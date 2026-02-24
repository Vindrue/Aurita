#!/usr/bin/env python3
"""
Aurita SymPy Bridge — JSON-RPC over stdin/stdout.

Reads line-delimited JSON requests from stdin, performs symbolic operations
using SymPy, and writes JSON responses to stdout.
"""

import sys
import json
import traceback
import sympy
from sympy import (
    Symbol, symbols, Integer, Rational, Float, pi, E, I, oo,
    sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
    exp, log, sqrt, Abs, sign, floor, ceiling,
    diff, integrate, limit, series, solve, simplify, expand, factor,
    latex, refine, Piecewise,
)
from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine


# Cache symbols so they share assumptions within a session.
_symbol_cache = {}


def get_symbol(name, **assumptions):
    """Get or create a symbol with the given assumptions."""
    key = (name, tuple(sorted(assumptions.items())))
    if key not in _symbol_cache:
        _symbol_cache[key] = Symbol(name, **assumptions)
    return _symbol_cache[key]


def expr_from_json(node):
    """Convert a JSON SymExpr tree to a SymPy expression."""
    t = node["type"]

    if t == "Sym":
        # Default: symbols are real (avoids Piecewise blowup in integration)
        return get_symbol(node["name"], real=True)
    elif t == "Int":
        return Integer(node["value"])
    elif t == "Rational":
        return Rational(node["num"], node["den"])
    elif t == "Float":
        return Float(node["value"])
    elif t == "Const":
        name = node["name"]
        if name == "Pi":
            return pi
        elif name == "E":
            return E
        elif name == "I":
            return I
        elif name == "Infinity":
            return oo
        elif name == "NegInfinity":
            return -oo
        else:
            raise ValueError(f"Unknown constant: {name}")
    elif t == "BinOp":
        lhs = expr_from_json(node["lhs"])
        rhs = expr_from_json(node["rhs"])
        op = node["op"]
        if op == "Add":
            return lhs + rhs
        elif op == "Sub":
            return lhs - rhs
        elif op == "Mul":
            return lhs * rhs
        elif op == "Div":
            return lhs / rhs
        elif op == "Pow":
            return lhs ** rhs
        else:
            raise ValueError(f"Unknown op: {op}")
    elif t == "Neg":
        return -expr_from_json(node["expr"])
    elif t == "Func":
        fname = node["name"]
        args = [expr_from_json(a) for a in node["args"]]
        func_map = {
            "sin": sin, "cos": cos, "tan": tan,
            "asin": asin, "acos": acos, "atan": atan,
            "sinh": sinh, "cosh": cosh, "tanh": tanh,
            "exp": exp, "ln": log, "log": log, "sqrt": sqrt,
            "abs": Abs, "sign": sign, "floor": floor, "ceil": ceiling,
            "erf": sympy.erf, "erfc": sympy.erfc,
            "gamma": sympy.gamma, "factorial": sympy.factorial,
        }
        if fname in func_map:
            return func_map[fname](*args)
        else:
            # Generic symbolic function
            f = sympy.Function(fname)
            return f(*args)
    elif t == "Vector":
        return sympy.Matrix([expr_from_json(e) for e in node["elements"]])
    elif t == "Undefined":
        return sympy.nan
    else:
        raise ValueError(f"Unknown node type: {t}")


def unwrap_piecewise(expr):
    """If expr is a Piecewise, extract the most useful branch.

    SymPy often returns Piecewise results when it doesn't know assumptions.
    Since we default to real symbols, we pick the first non-trivial branch
    (typically the main result, conditioned on the parameter being non-zero).
    """
    if not isinstance(expr, Piecewise):
        return expr

    # Look through the branches
    for branch_expr, condition in expr.args:
        # Skip the trivial "otherwise zero" or "otherwise nan" branches
        if branch_expr == 0 or branch_expr is sympy.nan:
            continue
        # Skip branches that are just the integral variable (unevaluated)
        if isinstance(branch_expr, sympy.Integral):
            continue
        # Take the first substantive branch
        return branch_expr

    # Fallback: return the first branch
    return expr.args[0][0] if expr.args else expr


def postprocess(expr):
    """Post-process a SymPy result: unwrap Piecewise, refine under real assumptions."""
    expr = unwrap_piecewise(expr)
    # Walk the expression tree to unwrap nested Piecewise
    if hasattr(expr, 'args') and expr.args:
        new_args = []
        changed = False
        for arg in expr.args:
            new_arg = postprocess(arg)
            new_args.append(new_arg)
            if new_arg is not arg:
                changed = True
        if changed:
            try:
                expr = expr.func(*new_args)
            except Exception:
                pass
    return expr


def expr_to_json(expr):
    """Convert a SymPy expression to a JSON SymExpr tree."""
    if expr is sympy.nan or expr is sympy.zoo:
        return {"type": "Undefined"}
    if expr is pi:
        return {"type": "Const", "name": "Pi"}
    if expr is E:
        return {"type": "Const", "name": "E"}
    if expr is I:
        return {"type": "Const", "name": "I"}
    if expr is oo:
        return {"type": "Const", "name": "Infinity"}
    if expr is -oo:
        return {"type": "Const", "name": "NegInfinity"}

    if isinstance(expr, sympy.Integer):
        return {"type": "Int", "value": int(expr)}
    if isinstance(expr, sympy.Rational):
        return {"type": "Rational", "num": int(expr.p), "den": int(expr.q)}
    if isinstance(expr, sympy.Float):
        return {"type": "Float", "value": float(expr)}
    if isinstance(expr, (int,)):
        return {"type": "Int", "value": expr}
    if isinstance(expr, (float,)):
        return {"type": "Float", "value": expr}

    if isinstance(expr, sympy.Symbol):
        return {"type": "Sym", "name": str(expr)}

    if isinstance(expr, sympy.Mul):
        # Handle -1 * expr as Neg
        args = expr.args
        if len(args) == 2 and args[0] == -1:
            return {"type": "Neg", "expr": expr_to_json(args[1])}
        # Chain of multiplications
        result = expr_to_json(args[0])
        for arg in args[1:]:
            result = {"type": "BinOp", "op": "Mul", "lhs": result, "rhs": expr_to_json(arg)}
        return result

    if isinstance(expr, sympy.Add):
        args = expr.args
        result = expr_to_json(args[0])
        for arg in args[1:]:
            # Check if arg is negative (Mul with -1)
            if isinstance(arg, sympy.Mul) and arg.args[0] == -1:
                rhs = expr_to_json(-arg)
                result = {"type": "BinOp", "op": "Sub", "lhs": result, "rhs": rhs}
            else:
                result = {"type": "BinOp", "op": "Add", "lhs": result, "rhs": expr_to_json(arg)}
        return result

    if isinstance(expr, sympy.Pow):
        base = expr_to_json(expr.base)
        exp_val = expr_to_json(expr.exp)
        # x^(-1) → 1/x
        if expr.exp == -1:
            return {"type": "BinOp", "op": "Div", "lhs": {"type": "Int", "value": 1}, "rhs": base}
        # x^(1/2) → sqrt(x)
        if isinstance(expr.exp, sympy.Rational) and expr.exp == sympy.Rational(1, 2):
            return {"type": "Func", "name": "sqrt", "args": [base]}
        # x^(-1/2) → 1/sqrt(x)
        if isinstance(expr.exp, sympy.Rational) and expr.exp == sympy.Rational(-1, 2):
            return {"type": "BinOp", "op": "Div",
                    "lhs": {"type": "Int", "value": 1},
                    "rhs": {"type": "Func", "name": "sqrt", "args": [base]}}
        return {"type": "BinOp", "op": "Pow", "lhs": base, "rhs": exp_val}

    # Piecewise — unwrap before converting
    if isinstance(expr, Piecewise):
        return expr_to_json(unwrap_piecewise(expr))

    # Functions
    if isinstance(expr, sympy.Function):
        fname = type(expr).__name__
        name_map = {
            "sin": "sin", "cos": "cos", "tan": "tan",
            "asin": "asin", "acos": "acos", "atan": "atan",
            "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
            "exp": "exp", "log": "ln",
            "Abs": "abs", "sign": "sign", "floor": "floor", "ceiling": "ceil",
            "erf": "erf", "erfc": "erfc",
            "gamma": "gamma", "factorial": "factorial",
        }
        out_name = name_map.get(fname, fname)
        args = [expr_to_json(a) for a in expr.args]
        return {"type": "Func", "name": out_name, "args": args}

    # Matrix/Vector
    if isinstance(expr, sympy.Matrix):
        elements = [expr_to_json(expr[i]) for i in range(len(expr))]
        return {"type": "Vector", "elements": elements}

    # Fallback: try to decompose
    if hasattr(expr, 'args') and expr.args:
        # Generic: treat as function
        fname = type(expr).__name__
        args = [expr_to_json(a) for a in expr.args]
        return {"type": "Func", "name": fname, "args": args}

    # Last resort: convert via string
    return {"type": "Sym", "name": str(expr)}


def handle_request(req):
    """Process a single CAS request and return a response dict."""
    req_id = req["id"]
    op_data = req["op"]
    op_name = op_data["op"]
    params = op_data["params"]

    try:
        if op_name == "differentiate":
            expr = expr_from_json(params["expr"])
            var = get_symbol(params["var"], real=True)
            order = params.get("order", 1)
            result = diff(expr, var, order)
            return {"id": req_id, "status": "ok", "result": expr_to_json(result)}

        elif op_name == "integrate":
            expr = expr_from_json(params["expr"])
            var = get_symbol(params["var"], real=True)
            lower = params.get("lower")
            upper = params.get("upper")
            if lower is not None and upper is not None:
                lo = expr_from_json(lower)
                hi = expr_from_json(upper)
                result = integrate(expr, (var, lo, hi))
            else:
                result = integrate(expr, var)
            result = postprocess(result)
            return {"id": req_id, "status": "ok", "result": expr_to_json(result)}

        elif op_name == "solve":
            equations = [expr_from_json(eq) for eq in params["equations"]]
            var_names = params["vars"]
            var_syms = [get_symbol(v, real=True) for v in var_names]
            if len(var_syms) == 1:
                result = solve(equations[0] if len(equations) == 1 else equations, var_syms[0])
            else:
                result = solve(equations, var_syms)

            # Convert results
            if isinstance(result, dict):
                # System of equations: {x: val, y: val}
                results = [expr_to_json(v) for v in result.values()]
            elif isinstance(result, list):
                if result and isinstance(result[0], dict):
                    # List of solution dicts
                    results = []
                    for sol in result:
                        results.append(expr_to_json(list(sol.values())[0]) if len(sol) == 1
                                       else {"type": "Vector", "elements": [expr_to_json(v) for v in sol.values()]})
                elif result and isinstance(result[0], tuple):
                    results = [{"type": "Vector", "elements": [expr_to_json(v) for v in tup]} for tup in result]
                else:
                    results = [expr_to_json(r) for r in result]
            else:
                results = [expr_to_json(result)]

            return {"id": req_id, "status": "ok", "results": results}

        elif op_name == "simplify":
            expr = expr_from_json(params["expr"])
            result = simplify(expr)
            result = postprocess(result)
            return {"id": req_id, "status": "ok", "result": expr_to_json(result)}

        elif op_name == "expand":
            expr = expr_from_json(params["expr"])
            result = expand(expr)
            return {"id": req_id, "status": "ok", "result": expr_to_json(result)}

        elif op_name == "factor":
            expr = expr_from_json(params["expr"])
            result = factor(expr)
            return {"id": req_id, "status": "ok", "result": expr_to_json(result)}

        elif op_name == "limit":
            expr = expr_from_json(params["expr"])
            var = get_symbol(params["var"], real=True)
            point = expr_from_json(params["point"])
            direction = params.get("dir")
            if direction == "+":
                result = limit(expr, var, point, "+")
            elif direction == "-":
                result = limit(expr, var, point, "-")
            else:
                result = limit(expr, var, point)
            return {"id": req_id, "status": "ok", "result": expr_to_json(result)}

        elif op_name == "taylor":
            expr = expr_from_json(params["expr"])
            var = get_symbol(params["var"], real=True)
            point = expr_from_json(params["point"])
            order = params.get("order", 5)
            s = series(expr, var, point, n=order + 1)
            result = s.removeO()
            return {"id": req_id, "status": "ok", "result": expr_to_json(result)}

        elif op_name == "latex":
            expr = expr_from_json(params["expr"])
            result = latex(expr)
            return {"id": req_id, "status": "ok", "result": {"type": "Sym", "name": result}, "latex": result}

        else:
            return {"id": req_id, "status": "error", "error": f"unknown operation: {op_name}"}

    except Exception as e:
        return {"id": req_id, "status": "error", "error": str(e)}


def main():
    """Main loop: read JSON requests from stdin, write responses to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
        except json.JSONDecodeError as e:
            resp = {"id": 0, "status": "error", "error": f"invalid JSON: {e}"}
        except Exception as e:
            resp = {"id": 0, "status": "error", "error": f"bridge error: {e}\n{traceback.format_exc()}"}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
