#!/usr/bin/env python3
"""
Aurita Maxima Bridge — JSON-RPC over stdin/stdout.

Spawns a Maxima subprocess and translates between the Aurita CAS protocol
(JSON over stdin/stdout) and Maxima's text-based interface.

Reuses expr_to_json / expr_from_json from python_bridge.py for expression
serialization (SymPy as the canonical AST format).
"""

import sys
import os
import json
import re
import select
import subprocess
import traceback

# Add the backends directory to path so we can import from python_bridge
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from python_bridge import expr_from_json, expr_to_json, get_symbol

import sympy
from sympy import parse_expr


class MaximaTimeout(Exception):
    """Raised when Maxima takes too long or asks an interactive question."""
    pass


# Seconds to wait for Maxima to produce the next byte of output.
MAXIMA_READ_TIMEOUT = 5


class MaximaProcess:
    """Manages a Maxima subprocess."""

    def __init__(self):
        self._spawn()

    def _spawn(self):
        """Spawn (or re-spawn) the Maxima process."""
        self.proc = subprocess.Popen(
            ["maxima"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._fd = self.proc.stdout.fileno()
        # Read startup banner until first prompt (generous timeout for startup)
        self._read_until_prompt(timeout=30)
        # Initialize: disable 2D display, set long line length
        self._send_raw("display2d:false$")
        self._read_until_prompt(timeout=10)
        self._send_raw("linel:10000$")
        self._read_until_prompt(timeout=10)

    def _send_raw(self, cmd):
        """Send a raw command to Maxima."""
        self.proc.stdin.write((cmd + "\n").encode())
        self.proc.stdin.flush()

    def _read_char(self, timeout):
        """Read a single byte from Maxima stdout with a timeout.

        Uses select() on the raw fd + os.read() to avoid Python's
        internal buffering which can defeat timeout logic.
        """
        ready, _, _ = select.select([self._fd], [], [], timeout)
        if not ready:
            raise MaximaTimeout("Maxima did not respond in time")
        data = os.read(self._fd, 1)
        if not data:
            return None
        return data.decode('utf-8', errors='replace')

    def _read_until_prompt(self, timeout=MAXIMA_READ_TIMEOUT):
        """Read Maxima output until we see an input prompt (%iN).

        Uses select()+os.read() with a per-byte timeout so we detect
        when Maxima stalls (e.g. asking an interactive question).
        """
        buf = []
        output_lines = []
        while True:
            ch = self._read_char(timeout)
            if ch is None:
                break
            buf.append(ch)
            if ch == '\n':
                line = ''.join(buf).rstrip('\n')
                buf = []
                if re.match(r'^\(%i\d+\)\s*$', line):
                    break
                output_lines.append(line)
            else:
                current = ''.join(buf)
                if re.match(r'^\(%i\d+\) $', current):
                    break
        return '\n'.join(output_lines)

    def execute(self, cmd):
        """Send a command to Maxima and return the output text.

        If Maxima hangs (e.g. asks an interactive question), kills and
        restarts the process, then raises MaximaTimeout.
        """
        self._send_raw(cmd)
        try:
            raw = self._read_until_prompt(timeout=MAXIMA_READ_TIMEOUT)
        except MaximaTimeout:
            self._restart()
            raise

        # Extract result: strip output labels like (%o2)
        result_lines = []
        for line in raw.split("\n"):
            cleaned = re.sub(r"^\(%o\d+\)\s*", "", line)
            result_lines.append(cleaned)

        result = "\n".join(result_lines).strip()
        return result

    def _restart(self):
        """Kill the current Maxima process and start a fresh one."""
        try:
            self.proc.kill()
            self.proc.wait(timeout=3)
        except Exception:
            pass
        self._spawn()

    def close(self):
        """Shut down the Maxima process."""
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def symexpr_to_maxima(node):
    """Convert a JSON SymExpr tree to a Maxima syntax string."""
    t = node["type"]

    if t == "Sym":
        return node["name"]
    elif t == "Int":
        v = node["value"]
        if v < 0:
            return f"({v})"
        return str(v)
    elif t == "Rational":
        return f"({node['num']}/{node['den']})"
    elif t == "Float":
        return str(node["value"])
    elif t == "Const":
        name = node["name"]
        const_map = {
            "Pi": "%pi",
            "E": "%e",
            "I": "%i",
            "Infinity": "inf",
            "NegInfinity": "minf",
        }
        return const_map.get(name, name)
    elif t == "BinOp":
        lhs = symexpr_to_maxima(node["lhs"])
        rhs = symexpr_to_maxima(node["rhs"])
        op = node["op"]
        op_map = {
            "Add": "+",
            "Sub": "-",
            "Mul": "*",
            "Div": "/",
            "Pow": "^",
        }
        op_str = op_map.get(op, op)
        return f"({lhs}{op_str}{rhs})"
    elif t == "Neg":
        inner = symexpr_to_maxima(node["expr"])
        return f"(-{inner})"
    elif t == "Func":
        fname = node["name"]
        args = [symexpr_to_maxima(a) for a in node["args"]]
        # Maxima uses log for natural log (ln)
        func_map = {
            "ln": "log",
            "abs": "abs",
            "ceil": "ceiling",
            "floor": "floor",
            "erf": "erf",
        }
        mname = func_map.get(fname, fname)
        return f"{mname}({', '.join(args)})"
    elif t == "Vector":
        elements = [symexpr_to_maxima(e) for e in node["elements"]]
        return f"[{', '.join(elements)}]"
    elif t == "Undefined":
        return "und"
    else:
        raise ValueError(f"Unknown node type for Maxima conversion: {t}")


def maxima_to_symexpr(text):
    """Parse Maxima output text into a JSON SymExpr tree.

    Strategy: convert Maxima-specific syntax to SymPy-compatible syntax,
    then use SymPy's parse_expr, then convert to JSON via expr_to_json.
    """
    text = text.strip()

    if not text or text == "done" or text == "false":
        return {"type": "Sym", "name": text}

    # Preprocessing: Maxima → SymPy syntax
    text = text.replace("%pi", "pi_const")
    text = text.replace("%e", "E_const")
    text = text.replace("%i", "I_const")

    # Handle inf/minf before general parsing
    text = re.sub(r'\bminf\b', '(-oo_const)', text)
    text = re.sub(r'\binf\b', 'oo_const', text)

    # Maxima uses log for natural log
    # But we need to be careful: log(x) in Maxima = ln(x) in our system
    # SymPy's log is already natural log, so this works out.

    # Build local_dict with real symbols
    # Extract all identifiers from the expression
    idents = set(re.findall(r'[a-zA-Z_]\w*', text))
    # Remove known functions and constants
    known = {
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
        'sinh', 'cosh', 'tanh', 'exp', 'log', 'sqrt',
        'abs', 'sign', 'floor', 'ceiling', 'erf', 'erfc',
        'gamma', 'factorial',
        'pi_const', 'E_const', 'I_const', 'oo_const',
    }
    local_dict = {}
    for ident in idents - known:
        local_dict[ident] = get_symbol(ident, real=True)

    # Add constants
    local_dict['pi_const'] = sympy.pi
    local_dict['E_const'] = sympy.E
    local_dict['I_const'] = sympy.I
    local_dict['oo_const'] = sympy.oo

    try:
        expr = parse_expr(text, local_dict=local_dict,
                         transformations='all')
        return expr_to_json(expr)
    except Exception as e:
        # Fallback: return as a symbol
        return {"type": "Sym", "name": text}


def parse_solve_output(text):
    """Parse Maxima's solve output: [x = val1, x = val2, ...] → list of SymExpr JSON."""
    text = text.strip()

    # Remove outer brackets
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    else:
        return [maxima_to_symexpr(text)]

    if not text.strip():
        return []

    # Split on commas, respecting brackets
    parts = []
    depth = 0
    current = ""
    for ch in text:
        if ch in "([":
            depth += 1
            current += ch
        elif ch in ")]":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())

    results = []
    for part in parts:
        # Each part is like "x = value"
        if "=" in part:
            rhs = part.split("=", 1)[1].strip()
            results.append(maxima_to_symexpr(rhs))
        else:
            results.append(maxima_to_symexpr(part))

    return results


def parse_latex_output(text):
    """Parse Maxima's tex() output to extract the LaTeX string."""
    # Maxima's tex() outputs something like:
    # $$expression$$
    # false
    text = text.strip()
    lines = text.split("\n")
    latex_parts = []
    for line in lines:
        line = line.strip()
        if line == "false" or line == "done":
            continue
        latex_parts.append(line)
    latex = "\n".join(latex_parts)
    # Remove $$ delimiters
    latex = re.sub(r'^\$\$', '', latex)
    latex = re.sub(r'\$\$$', '', latex)
    return latex.strip()


def handle_request(maxima, req):
    """Process a single CAS request via the Maxima subprocess."""
    req_id = req["id"]
    op_data = req["op"]
    op_name = op_data["op"]
    params = op_data["params"]

    try:
        if op_name == "differentiate":
            mexpr = symexpr_to_maxima(params["expr"])
            var = params["var"]
            order = params.get("order", 1)
            cmd = f"diff({mexpr}, {var}, {order});"
            result = maxima.execute(cmd)
            return {"id": req_id, "status": "ok", "result": maxima_to_symexpr(result)}

        elif op_name == "integrate":
            mexpr = symexpr_to_maxima(params["expr"])
            var = params["var"]
            lower = params.get("lower")
            upper = params.get("upper")
            if lower is not None and upper is not None:
                lo = symexpr_to_maxima(lower)
                hi = symexpr_to_maxima(upper)
                cmd = f"integrate({mexpr}, {var}, {lo}, {hi});"
            else:
                cmd = f"integrate({mexpr}, {var});"
            result = maxima.execute(cmd)
            return {"id": req_id, "status": "ok", "result": maxima_to_symexpr(result)}

        elif op_name == "solve":
            equations = params["equations"]
            var_names = params["vars"]
            if len(equations) == 1 and len(var_names) == 1:
                mexpr = symexpr_to_maxima(equations[0])
                cmd = f"solve({mexpr}, {var_names[0]});"
            else:
                meqs = ", ".join(symexpr_to_maxima(eq) for eq in equations)
                mvars = ", ".join(var_names)
                cmd = f"solve([{meqs}], [{mvars}]);"
            result = maxima.execute(cmd)
            results = parse_solve_output(result)
            return {"id": req_id, "status": "ok", "results": results}

        elif op_name == "simplify":
            mexpr = symexpr_to_maxima(params["expr"])
            cmd = f"ratsimp({mexpr});"
            result = maxima.execute(cmd)
            return {"id": req_id, "status": "ok", "result": maxima_to_symexpr(result)}

        elif op_name == "expand":
            mexpr = symexpr_to_maxima(params["expr"])
            cmd = f"expand({mexpr});"
            result = maxima.execute(cmd)
            return {"id": req_id, "status": "ok", "result": maxima_to_symexpr(result)}

        elif op_name == "factor":
            mexpr = symexpr_to_maxima(params["expr"])
            cmd = f"factor({mexpr});"
            result = maxima.execute(cmd)
            return {"id": req_id, "status": "ok", "result": maxima_to_symexpr(result)}

        elif op_name == "limit":
            mexpr = symexpr_to_maxima(params["expr"])
            var = params["var"]
            point = symexpr_to_maxima(params["point"])
            direction = params.get("dir")
            if direction == "+":
                cmd = f"limit({mexpr}, {var}, {point}, plus);"
            elif direction == "-":
                cmd = f"limit({mexpr}, {var}, {point}, minus);"
            else:
                cmd = f"limit({mexpr}, {var}, {point});"
            result = maxima.execute(cmd)
            return {"id": req_id, "status": "ok", "result": maxima_to_symexpr(result)}

        elif op_name == "taylor":
            mexpr = symexpr_to_maxima(params["expr"])
            var = params["var"]
            point = symexpr_to_maxima(params["point"])
            order = params.get("order", 5)
            # taylor(...) then ratdisrep to strip O() term
            cmd = f"ratdisrep(taylor({mexpr}, {var}, {point}, {order}));"
            result = maxima.execute(cmd)
            return {"id": req_id, "status": "ok", "result": maxima_to_symexpr(result)}

        elif op_name == "latex":
            mexpr = symexpr_to_maxima(params["expr"])
            cmd = f"tex({mexpr});"
            result = maxima.execute(cmd)
            latex_str = parse_latex_output(result)
            return {
                "id": req_id,
                "status": "ok",
                "result": {"type": "Sym", "name": latex_str},
                "latex": latex_str,
            }

        elif op_name == "lambdify":
            # Maxima can't do fast numpy-style evaluation — return error
            return {"id": req_id, "status": "error",
                    "error": "lambdify not supported by Maxima backend"}

        elif op_name == "render_plot":
            # Plot rendering not supported by Maxima backend
            return {"id": req_id, "status": "error",
                    "error": "render_plot not supported by Maxima backend"}

        else:
            return {"id": req_id, "status": "error",
                    "error": f"unknown operation: {op_name}"}

    except MaximaTimeout:
        return {"id": req_id, "status": "error",
                "error": "Maxima timed out (likely needs assumptions about variables)"}
    except Exception as e:
        return {"id": req_id, "status": "error", "error": str(e)}


def main():
    """Main loop: read JSON requests from stdin, write responses to stdout."""
    try:
        maxima = MaximaProcess()
    except FileNotFoundError:
        sys.stderr.write("Error: maxima not found in PATH\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error starting Maxima: {e}\n")
        sys.exit(1)

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = handle_request(maxima, req)
            except json.JSONDecodeError as e:
                resp = {"id": 0, "status": "error", "error": f"invalid JSON: {e}"}
            except Exception as e:
                resp = {"id": 0, "status": "error",
                        "error": f"bridge error: {e}\n{traceback.format_exc()}"}

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    finally:
        maxima.close()


if __name__ == "__main__":
    main()
