use crate::lang::env::EnvRef;
use crate::lang::types::*;
use crate::physics::constants;
use crate::symbolic::expr::{MathConst, SymExpr, SymOp};
use std::f64::consts;

/// Names of built-in mathematical constants (used by sidebar to filter them out of user variables).
pub const BUILTIN_CONSTANTS: &[&str] = &["pi", "e", "i", "inf", "Inf", "tau"];

/// Register all built-in functions and constants into the environment.
pub fn register_builtins(env: &EnvRef) {
    let mut e = env.borrow_mut();

    // Math functions (1 arg)
    for &(name, f) in MATH_UNARY {
        e.set(
            name.to_string(),
            Value::Function(Function::Builtin {
                name: name.to_string(),
                arity: Arity::Exact(1),
                func: BuiltinFn(f),
            }),
        );
    }

    // Math functions (2 args)
    e.set(
        "log".to_string(),
        Value::Function(Function::Builtin {
            name: "log".to_string(),
            arity: Arity::Range(1, 2),
            func: BuiltinFn(builtin_log),
        }),
    );
    e.set(
        "max".to_string(),
        Value::Function(Function::Builtin {
            name: "max".to_string(),
            arity: Arity::Range(2, 255),
            func: BuiltinFn(builtin_max),
        }),
    );
    e.set(
        "min".to_string(),
        Value::Function(Function::Builtin {
            name: "min".to_string(),
            arity: Arity::Range(2, 255),
            func: BuiltinFn(builtin_min),
        }),
    );

    // Utility functions
    e.set(
        "print".to_string(),
        Value::Function(Function::Builtin {
            name: "print".to_string(),
            arity: Arity::Variadic,
            func: BuiltinFn(builtin_print),
        }),
    );
    e.set(
        "len".to_string(),
        Value::Function(Function::Builtin {
            name: "len".to_string(),
            arity: Arity::Exact(1),
            func: BuiltinFn(builtin_len),
        }),
    );
    e.set(
        "typeof".to_string(),
        Value::Function(Function::Builtin {
            name: "typeof".to_string(),
            arity: Arity::Exact(1),
            func: BuiltinFn(builtin_typeof),
        }),
    );

    // Math constants — stored as symbolic so they propagate through CAS
    e.set("pi".to_string(), Value::Symbolic(SymExpr::Const { name: MathConst::Pi }));
    e.set("e".to_string(), Value::Symbolic(SymExpr::Const { name: MathConst::E }));
    e.set("i".to_string(), Value::Symbolic(SymExpr::Const { name: MathConst::I }));
    e.set("inf".to_string(), Value::Symbolic(SymExpr::Const { name: MathConst::Infinity }));
    e.set("Inf".to_string(), Value::Symbolic(SymExpr::Const { name: MathConst::Infinity }));
    e.set("tau".to_string(), Value::Symbolic(SymExpr::BinOp {
        op: crate::symbolic::expr::SymOp::Mul,
        lhs: Box::new(SymExpr::Int { value: 2 }),
        rhs: Box::new(SymExpr::Const { name: MathConst::Pi }),
    }));

    // eval() — force numeric evaluation of symbolic constants/expressions
    e.set(
        "eval".to_string(),
        Value::Function(Function::Builtin {
            name: "eval".to_string(),
            arity: Arity::Exact(1),
            func: BuiltinFn(builtin_numeric_eval),
        }),
    );

    // Physics builtins
    e.set(
        "pm".to_string(),
        Value::Function(Function::Builtin {
            name: "pm".to_string(),
            arity: Arity::Exact(2),
            func: BuiltinFn(builtin_pm),
        }),
    );
    e.set(
        "uncertainty".to_string(),
        Value::Function(Function::Builtin {
            name: "uncertainty".to_string(),
            arity: Arity::Exact(1),
            func: BuiltinFn(builtin_uncertainty),
        }),
    );
    e.set(
        "nominal".to_string(),
        Value::Function(Function::Builtin {
            name: "nominal".to_string(),
            arity: Arity::Exact(1),
            func: BuiltinFn(builtin_nominal),
        }),
    );
    e.set(
        "units".to_string(),
        Value::Function(Function::Builtin {
            name: "units".to_string(),
            arity: Arity::Exact(1),
            func: BuiltinFn(builtin_units),
        }),
    );

    // Vector constructor
    e.set(
        "vec".to_string(),
        Value::Function(Function::Builtin {
            name: "vec".to_string(),
            arity: Arity::Variadic,
            func: BuiltinFn(builtin_vec),
        }),
    );

    // Statistics
    e.set(
        "pdiff".to_string(),
        Value::Function(Function::Builtin {
            name: "pdiff".to_string(),
            arity: Arity::Exact(2),
            func: BuiltinFn(builtin_pdiff),
        }),
    );

    // Register CODATA physical constants
    register_physics_constants(&mut e);
}

/// Register all CODATA constants as Value::Quantity in the environment.
fn register_physics_constants(e: &mut std::cell::RefMut<'_, crate::lang::env::Env>) {
    for pc in constants::CODATA {
        e.set(
            pc.name.to_string(),
            Value::Quantity(Quantity::new(pc.value, pc.uncertainty, pc.unit)),
        );
    }
}

// --- Math unary functions ---

const MATH_UNARY: &[(&str, BuiltinFnPtr)] = &[
    ("sin", builtin_sin),
    ("cos", builtin_cos),
    ("tan", builtin_tan),
    ("asin", builtin_asin),
    ("acos", builtin_acos),
    ("atan", builtin_atan),
    ("sinh", builtin_sinh),
    ("cosh", builtin_cosh),
    ("tanh", builtin_tanh),
    ("exp", builtin_exp),
    ("ln", builtin_ln),
    ("sqrt", builtin_sqrt),
    ("abs", builtin_abs),
    ("abs2", builtin_abs2),
    ("conj", builtin_conj),
    ("floor", builtin_floor),
    ("ceil", builtin_ceil),
    ("round", builtin_round),
    ("sign", builtin_sign),
];

fn num_arg(args: &[Value], idx: usize, name: &str) -> Result<f64, String> {
    args.get(idx)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("{}: argument {} must be a number", name, idx + 1))
}

fn builtin_sin(args: &[Value]) -> Result<Value, String> {
    Ok(Value::Number(Number::Float(num_arg(args, 0, "sin")?.sin())))
}

fn builtin_cos(args: &[Value]) -> Result<Value, String> {
    Ok(Value::Number(Number::Float(num_arg(args, 0, "cos")?.cos())))
}

fn builtin_tan(args: &[Value]) -> Result<Value, String> {
    Ok(Value::Number(Number::Float(num_arg(args, 0, "tan")?.tan())))
}

fn builtin_asin(args: &[Value]) -> Result<Value, String> {
    Ok(Value::Number(Number::Float(
        num_arg(args, 0, "asin")?.asin(),
    )))
}

fn builtin_acos(args: &[Value]) -> Result<Value, String> {
    Ok(Value::Number(Number::Float(
        num_arg(args, 0, "acos")?.acos(),
    )))
}

fn builtin_atan(args: &[Value]) -> Result<Value, String> {
    Ok(Value::Number(Number::Float(
        num_arg(args, 0, "atan")?.atan(),
    )))
}

fn builtin_sinh(args: &[Value]) -> Result<Value, String> {
    Ok(Value::Number(Number::Float(
        num_arg(args, 0, "sinh")?.sinh(),
    )))
}

fn builtin_cosh(args: &[Value]) -> Result<Value, String> {
    Ok(Value::Number(Number::Float(
        num_arg(args, 0, "cosh")?.cosh(),
    )))
}

fn builtin_tanh(args: &[Value]) -> Result<Value, String> {
    Ok(Value::Number(Number::Float(
        num_arg(args, 0, "tanh")?.tanh(),
    )))
}

fn builtin_exp(args: &[Value]) -> Result<Value, String> {
    Ok(Value::Number(Number::Float(num_arg(args, 0, "exp")?.exp())))
}

fn builtin_ln(args: &[Value]) -> Result<Value, String> {
    Ok(Value::Number(Number::Float(num_arg(args, 0, "ln")?.ln())))
}

fn builtin_sqrt(args: &[Value]) -> Result<Value, String> {
    Ok(Value::Number(Number::Float(
        num_arg(args, 0, "sqrt")?.sqrt(),
    )))
}

fn builtin_abs(args: &[Value]) -> Result<Value, String> {
    let n = num_arg(args, 0, "abs")?;
    if let Some(Value::Number(Number::Int(i))) = args.first() {
        Ok(Value::Number(Number::Int(i.abs())))
    } else {
        Ok(Value::Number(Number::Float(n.abs())))
    }
}

fn builtin_abs2(args: &[Value]) -> Result<Value, String> {
    let n = num_arg(args, 0, "abs2")?;
    if let Some(Value::Number(Number::Int(i))) = args.first() {
        Ok(Value::Number(Number::Int(i * i)))
    } else {
        Ok(Value::Number(Number::Float(n * n)))
    }
}

fn builtin_conj(args: &[Value]) -> Result<Value, String> {
    // For real numbers, conjugate is identity
    if let Some(Value::Number(Number::Int(i))) = args.first() {
        return Ok(Value::Number(Number::Int(*i)));
    }
    let n = num_arg(args, 0, "conj")?;
    Ok(Value::Number(Number::Float(n)))
}

fn builtin_floor(args: &[Value]) -> Result<Value, String> {
    Ok(Value::Number(Number::Float(
        num_arg(args, 0, "floor")?.floor(),
    )))
}

fn builtin_ceil(args: &[Value]) -> Result<Value, String> {
    Ok(Value::Number(Number::Float(
        num_arg(args, 0, "ceil")?.ceil(),
    )))
}

fn builtin_round(args: &[Value]) -> Result<Value, String> {
    Ok(Value::Number(Number::Float(
        num_arg(args, 0, "round")?.round(),
    )))
}

fn builtin_sign(args: &[Value]) -> Result<Value, String> {
    let n = num_arg(args, 0, "sign")?;
    Ok(Value::Number(Number::Float(n.signum())))
}

fn builtin_log(args: &[Value]) -> Result<Value, String> {
    if args.len() == 1 {
        // Natural log
        Ok(Value::Number(Number::Float(
            num_arg(args, 0, "log")?.ln(),
        )))
    } else {
        // log(base, x)
        let base = num_arg(args, 0, "log")?;
        let x = num_arg(args, 1, "log")?;
        Ok(Value::Number(Number::Float(x.log(base))))
    }
}

fn builtin_max(args: &[Value]) -> Result<Value, String> {
    let mut max = num_arg(args, 0, "max")?;
    for i in 1..args.len() {
        let v = num_arg(args, i, "max")?;
        if v > max {
            max = v;
        }
    }
    Ok(Value::Number(Number::Float(max)))
}

fn builtin_min(args: &[Value]) -> Result<Value, String> {
    let mut min = num_arg(args, 0, "min")?;
    for i in 1..args.len() {
        let v = num_arg(args, i, "min")?;
        if v < min {
            min = v;
        }
    }
    Ok(Value::Number(Number::Float(min)))
}

fn builtin_print(args: &[Value]) -> Result<Value, String> {
    let parts: Vec<String> = args.iter().map(|a| format!("{}", a)).collect();
    // In the TUI, print output will be captured by the evaluator.
    // For now, just return the string.
    Ok(Value::Str(parts.join(" ")))
}

fn builtin_len(args: &[Value]) -> Result<Value, String> {
    match &args[0] {
        Value::Vector(v) => Ok(Value::Number(Number::Int(v.len() as i64))),
        Value::Str(s) => Ok(Value::Number(Number::Int(s.len() as i64))),
        other => Err(format!("len: expected vector or string, got {}", other.type_name())),
    }
}

fn builtin_typeof(args: &[Value]) -> Result<Value, String> {
    Ok(Value::Str(args[0].type_name().to_string()))
}

// --- Physics builtins ---

fn builtin_pm(args: &[Value]) -> Result<Value, String> {
    let value = args[0].as_f64()
        .ok_or_else(|| "pm: first argument must be a number".to_string())?;
    let unc = args[1].as_f64()
        .ok_or_else(|| "pm: second argument must be a number".to_string())?;
    Ok(Value::Quantity(Quantity::dimensionless(value, unc)))
}

fn builtin_uncertainty(args: &[Value]) -> Result<Value, String> {
    match &args[0] {
        Value::Quantity(q) => Ok(Value::Number(Number::Float(q.uncertainty))),
        Value::Number(_) => Ok(Value::Number(Number::Float(0.0))),
        _ => Err(format!("uncertainty: expected number or quantity, got {}", args[0].type_name())),
    }
}

fn builtin_nominal(args: &[Value]) -> Result<Value, String> {
    match &args[0] {
        Value::Quantity(q) => Ok(Value::Number(Number::Float(q.value))),
        Value::Number(n) => Ok(Value::Number(Number::Float(n.as_f64()))),
        _ => Err(format!("nominal: expected number or quantity, got {}", args[0].type_name())),
    }
}

fn builtin_vec(args: &[Value]) -> Result<Value, String> {
    if args.is_empty() {
        return Err("vec: expected at least 1 argument".to_string());
    }
    Ok(Value::Vector(args.to_vec()))
}

fn builtin_pdiff(args: &[Value]) -> Result<Value, String> {
    let a = num_arg(args, 0, "pdiff")?;
    let b = num_arg(args, 1, "pdiff")?;
    let avg = (a + b) / 2.0;
    if avg == 0.0 {
        return Err("pdiff: average of values is zero, percent difference undefined".to_string());
    }
    let pd = ((a - b).abs() / avg.abs()) * 100.0;
    Ok(Value::Number(Number::Float(pd)))
}

fn builtin_units(args: &[Value]) -> Result<Value, String> {
    match &args[0] {
        Value::Quantity(q) => {
            if q.unit.is_dimensionless() {
                Ok(Value::Str("dimensionless".to_string()))
            } else {
                Ok(Value::Str(format!("{}", q.unit)))
            }
        }
        _ => Ok(Value::Str("dimensionless".to_string())),
    }
}

/// Evaluate a symbolic expression numerically.
fn builtin_numeric_eval(args: &[Value]) -> Result<Value, String> {
    match &args[0] {
        Value::Symbolic(expr) => {
            // Try complex evaluation first (handles expressions with `i`)
            if let Some((re, im)) = complex_eval_sym(expr) {
                return Ok(complex_to_value(re, im));
            }
            // Fall back to partial evaluation: collapse numeric subtrees, keep free vars
            let simplified = partial_eval_sym(expr);
            // If the simplified result is purely numeric, return a Number
            if let Some((re, im)) = complex_eval_sym(&simplified) {
                return Ok(complex_to_value(re, im));
            }
            // Return the partially simplified symbolic expression
            // (even if it looks the same — subtrees may have been folded)
            Ok(Value::Symbolic(simplified))
        }
        Value::Number(_) => Ok(args[0].clone()),
        _ => Err(format!("eval: expected numeric or symbolic, got {}", args[0].type_name())),
    }
}

/// Convert a complex (re, im) pair to a Value.
/// Returns Number if purely real, Symbolic if complex.
fn complex_to_value(re: f64, im: f64) -> Value {
    const EPS: f64 = 1e-15;
    if im.abs() < EPS {
        // Purely real
        let re = if re.abs() < EPS { 0.0 } else { re };
        if re.fract() == 0.0 && re.abs() < i64::MAX as f64 {
            Value::Number(Number::Int(re as i64))
        } else {
            Value::Number(Number::Float(re))
        }
    } else if re.abs() < EPS {
        // Purely imaginary: b*i
        let im = if (im - 1.0).abs() < EPS {
            SymExpr::Const { name: MathConst::I }
        } else if (im + 1.0).abs() < EPS {
            SymExpr::Neg { expr: Box::new(SymExpr::Const { name: MathConst::I }) }
        } else {
            SymExpr::BinOp {
                op: SymOp::Mul,
                lhs: Box::new(SymExpr::Float { value: im }),
                rhs: Box::new(SymExpr::Const { name: MathConst::I }),
            }
        };
        Value::Symbolic(im)
    } else {
        // a + b*i
        let im_part = if (im - 1.0).abs() < EPS {
            SymExpr::Const { name: MathConst::I }
        } else if (im + 1.0).abs() < EPS {
            SymExpr::Neg { expr: Box::new(SymExpr::Const { name: MathConst::I }) }
        } else {
            SymExpr::BinOp {
                op: SymOp::Mul,
                lhs: Box::new(SymExpr::Float { value: im.abs() }),
                rhs: Box::new(SymExpr::Const { name: MathConst::I }),
            }
        };
        let re_expr = SymExpr::Float { value: re };
        if im > 0.0 {
            Value::Symbolic(SymExpr::BinOp {
                op: SymOp::Add,
                lhs: Box::new(re_expr),
                rhs: Box::new(im_part),
            })
        } else {
            // a - |b|*i  (im_part already has abs(im) as coefficient)
            Value::Symbolic(SymExpr::BinOp {
                op: SymOp::Sub,
                lhs: Box::new(re_expr),
                rhs: Box::new(im_part),
            })
        }
    }
}

/// Recursively evaluate a SymExpr to a real float, if possible.
/// Returns None if the expression contains free variables or complex values.
pub fn numeric_eval_sym(expr: &SymExpr) -> Option<f64> {
    // Use complex eval, but only return if the result is purely real
    let (re, im) = complex_eval_sym(expr)?;
    if im.abs() < 1e-15 {
        Some(re)
    } else {
        None
    }
}

/// Recursively evaluate a SymExpr to a complex (re, im) pair, if possible.
/// Returns None if the expression contains free variables.
pub fn complex_eval_sym(expr: &SymExpr) -> Option<(f64, f64)> {
    match expr {
        SymExpr::Int { value } => Some((*value as f64, 0.0)),
        SymExpr::Float { value } => Some((*value, 0.0)),
        SymExpr::Rational { num, den } => Some((*num as f64 / *den as f64, 0.0)),
        SymExpr::Const { name } => match name {
            MathConst::Pi => Some((consts::PI, 0.0)),
            MathConst::E => Some((consts::E, 0.0)),
            MathConst::Infinity => Some((f64::INFINITY, 0.0)),
            MathConst::NegInfinity => Some((f64::NEG_INFINITY, 0.0)),
            MathConst::I => Some((0.0, 1.0)),
        },
        SymExpr::BinOp { op, lhs, rhs } => {
            let (lr, li) = complex_eval_sym(lhs)?;
            let (rr, ri) = complex_eval_sym(rhs)?;
            Some(match op {
                SymOp::Add => (lr + rr, li + ri),
                SymOp::Sub => (lr - rr, li - ri),
                SymOp::Mul => (lr * rr - li * ri, lr * ri + li * rr),
                SymOp::Div => {
                    let denom = rr * rr + ri * ri;
                    if denom == 0.0 {
                        return None;
                    }
                    ((lr * rr + li * ri) / denom, (li * rr - lr * ri) / denom)
                }
                SymOp::Pow => complex_pow((lr, li), (rr, ri)),
            })
        }
        SymExpr::Neg { expr } => {
            let (re, im) = complex_eval_sym(expr)?;
            Some((-re, -im))
        }
        SymExpr::Func { name, args } => {
            let vals: Option<Vec<(f64, f64)>> = args.iter().map(complex_eval_sym).collect();
            let vals = vals?;
            // For most functions, only support real arguments for now
            // (complex trig/exp would be a future extension)
            let is_real = |z: &(f64, f64)| z.1.abs() < 1e-15;
            match name.as_str() {
                "abs" if vals.len() == 1 => {
                    // |a+bi| = sqrt(a²+b²) — always real
                    let (re, im) = vals[0];
                    Some(((re * re + im * im).sqrt(), 0.0))
                }
                "abs2" if vals.len() == 1 => {
                    // |a+bi|² = a²+b² — always real
                    let (re, im) = vals[0];
                    Some((re * re + im * im, 0.0))
                }
                "conj" if vals.len() == 1 => {
                    Some((vals[0].0, -vals[0].1))
                }
                "exp" if vals.len() == 1 => {
                    // exp(a+bi) = e^a * (cos(b) + i*sin(b))
                    let (re, im) = vals[0];
                    let r = re.exp();
                    Some((r * im.cos(), r * im.sin()))
                }
                "sqrt" if vals.len() == 1 => {
                    let (re, im) = vals[0];
                    if im.abs() < 1e-15 && re >= 0.0 {
                        Some((re.sqrt(), 0.0))
                    } else {
                        // sqrt(a+bi) via polar form
                        let r = (re * re + im * im).sqrt().sqrt();
                        let theta = im.atan2(re) / 2.0;
                        Some((r * theta.cos(), r * theta.sin()))
                    }
                }
                _ if vals.len() == 1 && is_real(&vals[0]) => {
                    // Real-only functions
                    let v = vals[0].0;
                    let result = match name.as_str() {
                        "sin" => Some(v.sin()),
                        "cos" => Some(v.cos()),
                        "tan" => Some(v.tan()),
                        "asin" => Some(v.asin()),
                        "acos" => Some(v.acos()),
                        "atan" => Some(v.atan()),
                        "sinh" => Some(v.sinh()),
                        "cosh" => Some(v.cosh()),
                        "tanh" => Some(v.tanh()),
                        "ln" | "log" => {
                            if v > 0.0 { Some(v.ln()) } else { None }
                        }
                        "floor" => Some(v.floor()),
                        "ceil" => Some(v.ceil()),
                        _ => None,
                    };
                    result.map(|r| (r, 0.0))
                }
                // ln of negative real: ln(-x) = ln(x) + i*pi
                "ln" | "log" if vals.len() == 1 && is_real(&vals[0]) && vals[0].0 < 0.0 => {
                    let v = vals[0].0;
                    Some(((-v).ln(), consts::PI))
                }
                _ => None,
            }
        }
        SymExpr::Sym { .. } => None,
        SymExpr::Vector { .. } | SymExpr::Undefined => None,
    }
}

/// Complex exponentiation: (a+bi)^(c+di)
fn complex_pow(base: (f64, f64), exp: (f64, f64)) -> (f64, f64) {
    let (a, b) = base;
    let (c, d) = exp;

    // Special case: 0^anything
    if a == 0.0 && b == 0.0 {
        if c > 0.0 { return (0.0, 0.0); }
        return (f64::NAN, f64::NAN);
    }

    // Use polar form: base = r*e^(i*theta)
    // base^exp = r^c * e^(-d*theta) * e^(i*(c*theta + d*ln(r)))
    let r = (a * a + b * b).sqrt();
    let theta = b.atan2(a);
    let ln_r = r.ln();

    let new_r = (c * ln_r - d * theta).exp();
    let new_theta = c * theta + d * ln_r;

    (new_r * new_theta.cos(), new_r * new_theta.sin())
}

/// Partially evaluate a SymExpr: collapse fully-numeric subtrees to Float,
/// leave free variables and irreducible parts as-is.
pub fn partial_eval_sym(expr: &SymExpr) -> SymExpr {
    // If the whole expression evaluates to a complex number, convert it
    if let Some((re, im)) = complex_eval_sym(expr) {
        if im.abs() < 1e-15 {
            return SymExpr::Float { value: re };
        }
        // Return complex as symbolic: re + im*i
        return complex_to_sym(re, im);
    }

    // Otherwise, recurse and simplify children
    match expr {
        // Leaf nodes that can't be simplified further
        SymExpr::Sym { .. } | SymExpr::Int { .. } | SymExpr::Float { .. }
        | SymExpr::Rational { .. } | SymExpr::Const { .. } | SymExpr::Undefined => expr.clone(),

        SymExpr::BinOp { op, lhs, rhs } => {
            let l = partial_eval_sym(lhs);
            let r = partial_eval_sym(rhs);
            // Try to evaluate the simplified binop numerically
            if let Some((re, im)) = complex_eval_sym(&SymExpr::BinOp {
                op: *op, lhs: Box::new(l.clone()), rhs: Box::new(r.clone()),
            }) {
                if im.abs() < 1e-15 {
                    return SymExpr::Float { value: re };
                }
                return complex_to_sym(re, im);
            }
            SymExpr::BinOp { op: *op, lhs: Box::new(l), rhs: Box::new(r) }
        }

        SymExpr::Neg { expr: inner } => {
            let simplified = partial_eval_sym(inner);
            if let Some((re, im)) = complex_eval_sym(&simplified) {
                if im.abs() < 1e-15 {
                    return SymExpr::Float { value: -re };
                }
                return complex_to_sym(-re, -im);
            }
            SymExpr::Neg { expr: Box::new(simplified) }
        }

        SymExpr::Func { name, args } => {
            let simplified_args: Vec<SymExpr> = args.iter().map(partial_eval_sym).collect();
            let new_expr = SymExpr::Func { name: name.clone(), args: simplified_args.clone() };
            if let Some((re, im)) = complex_eval_sym(&new_expr) {
                if im.abs() < 1e-15 {
                    return SymExpr::Float { value: re };
                }
                return complex_to_sym(re, im);
            }
            new_expr
        }

        SymExpr::Vector { elements } => {
            let simplified: Vec<SymExpr> = elements.iter().map(partial_eval_sym).collect();
            SymExpr::Vector { elements: simplified }
        }
    }
}

/// Convert a complex (re, im) pair to a SymExpr.
fn complex_to_sym(re: f64, im: f64) -> SymExpr {
    const EPS: f64 = 1e-15;
    if im.abs() < EPS {
        SymExpr::Float { value: re }
    } else if re.abs() < EPS {
        // Purely imaginary
        if (im - 1.0).abs() < EPS {
            SymExpr::Const { name: MathConst::I }
        } else if (im + 1.0).abs() < EPS {
            SymExpr::Neg { expr: Box::new(SymExpr::Const { name: MathConst::I }) }
        } else {
            SymExpr::BinOp {
                op: SymOp::Mul,
                lhs: Box::new(SymExpr::Float { value: im }),
                rhs: Box::new(SymExpr::Const { name: MathConst::I }),
            }
        }
    } else {
        let im_part = if (im.abs() - 1.0).abs() < EPS {
            SymExpr::Const { name: MathConst::I }
        } else {
            SymExpr::BinOp {
                op: SymOp::Mul,
                lhs: Box::new(SymExpr::Float { value: im.abs() }),
                rhs: Box::new(SymExpr::Const { name: MathConst::I }),
            }
        };
        if im > 0.0 {
            SymExpr::BinOp {
                op: SymOp::Add,
                lhs: Box::new(SymExpr::Float { value: re }),
                rhs: Box::new(im_part),
            }
        } else {
            SymExpr::BinOp {
                op: SymOp::Sub,
                lhs: Box::new(SymExpr::Float { value: re }),
                rhs: Box::new(im_part),
            }
        }
    }
}
