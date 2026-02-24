use crate::lang::env::EnvRef;
use crate::lang::types::*;
use crate::symbolic::expr::{MathConst, SymExpr};
use std::f64::consts;

/// Names of built-in mathematical constants (used by sidebar to filter them out of user variables).
pub const BUILTIN_CONSTANTS: &[&str] = &["pi", "e", "inf", "Inf", "tau"];

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

/// Evaluate a symbolic expression numerically.
fn builtin_numeric_eval(args: &[Value]) -> Result<Value, String> {
    match &args[0] {
        Value::Symbolic(expr) => {
            match numeric_eval_sym(expr) {
                Some(f) => Ok(Value::Number(Number::Float(f))),
                None => Err("eval: cannot evaluate symbolically — expression contains free variables".to_string()),
            }
        }
        Value::Number(_) => Ok(args[0].clone()),
        _ => Err(format!("eval: expected numeric or symbolic, got {}", args[0].type_name())),
    }
}

/// Recursively evaluate a SymExpr to a float, if possible.
pub fn numeric_eval_sym(expr: &SymExpr) -> Option<f64> {
    match expr {
        SymExpr::Int { value } => Some(*value as f64),
        SymExpr::Float { value } => Some(*value),
        SymExpr::Rational { num, den } => Some(*num as f64 / *den as f64),
        SymExpr::Const { name } => match name {
            MathConst::Pi => Some(consts::PI),
            MathConst::E => Some(consts::E),
            MathConst::Infinity => Some(f64::INFINITY),
            MathConst::NegInfinity => Some(f64::NEG_INFINITY),
            MathConst::I => None, // complex
        },
        SymExpr::BinOp { op, lhs, rhs } => {
            let l = numeric_eval_sym(lhs)?;
            let r = numeric_eval_sym(rhs)?;
            Some(match op {
                crate::symbolic::expr::SymOp::Add => l + r,
                crate::symbolic::expr::SymOp::Sub => l - r,
                crate::symbolic::expr::SymOp::Mul => l * r,
                crate::symbolic::expr::SymOp::Div => l / r,
                crate::symbolic::expr::SymOp::Pow => l.powf(r),
            })
        }
        SymExpr::Neg { expr } => numeric_eval_sym(expr).map(|v| -v),
        SymExpr::Func { name, args } => {
            let vals: Option<Vec<f64>> = args.iter().map(numeric_eval_sym).collect();
            let vals = vals?;
            match name.as_str() {
                "sin" => Some(vals[0].sin()),
                "cos" => Some(vals[0].cos()),
                "tan" => Some(vals[0].tan()),
                "asin" => Some(vals[0].asin()),
                "acos" => Some(vals[0].acos()),
                "atan" => Some(vals[0].atan()),
                "sinh" => Some(vals[0].sinh()),
                "cosh" => Some(vals[0].cosh()),
                "tanh" => Some(vals[0].tanh()),
                "exp" => Some(vals[0].exp()),
                "ln" | "log" => Some(vals[0].ln()),
                "sqrt" => Some(vals[0].sqrt()),
                "abs" => Some(vals[0].abs()),
                "floor" => Some(vals[0].floor()),
                "ceil" => Some(vals[0].ceil()),
                _ => None,
            }
        }
        SymExpr::Sym { .. } => None, // free variable
        SymExpr::Vector { .. } | SymExpr::Undefined => None,
    }
}
