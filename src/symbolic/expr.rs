use serde::{Deserialize, Serialize};
use std::fmt;

/// Symbolic expression tree — the Rust-native representation
/// for symbolic math that gets serialized to JSON for CAS backends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SymExpr {
    /// A symbolic variable: x, theta, alpha
    Sym { name: String },
    /// An integer constant (exact)
    Int { value: i64 },
    /// A rational constant (exact)
    Rational { num: i64, den: i64 },
    /// A floating-point constant
    Float { value: f64 },
    /// Known constant: pi, e, i
    Const { name: MathConst },
    /// Binary operation
    BinOp {
        op: SymOp,
        lhs: Box<SymExpr>,
        rhs: Box<SymExpr>,
    },
    /// Unary negation
    Neg { expr: Box<SymExpr> },
    /// Function application: sin(x), ln(x), etc.
    Func { name: String, args: Vec<SymExpr> },
    /// Vector
    Vector { elements: Vec<SymExpr> },
    /// Undefined / could not compute
    Undefined,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SymOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MathConst {
    Pi,
    E,
    I,
    Infinity,
    NegInfinity,
}

// --- Constructors ---

impl SymExpr {
    pub fn sym(name: &str) -> Self {
        SymExpr::Sym {
            name: name.to_string(),
        }
    }

    pub fn int(value: i64) -> Self {
        SymExpr::Int { value }
    }

    pub fn float(value: f64) -> Self {
        SymExpr::Float { value }
    }

    pub fn add(lhs: SymExpr, rhs: SymExpr) -> Self {
        SymExpr::BinOp {
            op: SymOp::Add,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn sub(lhs: SymExpr, rhs: SymExpr) -> Self {
        SymExpr::BinOp {
            op: SymOp::Sub,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn mul(lhs: SymExpr, rhs: SymExpr) -> Self {
        SymExpr::BinOp {
            op: SymOp::Mul,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn div(lhs: SymExpr, rhs: SymExpr) -> Self {
        SymExpr::BinOp {
            op: SymOp::Div,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn pow(lhs: SymExpr, rhs: SymExpr) -> Self {
        SymExpr::BinOp {
            op: SymOp::Pow,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn neg(expr: SymExpr) -> Self {
        SymExpr::Neg {
            expr: Box::new(expr),
        }
    }

    pub fn func(name: &str, args: Vec<SymExpr>) -> Self {
        SymExpr::Func {
            name: name.to_string(),
            args,
        }
    }

    /// Collect all free symbolic variable names in this expression.
    pub fn free_symbols(&self) -> Vec<String> {
        let mut syms = Vec::new();
        self.collect_symbols(&mut syms);
        syms.sort();
        syms.dedup();
        syms
    }

    fn collect_symbols(&self, syms: &mut Vec<String>) {
        match self {
            SymExpr::Sym { name } => syms.push(name.clone()),
            SymExpr::BinOp { lhs, rhs, .. } => {
                lhs.collect_symbols(syms);
                rhs.collect_symbols(syms);
            }
            SymExpr::Neg { expr } => expr.collect_symbols(syms),
            SymExpr::Func { args, .. } => {
                for arg in args {
                    arg.collect_symbols(syms);
                }
            }
            SymExpr::Vector { elements } => {
                for el in elements {
                    el.collect_symbols(syms);
                }
            }
            _ => {}
        }
    }
}

// --- Display (pretty-print) ---

impl fmt::Display for SymExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymExpr::Sym { name } => write!(f, "{}", name),
            SymExpr::Int { value } => write!(f, "{}", value),
            SymExpr::Rational { num, den } => write!(f, "{}/{}", num, den),
            SymExpr::Float { value } => {
                if value.fract() == 0.0 && value.abs() < 1e15 {
                    write!(f, "{:.1}", value)
                } else {
                    write!(f, "{}", value)
                }
            }
            SymExpr::Const { name } => match name {
                MathConst::Pi => write!(f, "pi"),
                MathConst::E => write!(f, "e"),
                MathConst::I => write!(f, "i"),
                MathConst::Infinity => write!(f, "inf"),
                MathConst::NegInfinity => write!(f, "-inf"),
            },
            SymExpr::BinOp { op, lhs, rhs } => {
                let (op_str, prec) = match op {
                    SymOp::Add => (" + ", 1),
                    SymOp::Sub => (" - ", 1),
                    SymOp::Mul => ("*", 2),
                    SymOp::Div => ("/", 2),
                    SymOp::Pow => ("^", 3),
                };

                let needs_lhs_parens = lhs_needs_parens(lhs, prec);
                let needs_rhs_parens = rhs_needs_parens(rhs, prec, *op);

                if needs_lhs_parens {
                    write!(f, "({})", lhs)?;
                } else {
                    write!(f, "{}", lhs)?;
                }

                write!(f, "{}", op_str)?;

                if needs_rhs_parens {
                    write!(f, "({})", rhs)?;
                } else {
                    write!(f, "{}", rhs)?;
                }
                Ok(())
            }
            SymExpr::Neg { expr } => {
                let needs_parens = matches!(
                    expr.as_ref(),
                    SymExpr::BinOp {
                        op: SymOp::Add | SymOp::Sub,
                        ..
                    }
                );
                if needs_parens {
                    write!(f, "-({})", expr)
                } else {
                    write!(f, "-{}", expr)
                }
            }
            SymExpr::Func { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            SymExpr::Vector { elements } => {
                write!(f, "[")?;
                for (i, el) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", el)?;
                }
                write!(f, "]")
            }
            SymExpr::Undefined => write!(f, "undefined"),
        }
    }
}

fn expr_precedence(expr: &SymExpr) -> u8 {
    match expr {
        SymExpr::BinOp { op, .. } => match op {
            SymOp::Add | SymOp::Sub => 1,
            SymOp::Mul | SymOp::Div => 2,
            SymOp::Pow => 3,
        },
        _ => 10, // atoms don't need parens
    }
}

fn lhs_needs_parens(lhs: &SymExpr, parent_prec: u8) -> bool {
    expr_precedence(lhs) < parent_prec
}

fn rhs_needs_parens(rhs: &SymExpr, parent_prec: u8, parent_op: SymOp) -> bool {
    let rhs_prec = expr_precedence(rhs);
    if rhs_prec < parent_prec {
        return true;
    }
    // For subtraction and division, same-precedence on RHS needs parens
    // e.g. a - (b - c), a / (b / c)
    if rhs_prec == parent_prec && matches!(parent_op, SymOp::Sub | SymOp::Div) {
        return true;
    }
    false
}

impl fmt::Display for MathConst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathConst::Pi => write!(f, "pi"),
            MathConst::E => write!(f, "e"),
            MathConst::I => write!(f, "i"),
            MathConst::Infinity => write!(f, "inf"),
            MathConst::NegInfinity => write!(f, "-inf"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_simple() {
        let expr = SymExpr::mul(SymExpr::int(2), SymExpr::sym("x"));
        assert_eq!(format!("{}", expr), "2*x");
    }

    #[test]
    fn test_display_precedence() {
        // 2*x + 1  (no parens needed)
        let expr = SymExpr::add(
            SymExpr::mul(SymExpr::int(2), SymExpr::sym("x")),
            SymExpr::int(1),
        );
        assert_eq!(format!("{}", expr), "2*x + 1");
    }

    #[test]
    fn test_display_parens_needed() {
        // 2*(x + 1)
        let expr = SymExpr::mul(
            SymExpr::int(2),
            SymExpr::add(SymExpr::sym("x"), SymExpr::int(1)),
        );
        assert_eq!(format!("{}", expr), "2*(x + 1)");
    }

    #[test]
    fn test_json_roundtrip() {
        let expr = SymExpr::add(
            SymExpr::mul(SymExpr::int(3), SymExpr::pow(SymExpr::sym("x"), SymExpr::int(2))),
            SymExpr::int(1),
        );
        let json = serde_json::to_string(&expr).unwrap();
        let back: SymExpr = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, back);
    }

    #[test]
    fn test_free_symbols() {
        let expr = SymExpr::add(
            SymExpr::mul(SymExpr::sym("x"), SymExpr::sym("y")),
            SymExpr::int(1),
        );
        assert_eq!(expr.free_symbols(), vec!["x", "y"]);
    }
}
