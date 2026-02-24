use crate::lang::ast::Expr;
use crate::lang::env::EnvRef;
use crate::physics::units::UnitExpr;
use crate::plot::types::RenderedPlot;
use crate::symbolic::expr::SymExpr;
use std::fmt;

/// A numeric value with optional uncertainty and units.
#[derive(Debug, Clone)]
pub struct Quantity {
    pub value: f64,
    pub uncertainty: f64, // >= 0, 0.0 = exact
    pub unit: UnitExpr,
}

impl Quantity {
    pub fn new(value: f64, uncertainty: f64, unit: UnitExpr) -> Self {
        Self { value, uncertainty: uncertainty.abs(), unit }
    }

    pub fn exact(value: f64, unit: UnitExpr) -> Self {
        Self { value, uncertainty: 0.0, unit }
    }

    pub fn dimensionless(value: f64, uncertainty: f64) -> Self {
        Self { value, uncertainty: uncertainty.abs(), unit: UnitExpr::dimensionless() }
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format the value
        write!(f, "{}", format_number(self.value))?;

        // Show uncertainty if nonzero
        if self.uncertainty > 0.0 {
            write!(f, " +/- {}", format_number(self.uncertainty))?;
        }

        // Show units if not dimensionless
        if !self.unit.is_dimensionless() {
            write!(f, " [{}]", self.unit)?;
        }

        Ok(())
    }
}

/// Format a float nicely: avoid trailing zeros for whole numbers, use scientific for tiny/huge.
fn format_number(v: f64) -> String {
    if v == 0.0 {
        return "0.0".to_string();
    }
    let abs = v.abs();
    if abs >= 1e15 || (abs < 1e-3 && abs > 0.0) {
        format!("{:e}", v)
    } else if v.fract() == 0.0 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

/// Runtime value produced by the evaluator.
#[derive(Debug, Clone)]
pub enum Value {
    /// Concrete number.
    Number(Number),
    /// Symbolic expression (unevaluated / partially evaluated).
    Symbolic(SymExpr),
    /// Boolean.
    Bool(bool),
    /// String.
    Str(String),
    /// Vector/list of values.
    Vector(Vec<Value>),
    /// A callable function.
    Function(Function),
    /// A rendered plot image.
    Plot(RenderedPlot),
    /// Physical quantity with value, uncertainty, and units.
    Quantity(Quantity),
    /// Unit value (result of statements with no return value).
    Unit,
}

#[derive(Debug, Clone)]
pub enum Number {
    Int(i64),
    Float(f64),
}

impl Number {
    pub fn as_f64(&self) -> f64 {
        match self {
            Number::Int(n) => *n as f64,
            Number::Float(f) => *f,
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Number::Int(n) => *n == 0,
            Number::Float(f) => *f == 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Function {
    Builtin {
        name: String,
        arity: Arity,
        func: BuiltinFn,
    },
    UserDefined {
        name: String,
        params: Vec<String>,
        body: Expr,
        closure_env: EnvRef,
    },
}

#[derive(Debug, Clone)]
pub enum Arity {
    Exact(usize),
    Range(usize, usize),
    Variadic,
}

impl Arity {
    pub fn accepts(&self, n: usize) -> bool {
        match self {
            Arity::Exact(expected) => n == *expected,
            Arity::Range(min, max) => n >= *min && n <= *max,
            Arity::Variadic => true,
        }
    }
}

/// A built-in function pointer.
pub type BuiltinFnPtr = fn(&[Value]) -> Result<Value, String>;

/// Wrapper to make function pointers Debug + Clone.
#[derive(Clone)]
pub struct BuiltinFn(pub BuiltinFnPtr);

impl fmt::Debug for BuiltinFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<builtin>")
    }
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Number(Number::Int(_)) => "integer",
            Value::Number(Number::Float(_)) => "float",
            Value::Symbolic(_) => "symbolic",
            Value::Bool(_) => "boolean",
            Value::Str(_) => "string",
            Value::Vector(_) => "vector",
            Value::Function(_) => "function",
            Value::Plot(_) => "plot",
            Value::Quantity(_) => "quantity",
            Value::Unit => "unit",
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Number(Number::Int(n)) => *n != 0,
            Value::Number(Number::Float(f)) => *f != 0.0,
            Value::Symbolic(_) => true,
            Value::Str(s) => !s.is_empty(),
            Value::Vector(v) => !v.is_empty(),
            Value::Unit => false,
            Value::Function(_) => true,
            Value::Plot(_) => true,
            Value::Quantity(q) => q.value != 0.0,
        }
    }

    pub fn is_symbolic(&self) -> bool {
        matches!(self, Value::Symbolic(_))
    }

    pub fn to_sym_expr(&self) -> Option<SymExpr> {
        match self {
            Value::Symbolic(s) => Some(s.clone()),
            Value::Number(Number::Int(n)) => Some(SymExpr::int(*n)),
            Value::Number(Number::Float(f)) => Some(SymExpr::float(*f)),
            _ => None,
        }
    }

    pub fn as_number(&self) -> Option<&Number> {
        match self {
            Value::Number(n) => Some(n),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        self.as_number().map(|n| n.as_f64())
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Number(Number::Int(n)) => Some(*n),
            _ => None,
        }
    }

    /// Extract as a (value, uncertainty) pair. Works for Number (0 uncertainty) and Quantity.
    pub fn as_measurement(&self) -> Option<(f64, f64)> {
        match self {
            Value::Number(n) => Some((n.as_f64(), 0.0)),
            Value::Quantity(q) => Some((q.value, q.uncertainty)),
            _ => None,
        }
    }

    pub fn as_quantity(&self) -> Option<&Quantity> {
        match self {
            Value::Quantity(q) => Some(q),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(Number::Int(n)) => write!(f, "{}", n),
            Value::Number(Number::Float(n)) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    write!(f, "{:.1}", n)
                } else {
                    write!(f, "{}", n)
                }
            }
            Value::Symbolic(expr) => write!(f, "{}", expr),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Str(s) => write!(f, "{}", s),
            Value::Vector(v) => {
                write!(f, "[")?;
                for (i, val) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", val)?;
                }
                write!(f, "]")
            }
            Value::Function(Function::Builtin { name, .. }) => {
                write!(f, "<builtin: {}>", name)
            }
            Value::Function(Function::UserDefined { name, params, .. }) => {
                write!(f, "{}({})", name, params.join(", "))
            }
            Value::Plot(p) => {
                let labels: Vec<&str> = p.spec.series.iter().map(|s| s.label.as_str()).collect();
                write!(f, "[plot: {}]", labels.join(", "))
            }
            Value::Quantity(q) => write!(f, "{}", q),
            Value::Unit => write!(f, "()"),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Number(a), Value::Number(b)) => a.as_f64() == b.as_f64(),
            (Value::Symbolic(a), Value::Symbolic(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::Quantity(a), Value::Quantity(b)) => {
                a.value == b.value && a.uncertainty == b.uncertainty && a.unit == b.unit
            }
            (Value::Unit, Value::Unit) => true,
            _ => false,
        }
    }
}
