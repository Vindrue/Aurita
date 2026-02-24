use crate::lang::token::Span;

/// Top-level statement.
#[derive(Debug, Clone)]
pub enum Stmt {
    /// Expression evaluated for its result.
    Expr(Expr),

    /// Variable binding: `a = 5`
    Assign {
        name: String,
        value: Expr,
        span: Span,
    },

    /// Function definition: `f(x) = 3x` or `f(x, y) = x^2 + y^2`
    FuncDef {
        name: String,
        params: Vec<String>,
        body: Expr,
        span: Span,
    },

    /// If/else.
    If {
        condition: Expr,
        then_body: Vec<Stmt>,
        else_body: Option<Vec<Stmt>>,
        span: Span,
    },

    /// For loop: `for i in 1..10 { ... }`
    For {
        var: String,
        iterable: Expr,
        body: Vec<Stmt>,
        span: Span,
    },

    /// While loop.
    While {
        condition: Expr,
        body: Vec<Stmt>,
        span: Span,
    },

    /// Return from function.
    Return(Option<Expr>, Span),

    /// Break.
    Break(Span),

    /// Continue.
    Continue(Span),
}

/// Expression node.
#[derive(Debug, Clone)]
pub enum Expr {
    /// Numeric literal: `42`, `3.14`
    Number(NumberLit, Span),

    /// String literal.
    StringLit(String, Span),

    /// Boolean literal.
    Bool(bool, Span),

    /// Variable / symbol reference: `x`, `theta`
    Ident(String, Span),

    /// Binary operation: `a + b`, `x^2`
    BinOp {
        op: BinOpKind,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },

    /// Unary operation: `-x`, `not p`
    UnaryOp {
        op: UnaryOpKind,
        operand: Box<Expr>,
        span: Span,
    },

    /// Function call: `sin(x)`, `dif(f(x), x)`
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },

    /// Vector/list literal: `[1, 2, 3]`
    Vector(Vec<Expr>, Span),

    /// Index: `v[2]`
    Index {
        object: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },

    /// Range: `1..10`
    Range {
        start: Box<Expr>,
        end: Box<Expr>,
        span: Span,
    },

    /// Unit annotation: `3[m/s^2]`, `(10 +/- 1)[kg]`
    UnitAnnotation {
        expr: Box<Expr>,
        unit_text: String,
        span: Span,
    },

    /// Lambda: `(x) -> x^2` (future)
    Lambda {
        params: Vec<String>,
        body: Box<Expr>,
        span: Span,
    },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Number(_, s) => *s,
            Expr::StringLit(_, s) => *s,
            Expr::Bool(_, s) => *s,
            Expr::Ident(_, s) => *s,
            Expr::BinOp { span, .. } => *span,
            Expr::UnaryOp { span, .. } => *span,
            Expr::Call { span, .. } => *span,
            Expr::Vector(_, s) => *s,
            Expr::Index { span, .. } => *span,
            Expr::Range { span, .. } => *span,
            Expr::UnitAnnotation { span, .. } => *span,
            Expr::Lambda { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NumberLit {
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    Eq,
    Neq,
    Lt,
    Gt,
    Leq,
    Geq,
    And,
    Or,
    PlusMinus,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOpKind {
    Neg,
    Not,
}
