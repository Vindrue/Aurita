use crate::lang::token::Span;
use std::fmt;

#[derive(Debug, Clone)]
pub struct LangError {
    pub kind: ErrorKind,
    pub message: String,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorKind {
    LexError,
    ParseError,
    EvalError,
    TypeError,
    ArityError,
    NameError,
    DivisionByZero,
    CasError,
}

impl LangError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            span: None,
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn lex(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::LexError, message)
    }

    pub fn parse(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::ParseError, message)
    }

    pub fn eval(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::EvalError, message)
    }

    pub fn type_err(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::TypeError, message)
    }

    pub fn arity(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::ArityError, message)
    }

    pub fn name(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::NameError, message)
    }
}

impl fmt::Display for LangError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for LangError {}

pub type LangResult<T> = Result<T, LangError>;
