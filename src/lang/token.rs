/// Source location span (byte offsets).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// A token produced by the lexer.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    Integer(i64),
    Float(f64),
    Ident(String),
    StringLit(String),

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Percent,

    // Assignment & comparison
    Eq,       // =
    EqEq,     // ==
    BangEq,   // !=
    Lt,       // <
    Gt,       // >
    LtEq,     // <=
    GtEq,     // >=

    // Compound assignment
    PlusEq,   // +=
    MinusEq,  // -=
    StarEq,   // *=
    SlashEq,  // /=

    // Delimiters
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Semicolon,
    Colon,
    Pipe,     // |
    DotDot,   // ..

    // Keywords
    If,
    Else,
    For,
    While,
    In,
    Return,
    Break,
    Continue,
    True,
    False,
    And,
    Or,
    Not,

    // Special
    Newline,
    Eof,
}

impl TokenKind {
    /// Whether this token can appear as the last token before implicit multiplication.
    pub fn can_end_implicit_mul(&self) -> bool {
        matches!(
            self,
            TokenKind::Integer(_)
                | TokenKind::Float(_)
                | TokenKind::Ident(_)
                | TokenKind::RParen
                | TokenKind::RBracket
                | TokenKind::True
                | TokenKind::False
        )
    }

    /// Whether this token can appear as the first token after implicit multiplication.
    pub fn can_start_implicit_mul(&self) -> bool {
        matches!(
            self,
            TokenKind::Integer(_)
                | TokenKind::Float(_)
                | TokenKind::Ident(_)
                | TokenKind::LParen
        )
    }
}
