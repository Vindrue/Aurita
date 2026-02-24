use crate::lang::error::{LangError, LangResult};
use crate::lang::token::{Span, Token, TokenKind};

pub struct Lexer {
    source: Vec<char>,
    pos: usize,
    tokens: Vec<Token>,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Self {
            source: source.chars().collect(),
            pos: 0,
            tokens: Vec::new(),
        }
    }

    pub fn tokenize(mut self) -> LangResult<Vec<Token>> {
        while !self.is_at_end() {
            self.skip_whitespace_not_newline();
            if self.is_at_end() {
                break;
            }
            let token = self.next_token()?;
            // Insert implicit multiplication if applicable
            if let Some(prev) = self.tokens.last() {
                if prev.kind.can_end_implicit_mul() && token.kind.can_start_implicit_mul() {
                    // Don't insert implicit mul before '(' if previous token is an identifier
                    // (that's a function call, not multiplication)
                    let is_func_call = matches!(&prev.kind, TokenKind::Ident(_))
                        && matches!(&token.kind, TokenKind::LParen);
                    if !is_func_call {
                        let span = Span::new(prev.span.end, token.span.start);
                        self.tokens.push(Token::new(TokenKind::Star, span));
                    }
                }
            }
            self.tokens.push(token);
        }
        self.tokens
            .push(Token::new(TokenKind::Eof, Span::new(self.pos, self.pos)));
        Ok(self.tokens)
    }

    fn next_token(&mut self) -> LangResult<Token> {
        let start = self.pos;
        let ch = self.advance();

        match ch {
            '\n' => Ok(Token::new(TokenKind::Newline, Span::new(start, self.pos))),
            '+' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(Token::new(TokenKind::PlusEq, Span::new(start, self.pos)))
                } else if self.peek() == Some('/') && self.peek_at(1) == Some('-') {
                    self.advance(); // consume '/'
                    self.advance(); // consume '-'
                    Ok(Token::new(TokenKind::PlusMinus, Span::new(start, self.pos)))
                } else {
                    Ok(Token::new(TokenKind::Plus, Span::new(start, self.pos)))
                }
            }
            '-' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(Token::new(TokenKind::MinusEq, Span::new(start, self.pos)))
                } else {
                    Ok(Token::new(TokenKind::Minus, Span::new(start, self.pos)))
                }
            }
            '*' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(Token::new(TokenKind::StarEq, Span::new(start, self.pos)))
                } else {
                    Ok(Token::new(TokenKind::Star, Span::new(start, self.pos)))
                }
            }
            '/' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(Token::new(TokenKind::SlashEq, Span::new(start, self.pos)))
                } else if self.peek() == Some('/') {
                    // Line comment
                    while !self.is_at_end() && self.peek() != Some('\n') {
                        self.advance();
                    }
                    // Recurse to get next real token
                    if self.is_at_end() {
                        Ok(Token::new(TokenKind::Eof, Span::new(self.pos, self.pos)))
                    } else {
                        self.next_token()
                    }
                } else {
                    Ok(Token::new(TokenKind::Slash, Span::new(start, self.pos)))
                }
            }
            '^' => Ok(Token::new(TokenKind::Caret, Span::new(start, self.pos))),
            '%' => Ok(Token::new(TokenKind::Percent, Span::new(start, self.pos))),
            '(' => Ok(Token::new(TokenKind::LParen, Span::new(start, self.pos))),
            ')' => Ok(Token::new(TokenKind::RParen, Span::new(start, self.pos))),
            '[' => Ok(Token::new(TokenKind::LBracket, Span::new(start, self.pos))),
            ']' => Ok(Token::new(TokenKind::RBracket, Span::new(start, self.pos))),
            '{' => Ok(Token::new(TokenKind::LBrace, Span::new(start, self.pos))),
            '}' => Ok(Token::new(TokenKind::RBrace, Span::new(start, self.pos))),
            ',' => Ok(Token::new(TokenKind::Comma, Span::new(start, self.pos))),
            ';' => Ok(Token::new(TokenKind::Semicolon, Span::new(start, self.pos))),
            ':' => Ok(Token::new(TokenKind::Colon, Span::new(start, self.pos))),
            '|' => Ok(Token::new(TokenKind::Pipe, Span::new(start, self.pos))),
            '.' => {
                if self.peek() == Some('.') {
                    self.advance();
                    Ok(Token::new(TokenKind::DotDot, Span::new(start, self.pos)))
                } else {
                    Err(LangError::lex("unexpected '.'").with_span(Span::new(start, self.pos)))
                }
            }
            '=' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(Token::new(TokenKind::EqEq, Span::new(start, self.pos)))
                } else {
                    Ok(Token::new(TokenKind::Eq, Span::new(start, self.pos)))
                }
            }
            '!' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(Token::new(TokenKind::BangEq, Span::new(start, self.pos)))
                } else {
                    Err(
                        LangError::lex("unexpected '!', did you mean 'not'?")
                            .with_span(Span::new(start, self.pos)),
                    )
                }
            }
            '<' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(Token::new(TokenKind::LtEq, Span::new(start, self.pos)))
                } else {
                    Ok(Token::new(TokenKind::Lt, Span::new(start, self.pos)))
                }
            }
            '>' => {
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(Token::new(TokenKind::GtEq, Span::new(start, self.pos)))
                } else {
                    Ok(Token::new(TokenKind::Gt, Span::new(start, self.pos)))
                }
            }
            '#' => {
                // Line comment
                while !self.is_at_end() && self.peek() != Some('\n') {
                    self.advance();
                }
                if self.is_at_end() {
                    Ok(Token::new(TokenKind::Eof, Span::new(self.pos, self.pos)))
                } else {
                    self.next_token()
                }
            }
            '"' => self.read_string(start),
            c if c.is_ascii_digit() => self.read_number(start),
            c if is_ident_start(c) => self.read_identifier(start),
            // Unicode math operators
            '\u{00D7}' => Ok(Token::new(TokenKind::Star, Span::new(start, self.pos))),  // ×
            '\u{00F7}' => Ok(Token::new(TokenKind::Slash, Span::new(start, self.pos))), // ÷
            '\u{2264}' => Ok(Token::new(TokenKind::LtEq, Span::new(start, self.pos))),  // ≤
            '\u{2265}' => Ok(Token::new(TokenKind::GtEq, Span::new(start, self.pos))),  // ≥
            '\u{2260}' => Ok(Token::new(TokenKind::BangEq, Span::new(start, self.pos))),// ≠
            '\u{22C5}' => Ok(Token::new(TokenKind::Star, Span::new(start, self.pos))),  // ⋅
            '\u{00B1}' => Ok(Token::new(TokenKind::PlusMinus, Span::new(start, self.pos))), // ±
            _ => Err(
                LangError::lex(format!("unexpected character: '{}'", ch))
                    .with_span(Span::new(start, self.pos)),
            ),
        }
    }

    fn read_number(&mut self, start: usize) -> LangResult<Token> {
        // Read integer part
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        // Check for decimal point (but not '..' range operator)
        let is_float = self.peek() == Some('.')
            && self.peek_at(1) != Some('.');

        if is_float {
            self.advance(); // consume '.'
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() || c == '_' {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        // Check for scientific notation
        if self.peek() == Some('e') || self.peek() == Some('E') {
            self.advance();
            if self.peek() == Some('+') || self.peek() == Some('-') {
                self.advance();
            }
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
            let text: String = self.source[start..self.pos]
                .iter()
                .filter(|c| **c != '_')
                .collect();
            let val: f64 = text
                .parse()
                .map_err(|_| LangError::lex(format!("invalid number: {}", text)))?;
            return Ok(Token::new(
                TokenKind::Float(val),
                Span::new(start, self.pos),
            ));
        }

        let text: String = self.source[start..self.pos]
            .iter()
            .filter(|c| **c != '_')
            .collect();

        if is_float {
            let val: f64 = text
                .parse()
                .map_err(|_| LangError::lex(format!("invalid float: {}", text)))?;
            Ok(Token::new(
                TokenKind::Float(val),
                Span::new(start, self.pos),
            ))
        } else {
            let val: i64 = text
                .parse()
                .map_err(|_| LangError::lex(format!("invalid integer: {}", text)))?;
            Ok(Token::new(
                TokenKind::Integer(val),
                Span::new(start, self.pos),
            ))
        }
    }

    fn read_identifier(&mut self, start: usize) -> LangResult<Token> {
        while let Some(c) = self.peek() {
            if is_ident_continue(c) {
                self.advance();
            } else {
                break;
            }
        }

        let text: String = self.source[start..self.pos].iter().collect();
        let kind = match text.as_str() {
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "for" => TokenKind::For,
            "while" => TokenKind::While,
            "in" => TokenKind::In,
            "return" => TokenKind::Return,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "and" => TokenKind::And,
            "or" => TokenKind::Or,
            "not" => TokenKind::Not,
            _ => TokenKind::Ident(text),
        };

        Ok(Token::new(kind, Span::new(start, self.pos)))
    }

    fn read_string(&mut self, start: usize) -> LangResult<Token> {
        let mut value = String::new();
        while !self.is_at_end() {
            match self.peek() {
                Some('"') => {
                    self.advance();
                    return Ok(Token::new(
                        TokenKind::StringLit(value),
                        Span::new(start, self.pos),
                    ));
                }
                Some('\\') => {
                    self.advance();
                    match self.peek() {
                        Some('n') => {
                            self.advance();
                            value.push('\n');
                        }
                        Some('t') => {
                            self.advance();
                            value.push('\t');
                        }
                        Some('\\') => {
                            self.advance();
                            value.push('\\');
                        }
                        Some('"') => {
                            self.advance();
                            value.push('"');
                        }
                        _ => {
                            value.push('\\');
                        }
                    }
                }
                Some(c) => {
                    self.advance();
                    value.push(c);
                }
                None => break,
            }
        }
        Err(LangError::lex("unterminated string").with_span(Span::new(start, self.pos)))
    }

    fn skip_whitespace_not_newline(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() && c != '\n' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn advance(&mut self) -> char {
        let ch = self.source[self.pos];
        self.pos += 1;
        ch
    }

    fn peek(&self) -> Option<char> {
        self.source.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.source.get(self.pos + offset).copied()
    }

    fn is_at_end(&self) -> bool {
        self.pos >= self.source.len()
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(input: &str) -> Vec<TokenKind> {
        Lexer::new(input)
            .tokenize()
            .unwrap()
            .into_iter()
            .map(|t| t.kind)
            .filter(|k| !matches!(k, TokenKind::Eof | TokenKind::Newline))
            .collect()
    }

    #[test]
    fn test_basic_arithmetic() {
        assert_eq!(
            lex("3 + 4"),
            vec![TokenKind::Integer(3), TokenKind::Plus, TokenKind::Integer(4)]
        );
    }

    #[test]
    fn test_implicit_multiplication() {
        // 3x -> 3 * x
        assert_eq!(
            lex("3x"),
            vec![
                TokenKind::Integer(3),
                TokenKind::Star,
                TokenKind::Ident("x".into()),
            ]
        );
        // 2(x+1) -> 2 * (x + 1)
        let tokens = lex("2(x+1)");
        assert_eq!(tokens[0], TokenKind::Integer(2));
        assert_eq!(tokens[1], TokenKind::Star);
        assert_eq!(tokens[2], TokenKind::LParen);
    }

    #[test]
    fn test_no_implicit_mul_for_func_call() {
        // sin(x) should NOT insert * between sin and (
        let tokens = lex("sin(x)");
        assert_eq!(tokens[0], TokenKind::Ident("sin".into()));
        assert_eq!(tokens[1], TokenKind::LParen);
    }

    #[test]
    fn test_float() {
        assert_eq!(lex("3.14"), vec![TokenKind::Float(3.14)]);
    }

    #[test]
    fn test_range() {
        let tokens = lex("1..10");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Integer(1),
                TokenKind::DotDot,
                TokenKind::Integer(10),
            ]
        );
    }

    #[test]
    fn test_keywords() {
        let tokens = lex("if else for while in true false and or not");
        assert_eq!(
            tokens,
            vec![
                TokenKind::If,
                TokenKind::Else,
                TokenKind::For,
                TokenKind::While,
                TokenKind::In,
                TokenKind::True,
                TokenKind::False,
                TokenKind::And,
                TokenKind::Or,
                TokenKind::Not,
            ]
        );
    }

    #[test]
    fn test_comparison_operators() {
        let tokens = lex("== != <= >= < >");
        assert_eq!(
            tokens,
            vec![
                TokenKind::EqEq,
                TokenKind::BangEq,
                TokenKind::LtEq,
                TokenKind::GtEq,
                TokenKind::Lt,
                TokenKind::Gt,
            ]
        );
    }

    #[test]
    fn test_comment() {
        assert_eq!(
            lex("3 + 4 # comment"),
            vec![TokenKind::Integer(3), TokenKind::Plus, TokenKind::Integer(4)]
        );
    }

    #[test]
    fn test_string() {
        assert_eq!(
            lex("\"hello world\""),
            vec![TokenKind::StringLit("hello world".into())]
        );
    }

    #[test]
    fn test_scientific_notation() {
        assert_eq!(lex("1e10"), vec![TokenKind::Float(1e10)]);
        assert_eq!(lex("3.14e-2"), vec![TokenKind::Float(3.14e-2)]);
    }
}
