use crate::lang::ast::*;
use crate::lang::error::{LangError, LangResult};
use crate::lang::token::{Span, Token, TokenKind};

/// Pratt parser for the Aurita language.
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Parse all statements until EOF.
    pub fn parse_program(&mut self) -> LangResult<Vec<Stmt>> {
        let mut stmts = Vec::new();
        self.skip_newlines();
        while !self.is_at_end() {
            stmts.push(self.parse_statement()?);
            self.skip_newlines();
        }
        Ok(stmts)
    }

    /// Parse a single line / statement.
    pub fn parse_line(&mut self) -> LangResult<Stmt> {
        self.skip_newlines();
        if self.is_at_end() {
            return Err(LangError::parse("empty input"));
        }
        self.parse_statement()
    }

    fn parse_statement(&mut self) -> LangResult<Stmt> {
        match self.peek_kind() {
            TokenKind::If => self.parse_if(),
            TokenKind::For => self.parse_for(),
            TokenKind::While => self.parse_while(),
            TokenKind::Return => self.parse_return(),
            TokenKind::Break => {
                let span = self.advance().span;
                Ok(Stmt::Break(span))
            }
            TokenKind::Continue => {
                let span = self.advance().span;
                Ok(Stmt::Continue(span))
            }
            _ => self.parse_expr_or_assign(),
        }
    }

    /// Parse expression, then check if it's followed by `=` (assignment/funcdef).
    fn parse_expr_or_assign(&mut self) -> LangResult<Stmt> {
        let expr = self.parse_expr(0)?;

        // Check for assignment: `a = 5` or function def: `f(x) = 3x`
        if self.peek_kind() == TokenKind::Eq {
            let eq_span = self.advance().span;
            let value = self.parse_expr(0)?;
            let span = expr.span().merge(value.span());

            match &expr {
                // Simple assignment: `a = 5`
                Expr::Ident(name, _) => {
                    return Ok(Stmt::Assign {
                        name: name.clone(),
                        value,
                        span,
                    });
                }
                // Function definition: `f(x, y) = expr`
                Expr::Call { func, args, .. } => {
                    if let Expr::Ident(name, _) = func.as_ref() {
                        let params: LangResult<Vec<String>> = args
                            .iter()
                            .map(|a| match a {
                                Expr::Ident(n, _) => Ok(n.clone()),
                                _ => Err(LangError::parse(
                                    "function parameters must be identifiers",
                                )
                                .with_span(a.span())),
                            })
                            .collect();
                        return Ok(Stmt::FuncDef {
                            name: name.clone(),
                            params: params?,
                            body: value,
                            span,
                        });
                    }
                    return Err(
                        LangError::parse("invalid left-hand side of assignment").with_span(eq_span)
                    );
                }
                _ => {
                    return Err(
                        LangError::parse("invalid left-hand side of assignment").with_span(eq_span)
                    );
                }
            }
        }

        // Compound assignment: +=, -=, *=, /=
        let compound_op = match self.peek_kind() {
            TokenKind::PlusEq => Some(BinOpKind::Add),
            TokenKind::MinusEq => Some(BinOpKind::Sub),
            TokenKind::StarEq => Some(BinOpKind::Mul),
            TokenKind::SlashEq => Some(BinOpKind::Div),
            _ => None,
        };
        if let Some(op) = compound_op {
            let op_span = self.advance().span;
            let rhs = self.parse_expr(0)?;
            let span = expr.span().merge(rhs.span());
            if let Expr::Ident(name, _) = &expr {
                return Ok(Stmt::Assign {
                    name: name.clone(),
                    value: Expr::BinOp {
                        op,
                        lhs: Box::new(expr),
                        rhs: Box::new(rhs),
                        span,
                    },
                    span,
                });
            }
            return Err(
                LangError::parse("compound assignment target must be a variable").with_span(op_span)
            );
        }

        self.skip_terminators();
        Ok(Stmt::Expr(expr))
    }

    /// Pratt parser: parse expression with given minimum binding power.
    fn parse_expr(&mut self, min_bp: u8) -> LangResult<Expr> {
        let mut lhs = self.parse_prefix()?;

        loop {
            // Check for postfix: function call `(`, index `[`
            match self.peek_kind() {
                TokenKind::LParen => {
                    if let Expr::Ident(_, _) = &lhs {
                        lhs = self.parse_call(lhs)?;
                        continue;
                    }
                }
                TokenKind::LBracket => {
                    lhs = self.parse_index(lhs)?;
                    continue;
                }
                _ => {}
            }

            // Check for infix operator
            let (op, left_bp, right_bp) = match self.peek_kind() {
                TokenKind::Or => (BinOpKind::Or, 1, 2),
                TokenKind::And => (BinOpKind::And, 3, 4),
                TokenKind::EqEq => (BinOpKind::Eq, 5, 6),
                TokenKind::BangEq => (BinOpKind::Neq, 5, 6),
                TokenKind::Lt => (BinOpKind::Lt, 7, 8),
                TokenKind::Gt => (BinOpKind::Gt, 7, 8),
                TokenKind::LtEq => (BinOpKind::Leq, 7, 8),
                TokenKind::GtEq => (BinOpKind::Geq, 7, 8),
                TokenKind::DotDot => {
                    // Range operator at very low precedence
                    if min_bp > 1 {
                        break;
                    }
                    self.advance();
                    let rhs = self.parse_expr(2)?;
                    let span = lhs.span().merge(rhs.span());
                    lhs = Expr::Range {
                        start: Box::new(lhs),
                        end: Box::new(rhs),
                        span,
                    };
                    continue;
                }
                TokenKind::Plus => (BinOpKind::Add, 9, 10),
                TokenKind::Minus => (BinOpKind::Sub, 9, 10),
                TokenKind::Star => (BinOpKind::Mul, 11, 12),
                TokenKind::Slash => (BinOpKind::Div, 11, 12),
                TokenKind::Percent => (BinOpKind::Mod, 11, 12),
                TokenKind::Caret => (BinOpKind::Pow, 16, 15), // right-associative
                _ => break,
            };

            if left_bp < min_bp {
                break;
            }

            self.advance(); // consume operator
            let rhs = self.parse_expr(right_bp)?;
            let span = lhs.span().merge(rhs.span());
            lhs = Expr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }

        Ok(lhs)
    }

    /// Parse prefix expression (atom or unary operator).
    fn parse_prefix(&mut self) -> LangResult<Expr> {
        match self.peek_kind() {
            TokenKind::Integer(_) | TokenKind::Float(_) => self.parse_number(),
            TokenKind::StringLit(_) => self.parse_string(),
            TokenKind::True => {
                let span = self.advance().span;
                Ok(Expr::Bool(true, span))
            }
            TokenKind::False => {
                let span = self.advance().span;
                Ok(Expr::Bool(false, span))
            }
            TokenKind::Ident(_) => self.parse_ident(),
            TokenKind::LParen => self.parse_grouped(),
            TokenKind::LBracket => self.parse_vector(),
            TokenKind::Minus => {
                let op_span = self.advance().span;
                let operand = self.parse_expr(13)?; // unary - binds tighter than + -
                let span = op_span.merge(operand.span());
                Ok(Expr::UnaryOp {
                    op: UnaryOpKind::Neg,
                    operand: Box::new(operand),
                    span,
                })
            }
            TokenKind::Not => {
                let op_span = self.advance().span;
                let operand = self.parse_expr(3)?;
                let span = op_span.merge(operand.span());
                Ok(Expr::UnaryOp {
                    op: UnaryOpKind::Not,
                    operand: Box::new(operand),
                    span,
                })
            }
            TokenKind::Pipe => self.parse_abs(),
            _ => {
                let tok = self.peek();
                Err(LangError::parse(format!(
                    "expected expression, found {:?}",
                    tok.kind
                ))
                .with_span(tok.span))
            }
        }
    }

    fn parse_number(&mut self) -> LangResult<Expr> {
        let tok = self.advance();
        match tok.kind {
            TokenKind::Integer(n) => Ok(Expr::Number(NumberLit::Int(n), tok.span)),
            TokenKind::Float(f) => Ok(Expr::Number(NumberLit::Float(f), tok.span)),
            _ => unreachable!(),
        }
    }

    fn parse_string(&mut self) -> LangResult<Expr> {
        let tok = self.advance();
        if let TokenKind::StringLit(s) = tok.kind {
            Ok(Expr::StringLit(s, tok.span))
        } else {
            unreachable!()
        }
    }

    fn parse_ident(&mut self) -> LangResult<Expr> {
        let tok = self.advance();
        if let TokenKind::Ident(name) = tok.kind {
            Ok(Expr::Ident(name, tok.span))
        } else {
            unreachable!()
        }
    }

    fn parse_grouped(&mut self) -> LangResult<Expr> {
        self.expect(TokenKind::LParen)?;
        let expr = self.parse_expr(0)?;
        self.expect(TokenKind::RParen)?;
        Ok(expr)
    }

    fn parse_vector(&mut self) -> LangResult<Expr> {
        let start = self.expect(TokenKind::LBracket)?.span;
        let mut elements = Vec::new();

        if self.peek_kind() != TokenKind::RBracket {
            elements.push(self.parse_expr(0)?);
            while self.peek_kind() == TokenKind::Comma {
                self.advance();
                if self.peek_kind() == TokenKind::RBracket {
                    break; // trailing comma
                }
                elements.push(self.parse_expr(0)?);
            }
        }

        let end = self.expect(TokenKind::RBracket)?.span;
        Ok(Expr::Vector(elements, start.merge(end)))
    }

    fn parse_call(&mut self, func: Expr) -> LangResult<Expr> {
        let start = func.span();
        self.expect(TokenKind::LParen)?;
        let mut args = Vec::new();

        if self.peek_kind() != TokenKind::RParen {
            args.push(self.parse_expr(0)?);
            while self.peek_kind() == TokenKind::Comma {
                self.advance();
                if self.peek_kind() == TokenKind::RParen {
                    break;
                }
                args.push(self.parse_expr(0)?);
            }
        }

        let end = self.expect(TokenKind::RParen)?.span;
        Ok(Expr::Call {
            func: Box::new(func),
            args,
            span: start.merge(end),
        })
    }

    fn parse_index(&mut self, object: Expr) -> LangResult<Expr> {
        let start = object.span();
        self.expect(TokenKind::LBracket)?;
        let index = self.parse_expr(0)?;
        let end = self.expect(TokenKind::RBracket)?.span;
        Ok(Expr::Index {
            object: Box::new(object),
            index: Box::new(index),
            span: start.merge(end),
        })
    }

    fn parse_abs(&mut self) -> LangResult<Expr> {
        let start = self.expect(TokenKind::Pipe)?.span;
        let inner = self.parse_expr(0)?;
        let end = self.expect(TokenKind::Pipe)?.span;
        // Desugar |x| to abs(x)
        Ok(Expr::Call {
            func: Box::new(Expr::Ident("abs".to_string(), start)),
            args: vec![inner],
            span: start.merge(end),
        })
    }

    fn parse_if(&mut self) -> LangResult<Stmt> {
        let start = self.expect(TokenKind::If)?.span;
        let condition = self.parse_expr(0)?;
        let then_body = self.parse_block()?;

        let else_body = if self.peek_kind() == TokenKind::Else {
            self.advance();
            if self.peek_kind() == TokenKind::If {
                // else if => wrap in a single-element vec
                let elif = self.parse_if()?;
                Some(vec![elif])
            } else {
                Some(self.parse_block()?)
            }
        } else {
            None
        };

        let end_span = else_body
            .as_ref()
            .and_then(|b| b.last())
            .map(|s| stmt_span(s))
            .unwrap_or(start);

        Ok(Stmt::If {
            condition,
            then_body,
            else_body,
            span: start.merge(end_span),
        })
    }

    fn parse_for(&mut self) -> LangResult<Stmt> {
        let start = self.expect(TokenKind::For)?.span;
        let var_tok = self.advance();
        let var = match var_tok.kind {
            TokenKind::Ident(name) => name,
            _ => {
                return Err(
                    LangError::parse("expected variable name after 'for'").with_span(var_tok.span)
                )
            }
        };
        self.expect(TokenKind::In)?;
        let iterable = self.parse_expr(0)?;
        let body = self.parse_block()?;
        Ok(Stmt::For {
            var,
            iterable,
            body,
            span: start,
        })
    }

    fn parse_while(&mut self) -> LangResult<Stmt> {
        let start = self.expect(TokenKind::While)?.span;
        let condition = self.parse_expr(0)?;
        let body = self.parse_block()?;
        Ok(Stmt::While {
            condition,
            body,
            span: start,
        })
    }

    fn parse_return(&mut self) -> LangResult<Stmt> {
        let span = self.expect(TokenKind::Return)?.span;
        if self.is_at_end()
            || matches!(
                self.peek_kind(),
                TokenKind::Newline | TokenKind::Semicolon | TokenKind::RBrace | TokenKind::Eof
            )
        {
            Ok(Stmt::Return(None, span))
        } else {
            let expr = self.parse_expr(0)?;
            Ok(Stmt::Return(Some(expr), span))
        }
    }

    /// Parse a `{ ... }` block of statements.
    fn parse_block(&mut self) -> LangResult<Vec<Stmt>> {
        self.skip_newlines();
        self.expect(TokenKind::LBrace)?;
        self.skip_newlines();

        let mut stmts = Vec::new();
        while self.peek_kind() != TokenKind::RBrace && !self.is_at_end() {
            stmts.push(self.parse_statement()?);
            self.skip_newlines();
        }

        self.expect(TokenKind::RBrace)?;
        Ok(stmts)
    }

    // --- Token helpers ---

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn peek_kind(&self) -> TokenKind {
        self.tokens[self.pos].kind.clone()
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens[self.pos].clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, kind: TokenKind) -> LangResult<Token> {
        let tok = self.peek().clone();
        if std::mem::discriminant(&tok.kind) == std::mem::discriminant(&kind) {
            Ok(self.advance())
        } else {
            Err(LangError::parse(format!(
                "expected {:?}, found {:?}",
                kind, tok.kind
            ))
            .with_span(tok.span))
        }
    }

    fn is_at_end(&self) -> bool {
        matches!(self.tokens[self.pos].kind, TokenKind::Eof)
    }

    fn skip_newlines(&mut self) {
        while matches!(
            self.peek_kind(),
            TokenKind::Newline | TokenKind::Semicolon
        ) {
            self.advance();
        }
    }

    fn skip_terminators(&mut self) {
        if matches!(
            self.peek_kind(),
            TokenKind::Newline | TokenKind::Semicolon
        ) {
            self.advance();
        }
    }
}

fn stmt_span(stmt: &Stmt) -> Span {
    match stmt {
        Stmt::Expr(e) => e.span(),
        Stmt::Assign { span, .. } => *span,
        Stmt::FuncDef { span, .. } => *span,
        Stmt::If { span, .. } => *span,
        Stmt::For { span, .. } => *span,
        Stmt::While { span, .. } => *span,
        Stmt::Return(_, span) => *span,
        Stmt::Break(span) => *span,
        Stmt::Continue(span) => *span,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::lexer::Lexer;

    fn parse(input: &str) -> Vec<Stmt> {
        let tokens = Lexer::new(input).tokenize().unwrap();
        Parser::new(tokens).parse_program().unwrap()
    }

    fn parse_expr(input: &str) -> Expr {
        let stmts = parse(input);
        match stmts.into_iter().next().unwrap() {
            Stmt::Expr(e) => e,
            other => panic!("expected Expr, got {:?}", other),
        }
    }

    #[test]
    fn test_simple_arithmetic() {
        let expr = parse_expr("3 + 4 * 2");
        // Should be Add(3, Mul(4, 2)) due to precedence
        match expr {
            Expr::BinOp {
                op: BinOpKind::Add,
                lhs,
                rhs,
                ..
            } => {
                assert!(matches!(*lhs, Expr::Number(NumberLit::Int(3), _)));
                assert!(matches!(
                    *rhs,
                    Expr::BinOp {
                        op: BinOpKind::Mul,
                        ..
                    }
                ));
            }
            _ => panic!("unexpected: {:?}", expr),
        }
    }

    #[test]
    fn test_exponentiation_right_assoc() {
        let expr = parse_expr("2^3^4");
        // Should be Pow(2, Pow(3, 4)) — right-associative
        match expr {
            Expr::BinOp {
                op: BinOpKind::Pow,
                rhs,
                ..
            } => {
                assert!(matches!(
                    *rhs,
                    Expr::BinOp {
                        op: BinOpKind::Pow,
                        ..
                    }
                ));
            }
            _ => panic!("unexpected: {:?}", expr),
        }
    }

    #[test]
    fn test_assignment() {
        let stmts = parse("a = 5");
        match &stmts[0] {
            Stmt::Assign { name, .. } => assert_eq!(name, "a"),
            other => panic!("expected Assign, got {:?}", other),
        }
    }

    #[test]
    fn test_func_def() {
        let stmts = parse("f(x) = 3x + 1");
        match &stmts[0] {
            Stmt::FuncDef { name, params, .. } => {
                assert_eq!(name, "f");
                assert_eq!(params, &["x"]);
            }
            other => panic!("expected FuncDef, got {:?}", other),
        }
    }

    #[test]
    fn test_func_call() {
        let expr = parse_expr("sin(3.14)");
        match expr {
            Expr::Call { func, args, .. } => {
                assert!(matches!(*func, Expr::Ident(ref n, _) if n == "sin"));
                assert_eq!(args.len(), 1);
            }
            _ => panic!("unexpected: {:?}", expr),
        }
    }

    #[test]
    fn test_vector() {
        let expr = parse_expr("[1, 2, 3]");
        match expr {
            Expr::Vector(elements, _) => assert_eq!(elements.len(), 3),
            _ => panic!("unexpected: {:?}", expr),
        }
    }

    #[test]
    fn test_unary_neg() {
        let expr = parse_expr("-x");
        assert!(matches!(
            expr,
            Expr::UnaryOp {
                op: UnaryOpKind::Neg,
                ..
            }
        ));
    }

    #[test]
    fn test_implicit_mul() {
        // 3x should parse as 3 * x
        let expr = parse_expr("3x");
        assert!(matches!(
            expr,
            Expr::BinOp {
                op: BinOpKind::Mul,
                ..
            }
        ));
    }

    #[test]
    fn test_for_loop() {
        let stmts = parse("for i in 1..10 { a = i }");
        assert!(matches!(&stmts[0], Stmt::For { var, .. } if var == "i"));
    }

    #[test]
    fn test_if_else() {
        let stmts = parse("if x > 0 { 1 } else { 2 }");
        assert!(matches!(&stmts[0], Stmt::If { else_body: Some(_), .. }));
    }

    #[test]
    fn test_range() {
        let expr = parse_expr("1..10");
        assert!(matches!(expr, Expr::Range { .. }));
    }
}
