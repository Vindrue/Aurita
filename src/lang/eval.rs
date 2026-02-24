use crate::cas::backend::CasBackend;
use crate::cas::protocol::CasOp;
use crate::lang::ast::*;
use crate::lang::builtins::register_builtins;
use crate::lang::env::{Env, EnvRef};
use crate::lang::error::{LangError, LangResult};
use crate::lang::token::Span;
use crate::lang::types::*;
use crate::symbolic::expr::{SymExpr, SymOp};

/// Signals for control flow (break, continue, return).
#[derive(Debug)]
enum Signal {
    Return(Value),
    Break,
    Continue,
}

/// The tree-walking evaluator.
pub struct Evaluator {
    pub env: EnvRef,
    /// Output lines captured from `print()` calls.
    pub output: Vec<String>,
    /// Optional CAS backend for symbolic operations.
    pub cas: Option<CasBackend>,
}

impl Evaluator {
    pub fn new() -> Self {
        let env = Env::new_global();
        register_builtins(&env);
        Self {
            env,
            output: Vec::new(),
            cas: None,
        }
    }

    /// Try to connect to the SymPy backend.
    pub fn init_cas(&mut self, bridge_path: &str) {
        match CasBackend::spawn("sympy", "python3", &[bridge_path]) {
            Ok(backend) => {
                self.cas = Some(backend);
            }
            Err(e) => {
                self.output.push(format!("Warning: CAS backend unavailable: {}", e));
            }
        }
    }

    pub fn has_cas(&self) -> bool {
        self.cas.is_some()
    }

    /// Evaluate a program (list of statements). Returns the value of the last expression.
    pub fn eval_program(&mut self, stmts: &[Stmt]) -> LangResult<Value> {
        let mut last = Value::Unit;
        for stmt in stmts {
            match self.eval_stmt(stmt) {
                Ok(val) => last = val,
                Err(e) => return Err(e),
            }
        }
        Ok(last)
    }

    /// Evaluate a single statement.
    pub fn eval_stmt(&mut self, stmt: &Stmt) -> LangResult<Value> {
        match self.eval_stmt_inner(stmt) {
            Ok(val) => Ok(val),
            Err(EvalOutcome::Error(e)) => Err(e),
            Err(EvalOutcome::Signal(Signal::Return(v))) => Ok(v),
            Err(EvalOutcome::Signal(Signal::Break)) => {
                Err(LangError::eval("break outside of loop"))
            }
            Err(EvalOutcome::Signal(Signal::Continue)) => {
                Err(LangError::eval("continue outside of loop"))
            }
        }
    }

    fn eval_stmt_inner(&mut self, stmt: &Stmt) -> Result<Value, EvalOutcome> {
        match stmt {
            Stmt::Expr(expr) => self.eval_expr(expr).map_err(EvalOutcome::Error),

            Stmt::Assign { name, value, .. } => {
                let val = self.eval_expr(value).map_err(EvalOutcome::Error)?;
                self.env.borrow_mut().update(name, val.clone());
                Ok(val)
            }

            Stmt::FuncDef {
                name,
                params,
                body,
                ..
            } => {
                let func = Value::Function(Function::UserDefined {
                    name: name.clone(),
                    params: params.clone(),
                    body: body.clone(),
                    closure_env: self.env.clone(),
                });
                self.env.borrow_mut().set(name.clone(), func.clone());
                Ok(func)
            }

            Stmt::If {
                condition,
                then_body,
                else_body,
                ..
            } => {
                let cond = self.eval_expr(condition).map_err(EvalOutcome::Error)?;
                if cond.is_truthy() {
                    self.eval_block(then_body)
                } else if let Some(else_stmts) = else_body {
                    self.eval_block(else_stmts)
                } else {
                    Ok(Value::Unit)
                }
            }

            Stmt::For {
                var,
                iterable,
                body,
                ..
            } => {
                let iter_val = self.eval_expr(iterable).map_err(EvalOutcome::Error)?;
                let items = self.to_iterable(iter_val).map_err(EvalOutcome::Error)?;

                let mut last = Value::Unit;
                for item in items {
                    self.env.borrow_mut().set(var.clone(), item);
                    match self.eval_block(body) {
                        Ok(v) => last = v,
                        Err(EvalOutcome::Signal(Signal::Break)) => break,
                        Err(EvalOutcome::Signal(Signal::Continue)) => continue,
                        Err(e) => return Err(e),
                    }
                }
                Ok(last)
            }

            Stmt::While {
                condition, body, ..
            } => {
                let mut last = Value::Unit;
                loop {
                    let cond = self.eval_expr(condition).map_err(EvalOutcome::Error)?;
                    if !cond.is_truthy() {
                        break;
                    }
                    match self.eval_block(body) {
                        Ok(v) => last = v,
                        Err(EvalOutcome::Signal(Signal::Break)) => break,
                        Err(EvalOutcome::Signal(Signal::Continue)) => continue,
                        Err(e) => return Err(e),
                    }
                }
                Ok(last)
            }

            Stmt::Return(expr, _) => {
                let val = if let Some(e) = expr {
                    self.eval_expr(e).map_err(EvalOutcome::Error)?
                } else {
                    Value::Unit
                };
                Err(EvalOutcome::Signal(Signal::Return(val)))
            }

            Stmt::Break(_) => Err(EvalOutcome::Signal(Signal::Break)),
            Stmt::Continue(_) => Err(EvalOutcome::Signal(Signal::Continue)),
        }
    }

    fn eval_block(&mut self, stmts: &[Stmt]) -> Result<Value, EvalOutcome> {
        let mut last = Value::Unit;
        for stmt in stmts {
            last = self.eval_stmt_inner(stmt)?;
        }
        Ok(last)
    }

    fn eval_expr(&mut self, expr: &Expr) -> LangResult<Value> {
        match expr {
            Expr::Number(NumberLit::Int(n), _) => Ok(Value::Number(Number::Int(*n))),
            Expr::Number(NumberLit::Float(f), _) => Ok(Value::Number(Number::Float(*f))),

            Expr::StringLit(s, _) => Ok(Value::Str(s.clone())),
            Expr::Bool(b, _) => Ok(Value::Bool(*b)),

            // AUTO-SYMBOLIC: unknown identifiers become symbolic variables
            Expr::Ident(name, _span) => {
                match self.env.borrow().get(name) {
                    Some(val) => Ok(val),
                    None => Ok(Value::Symbolic(SymExpr::sym(name))),
                }
            }

            Expr::BinOp {
                op, lhs, rhs, span, ..
            } => {
                let left = self.eval_expr(lhs)?;
                let right = self.eval_expr(rhs)?;
                self.eval_binop(*op, &left, &right, *span)
            }

            Expr::UnaryOp {
                op, operand, span, ..
            } => {
                let val = self.eval_expr(operand)?;
                self.eval_unaryop(*op, &val, *span)
            }

            Expr::Call { func, args, span } => {
                // Check for CAS operations by name before evaluating func
                if let Expr::Ident(name, _) = func.as_ref() {
                    if let Some(result) = self.try_cas_call(name, args, *span)? {
                        return Ok(result);
                    }
                }

                let func_val = self.eval_expr(func)?;
                let mut arg_vals = Vec::with_capacity(args.len());
                for arg in args {
                    arg_vals.push(self.eval_expr(arg)?);
                }
                self.call_function(&func_val, &arg_vals, *span)
            }

            Expr::Vector(elements, _) => {
                let vals: LangResult<Vec<Value>> =
                    elements.iter().map(|e| self.eval_expr(e)).collect();
                Ok(Value::Vector(vals?))
            }

            Expr::Index {
                object,
                index,
                span,
            } => {
                let obj = self.eval_expr(object)?;
                let idx = self.eval_expr(index)?;
                self.eval_index(&obj, &idx, *span)
            }

            Expr::Range { start, end, .. } => {
                let s = self.eval_expr(start)?;
                let e = self.eval_expr(end)?;
                match (&s, &e) {
                    (Value::Number(Number::Int(a)), Value::Number(Number::Int(b))) => {
                        let items: Vec<Value> =
                            (*a..=*b).map(|i| Value::Number(Number::Int(i))).collect();
                        Ok(Value::Vector(items))
                    }
                    _ => {
                        let a = s.as_f64().ok_or_else(|| {
                            LangError::type_err("range bounds must be numbers")
                        })?;
                        let b = e.as_f64().ok_or_else(|| {
                            LangError::type_err("range bounds must be numbers")
                        })?;
                        let mut items = Vec::new();
                        let mut x = a;
                        while x <= b + 1e-10 {
                            items.push(Value::Number(Number::Float(x)));
                            x += 1.0;
                        }
                        Ok(Value::Vector(items))
                    }
                }
            }

            Expr::Lambda { params, body, .. } => Ok(Value::Function(Function::UserDefined {
                name: "<lambda>".to_string(),
                params: params.clone(),
                body: *body.clone(),
                closure_env: self.env.clone(),
            })),
        }
    }

    // --- Symbolic-aware binary operations ---

    fn eval_binop(
        &self,
        op: BinOpKind,
        left: &Value,
        right: &Value,
        span: Span,
    ) -> LangResult<Value> {
        // If either side is symbolic, promote to symbolic
        if left.is_symbolic() || right.is_symbolic() {
            if let Some(sym_op) = binop_to_symop(op) {
                let lhs = left.to_sym_expr().ok_or_else(|| {
                    LangError::type_err(format!("cannot use {} in symbolic expression", left.type_name()))
                        .with_span(span)
                })?;
                let rhs = right.to_sym_expr().ok_or_else(|| {
                    LangError::type_err(format!("cannot use {} in symbolic expression", right.type_name()))
                        .with_span(span)
                })?;
                return Ok(Value::Symbolic(SymExpr::BinOp {
                    op: sym_op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }));
            }
            // Comparison operators on symbolic values are not supported yet
            return Err(LangError::type_err(format!(
                "cannot apply {:?} to symbolic expressions (yet)",
                op
            )).with_span(span));
        }

        // Vector operations
        if let (Value::Vector(a), Value::Vector(b)) = (left, right) {
            return self.eval_vector_binop(op, a, b, span);
        }

        // Scalar * vector or vector * scalar
        if let (Value::Number(_), Value::Vector(v)) = (left, right) {
            if matches!(op, BinOpKind::Mul) {
                let result: LangResult<Vec<Value>> =
                    v.iter().map(|e| self.eval_binop(op, left, e, span)).collect();
                return Ok(Value::Vector(result?));
            }
        }
        if let (Value::Vector(v), Value::Number(_)) = (left, right) {
            if matches!(op, BinOpKind::Mul | BinOpKind::Div) {
                let result: LangResult<Vec<Value>> =
                    v.iter().map(|e| self.eval_binop(op, e, right, span)).collect();
                return Ok(Value::Vector(result?));
            }
        }

        // Numeric operations
        match (left, right) {
            (Value::Number(a), Value::Number(b)) => self.eval_numeric_binop(op, a, b, span),
            (Value::Str(a), Value::Str(b)) if op == BinOpKind::Add => {
                Ok(Value::Str(format!("{}{}", a, b)))
            }
            (Value::Bool(a), Value::Bool(b)) => match op {
                BinOpKind::And => Ok(Value::Bool(*a && *b)),
                BinOpKind::Or => Ok(Value::Bool(*a || *b)),
                BinOpKind::Eq => Ok(Value::Bool(a == b)),
                BinOpKind::Neq => Ok(Value::Bool(a != b)),
                _ => Err(LangError::type_err(format!(
                    "cannot apply {:?} to booleans",
                    op
                ))
                .with_span(span)),
            },
            _ => Err(LangError::type_err(format!(
                "cannot apply {:?} to {} and {}",
                op,
                left.type_name(),
                right.type_name()
            ))
            .with_span(span)),
        }
    }

    fn eval_numeric_binop(
        &self,
        op: BinOpKind,
        a: &Number,
        b: &Number,
        span: Span,
    ) -> LangResult<Value> {
        // Try integer arithmetic when both are ints
        if let (Number::Int(ai), Number::Int(bi)) = (a, b) {
            match op {
                BinOpKind::Add => return Ok(Value::Number(Number::Int(ai.wrapping_add(*bi)))),
                BinOpKind::Sub => return Ok(Value::Number(Number::Int(ai.wrapping_sub(*bi)))),
                BinOpKind::Mul => return Ok(Value::Number(Number::Int(ai.wrapping_mul(*bi)))),
                BinOpKind::Div => {
                    if *bi == 0 {
                        return Err(
                            LangError::new(crate::lang::error::ErrorKind::DivisionByZero, "division by zero")
                                .with_span(span),
                        );
                    }
                    if ai % bi == 0 {
                        return Ok(Value::Number(Number::Int(ai / bi)));
                    }
                }
                BinOpKind::Mod => {
                    if *bi == 0 {
                        return Err(
                            LangError::new(crate::lang::error::ErrorKind::DivisionByZero, "modulo by zero")
                                .with_span(span),
                        );
                    }
                    return Ok(Value::Number(Number::Int(ai % bi)));
                }
                BinOpKind::Pow => {
                    if *bi >= 0 && *bi <= 63 {
                        return Ok(Value::Number(Number::Int(ai.wrapping_pow(*bi as u32))));
                    }
                }
                BinOpKind::Eq => return Ok(Value::Bool(ai == bi)),
                BinOpKind::Neq => return Ok(Value::Bool(ai != bi)),
                BinOpKind::Lt => return Ok(Value::Bool(ai < bi)),
                BinOpKind::Gt => return Ok(Value::Bool(ai > bi)),
                BinOpKind::Leq => return Ok(Value::Bool(ai <= bi)),
                BinOpKind::Geq => return Ok(Value::Bool(ai >= bi)),
                _ => {}
            }
        }

        let af = a.as_f64();
        let bf = b.as_f64();
        match op {
            BinOpKind::Add => Ok(Value::Number(Number::Float(af + bf))),
            BinOpKind::Sub => Ok(Value::Number(Number::Float(af - bf))),
            BinOpKind::Mul => Ok(Value::Number(Number::Float(af * bf))),
            BinOpKind::Div => {
                if bf == 0.0 {
                    return Err(
                        LangError::new(crate::lang::error::ErrorKind::DivisionByZero, "division by zero")
                            .with_span(span),
                    );
                }
                Ok(Value::Number(Number::Float(af / bf)))
            }
            BinOpKind::Pow => Ok(Value::Number(Number::Float(af.powf(bf)))),
            BinOpKind::Mod => Ok(Value::Number(Number::Float(af % bf))),
            BinOpKind::Eq => Ok(Value::Bool(af == bf)),
            BinOpKind::Neq => Ok(Value::Bool(af != bf)),
            BinOpKind::Lt => Ok(Value::Bool(af < bf)),
            BinOpKind::Gt => Ok(Value::Bool(af > bf)),
            BinOpKind::Leq => Ok(Value::Bool(af <= bf)),
            BinOpKind::Geq => Ok(Value::Bool(af >= bf)),
            _ => Err(LangError::type_err(format!("unexpected operator {:?} for numbers", op))
                .with_span(span)),
        }
    }

    fn eval_vector_binop(
        &self,
        op: BinOpKind,
        a: &[Value],
        b: &[Value],
        span: Span,
    ) -> LangResult<Value> {
        if a.len() != b.len() {
            return Err(LangError::type_err(format!(
                "vector size mismatch: {} vs {}",
                a.len(),
                b.len()
            ))
            .with_span(span));
        }
        match op {
            BinOpKind::Add | BinOpKind::Sub => {
                let result: LangResult<Vec<Value>> = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| self.eval_binop(op, x, y, span))
                    .collect();
                Ok(Value::Vector(result?))
            }
            BinOpKind::Eq => Ok(Value::Bool(a == b)),
            BinOpKind::Neq => Ok(Value::Bool(a != b)),
            _ => Err(LangError::type_err(format!(
                "cannot apply {:?} element-wise to vectors",
                op
            ))
            .with_span(span)),
        }
    }

    fn eval_unaryop(&self, op: UnaryOpKind, val: &Value, span: Span) -> LangResult<Value> {
        match op {
            UnaryOpKind::Neg => match val {
                Value::Number(Number::Int(n)) => Ok(Value::Number(Number::Int(-n))),
                Value::Number(Number::Float(f)) => Ok(Value::Number(Number::Float(-f))),
                Value::Symbolic(expr) => Ok(Value::Symbolic(SymExpr::neg(expr.clone()))),
                Value::Vector(v) => {
                    let negated: LangResult<Vec<Value>> = v
                        .iter()
                        .map(|e| self.eval_unaryop(UnaryOpKind::Neg, e, span))
                        .collect();
                    Ok(Value::Vector(negated?))
                }
                _ => Err(LangError::type_err(format!(
                    "cannot negate {}",
                    val.type_name()
                ))
                .with_span(span)),
            },
            UnaryOpKind::Not => match val {
                Value::Bool(b) => Ok(Value::Bool(!b)),
                _ => Err(LangError::type_err(format!(
                    "cannot apply 'not' to {}",
                    val.type_name()
                ))
                .with_span(span)),
            },
        }
    }

    // --- CAS operation dispatch ---

    /// Try to handle a function call as a CAS operation.
    /// Returns Some(value) if handled, None if not a CAS operation.
    fn try_cas_call(&mut self, name: &str, args: &[Expr], span: Span) -> LangResult<Option<Value>> {
        match name {
            "dif" => Ok(Some(self.cas_differentiate(args, span)?)),
            "int" => Ok(Some(self.cas_integrate(args, span)?)),
            "solve" => Ok(Some(self.cas_solve(args, span)?)),
            "simplify" => Ok(Some(self.cas_simplify(args, span)?)),
            "expand" => Ok(Some(self.cas_expand(args, span)?)),
            "factor" => Ok(Some(self.cas_factor(args, span)?)),
            "lim" => Ok(Some(self.cas_limit(args, span)?)),
            "taylor" => Ok(Some(self.cas_taylor(args, span)?)),
            "tex" => Ok(Some(self.cas_latex(args, span)?)),
            _ => Ok(None),
        }
    }

    fn require_cas(&self, op_name: &str, span: Span) -> LangResult<()> {
        if self.cas.is_none() {
            Err(LangError::eval(format!(
                "{}: CAS backend not available (is Python/SymPy installed?)", op_name
            )).with_span(span))
        } else {
            Ok(())
        }
    }

    fn cas_differentiate(&mut self, args: &[Expr], span: Span) -> LangResult<Value> {
        self.require_cas("dif", span)?;
        if args.is_empty() || args.len() > 3 {
            return Err(LangError::arity("dif: expected 1-3 arguments: dif(expr, var?, order?)").with_span(span));
        }

        let expr_val = self.eval_expr(&args[0])?;
        let sym_expr = expr_val.to_sym_expr().ok_or_else(|| {
            LangError::type_err("dif: first argument must be a numeric or symbolic expression").with_span(span)
        })?;

        // Determine variable: explicit or auto-detect from free symbols
        let var = if args.len() >= 2 {
            let v = self.eval_expr(&args[1])?;
            match v {
                Value::Symbolic(SymExpr::Sym { name }) => name,
                _ => return Err(LangError::type_err("dif: second argument must be a variable").with_span(span)),
            }
        } else {
            let syms = sym_expr.free_symbols();
            if syms.len() == 1 {
                syms[0].clone()
            } else {
                return Err(LangError::eval(
                    "dif: expression has multiple free variables, specify which: dif(expr, var)"
                ).with_span(span));
            }
        };

        let order = if args.len() == 3 {
            let o = self.eval_expr(&args[2])?;
            o.as_int().ok_or_else(|| {
                LangError::type_err("dif: order must be an integer").with_span(span)
            })? as u32
        } else {
            1
        };

        let cas = self.cas.as_mut().unwrap();
        let result = cas.differentiate(&sym_expr, &var, order)
            .map_err(|e| LangError::eval(format!("dif: {}", e)).with_span(span))?;

        Ok(Value::Symbolic(result))
    }

    fn cas_integrate(&mut self, args: &[Expr], span: Span) -> LangResult<Value> {
        self.require_cas("int", span)?;
        if args.is_empty() || args.len() > 4 {
            return Err(LangError::arity("int: expected 1-4 arguments: int(expr, var?, lower?, upper?)").with_span(span));
        }

        let expr_val = self.eval_expr(&args[0])?;
        let sym_expr = expr_val.to_sym_expr().ok_or_else(|| {
            LangError::type_err("int: first argument must be an expression").with_span(span)
        })?;

        let var = if args.len() >= 2 {
            let v = self.eval_expr(&args[1])?;
            match v {
                Value::Symbolic(SymExpr::Sym { name }) => name,
                _ => return Err(LangError::type_err("int: second argument must be a variable").with_span(span)),
            }
        } else {
            let syms = sym_expr.free_symbols();
            if syms.len() == 1 {
                syms[0].clone()
            } else {
                return Err(LangError::eval(
                    "int: expression has multiple free variables, specify which: int(expr, var)"
                ).with_span(span));
            }
        };

        let (lower, upper) = if args.len() == 4 {
            let lo = self.eval_expr(&args[2])?.to_sym_expr().ok_or_else(|| {
                LangError::type_err("int: lower bound must be numeric or symbolic").with_span(span)
            })?;
            let hi = self.eval_expr(&args[3])?.to_sym_expr().ok_or_else(|| {
                LangError::type_err("int: upper bound must be numeric or symbolic").with_span(span)
            })?;
            (Some(lo), Some(hi))
        } else {
            (None, None)
        };

        let cas = self.cas.as_mut().unwrap();
        let result = cas.integrate(&sym_expr, &var, lower.as_ref(), upper.as_ref())
            .map_err(|e| LangError::eval(format!("int: {}", e)).with_span(span))?;

        Ok(Value::Symbolic(result))
    }

    fn cas_solve(&mut self, args: &[Expr], span: Span) -> LangResult<Value> {
        self.require_cas("solve", span)?;
        if args.is_empty() || args.len() > 2 {
            return Err(LangError::arity("solve: expected 1-2 arguments: solve(expr, var?)").with_span(span));
        }

        let expr_val = self.eval_expr(&args[0])?;

        // Handle vector of equations
        let (equations, all_syms) = match &expr_val {
            Value::Vector(eqs) => {
                let mut sym_eqs = Vec::new();
                let mut syms = Vec::new();
                for eq in eqs {
                    let se = eq.to_sym_expr().ok_or_else(|| {
                        LangError::type_err("solve: equation must be symbolic").with_span(span)
                    })?;
                    for s in se.free_symbols() {
                        if !syms.contains(&s) {
                            syms.push(s);
                        }
                    }
                    sym_eqs.push(se);
                }
                (sym_eqs, syms)
            }
            _ => {
                let se = expr_val.to_sym_expr().ok_or_else(|| {
                    LangError::type_err("solve: argument must be symbolic").with_span(span)
                })?;
                let syms = se.free_symbols();
                (vec![se], syms)
            }
        };

        let vars = if args.len() == 2 {
            let v = self.eval_expr(&args[1])?;
            match v {
                Value::Symbolic(SymExpr::Sym { name }) => vec![name],
                Value::Vector(items) => {
                    let mut names = Vec::new();
                    for item in items {
                        match item {
                            Value::Symbolic(SymExpr::Sym { name }) => names.push(name),
                            _ => return Err(LangError::type_err("solve: variables must be symbolic names").with_span(span)),
                        }
                    }
                    names
                }
                _ => return Err(LangError::type_err("solve: second argument must be a variable or vector of variables").with_span(span)),
            }
        } else if all_syms.len() == 1 {
            all_syms
        } else {
            return Err(LangError::eval(
                "solve: multiple free variables, specify which: solve(expr, var)"
            ).with_span(span));
        };

        let cas = self.cas.as_mut().unwrap();
        let results = cas.solve(&equations, &vars)
            .map_err(|e| LangError::eval(format!("solve: {}", e)).with_span(span))?;

        if results.len() == 1 {
            Ok(Value::Symbolic(results.into_iter().next().unwrap()))
        } else {
            Ok(Value::Vector(results.into_iter().map(Value::Symbolic).collect()))
        }
    }

    fn cas_simplify(&mut self, args: &[Expr], span: Span) -> LangResult<Value> {
        self.require_cas("simplify", span)?;
        if args.len() != 1 {
            return Err(LangError::arity("simplify: expected 1 argument").with_span(span));
        }
        let expr_val = self.eval_expr(&args[0])?;
        let sym_expr = expr_val.to_sym_expr().ok_or_else(|| {
            LangError::type_err("simplify: argument must be symbolic").with_span(span)
        })?;
        let cas = self.cas.as_mut().unwrap();
        let result = cas.simplify(&sym_expr)
            .map_err(|e| LangError::eval(format!("simplify: {}", e)).with_span(span))?;
        Ok(Value::Symbolic(result))
    }

    fn cas_expand(&mut self, args: &[Expr], span: Span) -> LangResult<Value> {
        self.require_cas("expand", span)?;
        if args.len() != 1 {
            return Err(LangError::arity("expand: expected 1 argument").with_span(span));
        }
        let expr_val = self.eval_expr(&args[0])?;
        let sym_expr = expr_val.to_sym_expr().ok_or_else(|| {
            LangError::type_err("expand: argument must be symbolic").with_span(span)
        })?;
        let cas = self.cas.as_mut().unwrap();
        let result = cas.expand(&sym_expr)
            .map_err(|e| LangError::eval(format!("expand: {}", e)).with_span(span))?;
        Ok(Value::Symbolic(result))
    }

    fn cas_factor(&mut self, args: &[Expr], span: Span) -> LangResult<Value> {
        self.require_cas("factor", span)?;
        if args.len() != 1 {
            return Err(LangError::arity("factor: expected 1 argument").with_span(span));
        }
        let expr_val = self.eval_expr(&args[0])?;
        let sym_expr = expr_val.to_sym_expr().ok_or_else(|| {
            LangError::type_err("factor: argument must be symbolic").with_span(span)
        })?;
        let cas = self.cas.as_mut().unwrap();
        let result = cas.factor(&sym_expr)
            .map_err(|e| LangError::eval(format!("factor: {}", e)).with_span(span))?;
        Ok(Value::Symbolic(result))
    }

    fn cas_limit(&mut self, args: &[Expr], span: Span) -> LangResult<Value> {
        self.require_cas("lim", span)?;
        if args.len() < 3 || args.len() > 4 {
            return Err(LangError::arity("lim: expected 3-4 arguments: lim(expr, var, point, dir?)").with_span(span));
        }
        let expr_val = self.eval_expr(&args[0])?;
        let sym_expr = expr_val.to_sym_expr().ok_or_else(|| {
            LangError::type_err("lim: first argument must be symbolic").with_span(span)
        })?;
        let var = match self.eval_expr(&args[1])? {
            Value::Symbolic(SymExpr::Sym { name }) => name,
            _ => return Err(LangError::type_err("lim: second argument must be a variable").with_span(span)),
        };
        let point = self.eval_expr(&args[2])?.to_sym_expr().ok_or_else(|| {
            LangError::type_err("lim: third argument must be numeric or symbolic").with_span(span)
        })?;
        let dir = if args.len() == 4 {
            match self.eval_expr(&args[3])? {
                Value::Str(s) => Some(s),
                _ => return Err(LangError::type_err("lim: direction must be \"+\" or \"-\"").with_span(span)),
            }
        } else {
            None
        };

        let cas = self.cas.as_mut().unwrap();
        let result = cas.limit(&sym_expr, &var, &point, dir.as_deref())
            .map_err(|e| LangError::eval(format!("lim: {}", e)).with_span(span))?;
        Ok(Value::Symbolic(result))
    }

    fn cas_taylor(&mut self, args: &[Expr], span: Span) -> LangResult<Value> {
        self.require_cas("taylor", span)?;
        if args.len() < 3 || args.len() > 4 {
            return Err(LangError::arity("taylor: expected 3-4 arguments: taylor(expr, var, point, order?)").with_span(span));
        }
        let expr_val = self.eval_expr(&args[0])?;
        let sym_expr = expr_val.to_sym_expr().ok_or_else(|| {
            LangError::type_err("taylor: first argument must be symbolic").with_span(span)
        })?;
        let var = match self.eval_expr(&args[1])? {
            Value::Symbolic(SymExpr::Sym { name }) => name,
            _ => return Err(LangError::type_err("taylor: second argument must be a variable").with_span(span)),
        };
        let point = self.eval_expr(&args[2])?.to_sym_expr().ok_or_else(|| {
            LangError::type_err("taylor: third argument must be numeric or symbolic").with_span(span)
        })?;
        let order = if args.len() == 4 {
            self.eval_expr(&args[3])?.as_int().ok_or_else(|| {
                LangError::type_err("taylor: order must be an integer").with_span(span)
            })? as u32
        } else {
            5
        };

        let cas = self.cas.as_mut().unwrap();
        let result = cas.taylor(&sym_expr, &var, &point, order)
            .map_err(|e| LangError::eval(format!("taylor: {}", e)).with_span(span))?;
        Ok(Value::Symbolic(result))
    }

    fn cas_latex(&mut self, args: &[Expr], span: Span) -> LangResult<Value> {
        self.require_cas("tex", span)?;
        if args.len() != 1 {
            return Err(LangError::arity("tex: expected 1 argument").with_span(span));
        }
        let expr_val = self.eval_expr(&args[0])?;
        let sym_expr = expr_val.to_sym_expr().ok_or_else(|| {
            LangError::type_err("tex: argument must be symbolic").with_span(span)
        })?;
        let cas = self.cas.as_mut().unwrap();
        let resp = cas.request(CasOp::Latex { expr: sym_expr })
            .map_err(|e| LangError::eval(format!("tex: {}", e)).with_span(span))?;
        let latex_str = resp.latex.unwrap_or_else(|| "?".to_string());
        Ok(Value::Str(latex_str))
    }

    // --- Function calls ---

    fn call_function(&mut self, func: &Value, args: &[Value], span: Span) -> LangResult<Value> {
        match func {
            Value::Function(Function::Builtin {
                name, arity, func, ..
            }) => {
                if !arity.accepts(args.len()) {
                    return Err(LangError::arity(format!(
                        "{}: expected {:?} arguments, got {}",
                        name,
                        arity,
                        args.len()
                    ))
                    .with_span(span));
                }

                // If any argument is symbolic, try numeric eval first (for constants),
                // otherwise build a symbolic function call
                if args.iter().any(|a| a.is_symbolic()) {
                    // Try to evaluate all args numerically (works for pi, e, 2*pi, etc.)
                    let numeric_args: Option<Vec<Value>> = args
                        .iter()
                        .map(|a| match a {
                            Value::Symbolic(expr) => {
                                crate::lang::builtins::numeric_eval_sym(expr)
                                    .map(|f| Value::Number(Number::Float(f)))
                            }
                            Value::Number(_) => Some(a.clone()),
                            _ => None,
                        })
                        .collect();

                    if let Some(num_args) = numeric_args {
                        // All args are evaluable constants — call the function numerically
                        let result = (func.0)(&num_args).map_err(|e| LangError::eval(e).with_span(span))?;
                        return Ok(result);
                    }

                    // Has free variables — build symbolic function call
                    let sym_args: Option<Vec<SymExpr>> = args.iter().map(|a| a.to_sym_expr()).collect();
                    if let Some(sym_args) = sym_args {
                        return Ok(Value::Symbolic(SymExpr::func(name, sym_args)));
                    }
                }

                let result = (func.0)(args).map_err(|e| LangError::eval(e).with_span(span))?;
                if name == "print" {
                    if let Value::Str(s) = &result {
                        self.output.push(s.clone());
                    }
                    return Ok(Value::Unit);
                }
                Ok(result)
            }
            Value::Function(Function::UserDefined {
                name,
                params,
                body,
                closure_env,
            }) => {
                if args.len() != params.len() {
                    return Err(LangError::arity(format!(
                        "{}: expected {} arguments, got {}",
                        name,
                        params.len(),
                        args.len()
                    ))
                    .with_span(span));
                }

                let call_env = Env::new_child(closure_env.clone());
                for (param, arg) in params.iter().zip(args.iter()) {
                    call_env.borrow_mut().set(param.clone(), arg.clone());
                }

                let prev_env = self.env.clone();
                self.env = call_env;
                let result = self.eval_expr(body);
                self.env = prev_env;
                result
            }
            _ => Err(
                LangError::type_err(format!("{} is not callable", func.type_name())).with_span(span)
            ),
        }
    }

    fn eval_index(&self, object: &Value, index: &Value, span: Span) -> LangResult<Value> {
        match object {
            Value::Vector(v) => {
                let idx = index
                    .as_int()
                    .ok_or_else(|| LangError::type_err("index must be an integer").with_span(span))?;
                let idx = if idx > 0 { (idx - 1) as usize } else {
                    return Err(LangError::eval("index must be positive (1-indexed)").with_span(span));
                };
                v.get(idx).cloned().ok_or_else(|| {
                    LangError::eval(format!(
                        "index {} out of bounds (length {})",
                        idx + 1,
                        v.len()
                    ))
                    .with_span(span)
                })
            }
            _ => Err(LangError::type_err(format!(
                "cannot index into {}",
                object.type_name()
            ))
            .with_span(span)),
        }
    }

    fn to_iterable(&self, val: Value) -> LangResult<Vec<Value>> {
        match val {
            Value::Vector(v) => Ok(v),
            _ => Err(LangError::type_err(format!(
                "cannot iterate over {}",
                val.type_name()
            ))),
        }
    }
}

/// Map Aurita BinOpKind to symbolic SymOp (only for arithmetic ops).
fn binop_to_symop(op: BinOpKind) -> Option<SymOp> {
    match op {
        BinOpKind::Add => Some(SymOp::Add),
        BinOpKind::Sub => Some(SymOp::Sub),
        BinOpKind::Mul => Some(SymOp::Mul),
        BinOpKind::Div => Some(SymOp::Div),
        BinOpKind::Pow => Some(SymOp::Pow),
        _ => None,
    }
}

/// Internal result type that distinguishes errors from control-flow signals.
#[derive(Debug)]
enum EvalOutcome {
    Error(LangError),
    Signal(Signal),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::lexer::Lexer;
    use crate::lang::parser::Parser;

    fn eval(input: &str) -> Value {
        let tokens = Lexer::new(input).tokenize().unwrap();
        let stmts = Parser::new(tokens).parse_program().unwrap();
        let mut evaluator = Evaluator::new();
        evaluator.eval_program(&stmts).unwrap()
    }

    fn eval_with(evaluator: &mut Evaluator, input: &str) -> Value {
        let tokens = Lexer::new(input).tokenize().unwrap();
        let stmts = Parser::new(tokens).parse_program().unwrap();
        evaluator.eval_program(&stmts).unwrap()
    }

    #[test]
    fn test_arithmetic() {
        assert_eq!(eval("3 + 4"), Value::Number(Number::Int(7)));
        assert_eq!(eval("10 - 3"), Value::Number(Number::Int(7)));
        assert_eq!(eval("6 * 7"), Value::Number(Number::Int(42)));
        assert_eq!(eval("10 / 2"), Value::Number(Number::Int(5)));
        assert_eq!(eval("2 ^ 10"), Value::Number(Number::Int(1024)));
    }

    #[test]
    fn test_float_division() {
        if let Value::Number(Number::Float(f)) = eval("10 / 3") {
            assert!((f - 3.333333333333333).abs() < 1e-10);
        } else {
            panic!("expected float");
        }
    }

    #[test]
    fn test_precedence() {
        assert_eq!(eval("3 + 4 * 2"), Value::Number(Number::Int(11)));
        assert_eq!(eval("(3 + 4) * 2"), Value::Number(Number::Int(14)));
    }

    #[test]
    fn test_variables() {
        let mut e = Evaluator::new();
        eval_with(&mut e, "a = 5");
        assert_eq!(eval_with(&mut e, "a + 3"), Value::Number(Number::Int(8)));
    }

    #[test]
    fn test_function_def_and_call() {
        let mut e = Evaluator::new();
        eval_with(&mut e, "f(x) = x^2 + 1");
        assert_eq!(eval_with(&mut e, "f(5)"), Value::Number(Number::Int(26)));
    }

    #[test]
    fn test_implicit_mul() {
        let mut e = Evaluator::new();
        eval_with(&mut e, "x = 3");
        assert_eq!(eval_with(&mut e, "2x"), Value::Number(Number::Int(6)));
    }

    #[test]
    fn test_builtin_sin() {
        if let Value::Number(Number::Float(f)) = eval("sin(0.0)") {
            assert!(f.abs() < 1e-10);
        } else {
            panic!("expected float");
        }
    }

    #[test]
    fn test_pi_constant() {
        // pi is a symbolic constant
        assert!(matches!(eval("pi"), Value::Symbolic(_)));
        // eval(pi) evaluates to float
        if let Value::Number(Number::Float(f)) = eval("eval(pi)") {
            assert!((f - std::f64::consts::PI).abs() < 1e-10);
        } else {
            panic!("expected float from eval(pi)");
        }
    }

    #[test]
    fn test_for_loop() {
        let mut e = Evaluator::new();
        eval_with(&mut e, "total = 0");
        eval_with(&mut e, "for i in 1..10 { total += i }");
        assert_eq!(eval_with(&mut e, "total"), Value::Number(Number::Int(55)));
    }

    #[test]
    fn test_while_loop() {
        let mut e = Evaluator::new();
        eval_with(&mut e, "n = 1");
        eval_with(&mut e, "while n < 100 { n *= 2 }");
        assert_eq!(eval_with(&mut e, "n"), Value::Number(Number::Int(128)));
    }

    #[test]
    fn test_if_else() {
        assert_eq!(eval("if true { 1 } else { 2 }"), Value::Number(Number::Int(1)));
        assert_eq!(eval("if false { 1 } else { 2 }"), Value::Number(Number::Int(2)));
    }

    #[test]
    fn test_vector() {
        let v = eval("[1, 2, 3]");
        assert!(matches!(v, Value::Vector(_)));
    }

    #[test]
    fn test_vector_add() {
        let v = eval("[1, 2, 3] + [4, 5, 6]");
        if let Value::Vector(items) = v {
            assert_eq!(items.len(), 3);
            assert_eq!(items[0], Value::Number(Number::Int(5)));
            assert_eq!(items[1], Value::Number(Number::Int(7)));
            assert_eq!(items[2], Value::Number(Number::Int(9)));
        } else {
            panic!("expected vector");
        }
    }

    #[test]
    fn test_scalar_vector_mul() {
        let v = eval("3 * [1, 2, 3]");
        if let Value::Vector(items) = v {
            assert_eq!(items[0], Value::Number(Number::Int(3)));
            assert_eq!(items[1], Value::Number(Number::Int(6)));
            assert_eq!(items[2], Value::Number(Number::Int(9)));
        } else {
            panic!("expected vector");
        }
    }

    #[test]
    fn test_index() {
        let mut e = Evaluator::new();
        eval_with(&mut e, "v = [10, 20, 30]");
        assert_eq!(eval_with(&mut e, "v[1]"), Value::Number(Number::Int(10)));
        assert_eq!(eval_with(&mut e, "v[3]"), Value::Number(Number::Int(30)));
    }

    #[test]
    fn test_nested_function() {
        let mut e = Evaluator::new();
        eval_with(&mut e, "f(x) = x^2");
        eval_with(&mut e, "g(x) = f(x) + 1");
        assert_eq!(eval_with(&mut e, "g(3)"), Value::Number(Number::Int(10)));
    }

    #[test]
    fn test_multiarg_function() {
        let mut e = Evaluator::new();
        eval_with(&mut e, "f(x, y) = x^2 + y^2");
        assert_eq!(eval_with(&mut e, "f(3, 4)"), Value::Number(Number::Int(25)));
    }

    #[test]
    fn test_comparison() {
        assert_eq!(eval("3 > 2"), Value::Bool(true));
        assert_eq!(eval("3 < 2"), Value::Bool(false));
        assert_eq!(eval("3 == 3"), Value::Bool(true));
        assert_eq!(eval("3 != 4"), Value::Bool(true));
    }

    #[test]
    fn test_boolean_logic() {
        assert_eq!(eval("true and false"), Value::Bool(false));
        assert_eq!(eval("true or false"), Value::Bool(true));
        assert_eq!(eval("not true"), Value::Bool(false));
    }

    // --- Symbolic tests ---

    #[test]
    fn test_auto_symbolic() {
        // Unknown variable 'x' becomes symbolic
        let v = eval("x");
        assert!(matches!(v, Value::Symbolic(SymExpr::Sym { .. })));
    }

    #[test]
    fn test_symbolic_arithmetic() {
        let v = eval("x + 1");
        assert!(matches!(v, Value::Symbolic(_)));
        assert_eq!(format!("{}", v), "x + 1");
    }

    #[test]
    fn test_symbolic_mul() {
        let v = eval("3x");
        assert!(matches!(v, Value::Symbolic(_)));
        assert_eq!(format!("{}", v), "3*x");
    }

    #[test]
    fn test_symbolic_function() {
        // sin(x) where x is undefined -> symbolic sin(x)
        let v = eval("sin(x)");
        assert!(matches!(v, Value::Symbolic(_)));
        assert_eq!(format!("{}", v), "sin(x)");
    }

    #[test]
    fn test_symbolic_complex_expr() {
        let v = eval("3x^2 + 2x + 1");
        assert!(matches!(v, Value::Symbolic(_)));
    }

    #[test]
    fn test_user_func_with_symbolic_arg() {
        let mut e = Evaluator::new();
        eval_with(&mut e, "f(x) = x^2 + 1");
        // f(t) where t is undefined -> symbolic
        let v = eval_with(&mut e, "f(t)");
        assert!(matches!(v, Value::Symbolic(_)));
    }
}
