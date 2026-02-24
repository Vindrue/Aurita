use crate::cas::manager::{CasManager, CasResult, RoutingMode};
use crate::cas::protocol::PlotSeriesData;
use crate::lang::ast::*;
use crate::lang::builtins::{numeric_eval_sym, register_builtins};
use crate::lang::env::{Env, EnvRef};
use crate::lang::error::{LangError, LangResult};
use crate::lang::token::Span;
use crate::lang::types::*;
use crate::plot::render::render_plot;
use crate::plot::types::*;
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
    /// Optional CAS manager for symbolic operations (multiple backends).
    pub cas_manager: Option<CasManager>,
}

impl Evaluator {
    pub fn new() -> Self {
        let env = Env::new_global();
        register_builtins(&env);
        Self {
            env,
            output: Vec::new(),
            cas_manager: None,
        }
    }

    /// Initialize CAS backends.
    pub fn init_cas(&mut self, sympy_bridge: Option<&str>, maxima_bridge: Option<&str>) {
        let mut manager = CasManager::new();
        if let Some(path) = sympy_bridge {
            if let Err(e) = manager.add_backend("sympy", "python3", &[path]) {
                self.output.push(format!("Warning: SymPy unavailable: {}", e));
            }
        }
        if let Some(path) = maxima_bridge {
            if let Err(e) = manager.add_backend("maxima", "python3", &[path]) {
                self.output.push(format!("Warning: Maxima unavailable: {}", e));
            }
        }
        if manager.has_any_backend() {
            self.cas_manager = Some(manager);
        }
    }

    pub fn has_cas(&self) -> bool {
        self.cas_manager.is_some()
    }

    /// Get the CAS status string for the status bar.
    pub fn cas_status(&self) -> String {
        match &self.cas_manager {
            Some(m) => m.status_display(),
            None => "Numeric".to_string(),
        }
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
            "plot" => Ok(Some(self.eval_plot(args, span)?)),
            "backend" => Ok(Some(self.set_backend(args, span)?)),
            "using" => Ok(Some(self.eval_using(args, span)?)),
            _ => Ok(None),
        }
    }

    fn require_cas(&self, op_name: &str, span: Span) -> LangResult<()> {
        if self.cas_manager.is_none() {
            Err(LangError::eval(format!(
                "{}: CAS backend not available (is Python/SymPy installed?)", op_name
            )).with_span(span))
        } else {
            Ok(())
        }
    }

    /// Convert a CasResult to a Value.
    fn cas_result_to_value(&self, result: CasResult) -> Value {
        match result {
            CasResult::Single(expr) | CasResult::Agreed(expr) => Value::Symbolic(expr),
            CasResult::Multiple(exprs) | CasResult::AgreedMultiple(exprs) => {
                if exprs.len() == 1 {
                    Value::Symbolic(exprs.into_iter().next().unwrap())
                } else {
                    Value::Vector(exprs.into_iter().map(Value::Symbolic).collect())
                }
            }
            CasResult::Latex(s) => Value::Str(s),
            CasResult::Disagreed { results } => {
                let parts: Vec<String> = results
                    .iter()
                    .map(|(name, expr)| format!("{}: {}", name, expr))
                    .collect();
                Value::Str(parts.join(" | "))
            }
            CasResult::DisagreedMultiple { results } => {
                let parts: Vec<String> = results
                    .iter()
                    .map(|(name, exprs)| {
                        let vals: Vec<String> = exprs.iter().map(|e| format!("{}", e)).collect();
                        format!("{}: [{}]", name, vals.join(", "))
                    })
                    .collect();
                Value::Str(parts.join(" | "))
            }
        }
    }

    /// Set the active CAS backend: backend("sympy"), backend("maxima"), backend("both")
    fn set_backend(&mut self, args: &[Expr], span: Span) -> LangResult<Value> {
        self.require_cas("backend", span)?;
        if args.len() != 1 {
            return Err(LangError::arity("backend: expected 1 argument: backend(\"sympy\"|\"maxima\"|\"both\")").with_span(span));
        }
        let val = self.eval_expr(&args[0])?;
        let name = match &val {
            Value::Str(s) => s.clone(),
            Value::Symbolic(SymExpr::Sym { name }) => name.clone(),
            _ => return Err(LangError::type_err("backend: argument must be a string").with_span(span)),
        };

        let mode = match name.as_str() {
            "both" => RoutingMode::Both,
            other => RoutingMode::Single(other.to_string()),
        };

        let manager = self.cas_manager.as_mut().unwrap();
        manager.set_routing(mode)
            .map_err(|e| LangError::eval(format!("backend: {}", e)).with_span(span))?;

        Ok(Value::Str(format!("Backend set to {}", manager.status_display())))
    }

    /// Evaluate an expression with a specific backend: using("maxima", dif(x^2, x))
    fn eval_using(&mut self, args: &[Expr], span: Span) -> LangResult<Value> {
        self.require_cas("using", span)?;
        if args.len() != 2 {
            return Err(LangError::arity("using: expected 2 arguments: using(\"backend\", expr)").with_span(span));
        }

        // Evaluate first arg to get backend name
        let val = self.eval_expr(&args[0])?;
        let name = match &val {
            Value::Str(s) => s.clone(),
            Value::Symbolic(SymExpr::Sym { name }) => name.clone(),
            _ => return Err(LangError::type_err("using: first argument must be a backend name string").with_span(span)),
        };

        let manager = self.cas_manager.as_mut().unwrap();
        let saved_routing = manager.routing.clone();

        let mode = match name.as_str() {
            "both" => RoutingMode::Both,
            other => RoutingMode::Single(other.to_string()),
        };
        manager.set_routing(mode)
            .map_err(|e| LangError::eval(format!("using: {}", e)).with_span(span))?;

        // Evaluate the expression (lazy — args[1] is unevaluated AST)
        let result = self.eval_expr(&args[1]);

        // Restore routing
        let manager = self.cas_manager.as_mut().unwrap();
        manager.routing = saved_routing;

        result
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

        let manager = self.cas_manager.as_mut().unwrap();
        let result = manager.differentiate(&sym_expr, &var, order)
            .map_err(|e| LangError::eval(format!("dif: {}", e)).with_span(span))?;

        Ok(self.cas_result_to_value(result))
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
        } else if args.len() == 3 {
            // int(expr, var, lo..hi) — range syntax
            if let Expr::Range { start, end, .. } = &args[2] {
                let lo = self.eval_expr(start)?.to_sym_expr().ok_or_else(|| {
                    LangError::type_err("int: lower bound must be numeric or symbolic").with_span(span)
                })?;
                let hi = self.eval_expr(end)?.to_sym_expr().ok_or_else(|| {
                    LangError::type_err("int: upper bound must be numeric or symbolic").with_span(span)
                })?;
                (Some(lo), Some(hi))
            } else {
                return Err(LangError::eval(
                    "int: 3rd argument must be a range (e.g. 0..1), or use 4-arg form: int(expr, var, lo, hi)"
                ).with_span(span));
            }
        } else {
            (None, None)
        };

        let manager = self.cas_manager.as_mut().unwrap();
        let result = manager.integrate(&sym_expr, &var, lower.as_ref(), upper.as_ref())
            .map_err(|e| LangError::eval(format!("int: {}", e)).with_span(span))?;

        Ok(self.cas_result_to_value(result))
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

        let manager = self.cas_manager.as_mut().unwrap();
        let result = manager.solve(&equations, &vars)
            .map_err(|e| LangError::eval(format!("solve: {}", e)).with_span(span))?;

        Ok(self.cas_result_to_value(result))
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
        let manager = self.cas_manager.as_mut().unwrap();
        let result = manager.simplify(&sym_expr)
            .map_err(|e| LangError::eval(format!("simplify: {}", e)).with_span(span))?;
        Ok(self.cas_result_to_value(result))
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
        let manager = self.cas_manager.as_mut().unwrap();
        let result = manager.expand(&sym_expr)
            .map_err(|e| LangError::eval(format!("expand: {}", e)).with_span(span))?;
        Ok(self.cas_result_to_value(result))
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
        let manager = self.cas_manager.as_mut().unwrap();
        let result = manager.factor(&sym_expr)
            .map_err(|e| LangError::eval(format!("factor: {}", e)).with_span(span))?;
        Ok(self.cas_result_to_value(result))
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

        let manager = self.cas_manager.as_mut().unwrap();
        let result = manager.limit(&sym_expr, &var, &point, dir.as_deref())
            .map_err(|e| LangError::eval(format!("lim: {}", e)).with_span(span))?;
        Ok(self.cas_result_to_value(result))
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

        let manager = self.cas_manager.as_mut().unwrap();
        let result = manager.taylor(&sym_expr, &var, &point, order)
            .map_err(|e| LangError::eval(format!("taylor: {}", e)).with_span(span))?;
        Ok(self.cas_result_to_value(result))
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
        let manager = self.cas_manager.as_mut().unwrap();
        let result = manager.latex(&sym_expr)
            .map_err(|e| LangError::eval(format!("tex: {}", e)).with_span(span))?;
        Ok(self.cas_result_to_value(result))
    }

    // --- Plotting ---

    fn eval_plot(&mut self, args: &[Expr], span: Span) -> LangResult<Value> {
        if args.is_empty() || args.len() > 2 {
            return Err(LangError::arity(
                "plot: expected 1-2 arguments: plot(expr, range?) or plot([e1, e2, ...], range?)"
            ).with_span(span));
        }

        // Parse range from 2nd arg (or default)
        let (x_min, x_max) = if args.len() == 2 {
            self.parse_plot_range(&args[1], span)?
        } else {
            (DEFAULT_X_MIN, DEFAULT_X_MAX)
        };

        // Parse expression(s): single expr or vector of exprs
        let exprs: Vec<&Expr> = match &args[0] {
            Expr::Vector(elements, _) => elements.iter().collect(),
            other => vec![other],
        };

        // Collect all free variables across all exprs
        let mut all_free = Vec::new();
        for expr in &exprs {
            let val = self.eval_expr(expr)?;
            if let Some(sym) = val.to_sym_expr() {
                for s in sym.free_symbols() {
                    if !all_free.contains(&s) {
                        all_free.push(s);
                    }
                }
            }
        }

        let var = if all_free.len() == 1 {
            all_free[0].clone()
        } else if all_free.contains(&"x".to_string()) {
            "x".to_string()
        } else if all_free.is_empty() {
            // Constant expression — still plot it
            "x".to_string()
        } else {
            return Err(LangError::eval(format!(
                "plot: multiple free variables ({}) — only single-variable plots are supported",
                all_free.join(", ")
            )).with_span(span));
        };

        // Generate labels before binding the variable
        let labels: Vec<String> = exprs.iter().map(|e| {
            match self.eval_expr(e) {
                Ok(v) => format!("{}", v),
                Err(_) => "?".to_string(),
            }
        }).collect();

        // Sample each expression
        let mut series_list = Vec::new();
        for (i, expr) in exprs.iter().enumerate() {
            let mut points = self.sample_expression(expr, &var, x_min, x_max, DEFAULT_SAMPLES, span)?;
            insert_discontinuity_gaps(&mut points);
            series_list.push(Series {
                label: labels[i].clone(),
                points,
            });
        }

        let spec = PlotSpec {
            series: series_list,
            x_min,
            x_max,
            title: None,
        };

        // Prefer matplotlib rendering via CAS backend
        if self.cas_manager.is_some() {
            match self.render_plot_matplotlib(&spec, span) {
                Ok(rendered) => return Ok(Value::Plot(rendered)),
                Err(_) => {} // fall through to plotters
            }
        }

        // Fallback: plotters (no labels/ticks, but works without Python)
        let rendered = render_plot(&spec)
            .map_err(|e| LangError::eval(format!("plot render: {}", e)).with_span(span))?;

        Ok(Value::Plot(rendered))
    }

    /// Parse a range argument for plot: either Expr::Range or evaluate to get bounds.
    fn parse_plot_range(&mut self, arg: &Expr, span: Span) -> LangResult<(f64, f64)> {
        if let Expr::Range { start, end, .. } = arg {
            let lo = self.eval_expr(start)?;
            let hi = self.eval_expr(end)?;
            let lo_f = self.value_to_f64(&lo).ok_or_else(|| {
                LangError::type_err("plot: range start must be numeric").with_span(span)
            })?;
            let hi_f = self.value_to_f64(&hi).ok_or_else(|| {
                LangError::type_err("plot: range end must be numeric").with_span(span)
            })?;
            Ok((lo_f, hi_f))
        } else {
            Err(LangError::type_err(
                "plot: second argument must be a range (e.g. -5..5)"
            ).with_span(span))
        }
    }

    /// Sample an expression at evenly spaced x values, returning points with None for discontinuities.
    fn sample_expression(
        &mut self,
        expr: &Expr,
        var: &str,
        x_min: f64,
        x_max: f64,
        n: usize,
        span: Span,
    ) -> LangResult<Vec<Option<(f64, f64)>>> {
        // Save and remove current binding of var
        let saved = self.env.borrow().get(var);
        let mut points = Vec::with_capacity(n);
        let mut valid_count = 0;

        for i in 0..n {
            let x = x_min + (x_max - x_min) * i as f64 / (n - 1) as f64;

            // Bind var to x
            self.env.borrow_mut().set(var.to_string(), Value::Number(Number::Float(x)));

            let y = match self.eval_expr(expr) {
                Ok(val) => self.value_to_f64(&val),
                Err(_) => None,
            };

            match y {
                Some(yf) if yf.is_finite() => {
                    points.push(Some((x, yf)));
                    valid_count += 1;
                }
                _ => points.push(None),
            }
        }

        // Restore var binding
        if let Some(prev) = saved {
            self.env.borrow_mut().set(var.to_string(), prev);
        } else {
            self.env.borrow_mut().remove(var);
        }

        // If too few valid points and CAS is available, try lambdify fallback
        if valid_count * 4 < n && self.cas_manager.is_some() {
            if let Ok(cas_points) = self.cas_lambdify_sample(expr, var, x_min, x_max, n, span) {
                let cas_valid = cas_points.iter().filter(|p| p.is_some()).count();
                if cas_valid > valid_count {
                    return Ok(cas_points);
                }
            }
        }

        Ok(points)
    }

    /// CAS lambdify fallback for sampling expressions that can't be evaluated in Rust.
    fn cas_lambdify_sample(
        &mut self,
        expr: &Expr,
        var: &str,
        x_min: f64,
        x_max: f64,
        n: usize,
        span: Span,
    ) -> LangResult<Vec<Option<(f64, f64)>>> {
        // Evaluate expression to get SymExpr
        // Temporarily unbind var to get symbolic form
        let saved = self.env.borrow().get(var);
        self.env.borrow_mut().remove(var);
        let val = self.eval_expr(expr)?;
        if let Some(prev) = saved {
            self.env.borrow_mut().set(var.to_string(), prev);
        }

        let sym_expr = val.to_sym_expr().ok_or_else(|| {
            LangError::type_err("plot: expression must be numeric or symbolic").with_span(span)
        })?;

        let x_values: Vec<f64> = (0..n)
            .map(|i| x_min + (x_max - x_min) * i as f64 / (n - 1) as f64)
            .collect();

        let manager = self.cas_manager.as_mut().unwrap();
        let y_values = manager.lambdify_eval(&sym_expr, var, &x_values)
            .map_err(|e| LangError::eval(format!("plot CAS fallback: {}", e)).with_span(span))?;

        let points: Vec<Option<(f64, f64)>> = x_values
            .iter()
            .zip(y_values.iter())
            .map(|(&x, y)| y.map(|yv| (x, yv)))
            .collect();

        Ok(points)
    }

    /// Convert a Value to f64, handling Number, Symbolic constants, and evaluable symbolic exprs.
    fn value_to_f64(&self, val: &Value) -> Option<f64> {
        match val {
            Value::Number(n) => Some(n.as_f64()),
            Value::Symbolic(expr) => numeric_eval_sym(expr),
            _ => None,
        }
    }

    /// Render a PlotSpec via the CAS backend (matplotlib).
    fn render_plot_matplotlib(&mut self, spec: &PlotSpec, span: Span) -> LangResult<RenderedPlot> {
        let manager = self.cas_manager.as_mut().unwrap();

        // Convert Series to PlotSeriesData (split points into separate x/y arrays)
        let series_data: Vec<PlotSeriesData> = spec.series.iter().map(|s| {
            let mut xs = Vec::with_capacity(s.points.len());
            let mut ys = Vec::with_capacity(s.points.len());
            for pt in &s.points {
                match pt {
                    Some((x, y)) => {
                        xs.push(*x);
                        ys.push(Some(*y));
                    }
                    None => {
                        // Insert a NaN marker at the midpoint of the gap
                        if let Some(last_x) = xs.last() {
                            xs.push(*last_x);
                        } else {
                            xs.push(spec.x_min);
                        }
                        ys.push(None);
                    }
                }
            }
            PlotSeriesData {
                label: s.label.clone(),
                x: xs,
                y: ys,
            }
        }).collect();

        let png_bytes = manager.render_plot(
            series_data,
            spec.x_min,
            spec.x_max,
            PLOT_WIDTH,
            PLOT_HEIGHT,
            150,
        ).map_err(|e| LangError::eval(format!("matplotlib render: {}", e)).with_span(span))?;

        Ok(RenderedPlot {
            png_bytes,
            spec: spec.clone(),
            width: PLOT_WIDTH,
            height: PLOT_HEIGHT,
        })
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
            // Unknown symbolic variable used as function — build a symbolic call
            Value::Symbolic(SymExpr::Sym { name }) => {
                let sym_args: Option<Vec<SymExpr>> = args.iter().map(|a| a.to_sym_expr()).collect();
                if let Some(sym_args) = sym_args {
                    Ok(Value::Symbolic(SymExpr::func(name, sym_args)))
                } else {
                    Err(LangError::type_err(format!(
                        "cannot call '{}' with non-numeric/symbolic arguments", name
                    )).with_span(span))
                }
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

/// Detect and insert None gaps at likely discontinuities (vertical asymptotes).
///
/// Scans consecutive valid points; if |dy| exceeds a threshold relative to the
/// overall y-range, inserts a None between them to break the line.
fn insert_discontinuity_gaps(points: &mut Vec<Option<(f64, f64)>>) {
    // Compute y-range from valid points
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for pt in points.iter() {
        if let Some((_, y)) = pt {
            if y.is_finite() {
                y_min = y_min.min(*y);
                y_max = y_max.max(*y);
            }
        }
    }
    let y_range = y_max - y_min;
    if y_range < 1e-10 {
        return; // constant or near-constant — no discontinuities
    }

    // Threshold: if a single step jumps more than 50% of the total y-range,
    // it's likely a discontinuity (works for tan, 1/x, etc.)
    let threshold = y_range * 0.5;

    // Scan backwards so insertions don't shift indices we haven't processed
    let mut i = points.len().saturating_sub(1);
    while i > 0 {
        if let (Some((_, y1)), Some((_, y2))) = (&points[i - 1], &points[i]) {
            if (y2 - y1).abs() > threshold {
                points.insert(i, None);
            }
        }
        i -= 1;
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

    // --- Plot tests ---

    #[test]
    fn test_plot_basic() {
        let v = eval("plot(x^2)");
        assert!(matches!(v, Value::Plot(_)));
        if let Value::Plot(p) = &v {
            assert_eq!(p.spec.series.len(), 1);
            assert!(!p.png_bytes.is_empty());
            assert_eq!(p.spec.x_min, -10.0);
            assert_eq!(p.spec.x_max, 10.0);
        }
    }

    #[test]
    fn test_plot_with_range() {
        let v = eval("plot(x^2, -5..5)");
        assert!(matches!(v, Value::Plot(_)));
        if let Value::Plot(p) = &v {
            assert_eq!(p.spec.x_min, -5.0);
            assert_eq!(p.spec.x_max, 5.0);
        }
    }

    #[test]
    fn test_plot_multi_curve() {
        let v = eval("plot([sin(x), cos(x)])");
        assert!(matches!(v, Value::Plot(_)));
        if let Value::Plot(p) = &v {
            assert_eq!(p.spec.series.len(), 2);
        }
    }

    #[test]
    fn test_plot_user_function() {
        let mut e = Evaluator::new();
        eval_with(&mut e, "f(x) = x^2 + 1");
        let v = eval_with(&mut e, "plot(f(x))");
        assert!(matches!(v, Value::Plot(_)));
        if let Value::Plot(p) = &v {
            // Check some sample points: f(0) = 1, f(1) = 2, f(-1) = 2
            let pts: Vec<_> = p.spec.series[0].points.iter().filter_map(|p| *p).collect();
            assert!(!pts.is_empty());
            // Find point near x=0
            let near_zero = pts.iter().find(|(x, _)| x.abs() < 0.05).unwrap();
            assert!((near_zero.1 - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_plot_constant_expr() {
        // Constant expression — should produce a flat line
        let v = eval("plot(5, -2..2)");
        assert!(matches!(v, Value::Plot(_)));
        if let Value::Plot(p) = &v {
            let pts: Vec<_> = p.spec.series[0].points.iter().filter_map(|p| *p).collect();
            assert!(pts.iter().all(|(_, y)| (*y - 5.0).abs() < 1e-10));
        }
    }

    #[test]
    fn test_plot_trig() {
        let v = eval("plot(sin(x), 0..6)");
        assert!(matches!(v, Value::Plot(_)));
        if let Value::Plot(p) = &v {
            let pts: Vec<_> = p.spec.series[0].points.iter().filter_map(|p| *p).collect();
            // sin(0) ≈ 0
            assert!(pts[0].1.abs() < 0.01);
        }
    }

    #[test]
    fn test_plot_preserves_env() {
        // plot() should not leave the sampling variable bound
        let mut e = Evaluator::new();
        eval_with(&mut e, "plot(x^2)");
        // x should still be symbolic (unbound)
        let v = eval_with(&mut e, "x");
        assert!(matches!(v, Value::Symbolic(_)));
    }

    #[test]
    fn test_plot_display() {
        let v = eval("plot(sin(x))");
        let display = format!("{}", v);
        assert!(display.contains("plot:"));
        assert!(display.contains("sin(x)"));
    }
}
