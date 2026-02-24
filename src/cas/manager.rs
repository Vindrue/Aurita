use std::collections::HashMap;

use crate::cas::backend::CasBackend;
use crate::cas::cache::CasCache;
use crate::cas::protocol::{CasOp, CasResponse, PlotSeriesData};
use crate::symbolic::expr::SymExpr;

/// How to route CAS operations.
#[derive(Debug, Clone)]
pub enum RoutingMode {
    Single(String),
    Both,
}

/// Result from the CAS manager, which may come from one or two backends.
#[derive(Debug)]
pub enum CasResult {
    /// Single result from one backend.
    Single(SymExpr),
    /// Multiple results (e.g. solve).
    Multiple(Vec<SymExpr>),
    /// LaTeX string.
    Latex(String),
    /// Both backends agree on the result.
    Agreed(SymExpr),
    /// Both backends disagree — labeled results.
    Disagreed { results: Vec<(String, SymExpr)> },
    /// Both backends agree on multiple results.
    AgreedMultiple(Vec<SymExpr>),
    /// Both backends disagree on multiple results.
    DisagreedMultiple { results: Vec<(String, Vec<SymExpr>)> },
}

const DEFAULT_CACHE_CAPACITY: usize = 256;

/// Central orchestration for multiple CAS backends.
pub struct CasManager {
    backends: HashMap<String, CasBackend>,
    pub routing: RoutingMode,
    cache: CasCache,
}

impl CasManager {
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
            routing: RoutingMode::Single("sympy".to_string()),
            cache: CasCache::new(DEFAULT_CACHE_CAPACITY),
        }
    }

    /// Spawn and register a backend.
    pub fn add_backend(&mut self, name: &str, command: &str, args: &[&str]) -> Result<(), String> {
        let backend = CasBackend::spawn(name, command, args)?;
        self.backends.insert(name.to_string(), backend);
        Ok(())
    }

    /// Set routing mode. Returns error if the requested backend is not available.
    pub fn set_routing(&mut self, mode: RoutingMode) -> Result<(), String> {
        match &mode {
            RoutingMode::Single(name) => {
                if !self.backends.contains_key(name) {
                    return Err(format!(
                        "backend '{}' not available (available: {})",
                        name,
                        self.backend_names().join(", ")
                    ));
                }
            }
            RoutingMode::Both => {
                if self.backends.len() < 2 {
                    return Err(format!(
                        "need at least 2 backends for 'both' mode (available: {})",
                        self.backend_names().join(", ")
                    ));
                }
            }
        }
        self.routing = mode;
        Ok(())
    }

    /// Get a display string for the status bar.
    pub fn status_display(&self) -> String {
        match &self.routing {
            RoutingMode::Single(name) => {
                // Capitalize first letter
                let mut chars = name.chars();
                match chars.next() {
                    Some(c) => format!("{}{}", c.to_uppercase(), chars.as_str()),
                    None => name.clone(),
                }
            }
            RoutingMode::Both => "Both".to_string(),
        }
    }

    pub fn has_any_backend(&self) -> bool {
        !self.backends.is_empty()
    }

    pub fn has_backend(&self, name: &str) -> bool {
        self.backends.contains_key(name)
    }

    pub fn backend_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.backends.keys().cloned().collect();
        names.sort();
        names
    }

    // --- Convenience methods that mirror CasBackend ---

    pub fn differentiate(&mut self, expr: &SymExpr, var: &str, order: u32) -> Result<CasResult, String> {
        let op = CasOp::Differentiate {
            expr: expr.clone(),
            var: var.to_string(),
            order,
        };
        self.execute_op(op, ResponseKind::Single)
    }

    pub fn integrate(
        &mut self,
        expr: &SymExpr,
        var: &str,
        lower: Option<&SymExpr>,
        upper: Option<&SymExpr>,
    ) -> Result<CasResult, String> {
        let op = CasOp::Integrate {
            expr: expr.clone(),
            var: var.to_string(),
            lower: lower.cloned(),
            upper: upper.cloned(),
        };
        self.execute_op(op, ResponseKind::Single)
    }

    pub fn solve(&mut self, equations: &[SymExpr], vars: &[String]) -> Result<CasResult, String> {
        let op = CasOp::Solve {
            equations: equations.to_vec(),
            vars: vars.to_vec(),
        };
        self.execute_op(op, ResponseKind::Multiple)
    }

    pub fn simplify(&mut self, expr: &SymExpr) -> Result<CasResult, String> {
        let op = CasOp::Simplify { expr: expr.clone() };
        self.execute_op(op, ResponseKind::Single)
    }

    pub fn expand(&mut self, expr: &SymExpr) -> Result<CasResult, String> {
        let op = CasOp::Expand { expr: expr.clone() };
        self.execute_op(op, ResponseKind::Single)
    }

    pub fn factor(&mut self, expr: &SymExpr) -> Result<CasResult, String> {
        let op = CasOp::Factor { expr: expr.clone() };
        self.execute_op(op, ResponseKind::Single)
    }

    pub fn limit(
        &mut self,
        expr: &SymExpr,
        var: &str,
        point: &SymExpr,
        dir: Option<&str>,
    ) -> Result<CasResult, String> {
        let op = CasOp::Limit {
            expr: expr.clone(),
            var: var.to_string(),
            point: point.clone(),
            dir: dir.map(|s| s.to_string()),
        };
        self.execute_op(op, ResponseKind::Single)
    }

    pub fn taylor(
        &mut self,
        expr: &SymExpr,
        var: &str,
        point: &SymExpr,
        order: u32,
    ) -> Result<CasResult, String> {
        let op = CasOp::Taylor {
            expr: expr.clone(),
            var: var.to_string(),
            point: point.clone(),
            order,
        };
        self.execute_op(op, ResponseKind::Single)
    }

    pub fn latex(&mut self, expr: &SymExpr) -> Result<CasResult, String> {
        let op = CasOp::Latex { expr: expr.clone() };
        self.execute_op(op, ResponseKind::Latex)
    }

    /// Render a plot — always routes to SymPy (matplotlib/numpy).
    pub fn render_plot(
        &mut self,
        series: Vec<PlotSeriesData>,
        x_min: f64,
        x_max: f64,
        width: u32,
        height: u32,
        dpi: u32,
    ) -> Result<Vec<u8>, String> {
        let backend = self.backends.get_mut("sympy")
            .ok_or_else(|| "render_plot requires SymPy backend".to_string())?;
        backend.render_plot(series, x_min, x_max, width, height, dpi)
    }

    /// Lambdify + evaluate — always routes to SymPy (numpy).
    pub fn lambdify_eval(
        &mut self,
        expr: &SymExpr,
        var: &str,
        x_values: &[f64],
    ) -> Result<Vec<Option<f64>>, String> {
        let backend = self.backends.get_mut("sympy")
            .ok_or_else(|| "lambdify requires SymPy backend".to_string())?;
        backend.lambdify_eval(expr, var, x_values)
    }

    // --- Internal routing logic ---

    fn execute_op(&mut self, op: CasOp, kind: ResponseKind) -> Result<CasResult, String> {
        match self.routing.clone() {
            RoutingMode::Single(name) => self.execute_on_single(&name, op, kind),
            RoutingMode::Both => self.execute_on_both(op, kind),
        }
    }

    fn execute_on_single(&mut self, backend_name: &str, op: CasOp, kind: ResponseKind) -> Result<CasResult, String> {
        // Check cache
        if let Some(cached) = self.cache.get(&op, backend_name) {
            return response_to_result(cached, kind);
        }

        let backend = self.backends.get_mut(backend_name)
            .ok_or_else(|| format!("backend '{}' not available", backend_name))?;

        let response = backend.request(op.clone())?;

        if response.is_ok() {
            self.cache.put(&op, backend_name, response.clone());
        }

        response_to_result(response, kind)
    }

    fn execute_on_both(&mut self, op: CasOp, kind: ResponseKind) -> Result<CasResult, String> {
        let names: Vec<String> = self.backend_names();

        let mut results: Vec<(String, CasResponse)> = Vec::new();
        for name in &names {
            // Check cache first
            if let Some(cached) = self.cache.get(&op, name) {
                results.push((name.clone(), cached));
                continue;
            }

            let backend = self.backends.get_mut(name).unwrap();
            match backend.request(op.clone()) {
                Ok(resp) => {
                    if resp.is_ok() {
                        self.cache.put(&op, name, resp.clone());
                    }
                    results.push((name.clone(), resp));
                }
                Err(e) => {
                    // If one backend fails, continue with the other
                    results.push((name.clone(), CasResponse {
                        id: 0,
                        status: "error".to_string(),
                        result: None,
                        error: Some(e),
                        results: None,
                        latex: None,
                        y_values: None,
                        png_base64: None,
                    }));
                }
            }
        }

        // Filter to successful responses
        let ok_results: Vec<(String, CasResponse)> = results
            .into_iter()
            .filter(|(_, r)| r.is_ok())
            .collect();

        if ok_results.is_empty() {
            return Err("all backends failed".to_string());
        }

        if ok_results.len() == 1 {
            let (_, resp) = ok_results.into_iter().next().unwrap();
            return response_to_result(resp, kind);
        }

        // Compare results from multiple backends
        match kind {
            ResponseKind::Single => {
                let exprs: Vec<(String, SymExpr)> = ok_results
                    .into_iter()
                    .filter_map(|(name, resp)| resp.result.map(|e| (name, e)))
                    .collect();

                if exprs.len() < 2 {
                    return match exprs.into_iter().next() {
                        Some((_, expr)) => Ok(CasResult::Agreed(expr)),
                        None => Err("no results from backends".to_string()),
                    };
                }

                // Compare by Display string
                let first_str = format!("{}", exprs[0].1);
                let all_agree = exprs.iter().all(|(_, e)| format!("{}", e) == first_str);

                if all_agree {
                    Ok(CasResult::Agreed(exprs.into_iter().next().unwrap().1))
                } else {
                    Ok(CasResult::Disagreed { results: exprs })
                }
            }
            ResponseKind::Multiple => {
                let multi: Vec<(String, Vec<SymExpr>)> = ok_results
                    .into_iter()
                    .filter_map(|(name, resp)| resp.results.map(|r| (name, r)))
                    .collect();

                if multi.len() < 2 {
                    return match multi.into_iter().next() {
                        Some((_, exprs)) => Ok(CasResult::AgreedMultiple(exprs)),
                        None => Err("no results from backends".to_string()),
                    };
                }

                // Compare by Display strings of sorted results
                let strs: Vec<Vec<String>> = multi
                    .iter()
                    .map(|(_, exprs)| {
                        let mut s: Vec<String> = exprs.iter().map(|e| format!("{}", e)).collect();
                        s.sort();
                        s
                    })
                    .collect();

                if strs.windows(2).all(|w| w[0] == w[1]) {
                    Ok(CasResult::AgreedMultiple(multi.into_iter().next().unwrap().1))
                } else {
                    Ok(CasResult::DisagreedMultiple { results: multi })
                }
            }
            ResponseKind::Latex => {
                // For latex, just use the first successful result
                let (_, resp) = ok_results.into_iter().next().unwrap();
                response_to_result(resp, kind)
            }
        }
    }
}

/// What kind of response we expect from the backend.
#[derive(Debug, Clone, Copy)]
enum ResponseKind {
    Single,
    Multiple,
    Latex,
}

fn response_to_result(resp: CasResponse, kind: ResponseKind) -> Result<CasResult, String> {
    if !resp.is_ok() {
        return Err(resp.error.unwrap_or_else(|| "unknown error".to_string()));
    }
    match kind {
        ResponseKind::Single => {
            let expr = resp.result.ok_or("backend returned ok but no result")?;
            Ok(CasResult::Single(expr))
        }
        ResponseKind::Multiple => {
            let exprs = resp.results.ok_or("backend returned ok but no results")?;
            Ok(CasResult::Multiple(exprs))
        }
        ResponseKind::Latex => {
            let latex = resp.latex.unwrap_or_else(|| "?".to_string());
            Ok(CasResult::Latex(latex))
        }
    }
}
