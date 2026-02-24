use crate::cas::protocol::{CasOp, CasRequest, CasResponse};
use crate::symbolic::expr::SymExpr;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

/// A running CAS backend subprocess communicating via JSON over stdin/stdout.
pub struct CasBackend {
    pub name: String,
    child: Child,
    stdin: std::process::ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
    next_id: u64,
}

impl CasBackend {
    /// Spawn a backend process.
    pub fn spawn(name: &str, command: &str, args: &[&str]) -> Result<Self, String> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to spawn {} backend: {}", name, e))?;

        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let reader = BufReader::new(stdout);

        Ok(Self {
            name: name.to_string(),
            child,
            stdin,
            reader,
            next_id: 1,
        })
    }

    /// Send a CAS operation and wait for the response.
    pub fn request(&mut self, op: CasOp) -> Result<CasResponse, String> {
        let id = self.next_id;
        self.next_id += 1;

        let req = CasRequest { id, op };
        let json = serde_json::to_string(&req)
            .map_err(|e| format!("failed to serialize request: {}", e))?;

        // Send request (line-delimited JSON)
        writeln!(self.stdin, "{}", json)
            .map_err(|e| format!("failed to write to {} backend: {}", self.name, e))?;
        self.stdin
            .flush()
            .map_err(|e| format!("failed to flush {} backend stdin: {}", self.name, e))?;

        // Read response
        let mut line = String::new();
        self.reader
            .read_line(&mut line)
            .map_err(|e| format!("failed to read from {} backend: {}", self.name, e))?;

        if line.is_empty() {
            return Err(format!("{} backend process closed unexpectedly", self.name));
        }

        let response: CasResponse = serde_json::from_str(&line)
            .map_err(|e| format!("failed to parse {} response: {} (raw: {})", self.name, e, line.trim()))?;

        Ok(response)
    }

    /// Convenience: differentiate.
    pub fn differentiate(&mut self, expr: &SymExpr, var: &str, order: u32) -> Result<SymExpr, String> {
        let resp = self.request(CasOp::Differentiate {
            expr: expr.clone(),
            var: var.to_string(),
            order,
        })?;
        resp.into_result()
    }

    /// Convenience: integrate.
    pub fn integrate(
        &mut self,
        expr: &SymExpr,
        var: &str,
        lower: Option<&SymExpr>,
        upper: Option<&SymExpr>,
    ) -> Result<SymExpr, String> {
        let resp = self.request(CasOp::Integrate {
            expr: expr.clone(),
            var: var.to_string(),
            lower: lower.cloned(),
            upper: upper.cloned(),
        })?;
        resp.into_result()
    }

    /// Convenience: solve.
    pub fn solve(&mut self, equations: &[SymExpr], vars: &[String]) -> Result<Vec<SymExpr>, String> {
        let resp = self.request(CasOp::Solve {
            equations: equations.to_vec(),
            vars: vars.to_vec(),
        })?;
        resp.into_results_vec()
    }

    /// Convenience: simplify.
    pub fn simplify(&mut self, expr: &SymExpr) -> Result<SymExpr, String> {
        let resp = self.request(CasOp::Simplify { expr: expr.clone() })?;
        resp.into_result()
    }

    /// Convenience: expand.
    pub fn expand(&mut self, expr: &SymExpr) -> Result<SymExpr, String> {
        let resp = self.request(CasOp::Expand { expr: expr.clone() })?;
        resp.into_result()
    }

    /// Convenience: factor.
    pub fn factor(&mut self, expr: &SymExpr) -> Result<SymExpr, String> {
        let resp = self.request(CasOp::Factor { expr: expr.clone() })?;
        resp.into_result()
    }

    /// Convenience: limit.
    pub fn limit(
        &mut self,
        expr: &SymExpr,
        var: &str,
        point: &SymExpr,
        dir: Option<&str>,
    ) -> Result<SymExpr, String> {
        let resp = self.request(CasOp::Limit {
            expr: expr.clone(),
            var: var.to_string(),
            point: point.clone(),
            dir: dir.map(|s| s.to_string()),
        })?;
        resp.into_result()
    }

    /// Convenience: Taylor series.
    pub fn taylor(
        &mut self,
        expr: &SymExpr,
        var: &str,
        point: &SymExpr,
        order: u32,
    ) -> Result<SymExpr, String> {
        let resp = self.request(CasOp::Taylor {
            expr: expr.clone(),
            var: var.to_string(),
            point: point.clone(),
            order,
        })?;
        resp.into_result()
    }

    /// Check if the backend process is still running.
    pub fn is_alive(&mut self) -> bool {
        self.child.try_wait().ok().flatten().is_none()
    }

    /// Kill the backend process.
    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for CasBackend {
    fn drop(&mut self) {
        self.kill();
    }
}
