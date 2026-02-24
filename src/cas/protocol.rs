use crate::symbolic::expr::SymExpr;
use serde::{Deserialize, Serialize};

/// A request sent to a CAS backend.
#[derive(Debug, Clone, Serialize)]
pub struct CasRequest {
    pub id: u64,
    pub op: CasOp,
}

/// The operation to perform.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "op", content = "params")]
pub enum CasOp {
    #[serde(rename = "differentiate")]
    Differentiate {
        expr: SymExpr,
        var: String,
        order: u32,
    },
    #[serde(rename = "integrate")]
    Integrate {
        expr: SymExpr,
        var: String,
        lower: Option<SymExpr>,
        upper: Option<SymExpr>,
    },
    #[serde(rename = "solve")]
    Solve {
        equations: Vec<SymExpr>,
        vars: Vec<String>,
    },
    #[serde(rename = "simplify")]
    Simplify { expr: SymExpr },
    #[serde(rename = "expand")]
    Expand { expr: SymExpr },
    #[serde(rename = "factor")]
    Factor { expr: SymExpr },
    #[serde(rename = "limit")]
    Limit {
        expr: SymExpr,
        var: String,
        point: SymExpr,
        dir: Option<String>, // "+", "-", or None for both
    },
    #[serde(rename = "taylor")]
    Taylor {
        expr: SymExpr,
        var: String,
        point: SymExpr,
        order: u32,
    },
    #[serde(rename = "latex")]
    Latex { expr: SymExpr },
    #[serde(rename = "lambdify")]
    Lambdify {
        expr: SymExpr,
        var: String,
        x_values: Vec<f64>,
    },
    #[serde(rename = "render_plot")]
    RenderPlot {
        series: Vec<PlotSeriesData>,
        x_min: f64,
        x_max: f64,
        width: u32,
        height: u32,
        dpi: u32,
    },
}

/// Series data for the render_plot CAS operation.
#[derive(Debug, Clone, Serialize)]
pub struct PlotSeriesData {
    pub label: String,
    pub x: Vec<f64>,
    pub y: Vec<Option<f64>>,
}

/// A response from a CAS backend.
#[derive(Debug, Clone, Deserialize)]
pub struct CasResponse {
    pub id: u64,
    pub status: String,
    pub result: Option<SymExpr>,
    pub error: Option<String>,
    /// Results for solve (multiple solutions)
    pub results: Option<Vec<SymExpr>>,
    /// LaTeX string
    pub latex: Option<String>,
    /// Numeric y-values from lambdify evaluation
    pub y_values: Option<Vec<Option<f64>>>,
    /// Base64-encoded PNG from render_plot
    pub png_base64: Option<String>,
}

impl CasResponse {
    pub fn is_ok(&self) -> bool {
        self.status == "ok"
    }

    pub fn into_result(self) -> Result<SymExpr, String> {
        if self.is_ok() {
            self.result
                .ok_or_else(|| "backend returned ok but no result".to_string())
        } else {
            Err(self.error.unwrap_or_else(|| "unknown error".to_string()))
        }
    }

    pub fn into_results_vec(self) -> Result<Vec<SymExpr>, String> {
        if self.is_ok() {
            self.results
                .ok_or_else(|| "backend returned ok but no results".to_string())
        } else {
            Err(self.error.unwrap_or_else(|| "unknown error".to_string()))
        }
    }
}
