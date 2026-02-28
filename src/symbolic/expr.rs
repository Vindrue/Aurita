use serde::{Deserialize, Serialize};
use std::fmt;

/// Symbolic expression tree — the Rust-native representation
/// for symbolic math that gets serialized to JSON for CAS backends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SymExpr {
    /// A symbolic variable: x, theta, alpha
    Sym { name: String },
    /// An integer constant (exact)
    Int { value: i64 },
    /// A rational constant (exact)
    Rational { num: i64, den: i64 },
    /// A floating-point constant
    Float { value: f64 },
    /// Known constant: pi, e, i
    Const { name: MathConst },
    /// Binary operation
    BinOp {
        op: SymOp,
        lhs: Box<SymExpr>,
        rhs: Box<SymExpr>,
    },
    /// Unary negation
    Neg { expr: Box<SymExpr> },
    /// Function application: sin(x), ln(x), etc.
    Func { name: String, args: Vec<SymExpr> },
    /// Vector
    Vector { elements: Vec<SymExpr> },
    /// Undefined / could not compute
    Undefined,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SymOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MathConst {
    Pi,
    E,
    I,
    Infinity,
    NegInfinity,
}

// --- Constructors ---

impl SymExpr {
    pub fn sym(name: &str) -> Self {
        SymExpr::Sym {
            name: name.to_string(),
        }
    }

    pub fn int(value: i64) -> Self {
        SymExpr::Int { value }
    }

    pub fn float(value: f64) -> Self {
        SymExpr::Float { value }
    }

    pub fn add(lhs: SymExpr, rhs: SymExpr) -> Self {
        SymExpr::BinOp {
            op: SymOp::Add,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn sub(lhs: SymExpr, rhs: SymExpr) -> Self {
        SymExpr::BinOp {
            op: SymOp::Sub,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn mul(lhs: SymExpr, rhs: SymExpr) -> Self {
        SymExpr::BinOp {
            op: SymOp::Mul,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn div(lhs: SymExpr, rhs: SymExpr) -> Self {
        SymExpr::BinOp {
            op: SymOp::Div,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn pow(lhs: SymExpr, rhs: SymExpr) -> Self {
        SymExpr::BinOp {
            op: SymOp::Pow,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn neg(expr: SymExpr) -> Self {
        SymExpr::Neg {
            expr: Box::new(expr),
        }
    }

    pub fn func(name: &str, args: Vec<SymExpr>) -> Self {
        SymExpr::Func {
            name: name.to_string(),
            args,
        }
    }

    /// Collect all free symbolic variable names in this expression.
    pub fn free_symbols(&self) -> Vec<String> {
        let mut syms = Vec::new();
        self.collect_symbols(&mut syms);
        syms.sort();
        syms.dedup();
        syms
    }

    fn collect_symbols(&self, syms: &mut Vec<String>) {
        match self {
            SymExpr::Sym { name } => syms.push(name.clone()),
            SymExpr::BinOp { lhs, rhs, .. } => {
                lhs.collect_symbols(syms);
                rhs.collect_symbols(syms);
            }
            SymExpr::Neg { expr } => expr.collect_symbols(syms),
            SymExpr::Func { args, .. } => {
                for arg in args {
                    arg.collect_symbols(syms);
                }
            }
            SymExpr::Vector { elements } => {
                for el in elements {
                    el.collect_symbols(syms);
                }
            }
            _ => {}
        }
    }
}

// --- Display (pretty-print) ---

/// Result of analyzing a multiplication chain into sign + numerator/denominator.
struct MulAnalysis<'a> {
    negative: bool,
    numer: Vec<&'a SymExpr>,
    denom: Vec<DenomFactor<'a>>,
}

struct DenomFactor<'a> {
    base: &'a SymExpr,
    /// The positive exponent (after negation). None means exponent was 1.
    exp: Option<&'a SymExpr>,
}

/// Flatten a Mul chain into sign, numerator factors, and denominator factors.
fn analyze_mul(expr: &SymExpr) -> MulAnalysis<'_> {
    let mut result = MulAnalysis {
        negative: false,
        numer: Vec::new(),
        denom: Vec::new(),
    };
    collect_mul_factors(expr, &mut result);
    result
}

fn collect_mul_factors<'a>(expr: &'a SymExpr, out: &mut MulAnalysis<'a>) {
    match expr {
        SymExpr::BinOp { op: SymOp::Mul, lhs, rhs } => {
            collect_mul_factors(lhs, out);
            collect_mul_factors(rhs, out);
        }
        SymExpr::BinOp { op: SymOp::Div, lhs, rhs } => {
            collect_mul_factors(lhs, out);
            collect_denom_factor(rhs, out);
        }
        SymExpr::Neg { expr: inner } => {
            out.negative = !out.negative;
            collect_mul_factors(inner, out);
        }
        SymExpr::Int { value: 1 } => { /* skip multiplicative identity */ }
        SymExpr::Int { value: -1 } => {
            out.negative = !out.negative;
        }
        // Negative integer: extract sign, keep magnitude
        SymExpr::Int { value } if *value < 0 => {
            out.negative = !out.negative;
            // We can't create a new SymExpr here since we're borrowing,
            // so push the original and let display handle the sign
            out.numer.push(expr);
        }
        // x^(negative_int) → denominator
        SymExpr::BinOp { op: SymOp::Pow, lhs: base, rhs } => {
            match rhs.as_ref() {
                SymExpr::Int { value } if *value < 0 => {
                    if *value == -1 {
                        out.denom.push(DenomFactor { base, exp: None });
                    } else {
                        out.denom.push(DenomFactor { base, exp: Some(rhs) });
                    }
                }
                SymExpr::Neg { expr: inner } => {
                    match inner.as_ref() {
                        SymExpr::Int { value: 1 } => {
                            out.denom.push(DenomFactor { base, exp: None });
                        }
                        _ => {
                            out.denom.push(DenomFactor { base, exp: Some(inner) });
                        }
                    }
                }
                _ => out.numer.push(expr),
            }
        }
        _ => out.numer.push(expr),
    }
}

fn collect_denom_factor<'a>(expr: &'a SymExpr, out: &mut MulAnalysis<'a>) {
    // For the RHS of a Div, the whole thing goes to denominator
    out.denom.push(DenomFactor { base: expr, exp: None });
}

/// Write a factor with parens if it's Add/Sub (lower precedence than Mul).
fn write_mul_factor(f: &mut fmt::Formatter<'_>, expr: &SymExpr) -> fmt::Result {
    let needs_parens = matches!(
        expr,
        SymExpr::BinOp { op: SymOp::Add | SymOp::Sub, .. } | SymExpr::Neg { .. }
    );
    if needs_parens {
        write!(f, "({})", expr)
    } else {
        write!(f, "{}", expr)
    }
}

/// Write a denominator factor: base^pos_exp.
fn write_denom_factor(f: &mut fmt::Formatter<'_>, df: &DenomFactor<'_>) -> fmt::Result {
    match df.exp {
        None => {
            // Just the base (exponent was 1)
            write_mul_factor(f, df.base)
        }
        Some(exp) => {
            // base^positive_exp — need to display the positive version of the exponent
            let base_needs_parens = matches!(
                df.base,
                SymExpr::BinOp { .. } | SymExpr::Neg { .. }
            );
            if base_needs_parens {
                write!(f, "({})", df.base)?;
            } else {
                write!(f, "{}", df.base)?;
            }
            write!(f, "^")?;
            // Display the positive exponent
            match exp {
                // Original was Int(negative), display the absolute value
                SymExpr::Int { value } if *value < 0 => write!(f, "{}", -value),
                // Pow rhs was Neg(inner), we already have inner
                _ => {
                    let exp_needs_parens = matches!(
                        exp,
                        SymExpr::BinOp { .. } | SymExpr::Neg { .. }
                    );
                    if exp_needs_parens {
                        write!(f, "({})", exp)
                    } else {
                        write!(f, "{}", exp)
                    }
                }
            }
        }
    }
}

/// Format a Mul from its analysis. `show_sign` controls whether to write the `-` prefix;
/// `analysis.negative` always controls whether negative-int factors display as absolute values
/// (since analyze_mul already extracted the sign from them).
fn fmt_mul_analysis(analysis: &MulAnalysis<'_>, show_sign: bool, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if show_sign && analysis.negative {
        write!(f, "-")?;
    }

    if analysis.denom.is_empty() {
        // No denominator — just display numerator factors
        if analysis.numer.is_empty() {
            write!(f, "1")?;
        } else {
            for (i, factor) in analysis.numer.iter().enumerate() {
                if i > 0 {
                    write!(f, "*")?;
                }
                // If analyze_mul extracted sign from a negative int, display its absolute value
                if analysis.negative && i == 0 {
                    if let SymExpr::Int { value } = factor {
                        if *value < 0 {
                            write!(f, "{}", -value)?;
                            continue;
                        }
                    }
                }
                write_mul_factor(f, factor)?;
            }
        }
    } else {
        // Has denominator — display as fraction
        // Numerator
        if analysis.numer.is_empty() {
            write!(f, "1")?;
        } else if analysis.numer.len() == 1 {
            let factor = analysis.numer[0];
            if analysis.negative {
                if let SymExpr::Int { value } = factor {
                    if *value < 0 {
                        write!(f, "{}", -value)?;
                    } else {
                        write_mul_factor(f, factor)?;
                    }
                } else {
                    write_mul_factor(f, factor)?;
                }
            } else {
                write_mul_factor(f, factor)?;
            }
        } else {
            for (i, factor) in analysis.numer.iter().enumerate() {
                if i > 0 {
                    write!(f, "*")?;
                }
                if analysis.negative && i == 0 {
                    if let SymExpr::Int { value } = factor {
                        if *value < 0 {
                            write!(f, "{}", -value)?;
                            continue;
                        }
                    }
                }
                write_mul_factor(f, factor)?;
            }
        }

        write!(f, "/")?;

        // Denominator
        if analysis.denom.len() == 1 {
            write_denom_factor(f, &analysis.denom[0])?;
        } else {
            write!(f, "(")?;
            for (i, df) in analysis.denom.iter().enumerate() {
                if i > 0 {
                    write!(f, "*")?;
                }
                write_denom_factor(f, df)?;
            }
            write!(f, ")")?;
        }
    }
    Ok(())
}

/// Format a Mul expression using fraction analysis.
fn fmt_mul(expr: &SymExpr, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let analysis = analyze_mul(expr);
    fmt_mul_analysis(&analysis, true, f)
}

/// Check if an expression would display with a leading negative sign.
fn is_effectively_negative(expr: &SymExpr) -> bool {
    match expr {
        SymExpr::Neg { .. } => true,
        SymExpr::Int { value } => *value < 0,
        SymExpr::Float { value } => *value < 0.0,
        SymExpr::BinOp { op: SymOp::Mul, .. } => analyze_mul(expr).negative,
        _ => false,
    }
}

/// Format the absolute value of a negative expression (for use in subtraction display).
fn fmt_abs(expr: &SymExpr, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match expr {
        SymExpr::Neg { expr: inner } => {
            let needs_parens = matches!(
                inner.as_ref(),
                SymExpr::BinOp { op: SymOp::Add | SymOp::Sub, .. }
            );
            if needs_parens {
                write!(f, "({})", inner)
            } else {
                write!(f, "{}", inner)
            }
        }
        SymExpr::Int { value } => write!(f, "{}", value.abs()),
        SymExpr::Float { value } => {
            let v = value.abs();
            if v.fract() == 0.0 && v < 1e15 {
                write!(f, "{:.1}", v)
            } else {
                write!(f, "{}", v)
            }
        }
        SymExpr::BinOp { op: SymOp::Mul, .. } => {
            let analysis = analyze_mul(expr);
            fmt_mul_analysis(&analysis, false, f)
        }
        _ => write!(f, "{}", expr),
    }
}

impl fmt::Display for SymExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymExpr::Sym { name } => write!(f, "{}", name),
            SymExpr::Int { value } => write!(f, "{}", value),
            SymExpr::Rational { num, den } => write!(f, "{}/{}", num, den),
            SymExpr::Float { value } => {
                if value.fract() == 0.0 && value.abs() < 1e15 {
                    write!(f, "{:.1}", value)
                } else {
                    write!(f, "{}", value)
                }
            }
            SymExpr::Const { name } => match name {
                MathConst::Pi => write!(f, "pi"),
                MathConst::E => write!(f, "e"),
                MathConst::I => write!(f, "i"),
                MathConst::Infinity => write!(f, "inf"),
                MathConst::NegInfinity => write!(f, "-inf"),
            },
            // Mul/Div: use fraction-aware formatter
            SymExpr::BinOp { op: SymOp::Mul, .. } => fmt_mul(self, f),
            SymExpr::BinOp { op: SymOp::Div, lhs, rhs } => {
                // Treat Div as a simple fraction
                let lhs_needs_parens = matches!(
                    lhs.as_ref(),
                    SymExpr::BinOp { op: SymOp::Add | SymOp::Sub, .. }
                );
                let rhs_needs_parens = matches!(
                    rhs.as_ref(),
                    SymExpr::BinOp { .. } | SymExpr::Neg { .. }
                );

                if lhs_needs_parens {
                    write!(f, "({})", lhs)?;
                } else {
                    write!(f, "{}", lhs)?;
                }
                write!(f, "/")?;
                if rhs_needs_parens {
                    write!(f, "({})", rhs)?;
                } else {
                    write!(f, "{}", rhs)?;
                }
                Ok(())
            }
            // Add: check if rhs is effectively negative for sign handling
            SymExpr::BinOp { op: SymOp::Add, lhs, rhs } => {
                write!(f, "{}", lhs)?;

                if is_effectively_negative(rhs) {
                    write!(f, " - ")?;
                    fmt_abs(rhs, f)
                } else {
                    write!(f, " + ")?;
                    write!(f, "{}", rhs)
                }
            }
            SymExpr::BinOp { op: SymOp::Sub, lhs, rhs } => {
                write!(f, "{}", lhs)?;
                write!(f, " - ")?;
                let rhs_needs_parens = matches!(
                    rhs.as_ref(),
                    SymExpr::BinOp { op: SymOp::Add | SymOp::Sub, .. }
                );
                if rhs_needs_parens {
                    write!(f, "({})", rhs)
                } else {
                    write!(f, "{}", rhs)
                }
            }
            SymExpr::BinOp { op: SymOp::Pow, lhs, rhs } => {
                let needs_lhs_parens = matches!(
                    lhs.as_ref(),
                    SymExpr::BinOp { .. } | SymExpr::Neg { .. }
                );
                if needs_lhs_parens {
                    write!(f, "({})", lhs)?;
                } else {
                    write!(f, "{}", lhs)?;
                }
                write!(f, "^")?;
                // Pow is right-associative, so no parens needed for Pow on rhs
                let needs_rhs_parens = matches!(
                    rhs.as_ref(),
                    SymExpr::BinOp { op: SymOp::Add | SymOp::Sub | SymOp::Mul | SymOp::Div, .. }
                    | SymExpr::Neg { .. }
                );
                if needs_rhs_parens {
                    write!(f, "({})", rhs)
                } else {
                    write!(f, "{}", rhs)
                }
            }
            SymExpr::Neg { expr } => {
                let needs_parens = matches!(
                    expr.as_ref(),
                    SymExpr::BinOp { op: SymOp::Add | SymOp::Sub, .. }
                );
                if needs_parens {
                    write!(f, "-({})", expr)
                } else {
                    write!(f, "-{}", expr)
                }
            }
            SymExpr::Func { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            SymExpr::Vector { elements } => {
                write!(f, "[")?;
                for (i, el) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", el)?;
                }
                write!(f, "]")
            }
            SymExpr::Undefined => write!(f, "undefined"),
        }
    }
}

/// Convert an integer to Unicode superscript characters.
fn to_superscript(n: i64) -> String {
    const SUPERSCRIPTS: &[char] = &['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    let s = n.abs().to_string();
    let mut result = String::new();
    if n < 0 {
        result.push('⁻');
    }
    for ch in s.chars() {
        let digit = ch as usize - '0' as usize;
        result.push(SUPERSCRIPTS[digit]);
    }
    result
}

/// Pretty-mode constants display.
fn pretty_const(name: &MathConst) -> &'static str {
    match name {
        MathConst::Pi => "π",
        MathConst::E => "e",
        MathConst::I => "i",
        MathConst::Infinity => "∞",
        MathConst::NegInfinity => "-∞",
    }
}

impl SymExpr {
    /// Format with the given output mode.
    pub fn format_pretty(&self) -> String {
        match self {
            SymExpr::Sym { name } => name.clone(),
            SymExpr::Int { value } => format!("{}", value),
            SymExpr::Rational { num, den } => format!("{}/{}", num, den),
            SymExpr::Float { value } => {
                if value.fract() == 0.0 && value.abs() < 1e15 {
                    format!("{:.1}", value)
                } else {
                    format!("{}", value)
                }
            }
            SymExpr::Const { name } => pretty_const(name).to_string(),
            SymExpr::BinOp { op: SymOp::Pow, lhs, rhs } => {
                // Pretty mode: use superscript for integer exponents
                if let Some(n) = rhs.as_int() {
                    let base = lhs.format_pretty();
                    let needs_parens = matches!(
                        lhs.as_ref(),
                        SymExpr::BinOp { .. } | SymExpr::Neg { .. }
                    );
                    if needs_parens {
                        format!("({}){}", base, to_superscript(n))
                    } else {
                        format!("{}{}", base, to_superscript(n))
                    }
                } else {
                    // Non-integer exponent: fall back to ^ notation
                    let base = lhs.format_pretty();
                    let exp = rhs.format_pretty();
                    let needs_lhs_parens = matches!(
                        lhs.as_ref(),
                        SymExpr::BinOp { .. } | SymExpr::Neg { .. }
                    );
                    let needs_rhs_parens = matches!(
                        rhs.as_ref(),
                        SymExpr::BinOp { op: SymOp::Add | SymOp::Sub | SymOp::Mul | SymOp::Div, .. }
                        | SymExpr::Neg { .. }
                    );
                    format!("{}^{}",
                        if needs_lhs_parens { format!("({})", base) } else { base },
                        if needs_rhs_parens { format!("({})", exp) } else { exp },
                    )
                }
            }
            SymExpr::BinOp { op: SymOp::Mul, .. } => self.format_pretty_mul(),
            SymExpr::BinOp { op: SymOp::Div, lhs, rhs } => {
                let l = lhs.format_pretty();
                let r = rhs.format_pretty();
                let lp = matches!(lhs.as_ref(), SymExpr::BinOp { op: SymOp::Add | SymOp::Sub, .. });
                let rp = matches!(rhs.as_ref(), SymExpr::BinOp { .. } | SymExpr::Neg { .. });
                format!("{}/{}",
                    if lp { format!("({})", l) } else { l },
                    if rp { format!("({})", r) } else { r },
                )
            }
            SymExpr::BinOp { op: SymOp::Add, lhs, rhs } => {
                let l = lhs.format_pretty();
                if is_effectively_negative(rhs) {
                    format!("{} - {}", l, self.format_pretty_abs(rhs))
                } else {
                    format!("{} + {}", l, rhs.format_pretty())
                }
            }
            SymExpr::BinOp { op: SymOp::Sub, lhs, rhs } => {
                let l = lhs.format_pretty();
                let r = rhs.format_pretty();
                let rp = matches!(rhs.as_ref(), SymExpr::BinOp { op: SymOp::Add | SymOp::Sub, .. });
                format!("{} - {}", l, if rp { format!("({})", r) } else { r })
            }
            SymExpr::Neg { expr } => {
                let inner = expr.format_pretty();
                let needs_parens = matches!(
                    expr.as_ref(),
                    SymExpr::BinOp { op: SymOp::Add | SymOp::Sub, .. }
                );
                if needs_parens { format!("-({})", inner) } else { format!("-{}", inner) }
            }
            SymExpr::Func { name, args } => {
                let arg_strs: Vec<String> = args.iter().map(|a| a.format_pretty()).collect();
                format!("{}({})", name, arg_strs.join(", "))
            }
            SymExpr::Vector { elements } => {
                let el_strs: Vec<String> = elements.iter().map(|e| e.format_pretty()).collect();
                format!("[{}]", el_strs.join(", "))
            }
            SymExpr::Undefined => "undefined".to_string(),
        }
    }

    /// Extract integer value if this is an Int node.
    fn as_int(&self) -> Option<i64> {
        match self {
            SymExpr::Int { value } => Some(*value),
            _ => None,
        }
    }

    /// Pretty-format a Mul chain using fraction analysis.
    fn format_pretty_mul(&self) -> String {
        let analysis = analyze_mul(self);
        let mut result = String::new();
        if analysis.negative {
            result.push('-');
        }

        let numer_str = if analysis.numer.is_empty() {
            "1".to_string()
        } else {
            let mut parts = Vec::new();
            for (i, factor) in analysis.numer.iter().enumerate() {
                let s = if analysis.negative && i == 0 {
                    if let SymExpr::Int { value } = factor {
                        if *value < 0 { format!("{}", -value) } else { factor.format_pretty() }
                    } else { factor.format_pretty() }
                } else { factor.format_pretty() };
                let needs_parens = matches!(factor, SymExpr::BinOp { op: SymOp::Add | SymOp::Sub, .. } | SymExpr::Neg { .. });
                if needs_parens { parts.push(format!("({})", s)); } else { parts.push(s); }
            }
            parts.join("*")
        };

        if analysis.denom.is_empty() {
            result.push_str(&numer_str);
        } else {
            result.push_str(&numer_str);
            result.push('/');
            if analysis.denom.len() == 1 {
                result.push_str(&self.format_pretty_denom_factor(&analysis.denom[0]));
            } else {
                result.push('(');
                let parts: Vec<String> = analysis.denom.iter()
                    .map(|df| self.format_pretty_denom_factor(df))
                    .collect();
                result.push_str(&parts.join("*"));
                result.push(')');
            }
        }
        result
    }

    fn format_pretty_denom_factor(&self, df: &DenomFactor<'_>) -> String {
        match df.exp {
            None => {
                let s = df.base.format_pretty();
                let needs_parens = matches!(df.base, SymExpr::BinOp { op: SymOp::Add | SymOp::Sub, .. } | SymExpr::Neg { .. });
                if needs_parens { format!("({})", s) } else { s }
            }
            Some(exp) => {
                let base_s = df.base.format_pretty();
                let needs_parens = matches!(df.base, SymExpr::BinOp { .. } | SymExpr::Neg { .. });
                let base = if needs_parens { format!("({})", base_s) } else { base_s };
                // Use superscript for the positive exponent if it's an integer
                match exp {
                    SymExpr::Int { value } if *value < 0 => format!("{}{}", base, to_superscript(-value)),
                    SymExpr::Int { value } => format!("{}{}", base, to_superscript(*value)),
                    _ => {
                        if let Some(n) = exp.as_int() {
                            format!("{}{}", base, to_superscript(n))
                        } else {
                            format!("{}^{}", base, exp.format_pretty())
                        }
                    }
                }
            }
        }
    }

    fn format_pretty_abs(&self, expr: &SymExpr) -> String {
        match expr {
            SymExpr::Neg { expr: inner } => {
                let s = inner.format_pretty();
                let needs_parens = matches!(inner.as_ref(), SymExpr::BinOp { op: SymOp::Add | SymOp::Sub, .. });
                if needs_parens { format!("({})", s) } else { s }
            }
            SymExpr::Int { value } => format!("{}", value.abs()),
            SymExpr::Float { value } => {
                let v = value.abs();
                if v.fract() == 0.0 && v < 1e15 { format!("{:.1}", v) } else { format!("{}", v) }
            }
            SymExpr::BinOp { op: SymOp::Mul, .. } => {
                let analysis = analyze_mul(expr);
                // Format with negative suppressed — reuse logic
                let mut s = String::new();
                // Same as format_pretty_mul but without sign
                let numer_str = if analysis.numer.is_empty() {
                    "1".to_string()
                } else {
                    let mut parts = Vec::new();
                    for (i, factor) in analysis.numer.iter().enumerate() {
                        let fs = if analysis.negative && i == 0 {
                            if let SymExpr::Int { value } = factor {
                                if *value < 0 { format!("{}", -value) } else { factor.format_pretty() }
                            } else { factor.format_pretty() }
                        } else { factor.format_pretty() };
                        let needs_parens = matches!(factor, SymExpr::BinOp { op: SymOp::Add | SymOp::Sub, .. } | SymExpr::Neg { .. });
                        if needs_parens { parts.push(format!("({})", fs)); } else { parts.push(fs); }
                    }
                    parts.join("*")
                };
                if analysis.denom.is_empty() {
                    s.push_str(&numer_str);
                } else {
                    s.push_str(&numer_str);
                    s.push('/');
                    if analysis.denom.len() == 1 {
                        s.push_str(&expr.format_pretty_denom_factor(&analysis.denom[0]));
                    } else {
                        s.push('(');
                        let parts: Vec<String> = analysis.denom.iter()
                            .map(|df| expr.format_pretty_denom_factor(df))
                            .collect();
                        s.push_str(&parts.join("*"));
                        s.push(')');
                    }
                }
                s
            }
            _ => expr.format_pretty(),
        }
    }
}

impl fmt::Display for MathConst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathConst::Pi => write!(f, "pi"),
            MathConst::E => write!(f, "e"),
            MathConst::I => write!(f, "i"),
            MathConst::Infinity => write!(f, "inf"),
            MathConst::NegInfinity => write!(f, "-inf"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_simple() {
        let expr = SymExpr::mul(SymExpr::int(2), SymExpr::sym("x"));
        assert_eq!(format!("{}", expr), "2*x");
    }

    #[test]
    fn test_display_precedence() {
        let expr = SymExpr::add(
            SymExpr::mul(SymExpr::int(2), SymExpr::sym("x")),
            SymExpr::int(1),
        );
        assert_eq!(format!("{}", expr), "2*x + 1");
    }

    #[test]
    fn test_display_parens_needed() {
        let expr = SymExpr::mul(
            SymExpr::int(2),
            SymExpr::add(SymExpr::sym("x"), SymExpr::int(1)),
        );
        assert_eq!(format!("{}", expr), "2*(x + 1)");
    }

    #[test]
    fn test_json_roundtrip() {
        let expr = SymExpr::add(
            SymExpr::mul(SymExpr::int(3), SymExpr::pow(SymExpr::sym("x"), SymExpr::int(2))),
            SymExpr::int(1),
        );
        let json = serde_json::to_string(&expr).unwrap();
        let back: SymExpr = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, back);
    }

    #[test]
    fn test_free_symbols() {
        let expr = SymExpr::add(
            SymExpr::mul(SymExpr::sym("x"), SymExpr::sym("y")),
            SymExpr::int(1),
        );
        assert_eq!(expr.free_symbols(), vec!["x", "y"]);
    }

    // --- Pretty output tests ---

    #[test]
    fn test_add_neg_sign_handling() {
        // a + Neg(b) → "a - b"
        let expr = SymExpr::add(
            SymExpr::sym("a"),
            SymExpr::neg(SymExpr::sym("b")),
        );
        assert_eq!(format!("{}", expr), "a - b");
    }

    #[test]
    fn test_add_neg_compound() {
        // a + Neg(x + y) → "a - (x + y)"
        let expr = SymExpr::add(
            SymExpr::sym("a"),
            SymExpr::neg(SymExpr::add(SymExpr::sym("x"), SymExpr::sym("y"))),
        );
        assert_eq!(format!("{}", expr), "a - (x + y)");
    }

    #[test]
    fn test_fraction_simple() {
        // x * y^(-1) → "x/y"
        let expr = SymExpr::mul(
            SymExpr::sym("x"),
            SymExpr::pow(SymExpr::sym("y"), SymExpr::int(-1)),
        );
        assert_eq!(format!("{}", expr), "x/y");
    }

    #[test]
    fn test_fraction_higher_neg_exp() {
        // a * x^(-2) → "a/x^2"
        let expr = SymExpr::mul(
            SymExpr::sym("a"),
            SymExpr::pow(SymExpr::sym("x"), SymExpr::int(-2)),
        );
        assert_eq!(format!("{}", expr), "a/x^2");
    }

    #[test]
    fn test_fraction_multi_denom() {
        // a * x^(-1) * y^(-1) → "a/(x*y)"
        let expr = SymExpr::mul(
            SymExpr::mul(
                SymExpr::sym("a"),
                SymExpr::pow(SymExpr::sym("x"), SymExpr::int(-1)),
            ),
            SymExpr::pow(SymExpr::sym("y"), SymExpr::int(-1)),
        );
        assert_eq!(format!("{}", expr), "a/(x*y)");
    }

    #[test]
    fn test_fraction_empty_numer() {
        // x^(-1) → "1/x"
        // x^(-1) as a Pow goes through Div conversion in python_bridge,
        // but wrapped in Mul: 1 * x^(-1) gets fraction display via the mul chain
        let expr2 = SymExpr::mul(
            SymExpr::int(1),
            SymExpr::pow(SymExpr::sym("x"), SymExpr::int(-1)),
        );
        assert_eq!(format!("{}", expr2), "1/x");
    }

    #[test]
    fn test_coeff_cleanup_one() {
        // 1 * x → "x"
        let expr = SymExpr::mul(SymExpr::int(1), SymExpr::sym("x"));
        assert_eq!(format!("{}", expr), "x");
    }

    #[test]
    fn test_coeff_cleanup_neg_one() {
        // -1 * x → "-x"
        let expr = SymExpr::mul(SymExpr::int(-1), SymExpr::sym("x"));
        assert_eq!(format!("{}", expr), "-x");
    }

    #[test]
    fn test_neg_mul_cleanup() {
        // Neg(Mul(2, x)) → "-2*x"
        let expr = SymExpr::neg(SymExpr::mul(SymExpr::int(2), SymExpr::sym("x")));
        assert_eq!(format!("{}", expr), "-2*x");
    }

    #[test]
    fn test_imaginary_const() {
        let expr = SymExpr::Const { name: MathConst::I };
        assert_eq!(format!("{}", expr), "i");
    }

    #[test]
    fn test_add_neg_coeff_mul() {
        // a + Mul(Int(-2), f, i, m) → "a - 2*f*i*m"
        // This is what CAS returns for negative-coefficient terms
        let expr = SymExpr::add(
            SymExpr::sym("a"),
            SymExpr::mul(
                SymExpr::mul(
                    SymExpr::mul(SymExpr::int(-2), SymExpr::sym("f")),
                    SymExpr::Const { name: MathConst::I },
                ),
                SymExpr::sym("m"),
            ),
        );
        assert_eq!(format!("{}", expr), "a - 2*f*i*m");
    }

    #[test]
    fn test_add_neg_int() {
        // a + Int(-3) → "a - 3"
        let expr = SymExpr::add(SymExpr::sym("a"), SymExpr::int(-3));
        assert_eq!(format!("{}", expr), "a - 3");
    }
}
