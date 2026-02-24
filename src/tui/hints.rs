/// Function signature hints for the input bar.
///
/// Detects when the cursor is inside a function call and provides
/// the function's signature and a one-line description.

pub struct FuncHint {
    pub name: &'static str,
    pub signature: &'static str,
    pub description: &'static str,
}

/// All known function signatures (CAS ops + builtins).
pub static FUNCTION_HINTS: &[FuncHint] = &[
    // CAS operations
    FuncHint { name: "dif",      signature: "dif(expr, var?, order?)",         description: "Differentiate expression" },
    FuncHint { name: "int",      signature: "int(expr, var?, lo..hi or lo, hi)", description: "Integrate expression" },
    FuncHint { name: "solve",    signature: "solve(expr, var?)",               description: "Solve equation for variable" },
    FuncHint { name: "simplify", signature: "simplify(expr)",                  description: "Simplify expression" },
    FuncHint { name: "expand",   signature: "expand(expr)",                    description: "Expand expression" },
    FuncHint { name: "factor",   signature: "factor(expr)",                    description: "Factor expression" },
    FuncHint { name: "lim",      signature: "lim(expr, var, point, dir?)",     description: "Compute limit" },
    FuncHint { name: "taylor",   signature: "taylor(expr, var, point, order?)", description: "Taylor series expansion" },
    FuncHint { name: "tex",      signature: "tex(expr)",                       description: "Convert to LaTeX string" },
    // Plot
    FuncHint { name: "plot",     signature: "plot(expr, range?) or plot([e1, e2, ...], range?)", description: "Plot expression(s)" },
    // Numeric eval
    FuncHint { name: "eval",     signature: "eval(expr)",                      description: "Force numeric evaluation" },
    // Backend control
    FuncHint { name: "backend",  signature: "backend(\"sympy\"|\"maxima\"|\"both\")", description: "Set active CAS backend" },
    FuncHint { name: "using",    signature: "using(\"backend\", expr)",        description: "Evaluate with specific backend" },
    // Math (1 arg)
    FuncHint { name: "sin",      signature: "sin(x)",     description: "Sine" },
    FuncHint { name: "cos",      signature: "cos(x)",     description: "Cosine" },
    FuncHint { name: "tan",      signature: "tan(x)",     description: "Tangent" },
    FuncHint { name: "asin",     signature: "asin(x)",    description: "Inverse sine" },
    FuncHint { name: "acos",     signature: "acos(x)",    description: "Inverse cosine" },
    FuncHint { name: "atan",     signature: "atan(x)",    description: "Inverse tangent" },
    FuncHint { name: "sinh",     signature: "sinh(x)",    description: "Hyperbolic sine" },
    FuncHint { name: "cosh",     signature: "cosh(x)",    description: "Hyperbolic cosine" },
    FuncHint { name: "tanh",     signature: "tanh(x)",    description: "Hyperbolic tangent" },
    FuncHint { name: "exp",      signature: "exp(x)",     description: "Exponential (e^x)" },
    FuncHint { name: "ln",       signature: "ln(x)",      description: "Natural logarithm" },
    FuncHint { name: "sqrt",     signature: "sqrt(x)",    description: "Square root" },
    FuncHint { name: "abs",      signature: "abs(x)",     description: "Absolute value" },
    FuncHint { name: "floor",    signature: "floor(x)",   description: "Floor (round down)" },
    FuncHint { name: "ceil",     signature: "ceil(x)",    description: "Ceiling (round up)" },
    FuncHint { name: "round",    signature: "round(x)",   description: "Round to nearest integer" },
    FuncHint { name: "sign",     signature: "sign(x)",    description: "Sign (-1, 0, or 1)" },
    // Math (variadic)
    FuncHint { name: "log",      signature: "log(x) or log(base, x)", description: "Logarithm" },
    FuncHint { name: "max",      signature: "max(a, b, ...)",         description: "Maximum value" },
    FuncHint { name: "min",      signature: "min(a, b, ...)",         description: "Minimum value" },
    // Utility
    FuncHint { name: "print",    signature: "print(args...)",  description: "Print values" },
    FuncHint { name: "len",      signature: "len(v)",          description: "Length of vector or string" },
    FuncHint { name: "typeof",   signature: "typeof(x)",       description: "Type name of value" },
];

/// Detect the innermost function call surrounding the cursor position.
///
/// Walks backwards from `cursor` tracking parenthesis depth to find the
/// nearest unmatched `(` preceded by an identifier. Returns the matching
/// hint if one exists.
pub fn detect_active_function(text: &str, cursor: usize) -> Option<&'static FuncHint> {
    let bytes = text.as_bytes();
    let pos = cursor.min(bytes.len());
    let mut depth: i32 = 0;

    // Walk backwards from cursor
    let mut i = pos;
    while i > 0 {
        i -= 1;
        match bytes[i] {
            b')' => depth += 1,
            b'(' => {
                if depth > 0 {
                    depth -= 1;
                } else {
                    // Found unmatched '(' — extract the identifier before it
                    let name = extract_ident_before(bytes, i);
                    if let Some(name) = name {
                        if let Some(hint) = lookup_hint(name) {
                            return Some(hint);
                        }
                    }
                    // Not a known function; keep walking to find outer call
                }
            }
            _ => {}
        }
    }
    None
}

/// Extract an identifier immediately before position `pos` in `bytes`.
fn extract_ident_before(bytes: &[u8], pos: usize) -> Option<&str> {
    if pos == 0 {
        return None;
    }
    let mut start = pos;
    // Skip whitespace between identifier and '('
    while start > 0 && bytes[start - 1] == b' ' {
        start -= 1;
    }
    let end = start;
    // Walk back through identifier characters
    while start > 0 && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_') {
        start -= 1;
    }
    if start == end {
        return None;
    }
    std::str::from_utf8(&bytes[start..end]).ok()
}

fn lookup_hint(name: &str) -> Option<&'static FuncHint> {
    FUNCTION_HINTS.iter().find(|h| h.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_function() {
        let text = "sin(";
        let hint = detect_active_function(text, 4);
        assert_eq!(hint.unwrap().name, "sin");
    }

    #[test]
    fn test_nested_inner() {
        // cursor at end → innermost is cos
        let text = "sin(cos(";
        let hint = detect_active_function(text, 8);
        assert_eq!(hint.unwrap().name, "cos");
    }

    #[test]
    fn test_nested_outer() {
        // cos is closed, cursor after comma → outer is sin
        let text = "sin(cos(x), ";
        let hint = detect_active_function(text, 12);
        assert_eq!(hint.unwrap().name, "sin");
    }

    #[test]
    fn test_no_function() {
        let text = "x + 1";
        assert!(detect_active_function(text, 5).is_none());
    }

    #[test]
    fn test_cursor_after_close() {
        let text = "sin(x)";
        // cursor after closing paren — no active function
        assert!(detect_active_function(text, 6).is_none());
    }

    #[test]
    fn test_cas_function() {
        let text = "dif(x^2, ";
        let hint = detect_active_function(text, 9);
        assert_eq!(hint.unwrap().name, "dif");
    }

    #[test]
    fn test_unknown_function() {
        let text = "foo(";
        assert!(detect_active_function(text, 4).is_none());
    }
}
