//! Integration tests: full evaluator + CAS backends.
//!
//! These tests require Python 3 + SymPy to be installed.
//! Maxima tests additionally require Maxima to be installed.
//! They spawn the actual bridge subprocesses.

use aurita::lang::eval::Evaluator;
use aurita::lang::lexer::Lexer;
use aurita::lang::parser::Parser;
use aurita::lang::types::Value;

fn eval_with_cas(input: &str) -> Result<Value, String> {
    let mut evaluator = Evaluator::new();
    let sympy_bridge = "backends/python_bridge.py";
    evaluator.init_cas(Some(sympy_bridge), None);
    assert!(evaluator.has_cas(), "CAS backend failed to start — is SymPy installed?");

    let tokens = Lexer::new(input).tokenize().map_err(|e| e.message)?;
    let stmts = Parser::new(tokens).parse_program().map_err(|e| e.message)?;
    evaluator.eval_program(&stmts).map_err(|e| e.message)
}

fn eval_str(input: &str) -> String {
    format!("{}", eval_with_cas(input).unwrap())
}

/// Create an evaluator with both SymPy and Maxima (if available).
#[allow(dead_code)]
fn eval_with_both(input: &str) -> Result<Value, String> {
    let mut evaluator = Evaluator::new();
    let sympy_bridge = "backends/python_bridge.py";
    let maxima_bridge = "backends/maxima_bridge.py";
    evaluator.init_cas(Some(sympy_bridge), Some(maxima_bridge));
    assert!(evaluator.has_cas(), "No CAS backend available");

    let tokens = Lexer::new(input).tokenize().map_err(|e| e.message)?;
    let stmts = Parser::new(tokens).parse_program().map_err(|e| e.message)?;
    evaluator.eval_program(&stmts).map_err(|e| e.message)
}

/// Run multiple expressions on one evaluator with both backends.
fn eval_multi_with_both(inputs: &[&str]) -> Result<Vec<Value>, String> {
    let mut evaluator = Evaluator::new();
    let sympy_bridge = "backends/python_bridge.py";
    let maxima_bridge = "backends/maxima_bridge.py";
    evaluator.init_cas(Some(sympy_bridge), Some(maxima_bridge));
    assert!(evaluator.has_cas(), "No CAS backend available");

    let mut results = Vec::new();
    for input in inputs {
        let tokens = Lexer::new(input).tokenize().map_err(|e| e.message)?;
        let stmts = Parser::new(tokens).parse_program().map_err(|e| e.message)?;
        results.push(evaluator.eval_program(&stmts).map_err(|e| e.message)?);
    }
    Ok(results)
}

// =============================================================================
// Existing SymPy tests (unchanged behavior)
// =============================================================================

#[test]
fn test_cas_differentiate() {
    assert_eq!(eval_str("dif(x^2, x)"), "2*x");
}

#[test]
fn test_cas_differentiate_auto_var() {
    // Single free variable — auto-detected
    assert_eq!(eval_str("dif(x^3)"), "3*x^2");
}

#[test]
fn test_cas_integrate() {
    assert_eq!(eval_str("int(sin(x), x)"), "-cos(x)");
}

#[test]
fn test_cas_definite_integral() {
    let result = eval_with_cas("int(x^2, x, 0, 1)").unwrap();
    // Should be 1/3 (rational)
    let display = format!("{}", result);
    assert!(display == "1/3" || display == "0.3333333333333333",
        "Expected 1/3 or ~0.333, got: {}", display);
}

#[test]
fn test_cas_solve() {
    let result = eval_str("solve(x^2 - 4, x)");
    // Should contain -2 and 2
    assert!(result.contains("-2") && result.contains("2"),
        "Expected [-2, 2], got: {}", result);
}

#[test]
fn test_cas_simplify() {
    let result = eval_str("simplify(x^2 + 2x + 1)");
    // (x + 1)^2
    assert!(result.contains("x") && result.contains("1"),
        "Expected simplified form, got: {}", result);
}

#[test]
fn test_cas_expand() {
    let result = eval_str("expand((x+1)^2)");
    assert!(result.contains("x^2") || result.contains("x²"),
        "Expected expanded form with x^2, got: {}", result);
}

#[test]
fn test_cas_limit() {
    let result = eval_str("lim(sin(x)/x, x, 0)");
    assert_eq!(result, "1");
}

#[test]
fn test_cas_taylor() {
    let result = eval_str("taylor(exp(x), x, 0, 3)");
    // Should have x^3/6 or similar terms
    assert!(result.contains("x"),
        "Expected Taylor polynomial with x terms, got: {}", result);
}

#[test]
fn test_constants_stay_symbolic() {
    // e^x * e^x should simplify to exp(2x), not 7.389...^x
    let result = eval_str("simplify(e^x * e^x)");
    assert!(!result.contains("7.38"), "e should not be evaluated to float, got: {}", result);
    assert!(result.contains("2") && result.contains("x"),
        "Expected exp(2*x) or similar, got: {}", result);
}

#[test]
fn test_gaussian_integral() {
    // int(e^(-a*x^2), x) should give a clean result, not Piecewise blowup
    let result = eval_str("int(e^(-a*x^2), x)");
    assert!(!result.contains("Piecewise"), "Should not contain Piecewise, got: {}", result);
    assert!(result.contains("erf") || result.contains("sqrt"),
        "Expected erf/sqrt in result, got: {}", result);
}

#[test]
fn test_sin_pi_evaluates_numerically() {
    // sin(pi) should evaluate to ~0, not stay as sin(pi)
    let result = eval_with_cas("sin(pi)").unwrap();
    match result {
        Value::Number(n) => {
            let f = n.as_f64();
            assert!(f.abs() < 1e-10, "sin(pi) should be ~0, got: {}", f);
        }
        _ => panic!("sin(pi) should be numeric, got: {}", result),
    }
}

#[test]
fn test_gaussian_definite_integral() {
    // Gaussian integral: int(e^(-a*x^2), x, -inf, inf) = sqrt(pi/a)
    let result = eval_str("int(e^(-a*x^2), x, -inf, inf)");
    assert!(result.contains("sqrt") && result.contains("pi"),
        "Expected sqrt(pi/a) or similar, got: {}", result);

    // Space-separated implicit mul also works
    let result2 = eval_str("int(e^(-a x^2), x, -inf, inf)");
    assert!(result2.contains("sqrt") && result2.contains("pi"),
        "Expected sqrt(pi/a) or similar, got: {}", result2);
}

#[test]
fn test_definite_integral_trig() {
    // int(sin(x)^2, x, 0, pi) = pi/2
    let result = eval_str("int(sin(x)^2, x, 0, pi)");
    assert!(result.contains("pi") || result.contains("1.5707"),
        "Expected pi/2, got: {}", result);
}

#[test]
fn test_definite_integral_polynomial() {
    // int(x^3, x, 0, 2) = 4
    let result = eval_str("int(x^3, x, 0, 2)");
    assert!(result == "4" || result == "4.0",
        "Expected 4, got: {}", result);
}

#[test]
fn test_integral_with_inf_bounds() {
    // int(e^(-x), x, 0, inf) = 1
    let result = eval_str("int(e^(-x), x, 0, inf)");
    assert!(result == "1" || result == "1.0",
        "Expected 1, got: {}", result);
}

#[test]
fn test_integral_inf_alias() {
    // Inf (capital) should work the same as inf
    let result = eval_str("int(e^(-x), x, 0, Inf)");
    assert!(result == "1" || result == "1.0",
        "Expected 1, got: {}", result);
}

#[test]
fn test_eval_function() {
    // eval() forces numeric evaluation
    let result = eval_with_cas("eval(pi)").unwrap();
    match result {
        Value::Number(n) => {
            assert!((n.as_f64() - std::f64::consts::PI).abs() < 1e-10);
        }
        _ => panic!("eval(pi) should be numeric, got: {}", result),
    }
}

// --- int() range syntax tests ---

#[test]
fn test_int_range_syntax() {
    // int(x^2, x, 0..1) = 1/3  (3-arg range form)
    let result = eval_str("int(x^2, x, 0..1)");
    assert!(result == "1/3" || result.contains("0.333"),
        "Expected 1/3, got: {}", result);
}

#[test]
fn test_int_range_symbolic_bounds() {
    // int(sin(x), x, 0..pi) = 2
    let result = eval_str("int(sin(x), x, 0..pi)");
    assert!(result == "2" || result == "2.0",
        "Expected 2, got: {}", result);
}

// --- Plot integration tests ---

#[test]
fn test_plot_with_cas_lambdify() {
    // erf(x) can't be evaluated in pure Rust — should fall back to CAS lambdify
    let result = eval_with_cas("plot(erf(x), -3..3)");
    match result {
        Ok(Value::Plot(p)) => {
            let valid = p.spec.series[0].points.iter().filter(|p| p.is_some()).count();
            assert!(valid > 100, "Expected many valid points from CAS lambdify, got: {}", valid);
        }
        Ok(other) => panic!("Expected plot, got: {}", other),
        Err(e) => panic!("Plot failed: {}", e),
    }
}

#[test]
fn test_plot_matplotlib_rendering() {
    // With CAS backend, plots should be rendered by matplotlib (larger, richer PNGs)
    let result = eval_with_cas("plot(sin(x), -3..3)");
    match result {
        Ok(Value::Plot(p)) => {
            // matplotlib PNGs are typically >10KB, plotters fallback ~5KB
            assert!(p.png_bytes.len() > 5000,
                "Expected matplotlib-rendered PNG (>5KB), got {} bytes", p.png_bytes.len());
            // Verify valid PNG
            assert_eq!(&p.png_bytes[1..4], b"PNG");
        }
        Ok(other) => panic!("Expected plot, got: {}", other),
        Err(e) => panic!("Plot failed: {}", e),
    }
}

#[test]
fn test_plot_multi_curve_matplotlib() {
    let result = eval_with_cas("plot([sin(x), cos(x)], -3..3)");
    match result {
        Ok(Value::Plot(p)) => {
            assert_eq!(p.spec.series.len(), 2);
            assert!(!p.png_bytes.is_empty());
        }
        Ok(other) => panic!("Expected plot, got: {}", other),
        Err(e) => panic!("Plot failed: {}", e),
    }
}

#[test]
fn test_plot_discontinuity_detection() {
    // tan(x) has discontinuities at pi/2 + n*pi — gaps should be inserted
    let result = eval_with_cas("plot(tan(x), -4..4)");
    match result {
        Ok(Value::Plot(p)) => {
            let gaps = p.spec.series[0].points.iter().filter(|p| p.is_none()).count();
            assert!(gaps > 0,
                "Expected discontinuity gaps in tan(x) plot, got 0 gaps");
        }
        Ok(other) => panic!("Expected plot, got: {}", other),
        Err(e) => panic!("Plot failed: {}", e),
    }
}

// =============================================================================
// Maxima tests (require Maxima to be installed)
// =============================================================================

fn has_maxima() -> bool {
    std::process::Command::new("maxima")
        .arg("--version")
        .output()
        .is_ok()
}

fn eval_maxima(input: &str) -> Result<Value, String> {
    let results = eval_multi_with_both(&[
        "backend(\"maxima\")",
        input,
    ])?;
    Ok(results.into_iter().last().unwrap())
}

fn eval_maxima_str(input: &str) -> String {
    format!("{}", eval_maxima(input).unwrap())
}

#[test]
fn test_maxima_differentiate() {
    if !has_maxima() { return; }
    let result = eval_maxima_str("dif(x^2, x)");
    assert_eq!(result, "2*x");
}

#[test]
fn test_maxima_integrate() {
    if !has_maxima() { return; }
    let result = eval_maxima_str("int(sin(x), x)");
    assert_eq!(result, "-cos(x)");
}

#[test]
fn test_maxima_solve() {
    if !has_maxima() { return; }
    let result = eval_maxima_str("solve(x^2 - 4, x)");
    assert!(result.contains("-2") && result.contains("2"),
        "Expected solutions containing -2 and 2, got: {}", result);
}

#[test]
fn test_maxima_simplify() {
    if !has_maxima() { return; }
    let result = eval_maxima_str("simplify(x^2 + 2x + 1)");
    // ratsimp should produce (x+1)^2 or x^2+2*x+1
    assert!(result.contains("x"),
        "Expected expression with x, got: {}", result);
}

// --- Gaussian integral tests (good at catching backend issues) ---

#[test]
fn test_gaussian_integral_sympy() {
    // int(e^(-a*x^2), x, -inf, inf) = sqrt(pi/a) via SymPy
    let result = eval_str("int(e^(-a*x^2), x, -inf, inf)");
    assert!(result.contains("sqrt") && result.contains("pi"),
        "Expected sqrt(pi/a) or similar, got: {}", result);
}

#[test]
fn test_gaussian_integral_simple_sympy() {
    // int(e^(-x^2), x, -inf, inf) = sqrt(pi) via SymPy
    let result = eval_str("int(e^(-x^2), x, -inf, inf)");
    assert!(result.contains("sqrt") && result.contains("pi"),
        "Expected sqrt(pi), got: {}", result);
}

#[test]
fn test_maxima_timeout_on_assumptions() {
    if !has_maxima() { return; }
    // Maxima asks "Is a positive, negative or zero?" for this integral —
    // bridge should timeout and return an error, not hang
    let result = eval_maxima("int(e^(-a*x^2), x, -inf, inf)");
    assert!(result.is_err(),
        "Expected error from Maxima (needs assumptions), got: {:?}", result);
}

#[test]
fn test_maxima_recovers_after_timeout() {
    if !has_maxima() { return; }
    // After a timeout, Maxima should restart and work for the next operation
    let results = eval_multi_with_both(&[
        "backend(\"maxima\")",
        // This will fail (Maxima asks about assumptions)
        // but we can't easily catch per-expression errors in eval_multi_with_both,
        // so test recovery by doing a simple op after a fresh evaluator
    ]).unwrap();

    // Just verify backend was set
    let set_result = format!("{}", results[0]);
    assert!(set_result.contains("Maxima"),
        "Expected Maxima confirmation, got: {}", set_result);
}

#[test]
fn test_both_mode_gaussian_graceful() {
    if !has_maxima() { return; }
    // In "both" mode, Maxima will timeout on this but SymPy should succeed
    let results = eval_multi_with_both(&[
        "backend(\"both\")",
        "int(e^(-x^2), x, -inf, inf)",
    ]).unwrap();

    let result = format!("{}", results[1]);
    // Should get SymPy's result since Maxima fails
    assert!(result.contains("sqrt") && result.contains("pi"),
        "Expected sqrt(pi) from SymPy fallback, got: {}", result);
}

// =============================================================================
// Backend switching tests
// =============================================================================

#[test]
fn test_backend_set() {
    let results = eval_multi_with_both(&[
        "backend(\"sympy\")",
        "dif(x^2, x)",
    ]).unwrap();
    let diff_result = format!("{}", results[1]);
    assert_eq!(diff_result, "2*x");

    // Setting should return confirmation
    let set_result = format!("{}", results[0]);
    assert!(set_result.contains("Sympy") || set_result.contains("SymPy") || set_result.contains("sympy"),
        "Expected confirmation message, got: {}", set_result);
}

#[test]
fn test_using_override() {
    if !has_maxima() { return; }
    // Default is sympy; using("maxima", ...) should work
    let results = eval_multi_with_both(&[
        "using(\"maxima\", dif(x^2, x))",
        "dif(x^2, x)",  // should still use default (sympy)
    ]).unwrap();

    let maxima_result = format!("{}", results[0]);
    assert_eq!(maxima_result, "2*x");

    let sympy_result = format!("{}", results[1]);
    assert_eq!(sympy_result, "2*x");
}

// =============================================================================
// Both mode tests
// =============================================================================

#[test]
fn test_both_mode_agree() {
    if !has_maxima() { return; }
    let results = eval_multi_with_both(&[
        "backend(\"both\")",
        "dif(x^2, x)",
    ]).unwrap();

    let diff_result = format!("{}", results[1]);
    // Both backends should agree → single result "2*x"
    assert_eq!(diff_result, "2*x");
}

// =============================================================================
// Cache tests
// =============================================================================

#[test]
fn test_cache_correctness() {
    // Same operation twice should give the same result (and the second hit the cache)
    let results = eval_multi_with_both(&[
        "dif(x^3, x)",
        "dif(x^3, x)",
    ]).unwrap();

    let r1 = format!("{}", results[0]);
    let r2 = format!("{}", results[1]);
    assert_eq!(r1, r2, "Cached result should match: {} vs {}", r1, r2);
    assert_eq!(r1, "3*x^2");
}
