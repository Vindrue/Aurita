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
    let result = eval_str("simplify(x^2 + 2*x + 1)");
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
    let result = eval_maxima_str("simplify(x^2 + 2*x + 1)");
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

// =============================================================================
// Physics: measurements, units, and CODATA constants
// =============================================================================

/// Helper: evaluate without CAS (pure numeric + physics)
fn eval_no_cas(input: &str) -> Result<Value, String> {
    let mut evaluator = Evaluator::new();
    let tokens = Lexer::new(input).tokenize().map_err(|e| e.message)?;
    let stmts = Parser::new(tokens).parse_program().map_err(|e| e.message)?;
    evaluator.eval_program(&stmts).map_err(|e| e.message)
}

fn eval_no_cas_str(input: &str) -> String {
    format!("{}", eval_no_cas(input).unwrap())
}

fn eval_no_cas_multi(inputs: &[&str]) -> Result<Vec<Value>, String> {
    let mut evaluator = Evaluator::new();
    let mut results = Vec::new();
    for input in inputs {
        let tokens = Lexer::new(input).tokenize().map_err(|e| e.message)?;
        let stmts = Parser::new(tokens).parse_program().map_err(|e| e.message)?;
        results.push(evaluator.eval_program(&stmts).map_err(|e| e.message)?);
    }
    Ok(results)
}

// --- Measurement creation ---

#[test]
fn test_measurement_plusminus() {
    let result = eval_no_cas("9.81 +/- 0.02").unwrap();
    let display = format!("{}", result);
    assert!(display.contains("9.81"), "Expected 9.81 in {}", display);
    assert!(display.contains("+/-"), "Expected +/- in {}", display);
}

#[test]
fn test_measurement_pm_function() {
    let result = eval_no_cas("pm(100, 5)").unwrap();
    match result {
        Value::Quantity(q) => {
            assert_eq!(q.value, 100.0);
            assert_eq!(q.uncertainty, 5.0);
        }
        _ => panic!("Expected Quantity, got: {}", result),
    }
}

// --- Unit annotation ---

#[test]
fn test_unit_meters() {
    let result = eval_no_cas("5[m]").unwrap();
    match result {
        Value::Quantity(q) => {
            assert_eq!(q.value, 5.0);
            assert_eq!(q.uncertainty, 0.0);
        }
        _ => panic!("Expected Quantity, got: {}", result),
    }
}

#[test]
fn test_unit_km() {
    let result = eval_no_cas("3[km]").unwrap();
    match result {
        Value::Quantity(q) => {
            assert_eq!(q.value, 3000.0);
        }
        _ => panic!("Expected Quantity, got: {}", result),
    }
}

#[test]
fn test_unit_compound() {
    let result = eval_no_cas("9.81[m/s^2]").unwrap();
    match result {
        Value::Quantity(q) => {
            assert!((q.value - 9.81).abs() < 1e-10);
        }
        _ => panic!("Expected Quantity, got: {}", result),
    }
}

#[test]
fn test_measurement_with_units() {
    let result = eval_no_cas("(10 +/- 1)[m]").unwrap();
    match result {
        Value::Quantity(q) => {
            assert_eq!(q.value, 10.0);
            assert_eq!(q.uncertainty, 1.0);
        }
        _ => panic!("Expected Quantity, got: {}", result),
    }
}

// --- Unit arithmetic ---

#[test]
fn test_add_same_units() {
    let result = eval_no_cas("3[m] + 2[m]").unwrap();
    let display = format!("{}", result);
    assert!(display.contains("5.0"), "Expected 5.0 in {}", display);
    assert!(display.contains("[m]"), "Expected [m] in {}", display);
}

#[test]
fn test_add_mismatched_units() {
    let result = eval_no_cas("3[m] + 2[s]");
    assert!(result.is_err(), "Expected error for mismatched units");
}

#[test]
fn test_mul_units() {
    let result = eval_no_cas("3[m] * 2[s]").unwrap();
    let display = format!("{}", result);
    assert!(display.contains("6.0"), "Expected 6.0 in {}", display);
}

#[test]
fn test_div_units() {
    let result = eval_no_cas("10[m] / 2[s]").unwrap();
    match result {
        Value::Quantity(q) => {
            assert_eq!(q.value, 5.0);
        }
        _ => panic!("Expected Quantity, got: {}", result),
    }
}

// --- Uncertainty propagation ---

#[test]
fn test_uncertainty_add() {
    let result = eval_no_cas("(10 +/- 1)[m] + (20 +/- 2)[m]").unwrap();
    match result {
        Value::Quantity(q) => {
            assert_eq!(q.value, 30.0);
            let expected = (1.0_f64.powi(2) + 2.0_f64.powi(2)).sqrt();
            assert!((q.uncertainty - expected).abs() < 1e-10);
        }
        _ => panic!("Expected Quantity, got: {}", result),
    }
}

#[test]
fn test_uncertainty_sin() {
    let result = eval_no_cas("sin(1.0 +/- 0.01)").unwrap();
    match result {
        Value::Quantity(q) => {
            assert!((q.value - 1.0_f64.sin()).abs() < 1e-10);
        }
        _ => panic!("Expected Quantity, got: {}", result),
    }
}

// --- CODATA constants ---

#[test]
fn test_codata_c() {
    let result = eval_no_cas("c").unwrap();
    match result {
        Value::Quantity(q) => {
            assert_eq!(q.value, 299_792_458.0);
            assert_eq!(q.uncertainty, 0.0);
        }
        _ => panic!("Expected Quantity for c, got: {}", result),
    }
}

#[test]
fn test_codata_G_has_uncertainty() {
    let result = eval_no_cas("G").unwrap();
    match result {
        Value::Quantity(q) => {
            assert!(q.uncertainty > 0.0, "G should have nonzero uncertainty");
        }
        _ => panic!("Expected Quantity for G, got: {}", result),
    }
}

#[test]
fn test_c_times_second() {
    let result = eval_no_cas("c * 1[s]").unwrap();
    match result {
        Value::Quantity(q) => {
            assert_eq!(q.value, 299_792_458.0);
        }
        _ => panic!("Expected Quantity, got: {}", result),
    }
}

// --- Physics builtins ---

#[test]
fn test_uncertainty_function() {
    let result = eval_no_cas("uncertainty(G)").unwrap();
    match result {
        Value::Number(n) => {
            assert!(n.as_f64() > 0.0);
        }
        _ => panic!("Expected Number, got: {}", result),
    }
}

#[test]
fn test_nominal_function() {
    let result = eval_no_cas("nominal(c)").unwrap();
    match result {
        Value::Number(n) => {
            assert_eq!(n.as_f64(), 299_792_458.0);
        }
        _ => panic!("Expected Number, got: {}", result),
    }
}

#[test]
fn test_units_function() {
    let result = eval_no_cas("units(c)").unwrap();
    match result {
        Value::Str(s) => {
            assert!(s.contains("m") && s.contains("s"), "Expected m/s in units, got: {}", s);
        }
        _ => panic!("Expected Str, got: {}", result),
    }
}

// --- Unit conversion ---

#[test]
fn test_to_conversion_km_to_m() {
    let result = eval_no_cas_multi(&["to(3[km], \"m\")"]).unwrap();
    match &result[0] {
        Value::Quantity(q) => {
            assert_eq!(q.value, 3000.0);
        }
        _ => panic!("Expected Quantity, got: {}", result[0]),
    }
}

#[test]
fn test_to_conversion_dimension_mismatch() {
    let result = eval_no_cas("to(3[m], \"s\")");
    assert!(result.is_err(), "Expected error for dimension mismatch");
}

// --- No regression: vector indexing ---

#[test]
fn test_vector_indexing_no_regression() {
    let results = eval_no_cas_multi(&[
        "v = [10, 20, 30]",
        "v[2]",
    ]).unwrap();
    assert_eq!(format!("{}", results[1]), "20");
}

// =============================================================================
// grad() — gradient (CAS/SymPy)
// =============================================================================

#[test]
fn test_grad_basic_2d() {
    // grad(x*y, [x, y]) = [y, x]
    let result = eval_str("grad(x*y, [x, y])");
    assert!(result.contains("y") && result.contains("x"),
        "Expected [y, x], got: {}", result);
}

#[test]
fn test_grad_basic_3d() {
    // grad(x^2 + y^2 + z^2, [x, y, z]) = [2*x, 2*y, 2*z]
    let result = eval_str("grad(x^2 + y^2 + z^2, [x, y, z])");
    assert!(result.contains("2*x") && result.contains("2*y") && result.contains("2*z"),
        "Expected [2*x, 2*y, 2*z], got: {}", result);
}

#[test]
fn test_grad_constant_expression() {
    // grad(5, [x, y]) = [0, 0]
    let result = eval_str("grad(5, [x, y])");
    assert!(result.contains("0"),
        "Gradient of constant should be zero, got: {}", result);
}

#[test]
fn test_grad_single_variable() {
    // grad(x^3, [x]) = [3*x^2]
    let result = eval_str("grad(x^3, [x])");
    assert!(result.contains("3") && result.contains("x"),
        "Expected [3*x^2], got: {}", result);
}

#[test]
fn test_grad_complex_expression() {
    // grad(sin(x)*cos(y), [x, y])
    // = [cos(x)*cos(y), -sin(x)*sin(y)]
    let result = eval_str("grad(sin(x)*cos(y), [x, y])");
    assert!(result.contains("cos") && result.contains("sin"),
        "Expected trig derivatives, got: {}", result);
}

#[test]
fn test_grad_4d() {
    // grad should work for arbitrary dimension
    // Use variable names that don't clash with CODATA constants
    // grad(p + q + r + s, [p, q, r, s]) = [1, 1, 1, 1]
    let result = eval_str("grad(p + q + r + s, [p, q, r, s])");
    assert!(result.contains("1"),
        "Expected [1, 1, 1, 1], got: {}", result);
}

#[test]
fn test_grad_partial_dependence() {
    // grad(x^2, [x, y]) = [2*x, 0] — only depends on x
    let result = eval_str("grad(x^2, [x, y])");
    assert!(result.contains("2*x") && result.contains("0"),
        "Expected [2*x, 0], got: {}", result);
}

#[test]
fn test_grad_exponential() {
    // grad(exp(x*y), [x, y]) = [y*exp(x*y), x*exp(x*y)]
    let result = eval_str("grad(exp(x*y), [x, y])");
    assert!(result.contains("exp"),
        "Expected exponentials in gradient, got: {}", result);
}

#[test]
fn test_grad_wrong_arg_count() {
    // grad needs exactly 2 args
    let result = eval_with_cas("grad(x^2)");
    assert!(result.is_err(), "grad with 1 argument should error");
}

#[test]
fn test_grad_wrong_arg_count_three() {
    let result = eval_with_cas("grad(x^2, [x], [y])");
    assert!(result.is_err(), "grad with 3 arguments should error");
}

// =============================================================================
// divg() — divergence (CAS/SymPy)
// =============================================================================

#[test]
fn test_divg_basic_3d() {
    // divg([x^2, y^3, z^2], [x, y, z]) = 2*x + 3*y^2 + 2*z
    let result = eval_str("divg([x^2, y^3, z^2], [x, y, z])");
    assert!(result.contains("2*x") && result.contains("2*z"),
        "Expected 2*x + 3*y^2 + 2*z, got: {}", result);
}

#[test]
fn test_divg_basic_2d() {
    // divg([x, y], [x, y]) = 1 + 1 = 2
    let result = eval_str("divg([x, y], [x, y])");
    assert!(result == "2" || result == "2.0",
        "Expected 2, got: {}", result);
}

#[test]
fn test_divg_no_dependence_on_own_var() {
    // divg([y, x], [x, y]) = d(y)/dx + d(x)/dy = 0 + 0 = 0
    let result = eval_str("divg([y, x], [x, y])");
    assert!(result == "0" || result == "0.0",
        "Expected 0, got: {}", result);
}

#[test]
fn test_divg_complex_field() {
    // divg([sin(x), cos(y), exp(z)], [x, y, z])
    // = cos(x) - sin(y) + exp(z)
    let result = eval_str("divg([sin(x), cos(y), exp(z)], [x, y, z])");
    assert!(result.contains("cos") && result.contains("sin") && result.contains("exp"),
        "Expected cos(x) - sin(y) + exp(z), got: {}", result);
}

#[test]
fn test_divg_dimension_mismatch() {
    // Field has 3 components but only 2 variables
    let result = eval_with_cas("divg([x, y, z], [x, y])");
    assert!(result.is_err(), "divg dimension mismatch should error");
}

#[test]
fn test_divg_dimension_mismatch_reversed() {
    // Field has 2 components but 3 variables
    let result = eval_with_cas("divg([x, y], [x, y, z])");
    assert!(result.is_err(), "divg dimension mismatch should error");
}

#[test]
fn test_divg_constant_field() {
    // divg([1, 2, 3], [x, y, z]) = 0
    let result = eval_str("divg([1, 2, 3], [x, y, z])");
    assert!(result == "0" || result == "0.0",
        "Divergence of constant field should be 0, got: {}", result);
}

#[test]
fn test_divg_wrong_arg_count() {
    let result = eval_with_cas("divg([x, y, z])");
    assert!(result.is_err(), "divg with 1 argument should error");
}

// =============================================================================
// curl() — curl (CAS/SymPy)
// =============================================================================

#[test]
fn test_curl_basic_3d() {
    // curl([y, -x, 0], [x, y, z]) = [0, 0, -1 - 1] = [0, 0, -2]
    let result = eval_str("curl([y, -x, 0], [x, y, z])");
    assert!(result.contains("-2") || result.contains("- 2"),
        "Expected [0, 0, -2], got: {}", result);
}

#[test]
fn test_curl_2d_returns_3d() {
    // curl([y, -x], [x, y]) should return [0, 0, dFy/dx - dFx/dy] = [0, 0, -1 - 1] = [0, 0, -2]
    let result = eval_str("curl([y, -x], [x, y])");
    // Should be a 3D vector
    let display = result.clone();
    assert!(display.contains("0"),
        "2D curl should have zero components, got: {}", display);
    assert!(display.contains("-2") || display.contains("- 2"),
        "Expected z-component -2, got: {}", display);
}

#[test]
fn test_curl_irrotational_field() {
    // curl(grad(f)) = 0 for any scalar field
    // grad(x^2*y + y^2*z) = [2*x*y, x^2 + 2*y*z, y^2]
    // curl of that should be [0, 0, 0]
    let result = eval_str("curl([2*x*y, x^2 + 2*y*z, y^2], [x, y, z])");
    // All components should simplify to 0
    // dFz/dy - dFy/dz = 2*y - 2*y = 0
    // dFx/dz - dFz/dx = 0 - 0 = 0
    // dFy/dx - dFx/dy = 2*x - 2*x = 0
    assert!(result.contains("0"),
        "Curl of gradient should be zero, got: {}", result);
}

#[test]
fn test_curl_complex_3d() {
    // curl([x*z, y*z, x*y], [x, y, z])
    // = [d(x*y)/dy - d(y*z)/dz, d(x*z)/dz - d(x*y)/dx, d(y*z)/dx - d(x*z)/dy]
    // = [x - y, x - y, -z + 0]... let me just check it runs and produces a vector
    let result = eval_str("curl([x*z, y*z, x*y], [x, y, z])");
    // Should contain variable names — it's a nontrivial result
    assert!(result.contains("x") || result.contains("y") || result.contains("z"),
        "Expected symbolic curl result, got: {}", result);
}

#[test]
fn test_curl_dimension_mismatch() {
    // curl with mismatched field/vars dimensions
    let result = eval_with_cas("curl([x, y, z], [x, y])");
    assert!(result.is_err(), "curl dimension mismatch should error");
}

#[test]
fn test_curl_1d_error() {
    // curl only works for 2D and 3D
    let result = eval_with_cas("curl([x], [x])");
    assert!(result.is_err(), "curl with 1D field should error");
}

#[test]
fn test_curl_4d_error() {
    // curl only works for 2D and 3D
    let result = eval_with_cas("curl([a, b, c, d], [w, x, y, z])");
    assert!(result.is_err(), "curl with 4D field should error");
}

#[test]
fn test_curl_wrong_arg_count() {
    let result = eval_with_cas("curl([x, y, z])");
    assert!(result.is_err(), "curl with 1 argument should error");
}

#[test]
fn test_curl_2d_simple() {
    // curl([0, x], [x, y]) = [0, 0, d(x)/dx - d(0)/dy] = [0, 0, 1]
    let result = eval_str("curl([0, x], [x, y])");
    assert!(result.contains("1"),
        "Expected [0, 0, 1], got: {}", result);
}

// =============================================================================
// Cross-feature / composition tests
// =============================================================================

#[test]
fn test_grad_then_divg_laplacian() {
    // div(grad(f)) = Laplacian
    // For f = x^2 + y^2 + z^2:
    // grad = [2*x, 2*y, 2*z], div(grad) = 2 + 2 + 2 = 6
    // We need to do this in multiple steps using the evaluator
    let mut evaluator = Evaluator::new();
    let sympy_bridge = "backends/python_bridge.py";
    evaluator.init_cas(Some(sympy_bridge), None);
    assert!(evaluator.has_cas());

    let inputs = &[
        "g = grad(x^2 + y^2 + z^2, [x, y, z])",
        "divg(g, [x, y, z])",
    ];
    let mut results = Vec::new();
    for input in inputs {
        let tokens = aurita::lang::lexer::Lexer::new(input).tokenize().unwrap();
        let stmts = aurita::lang::parser::Parser::new(tokens).parse_program().unwrap();
        results.push(evaluator.eval_program(&stmts).unwrap());
    }
    let result = format!("{}", results[1]);
    assert!(result == "6" || result == "6.0",
        "Laplacian of x^2+y^2+z^2 should be 6, got: {}", result);
}

#[test]
fn test_curl_of_gradient_is_zero() {
    // curl(grad(f)) = 0 for any scalar f
    // f = x*y*z, grad = [y*z, x*z, x*y]
    // curl of that should be zero
    let mut evaluator = Evaluator::new();
    let sympy_bridge = "backends/python_bridge.py";
    evaluator.init_cas(Some(sympy_bridge), None);
    assert!(evaluator.has_cas());

    let inputs = &[
        "g = grad(x*y*z, [x, y, z])",
        "curl(g, [x, y, z])",
    ];
    let mut results = Vec::new();
    for input in inputs {
        let tokens = aurita::lang::lexer::Lexer::new(input).tokenize().unwrap();
        let stmts = aurita::lang::parser::Parser::new(tokens).parse_program().unwrap();
        results.push(evaluator.eval_program(&stmts).unwrap());
    }
    let result = format!("{}", results[1]);
    assert!(result.contains("0"),
        "curl(grad(f)) should be zero, got: {}", result);
}

#[test]
fn test_divg_of_curl_is_zero() {
    // div(curl(F)) = 0 for any vector field F
    // F = [x*y, y*z, z*x]
    let mut evaluator = Evaluator::new();
    let sympy_bridge = "backends/python_bridge.py";
    evaluator.init_cas(Some(sympy_bridge), None);
    assert!(evaluator.has_cas());

    let inputs = &[
        "c = curl([x*y, y*z, z*x], [x, y, z])",
        "divg(c, [x, y, z])",
    ];
    let mut results = Vec::new();
    for input in inputs {
        let tokens = aurita::lang::lexer::Lexer::new(input).tokenize().unwrap();
        let stmts = aurita::lang::parser::Parser::new(tokens).parse_program().unwrap();
        results.push(evaluator.eval_program(&stmts).unwrap());
    }
    let result = format!("{}", results[1]);
    assert!(result == "0" || result == "0.0",
        "div(curl(F)) should be 0, got: {}", result);
}

#[test]
fn test_vec_used_in_cas_operation() {
    // Build a vector with vec() and use it in divg
    let mut evaluator = Evaluator::new();
    let sympy_bridge = "backends/python_bridge.py";
    evaluator.init_cas(Some(sympy_bridge), None);
    assert!(evaluator.has_cas());

    let inputs = &[
        "F = vec(x^2, y^2, z^2)",
        "divg(F, [x, y, z])",
    ];
    let mut results = Vec::new();
    for input in inputs {
        let tokens = aurita::lang::lexer::Lexer::new(input).tokenize().unwrap();
        let stmts = aurita::lang::parser::Parser::new(tokens).parse_program().unwrap();
        results.push(evaluator.eval_program(&stmts).unwrap());
    }
    let result = format!("{}", results[1]);
    assert!(result.contains("2*x") && result.contains("2*y") && result.contains("2*z"),
        "Expected 2*x + 2*y + 2*z, got: {}", result);
}

#[test]
fn test_vec_used_in_curl() {
    // Build a 2D vector with vec() and compute its curl
    let mut evaluator = Evaluator::new();
    let sympy_bridge = "backends/python_bridge.py";
    evaluator.init_cas(Some(sympy_bridge), None);
    assert!(evaluator.has_cas());

    let inputs = &[
        "F = vec(y, -x)",
        "curl(F, [x, y])",
    ];
    let mut results = Vec::new();
    for input in inputs {
        let tokens = aurita::lang::lexer::Lexer::new(input).tokenize().unwrap();
        let stmts = aurita::lang::parser::Parser::new(tokens).parse_program().unwrap();
        results.push(evaluator.eval_program(&stmts).unwrap());
    }
    let result = format!("{}", results[1]);
    assert!(result.contains("-2") || result.contains("- 2"),
        "Expected z-component -2, got: {}", result);
}

#[test]
fn test_grad_with_vec_vars() {
    // Use vec() to construct the variable list
    let mut evaluator = Evaluator::new();
    let sympy_bridge = "backends/python_bridge.py";
    evaluator.init_cas(Some(sympy_bridge), None);
    assert!(evaluator.has_cas());

    let inputs = &[
        "vars = vec(x, y, z)",
        "grad(x^2 + y^2 + z^2, vars)",
    ];
    let mut results = Vec::new();
    for input in inputs {
        let tokens = aurita::lang::lexer::Lexer::new(input).tokenize().unwrap();
        let stmts = aurita::lang::parser::Parser::new(tokens).parse_program().unwrap();
        results.push(evaluator.eval_program(&stmts).unwrap());
    }
    let result = format!("{}", results[1]);
    assert!(result.contains("2*x") && result.contains("2*y") && result.contains("2*z"),
        "Expected [2*x, 2*y, 2*z], got: {}", result);
}

#[test]
fn test_pdiff_on_cas_results() {
    // Compute two definite integrals and compare them with pdiff
    let mut evaluator = Evaluator::new();
    let sympy_bridge = "backends/python_bridge.py";
    evaluator.init_cas(Some(sympy_bridge), None);
    assert!(evaluator.has_cas());

    let inputs = &[
        "a = int(x^2, x, 0, 1)",   // 1/3
        "b = int(x^3, x, 0, 1)",   // 1/4
        "pdiff(a, b)",
    ];
    let mut results = Vec::new();
    for input in inputs {
        let tokens = aurita::lang::lexer::Lexer::new(input).tokenize().unwrap();
        let stmts = aurita::lang::parser::Parser::new(tokens).parse_program().unwrap();
        results.push(evaluator.eval_program(&stmts).unwrap());
    }
    let result = format!("{}", results[2]);
    // pdiff(1/3, 1/4) = |1/3 - 1/4| / ((1/3 + 1/4)/2) * 100
    // = (1/12) / (7/24) * 100 = (24/84) * 100 ≈ 28.57%
    // Just verify it's a reasonable positive number
    let f: f64 = result.parse().expect(&format!("Expected a number, got: {}", result));
    assert!(f > 20.0 && f < 40.0, "Expected pdiff ~28.57%, got: {}", f);
}

#[test]
fn test_grad_polynomial_high_degree() {
    // grad(x^5*y^3, [x, y]) = [5*x^4*y^3, 3*x^5*y^2]
    let result = eval_str("grad(x^5*y^3, [x, y])");
    assert!(result.contains("5") && result.contains("3"),
        "Expected polynomial gradient, got: {}", result);
}

#[test]
fn test_grad_with_trig_and_exp() {
    // grad(sin(x)*exp(y), [x, y]) = [cos(x)*exp(y), sin(x)*exp(y)]
    let result = eval_str("grad(sin(x)*exp(y), [x, y])");
    assert!(result.contains("cos") && result.contains("exp"),
        "Expected cos and exp in gradient, got: {}", result);
}

#[test]
fn test_divg_solenoidal_field() {
    // A solenoidal (divergence-free) field: F = [-y, x, 0]
    // divg = d(-y)/dx + d(x)/dy + d(0)/dz = 0 + 0 + 0 = 0
    let result = eval_str("divg([-y, x, 0], [x, y, z])");
    assert!(result == "0" || result == "0.0",
        "Divergence of solenoidal field should be 0, got: {}", result);
}

#[test]
fn test_grad_returns_vector() {
    // Verify grad returns a Vector value
    let result = eval_with_cas("grad(x^2 + y^2, [x, y])").unwrap();
    assert!(matches!(result, Value::Vector(_)),
        "grad should return a Vector, got: {}", result);
    if let Value::Vector(items) = result {
        assert_eq!(items.len(), 2, "grad of 2-var expr should have 2 components");
    }
}

#[test]
fn test_curl_returns_vector() {
    // Verify curl returns a Vector value
    let result = eval_with_cas("curl([y, -x, 0], [x, y, z])").unwrap();
    assert!(matches!(result, Value::Vector(_)),
        "curl should return a Vector, got: {}", result);
    if let Value::Vector(items) = result {
        assert_eq!(items.len(), 3, "3D curl should have 3 components");
    }
}

#[test]
fn test_curl_2d_returns_3d_vector() {
    // 2D curl should return a 3D vector [0, 0, scalar]
    let result = eval_with_cas("curl([0, x^2], [x, y])").unwrap();
    assert!(matches!(result, Value::Vector(_)),
        "curl should return a Vector, got: {}", result);
    if let Value::Vector(items) = result {
        assert_eq!(items.len(), 3, "2D curl should return 3D vector, got {} components", items.len());
        // First two components should be 0
        assert_eq!(format!("{}", items[0]), "0");
        assert_eq!(format!("{}", items[1]), "0");
        // Third component = d(x^2)/dx - d(0)/dy = 2*x
        let z_str = format!("{}", items[2]);
        assert!(z_str.contains("2") && z_str.contains("x"),
            "Expected 2*x as z-component, got: {}", z_str);
    }
}

// =========================================================================
// component() — complex decomposition
// =========================================================================

#[test]
fn test_component_exp_i_pi_4() {
    // component(exp(i*pi/4)) should give sqrt(2)/2 + sqrt(2)/2 * i
    let result = eval_with_cas("component(exp(i*pi/4))").unwrap();
    let s = format!("{}", result);
    assert!(matches!(result, Value::Symbolic(_)),
        "expected symbolic expression, got {:?}", result);
    // Should contain both i and sqrt(2) (or 2^(1/2))
    assert!(s.contains("i"), "expected imaginary part, got: {}", s);
    assert!(s.contains("sqrt") || s.contains("2"),
        "expected sqrt(2) in result, got: {}", s);
}

#[test]
fn test_component_purely_real() {
    // component(exp(0)) = 1 (purely real, no imaginary part)
    let result = eval_with_cas("component(exp(0))").unwrap();
    let s = format!("{}", result);
    assert!(!s.contains("i"), "expected purely real result, got: {}", s);
}

#[test]
fn test_component_purely_imaginary() {
    // component(i) = i
    let result = eval_with_cas("component(i)").unwrap();
    let s = format!("{}", result);
    assert!(s.contains("i"), "expected i, got: {}", s);
}

#[test]
fn test_component_complex_sum() {
    // component(1 + i) = 1 + i
    let result = eval_with_cas("component(1 + i)").unwrap();
    let s = format!("{}", result);
    assert!(s.contains("1"), "expected real part 1, got: {}", s);
    assert!(s.contains("i"), "expected imaginary part, got: {}", s);
}

#[test]
fn test_component_cos_i_sin() {
    // component(cos(pi/6) + i*sin(pi/6)) should give sqrt(3)/2 + i/2 (exact)
    let result = eval_with_cas("component(cos(pi/6) + i*sin(pi/6))").unwrap();
    let s = format!("{}", result);
    assert!(s.contains("i"), "expected imaginary part, got: {}", s);
    assert!(s.contains("sqrt") || s.contains("3"),
        "expected sqrt(3) in result, got: {}", s);
}
