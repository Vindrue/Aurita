use crate::physics::units::UnitExpr;

/// Result of a propagated operation.
pub struct PropResult {
    pub value: f64,
    pub uncertainty: f64,
    pub unit: UnitExpr,
}

/// Add two quantities (units must match — caller checks).
pub fn prop_add(a: f64, da: f64, b: f64, db: f64, unit: UnitExpr) -> PropResult {
    PropResult {
        value: a + b,
        uncertainty: (da * da + db * db).sqrt(),
        unit,
    }
}

/// Subtract two quantities (units must match — caller checks).
pub fn prop_sub(a: f64, da: f64, b: f64, db: f64, unit: UnitExpr) -> PropResult {
    PropResult {
        value: a - b,
        uncertainty: (da * da + db * db).sqrt(),
        unit,
    }
}

/// Multiply two quantities.
pub fn prop_mul(
    a: f64, da: f64, ua: UnitExpr,
    b: f64, db: f64, ub: UnitExpr,
) -> PropResult {
    let value = a * b;
    let rel_a = if a != 0.0 { da / a.abs() } else { 0.0 };
    let rel_b = if b != 0.0 { db / b.abs() } else { 0.0 };
    let uncertainty = value.abs() * (rel_a * rel_a + rel_b * rel_b).sqrt();
    PropResult {
        value,
        uncertainty,
        unit: ua.mul(ub),
    }
}

/// Divide two quantities.
pub fn prop_div(
    a: f64, da: f64, ua: UnitExpr,
    b: f64, db: f64, ub: UnitExpr,
) -> PropResult {
    let value = a / b;
    let rel_a = if a != 0.0 { da / a.abs() } else { 0.0 };
    let rel_b = if b != 0.0 { db / b.abs() } else { 0.0 };
    let uncertainty = value.abs() * (rel_a * rel_a + rel_b * rel_b).sqrt();
    PropResult {
        value,
        uncertainty,
        unit: ua.div(ub),
    }
}

/// Raise a quantity to an exact integer power.
pub fn prop_pow_int(a: f64, da: f64, ua: UnitExpr, n: i8) -> PropResult {
    let value = a.powi(n as i32);
    let uncertainty = if n != 0 && a != 0.0 {
        ((n as f64) * a.powi(n as i32 - 1)).abs() * da
    } else {
        0.0
    };
    PropResult {
        value,
        uncertainty,
        unit: ua.pow(n),
    }
}

/// Raise a quantity to a non-integer power (units must be dimensionless for non-integer exponents).
pub fn prop_pow_float(a: f64, da: f64, n: f64) -> PropResult {
    let value = a.powf(n);
    let uncertainty = if a != 0.0 {
        (n * a.powf(n - 1.0)).abs() * da
    } else {
        0.0
    };
    PropResult {
        value,
        uncertainty,
        unit: UnitExpr::dimensionless(),
    }
}

/// Negate a quantity.
pub fn prop_neg(a: f64, da: f64, ua: UnitExpr) -> PropResult {
    PropResult {
        value: -a,
        uncertainty: da,
        unit: ua,
    }
}

/// Propagate uncertainty through a single-argument function.
///
/// Given f(a) with uncertainty da, the propagated uncertainty is |f'(a)| * da.
/// The input must be dimensionless for trig/transcendental functions.
pub fn prop_func(name: &str, a: f64, da: f64) -> Result<PropResult, String> {
    let (value, deriv) = match name {
        "sin"   => (a.sin(),   a.cos()),
        "cos"   => (a.cos(),  -a.sin()),
        "tan"   => {
            let c = a.cos();
            (a.tan(), 1.0 / (c * c))
        }
        "asin"  => (a.asin(),  1.0 / (1.0 - a * a).sqrt()),
        "acos"  => (a.acos(), -1.0 / (1.0 - a * a).sqrt()),
        "atan"  => (a.atan(),  1.0 / (1.0 + a * a)),
        "sinh"  => (a.sinh(),  a.cosh()),
        "cosh"  => (a.cosh(),  a.sinh()),
        "tanh"  => {
            let c = a.cosh();
            (a.tanh(), 1.0 / (c * c))
        }
        "exp"   => (a.exp(),   a.exp()),
        "ln"    => (a.ln(),    1.0 / a),
        "sqrt"  => (a.sqrt(),  0.5 / a.sqrt()),
        "abs"   => (a.abs(),   a.signum()),
        _ => return Err(format!("cannot propagate uncertainty through '{}'", name)),
    };

    Ok(PropResult {
        value,
        uncertainty: deriv.abs() * da,
        unit: UnitExpr::dimensionless(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_uncertainty() {
        let r = prop_add(10.0, 1.0, 20.0, 2.0, UnitExpr::dimensionless());
        assert_eq!(r.value, 30.0);
        assert!((r.uncertainty - (1.0_f64.powi(2) + 2.0_f64.powi(2)).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_mul_uncertainty() {
        let u = UnitExpr::dimensionless();
        let r = prop_mul(10.0, 1.0, u, 20.0, 2.0, u);
        assert_eq!(r.value, 200.0);
        // relative: sqrt((1/10)^2 + (2/20)^2) = sqrt(0.01 + 0.01) = sqrt(0.02)
        let expected = 200.0 * (0.01_f64 + 0.01).sqrt();
        assert!((r.uncertainty - expected).abs() < 1e-10);
    }

    #[test]
    fn test_pow_uncertainty() {
        let u = UnitExpr::dimensionless();
        let r = prop_pow_int(3.0, 0.1, u, 2);
        assert_eq!(r.value, 9.0);
        // d/dx(x^2) = 2x, so uncertainty = |2*3| * 0.1 = 0.6
        assert!((r.uncertainty - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_sin_uncertainty() {
        let r = prop_func("sin", 0.0, 0.01).unwrap();
        assert!((r.value - 0.0).abs() < 1e-10);
        // d/dx sin(x) at x=0 = cos(0) = 1, so uncertainty = 0.01
        assert!((r.uncertainty - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_exp_uncertainty() {
        let r = prop_func("exp", 1.0, 0.1).unwrap();
        let e = std::f64::consts::E;
        assert!((r.value - e).abs() < 1e-10);
        // d/dx exp(x) at x=1 = e, so uncertainty = e * 0.1
        assert!((r.uncertainty - e * 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_unit_propagation_mul() {
        let m = UnitExpr { m: 1, kg: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 };
        let s = UnitExpr { m: 0, kg: 0, s: 1, a: 0, k: 0, mol: 0, cd: 0 };
        let r = prop_mul(3.0, 0.0, m, 2.0, 0.0, s);
        assert_eq!(r.unit.m, 1);
        assert_eq!(r.unit.s, 1);
    }

    #[test]
    fn test_unit_propagation_div() {
        let m = UnitExpr { m: 1, kg: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 };
        let s = UnitExpr { m: 0, kg: 0, s: 1, a: 0, k: 0, mol: 0, cd: 0 };
        let r = prop_div(6.0, 0.0, m, 2.0, 0.0, s);
        assert_eq!(r.value, 3.0);
        assert_eq!(r.unit.m, 1);
        assert_eq!(r.unit.s, -1);
    }
}
