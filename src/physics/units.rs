use std::fmt;

/// SI base unit exponents (7 dimensions).
///
/// A physical unit is represented as a product of SI base units raised to integer powers:
///   m^a · kg^b · s^c · A^d · K^e · mol^f · cd^g
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitExpr {
    pub m: i8,
    pub kg: i8,
    pub s: i8,
    pub a: i8,
    pub k: i8,
    pub mol: i8,
    pub cd: i8,
}

impl UnitExpr {
    pub fn dimensionless() -> Self {
        Self { m: 0, kg: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 }
    }

    pub fn is_dimensionless(&self) -> bool {
        self.m == 0 && self.kg == 0 && self.s == 0
            && self.a == 0 && self.k == 0 && self.mol == 0 && self.cd == 0
    }

    pub fn mul(self, other: Self) -> Self {
        Self {
            m: self.m + other.m,
            kg: self.kg + other.kg,
            s: self.s + other.s,
            a: self.a + other.a,
            k: self.k + other.k,
            mol: self.mol + other.mol,
            cd: self.cd + other.cd,
        }
    }

    pub fn div(self, other: Self) -> Self {
        Self {
            m: self.m - other.m,
            kg: self.kg - other.kg,
            s: self.s - other.s,
            a: self.a - other.a,
            k: self.k - other.k,
            mol: self.mol - other.mol,
            cd: self.cd - other.cd,
        }
    }

    pub fn pow(self, n: i8) -> Self {
        Self {
            m: self.m * n,
            kg: self.kg * n,
            s: self.s * n,
            a: self.a * n,
            k: self.k * n,
            mol: self.mol * n,
            cd: self.cd * n,
        }
    }

    pub fn inv(self) -> Self {
        Self {
            m: -self.m,
            kg: -self.kg,
            s: -self.s,
            a: -self.a,
            k: -self.k,
            mol: -self.mol,
            cd: -self.cd,
        }
    }
}

impl fmt::Display for UnitExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dimensionless() {
            return Ok(());
        }

        let dims: &[(&str, i8)] = &[
            ("kg", self.kg), ("m", self.m), ("s", self.s),
            ("A", self.a), ("K", self.k), ("mol", self.mol), ("cd", self.cd),
        ];

        let mut numer = Vec::new();
        let mut denom = Vec::new();

        for &(name, exp) in dims {
            if exp > 0 {
                if exp == 1 {
                    numer.push(name.to_string());
                } else {
                    numer.push(format!("{}^{}", name, exp));
                }
            } else if exp < 0 {
                if exp == -1 {
                    denom.push(name.to_string());
                } else {
                    denom.push(format!("{}^{}", name, -exp));
                }
            }
        }

        if denom.is_empty() {
            write!(f, "{}", numer.join("*"))
        } else if numer.is_empty() {
            // All negative: show as 1/(...)
            if denom.len() == 1 {
                write!(f, "1/{}", denom[0])
            } else {
                write!(f, "1/({})", denom.join("*"))
            }
        } else {
            let n = numer.join("*");
            if denom.len() == 1 {
                write!(f, "{}/{}", n, denom[0])
            } else {
                write!(f, "{}/({})", n, denom.join("*"))
            }
        }
    }
}

// =============================================================================
// Unit registry
// =============================================================================

/// SI base units
const UNIT_M:   UnitExpr = UnitExpr { m: 1, kg: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 };
const UNIT_KG:  UnitExpr = UnitExpr { m: 0, kg: 1, s: 0, a: 0, k: 0, mol: 0, cd: 0 };
const UNIT_S:   UnitExpr = UnitExpr { m: 0, kg: 0, s: 1, a: 0, k: 0, mol: 0, cd: 0 };
const UNIT_A:   UnitExpr = UnitExpr { m: 0, kg: 0, s: 0, a: 1, k: 0, mol: 0, cd: 0 };
const UNIT_K:   UnitExpr = UnitExpr { m: 0, kg: 0, s: 0, a: 0, k: 1, mol: 0, cd: 0 };
const UNIT_MOL: UnitExpr = UnitExpr { m: 0, kg: 0, s: 0, a: 0, k: 0, mol: 1, cd: 0 };
const UNIT_CD:  UnitExpr = UnitExpr { m: 0, kg: 0, s: 0, a: 0, k: 0, mol: 0, cd: 1 };
const UNIT_DL:  UnitExpr = UnitExpr { m: 0, kg: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 };

/// Known named units: (name, multiplier_to_SI, dimensions)
static NAMED_UNITS: &[(&str, f64, UnitExpr)] = &[
    // SI base
    ("m",   1.0,      UNIT_M),
    ("kg",  1.0,      UNIT_KG),
    ("s",   1.0,      UNIT_S),
    ("A",   1.0,      UNIT_A),
    ("K",   1.0,      UNIT_K),
    ("mol", 1.0,      UNIT_MOL),
    ("cd",  1.0,      UNIT_CD),
    // Gram (so prefixed grams work: mg, ug, etc.)
    ("g",   1e-3,     UNIT_KG),
    // Derived SI
    ("N",   1.0,      UnitExpr { m: 1, kg: 1, s: -2, a: 0, k: 0, mol: 0, cd: 0 }),  // kg*m/s^2
    ("J",   1.0,      UnitExpr { m: 2, kg: 1, s: -2, a: 0, k: 0, mol: 0, cd: 0 }),  // kg*m^2/s^2
    ("W",   1.0,      UnitExpr { m: 2, kg: 1, s: -3, a: 0, k: 0, mol: 0, cd: 0 }),  // J/s
    ("Pa",  1.0,      UnitExpr { m: -1, kg: 1, s: -2, a: 0, k: 0, mol: 0, cd: 0 }), // N/m^2
    ("Hz",  1.0,      UnitExpr { m: 0, kg: 0, s: -1, a: 0, k: 0, mol: 0, cd: 0 }),  // 1/s
    ("C",   1.0,      UnitExpr { m: 0, kg: 0, s: 1, a: 1, k: 0, mol: 0, cd: 0 }),   // A*s
    ("V",   1.0,      UnitExpr { m: 2, kg: 1, s: -3, a: -1, k: 0, mol: 0, cd: 0 }), // W/A
    ("Ohm", 1.0,      UnitExpr { m: 2, kg: 1, s: -3, a: -2, k: 0, mol: 0, cd: 0 }), // V/A
    ("F",   1.0,      UnitExpr { m: -2, kg: -1, s: 4, a: 2, k: 0, mol: 0, cd: 0 }), // C/V
    ("H",   1.0,      UnitExpr { m: 2, kg: 1, s: -2, a: -2, k: 0, mol: 0, cd: 0 }), // V*s/A
    ("T",   1.0,      UnitExpr { m: 0, kg: 1, s: -2, a: -1, k: 0, mol: 0, cd: 0 }), // Wb/m^2
    ("Wb",  1.0,      UnitExpr { m: 2, kg: 1, s: -2, a: -1, k: 0, mol: 0, cd: 0 }), // V*s
    ("S",   1.0,      UnitExpr { m: -2, kg: -1, s: 3, a: 2, k: 0, mol: 0, cd: 0 }), // 1/Ohm
    // Practical
    ("eV",  1.602176634e-19, UnitExpr { m: 2, kg: 1, s: -2, a: 0, k: 0, mol: 0, cd: 0 }), // energy
    ("L",   1e-3,     UnitExpr { m: 3, kg: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 }),         // litre
    // Dimensionless
    ("rad", 1.0,      UNIT_DL),
    ("sr",  1.0,      UNIT_DL),
];

/// SI prefix table
static SI_PREFIXES: &[(&str, f64)] = &[
    ("T",  1e12),
    ("G",  1e9),
    ("M",  1e6),
    ("k",  1e3),
    ("h",  1e2),
    ("da", 1e1),
    ("d",  1e-1),
    ("c",  1e-2),
    ("m",  1e-3),
    ("u",  1e-6),
    ("\u{03BC}", 1e-6),  // μ
    ("n",  1e-9),
    ("p",  1e-12),
    ("f",  1e-15),
    ("a",  1e-18),
];

/// Look up a unit name, returning (multiplier_to_SI, dimensions).
/// Tries full name first, then prefix+remainder.
pub fn lookup_unit(name: &str) -> Option<(f64, UnitExpr)> {
    // Direct match
    for &(n, mult, dims) in NAMED_UNITS {
        if n == name {
            return Some((mult, dims));
        }
    }

    // Try SI prefix + base unit (but not for "kg" which is already listed,
    // and not for single-char names that might be a prefix themselves)
    if name.len() >= 2 {
        for &(prefix, pmult) in SI_PREFIXES {
            if let Some(remainder) = name.strip_prefix(prefix) {
                if !remainder.is_empty() {
                    // Don't allow prefix on "kg" (already has k- prefix built in)
                    if remainder == "kg" {
                        continue;
                    }
                    for &(n, umult, dims) in NAMED_UNITS {
                        if n == remainder {
                            return Some((pmult * umult, dims));
                        }
                    }
                }
            }
        }
    }

    None
}

// =============================================================================
// Mini unit expression parser
// =============================================================================

/// Parse a unit expression string like "m/s^2", "kg*m/s^2", "km", "MeV".
///
/// Grammar:
///   expr       = term (('*'|'·'|'/') term)*
///   term       = atom ('^' integer)?
///   atom       = identifier
///   integer    = ['-'] digits
pub fn parse_unit_expr(text: &str) -> Result<(f64, UnitExpr), String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("empty unit expression".to_string());
    }
    let mut parser = UnitParser::new(text);
    let result = parser.parse_expr()?;
    if parser.pos < parser.chars.len() {
        return Err(format!("unexpected character '{}' in unit expression", parser.chars[parser.pos]));
    }
    Ok(result)
}

struct UnitParser {
    chars: Vec<char>,
    pos: usize,
}

impl UnitParser {
    fn new(text: &str) -> Self {
        Self {
            chars: text.chars().collect(),
            pos: 0,
        }
    }

    fn parse_expr(&mut self) -> Result<(f64, UnitExpr), String> {
        let (mut mult, mut dims) = self.parse_term()?;

        while self.pos < self.chars.len() {
            self.skip_spaces();
            match self.peek() {
                Some('*') | Some('\u{00B7}') | Some('\u{22C5}') => {
                    self.advance();
                    let (m, d) = self.parse_term()?;
                    mult *= m;
                    dims = dims.mul(d);
                }
                Some('/') => {
                    self.advance();
                    let (m, d) = self.parse_term()?;
                    mult /= m;
                    dims = dims.div(d);
                }
                _ => break,
            }
        }

        Ok((mult, dims))
    }

    fn parse_term(&mut self) -> Result<(f64, UnitExpr), String> {
        self.skip_spaces();
        let (mult, dims) = self.parse_atom()?;
        self.skip_spaces();

        if self.peek() == Some('^') {
            self.advance();
            let exp = self.parse_integer()?;
            Ok((mult.powi(exp as i32), dims.pow(exp)))
        } else {
            Ok((mult, dims))
        }
    }

    fn parse_atom(&mut self) -> Result<(f64, UnitExpr), String> {
        self.skip_spaces();

        // Handle parenthesized sub-expression
        if self.peek() == Some('(') {
            self.advance();
            let result = self.parse_expr()?;
            if self.peek() != Some(')') {
                return Err("expected ')' in unit expression".to_string());
            }
            self.advance();
            return Ok(result);
        }

        // Read identifier
        let start = self.pos;
        while self.pos < self.chars.len() {
            let c = self.chars[self.pos];
            if c.is_alphanumeric() || c == '_' || c == '\u{03BC}' {
                self.pos += 1;
            } else {
                break;
            }
        }

        if self.pos == start {
            return Err("expected unit name".to_string());
        }

        let name: String = self.chars[start..self.pos].iter().collect();
        lookup_unit(&name)
            .ok_or_else(|| format!("unknown unit: '{}'", name))
    }

    fn parse_integer(&mut self) -> Result<i8, String> {
        self.skip_spaces();
        let mut neg = false;
        if self.peek() == Some('-') {
            neg = true;
            self.advance();
        }

        let start = self.pos;
        while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
            self.pos += 1;
        }

        if self.pos == start {
            return Err("expected integer exponent".to_string());
        }

        let digits: String = self.chars[start..self.pos].iter().collect();
        let n: i8 = digits.parse().map_err(|_| format!("invalid exponent: {}", digits))?;
        Ok(if neg { -n } else { n })
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn skip_spaces(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos] == ' ' {
            self.pos += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensionless() {
        let u = UnitExpr::dimensionless();
        assert!(u.is_dimensionless());
        assert_eq!(format!("{}", u), "");
    }

    #[test]
    fn test_display_simple() {
        let u = UNIT_M;
        assert_eq!(format!("{}", u), "m");
    }

    #[test]
    fn test_display_compound() {
        // kg*m/s^2 (Newton)
        let u = UnitExpr { m: 1, kg: 1, s: -2, a: 0, k: 0, mol: 0, cd: 0 };
        assert_eq!(format!("{}", u), "kg*m/s^2");
    }

    #[test]
    fn test_display_denominator_only() {
        // 1/s (Hertz)
        let u = UnitExpr { m: 0, kg: 0, s: -1, a: 0, k: 0, mol: 0, cd: 0 };
        assert_eq!(format!("{}", u), "1/s");
    }

    #[test]
    fn test_mul_div() {
        let m = UNIT_M;
        let s = UNIT_S;
        let ms = m.div(s);
        assert_eq!(ms.m, 1);
        assert_eq!(ms.s, -1);
    }

    #[test]
    fn test_pow() {
        let m = UNIT_M;
        let m2 = m.pow(2);
        assert_eq!(m2.m, 2);
    }

    #[test]
    fn test_lookup_base() {
        let (mult, dims) = lookup_unit("m").unwrap();
        assert_eq!(mult, 1.0);
        assert_eq!(dims, UNIT_M);
    }

    #[test]
    fn test_lookup_kg() {
        let (mult, dims) = lookup_unit("kg").unwrap();
        assert_eq!(mult, 1.0);
        assert_eq!(dims, UNIT_KG);
    }

    #[test]
    fn test_lookup_prefixed() {
        let (mult, dims) = lookup_unit("km").unwrap();
        assert_eq!(mult, 1e3);
        assert_eq!(dims, UNIT_M);
    }

    #[test]
    fn test_lookup_mg() {
        let (mult, dims) = lookup_unit("mg").unwrap();
        assert!((mult - 1e-6).abs() < 1e-20); // milli * gram = 1e-3 * 1e-3
        assert_eq!(dims, UNIT_KG);
    }

    #[test]
    fn test_lookup_mev() {
        let (mult, dims) = lookup_unit("MeV").unwrap();
        assert!((mult - 1e6 * 1.602176634e-19).abs() / mult < 1e-10);
        // Energy dimensions: kg*m^2/s^2
        assert_eq!(dims.m, 2);
        assert_eq!(dims.kg, 1);
        assert_eq!(dims.s, -2);
    }

    #[test]
    fn test_parse_simple() {
        let (mult, dims) = parse_unit_expr("m").unwrap();
        assert_eq!(mult, 1.0);
        assert_eq!(dims, UNIT_M);
    }

    #[test]
    fn test_parse_compound() {
        let (mult, dims) = parse_unit_expr("m/s^2").unwrap();
        assert_eq!(mult, 1.0);
        assert_eq!(dims.m, 1);
        assert_eq!(dims.s, -2);
    }

    #[test]
    fn test_parse_newton() {
        let (mult, dims) = parse_unit_expr("kg*m/s^2").unwrap();
        assert_eq!(mult, 1.0);
        assert_eq!(dims, UnitExpr { m: 1, kg: 1, s: -2, a: 0, k: 0, mol: 0, cd: 0 });
    }

    #[test]
    fn test_parse_prefixed() {
        let (mult, _) = parse_unit_expr("km").unwrap();
        assert_eq!(mult, 1e3);
    }

    #[test]
    fn test_parse_km_per_s() {
        let (mult, dims) = parse_unit_expr("km/s").unwrap();
        assert_eq!(mult, 1e3);
        assert_eq!(dims.m, 1);
        assert_eq!(dims.s, -1);
    }

    #[test]
    fn test_parse_unknown_unit() {
        assert!(parse_unit_expr("xyz").is_err());
    }
}
