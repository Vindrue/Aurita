use crate::physics::units::UnitExpr;

/// A CODATA physical constant.
pub struct PhysConst {
    pub name: &'static str,
    pub value: f64,
    pub uncertainty: f64,
    pub unit: UnitExpr,
    pub unit_display: &'static str,
    pub description: &'static str,
}

// SI dimension helpers (const)
#[allow(dead_code)]
const DL:  UnitExpr = UnitExpr { m: 0, kg: 0, s: 0,  a: 0,  k: 0, mol: 0, cd: 0 };
const M:   UnitExpr = UnitExpr { m: 1, kg: 0, s: 0,  a: 0,  k: 0, mol: 0, cd: 0 };
const KG:  UnitExpr = UnitExpr { m: 0, kg: 1, s: 0,  a: 0,  k: 0, mol: 0, cd: 0 };
const S:   UnitExpr = UnitExpr { m: 0, kg: 0, s: 1,  a: 0,  k: 0, mol: 0, cd: 0 };
const MS1: UnitExpr = UnitExpr { m: 1, kg: 0, s: -1, a: 0,  k: 0, mol: 0, cd: 0 }; // m/s
const MS2: UnitExpr = UnitExpr { m: 1, kg: 0, s: -2, a: 0,  k: 0, mol: 0, cd: 0 }; // m/s^2
const J:   UnitExpr = UnitExpr { m: 2, kg: 1, s: -2, a: 0,  k: 0, mol: 0, cd: 0 }; // J
const JS:  UnitExpr = UnitExpr { m: 2, kg: 1, s: -1, a: 0,  k: 0, mol: 0, cd: 0 }; // J*s
const JK:  UnitExpr = UnitExpr { m: 2, kg: 1, s: -2, a: 0,  k: -1, mol: 0, cd: 0 }; // J/K
const PA:  UnitExpr = UnitExpr { m: -1, kg: 1, s: -2, a: 0, k: 0, mol: 0, cd: 0 }; // Pa
#[allow(dead_code)]
const V:   UnitExpr = UnitExpr { m: 2, kg: 1, s: -3, a: -1, k: 0, mol: 0, cd: 0 }; // V
#[allow(dead_code)]
const T:   UnitExpr = UnitExpr { m: 0, kg: 1, s: -2, a: -1, k: 0, mol: 0, cd: 0 }; // T
const C:   UnitExpr = UnitExpr { m: 0, kg: 0, s: 1,  a: 1,  k: 0, mol: 0, cd: 0 }; // C

pub static CODATA: &[PhysConst] = &[
    // =========================================================================
    // Universal / defining constants (exact in 2019 SI)
    // =========================================================================
    PhysConst {
        name: "c", value: 299_792_458.0, uncertainty: 0.0,
        unit: MS1, unit_display: "m/s",
        description: "Speed of light in vacuum",
    },
    PhysConst {
        name: "h", value: 6.626_070_15e-34, uncertainty: 0.0,
        unit: JS, unit_display: "J*s",
        description: "Planck constant",
    },
    PhysConst {
        name: "hbar", value: 1.054_571_817e-34, uncertainty: 0.0,
        unit: JS, unit_display: "J*s",
        description: "Reduced Planck constant",
    },
    PhysConst {
        name: "k_B", value: 1.380_649e-23, uncertainty: 0.0,
        unit: JK, unit_display: "J/K",
        description: "Boltzmann constant",
    },
    PhysConst {
        name: "N_A", value: 6.022_140_76e23, uncertainty: 0.0,
        unit: UnitExpr { m: 0, kg: 0, s: 0, a: 0, k: 0, mol: -1, cd: 0 },
        unit_display: "1/mol",
        description: "Avogadro constant",
    },
    PhysConst {
        name: "e_charge", value: 1.602_176_634e-19, uncertainty: 0.0,
        unit: C, unit_display: "C",
        description: "Elementary charge",
    },

    // =========================================================================
    // Electromagnetic
    // =========================================================================
    PhysConst {
        name: "mu_0", value: 1.256_637_062_12e-6, uncertainty: 1.9e-16,
        unit: UnitExpr { m: 1, kg: 1, s: -2, a: -2, k: 0, mol: 0, cd: 0 },
        unit_display: "N/A^2",
        description: "Vacuum permeability",
    },
    PhysConst {
        name: "eps_0", value: 8.854_187_8128e-12, uncertainty: 1.3e-21,
        unit: UnitExpr { m: -3, kg: -1, s: 4, a: 2, k: 0, mol: 0, cd: 0 },
        unit_display: "F/m",
        description: "Vacuum permittivity",
    },
    PhysConst {
        name: "mu_B", value: 9.274_010_0783e-24, uncertainty: 2.8e-33,
        unit: UnitExpr { m: 2, kg: 0, s: 0, a: 1, k: 0, mol: 0, cd: 0 },
        unit_display: "J/T",
        description: "Bohr magneton",
    },
    PhysConst {
        name: "mu_N", value: 5.050_783_7461e-27, uncertainty: 1.5e-36,
        unit: UnitExpr { m: 2, kg: 0, s: 0, a: 1, k: 0, mol: 0, cd: 0 },
        unit_display: "J/T",
        description: "Nuclear magneton",
    },
    PhysConst {
        name: "Phi_0", value: 2.067_833_848e-15, uncertainty: 0.0,
        unit: UnitExpr { m: 2, kg: 1, s: -2, a: -1, k: 0, mol: 0, cd: 0 },
        unit_display: "Wb",
        description: "Magnetic flux quantum",
    },
    PhysConst {
        name: "G_0", value: 7.748_091_729e-5, uncertainty: 0.0,
        unit: UnitExpr { m: -2, kg: -1, s: 3, a: 2, k: 0, mol: 0, cd: 0 },
        unit_display: "S",
        description: "Conductance quantum",
    },
    PhysConst {
        name: "K_J", value: 483_597.8484e9, uncertainty: 0.0,
        unit: UnitExpr { m: -2, kg: -1, s: 2, a: 1, k: 0, mol: 0, cd: 0 },
        unit_display: "Hz/V",
        description: "Josephson constant",
    },
    PhysConst {
        name: "R_K", value: 25_812.807_45, uncertainty: 0.0,
        unit: UnitExpr { m: 2, kg: 1, s: -3, a: -2, k: 0, mol: 0, cd: 0 },
        unit_display: "Ohm",
        description: "von Klitzing constant",
    },
    PhysConst {
        name: "Z_0", value: 376.730_313_668, uncertainty: 5.7e-8,
        unit: UnitExpr { m: 2, kg: 1, s: -3, a: -2, k: 0, mol: 0, cd: 0 },
        unit_display: "Ohm",
        description: "Impedance of free space",
    },

    // =========================================================================
    // Atomic / nuclear
    // =========================================================================
    PhysConst {
        name: "m_e", value: 9.109_383_7015e-31, uncertainty: 2.8e-40,
        unit: KG, unit_display: "kg",
        description: "Electron mass",
    },
    PhysConst {
        name: "m_p", value: 1.672_621_923_69e-27, uncertainty: 5.1e-37,
        unit: KG, unit_display: "kg",
        description: "Proton mass",
    },
    PhysConst {
        name: "m_n", value: 1.674_927_498_04e-27, uncertainty: 9.5e-37,
        unit: KG, unit_display: "kg",
        description: "Neutron mass",
    },
    PhysConst {
        name: "m_u", value: 1.660_539_066_60e-27, uncertainty: 5.0e-37,
        unit: KG, unit_display: "kg",
        description: "Atomic mass unit",
    },
    PhysConst {
        name: "a_0", value: 5.291_772_109_03e-11, uncertainty: 8.0e-21,
        unit: M, unit_display: "m",
        description: "Bohr radius",
    },
    PhysConst {
        name: "r_e", value: 2.817_940_3262e-15, uncertainty: 1.3e-24,
        unit: M, unit_display: "m",
        description: "Classical electron radius",
    },
    PhysConst {
        name: "R_inf", value: 10_973_731.568_160, uncertainty: 2.1e-5,
        unit: UnitExpr { m: -1, kg: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 },
        unit_display: "1/m",
        description: "Rydberg constant",
    },
    PhysConst {
        name: "E_h", value: 4.359_744_722_2071e-18, uncertainty: 8.5e-30,
        unit: J, unit_display: "J",
        description: "Hartree energy",
    },
    PhysConst {
        name: "lambda_C", value: 2.426_310_238_67e-12, uncertainty: 7.3e-22,
        unit: M, unit_display: "m",
        description: "Compton wavelength",
    },
    PhysConst {
        name: "sigma_T", value: 6.652_458_7321e-29, uncertainty: 6.0e-38,
        unit: UnitExpr { m: 2, kg: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 },
        unit_display: "m^2",
        description: "Thomson cross section",
    },

    // =========================================================================
    // Measured constants
    // =========================================================================
    PhysConst {
        name: "G", value: 6.674_30e-11, uncertainty: 1.5e-15,
        unit: UnitExpr { m: 3, kg: -1, s: -2, a: 0, k: 0, mol: 0, cd: 0 },
        unit_display: "m^3/(kg*s^2)",
        description: "Newtonian gravitational constant",
    },
    PhysConst {
        name: "alpha", value: 7.297_352_5693e-3, uncertainty: 1.1e-12,
        unit: DL, unit_display: "",
        description: "Fine-structure constant",
    },
    PhysConst {
        name: "g_e", value: -2.002_319_304_362_56, uncertainty: 3.5e-13,
        unit: DL, unit_display: "",
        description: "Electron g-factor",
    },
    PhysConst {
        name: "g_p", value: 5.585_694_6893, uncertainty: 1.6e-9,
        unit: DL, unit_display: "",
        description: "Proton g-factor",
    },

    // =========================================================================
    // Thermodynamic
    // =========================================================================
    PhysConst {
        name: "R", value: 8.314_462_618, uncertainty: 0.0,
        unit: UnitExpr { m: 2, kg: 1, s: -2, a: 0, k: -1, mol: -1, cd: 0 },
        unit_display: "J/(mol*K)",
        description: "Molar gas constant",
    },
    PhysConst {
        name: "sigma_SB", value: 5.670_374_419e-8, uncertainty: 0.0,
        unit: UnitExpr { m: 0, kg: 1, s: -3, a: 0, k: -4, mol: 0, cd: 0 },
        unit_display: "W/(m^2*K^4)",
        description: "Stefan-Boltzmann constant",
    },
    PhysConst {
        name: "b_Wien", value: 2.897_771_955e-3, uncertainty: 0.0,
        unit: UnitExpr { m: 1, kg: 0, s: 0, a: 0, k: 1, mol: 0, cd: 0 },
        unit_display: "m*K",
        description: "Wien displacement law constant",
    },

    // =========================================================================
    // Defined / conventional
    // =========================================================================
    PhysConst {
        name: "g_n", value: 9.806_65, uncertainty: 0.0,
        unit: MS2, unit_display: "m/s^2",
        description: "Standard gravity",
    },
    PhysConst {
        name: "atm", value: 101_325.0, uncertainty: 0.0,
        unit: PA, unit_display: "Pa",
        description: "Standard atmosphere",
    },
    PhysConst {
        name: "eV_J", value: 1.602_176_634e-19, uncertainty: 0.0,
        unit: J, unit_display: "J",
        description: "Electron volt (in joules)",
    },

    // =========================================================================
    // Additional useful constants
    // =========================================================================
    PhysConst {
        name: "F_const", value: 96_485.332_12, uncertainty: 0.0,
        unit: UnitExpr { m: 0, kg: 0, s: 1, a: 1, k: 0, mol: -1, cd: 0 },
        unit_display: "C/mol",
        description: "Faraday constant",
    },
    PhysConst {
        name: "m_tau", value: 3.167_54e-27, uncertainty: 4.4e-31,
        unit: KG, unit_display: "kg",
        description: "Tau mass",
    },
    PhysConst {
        name: "m_mu", value: 1.883_531_627e-28, uncertainty: 4.2e-36,
        unit: KG, unit_display: "kg",
        description: "Muon mass",
    },
    PhysConst {
        name: "m_W", value: 1.432_15e-25, uncertainty: 2.0e-28,
        unit: KG, unit_display: "kg",
        description: "W boson mass",
    },
    PhysConst {
        name: "m_Z", value: 1.625_39e-25, uncertainty: 3.3e-29,
        unit: KG, unit_display: "kg",
        description: "Z boson mass",
    },
    PhysConst {
        name: "m_H", value: 2.228_74e-25, uncertainty: 2.2e-28,
        unit: KG, unit_display: "kg",
        description: "Higgs boson mass",
    },
    PhysConst {
        name: "G_F", value: 1.166_3787e-5, uncertainty: 6.0e-12,
        unit: UnitExpr { m: 0, kg: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 }, // GeV^-2 (dimensionless in natural units)
        unit_display: "GeV^-2",
        description: "Fermi coupling constant",
    },
    PhysConst {
        name: "sin2_thetaW", value: 0.231_16, uncertainty: 1.3e-4,
        unit: DL, unit_display: "",
        description: "Weak mixing angle (sin^2)",
    },
    PhysConst {
        name: "m_Pl", value: 2.176_434e-8, uncertainty: 2.4e-13,
        unit: KG, unit_display: "kg",
        description: "Planck mass",
    },
    PhysConst {
        name: "l_Pl", value: 1.616_255e-35, uncertainty: 1.8e-40,
        unit: M, unit_display: "m",
        description: "Planck length",
    },
    PhysConst {
        name: "t_Pl", value: 5.391_247e-44, uncertainty: 6.0e-49,
        unit: S, unit_display: "s",
        description: "Planck time",
    },
    PhysConst {
        name: "T_Pl", value: 1.416_784e32, uncertainty: 1.6e27,
        unit: UnitExpr { m: 0, kg: 0, s: 0, a: 0, k: 1, mol: 0, cd: 0 },
        unit_display: "K",
        description: "Planck temperature",
    },
    PhysConst {
        name: "lambda_C_p", value: 1.321_409_855_39e-15, uncertainty: 4.0e-25,
        unit: M, unit_display: "m",
        description: "Proton Compton wavelength",
    },
    PhysConst {
        name: "m_d", value: 3.343_583_7724e-27, uncertainty: 1.0e-36,
        unit: KG, unit_display: "kg",
        description: "Deuteron mass",
    },
    PhysConst {
        name: "m_alpha", value: 6.644_657_3357e-27, uncertainty: 2.0e-36,
        unit: KG, unit_display: "kg",
        description: "Alpha particle mass",
    },
    PhysConst {
        name: "sigma_e", value: 6.652_458_7321e-29, uncertainty: 6.0e-38,
        unit: UnitExpr { m: 2, kg: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 },
        unit_display: "m^2",
        description: "Electron cross section",
    },
];

/// Look up a CODATA constant by name.
pub fn lookup(name: &str) -> Option<&'static PhysConst> {
    CODATA.iter().find(|c| c.name == name)
}

/// All physics constant names (for sidebar filtering).
pub fn all_names() -> Vec<&'static str> {
    CODATA.iter().map(|c| c.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_c() {
        let c = lookup("c").unwrap();
        assert_eq!(c.value, 299_792_458.0);
        assert_eq!(c.uncertainty, 0.0);
        assert_eq!(c.unit, UnitExpr { m: 1, kg: 0, s: -1, a: 0, k: 0, mol: 0, cd: 0 });
    }

    #[test]
    fn test_lookup_G() {
        let g = lookup("G").unwrap();
        assert!(g.uncertainty > 0.0);
        assert_eq!(g.unit.m, 3);
        assert_eq!(g.unit.kg, -1);
        assert_eq!(g.unit.s, -2);
    }

    #[test]
    fn test_lookup_nonexistent() {
        assert!(lookup("nonexistent").is_none());
    }

    #[test]
    fn test_all_names() {
        let names = all_names();
        assert!(names.contains(&"c"));
        assert!(names.contains(&"h"));
        assert!(names.contains(&"G"));
        assert!(names.len() >= 40);
    }
}
