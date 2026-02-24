# Aurita Wiki

Aurita is a terminal-based computer algebra system built in Rust. It combines a custom math language with symbolic computation backends (SymPy, Maxima), inline plotting, physics units with error propagation, and a full TUI with sidebar, tab completion, and session management.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Arithmetic & Variables](#arithmetic--variables)
- [Functions](#functions)
- [Vectors](#vectors)
- [Control Flow](#control-flow)
- [Symbolic Math (CAS)](#symbolic-math-cas)
- [Plotting](#plotting)
- [Physics: Units & Measurements](#physics-units--measurements)
- [CODATA Constants](#codata-constants)
- [Commands](#commands)
- [Keybindings](#keybindings)
- [Configuration](#configuration)

---

## Getting Started

```
cargo build
cargo run
```

Type expressions at the `aurita>` prompt and press Enter. Results appear in the worksheet above. Press Ctrl-D to quit.

Aurita requires a Kitty-compatible terminal for inline plot display. SymPy (Python 3) must be installed for CAS features. Maxima is optional.

---

## Arithmetic & Variables

### Basic Arithmetic

```
2 + 3              # 5
10 / 3             # 3.3333...
10 / 2             # 5 (integer preserved)
2 ^ 10             # 1024
17 % 5             # 2
```

### Implicit Multiplication

```
3x                 # 3 * x
2(x + 1)           # 2 * (x + 1)
2pi                # 2 * pi
```

### Variables

```
x = 5
y = x ^ 2         # 25
x += 1             # 6 (also -=, *=, /=)
```

### Constants

| Name  | Value |
|-------|-------|
| `pi`  | 3.14159... |
| `e`   | 2.71828... |
| `tau` | 6.28318... (2pi) |
| `inf` | Infinity |

Constants are symbolic — they propagate through CAS operations correctly. Use `eval()` to get the numeric value:

```
eval(pi)           # 3.141592653589793
eval(2pi + 1)      # 7.283185307179586
```

---

## Functions

### Defining Functions

```
f(x) = x^2 + 1
f(3)               # 10

g(x, y) = x^2 + y^2
g(3, 4)            # 25
```

### Built-in Math Functions

| Function | Description |
|----------|-------------|
| `sin(x)`, `cos(x)`, `tan(x)` | Trigonometric |
| `asin(x)`, `acos(x)`, `atan(x)` | Inverse trig |
| `sinh(x)`, `cosh(x)`, `tanh(x)` | Hyperbolic |
| `exp(x)` | e^x |
| `ln(x)` | Natural logarithm |
| `log(x)` | Natural logarithm |
| `log(base, x)` | Logarithm with base |
| `sqrt(x)` | Square root |
| `abs(x)` | Absolute value |
| `floor(x)`, `ceil(x)`, `round(x)` | Rounding |
| `sign(x)` | Sign (-1, 0, or 1) |
| `max(a, b, ...)` | Maximum |
| `min(a, b, ...)` | Minimum |

### Utility Functions

| Function | Description |
|----------|-------------|
| `eval(expr)` | Force numeric evaluation of symbolic expression |
| `print(args...)` | Print values |
| `len(v)` | Length of vector or string |
| `typeof(x)` | Type name of a value |

---

## Vectors

Vectors are 1-indexed (math convention).

```
v = [1, 2, 3]
v[1]               # 1
v[3]               # 3
len(v)             # 3
```

Element-wise arithmetic:

```
[1, 2, 3] + [4, 5, 6]    # [5, 7, 9]
[1, 2, 3] * 2             # [2, 4, 6]
```

---

## Control Flow

### If/Else

```
x = -5
if x > 0 { x } else { -x }    # 5
```

### For Loops

```
for i in 1..5 { print(i) }    # prints 1 2 3 4
```

### While Loops

```
x = 10
while x > 0 { x = x - 1 }
```

### Keywords

`break`, `continue`, `return` work as expected inside loops and functions.

### Boolean Operators

```
true and false     # false
true or false      # true
not true           # false
```

### Comparison

```
==  !=  <  >  <=  >=
```

---

## Symbolic Math (CAS)

CAS operations require a SymPy or Maxima backend. Aurita auto-detects bridge scripts at startup.

### Differentiation

```
dif(x^3, x)                # 3*x^2
dif(sin(x), x)             # cos(x)
dif(x^3, x, 2)             # 6*x (2nd derivative)
```

### Integration

```
int(x^2, x)                # x^3/3
int(x^2, x, 0..1)          # 1/3 (definite)
int(x^2, x, 0, 1)          # 1/3 (alternative syntax)
```

### Solving Equations

```
solve(x^2 - 4, x)          # [-2, 2]
solve(x^2 + 1, x)          # [-I, I]
```

### Simplify / Expand / Factor

```
simplify(sin(x)^2 + cos(x)^2)    # 1
expand((x + 1)^3)                  # x^3 + 3*x^2 + 3*x + 1
factor(x^2 - 1)                    # (x - 1)*(x + 1)
```

### Limits

```
lim(sin(x)/x, x, 0)       # 1
lim(1/x, x, 0, "+")       # oo (from the right)
```

### Taylor Series

```
taylor(sin(x), x, 0, 5)   # x - x^3/6 + x^5/120
```

### LaTeX Output

```
tex(x^2 + 1)               # "x^{2} + 1"
```

### Backend Control

```
backend("sympy")           # use SymPy
backend("maxima")          # use Maxima
backend("both")            # send to both, compare results
using("maxima", dif(x^2, x))   # one-off with specific backend
```

---

## Plotting

```
plot(sin(x))                      # plot sin(x) for default range
plot(x^2, -5..5)                  # plot with explicit range
plot([sin(x), cos(x)], -pi..pi)  # multiple curves
```

Plots render inline using the Kitty graphics protocol. Requires a compatible terminal (Kitty, iTerm2). Uses matplotlib via the Python bridge for high-quality output with axes, grid, and legend.

---

## Physics: Units & Measurements

### Unit Annotation

Attach units to numbers with bracket syntax:

```
3[m]               # 3.0 [m]
9.81[m/s^2]        # 9.81 [m/s^2]
3[km]              # 3000.0 [m] (SI prefix applied)
```

### Unit Arithmetic

```
3[m] + 2[m]        # 5.0 [m]
3[m] * 2[s]        # 6.0 [m*s]
10[m] / 2[s]       # 5.0 [m/s]
3[m] + 2[s]        # ERROR: unit mismatch
```

### Unit Conversion

```
to(3[km], "m")     # 3000.0 [m]
to(1[eV], "J")     # 1.602e-19 [J]
```

### Supported Units

**Base SI:** m, kg, s, A, K, mol, cd

**Derived:** N, J, W, Pa, Hz, C, V, Ohm, F, H, T, Wb, S, eV, L, g

**SI Prefixes:** T (10^12), G (10^9), M (10^6), k (10^3), m (10^-3), u/&mu; (10^-6), n (10^-9), p (10^-12), f (10^-15)

### Measurements with Uncertainty

Create measurements with the `+/-` operator or `pm()` function:

```
9.81 +/- 0.02                    # 9.81 +/- 0.02
pm(100, 5)                       # 100.0 +/- 5.0
(10 +/- 1)[m]                    # 10.0 +/- 1.0 [m]
```

### Error Propagation

Uncertainty propagates automatically through arithmetic and functions using Gaussian error propagation:

```
a = (10 +/- 1)[m]
b = (20 +/- 2)[m]
a + b              # 30.0 +/- 2.236 [m]     (quadrature)
a * b              # 200.0 +/- 24.166 [m^2]  (relative)

sin(1.0 +/- 0.01) # 0.8415 +/- 0.0054       (derivative-based)
```

### Physics Utility Functions

| Function | Description |
|----------|-------------|
| `pm(value, uncertainty)` | Create a measurement |
| `uncertainty(x)` | Extract uncertainty |
| `nominal(x)` | Extract nominal value |
| `units(x)` | Get unit string |
| `to(quantity, "unit")` | Convert to different unit |

---

## CODATA Constants

All constants carry their CODATA uncertainties and SI units as `Quantity` values.

| Name | Description | Value |
|------|-------------|-------|
| `c` | Speed of light | 299792458 m/s |
| `h` | Planck constant | 6.626e-34 J*s |
| `hbar` | Reduced Planck | 1.055e-34 J*s |
| `G` | Gravitational constant | 6.674e-11 m^3/(kg*s^2) |
| `k_B` | Boltzmann constant | 1.381e-23 J/K |
| `N_A` | Avogadro number | 6.022e23 1/mol |
| `e_charge` | Elementary charge | 1.602e-19 C |
| `m_e` | Electron mass | 9.109e-31 kg |
| `m_p` | Proton mass | 1.673e-27 kg |
| `m_n` | Neutron mass | 1.675e-27 kg |
| `R` | Gas constant | 8.314 J/(mol*K) |
| `alpha` | Fine-structure constant | 7.297e-3 |
| `a_0` | Bohr radius | 5.292e-11 m |
| `mu_0` | Vacuum permeability | 1.257e-6 kg*m/A^2 |
| `eps_0` | Vacuum permittivity | 8.854e-12 s^2*A^2/(kg*m^3) |
| `sigma_SB` | Stefan-Boltzmann | 5.670e-8 W/(m^2*K^4) |
| `eV` | Electron volt | 1.602e-19 J |
| `mu_B` | Bohr magneton | 9.274e-24 J/T |
| `R_inf` | Rydberg constant | 1.097e7 1/m |

...and ~35 more (Planck units, particle masses, nuclear magneton, Faraday constant, etc.). All are accessible by name at the prompt.

```
c                  # 299792458.0 [m/s]
uncertainty(G)     # 1.5e-15
c * 1[s]           # 299792458.0 [m]
```

---

## Commands

Commands start with `:` and are distinct from math expressions.

| Command | Description |
|---------|-------------|
| `:save [name]` | Save current session (default: timestamped name) |
| `:load <name>` | Load and replay a saved session |
| `:sessions` | List all saved sessions |
| `:clear` | Clear the worksheet |
| `:help` | Toggle the help panel |

Sessions are stored as JSON in `~/.local/share/aurita/sessions/`. Loading a session replays all inputs to rebuild the full environment state.

---

## Keybindings

### Input

| Key | Action |
|-----|--------|
| Enter | Evaluate input |
| Tab | Tab completion |
| Up / Down | History navigation |
| Ctrl-A | Move to start of line |
| Ctrl-E | Move to end of line |
| Ctrl-K | Kill to end of line |
| Ctrl-U | Clear line |
| Ctrl-W | Kill word back |

### App

| Key | Action |
|-----|--------|
| Ctrl-H / F1 | Toggle help panel |
| Ctrl-L | Clear worksheet |
| Ctrl-D | Quit |
| PageUp / PageDown | Scroll worksheet |

### Tab Completion

| Key | Action |
|-----|--------|
| Tab / Down | Next candidate |
| Shift-Tab / Up | Previous candidate |
| Enter | Accept selection |
| Esc | Dismiss popup |

### Help Panel

| Key | Action |
|-----|--------|
| j / Down | Scroll down |
| k / Up | Scroll up |
| PageUp / PageDown | Scroll by page |
| Esc / Ctrl-H / F1 | Close |

---

## Configuration

Config file: `~/.config/aurita/config.toml` (created automatically on first run)

```toml
# Maximum number of history entries to keep
history_limit = 1000

# Default CAS backend: "sympy", "maxima", or "both"
default_backend = "sympy"

# Height in terminal rows for plot images
plot_height = 20
```

### Data Files

| Path | Contents |
|------|----------|
| `~/.local/share/aurita/history` | Command history (one per line) |
| `~/.local/share/aurita/sessions/` | Saved session JSON files |
| `~/.config/aurita/config.toml` | Configuration |
