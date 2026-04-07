# Aurita — Terminal-Based Computer Algebra System

## What is this?
A Rust TUI application with a custom interpreted math language. Successor to Jellyfish (Julia/SymPy.jl).

## Architecture
- **TUI**: ratatui + crossterm (Kitty terminal primary target)
- **Interpreter**: Hand-written lexer → Pratt parser → tree-walking evaluator
- **CAS backends** (Phase 3+): Multi-backend via JSON-RPC over subprocess (SymPy, Symbolics.jl, Maxima)
- **Plotting** (Phase 4+): plotters → PNG → ratatui-image (Kitty graphics protocol)

## Project structure
```
src/lang/       — Aurita language: token, lexer, parser, ast, eval, builtins, types, env, error
src/cas/        — CAS backend abstraction + implementations (stubs for now)
src/symbolic/   — Rust-native SymExpr type (stubs for now)
src/tui/        — TUI: app, event, input, output, sidebar, status, theme
src/plot/       — Plotting pipeline (stubs for now)
src/physics/    — Constants and error propagation (stubs for now)
backends/       — Bridge scripts for CAS subprocesses (not yet created)
```

## Build & run
```
cargo build
cargo run       # launches TUI
cargo test      # 322 tests covering lexer, parser, evaluator, CAS integration
```

## Language features (currently implemented)
- Arithmetic: `+`, `-`, `*`, `/`, `^`, `%` with correct precedence
- Variables: `a = 5`
- Functions: `f(x) = x^2 + 1`, `f(x, y) = x^2 + y^2`
- Builtins: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, ln, sqrt, abs, abs2, conj, floor, ceil, round, sign, log, max, min, len, typeof, print, pdiff, vec, eval, component
- CAS vector calculus: grad(expr, [vars]), divg([field], [vars]), curl([field], [vars])
- Complex decomposition: `component(expr)` → exact `a + b*i` form via CAS (SymPy re/im, Maxima realpart/imagpart)
- Vector decomposition: `component([a, b, c])` → `a*e_1 + b*e_2 + c*e_3` string
- Numeric eval: `eval(expr)` supports complex numbers (`eval(i^2)` → `-1`) and partial evaluation (`eval(sqrt(2)*x)` → `1.4142...*x`)
- Constants: pi, e, i, inf, tau
- Imaginary unit: `i` (i^2 = -1, simplifies for integer powers)
- Vectors: `[1, 2, 3]`, element-wise `+`/`-`, scalar `*`/`/`, 1-indexed
- Control flow: `if {} else {}`, `for i in 1..10 {}`, `while cond {}`, break, continue, return
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Boolean: `and`, `or`, `not`
- Comments: `#` and `//`
- Strings: `"hello"`
- Compound assignment: `+=`, `-=`, `*=`, `/=`

## Key design decisions
- **Integer preservation**: `10 / 2` → `5` (int), `10 / 3` → `3.333...` (float)
- **Right-associative exponentiation**: `2^3^4` = `2^(3^4)`
- **1-indexed vectors**: Math convention, `v[1]` is the first element
- **Auto-scroll**: Worksheet auto-scrolls to latest output
- **Emacs keybindings**: Ctrl-A/E (home/end), Ctrl-U (clear), Ctrl-W (kill word), Ctrl-K (kill line)
- **Unit display toggles**: `reduceunits(true/false)` shows derived SI names (Pa, N, J, ...) instead of base units; off by default
- **Percent difference**: `pdiff(a,b)` = 2*|a-b|/|a+b| * 100 (symmetric, vs average)
- **:clearall command**: resets worksheet, variables, functions, and CAS state
- **Undefined detection**: bare unknown identifiers and unknown function calls show errors at display level
- **Vector calculus**: grad/divg/curl follow Jellyfish.jl syntax, curl on 2D returns 3D vector

## Current phase
All phases (0-7) complete. Full TUI with numeric interpreter, symbolic CAS (SymPy + Maxima), inline plotting, physics units, and polish.

## Dependencies
ratatui 0.30, crossterm 0.28, ratatui-image 10, serde/serde_json 1, tokio 1, plotters 0.3, image 0.25, lru 0.12, unicode-width 0.2, thiserror 2, anyhow 1, base64 0.22, dirs 6, toml 0.8
