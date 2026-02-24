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
cargo test      # 40 tests covering lexer, parser, evaluator
```

## Language features (currently implemented)
- Arithmetic: `+`, `-`, `*`, `/`, `^`, `%` with correct precedence
- Implicit multiplication: `3x` → `3 * x`, `2(x+1)` → `2 * (x + 1)`
- Variables: `a = 5`
- Functions: `f(x) = x^2 + 1`, `f(x, y) = x^2 + y^2`
- Builtins: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, ln, sqrt, abs, floor, ceil, round, sign, log, max, min, len, typeof, print
- Constants: pi, e, inf, tau
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

## Current phase
Phase 0-2 complete. Working TUI with numeric interpreter. Next: Phase 3 (symbolic expressions + CAS backend).

## Dependencies
ratatui 0.29, crossterm 0.28, serde/serde_json 1, tokio 1, thiserror 2, anyhow 1, lru 0.12, unicode-width 0.2
