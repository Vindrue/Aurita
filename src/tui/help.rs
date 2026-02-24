/// Help panel overlay with scrollable documentation sections.

pub struct HelpPanel {
    pub visible: bool,
    pub scroll: usize,
}

impl HelpPanel {
    pub fn new() -> Self {
        Self {
            visible: false,
            scroll: 0,
        }
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
        if self.visible {
            self.scroll = 0;
        }
    }

    pub fn scroll_up(&mut self, n: usize) {
        self.scroll = self.scroll.saturating_sub(n);
    }

    pub fn scroll_down(&mut self, n: usize) {
        self.scroll += n;
    }

    pub fn total_lines(&self) -> usize {
        HELP_SECTIONS.iter().map(|(_, content)| {
            // title line + blank + content lines + blank
            2 + content.lines().count() + 1
        }).sum()
    }
}

/// Help content: (section_title, content_text)
pub static HELP_SECTIONS: &[(&str, &str)] = &[
    ("Quick Start", "\
Aurita is a terminal-based computer algebra system.
Type expressions at the prompt and press Enter to evaluate.

  2 + 3           arithmetic
  x = 5           assign a variable
  f(x) = x^2      define a function
  f(3)            call a function  =>  9
  3x + 1          implicit multiplication"),

    ("Operators", "\
  +   addition           -   subtraction
  *   multiplication     /   division
  ^   power              %   modulo
  +/- uncertainty        ..  range (in for/int)

Unit annotation:  3[m]   9.81[m/s^2]
Comparison:  ==  !=  <  >  <=  >=
Boolean:     and  or  not
Assignment:  =  +=  -=  *=  /="),

    ("Math Functions", "\
  sin(x)  cos(x)  tan(x)     trigonometric
  asin(x) acos(x) atan(x)    inverse trig
  sinh(x) cosh(x) tanh(x)    hyperbolic
  exp(x)  ln(x)   log(b,x)   exponential/log
  sqrt(x) abs(x)  sign(x)    root/absolute
  floor(x) ceil(x) round(x)  rounding
  max(a,b,...) min(a,b,...)   extrema
  eval(expr)                  force numeric eval"),

    ("CAS Operations", "\
  dif(expr, var)              differentiate
  dif(expr, var, n)           nth derivative
  int(expr, var)              indefinite integral
  int(expr, var, a..b)        definite integral
  solve(expr, var)            solve equation
  simplify(expr)              simplify
  expand(expr)                expand
  factor(expr)                factor
  lim(expr, var, point)       limit
  taylor(expr, var, pt, n)    Taylor series
  tex(expr)                   LaTeX output"),

    ("Plotting", "\
  plot(expr)                  plot y=expr for x
  plot(expr, -5..5)           plot with range
  plot([e1, e2], -pi..pi)     multiple curves"),

    ("Physics & Units", "\
  3[m] + 2[m]                unit arithmetic
  3[km]                      SI prefix => 3000[m]
  9.81 +/- 0.02              measurement
  (10 +/- 1)[m]              measurement with units
  to(3[km], \"m\")             unit conversion

  pm(val, unc)               create measurement
  uncertainty(x)             get uncertainty
  nominal(x)                 get nominal value
  units(x)                   get unit string

  c, h, hbar, G, k_B, ...   CODATA constants"),

    ("Control Flow", "\
  if x > 0 { x } else { -x }
  for i in 1..10 { print(i) }
  while x > 0 { x = x - 1 }
  break   continue   return"),

    ("Keybindings", "\
  Enter        evaluate input
  Tab          tab completion
  Up/Down      history navigation
  Ctrl-A       move to start of line
  Ctrl-E       move to end of line
  Ctrl-K       kill to end of line
  Ctrl-U       clear line
  Ctrl-W       kill word back
  Ctrl-L       clear worksheet
  Ctrl-H / F1  toggle help panel
  Ctrl-D       quit
  PageUp/Down  scroll worksheet
  Esc          dismiss popup"),

    ("Commands", "\
  :save [name]      save session
  :load [name]      load session
  :sessions         list sessions
  :clear            clear worksheet
  :help             toggle help panel
  backend(\"sympy\")  set CAS backend
  using(\"b\", expr)  eval with backend"),
];
