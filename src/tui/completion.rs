use crate::lang::env::EnvRef;
use crate::lang::types::{Function, Value};
use crate::physics::constants;
use crate::tui::hints::FUNCTION_HINTS;

pub struct CompletionState {
    pub active: bool,
    pub candidates: Vec<CompletionItem>,
    pub selected: usize,
    pub prefix: String,
    pub prefix_start: usize,
}

#[derive(Clone)]
pub struct CompletionItem {
    pub text: String,
    pub kind: CompletionKind,
    pub description: String,
}

#[derive(Clone, Copy, PartialEq)]
pub enum CompletionKind {
    Function,
    Variable,
    Constant,
    Keyword,
}

impl CompletionKind {
    pub fn label(self) -> &'static str {
        match self {
            CompletionKind::Function => "fn",
            CompletionKind::Variable => "var",
            CompletionKind::Constant => "const",
            CompletionKind::Keyword => "kw",
        }
    }
}

const KEYWORDS: &[&str] = &[
    "if", "else", "for", "while", "in", "break", "continue", "return",
    "and", "or", "not", "true", "false",
];

impl CompletionState {
    pub fn new() -> Self {
        Self {
            active: false,
            candidates: Vec::new(),
            selected: 0,
            prefix: String::new(),
            prefix_start: 0,
        }
    }

    /// Compute completion candidates given current input text and cursor position.
    pub fn compute(&mut self, text: &str, cursor: usize, env: &EnvRef) {
        let (prefix, start) = extract_prefix(text, cursor);
        if prefix.is_empty() {
            self.dismiss();
            return;
        }

        self.prefix = prefix.to_string();
        self.prefix_start = start;

        let prefix_lower = prefix.to_ascii_lowercase();
        let mut candidates: Vec<CompletionItem> = Vec::new();

        // 1. Function hints (builtins + CAS ops)
        for hint in FUNCTION_HINTS.iter() {
            if hint.name.to_ascii_lowercase().starts_with(&prefix_lower) {
                candidates.push(CompletionItem {
                    text: hint.name.to_string(),
                    kind: CompletionKind::Function,
                    description: hint.description.to_string(),
                });
            }
        }

        // 2. Environment bindings (user variables + functions)
        let bindings = env.borrow().all_bindings();
        for (name, value) in &bindings {
            if !name.to_ascii_lowercase().starts_with(&prefix_lower) {
                continue;
            }
            // Skip if already added from hints
            if candidates.iter().any(|c| c.text == *name) {
                continue;
            }
            match value {
                Value::Function(Function::UserDefined { params, .. }) => {
                    candidates.push(CompletionItem {
                        text: name.clone(),
                        kind: CompletionKind::Function,
                        description: format!("({})", params.join(", ")),
                    });
                }
                Value::Function(Function::Builtin { .. }) => {
                    // Already covered by hints
                }
                _ => {
                    candidates.push(CompletionItem {
                        text: name.clone(),
                        kind: CompletionKind::Variable,
                        description: String::new(),
                    });
                }
            }
        }

        // 3. CODATA constant names (not already in env bindings above)
        for name in constants::all_names() {
            if !name.to_ascii_lowercase().starts_with(&prefix_lower) {
                continue;
            }
            if candidates.iter().any(|c| c.text == name) {
                continue;
            }
            let desc = constants::lookup(name)
                .map(|pc| pc.description.to_string())
                .unwrap_or_default();
            candidates.push(CompletionItem {
                text: name.to_string(),
                kind: CompletionKind::Constant,
                description: desc,
            });
        }

        // 4. Keywords
        for &kw in KEYWORDS {
            if kw.starts_with(&prefix_lower) {
                candidates.push(CompletionItem {
                    text: kw.to_string(),
                    kind: CompletionKind::Keyword,
                    description: String::new(),
                });
            }
        }

        // Sort: exact match first, then alphabetical
        candidates.sort_by(|a, b| {
            let a_exact = a.text == prefix;
            let b_exact = b.text == prefix;
            b_exact.cmp(&a_exact).then_with(|| a.text.cmp(&b.text))
        });

        if candidates.is_empty() {
            self.dismiss();
        } else {
            self.candidates = candidates;
            self.selected = 0;
            self.active = true;
        }
    }

    pub fn select_next(&mut self) {
        if !self.candidates.is_empty() {
            self.selected = (self.selected + 1) % self.candidates.len();
        }
    }

    pub fn select_prev(&mut self) {
        if !self.candidates.is_empty() {
            self.selected = if self.selected == 0 {
                self.candidates.len() - 1
            } else {
                self.selected - 1
            };
        }
    }

    /// Accept the current selection. Returns (prefix_start, prefix_end, replacement_text).
    pub fn accept(&mut self) -> Option<(usize, usize, String)> {
        if !self.active || self.candidates.is_empty() {
            return None;
        }
        let item = &self.candidates[self.selected];
        let result = (
            self.prefix_start,
            self.prefix_start + self.prefix.len(),
            item.text.clone(),
        );
        self.dismiss();
        Some(result)
    }

    pub fn dismiss(&mut self) {
        self.active = false;
        self.candidates.clear();
        self.selected = 0;
    }

    /// Maximum number of visible items in the popup.
    pub fn max_visible(&self) -> usize {
        8
    }

    /// Scroll offset so the selected item is visible.
    pub fn scroll_offset(&self) -> usize {
        let max_vis = self.max_visible();
        if self.selected >= max_vis {
            self.selected - max_vis + 1
        } else {
            0
        }
    }
}

/// Extract the identifier prefix at the cursor position.
/// Returns (prefix_str, byte_offset_of_prefix_start).
fn extract_prefix(text: &str, cursor: usize) -> (&str, usize) {
    let bytes = text.as_bytes();
    let pos = cursor.min(bytes.len());
    let mut start = pos;
    while start > 0 && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_') {
        start -= 1;
    }
    if start == pos {
        return ("", pos);
    }
    (&text[start..pos], start)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_prefix_simple() {
        let (prefix, start) = extract_prefix("sin", 3);
        assert_eq!(prefix, "sin");
        assert_eq!(start, 0);
    }

    #[test]
    fn test_extract_prefix_mid_expr() {
        let (prefix, start) = extract_prefix("x + si", 6);
        assert_eq!(prefix, "si");
        assert_eq!(start, 4);
    }

    #[test]
    fn test_extract_prefix_empty() {
        let (prefix, _) = extract_prefix("x + ", 4);
        assert_eq!(prefix, "");
    }

    #[test]
    fn test_extract_prefix_after_paren() {
        let (prefix, start) = extract_prefix("dif(co", 6);
        assert_eq!(prefix, "co");
        assert_eq!(start, 4);
    }

    #[test]
    fn test_completion_basic() {
        use crate::lang::env::Env;
        let env = Env::new_global();
        let mut state = CompletionState::new();
        state.compute("si", 2, &env);
        assert!(state.active);
        let names: Vec<&str> = state.candidates.iter().map(|c| c.text.as_str()).collect();
        assert!(names.contains(&"sin"));
        assert!(names.contains(&"sinh"));
        assert!(names.contains(&"sign"));
        assert!(names.contains(&"simplify"));
    }

    #[test]
    fn test_completion_no_match() {
        use crate::lang::env::Env;
        let env = Env::new_global();
        let mut state = CompletionState::new();
        state.compute("zzz", 3, &env);
        assert!(!state.active);
    }
}
