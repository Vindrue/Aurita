use crate::tui::theme::Theme;
use ratatui::style::Style;

/// A single entry in the worksheet output.
#[derive(Debug, Clone)]
pub struct WorksheetEntry {
    pub index: usize,
    pub input: String,
    pub output: OutputKind,
}

#[derive(Debug, Clone)]
pub enum OutputKind {
    Value(String),
    Error(String),
    PrintOutput(Vec<String>),
}

impl WorksheetEntry {
    pub fn input_line(&self) -> String {
        format!(" In[{}]: {}", self.index, self.input)
    }

    pub fn output_lines(&self) -> Vec<(String, Style)> {
        match &self.output {
            OutputKind::Value(v) => {
                if v == "()" {
                    vec![] // Don't show Unit values
                } else {
                    vec![(format!("Out[{}]: {}", self.index, v), Theme::output_value())]
                }
            }
            OutputKind::Error(e) => {
                vec![(format!("Err[{}]: {}", self.index, e), Theme::error())]
            }
            OutputKind::PrintOutput(lines) => {
                let mut result = Vec::new();
                for line in lines {
                    result.push((format!("       {}", line), Theme::output_value()));
                }
                result
            }
        }
    }
}

/// State for the worksheet panel.
pub struct WorksheetState {
    pub entries: Vec<WorksheetEntry>,
    pub next_index: usize,
    pub scroll_offset: usize,
}

impl WorksheetState {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_index: 1,
            scroll_offset: 0,
        }
    }

    pub fn add_entry(&mut self, input: String, output: OutputKind, print_lines: Vec<String>) {
        let index = self.next_index;
        self.next_index += 1;

        // Add print output first if any
        if !print_lines.is_empty() {
            self.entries.push(WorksheetEntry {
                index,
                input: input.clone(),
                output: OutputKind::PrintOutput(print_lines),
            });
        }

        self.entries.push(WorksheetEntry {
            index,
            input,
            output,
        });
    }

    pub fn scroll_up(&mut self, amount: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(amount);
    }

    pub fn scroll_down(&mut self, amount: usize, max_visible: usize) {
        let total_lines = self.total_lines();
        if total_lines > max_visible {
            self.scroll_offset = (self.scroll_offset + amount).min(total_lines - max_visible);
        }
    }

    pub fn scroll_to_bottom(&mut self, max_visible: usize) {
        let total_lines = self.total_lines();
        if total_lines > max_visible {
            self.scroll_offset = total_lines - max_visible;
        }
    }

    fn total_lines(&self) -> usize {
        self.entries
            .iter()
            .map(|e| 1 + e.output_lines().len() + 1) // input + output + blank line
            .sum()
    }
}
