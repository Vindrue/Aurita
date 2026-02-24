use crate::plot::types::RenderedPlot;
use crate::tui::theme::Theme;
use ratatui::style::Style;
use ratatui_image::protocol::StatefulProtocol;
use std::cell::RefCell;

/// Height in terminal rows allocated for a plot image.
pub const PLOT_ROWS: u16 = 20;

/// A single entry in the worksheet output.
pub struct WorksheetEntry {
    pub index: usize,
    pub input: String,
    pub output: OutputKind,
    /// Cached image protocol state for plot entries (avoids re-encoding per frame).
    pub image_state: RefCell<Option<StatefulProtocol>>,
}

impl Clone for WorksheetEntry {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            input: self.input.clone(),
            output: self.output.clone(),
            image_state: RefCell::new(None),
        }
    }
}

impl std::fmt::Debug for WorksheetEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorksheetEntry")
            .field("index", &self.index)
            .field("input", &self.input)
            .field("output", &self.output)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub enum OutputKind {
    Value(String),
    Error(String),
    PrintOutput(Vec<String>),
    Plot(RenderedPlot),
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
            OutputKind::Plot(p) => {
                let labels: Vec<&str> = p.spec.series.iter().map(|s| s.label.as_str()).collect();
                vec![(format!("Out[{}]: [plot: {}]", self.index, labels.join(", ")), Theme::output_value())]
            }
        }
    }

    /// Number of text lines this entry occupies (for scrolling calculations).
    /// Plot entries reserve PLOT_ROWS for the image.
    pub fn line_count(&self) -> usize {
        match &self.output {
            OutputKind::Plot(_) => 1 + PLOT_ROWS as usize + 1, // input + image + blank
            _ => 1 + self.output_lines().len() + 1, // input + output + blank
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
                image_state: RefCell::new(None),
            });
        }

        self.entries.push(WorksheetEntry {
            index,
            input,
            output,
            image_state: RefCell::new(None),
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
        self.entries.iter().map(|e| e.line_count()).sum()
    }
}
