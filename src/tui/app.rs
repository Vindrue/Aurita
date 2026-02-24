use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
use ratatui::Frame;

use crate::lang::builtins::{numeric_eval_sym, BUILTIN_CONSTANTS};
use crate::lang::eval::Evaluator;
use crate::lang::lexer::Lexer;
use crate::lang::parser::Parser;
use crate::lang::types::{Function, Value};
use crate::tui::hints::detect_active_function;
use crate::tui::input::InputState;
use crate::tui::output::{OutputKind, WorksheetState};
use crate::tui::status::render_status_bar;
use crate::tui::theme::Theme;

pub struct App {
    pub input: InputState,
    pub worksheet: WorksheetState,
    pub evaluator: Evaluator,
    pub should_quit: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            input: InputState::new(),
            worksheet: WorksheetState::new(),
            evaluator: Evaluator::new(),
            should_quit: false,
        }
    }

    /// Handle a key event. Returns true if the screen should be redrawn.
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        match key {
            // Quit
            KeyEvent {
                code: KeyCode::Char('d'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.should_quit = true;
                true
            }

            // Submit input
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } => {
                let text = self.input.submit();
                if !text.trim().is_empty() {
                    self.execute(&text);
                }
                true
            }

            // History navigation
            KeyEvent {
                code: KeyCode::Up, ..
            } => {
                self.input.history_up();
                true
            }
            KeyEvent {
                code: KeyCode::Down,
                ..
            } => {
                self.input.history_down();
                true
            }

            // Cursor movement
            KeyEvent {
                code: KeyCode::Left,
                ..
            } => {
                self.input.move_left();
                true
            }
            KeyEvent {
                code: KeyCode::Right,
                ..
            } => {
                self.input.move_right();
                true
            }
            KeyEvent {
                code: KeyCode::Home,
                ..
            } => {
                self.input.move_home();
                true
            }
            KeyEvent {
                code: KeyCode::End,
                ..
            } => {
                self.input.move_end();
                true
            }

            // Editing
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => {
                self.input.backspace();
                true
            }
            KeyEvent {
                code: KeyCode::Delete,
                ..
            } => {
                self.input.delete();
                true
            }
            KeyEvent {
                code: KeyCode::Char('u'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.input.clear();
                true
            }
            KeyEvent {
                code: KeyCode::Char('k'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.input.kill_line();
                true
            }
            KeyEvent {
                code: KeyCode::Char('w'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.input.kill_word_back();
                true
            }
            KeyEvent {
                code: KeyCode::Char('a'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.input.move_home();
                true
            }
            KeyEvent {
                code: KeyCode::Char('e'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.input.move_end();
                true
            }

            // Scroll worksheet
            KeyEvent {
                code: KeyCode::PageUp,
                ..
            } => {
                self.worksheet.scroll_up(10);
                true
            }
            KeyEvent {
                code: KeyCode::PageDown,
                ..
            } => {
                self.worksheet.scroll_down(10, 50);
                true
            }

            // Clear screen
            KeyEvent {
                code: KeyCode::Char('l'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.worksheet = WorksheetState::new();
                true
            }

            // Regular character input
            KeyEvent {
                code: KeyCode::Char(c),
                modifiers: KeyModifiers::NONE | KeyModifiers::SHIFT,
                ..
            } => {
                self.input.insert(c);
                true
            }

            _ => false,
        }
    }

    /// Execute a line of input.
    fn execute(&mut self, input: &str) {
        self.evaluator.output.clear();

        let result = Lexer::new(input)
            .tokenize()
            .and_then(|tokens| Parser::new(tokens).parse_program())
            .and_then(|stmts| self.evaluator.eval_program(&stmts));

        let print_lines = self.evaluator.output.clone();

        match result {
            Ok(value) => {
                self.worksheet.add_entry(
                    input.to_string(),
                    OutputKind::Value(format!("{}", value)),
                    print_lines,
                );
            }
            Err(err) => {
                self.worksheet.add_entry(
                    input.to_string(),
                    OutputKind::Error(err.message),
                    print_lines,
                );
            }
        }

        // Auto-scroll to bottom
        self.worksheet.scroll_to_bottom(50);
    }

    /// Render the full UI.
    pub fn render(&self, frame: &mut Frame) {
        let outer = Layout::vertical([
            Constraint::Length(1),  // Status bar
            Constraint::Min(5),    // Main area
            Constraint::Length(3), // Input bar
        ])
        .split(frame.area());

        // Status bar
        render_status_bar(frame, outer[0], self.evaluator.has_cas());

        // Main area: worksheet + sidebar
        let main = Layout::horizontal([
            Constraint::Percentage(70),
            Constraint::Percentage(30),
        ])
        .split(outer[1]);

        self.render_worksheet(frame, main[0]);
        self.render_sidebar(frame, main[1]);
        self.render_input(frame, outer[2]);
    }

    fn render_worksheet(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Theme::border_focused())
            .title(" Worksheet ");

        let inner = block.inner(area);
        frame.render_widget(block, area);

        // Build all lines
        let mut lines: Vec<Line> = Vec::new();
        for entry in &self.worksheet.entries {
            // Input line
            lines.push(Line::from(vec![
                Span::styled(
                    format!(" In[{}]: ", entry.index),
                    Theme::input_prompt(),
                ),
                Span::styled(&entry.input, Theme::input_text()),
            ]));

            // Output lines
            for (text, style) in entry.output_lines() {
                lines.push(Line::from(Span::styled(text, style)));
            }

            // Blank separator
            lines.push(Line::from(""));
        }

        // Handle scrolling
        let visible_height = inner.height as usize;
        let total = lines.len();
        let offset = if total > visible_height {
            total.saturating_sub(visible_height) // auto-scroll to bottom
        } else {
            0
        };

        let visible_lines: Vec<Line> = lines.into_iter().skip(offset).take(visible_height).collect();
        let paragraph = Paragraph::new(visible_lines);
        frame.render_widget(paragraph, inner);
    }

    fn render_sidebar(&self, frame: &mut Frame, area: Rect) {
        let has_hint = detect_active_function(&self.input.text, self.input.cursor).is_some();
        let hint_height = if has_hint { 4u16 } else { 0u16 };

        let sidebar = Layout::vertical([
            Constraint::Length(6),                          // Constants
            Constraint::Min(3),                            // Variables
            Constraint::Min(3),                            // Functions
            Constraint::Length(hint_height),                // Hint (conditional)
        ])
        .split(area);

        self.render_constants(frame, sidebar[0]);
        self.render_variables(frame, sidebar[1]);
        self.render_functions(frame, sidebar[2]);
        if has_hint {
            self.render_hint(frame, sidebar[3]);
        }
    }

    fn render_constants(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Theme::border())
            .title(Span::styled(" Constants ", Theme::sidebar_title()));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        let env = self.evaluator.env.borrow();
        let mut items: Vec<ListItem> = Vec::new();
        for &name in BUILTIN_CONSTANTS {
            if let Some(val) = env.get(name) {
                let approx = match &val {
                    Value::Symbolic(expr) => {
                        if name == "inf" {
                            "\u{221e}".to_string()
                        } else if let Some(f) = numeric_eval_sym(expr) {
                            format!("\u{2248} {:.5}", f)
                        } else {
                            format!("{}", val)
                        }
                    }
                    _ => format!("{}", val),
                };
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(
                        name.to_string(),
                        Style::default().fg(ratatui::style::Color::Cyan),
                    ),
                    Span::raw(" "),
                    Span::styled(approx, Theme::sidebar_item()),
                ])));
            }
        }

        let list = List::new(items);
        frame.render_widget(list, inner);
    }

    fn render_variables(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Theme::border())
            .title(Span::styled(" Variables ", Theme::sidebar_title()));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        let bindings = self.evaluator.env.borrow().all_bindings();
        let items: Vec<ListItem> = bindings
            .iter()
            .filter(|(name, v)| {
                !matches!(v, Value::Function(_))
                    && !BUILTIN_CONSTANTS.contains(&name.as_str())
            })
            .take(inner.height as usize)
            .map(|(name, value)| {
                let display = format!("{}", value);
                let truncated = if display.len() > (inner.width as usize).saturating_sub(name.len() + 4) {
                    format!("{}...", &display[..display.len().min(20)])
                } else {
                    display
                };
                ListItem::new(Line::from(vec![
                    Span::styled(name.to_string(), Style::default().fg(ratatui::style::Color::Cyan)),
                    Span::raw(" = "),
                    Span::styled(truncated, Theme::sidebar_item()),
                ]))
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, inner);
    }

    fn render_functions(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Theme::border())
            .title(Span::styled(" Functions ", Theme::sidebar_title()));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        let bindings = self.evaluator.env.borrow().all_bindings();
        let items: Vec<ListItem> = bindings
            .iter()
            .filter_map(|(name, v)| match v {
                Value::Function(Function::UserDefined { params, .. }) => {
                    Some(ListItem::new(Span::styled(
                        format!("{}({})", name, params.join(", ")),
                        Theme::sidebar_item(),
                    )))
                }
                _ => None,
            })
            .take(inner.height as usize)
            .collect();

        let list = List::new(items);
        frame.render_widget(list, inner);
    }

    fn render_hint(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Theme::border())
            .title(Span::styled(" Hint ", Theme::sidebar_title()));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        if let Some(hint) = detect_active_function(&self.input.text, self.input.cursor) {
            let lines = vec![
                Line::from(Span::styled(
                    hint.signature,
                    Style::default().fg(ratatui::style::Color::Cyan),
                )),
                Line::from(Span::styled(
                    hint.description,
                    Style::default().fg(ratatui::style::Color::DarkGray),
                )),
            ];
            let paragraph = Paragraph::new(lines);
            frame.render_widget(paragraph, inner);
        }
    }

    fn render_input(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Theme::border_focused())
            .title(" Input ");

        let inner = block.inner(area);
        frame.render_widget(block, area);

        let prompt = "aurita> ";
        let line = Line::from(vec![
            Span::styled(prompt, Theme::input_prompt()),
            Span::styled(&self.input.text, Theme::input_text()),
        ]);

        let paragraph = Paragraph::new(line);
        frame.render_widget(paragraph, inner);

        // Position cursor
        let cursor_x = inner.x + prompt.len() as u16 + self.input.cursor as u16;
        let cursor_y = inner.y;
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}
