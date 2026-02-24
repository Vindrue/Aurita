use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
use ratatui::Frame;
use ratatui_image::picker::Picker;
use ratatui_image::StatefulImage;

use crate::lang::builtins::{numeric_eval_sym, BUILTIN_CONSTANTS};
use crate::physics::constants;
use crate::lang::eval::Evaluator;
use crate::lang::lexer::Lexer;
use crate::lang::parser::Parser;
use crate::lang::types::{Function, Value};
use crate::tui::hints::detect_active_function;
use crate::tui::input::InputState;
use crate::tui::output::{OutputKind, WorksheetState, PLOT_ROWS};
use crate::tui::status::render_status_bar;
use crate::tui::theme::Theme;

pub struct App {
    pub input: InputState,
    pub worksheet: WorksheetState,
    pub evaluator: Evaluator,
    pub should_quit: bool,
    pub picker: Option<Picker>,
}

impl App {
    pub fn new(picker: Option<Picker>) -> Self {
        Self {
            input: InputState::new(),
            worksheet: WorksheetState::new(),
            evaluator: Evaluator::new(),
            should_quit: false,
            picker,
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
            Ok(Value::Plot(rendered)) => {
                self.worksheet.add_entry(
                    input.to_string(),
                    OutputKind::Plot(rendered),
                    print_lines,
                );
            }
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
    pub fn render(&mut self, frame: &mut Frame) {
        let outer = Layout::vertical([
            Constraint::Length(1),  // Status bar
            Constraint::Min(5),    // Main area
            Constraint::Length(3), // Input bar
        ])
        .split(frame.area());

        // Status bar
        let cas_status = self.evaluator.cas_status();
        render_status_bar(frame, outer[0], &cas_status);

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

    fn render_worksheet(&mut self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Theme::border_focused())
            .title(" Worksheet ");

        let inner = block.inner(area);
        frame.render_widget(block, area);

        // Build a flat list of render items with their heights
        enum RenderItem {
            TextLine(Line<'static>),
            PlotImage(usize), // index into worksheet entries
        }

        let mut items: Vec<RenderItem> = Vec::new();
        for (entry_idx, entry) in self.worksheet.entries.iter().enumerate() {
            // Input line
            items.push(RenderItem::TextLine(Line::from(vec![
                Span::styled(
                    format!(" In[{}]: ", entry.index),
                    Theme::input_prompt(),
                ),
                Span::styled(entry.input.clone(), Theme::input_text()),
            ])));

            match &entry.output {
                OutputKind::Plot(_) => {
                    // Reserve PLOT_ROWS for image rendering
                    items.push(RenderItem::PlotImage(entry_idx));
                }
                _ => {
                    for (text, style) in entry.output_lines() {
                        items.push(RenderItem::TextLine(Line::from(Span::styled(text, style))));
                    }
                }
            }

            // Blank separator
            items.push(RenderItem::TextLine(Line::from("")));
        }

        // Calculate total height
        let total_height: usize = items.iter().map(|item| match item {
            RenderItem::TextLine(_) => 1,
            RenderItem::PlotImage(_) => PLOT_ROWS as usize,
        }).sum();

        // Auto-scroll to bottom
        let visible_height = inner.height as usize;
        let offset = if total_height > visible_height {
            total_height.saturating_sub(visible_height)
        } else {
            0
        };

        // Render visible items
        let mut y_pos: usize = 0;
        for item in &items {
            let item_height = match item {
                RenderItem::TextLine(_) => 1,
                RenderItem::PlotImage(_) => PLOT_ROWS as usize,
            };

            // Skip items before the scroll offset
            if y_pos + item_height <= offset {
                y_pos += item_height;
                continue;
            }
            // Stop if past the visible area
            if y_pos >= offset + visible_height {
                break;
            }

            let render_y = (y_pos.saturating_sub(offset)) as u16;

            match item {
                RenderItem::TextLine(line) => {
                    let line_area = Rect {
                        x: inner.x,
                        y: inner.y + render_y,
                        width: inner.width,
                        height: 1,
                    };
                    frame.render_widget(Paragraph::new(line.clone()), line_area);
                }
                RenderItem::PlotImage(entry_idx) => {
                    let remaining = (inner.y + inner.height).saturating_sub(inner.y + render_y);
                    let plot_h = remaining.min(PLOT_ROWS);
                    if plot_h > 0 {
                        let plot_area = Rect {
                            x: inner.x,
                            y: inner.y + render_y,
                            width: inner.width,
                            height: plot_h,
                        };
                        self.render_plot_image(frame, plot_area, *entry_idx);
                    }
                }
            }

            y_pos += item_height;
        }
    }

    /// Render a plot image into the given area using ratatui-image.
    fn render_plot_image(&self, frame: &mut Frame, area: Rect, entry_idx: usize) {
        let entry = &self.worksheet.entries[entry_idx];
        let png_bytes = match &entry.output {
            OutputKind::Plot(p) => &p.png_bytes,
            _ => return,
        };

        // Initialize image state if needed
        let needs_init = entry.image_state.borrow().is_none();
        if needs_init {
            if let Some(picker) = &self.picker {
                match image::load_from_memory(png_bytes) {
                    Ok(dyn_image) => {
                        let protocol = picker.new_resize_protocol(dyn_image);
                        *entry.image_state.borrow_mut() = Some(protocol);
                    }
                    Err(e) => {
                        // Fallback: show error text
                        let text = format!("[plot decode error: {}]", e);
                        frame.render_widget(
                            Paragraph::new(Span::styled(text, Theme::error())),
                            area,
                        );
                        return;
                    }
                }
            } else {
                // No picker — show fallback text
                let labels: String = match &entry.output {
                    OutputKind::Plot(p) => p.spec.series.iter()
                        .map(|s| s.label.as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                    _ => String::new(),
                };
                frame.render_widget(
                    Paragraph::new(Span::styled(
                        format!("[plot: {} — image display requires Kitty/iTerm2]", labels),
                        Theme::output_value(),
                    )),
                    area,
                );
                return;
            }
        }

        let mut state = entry.image_state.borrow_mut();
        if let Some(protocol) = state.as_mut() {
            let image_widget = StatefulImage::default();
            frame.render_stateful_widget(image_widget, area, protocol);
        }
    }

    fn render_sidebar(&self, frame: &mut Frame, area: Rect) {
        let has_hint = detect_active_function(&self.input.text, self.input.cursor).is_some();
        let hint_height = if has_hint { 4u16 } else { 0u16 };

        let sidebar = Layout::vertical([
            Constraint::Length(6),                          // Constants
            Constraint::Length(10),                         // Physics
            Constraint::Min(3),                            // Variables
            Constraint::Min(3),                            // Functions
            Constraint::Length(hint_height),                // Hint (conditional)
        ])
        .split(area);

        self.render_constants(frame, sidebar[0]);
        self.render_physics(frame, sidebar[1]);
        self.render_variables(frame, sidebar[2]);
        self.render_functions(frame, sidebar[3]);
        if has_hint {
            self.render_hint(frame, sidebar[4]);
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

    fn render_physics(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Theme::border())
            .title(Span::styled(" Physics ", Theme::sidebar_title()));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        // Show a curated set of the most commonly used physics constants
        let show_names = ["c", "h", "hbar", "G", "k_B", "N_A", "e_charge", "m_e"];
        let mut items: Vec<ListItem> = Vec::new();

        for &name in &show_names {
            if items.len() >= inner.height as usize {
                break;
            }
            if let Some(pc) = constants::lookup(name) {
                let val_str = if pc.uncertainty > 0.0 {
                    format!("{:.3e} +/-", pc.value)
                } else {
                    format!("{:.4e}", pc.value)
                };
                let unit_part = if pc.unit_display.is_empty() {
                    String::new()
                } else {
                    format!(" {}", pc.unit_display)
                };
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("{:<8}", name),
                        Style::default().fg(ratatui::style::Color::Yellow),
                    ),
                    Span::styled(
                        format!("{}{}", val_str, unit_part),
                        Theme::sidebar_item(),
                    ),
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

        let physics_names = constants::all_names();
        let bindings = self.evaluator.env.borrow().all_bindings();
        let items: Vec<ListItem> = bindings
            .iter()
            .filter(|(name, v)| {
                !matches!(v, Value::Function(_))
                    && !BUILTIN_CONSTANTS.contains(&name.as_str())
                    && !physics_names.contains(&name.as_str())
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
