use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph};
use ratatui::Frame;
use ratatui_image::picker::Picker;
use ratatui_image::StatefulImage;

use crate::lang::builtins::{numeric_eval_sym, BUILTIN_CONSTANTS};
use crate::persistence;
use crate::physics::constants;
use crate::lang::eval::Evaluator;
use crate::lang::lexer::Lexer;
use crate::lang::parser::Parser;
use crate::lang::types::{Function, Value};
use crate::tui::completion::CompletionState;
use crate::tui::help::HelpPanel;
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
    pub picker: Option<Picker>,
    pub completion: CompletionState,
    pub help: HelpPanel,
    pub session_name: Option<String>,
    pub config: persistence::config::Config,
}

impl App {
    pub fn new(picker: Option<Picker>, history: Vec<String>, config: persistence::config::Config) -> Self {
        Self {
            input: InputState::new(history),
            worksheet: WorksheetState::new(),
            evaluator: Evaluator::new(),
            should_quit: false,
            picker,
            completion: CompletionState::new(),
            help: HelpPanel::new(),
            session_name: None,
            config,
        }
    }

    /// Handle a key event. Returns true if the screen should be redrawn.
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        // Help panel mode
        if self.help.visible {
            return self.handle_key_help(key);
        }

        // Completion mode: intercept keys for the popup
        if self.completion.active {
            return self.handle_key_completion(key);
        }

        self.handle_key_normal(key)
    }

    /// Key handling when the help panel is visible.
    fn handle_key_help(&mut self, key: KeyEvent) -> bool {
        match key {
            KeyEvent { code: KeyCode::Esc, .. }
            | KeyEvent { code: KeyCode::Char('h'), modifiers: KeyModifiers::CONTROL, .. }
            | KeyEvent { code: KeyCode::F(1), .. } => {
                self.help.toggle();
                true
            }
            KeyEvent { code: KeyCode::Up, .. } | KeyEvent { code: KeyCode::Char('k'), .. } => {
                self.help.scroll_up(1);
                true
            }
            KeyEvent { code: KeyCode::Down, .. } | KeyEvent { code: KeyCode::Char('j'), .. } => {
                self.help.scroll_down(1);
                true
            }
            KeyEvent { code: KeyCode::PageUp, .. } => {
                self.help.scroll_up(10);
                true
            }
            KeyEvent { code: KeyCode::PageDown, .. } => {
                self.help.scroll_down(10);
                true
            }
            // Any other key closes help
            _ => {
                self.help.visible = false;
                false
            }
        }
    }

    /// Key handling when the completion popup is active.
    fn handle_key_completion(&mut self, key: KeyEvent) -> bool {
        match key {
            // Accept selection
            KeyEvent { code: KeyCode::Enter, .. } => {
                if let Some((start, end, text)) = self.completion.accept() {
                    self.input.replace_range(start, end, &text);
                }
                true
            }
            // Next candidate
            KeyEvent { code: KeyCode::Tab, modifiers, .. }
                if !modifiers.contains(KeyModifiers::SHIFT) =>
            {
                self.completion.select_next();
                true
            }
            KeyEvent { code: KeyCode::Down, .. } => {
                self.completion.select_next();
                true
            }
            // Previous candidate
            KeyEvent { code: KeyCode::BackTab, .. }
            | KeyEvent { code: KeyCode::Tab, modifiers: KeyModifiers::SHIFT, .. } => {
                self.completion.select_prev();
                true
            }
            KeyEvent { code: KeyCode::Up, .. } => {
                self.completion.select_prev();
                true
            }
            // Dismiss
            KeyEvent { code: KeyCode::Esc, .. } => {
                self.completion.dismiss();
                true
            }
            // Any other key: dismiss completion and forward to normal handler
            _ => {
                self.completion.dismiss();
                self.handle_key_normal(key)
            }
        }
    }

    /// Normal mode key handling.
    fn handle_key_normal(&mut self, key: KeyEvent) -> bool {
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
                    persistence::history::append_history(&text);
                    self.execute(&text);
                }
                true
            }

            // Tab completion
            KeyEvent {
                code: KeyCode::Tab,
                modifiers,
                ..
            } if !modifiers.contains(KeyModifiers::SHIFT) => {
                self.completion.compute(
                    &self.input.text,
                    self.input.cursor,
                    &self.evaluator.env,
                );
                // If exactly one match, insert directly
                if self.completion.candidates.len() == 1 {
                    if let Some((start, end, text)) = self.completion.accept() {
                        self.input.replace_range(start, end, &text);
                    }
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

            // Help panel toggle
            KeyEvent {
                code: KeyCode::Char('h'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.help.toggle();
                true
            }
            KeyEvent {
                code: KeyCode::F(1),
                ..
            } => {
                self.help.toggle();
                true
            }

            // Escape (no-op in normal mode, but consume the key)
            KeyEvent {
                code: KeyCode::Esc,
                ..
            } => false,

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
        // Check for :commands
        if let Some(cmd) = input.strip_prefix(':') {
            self.execute_command(cmd.trim());
            return;
        }

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

    /// Execute a :command.
    fn execute_command(&mut self, cmd: &str) {
        let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
        let name = parts[0];
        let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

        match name {
            "help" => {
                self.help.toggle();
            }
            "clear" => {
                self.worksheet = WorksheetState::new();
            }
            "save" => {
                self.command_save(arg);
            }
            "load" => {
                self.command_load(arg);
            }
            "sessions" => {
                self.command_list_sessions();
            }
            _ => {
                self.worksheet.add_entry(
                    format!(":{}", cmd),
                    OutputKind::Error(format!("unknown command: :{}", name)),
                    vec![],
                );
            }
        }
        self.worksheet.scroll_to_bottom(50);
    }

    fn command_save(&mut self, arg: &str) {
        use crate::persistence::session::*;

        let name = if arg.is_empty() {
            timestamp_name()
        } else {
            arg.to_string()
        };

        let mut entries = Vec::new();
        for entry in &self.worksheet.entries {
            let output = match &entry.output {
                OutputKind::Value(v) => SavedOutput::Value(v.clone()),
                OutputKind::Error(e) => SavedOutput::Error(e.clone()),
                OutputKind::PrintOutput(lines) => SavedOutput::PrintOutput(lines.clone()),
                OutputKind::Plot(p) => SavedOutput::Plot {
                    label: p.spec.series.iter()
                        .map(|s| s.label.as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                },
            };
            entries.push(SavedEntry {
                index: entry.index,
                input: entry.input.clone(),
                output,
            });
        }

        let session = SavedSession {
            name: name.clone(),
            timestamp: timestamp_iso(),
            entries,
        };

        match save_session(&session) {
            Ok(()) => {
                self.session_name = Some(name.clone());
                self.worksheet.add_entry(
                    format!(":save {}", name),
                    OutputKind::Value(format!("Session saved: {}", name)),
                    vec![],
                );
            }
            Err(e) => {
                self.worksheet.add_entry(
                    format!(":save {}", name),
                    OutputKind::Error(format!("Save failed: {}", e)),
                    vec![],
                );
            }
        }
    }

    fn command_load(&mut self, arg: &str) {
        use crate::persistence::session::*;

        if arg.is_empty() {
            self.worksheet.add_entry(
                ":load".to_string(),
                OutputKind::Error("usage: :load <name>".to_string()),
                vec![],
            );
            return;
        }

        match load_session(arg) {
            Ok(session) => {
                // Reset state
                self.worksheet = WorksheetState::new();
                self.evaluator = Evaluator::new();
                // Re-init CAS from evaluator (preserve bridge paths)
                // Just replay the inputs
                let inputs: Vec<String> = session.entries.iter()
                    .map(|e| e.input.clone())
                    .collect();
                // Deduplicate (same index may appear twice if there was PrintOutput)
                let mut seen_inputs: Vec<String> = Vec::new();
                for input in &inputs {
                    if seen_inputs.last().map_or(true, |last| last != input) {
                        seen_inputs.push(input.clone());
                    }
                }
                for input in &seen_inputs {
                    if !input.starts_with(':') {
                        self.evaluator.output.clear();
                        let result = Lexer::new(input)
                            .tokenize()
                            .and_then(|tokens| Parser::new(tokens).parse_program())
                            .and_then(|stmts| self.evaluator.eval_program(&stmts));
                        let print_lines = self.evaluator.output.clone();
                        match result {
                            Ok(Value::Plot(rendered)) => {
                                self.worksheet.add_entry(input.clone(), OutputKind::Plot(rendered), print_lines);
                            }
                            Ok(value) => {
                                self.worksheet.add_entry(input.clone(), OutputKind::Value(format!("{}", value)), print_lines);
                            }
                            Err(err) => {
                                self.worksheet.add_entry(input.clone(), OutputKind::Error(err.message), print_lines);
                            }
                        }
                    }
                }
                self.session_name = Some(arg.to_string());
                self.worksheet.add_entry(
                    format!(":load {}", arg),
                    OutputKind::Value(format!("Session loaded: {} ({} entries)", arg, seen_inputs.len())),
                    vec![],
                );
                self.worksheet.scroll_to_bottom(50);
            }
            Err(e) => {
                self.worksheet.add_entry(
                    format!(":load {}", arg),
                    OutputKind::Error(format!("Load failed: {}", e)),
                    vec![],
                );
            }
        }
    }

    fn command_list_sessions(&mut self) {
        use crate::persistence::session::list_sessions;

        let sessions = list_sessions();
        if sessions.is_empty() {
            self.worksheet.add_entry(
                ":sessions".to_string(),
                OutputKind::Value("No saved sessions.".to_string()),
                vec![],
            );
        } else {
            let lines: Vec<String> = sessions.iter()
                .map(|(name, ts)| format!("  {} ({})", name, ts))
                .collect();
            self.worksheet.add_entry(
                ":sessions".to_string(),
                OutputKind::Value(format!("Saved sessions:\n{}", lines.join("\n"))),
                vec![],
            );
        }
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
        render_status_bar(frame, outer[0], &cas_status, self.session_name.as_deref());

        // Main area: worksheet + sidebar
        let main = Layout::horizontal([
            Constraint::Percentage(70),
            Constraint::Percentage(30),
        ])
        .split(outer[1]);

        self.render_worksheet(frame, main[0]);
        self.render_sidebar(frame, main[1]);
        self.render_input(frame, outer[2]);

        // Render completion popup as overlay above input bar
        if self.completion.active {
            self.render_completion(frame, outer[2]);
        }

        // Render help panel as centered overlay
        if self.help.visible {
            self.render_help(frame, frame.area());
        }
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
        let plot_rows = self.config.plot_height as usize;
        let total_height: usize = items.iter().map(|item| match item {
            RenderItem::TextLine(_) => 1,
            RenderItem::PlotImage(_) => plot_rows,
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
                RenderItem::PlotImage(_) => plot_rows,
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
                    let plot_h = remaining.min(self.config.plot_height);
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

    fn render_help(&self, frame: &mut Frame, area: Rect) {
        use crate::tui::help::HELP_SECTIONS;

        // 80% of screen, centered
        let w = (area.width * 4 / 5).max(40);
        let h = (area.height * 4 / 5).max(10);
        let x = area.x + (area.width.saturating_sub(w)) / 2;
        let y = area.y + (area.height.saturating_sub(h)) / 2;
        let popup_area = Rect { x, y, width: w, height: h };

        frame.render_widget(Clear, popup_area);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Theme::border_focused())
            .title(Span::styled(
                " Help \u{2014} Esc to close, \u{2191}/\u{2193} to scroll ",
                Theme::sidebar_title(),
            ));
        let inner = block.inner(popup_area);
        frame.render_widget(block, popup_area);

        // Build all lines
        let mut lines: Vec<Line<'static>> = Vec::new();
        for &(title, content) in HELP_SECTIONS {
            lines.push(Line::from(Span::styled(
                format!(" {} ", title),
                Style::default().fg(Color::Cyan).add_modifier(ratatui::style::Modifier::BOLD),
            )));
            lines.push(Line::from(""));
            for text_line in content.lines() {
                lines.push(Line::from(Span::styled(
                    format!("  {}", text_line),
                    Style::default(),
                )));
            }
            lines.push(Line::from(""));
        }

        // Clamp scroll
        let total = lines.len();
        let visible = inner.height as usize;
        let max_scroll = total.saturating_sub(visible);
        let scroll = self.help.scroll.min(max_scroll);

        let visible_lines: Vec<Line> = lines.into_iter().skip(scroll).take(visible).collect();
        let paragraph = Paragraph::new(visible_lines);
        frame.render_widget(paragraph, inner);
    }

    fn render_completion(&self, frame: &mut Frame, input_area: Rect) {
        let max_vis = self.completion.max_visible();
        let count = self.completion.candidates.len().min(max_vis);
        if count == 0 {
            return;
        }

        let popup_height = count as u16 + 2; // +2 for border
        let prompt_len = "aurita> ".len() as u16;
        let popup_x = input_area.x + 1 + prompt_len + self.completion.prefix_start as u16;
        let popup_y = input_area.y.saturating_sub(popup_height);
        let popup_width = 40u16.min(input_area.width.saturating_sub(popup_x.saturating_sub(input_area.x)));

        let popup_area = Rect {
            x: popup_x.min(input_area.x + input_area.width - popup_width),
            y: popup_y,
            width: popup_width,
            height: popup_height,
        };

        // Clear the area behind the popup
        frame.render_widget(Clear, popup_area);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));
        let inner = block.inner(popup_area);
        frame.render_widget(block, popup_area);

        let offset = self.completion.scroll_offset();
        let items: Vec<ListItem> = self.completion.candidates
            .iter()
            .skip(offset)
            .take(max_vis)
            .enumerate()
            .map(|(i, item)| {
                let is_selected = i + offset == self.completion.selected;
                let kind_label = item.kind.label();
                let desc = if item.description.is_empty() {
                    String::new()
                } else {
                    let avail = (inner.width as usize)
                        .saturating_sub(item.text.len() + kind_label.len() + 4);
                    if avail > 3 {
                        let d = if item.description.len() > avail {
                            format!("{}...", &item.description[..avail.saturating_sub(3)])
                        } else {
                            item.description.clone()
                        };
                        format!(" {}", d)
                    } else {
                        String::new()
                    }
                };

                let style = if is_selected {
                    Style::default().fg(Color::Black).bg(Color::Cyan)
                } else {
                    Style::default()
                };
                let kind_style = if is_selected {
                    Style::default().fg(Color::Black).bg(Color::Cyan)
                } else {
                    Style::default().fg(Color::DarkGray)
                };

                ListItem::new(Line::from(vec![
                    Span::styled(format!("{} ", kind_label), kind_style),
                    Span::styled(&item.text, style),
                    Span::styled(desc, kind_style),
                ]))
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, inner);
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
