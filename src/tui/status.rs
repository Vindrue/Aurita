use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::Frame;

use crate::tui::theme::Theme;

pub fn render_status_bar(frame: &mut Frame, area: Rect, cas_status: &str, session_name: Option<&str>) {
    let version = env!("CARGO_PKG_VERSION");
    let left_text = match session_name {
        Some(name) => format!(" Aurita v{} | {}", version, name),
        None => format!(" Aurita v{}", version),
    };
    let right_text = format!("{} ", cas_status);

    let left = Span::styled(left_text.clone(), Theme::status_bar());
    let right = Span::styled(right_text.clone(), Theme::status_bar());

    let width = area.width as usize;
    let padding = width.saturating_sub(left_text.len() + right_text.len());

    let line = Line::from(vec![
        left,
        Span::styled(" ".repeat(padding), Theme::status_bar()),
        right,
    ]);

    frame.render_widget(line, area);
}
