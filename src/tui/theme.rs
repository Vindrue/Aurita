use ratatui::style::{Color, Modifier, Style};

pub struct Theme;

impl Theme {
    pub fn status_bar() -> Style {
        Style::default()
            .fg(Color::Reset)
            .bg(Color::DarkGray)
    }

    pub fn input_prompt() -> Style {
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
    }

    pub fn input_text() -> Style {
        Style::default()
    }

    pub fn output_label() -> Style {
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD)
    }

    pub fn output_value() -> Style {
        Style::default()
    }

    pub fn error() -> Style {
        Style::default().fg(Color::Red)
    }

    pub fn sidebar_title() -> Style {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    }

    pub fn sidebar_item() -> Style {
        Style::default()
    }

    pub fn border() -> Style {
        Style::default().fg(Color::DarkGray)
    }

    pub fn border_focused() -> Style {
        Style::default().fg(Color::Cyan)
    }

    pub fn hint() -> Style {
        Style::default().fg(Color::DarkGray)
    }
}
