use std::io;
use std::time::Duration;

use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use ratatui_image::picker::Picker;

use aurita::tui::app::App;
use aurita::tui::event::{poll_event, AppEvent};

fn main() -> anyhow::Result<()> {
    // Query terminal for image protocol support BEFORE entering alternate screen
    let picker = Picker::from_query_stdio().ok();

    // Set up terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run app
    let result = run_app(&mut terminal, picker);

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(err) = result {
        eprintln!("Error: {}", err);
    }

    Ok(())
}

/// Locate the Python bridge script relative to the executable or in known paths.
fn locate_bridge() -> Option<String> {
    // 1. Next to the executable
    if let Ok(exe) = std::env::current_exe() {
        let dir = exe.parent()?;
        let candidate = dir.join("../backends/python_bridge.py");
        if candidate.exists() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }

    // 2. Current working directory
    let cwd_candidate = std::path::Path::new("backends/python_bridge.py");
    if cwd_candidate.exists() {
        return Some(cwd_candidate.to_string_lossy().to_string());
    }

    // 3. Cargo manifest directory (development)
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let candidate = std::path::Path::new(&manifest_dir).join("backends/python_bridge.py");
        if candidate.exists() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }

    None
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, picker: Option<Picker>) -> anyhow::Result<()> {
    let mut app = App::new(picker);

    // Try to locate and spawn the SymPy bridge
    let bridge_path = locate_bridge();
    if let Some(path) = &bridge_path {
        app.evaluator.init_cas(path);
    }

    loop {
        terminal.draw(|frame| app.render(frame))?; // render takes &mut self via app

        if let Some(event) = poll_event(Duration::from_millis(50)) {
            match event {
                AppEvent::Key(key) => {
                    app.handle_key(key);
                }
                AppEvent::Resize(_, _) => {
                    // Terminal will auto-redraw
                }
                AppEvent::Tick => {}
            }
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}
