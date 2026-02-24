pub mod history;
pub mod session;
pub mod config;

use std::path::PathBuf;

/// Get or create the Aurita data directory (~/.local/share/aurita/).
pub fn data_dir() -> Option<PathBuf> {
    let dir = dirs::data_dir()?.join("aurita");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Get or create the Aurita config directory (~/.config/aurita/).
pub fn config_dir() -> Option<PathBuf> {
    let dir = dirs::config_dir()?.join("aurita");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}
