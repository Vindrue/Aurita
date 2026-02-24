use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct Config {
    /// Maximum number of history entries to keep.
    pub history_limit: usize,
    /// Default CAS backend ("sympy", "maxima", or "both").
    pub default_backend: String,
    /// Height in terminal rows for plot images.
    pub plot_height: u16,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            history_limit: 1000,
            default_backend: "sympy".to_string(),
            plot_height: 20,
        }
    }
}

/// Path to the config file.
pub fn config_path() -> Option<PathBuf> {
    Some(super::config_dir()?.join("config.toml"))
}

/// Load config from disk, returning defaults if file doesn't exist or is invalid.
pub fn load_config() -> Config {
    let path = match config_path() {
        Some(p) => p,
        None => return Config::default(),
    };
    match std::fs::read_to_string(&path) {
        Ok(content) => toml::from_str(&content).unwrap_or_default(),
        Err(_) => {
            // Create default config file on first run
            let config = Config::default();
            let _ = write_default_config(&path, &config);
            config
        }
    }
}

/// Write a default config file with comments.
fn write_default_config(path: &PathBuf, config: &Config) -> Result<(), String> {
    let content = format!(
        "# Aurita configuration\n\
         \n\
         # Maximum number of history entries to keep\n\
         history_limit = {}\n\
         \n\
         # Default CAS backend: \"sympy\", \"maxima\", or \"both\"\n\
         default_backend = \"{}\"\n\
         \n\
         # Height in terminal rows for plot images\n\
         plot_height = {}\n",
        config.history_limit,
        config.default_backend,
        config.plot_height,
    );
    std::fs::write(path, content.as_bytes())
        .map_err(|e| format!("write error: {}", e))
}
