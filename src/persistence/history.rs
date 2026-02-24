use std::path::PathBuf;

const MAX_HISTORY: usize = 1000;

/// Path to the history file.
pub fn history_path() -> Option<PathBuf> {
    Some(super::data_dir()?.join("history"))
}

/// Load history from disk. Returns empty vec if file doesn't exist.
pub fn load_history() -> Vec<String> {
    let path = match history_path() {
        Some(p) => p,
        None => return Vec::new(),
    };
    match std::fs::read_to_string(&path) {
        Ok(content) => content
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.to_string())
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Save history to disk, keeping only the last MAX_HISTORY entries.
/// Deduplicates consecutive identical entries.
pub fn save_history(entries: &[String]) {
    let path = match history_path() {
        Some(p) => p,
        None => return,
    };

    let mut deduped: Vec<&str> = Vec::new();
    for entry in entries {
        if deduped.last().map_or(true, |last| *last != entry.as_str()) {
            deduped.push(entry.as_str());
        }
    }

    // Keep only the last MAX_HISTORY
    let start = deduped.len().saturating_sub(MAX_HISTORY);
    let trimmed = &deduped[start..];

    let content = trimmed.join("\n");
    let _ = std::fs::write(&path, content.as_bytes());
}

/// Append a single line to the history file (crash-safe incremental save).
pub fn append_history(line: &str) {
    let path = match history_path() {
        Some(p) => p,
        None => return,
    };
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(file, "{}", line);
    }
}
