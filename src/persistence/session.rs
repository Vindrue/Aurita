use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize)]
pub struct SavedSession {
    pub name: String,
    pub timestamp: String,
    pub entries: Vec<SavedEntry>,
}

#[derive(Serialize, Deserialize)]
pub struct SavedEntry {
    pub index: usize,
    pub input: String,
    pub output: SavedOutput,
}

#[derive(Serialize, Deserialize)]
pub enum SavedOutput {
    Value(String),
    Error(String),
    PrintOutput(Vec<String>),
    Plot { label: String },
}

/// Get or create the sessions directory.
pub fn sessions_dir() -> Option<PathBuf> {
    let dir = super::data_dir()?.join("sessions");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// List available sessions, sorted by timestamp (newest first).
pub fn list_sessions() -> Vec<(String, String)> {
    let dir = match sessions_dir() {
        Some(d) => d,
        None => return Vec::new(),
    };
    let mut sessions: Vec<(String, String)> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "json") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(session) = serde_json::from_str::<SavedSession>(&content) {
                        sessions.push((session.name, session.timestamp));
                    }
                }
            }
        }
    }
    sessions.sort_by(|a, b| b.1.cmp(&a.1));
    sessions
}

/// Save a session to disk.
pub fn save_session(session: &SavedSession) -> Result<(), String> {
    let dir = sessions_dir().ok_or("cannot determine sessions directory")?;
    let filename = format!("{}.json", sanitize_filename(&session.name));
    let path = dir.join(filename);
    let json = serde_json::to_string_pretty(session)
        .map_err(|e| format!("serialize error: {}", e))?;
    std::fs::write(&path, json.as_bytes())
        .map_err(|e| format!("write error: {}", e))?;
    Ok(())
}

/// Load a session from disk by name.
pub fn load_session(name: &str) -> Result<SavedSession, String> {
    let dir = sessions_dir().ok_or("cannot determine sessions directory")?;
    let filename = format!("{}.json", sanitize_filename(name));
    let path = dir.join(filename);
    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("read error: {}", e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("parse error: {}", e))
}

/// Sanitize a name for use as a filename.
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

/// Generate a timestamp string for session naming.
pub fn timestamp_name() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Simple readable format: seconds since epoch
    format!("session_{}", secs)
}

/// Generate an ISO 8601-ish timestamp.
pub fn timestamp_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Simple: just use epoch seconds (no chrono dependency)
    format!("{}", secs)
}
