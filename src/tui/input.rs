/// Input line state with history.
pub struct InputState {
    /// Current input text.
    pub text: String,
    /// Cursor position (byte offset).
    pub cursor: usize,
    /// Command history.
    pub history: Vec<String>,
    /// Current position in history (-1 = current input).
    pub history_pos: Option<usize>,
    /// Saved current input when browsing history.
    pub saved_input: String,
}

impl InputState {
    pub fn new(history: Vec<String>) -> Self {
        Self {
            text: String::new(),
            cursor: 0,
            history,
            history_pos: None,
            saved_input: String::new(),
        }
    }

    pub fn insert(&mut self, ch: char) {
        self.text.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            // Find the previous character boundary
            let prev = self.text[..self.cursor]
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.text.remove(prev);
            self.cursor = prev;
        }
    }

    pub fn delete(&mut self) {
        if self.cursor < self.text.len() {
            self.text.remove(self.cursor);
        }
    }

    pub fn move_left(&mut self) {
        if self.cursor > 0 {
            self.cursor = self.text[..self.cursor]
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
    }

    pub fn move_right(&mut self) {
        if self.cursor < self.text.len() {
            self.cursor += self.text[self.cursor..].chars().next().map(|c| c.len_utf8()).unwrap_or(0);
        }
    }

    pub fn move_home(&mut self) {
        self.cursor = 0;
    }

    pub fn move_end(&mut self) {
        self.cursor = self.text.len();
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }

    /// Submit current input: add to history, clear, return the text.
    pub fn submit(&mut self) -> String {
        let text = self.text.clone();
        if !text.trim().is_empty() {
            self.history.push(text.clone());
        }
        self.text.clear();
        self.cursor = 0;
        self.history_pos = None;
        self.saved_input.clear();
        text
    }

    pub fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        match self.history_pos {
            None => {
                self.saved_input = self.text.clone();
                self.history_pos = Some(self.history.len() - 1);
            }
            Some(pos) if pos > 0 => {
                self.history_pos = Some(pos - 1);
            }
            _ => return,
        }
        if let Some(pos) = self.history_pos {
            self.text = self.history[pos].clone();
            self.cursor = self.text.len();
        }
    }

    pub fn history_down(&mut self) {
        match self.history_pos {
            Some(pos) => {
                if pos + 1 < self.history.len() {
                    self.history_pos = Some(pos + 1);
                    self.text = self.history[pos + 1].clone();
                    self.cursor = self.text.len();
                } else {
                    // Back to current input
                    self.history_pos = None;
                    self.text = self.saved_input.clone();
                    self.cursor = self.text.len();
                }
            }
            None => {} // Already at current input
        }
    }

    pub fn kill_line(&mut self) {
        self.text.truncate(self.cursor);
    }

    /// Replace bytes [start..end) with `replacement` and update cursor.
    pub fn replace_range(&mut self, start: usize, end: usize, replacement: &str) {
        self.text.replace_range(start..end, replacement);
        self.cursor = start + replacement.len();
    }

    pub fn kill_word_back(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let before = &self.text[..self.cursor];
        let trimmed = before.trim_end();
        let new_end = trimmed
            .rfind(|c: char| c.is_whitespace() || c == '(' || c == ')' || c == ',')
            .map(|i| i + 1)
            .unwrap_or(0);
        self.text = format!("{}{}", &self.text[..new_end], &self.text[self.cursor..]);
        self.cursor = new_end;
    }
}
