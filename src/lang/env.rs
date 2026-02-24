use crate::lang::types::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub type EnvRef = Rc<RefCell<Env>>;

/// Scoped environment for variable and function bindings.
#[derive(Debug)]
pub struct Env {
    bindings: HashMap<String, Value>,
    parent: Option<EnvRef>,
}

impl Env {
    /// Create a new root (global) environment.
    pub fn new_global() -> EnvRef {
        Rc::new(RefCell::new(Env {
            bindings: HashMap::new(),
            parent: None,
        }))
    }

    /// Create a child scope that inherits from `parent`.
    pub fn new_child(parent: EnvRef) -> EnvRef {
        Rc::new(RefCell::new(Env {
            bindings: HashMap::new(),
            parent: Some(parent),
        }))
    }

    /// Look up a name, walking up the scope chain.
    pub fn get(&self, name: &str) -> Option<Value> {
        self.bindings
            .get(name)
            .cloned()
            .or_else(|| self.parent.as_ref()?.borrow().get(name))
    }

    /// Set a binding in this scope.
    pub fn set(&mut self, name: String, value: Value) {
        self.bindings.insert(name, value);
    }

    /// Update an existing binding in the nearest scope that has it,
    /// or create it in the current scope.
    pub fn update(&mut self, name: &str, value: Value) {
        if self.bindings.contains_key(name) {
            self.bindings.insert(name.to_string(), value);
        } else if let Some(ref parent) = self.parent {
            parent.borrow_mut().update(name, value);
        } else {
            self.bindings.insert(name.to_string(), value);
        }
    }

    /// Remove a binding from this scope (does not affect parents).
    pub fn remove(&mut self, name: &str) -> Option<Value> {
        self.bindings.remove(name)
    }

    /// Get all bindings visible from this scope (for the variable inspector).
    pub fn all_bindings(&self) -> Vec<(String, Value)> {
        let mut result = HashMap::new();

        // Walk up the chain, parent bindings first (so children override)
        self.collect_bindings(&mut result);

        let mut sorted: Vec<(String, Value)> = result.into_iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        sorted
    }

    fn collect_bindings(&self, result: &mut HashMap<String, Value>) {
        if let Some(ref parent) = self.parent {
            parent.borrow().collect_bindings(result);
        }
        for (k, v) in &self.bindings {
            result.insert(k.clone(), v.clone());
        }
    }
}
