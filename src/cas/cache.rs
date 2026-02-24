use lru::LruCache;
use std::num::NonZeroUsize;

use crate::cas::protocol::{CasOp, CasResponse};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    op_json: String,
    backend: String,
}

/// LRU cache for CAS operation results.
pub struct CasCache {
    cache: LruCache<CacheKey, CasResponse>,
}

impl CasCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(capacity).unwrap()),
        }
    }

    /// Check if an operation should be cached (skip large I/O ops).
    fn should_cache(op: &CasOp) -> bool {
        !matches!(op, CasOp::Lambdify { .. } | CasOp::RenderPlot { .. })
    }

    /// Look up a cached result for the given operation and backend.
    pub fn get(&mut self, op: &CasOp, backend: &str) -> Option<CasResponse> {
        if !Self::should_cache(op) {
            return None;
        }
        let key = Self::make_key(op, backend)?;
        self.cache.get(&key).cloned()
    }

    /// Store a result in the cache.
    pub fn put(&mut self, op: &CasOp, backend: &str, response: CasResponse) {
        if !Self::should_cache(op) {
            return;
        }
        if let Some(key) = Self::make_key(op, backend) {
            self.cache.put(key, response);
        }
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    fn make_key(op: &CasOp, backend: &str) -> Option<CacheKey> {
        let op_json = serde_json::to_string(op).ok()?;
        Some(CacheKey {
            op_json,
            backend: backend.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::expr::SymExpr;

    fn make_diff_op(var: &str) -> CasOp {
        CasOp::Differentiate {
            expr: SymExpr::sym("x"),
            var: var.to_string(),
            order: 1,
        }
    }

    fn make_response(id: u64) -> CasResponse {
        CasResponse {
            id,
            status: "ok".to_string(),
            result: Some(SymExpr::int(42)),
            error: None,
            results: None,
            latex: None,
            y_values: None,
            png_base64: None,
        }
    }

    #[test]
    fn test_cache_put_get() {
        let mut cache = CasCache::new(16);
        let op = make_diff_op("x");
        let resp = make_response(1);
        cache.put(&op, "sympy", resp.clone());
        let cached = cache.get(&op, "sympy");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().id, 1);
    }

    #[test]
    fn test_cache_miss_different_backend() {
        let mut cache = CasCache::new(16);
        let op = make_diff_op("x");
        cache.put(&op, "sympy", make_response(1));
        assert!(cache.get(&op, "maxima").is_none());
    }

    #[test]
    fn test_cache_miss_different_op() {
        let mut cache = CasCache::new(16);
        let op1 = make_diff_op("x");
        let op2 = make_diff_op("y");
        cache.put(&op1, "sympy", make_response(1));
        assert!(cache.get(&op2, "sympy").is_none());
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = CasCache::new(2);
        let op1 = make_diff_op("a");
        let op2 = make_diff_op("b");
        let op3 = make_diff_op("c");
        cache.put(&op1, "sympy", make_response(1));
        cache.put(&op2, "sympy", make_response(2));
        cache.put(&op3, "sympy", make_response(3));
        // op1 should have been evicted
        assert!(cache.get(&op1, "sympy").is_none());
        assert!(cache.get(&op2, "sympy").is_some());
        assert!(cache.get(&op3, "sympy").is_some());
    }

    #[test]
    fn test_cache_skip_lambdify() {
        let mut cache = CasCache::new(16);
        let op = CasOp::Lambdify {
            expr: SymExpr::sym("x"),
            var: "x".to_string(),
            x_values: vec![1.0, 2.0],
        };
        cache.put(&op, "sympy", make_response(1));
        assert!(cache.get(&op, "sympy").is_none());
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = CasCache::new(16);
        let op = make_diff_op("x");
        cache.put(&op, "sympy", make_response(1));
        cache.clear();
        assert!(cache.get(&op, "sympy").is_none());
    }
}
