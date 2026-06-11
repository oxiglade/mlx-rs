//! Clone-and-restore KV-cache snapshot for speculative rollback.
//!
//! Speculative decode over-commits the cache (it writes γ+1 verify tokens),
//! then either keeps the state (all-accept) or restores it to re-commit only
//! the accepted prefix (partial reject). `trim`-based rollback is unsound for
//! a wrapped rotating cache, so this snapshots the whole `Vec<T>` and restores
//! it — clone is shared-ptr cheap (the `Array`s are `mlx::core::array`
//! shared_ptr handles).

use log::warn;

/// Pre-step snapshot of one cache `Vec<T>`. Forces the caller to choose
/// between [`Self::commit`] (keep the over-committed state — full accept) and
/// [`Self::rollback_into`] (restore before re-committing the accepted prefix
/// — partial reject). Dropping without either is a logic bug; the destructor
/// logs.
pub struct CacheSnapshot<T: Clone> {
    saved: Option<Vec<T>>,
}

impl<T: Clone> CacheSnapshot<T> {
    pub fn new(cache: &[T]) -> Self {
        Self {
            saved: Some(cache.to_vec()),
        }
    }

    /// Discard the snapshot — the post-step cache state is what we want.
    pub fn commit(&mut self) {
        self.saved = None;
    }

    /// Restore the cache from the snapshot. Consumes the saved state so
    /// [`Drop`] doesn't warn.
    pub fn rollback_into(&mut self, cache: &mut Vec<T>) {
        if let Some(s) = self.saved.take() {
            *cache = s;
        }
    }
}

impl<T: Clone> Drop for CacheSnapshot<T> {
    fn drop(&mut self) {
        if self.saved.is_some() {
            warn!(
                "CacheSnapshot dropped without commit() or rollback_into(); \
                 KV cache state may be inconsistent"
            );
        }
    }
}
