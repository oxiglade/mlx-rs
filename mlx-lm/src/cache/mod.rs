pub use crate::error::CacheError;
use mlx_rs::{Array, Dtype};
use std::{marker::PhantomData, num::NonZeroUsize, ops::Range, rc::Rc};
mod full;
mod rotating;

/// Per-request cache storage options.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CacheOptions {
    /// Policy used to resolve each layer's storage.
    pub policy: CachePolicy,
}
/// Selects model defaults or an explicit storage policy.
#[derive(Debug, Clone, Default)]
pub enum CachePolicy {
    /// Uses the architecture's per-layer attention pattern.
    #[default]
    ModelDefault,
    /// Retains every processed position.
    Full,
    /// Retains a prefix and a bounded recent window.
    Rotating {
        /// Number of allocated token positions.
        capacity: NonZeroUsize,
        /// Number of initial positions retained across wraps.
        keep_prefix: usize,
    },
}
/// Logical state and allocation information for one cache layer.
#[derive(Debug, Clone)]
pub struct CacheInfo {
    /// Zero-based decoder layer index.
    pub layer: usize,
    /// Resolved storage kind.
    pub kind: CacheKind,
    /// Absolute count of committed input tokens.
    pub processed_tokens: usize,
    /// Absolute range of retained logical positions.
    pub retained_positions: Range<usize>,
    /// Allocated token capacity.
    pub capacity: usize,
}
/// Admitted cache storage kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheKind {
    /// Storage retaining all processed positions.
    Full,
    /// Fixed-capacity rotating storage.
    Rotating,
}
/// Thread-bound decoder cache with transactional layer state.
pub struct Cache {
    layers: Vec<Box<dyn LayerCache>>,
    _thread_bound: PhantomData<Rc<()>>,
}
/// Opaque thread-bound handles and metadata for restoring a cache.
pub struct CacheSnapshot {
    _thread_bound: PhantomData<Rc<()>>,
}
impl Cache {
    pub(crate) fn new(
        _layout: &[LayerCacheSpec],
        _options: &CacheOptions,
        _capacity: Option<NonZeroUsize>,
    ) -> Result<Self, CacheError> {
        Err(unsupported("cache allocation"))
    }
    /// Returns logical metadata without evaluating or copying array buffers.
    pub fn info(&self) -> impl ExactSizeIterator<Item = CacheInfo> + '_ {
        self.layers
            .iter()
            .enumerate()
            .map(|(layer, cache)| cache.info(layer))
    }
    /// Captures O(1) array handles without evaluating or copying buffers.
    pub fn snapshot(&mut self) -> Result<CacheSnapshot, CacheError> {
        Err(unsupported("cache snapshot"))
    }
    /// Restores compatible handles and absolute-position metadata.
    pub fn restore(&mut self, snapshot: CacheSnapshot) -> Result<(), CacheError> {
        let _ = snapshot;
        Err(unsupported("cache restore"))
    }
    pub(crate) fn step(&mut self) -> Result<CacheStep<'_>, CacheError> {
        Err(unsupported("cache transaction"))
    }
}
/// The resolved shape and attention policy for one decoder layer.
#[derive(Debug, Clone)]
pub(crate) struct LayerCacheSpec {
    pub(crate) attention: crate::AttentionKind,
    pub(crate) batch_size: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) dtype: Dtype,
}
mod sealed {
    pub trait Sealed {}
}
/// Sealed layer storage returning K/V in logical attention order.
pub(crate) trait LayerCache: sealed::Sealed {
    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), CacheError>;
    fn info(&self, layer: usize) -> CacheInfo;
}
/// A staged update whose state is committed only after successful evaluation.
pub(crate) struct CacheStep<'cache> {
    cache: &'cache mut Cache,
}
impl CacheStep<'_> {
    pub(crate) fn update_and_fetch(
        &mut self,
        _layer: usize,
        _keys: Array,
        _values: Array,
    ) -> Result<(Array, Array), CacheError> {
        Err(unsupported("cache layer update"))
    }
    pub(crate) fn commit(self) -> Result<(), CacheError> {
        Err(unsupported("cache commit"))
    }
}
fn unsupported(operation: &'static str) -> CacheError {
    CacheError::UnsupportedPolicy(crate::NotYetImplemented(operation).to_string())
}
