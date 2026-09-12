use super::{sealed, unsupported, CacheError, CacheInfo, CacheKind, LayerCache, LayerCacheSpec};
use mlx_rs::Array;

pub(crate) struct FullCache {
    processed_tokens: usize,
    capacity: usize,
}
impl FullCache {
    pub(crate) fn new(_spec: &LayerCacheSpec, _capacity: usize) -> Result<Self, CacheError> {
        Err(unsupported("full storage allocation"))
    }
}
impl sealed::Sealed for FullCache {}
impl LayerCache for FullCache {
    fn update_and_fetch(
        &mut self,
        _keys: Array,
        _values: Array,
    ) -> Result<(Array, Array), CacheError> {
        Err(unsupported("full storage update"))
    }
    fn info(&self, layer: usize) -> CacheInfo {
        CacheInfo {
            layer,
            kind: CacheKind::Full,
            processed_tokens: self.processed_tokens,
            retained_positions: 0..self.processed_tokens,
            capacity: self.capacity,
        }
    }
}
