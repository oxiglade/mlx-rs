use super::{sealed, unsupported, CacheError, CacheInfo, CacheKind, LayerCache, LayerCacheSpec};
use mlx_rs::Array;

pub(crate) struct RotatingCache {
    processed_tokens: usize,
    capacity: usize,
}
impl RotatingCache {
    pub(crate) fn new(
        _spec: &LayerCacheSpec,
        _capacity: usize,
        _keep_prefix: usize,
    ) -> Result<Self, CacheError> {
        Err(unsupported("rotating storage allocation"))
    }
}
impl sealed::Sealed for RotatingCache {}
impl LayerCache for RotatingCache {
    fn update_and_fetch(
        &mut self,
        _keys: Array,
        _values: Array,
    ) -> Result<(Array, Array), CacheError> {
        Err(unsupported("rotating storage update"))
    }
    fn info(&self, layer: usize) -> CacheInfo {
        CacheInfo {
            layer,
            kind: CacheKind::Rotating,
            processed_tokens: self.processed_tokens,
            retained_positions: self.processed_tokens.saturating_sub(self.capacity)
                ..self.processed_tokens,
            capacity: self.capacity,
        }
    }
}
