use super::{
    grown_capacity, invalid, sealed, validate, Buffer, CacheError, CacheInfo, CacheKind,
    LayerCache, LayerCacheSpec, LayerFingerprint,
};
use mlx_rs::Array;

#[derive(Clone)]
pub(crate) struct FullCache {
    spec: LayerCacheSpec,
    buffer: Buffer,
    processed_tokens: usize,
    capacity: usize,
}
impl FullCache {
    pub(crate) fn new(spec: &LayerCacheSpec, capacity: usize) -> Result<Self, CacheError> {
        if capacity == 0 {
            return Err(invalid("full cache capacity must be positive"));
        }
        Ok(Self {
            spec: spec.clone(),
            buffer: Buffer::new(spec, capacity)?,
            processed_tokens: 0,
            capacity,
        })
    }
}
impl sealed::Sealed for FullCache {}
impl LayerCache for FullCache {
    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), CacheError> {
        let length = validate(&self.spec, &keys, &values)?;
        let end = self
            .processed_tokens
            .checked_add(length)
            .ok_or_else(|| invalid("position overflow"))?;
        self.reserve(end)?;
        let buffer = &self.buffer;
        let buffer = buffer.replace(self.processed_tokens, &keys, &values)?;
        let logical = buffer.slice(0..end)?;
        self.buffer = buffer;
        self.processed_tokens = end;
        Ok(logical)
    }
    fn reserve(&mut self, required: usize) -> Result<(), CacheError> {
        let capacity = grown_capacity(self.capacity, required)?;
        if capacity != self.capacity {
            let (keys, values) = self.buffer.slice(0..self.processed_tokens)?;
            self.buffer = Buffer::new(&self.spec, capacity)?.replace(0, &keys, &values)?;
            self.capacity = capacity;
        }
        Ok(())
    }
    fn info(&self, layer: usize) -> CacheInfo {
        CacheInfo {
            layer,
            kind: CacheKind::Full,
            processed_tokens: self.processed_tokens,
            retained_prefix: 0..0,
            retained_positions: 0..self.processed_tokens,
            capacity: self.capacity,
        }
    }
    fn clone_box(&self) -> Box<dyn LayerCache> {
        Box::new(self.clone())
    }
    fn arrays(&self) -> (&Array, &Array) {
        (&self.buffer.keys, &self.buffer.values)
    }
    fn logical_arrays(&self) -> Result<(Array, Array), CacheError> {
        self.buffer.slice(0..self.processed_tokens)
    }
    fn fingerprint(&self) -> LayerFingerprint {
        LayerFingerprint {
            spec: self.spec.clone(),
            kind: CacheKind::Full,
            capacity: self.capacity,
            keep_prefix: 0,
            processed_tokens: self.processed_tokens,
            length: self.processed_tokens,
            index: self.processed_tokens,
            keys_shape: self.buffer.keys.shape().to_vec(),
            values_shape: self.buffer.values.shape().to_vec(),
            keys_dtype: self.buffer.keys.dtype(),
            values_dtype: self.buffer.values.dtype(),
        }
    }
}
