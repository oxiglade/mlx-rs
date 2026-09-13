use super::{
    concatenate, invalid, sealed, slice, validate, validate_rotation, Buffer, CacheError,
    CacheInfo, CacheKind, LayerCache, LayerCacheSpec, LayerFingerprint,
};
use mlx_rs::Array;

#[derive(Clone)]
pub(crate) struct RotatingCache {
    spec: LayerCacheSpec,
    buffer: Buffer,
    processed_tokens: usize,
    capacity: usize,
    keep_prefix: usize,
    length: usize,
    index: usize,
}
impl RotatingCache {
    pub(crate) fn new(
        spec: &LayerCacheSpec,
        capacity: usize,
        keep_prefix: usize,
    ) -> Result<Self, CacheError> {
        validate_rotation(capacity, keep_prefix)?;
        Ok(Self {
            spec: spec.clone(),
            buffer: Buffer::new(spec, capacity)?,
            processed_tokens: 0,
            capacity,
            keep_prefix,
            length: 0,
            index: 0,
        })
    }
    #[cfg(test)]
    pub(crate) fn trim(&mut self, count: usize) -> Result<usize, CacheError> {
        if count == 0 {
            return Ok(0);
        }
        // Once rotation starts, evicted positions cannot be recovered by moving the cursor.
        if self.processed_tokens >= self.capacity {
            return Err(invalid("rotating cache is no longer trimmable"));
        }
        let count = count.min(self.processed_tokens);
        let index = self
            .index
            .checked_sub(count)
            .ok_or_else(|| invalid("invalid trim cursor"))?;
        let length = self
            .length
            .checked_sub(count)
            .ok_or_else(|| invalid("invalid trim length"))?;
        self.processed_tokens -= count;
        self.index = index;
        self.length = length;
        Ok(count)
    }
    fn temporal(&self, array: &Array) -> Result<Array, CacheError> {
        if self.index == self.length {
            return slice(array, 0..self.length);
        }
        Ok(concatenate(
            &[
                slice(array, 0..self.keep_prefix)?,
                slice(array, self.index..self.length)?,
                slice(array, self.keep_prefix..self.index)?,
            ],
            2,
        )?)
    }
    fn trim_array(&self, array: &Array, trim: usize) -> Result<Array, CacheError> {
        if trim == 0 {
            return Ok(array.clone());
        }
        Ok(concatenate(
            &[
                slice(array, 0..self.keep_prefix.min(self.length))?,
                slice(
                    array,
                    (self.keep_prefix + trim).min(self.length)..self.length,
                )?,
            ],
            2,
        )?)
    }
    fn update_concat(
        &mut self,
        keys: Array,
        values: Array,
        count: usize,
    ) -> Result<(), CacheError> {
        let (keys, values) = if self.length == 0 {
            (keys, values)
        } else {
            let trim = self.length.saturating_sub(self.capacity - 1);
            let previous_keys = self.trim_array(&self.temporal(&self.buffer.keys)?, trim)?;
            let previous_values = self.trim_array(&self.temporal(&self.buffer.values)?, trim)?;
            (
                concatenate(&[previous_keys, keys], 2)?,
                concatenate(&[previous_values, values], 2)?,
            )
        };
        let length = super::token_length(&keys)?;
        // The oracle exposes up to capacity + chunk - 1 positions to chunked attention.
        self.buffer = if length > self.capacity {
            Buffer { keys, values }
        } else {
            let buffer = if super::token_length(&self.buffer.keys)? == self.capacity {
                self.buffer.clone()
            } else {
                Buffer::new(&self.spec, self.capacity)?
            };
            buffer.replace(0, &keys, &values)?
        };
        self.length = length;
        self.index = length;
        self.processed_tokens = self
            .processed_tokens
            .checked_add(count)
            .ok_or_else(|| invalid("position overflow"))?;
        Ok(())
    }
    fn update_single(&mut self, keys: Array, values: Array) -> Result<(), CacheError> {
        if self.length > self.capacity {
            let trim = self.length - self.capacity;
            self.buffer = Buffer {
                keys: self.trim_array(&self.buffer.keys, trim)?,
                values: self.trim_array(&self.buffer.values, trim)?,
            };
            self.length = self.capacity;
            self.index = self.capacity;
        }
        if self.index == self.capacity {
            self.index = self.keep_prefix;
        }
        self.buffer = self.buffer.replace(self.index, &keys, &values)?;
        self.index += 1;
        self.length = self.capacity.min(self.length + 1);
        self.processed_tokens = self
            .processed_tokens
            .checked_add(1)
            .ok_or_else(|| invalid("position overflow"))?;
        Ok(())
    }
}
impl sealed::Sealed for RotatingCache {}
impl LayerCache for RotatingCache {
    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), CacheError> {
        let count = validate(&self.spec, &keys, &values)?;
        let mut staged = self.clone();
        if count == 1 {
            staged.update_single(keys, values)?;
        } else {
            staged.update_concat(keys, values, count)?;
        }
        let logical = (
            staged.temporal(&staged.buffer.keys)?,
            staged.temporal(&staged.buffer.values)?,
        );
        *self = staged;
        Ok(logical)
    }
    fn info(&self, layer: usize) -> CacheInfo {
        let prefix = self.keep_prefix.min(self.length);
        CacheInfo {
            layer,
            kind: CacheKind::Rotating,
            processed_tokens: self.processed_tokens,
            retained_prefix: 0..prefix,
            retained_positions: self.processed_tokens - (self.length - prefix)
                ..self.processed_tokens,
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
        Ok((
            self.temporal(&self.buffer.keys)?,
            self.temporal(&self.buffer.values)?,
        ))
    }
    fn fingerprint(&self) -> LayerFingerprint {
        LayerFingerprint {
            spec: self.spec.clone(),
            kind: CacheKind::Rotating,
            capacity: self.capacity,
            keep_prefix: self.keep_prefix,
            processed_tokens: self.processed_tokens,
            length: self.length,
            index: self.index,
            keys_shape: self.buffer.keys.shape().to_vec(),
            values_shape: self.buffer.values.shape().to_vec(),
            keys_dtype: self.buffer.keys.dtype(),
            values_dtype: self.buffer.values.dtype(),
        }
    }
}
