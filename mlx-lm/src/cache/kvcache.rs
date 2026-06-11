//! [`KVCache`]: the default pre-allocated, step-grown KV cache.

use mlx_rs::{
    error::Exception,
    ops::{
        indexing::{Ellipsis, IndexOp, TryIndexMutOp},
        zeros_dtype,
    },
    Array,
};

use super::trait_def::KeyValueCache;

/// Initial KV buffer capacity. Doubles geometrically on overflow.
pub const DEFAULT_KV_CACHE_INIT_CAPACITY: i32 = 64;

/// Pre-allocated, geometrically-grown KV cache.
///
/// Holds `[B, H, capacity, D]` K/V buffers. First call allocates
/// `init_capacity` tokens; overflows double the buffer. Returns
/// graph-view slices over the populated `[..offset]` range, so no
/// per-step `concatenate_axis`.
#[derive(Debug, Clone)]
pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
    init_capacity: i32,
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KVCache {
    pub fn new() -> Self {
        Self::with_init_capacity(DEFAULT_KV_CACHE_INIT_CAPACITY)
    }

    pub fn with_init_capacity(init_capacity: i32) -> Self {
        assert!(init_capacity > 0, "KVCache init_capacity must be positive");
        Self {
            keys: None,
            values: None,
            offset: 0,
            init_capacity,
        }
    }

    pub fn init_capacity(&self) -> i32 {
        self.init_capacity
    }

    pub fn capacity(&self) -> i32 {
        self.keys
            .as_ref()
            .map(|k| k.shape()[k.shape().len() - 2])
            .unwrap_or(0)
    }

    fn alloc_like(template: &Array, capacity: i32) -> Result<Array, Exception> {
        let mut buf_shape = template.shape().to_vec();
        let t_axis = buf_shape.len() - 2;
        buf_shape[t_axis] = capacity;
        zeros_dtype(&buf_shape, template.dtype())
    }

    fn grow_to_fit(&mut self, new_keys: &Array, new_values: &Array) -> Result<(), Exception> {
        let new_shape = new_keys.shape();
        let s = new_shape[new_shape.len() - 2];
        let required = self.offset + s;
        let current_cap = self.capacity();
        if required <= current_cap {
            return Ok(());
        }

        let mut target_cap = if current_cap == 0 {
            self.init_capacity.max(required)
        } else {
            current_cap
        };
        while target_cap < required {
            target_cap *= 2;
        }

        let mut grown_k = Self::alloc_like(new_keys, target_cap)?;
        let mut grown_v = Self::alloc_like(new_values, target_cap)?;

        if let (Some(old_k), Some(old_v)) = (self.keys.take(), self.values.take()) {
            if self.offset > 0 {
                grown_k.try_index_mut(
                    (Ellipsis, 0..self.offset, ..),
                    old_k.index((Ellipsis, 0..self.offset, ..)),
                )?;
                grown_v.try_index_mut(
                    (Ellipsis, 0..self.offset, ..),
                    old_v.index((Ellipsis, 0..self.offset, ..)),
                )?;
            }
        }

        self.keys = Some(grown_k);
        self.values = Some(grown_v);
        Ok(())
    }
}

impl KeyValueCache for KVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let key_shape = keys.shape();
        let s = key_shape[key_shape.len() - 2];

        self.grow_to_fit(&keys, &values)?;

        let buf_k = self.keys.as_mut().expect("allocated by grow_to_fit");
        let buf_v = self.values.as_mut().expect("allocated by grow_to_fit");

        buf_k.try_index_mut((Ellipsis, self.offset..self.offset + s, ..), keys)?;
        buf_v.try_index_mut((Ellipsis, self.offset..self.offset + s, ..), values)?;

        self.offset += s;

        let end = self.offset;
        Ok((
            buf_k.index((Ellipsis, 0..end, ..)),
            buf_v.index((Ellipsis, 0..end, ..)),
        ))
    }

    fn current_kv(&self) -> Result<Option<(Array, Array)>, Exception> {
        if self.offset == 0 {
            return Ok(None);
        }
        let (buf_k, buf_v) = match (self.keys.as_ref(), self.values.as_ref()) {
            (Some(k), Some(v)) => (k, v),
            _ => return Ok(None),
        };
        let end = self.offset;
        Ok(Some((
            buf_k.index((Ellipsis, 0..end, ..)),
            buf_v.index((Ellipsis, 0..end, ..)),
        )))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    use super::*;
    use mlx_rs::ops::all_close;

    fn tok(v: f32) -> Array {
        // [B=1, H=1, S=1, D=2]
        Array::from_slice(&[v, v], &[1, 1, 1, 2])
    }

    #[test]
    fn current_kv_none_when_empty() {
        let c = KVCache::new();
        assert!(c.current_kv().unwrap().is_none());
    }

    #[test]
    fn current_kv_matches_fetch_without_advancing() {
        let mut c = KVCache::new();
        c.update_and_fetch(tok(1.0), tok(-1.0)).unwrap();
        c.update_and_fetch(tok(2.0), tok(-2.0)).unwrap();
        let off_before = c.offset();

        let (k, v) = c.current_kv().unwrap().expect("non-empty");
        // Offset unchanged — current_kv is read-only.
        assert_eq!(c.offset(), off_before);
        assert_eq!(k.shape(), &[1, 1, 2, 2]);

        // Same tensors a zero-length fetch would return.
        let (k_ref, v_ref) = c
            .update_and_fetch(
                Array::zeros::<f32>(&[1, 1, 0, 2]).unwrap(),
                Array::zeros::<f32>(&[1, 1, 0, 2]).unwrap(),
            )
            .unwrap();
        assert!(all_close(&k, &k_ref, None, None, None)
            .unwrap()
            .item::<bool>());
        assert!(all_close(&v, &v_ref, None, None, None)
            .unwrap()
            .item::<bool>());
    }
}
