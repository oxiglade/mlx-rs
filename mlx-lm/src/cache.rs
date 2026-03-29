use mlx_rs::{
    error::Exception,
    ops::{concatenate_axis, indexing::IndexOp},
    transforms::eval,
    Array,
};

// TODO: somehow move quantized methods to a separate trait?
pub trait KeyValueCache {
    fn is_quantized(&self) -> bool {
        false
    }

    /// Returns the group size used for quantization. `None` if not quantized.
    fn group_size(&self) -> Option<i32> {
        None
    }

    /// Returns the number of bits used for quantization. `None` if not quantized.
    fn bits(&self) -> Option<i32> {
        None
    }

    fn offset(&self) -> i32;

    fn max_size(&self) -> Option<i32>;

    fn update_and_fetch(&mut self, keys: Array, values: Array)
        -> Result<(Array, Array), Exception>;
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn is_quantized(&self) -> bool {
        T::is_quantized(self)
    }

    fn group_size(&self) -> Option<i32> {
        T::group_size(self)
    }

    fn bits(&self) -> Option<i32> {
        T::bits(self)
    }

    fn offset(&self) -> i32 {
        T::offset(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        T::update_and_fetch(self, keys, values)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ConcatKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl ConcatKeyValueCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn trim_to(&mut self, token_count: i32) -> Result<(), Exception> {
        let token_count = token_count.max(0);
        match (&self.keys, &self.values) {
            (Some(keys), Some(values)) => {
                let current = keys.shape()[keys.shape().len() - 2];
                if token_count >= current {
                    self.offset = current;
                    return Ok(());
                }

                let trimmed_keys = slice_kv_prefix(keys, token_count)?;
                let trimmed_values = slice_kv_prefix(values, token_count)?;
                eval([&trimmed_keys, &trimmed_values])?;
                let trimmed_keys = trimmed_keys.deep_clone();
                let trimmed_values = trimmed_values.deep_clone();

                self.keys = Some(trimmed_keys);
                self.values = Some(trimmed_values);
                self.offset = token_count;
                Ok(())
            }
            _ => {
                self.offset = 0;
                Ok(())
            }
        }
    }

    pub fn trimmed_to(&self, token_count: i32) -> Result<Self, Exception> {
        let mut clone = self.clone();
        clone.trim_to(token_count)?;
        Ok(clone)
    }
}

fn slice_kv_prefix(array: &Array, token_count: i32) -> Result<Array, Exception> {
    match array.shape().len() {
        4 => Ok(array.index((.., .., ..token_count, ..))),
        3 => Ok(array.index((.., ..token_count, ..))),
        other => Err(Exception::custom(format!(
            "unsupported KV cache rank {other} for prefix trim"
        ))),
    }
}

impl KeyValueCache for ConcatKeyValueCache {
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
        match (self.keys.take(), self.values.take()) {
            (Some(k), Some(v)) => {
                self.keys = Some(concatenate_axis(&[k, keys], -2)?);
                self.values = Some(concatenate_axis(&[v, values], -2)?);
            }
            _ => {
                self.keys = Some(keys);
                self.values = Some(values);
            }
        }
        let shape = self.keys.as_ref().expect("Keys cannot be None").shape();
        self.offset = shape[shape.len() - 2];

        Ok((
            self.keys.clone().expect("Keys cannot be None"),
            self.values.clone().expect("Values cannot be None"),
        ))
    }
}

/// TODO: A generic KV Cache
pub struct DefaultKeyValueCache {}

#[cfg(test)]
mod tests {
    use super::{ConcatKeyValueCache, KeyValueCache};
    use mlx_rs::Array;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    fn test_guard() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("cache test lock poisoned")
    }

    #[test]
    fn trimmed_cache_keeps_prefix_and_offset() {
        let _guard = test_guard();
        let mut cache = ConcatKeyValueCache::new();
        let keys = Array::from_slice(&[1f32, 2., 3., 4., 5., 6.], &[1, 1, 3, 2]);
        let values = Array::from_slice(&[7f32, 8., 9., 10., 11., 12.], &[1, 1, 3, 2]);
        let _ = cache.update_and_fetch(keys, values).expect("seed cache");

        let trimmed = cache.trimmed_to(2).expect("trimmed clone");
        assert_eq!(trimmed.offset(), 2);

        let mut trimmed = trimmed;
        let append_keys = Array::from_slice(&[13f32, 14.], &[1, 1, 1, 2]);
        let append_values = Array::from_slice(&[15f32, 16.], &[1, 1, 1, 2]);
        let (keys, values) = trimmed
            .update_and_fetch(append_keys, append_values)
            .expect("append after trim");

        assert_eq!(keys.shape(), &[1, 1, 3, 2]);
        assert_eq!(values.shape(), &[1, 1, 3, 2]);
        assert_eq!(keys.as_slice::<f32>(), &[1., 2., 3., 4., 13., 14.]);
        assert_eq!(values.as_slice::<f32>(), &[7., 8., 9., 10., 15., 16.]);
    }

    #[test]
    fn trim_to_larger_offset_is_noop() {
        let _guard = test_guard();
        let mut cache = ConcatKeyValueCache::new();
        let keys = Array::from_slice(&[1f32, 2., 3., 4.], &[1, 1, 2, 2]);
        let values = Array::from_slice(&[5f32, 6., 7., 8.], &[1, 1, 2, 2]);
        let _ = cache.update_and_fetch(keys, values).expect("seed cache");

        cache.trim_to(8).expect("trim beyond end");
        assert_eq!(cache.offset(), 2);
    }

    #[test]
    fn trimmed_to_does_not_mutate_original_cache() {
        let _guard = test_guard();
        let mut cache = ConcatKeyValueCache::new();
        let keys = Array::from_slice(&[1f32, 2., 3., 4., 5., 6.], &[1, 1, 3, 2]);
        let values = Array::from_slice(&[7f32, 8., 9., 10., 11., 12.], &[1, 1, 3, 2]);
        let _ = cache.update_and_fetch(keys, values).expect("seed cache");

        let trimmed = cache.trimmed_to(2).expect("trimmed clone");

        assert_eq!(cache.offset(), 3);
        assert_eq!(trimmed.offset(), 2);

        let mut original = cache;
        let mut trimmed = trimmed;
        let append_keys = Array::from_slice(&[13f32, 14.], &[1, 1, 1, 2]);
        let append_values = Array::from_slice(&[15f32, 16.], &[1, 1, 1, 2]);

        let (original_keys, _) = original
            .update_and_fetch(append_keys.deep_clone(), append_values.deep_clone())
            .expect("append original");
        let (trimmed_keys, _) = trimmed
            .update_and_fetch(append_keys, append_values)
            .expect("append trimmed");

        assert_eq!(original_keys.as_slice::<f32>(), &[1., 2., 3., 4., 5., 6., 13., 14.]);
        assert_eq!(trimmed_keys.as_slice::<f32>(), &[1., 2., 3., 4., 13., 14.]);
    }
}
