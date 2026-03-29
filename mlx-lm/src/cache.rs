use mlx_rs::{
    error::Exception,
    ops::{
        concatenate_axis,
        indexing::{IndexOp, TryIndexMutOp},
        zeros_dtype,
    },
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

#[derive(Debug, Clone, Default)]
pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl KVCache {
    const STEP: i32 = 256;

    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn arrays(&self) -> (Option<&Array>, Option<&Array>) {
        (self.keys.as_ref(), self.values.as_ref())
    }
}

impl ConcatKeyValueCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn arrays(&self) -> (Option<&Array>, Option<&Array>) {
        (self.keys.as_ref(), self.values.as_ref())
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
        let prev = self.offset;
        let seq_len = keys.shape()[2];

        if self.keys.is_none() && self.values.is_none() {
            self.offset = seq_len;
            self.keys = Some(keys.deep_clone());
            self.values = Some(values.deep_clone());
            return Ok((keys, values));
        }

        let needs_resize = self
            .keys
            .as_ref()
            .map(|cached| (prev + seq_len) > cached.shape()[2])
            .unwrap_or(true);

        let mut wrote_current = false;
        if needs_resize {
            let key_shape = keys.shape();
            let value_shape = values.shape();
            let expand_steps = ((Self::STEP + seq_len - 1) / Self::STEP) * Self::STEP;
            let total_capacity = prev + expand_steps;
            let mut zero_pad_keys = zeros_dtype(
                &[key_shape[0], key_shape[1], expand_steps - seq_len, key_shape[3]],
                keys.dtype(),
            )?;
            let mut zero_pad_values = zeros_dtype(
                &[value_shape[0], value_shape[1], expand_steps - seq_len, value_shape[3]],
                values.dtype(),
            )?;

            match (self.keys.take(), self.values.take()) {
                (Some(existing_keys), Some(existing_values)) => {
                    let existing_keys = if prev < existing_keys.shape()[2] {
                        existing_keys.index((.., .., ..prev, ..))
                    } else {
                        existing_keys
                    };
                    let existing_values = if prev < existing_values.shape()[2] {
                        existing_values.index((.., .., ..prev, ..))
                    } else {
                        existing_values
                    };
                    self.keys = Some(concatenate_axis(
                        &[existing_keys, keys.deep_clone(), zero_pad_keys],
                        -2,
                    )?);
                    self.values = Some(concatenate_axis(
                        &[existing_values, values.deep_clone(), zero_pad_values],
                        -2,
                    )?);
                    wrote_current = true;
                }
                _ => {
                    let mut new_keys = zeros_dtype(
                        &[key_shape[0], key_shape[1], total_capacity, key_shape[3]],
                        keys.dtype(),
                    )?;
                    let mut new_values = zeros_dtype(
                        &[value_shape[0], value_shape[1], total_capacity, value_shape[3]],
                        values.dtype(),
                    )?;
                    new_keys.try_index_mut((.., .., ..seq_len, ..), &keys)?;
                    new_values.try_index_mut((.., .., ..seq_len, ..), &values)?;
                    eval([&new_keys, &new_values])?;
                    self.keys = Some(new_keys);
                    self.values = Some(new_values);
                    wrote_current = true;
                }
            }
        }

        self.offset += seq_len;
        let end = self.offset;
        if !wrote_current {
            self.keys
                .as_mut()
                .expect("keys cache missing")
                .try_index_mut((.., .., prev..end, ..), &keys)?;
            self.values
                .as_mut()
                .expect("values cache missing")
                .try_index_mut((.., .., prev..end, ..), &values)?;
        }
        eval([
            self.keys.as_ref().expect("keys cache missing"),
            self.values.as_ref().expect("values cache missing"),
        ])?;

        let keys = self
            .keys
            .as_ref()
            .expect("keys cache missing")
            .index((.., .., ..end, ..));
        let values = self
            .values
            .as_ref()
            .expect("values cache missing")
            .index((.., .., ..end, ..));
        eval([&keys, &values])?;
        Ok((keys.deep_clone(), values.deep_clone()))
    }
}

/// TODO: A generic KV Cache
pub struct DefaultKeyValueCache {}

#[cfg(test)]
mod tests {
    use super::{ConcatKeyValueCache, KVCache, KeyValueCache};
    use mlx_rs::{
        ops::{concatenate_axis, indexing::{IndexOp, TryIndexMutOp}, zeros_dtype},
        Array,
    };
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

        assert_eq!(
            original_keys.as_slice::<f32>(),
            &[1., 2., 3., 4., 5., 6., 13., 14.]
        );
        assert_eq!(trimmed_keys.as_slice::<f32>(), &[1., 2., 3., 4., 13., 14.]);
    }

    #[test]
    fn kv_cache_appends_with_preallocated_capacity() {
        let _guard = test_guard();
        let mut cache = KVCache::new();
        let keys = Array::from_slice(&[1f32, 2., 3., 4.], &[1, 1, 2, 2]);
        let values = Array::from_slice(&[5f32, 6., 7., 8.], &[1, 1, 2, 2]);
        let (keys, values) = cache.update_and_fetch(keys, values).expect("seed cache");
        assert_eq!(cache.offset(), 2);
        assert_eq!(keys.shape(), &[1, 1, 2, 2]);
        assert_eq!(values.shape(), &[1, 1, 2, 2]);

        let append_keys = Array::from_slice(&[9f32, 10.], &[1, 1, 1, 2]);
        let append_values = Array::from_slice(&[11f32, 12.], &[1, 1, 1, 2]);
        let (keys, values) = cache
            .update_and_fetch(append_keys, append_values)
            .expect("append cache");
        assert_eq!(cache.offset(), 3);
        assert_eq!(keys.shape(), &[1, 1, 3, 2]);
        assert_eq!(values.shape(), &[1, 1, 3, 2]);
        assert_eq!(keys.as_slice::<f32>(), &[1., 2., 3., 4., 9., 10.]);
        assert_eq!(values.as_slice::<f32>(), &[5., 6., 7., 8., 11., 12.]);
    }

    #[test]
    fn kv_cache_matches_concat_for_repeated_single_token_appends() {
        let _guard = test_guard();
        let mut concat = ConcatKeyValueCache::new();
        let mut kv = KVCache::new();

        let prefix_len = 220usize;
        let heads = 2usize;
        let dim = 64usize;
        let prefix_elems = heads * prefix_len * dim;
        let prefix_keys: Vec<f32> = (0..prefix_elems).map(|i| i as f32 / 1000.0).collect();
        let prefix_values: Vec<f32> = (0..prefix_elems).map(|i| i as f32 / 2000.0).collect();
        let prefix_keys = Array::from_slice(&prefix_keys, &[1, heads as i32, prefix_len as i32, dim as i32]);
        let prefix_values =
            Array::from_slice(&prefix_values, &[1, heads as i32, prefix_len as i32, dim as i32]);

        let (concat_keys, concat_values) = concat
            .update_and_fetch(prefix_keys.deep_clone(), prefix_values.deep_clone())
            .expect("seed concat");
        let (kv_keys, kv_values) = kv
            .update_and_fetch(prefix_keys, prefix_values)
            .expect("seed kv");
        let seed_key_match = concat_keys
            .all_close(&kv_keys, 1e-6, 1e-6, None)
            .unwrap()
            .item::<bool>();
        let seed_value_match = concat_values
            .all_close(&kv_values, 1e-6, 1e-6, None)
            .unwrap()
            .item::<bool>();
        if !seed_key_match || !seed_value_match {
            let key_diff = (concat_keys.subtract(&kv_keys).unwrap())
                .abs()
                .unwrap()
                .max(false)
                .unwrap()
                .item::<f32>();
            let value_diff = (concat_values.subtract(&kv_values).unwrap())
                .abs()
                .unwrap()
                .max(false)
                .unwrap()
                .item::<f32>();
            panic!(
                "seed mismatch: concat_shape={:?} kv_shape={:?} key_diff={key_diff} value_diff={value_diff}",
                concat_keys.shape(),
                kv_keys.shape()
            );
        }

        for step in 0..8usize {
            let token_keys: Vec<f32> = (0..(heads * dim))
                .map(|i| (10_000 + step * heads * dim + i) as f32 / 1000.0)
                .collect();
            let token_values: Vec<f32> = (0..(heads * dim))
                .map(|i| (20_000 + step * heads * dim + i) as f32 / 1000.0)
                .collect();
            let token_keys =
                Array::from_slice(&token_keys, &[1, heads as i32, 1, dim as i32]);
            let token_values =
                Array::from_slice(&token_values, &[1, heads as i32, 1, dim as i32]);

            let (concat_keys, concat_values) = concat
                .update_and_fetch(token_keys.deep_clone(), token_values.deep_clone())
                .expect("append concat");
            let (kv_keys, kv_values) = kv
                .update_and_fetch(token_keys, token_values)
                .expect("append kv");

            let key_match = concat_keys
                .all_close(&kv_keys, 1e-6, 1e-6, None)
                .expect("compare key arrays")
                .item::<bool>();
            let value_match = concat_values
                .all_close(&kv_values, 1e-6, 1e-6, None)
                .expect("compare value arrays")
                .item::<bool>();

            if !key_match || !value_match {
                let concat_key_slice = concat_keys.as_slice::<f32>();
                let kv_key_slice = kv_keys.as_slice::<f32>();
                let compare_from = concat_key_slice.len().saturating_sub(16);
                let concat_tail = &concat_key_slice[compare_from..];
                let kv_tail = &kv_key_slice[compare_from..];
                let key_diff = (concat_keys.subtract(&kv_keys).unwrap())
                    .abs()
                    .unwrap()
                    .max(false)
                    .unwrap()
                    .item::<f32>();
                let value_diff = (concat_values.subtract(&kv_values).unwrap())
                    .abs()
                    .unwrap()
                    .max(false)
                    .unwrap()
                    .item::<f32>();
                panic!(
                    "kv cache diverged from concat at step {step}: key_diff={key_diff} value_diff={value_diff} concat_tail={concat_tail:?} kv_tail={kv_tail:?}"
                );
            }
        }
    }

    #[test]
    fn direct_index_mut_large_prefix_then_single_append_preserves_tail() {
        let _guard = test_guard();
        let heads = 2usize;
        let dim = 64usize;
        let prefix_len = 220usize;
        let total_capacity = 476usize;

        let prefix_keys: Vec<f32> = (0..(heads * prefix_len * dim))
            .map(|i| i as f32 / 1000.0)
            .collect();
        let token_keys: Vec<f32> = (0..(heads * dim))
            .map(|i| (10_000 + i) as f32 / 1000.0)
            .collect();

        let prefix_keys =
            Array::from_slice(&prefix_keys, &[1, heads as i32, prefix_len as i32, dim as i32]);
        let token_keys = Array::from_slice(&token_keys, &[1, heads as i32, 1, dim as i32]);
        let mut buffer = zeros_dtype(
            &[1, heads as i32, total_capacity as i32, dim as i32],
            prefix_keys.dtype(),
        )
        .unwrap();

        buffer
            .try_index_mut((.., .., ..prefix_len as i32, ..), &prefix_keys)
            .unwrap();
        buffer
            .try_index_mut(
                (.., .., prefix_len as i32..(prefix_len as i32 + 1), ..),
                &token_keys,
            )
            .unwrap();
        mlx_rs::transforms::eval([&buffer]).unwrap();

        let live = buffer.index((.., .., ..(prefix_len as i32 + 1), ..));
        mlx_rs::transforms::eval([&live]).unwrap();
        let live_tail = &live.as_slice::<f32>()[live.as_slice::<f32>().len() - 16..];
        let token_tail = &token_keys.as_slice::<f32>()[token_keys.as_slice::<f32>().len() - 16..];
        assert_eq!(live_tail, token_tail);
    }

    #[test]
    fn direct_concat_prefix_token_zeropad_preserves_tail() {
        let _guard = test_guard();
        let heads = 2usize;
        let dim = 64usize;
        let prefix_len = 220usize;
        let pad_len = 255usize;

        let prefix_keys: Vec<f32> = (0..(heads * prefix_len * dim))
            .map(|i| i as f32 / 1000.0)
            .collect();
        let token_keys: Vec<f32> = (0..(heads * dim))
            .map(|i| (10_000 + i) as f32 / 1000.0)
            .collect();

        let prefix_keys =
            Array::from_slice(&prefix_keys, &[1, heads as i32, prefix_len as i32, dim as i32]);
        let token_keys = Array::from_slice(&token_keys, &[1, heads as i32, 1, dim as i32]);
        let zero_pad = zeros_dtype(
            &[1, heads as i32, pad_len as i32, dim as i32],
            prefix_keys.dtype(),
        )
        .unwrap();

        let buffer = concatenate_axis(&[prefix_keys, token_keys.deep_clone(), zero_pad], -2).unwrap();
        mlx_rs::transforms::eval([&buffer]).unwrap();
        let live = buffer.index((.., .., ..(prefix_len as i32 + 1), ..));
        mlx_rs::transforms::eval([&live]).unwrap();
        let live_tail = &live.as_slice::<f32>()[live.as_slice::<f32>().len() - 16..];
        let token_tail = &token_keys.as_slice::<f32>()[token_keys.as_slice::<f32>().len() - 16..];
        assert_eq!(live_tail, token_tail);
    }

    #[test]
    fn direct_concat_prefix_and_token_preserves_tail() {
        let _guard = test_guard();
        let heads = 2usize;
        let dim = 64usize;
        let prefix_len = 220usize;

        let prefix_keys: Vec<f32> = (0..(heads * prefix_len * dim))
            .map(|i| i as f32 / 1000.0)
            .collect();
        let token_keys: Vec<f32> = (0..(heads * dim))
            .map(|i| (10_000 + i) as f32 / 1000.0)
            .collect();

        let prefix_keys =
            Array::from_slice(&prefix_keys, &[1, heads as i32, prefix_len as i32, dim as i32]);
        let token_keys = Array::from_slice(&token_keys, &[1, heads as i32, 1, dim as i32]);
        let buffer = concatenate_axis(&[prefix_keys, token_keys.deep_clone()], -2).unwrap();
        mlx_rs::transforms::eval([&buffer]).unwrap();
        let live_tail = &buffer.as_slice::<f32>()[buffer.as_slice::<f32>().len() - 16..];
        let token_tail = &token_keys.as_slice::<f32>()[token_keys.as_slice::<f32>().len() - 16..];
        assert_eq!(live_tail, token_tail);
    }
}
