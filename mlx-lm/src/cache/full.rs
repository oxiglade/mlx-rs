use super::{
    grown_capacity, invalid, sealed, shape, validate, Buffer, CacheError, CacheInfo, CacheKind,
    LayerCache, LayerCacheSpec, LayerFingerprint,
};
use mlx_rs::Array;

#[derive(Clone)]
pub(crate) struct FullCache {
    spec: LayerCacheSpec,
    buffer: Option<Buffer>,
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
            buffer: Some(Buffer::new(spec, capacity)?),
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
        let previous = self.buffer.as_ref().unwrap();
        let stream = mlx_rs::Stream::thread_local_or_default();
        let buffer = if matches!(stream.device()?.get_type()?, mlx_rs::DeviceType::Gpu) {
            Buffer {
                keys: super::append::append(&previous.keys, &keys, self.processed_tokens, &stream)?,
                values: super::append::append(
                    &previous.values,
                    &values,
                    self.processed_tokens,
                    &stream,
                )?,
            }
        } else {
            previous.replace(self.processed_tokens, &keys, &values)?
        };
        let logical = buffer.slice(0..end)?;
        self.buffer = Some(buffer);
        self.processed_tokens = end;
        Ok(logical)
    }
    fn reserve(&mut self, required: usize) -> Result<(), CacheError> {
        let capacity = grown_capacity(self.capacity, required)?;
        if capacity != self.capacity {
            let (keys, values) = self
                .buffer
                .as_ref()
                .unwrap()
                .slice(0..self.processed_tokens)?;
            self.buffer = Some(Buffer::new(&self.spec, capacity)?.replace(0, &keys, &values)?);
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
    fn snapshot_box(&self) -> Result<Box<dyn LayerCache>, CacheError> {
        let (keys, values) = self.logical_arrays()?;
        Ok(Box::new(Self {
            spec: self.spec.clone(),
            buffer: Some(Buffer::new(&self.spec, self.capacity)?.replace(0, &keys, &values)?),
            processed_tokens: self.processed_tokens,
            capacity: self.capacity,
        }))
    }
    fn rollback(&mut self, staged: &mut dyn LayerCache) {
        if self.buffer.is_none() {
            self.capacity = staged.info(0).capacity;
            self.buffer = staged.take_buffer();
        }
    }
    fn set_length(&mut self, length: usize) -> Result<(), CacheError> {
        if length > self.capacity {
            return Err(invalid("length exceeds full cache capacity"));
        }
        self.processed_tokens = length;
        Ok(())
    }
    fn take_buffer(&mut self) -> Option<Buffer> {
        self.buffer.take()
    }
    fn arrays(&self) -> Result<(Array, Array), CacheError> {
        let buffer = self.buffer.as_ref().unwrap();
        Ok((buffer.keys.clone(), buffer.values.clone()))
    }
    fn logical_arrays(&self) -> Result<(Array, Array), CacheError> {
        self.buffer
            .as_ref()
            .unwrap()
            .slice(0..self.processed_tokens)
    }
    fn fingerprint(&self) -> LayerFingerprint {
        let shape = shape(&self.spec, self.capacity).expect("validated cache shape");
        LayerFingerprint {
            spec: self.spec.clone(),
            kind: CacheKind::Full,
            capacity: self.capacity,
            keep_prefix: 0,
            processed_tokens: self.processed_tokens,
            length: self.processed_tokens,
            index: self.processed_tokens,
            keys_shape: shape.to_vec(),
            values_shape: shape.to_vec(),
            keys_dtype: self.spec.dtype,
            values_dtype: self.spec.dtype,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{Cache, CacheOptions};

    #[test]
    fn snapshot_failure_only_invalidates_failed_source_graphs(
    ) -> Result<(), Box<dyn std::error::Error>> {
        mlx_rs::with_device(
            mlx_rs::Device::cpu(),
            || -> Result<(), Box<dyn std::error::Error>> {
                let spec = LayerCacheSpec {
                    attention: crate::AttentionKind::Full,
                    batch_size: 1,
                    kv_heads: 1,
                    head_dim: 1,
                    dtype: mlx_rs::Dtype::Float32,
                };
                let mut cache = Cache::new(
                    crate::ModelType::new("llama"),
                    std::slice::from_ref(&spec),
                    &CacheOptions::default(),
                    None,
                )?;
                for failed_source in [false, true] {
                    let keys = if failed_source {
                        mlx_rs::linalg::inv(Array::from_slice(&[0.0f32], &[1, 1]))?
                            .reshape(&[1, 1, 1, 1])?
                    } else {
                        Array::from_slice(&[3.0f32], &[1, 1, 1, 1])
                    };
                    cache.layers[0] = Box::new(FullCache {
                        spec: spec.clone(),
                        buffer: Some(Buffer {
                            keys,
                            values: Array::from_slice(&[4.0f32], &[1, 1, 1, 1]),
                        }),
                        processed_tokens: 1,
                        // Force copy construction to fail without an actual oversized allocation.
                        capacity: if failed_source {
                            1
                        } else {
                            i32::MAX as usize + 1
                        },
                    });
                    assert!(cache.snapshot().is_err());
                    assert_eq!(cache.invalidated, failed_source);
                    if !failed_source {
                        let (keys, values) = cache.layers[0].logical_arrays()?;
                        assert_eq!(keys.to_vec_exact::<f32>()?, [3.]);
                        assert_eq!(values.to_vec_exact::<f32>()?, [4.]);
                    }
                }
                Ok(())
            },
        )
    }

    #[test]
    fn empty_submission_preserves_committed_buffers() -> Result<(), Box<dyn std::error::Error>> {
        for device in [mlx_rs::Device::cpu(), mlx_rs::Device::gpu()] {
            mlx_rs::with_device(device, || -> Result<(), Box<dyn std::error::Error>> {
                for capacity in [16, 512] {
                    let spec = LayerCacheSpec {
                        attention: crate::AttentionKind::Full,
                        batch_size: 1,
                        kv_heads: 1,
                        head_dim: 1,
                        dtype: mlx_rs::Dtype::Float32,
                    };
                    let mut cache = Cache::new(
                        crate::ModelType::new("llama"),
                        std::slice::from_ref(&spec),
                        &CacheOptions::default(),
                        None,
                    )?;
                    let keys = Array::from_slice(&[1.0f32; 16], &[1, 1, 16, 1]);
                    let values = Array::from_slice(&[-1.0f32; 16], &[1, 1, 16, 1]);
                    let before = (
                        keys.as_slice::<f32>().as_ptr(),
                        values.as_slice::<f32>().as_ptr(),
                    );
                    cache.layers[0] = Box::new(FullCache {
                        spec,
                        buffer: Some(Buffer { keys, values }),
                        processed_tokens: 2,
                        capacity: 16,
                    });
                    let pending = cache.ledger.prepare(&[1.into(), 2.into()])?;
                    cache.ledger.commit(pending);
                    let mut step = cache.speculative_step(2)?;
                    step.reserve(std::num::NonZeroUsize::new(capacity).unwrap())?;
                    step.update_and_fetch(
                        0,
                        Array::from_slice(&[3.0f32], &[1, 1, 1, 1]),
                        Array::from_slice(&[-3.0f32], &[1, 1, 1, 1]),
                    )?;
                    step.submit(&[])?;
                    assert_eq!(cache.layers[0].info(0).capacity, 16);
                    assert_eq!(cache.layers[0].info(0).processed_tokens, 2);
                    assert_eq!(cache.tokens()?, &[1.into(), 2.into()]);
                    let (keys, values) = cache.layers[0].arrays()?;
                    assert_eq!(keys.as_slice::<f32>(), &[1.0; 16]);
                    assert_eq!(values.as_slice::<f32>(), &[-1.0; 16]);
                    assert_eq!(
                        (
                            keys.as_slice::<f32>().as_ptr(),
                            values.as_slice::<f32>().as_ptr()
                        ),
                        before,
                    );
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    #[test]
    fn ordinary_append_donates_full_cache_buffers() -> Result<(), Box<dyn std::error::Error>> {
        for device in [mlx_rs::Device::cpu(), mlx_rs::Device::gpu()] {
            mlx_rs::with_device(device, || -> Result<(), Box<dyn std::error::Error>> {
                for mode in ["evaluate", "evaluate_outputs", "submit"] {
                    let spec = LayerCacheSpec {
                        attention: crate::AttentionKind::Full,
                        batch_size: 1,
                        kv_heads: 1,
                        head_dim: 1,
                        dtype: mlx_rs::Dtype::Float32,
                    };
                    let mut cache = Cache::new(
                        crate::ModelType::new("llama"),
                        std::slice::from_ref(&spec),
                        &CacheOptions::default(),
                        None,
                    )?;
                    let keys = Array::from_slice(&[1.0f32; 16], &[1, 1, 16, 1]);
                    let values = Array::from_slice(&[-1.0f32; 16], &[1, 1, 16, 1]);
                    let before = (
                        keys.as_slice::<f32>().as_ptr(),
                        values.as_slice::<f32>().as_ptr(),
                    );
                    cache.layers[0] = Box::new(FullCache {
                        spec,
                        buffer: Some(Buffer { keys, values }),
                        processed_tokens: 2,
                        capacity: 16,
                    });
                    let mut step = if mode == "submit" {
                        let pending = cache.ledger.prepare(&[1.into(), 2.into()])?;
                        cache.ledger.commit(pending);
                        cache.speculative_step(2)?
                    } else {
                        cache.step()?
                    };
                    let (updated_keys, updated_values) = step.update_and_fetch(
                        0,
                        Array::from_slice(&[3.0f32], &[1, 1, 1, 1]),
                        Array::from_slice(&[-3.0f32], &[1, 1, 1, 1]),
                    )?;
                    match mode {
                        "evaluate" => step.evaluate(&[])?.commit(),
                        "evaluate_outputs" => step
                            .evaluate_outputs(|arrays, evaluate| {
                                evaluate(arrays)?;
                                Ok(())
                            })?
                            .1
                            .commit(),
                        _ => step.submit(&[&updated_keys, &updated_values])?,
                    }
                    let (keys, values) = cache.layers[0].arrays()?;
                    assert_eq!(
                        (
                            keys.as_slice::<f32>().as_ptr(),
                            values.as_slice::<f32>().as_ptr()
                        ),
                        before,
                        "{mode}",
                    );
                    assert_eq!(&keys.as_slice::<f32>()[..3], &[1., 1., 3.]);
                    assert_eq!(&values.as_slice::<f32>()[..3], &[-1., -1., -3.]);
                }
                Ok(())
            })?;
        }
        Ok(())
    }
}
