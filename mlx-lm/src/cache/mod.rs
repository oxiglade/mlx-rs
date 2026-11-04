pub use crate::error::CacheError;
use mlx_rs::{
    ops::{
        concatenate,
        indexing::{IndexUpdateError, TryIndexOp, TryIndexUpdateOp, UpdateMode},
        zeros_dtype,
    },
    Array, Dtype,
};
use std::collections::BTreeMap;
use std::{marker::PhantomData, num::NonZeroUsize, ops::Range, rc::Rc};
#[cfg(mlx_lm_kv_append)]
mod append;
mod full;
mod ledger;
use ledger::{PendingTokens, TokenLedger};
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
        /// Single-token storage capacity; chunked prefill can temporarily retain more.
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
    /// Absolute count of processed input tokens.
    pub processed_tokens: usize,
    /// Retained prefix, separate from the recent range after a rotating wrap.
    pub retained_prefix: Range<usize>,
    /// Absolute recent range; excludes positions already in `retained_prefix`.
    pub retained_positions: Range<usize>,
    /// Full-cache allocation size or configured rotating capacity.
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
/// GPU full caches append in place when building MLX from source.
/// Prebuilt MLX, CPU and rotating caches use `slice_update`.
/// On an M5 Max, decode throughput is level at 128 tokens, about 7% higher at ~4k
/// and 0.4% lower at ~16k than `slice_update`.
/// Abandoned steps preserve committed tokens and K/V prefixes; spare storage may change.
/// Evaluation failure invalidates the cache. Call [`Cache::reset`] and prefill again.
pub struct Cache {
    architecture: crate::ModelType,
    model_identity: Rc<()>,
    ledger: TokenLedger,
    tracks_tokens: bool,
    invalidated: bool,
    last_submission: Option<(usize, mlx_rs::Stream)>,
    layers: Vec<Box<dyn LayerCache>>,
    _thread_bound: PhantomData<Rc<()>>,
}
/// Opaque thread-bound handles and metadata for restoring a cache.
/// Snapshots remain valid if a cache restored from them fails.
pub struct CacheSnapshot {
    architecture: crate::ModelType,
    model_identity: Rc<()>,
    ledger: TokenLedger,
    tracks_tokens: bool,
    layers: BTreeMap<usize, Box<dyn LayerCache>>,
    _thread_bound: PhantomData<Rc<()>>,
}
/// Contiguous K/V for one uninterrupted range of absolute positions.
#[cfg_attr(not(feature = "oracle-hooks"), allow(dead_code))]
pub(crate) struct LogicalLayerView {
    pub(crate) layer: usize,
    pub(crate) keys: Array,
    pub(crate) values: Array,
    pub(crate) positions: Range<usize>,
}
impl Cache {
    #[cfg_attr(not(feature = "oracle-hooks"), allow(dead_code))]
    pub(crate) fn new(
        architecture: crate::ModelType,
        layout: &[LayerCacheSpec],
        options: &CacheOptions,
        capacity: Option<NonZeroUsize>,
    ) -> Result<Self, CacheError> {
        let mut cache = Self::new_for_model(architecture, layout, options, capacity, Rc::new(()))?;
        cache.tracks_tokens = false;
        Ok(cache)
    }
    pub(crate) fn new_for_model(
        architecture: crate::ModelType,
        layout: &[LayerCacheSpec],
        options: &CacheOptions,
        capacity: Option<NonZeroUsize>,
        model_identity: Rc<()>,
    ) -> Result<Self, CacheError> {
        if layout.is_empty() {
            return Err(invalid("cache layout must contain a layer"));
        }
        let mut layers: Vec<Box<dyn LayerCache>> = Vec::with_capacity(layout.len());
        for spec in layout {
            let policy = resolve_policy(spec, &options.policy)?;
            layers.push(match policy {
                None => Box::new(full::FullCache::new(
                    spec,
                    capacity.map_or(256, NonZeroUsize::get),
                )?),
                Some((capacity, keep)) => {
                    Box::new(rotating::RotatingCache::new(spec, capacity, keep)?)
                }
            });
        }
        let mut ledger = TokenLedger::default();
        ledger.reserve(capacity.map_or(16, NonZeroUsize::get))?;
        Ok(Self {
            architecture,
            model_identity,
            ledger,
            tracks_tokens: true,
            invalidated: false,
            last_submission: None,
            layers,
            _thread_bound: PhantomData,
        })
    }
    /// Token IDs represented by committed cache state, including evicted positions.
    /// Returns [`CacheError::Invalidated`] after cache evaluation or submission fails.
    pub fn tokens(&self) -> Result<&[crate::TokenId], CacheError> {
        self.ensure_valid()?;
        Ok(self.ledger.tokens())
    }
    /// Returns logical metadata without evaluating or copying array buffers.
    /// Returns [`CacheError::Invalidated`] after cache evaluation or submission fails.
    pub fn info(&self) -> Result<impl ExactSizeIterator<Item = CacheInfo> + '_, CacheError> {
        self.ensure_valid()?;
        Ok(self
            .layers
            .iter()
            .enumerate()
            .map(|(layer, cache)| cache.info(layer)))
    }
    pub(crate) fn ensure_valid(&self) -> Result<(), CacheError> {
        if self.invalidated {
            Err(CacheError::Invalidated)
        } else {
            Ok(())
        }
    }
    pub(crate) fn invalidate(&mut self) {
        self.invalidated = true;
    }
    /// Discards all tokens and buffers, including failed graphs. Prefill again before reuse.
    pub fn reset(&mut self) -> Result<(), CacheError> {
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                let state = layer.fingerprint();
                Ok(match state.kind {
                    CacheKind::Full => Box::new(full::FullCache::new(&state.spec, state.capacity)?)
                        as Box<dyn LayerCache>,
                    CacheKind::Rotating => Box::new(rotating::RotatingCache::new(
                        &state.spec,
                        state.capacity,
                        state.keep_prefix,
                    )?),
                })
            })
            .collect::<Result<_, CacheError>>()?;
        self.layers = layers;
        self.ledger = TokenLedger::default();
        self.invalidated = false;
        self.last_submission = None;
        Ok(())
    }
    /// Returns temporal [batch, kv_heads, positions.len(), head_dim] arrays for
    /// exactly the positions `info` reports. Each layer yields its nonempty
    /// prefix followed by its tail; an empty layer yields one empty view.
    /// Views are ordered by layer, then position.
    #[cfg_attr(not(feature = "oracle-hooks"), allow(dead_code))]
    pub(crate) fn logical_layers(&self) -> Result<Vec<LogicalLayerView>, CacheError> {
        self.ensure_valid()?;
        let mut views = Vec::with_capacity(self.layers.len());
        for (layer, cache) in self.layers.iter().enumerate() {
            let info = cache.info(layer);
            let (keys, values) = cache.logical_arrays()?;
            let prefix = info.retained_prefix.len();
            if prefix > 0 {
                views.push(LogicalLayerView {
                    layer,
                    keys: slice(&keys, 0..prefix)?.contiguous()?,
                    values: slice(&values, 0..prefix)?.contiguous()?,
                    positions: info.retained_prefix.clone(),
                });
            }
            let positions = info.retained_positions.clone();
            if !positions.is_empty() || prefix == 0 {
                let range = prefix..prefix + positions.len();
                views.push(LogicalLayerView {
                    layer,
                    keys: slice(&keys, range.clone())?.contiguous()?,
                    values: slice(&values, range)?.contiguous()?,
                    positions,
                });
            }
        }
        Ok(views)
    }
    /// Evaluates source arrays, then copies committed prefixes of full layers with spare
    /// capacity into full-capacity allocations. All layer copies are evaluated together.
    /// Full layers without spare capacity and rotating layers may share storage until an
    /// update replaces it. Restore consumes the snapshot.
    /// A source evaluation failure invalidates this cache; a copy failure leaves it valid.
    pub fn snapshot(&mut self) -> Result<CacheSnapshot, CacheError> {
        self.ensure_valid()?;
        let source = self
            .layers
            .iter()
            .map(|layer| layer.arrays().map(|(keys, values)| [keys, values]))
            .collect::<Result<Vec<_>, _>>()?;
        mlx_rs::transforms::eval(source.iter().flatten()).inspect_err(|_| self.invalidate())?;
        let layers = self
            .layers
            .iter()
            .enumerate()
            .map(|(i, layer)| Ok((i, layer.snapshot_box()?)))
            .collect::<Result<BTreeMap<_, _>, CacheError>>()?;
        let copies = layers
            .values()
            .map(|layer| layer.arrays().map(|(keys, values)| [keys, values]))
            .collect::<Result<Vec<_>, _>>()?;
        mlx_rs::transforms::eval(copies.iter().flatten())?;
        Ok(CacheSnapshot {
            architecture: self.architecture.clone(),
            model_identity: Rc::clone(&self.model_identity),
            ledger: self.ledger.clone(),
            tracks_tokens: self.tracks_tokens,
            layers,
            _thread_bound: PhantomData,
        })
    }
    /// Adopts compatible snapshot storage and restores absolute-position metadata.
    /// Consumes the snapshot without copying or evaluating its arrays.
    pub fn restore(&mut self, snapshot: CacheSnapshot) -> Result<(), CacheError> {
        self.ensure_valid()?;
        if !Rc::ptr_eq(&self.model_identity, &snapshot.model_identity)
            || self.tracks_tokens != snapshot.tracks_tokens
            || self.architecture != snapshot.architecture
            || self.layers.len() != snapshot.layers.len()
        {
            return Err(CacheError::FingerprintMismatch);
        }
        for (i, current) in self.layers.iter().enumerate() {
            let saved = snapshot
                .layers
                .get(&i)
                .ok_or(CacheError::FingerprintMismatch)?;
            if !current.fingerprint().compatible(&saved.fingerprint()) {
                return Err(CacheError::FingerprintMismatch);
            }
        }
        self.layers = snapshot.layers.into_values().collect();
        self.ledger = snapshot.ledger;
        self.model_identity = snapshot.model_identity;
        self.last_submission = None;
        Ok(())
    }
    pub(crate) fn validate_reuse(
        &self,
        model_identity: &Rc<()>,
        architecture: &crate::ModelType,
        layout: &[LayerCacheSpec],
        options: &CacheOptions,
        prompt: &[crate::TokenId],
    ) -> Result<usize, crate::GenerationError> {
        self.ensure_valid()?;
        if !Rc::ptr_eq(&self.model_identity, model_identity)
            || &self.architecture != architecture
            || self.layers.len() != layout.len()
            || self
                .layers
                .iter()
                .zip(layout)
                .any(|(layer, spec)| layer.fingerprint().spec != *spec)
        {
            return Err(CacheError::FingerprintMismatch.into());
        }
        for (layer, spec) in self.layers.iter().zip(layout) {
            validate_policy(&layer.fingerprint(), spec, &options.policy)?;
        }
        self.validate_ledger()?;
        validate_prefix(self.tokens()?, prompt)
    }
    fn validate_ledger(&self) -> Result<(), CacheError> {
        self.ensure_valid()?;
        if self.layers.is_empty()
            || self
                .info()?
                .any(|info| info.processed_tokens != self.ledger.tokens().len())
        {
            return Err(invalid(
                "layer positions differ from represented token count",
            ));
        }
        Ok(())
    }
    pub(crate) fn supports_pipeline(&self) -> bool {
        self.layers
            .iter()
            .all(|layer| layer.info(0).kind == CacheKind::Full)
    }

    pub(crate) fn speculative_step(&mut self, offset: usize) -> Result<CacheStep<'_>, CacheError> {
        self.validate_ledger()?;
        if !self.supports_pipeline() {
            return Err(invalid("lookahead requires full caches"));
        }
        self.prepare_append(offset)?;
        let mut step = self.begin_step()?;
        for layer in &mut step.staged {
            layer.set_length(offset)?;
        }
        Ok(step)
    }

    pub(crate) fn commit_input(&mut self, token: crate::TokenId) -> Result<(), CacheError> {
        self.validate_ledger()?;
        let pending = self.ledger.prepare(&[token])?;
        let length = self.ledger.tokens().len() + 1;
        for layer in &mut self.layers {
            layer.set_length(length)?;
        }
        self.ledger.commit(pending);
        Ok(())
    }

    pub(crate) fn step_with_tokens(
        &mut self,
        tokens: &[crate::TokenId],
    ) -> Result<CacheStep<'_>, CacheError> {
        self.validate_ledger()?;
        if tokens.is_empty() {
            return Err(invalid("cache step must represent input tokens"));
        }
        let pending = self.ledger.prepare(tokens)?;
        self.prepare_append(self.ledger.tokens().len())?;
        let mut step = self.begin_step()?;
        step.token_count = Some(tokens.len());
        step.pending_tokens = Some(pending);
        Ok(step)
    }
    // Forward-only oracle sessions do not supply represented token IDs.
    #[cfg_attr(not(feature = "oracle-hooks"), allow(dead_code))]
    pub(crate) fn step(&mut self) -> Result<CacheStep<'_>, CacheError> {
        self.ensure_valid()?;
        if self.tracks_tokens {
            return Err(invalid(
                "model-bound steps must supply represented token IDs",
            ));
        }
        self.prepare_append(self.layers[0].info(0).processed_tokens)?;
        self.begin_step()
    }
    fn prepare_append(&mut self, offset: usize) -> Result<(), CacheError> {
        if let Some((end, stream)) = &self.last_submission {
            if offset < *end || *stream != mlx_rs::Stream::thread_local_or_default() {
                // Rolled-back suffixes can still have attention readers in flight.
                #[cfg(mlx_lm_kv_append)]
                append::synchronize(stream).inspect_err(|_| self.invalidated = true)?;
                self.last_submission = None;
            }
        }
        Ok(())
    }
    fn begin_step(&mut self) -> Result<CacheStep<'_>, CacheError> {
        self.ensure_valid()?;
        let staged = self.layers.iter_mut().map(|layer| layer.stage()).collect();
        let updated = vec![false; self.layers.len()];
        Ok(CacheStep {
            cache: self,
            staged,
            updated,
            failed: false,
            pending_tokens: None,
            token_count: None,
        })
    }
}
/// The resolved shape and attention policy for one decoder layer.
#[derive(Debug, Clone, PartialEq)]
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
    fn reserve(&mut self, _capacity: usize) -> Result<(), CacheError> {
        Ok(())
    }
    fn info(&self, layer: usize) -> CacheInfo;
    fn clone_box(&self) -> Box<dyn LayerCache>;
    fn snapshot_box(&self) -> Result<Box<dyn LayerCache>, CacheError> {
        Ok(self.clone_box())
    }
    fn stage(&mut self) -> Box<dyn LayerCache> {
        self.clone_box()
    }
    fn rollback(&mut self, _staged: &mut dyn LayerCache) {}
    fn set_length(&mut self, _length: usize) -> Result<(), CacheError> {
        Err(invalid("length rollback requires a full cache"))
    }
    fn take_growth(&mut self) -> Option<Buffer> {
        None
    }
    fn take_buffer(&mut self) -> Option<Buffer> {
        None
    }
    fn fingerprint(&self) -> LayerFingerprint;
    fn arrays(&self) -> Result<(Array, Array), CacheError>;
    #[cfg_attr(not(feature = "oracle-hooks"), allow(dead_code))]
    fn logical_arrays(&self) -> Result<(Array, Array), CacheError>;
}
/// Appends full caches in place and stages rotating caches on copied handles.
/// Drop preserves committed lengths and prefixes. Failed evaluation invalidates the cache.
pub(crate) struct CacheStep<'cache> {
    cache: &'cache mut Cache,
    staged: Vec<Box<dyn LayerCache>>,
    updated: Vec<bool>,
    failed: bool,
    pending_tokens: Option<PendingTokens>,
    token_count: Option<usize>,
}
/// Evaluated state held until fallible token and text preparation succeeds.
pub(crate) struct EvaluatedCacheStep<'cache> {
    step: CacheStep<'cache>,
}
impl EvaluatedCacheStep<'_> {
    pub(crate) fn commit(mut self) {
        if let Some(pending) = self.step.pending_tokens.take() {
            self.step.cache.ledger.commit(pending);
        }
        self.step.cache.layers = std::mem::take(&mut self.step.staged);
    }
}
impl Drop for CacheStep<'_> {
    fn drop(&mut self) {
        if !self.cache.invalidated {
            for (committed, staged) in self.cache.layers.iter_mut().zip(&mut self.staged) {
                committed.rollback(staged.as_mut());
            }
        }
    }
}
impl<'cache> CacheStep<'cache> {
    pub(crate) fn evaluate(
        self,
        outputs: &[&Array],
    ) -> Result<EvaluatedCacheStep<'cache>, CacheError> {
        self.evaluate_with(outputs, |arrays| {
            mlx_rs::transforms::eval(arrays.iter().copied()).map_err(CacheError::from)
        })
    }
    pub(crate) fn evaluate_outputs<T>(
        mut self,
        evaluate: impl FnOnce(
            &[&Array],
            &mut dyn FnMut(&[&Array]) -> Result<(), mlx_rs::error::Exception>,
        ) -> Result<T, crate::GenerationError>,
    ) -> Result<(T, EvaluatedCacheStep<'cache>), crate::GenerationError> {
        self.validate_complete()?;
        let arrays = self.evaluation_arrays()?;
        let references: Vec<_> = arrays.iter().collect();
        let output = evaluate(&references, &mut |outputs| {
            self.prepare_evaluation()?;
            mlx_rs::transforms::eval(outputs.iter().copied())
                .inspect_err(|_| self.cache.invalidate())
        })?;
        Ok((output, EvaluatedCacheStep { step: self }))
    }
    pub(crate) fn submit(mut self, outputs: &[&Array]) -> Result<(), crate::GenerationError> {
        self.validate_complete()?;
        if outputs.is_empty() {
            return Ok(());
        }
        self.prepare_evaluation()?;
        mlx_rs::transforms::async_eval(outputs.iter().copied())
            .inspect_err(|_| self.cache.invalidate())?;
        Ok(())
    }
    fn prepare_evaluation(&mut self) -> Result<(), mlx_rs::error::Exception> {
        self.cache.last_submission = self.staged.first().map(|layer| {
            (
                layer.info(0).processed_tokens,
                mlx_rs::Stream::thread_local_or_default(),
            )
        });
        for layer in &mut self.cache.layers {
            layer.take_buffer();
        }
        let growth: Vec<_> = self
            .staged
            .iter_mut()
            .filter_map(|layer| layer.take_growth())
            .collect();
        // Bound overlapping old and new storage to one layer during growth.
        for buffer in growth {
            mlx_rs::transforms::eval([&buffer.keys, &buffer.values])
                .inspect_err(|_| self.cache.invalidate())?;
        }
        Ok(())
    }
    fn evaluation_arrays(&self) -> Result<Vec<Array>, CacheError> {
        let mut arrays = Vec::with_capacity(self.staged.len() * 2);
        for layer in &self.staged {
            let (keys, values) = layer.arrays()?;
            arrays.extend([keys, values]);
        }
        Ok(arrays)
    }
    #[cfg(test)]
    pub(crate) fn reserve(&mut self, capacity: NonZeroUsize) -> Result<(), CacheError> {
        let result = self
            .staged
            .iter_mut()
            .try_for_each(|layer| layer.reserve(capacity.get()));
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    pub(crate) fn info(&self, layer: usize) -> Result<CacheInfo, CacheError> {
        self.staged
            .get(layer)
            .map(|cache| cache.info(layer))
            .ok_or_else(|| invalid("layer index out of range"))
    }
    pub(crate) fn update_and_fetch(
        &mut self,
        layer: usize,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), CacheError> {
        let result = self.update_layer(layer, keys, values);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn update_layer(
        &mut self,
        layer: usize,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), CacheError> {
        if self.failed {
            return Err(invalid("transaction already failed"));
        }
        let updated = self
            .updated
            .get_mut(layer)
            .ok_or_else(|| invalid("layer index out of range"))?;
        if *updated {
            return Err(invalid("layer already updated in this step"));
        }
        let cache = self
            .staged
            .get_mut(layer)
            .ok_or_else(|| invalid("layer index out of range"))?;
        let result = cache
            .update_and_fetch(keys, values)
            .map_err(|error| match error {
                CacheError::ShapeMismatch {
                    expected, actual, ..
                } => CacheError::ShapeMismatch {
                    layer,
                    expected,
                    actual,
                },
                other => other,
            })?;
        *updated = true;
        Ok(result)
    }
    #[cfg(test)]
    pub(crate) fn commit(self) -> Result<(), CacheError> {
        self.evaluate_and_commit(&[])
    }
    #[cfg_attr(not(feature = "oracle-hooks"), allow(dead_code))]
    pub(crate) fn evaluate_and_commit(self, outputs: &[&Array]) -> Result<(), CacheError> {
        self.evaluate(outputs)?.commit();
        Ok(())
    }
    fn evaluate_with(
        mut self,
        outputs: &[&Array],
        evaluate: impl FnOnce(&[&Array]) -> Result<(), CacheError>,
    ) -> Result<EvaluatedCacheStep<'cache>, CacheError> {
        self.validate_complete()?;
        let arrays = self.evaluation_arrays()?;
        let references: Vec<_> = arrays.iter().chain(outputs.iter().copied()).collect();
        self.prepare_evaluation()?;
        evaluate(&references).inspect_err(|_| self.cache.invalidate())?;
        Ok(EvaluatedCacheStep { step: self })
    }
    fn validate_complete(&self) -> Result<(), CacheError> {
        if self.failed || self.updated.iter().any(|updated| !updated) {
            return Err(invalid("cannot commit a failed or incomplete step"));
        }
        let offset = self
            .staged
            .first()
            .map(|layer| layer.info(0).processed_tokens);
        if self
            .staged
            .iter()
            .any(|layer| Some(layer.info(0).processed_tokens) != offset)
        {
            return Err(invalid("layer positions differ at step boundary"));
        }
        if let Some(count) = self.token_count {
            let expected = self
                .cache
                .tokens()?
                .len()
                .checked_add(count)
                .ok_or_else(|| invalid("token ledger overflow"))?;
            if offset != Some(expected) {
                return Err(invalid(
                    "staged positions differ from represented input token count",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LayerFingerprint {
    spec: LayerCacheSpec,
    kind: CacheKind,
    capacity: usize,
    keep_prefix: usize,
    processed_tokens: usize,
    length: usize,
    index: usize,
    keys_shape: Vec<i32>,
    values_shape: Vec<i32>,
    keys_dtype: Dtype,
    values_dtype: Dtype,
}
impl LayerFingerprint {
    fn compatible(&self, other: &Self) -> bool {
        self.spec == other.spec
            && self.kind == other.kind
            && self.keep_prefix == other.keep_prefix
            && (self.kind == CacheKind::Full || self.capacity == other.capacity)
    }
}

#[derive(Clone)]
pub(crate) struct Buffer {
    keys: Array,
    values: Array,
}
impl Buffer {
    fn new(spec: &LayerCacheSpec, capacity: usize) -> Result<Self, CacheError> {
        let shape = shape(spec, capacity)?;
        Ok(Self {
            keys: zeros_dtype(&shape, spec.dtype)?,
            values: zeros_dtype(&shape, spec.dtype)?,
        })
    }
    fn replace(&self, start: usize, keys: &Array, values: &Array) -> Result<Self, CacheError> {
        let end = start
            .checked_add(token_length(keys)?)
            .ok_or_else(|| invalid("position overflow"))?;
        let range = dimension(start)?..dimension(end)?;
        let update = |array: &Array, value: &Array| {
            array
                .try_index_update((.., .., range.clone(), ..), value, UpdateMode::Replace)
                .map_err(|error| match error {
                    IndexUpdateError::Exception(source) => CacheError::Exception(source),
                    IndexUpdateError::ZeroStride { .. } => invalid("zero stride in cache update"),
                })
        };
        Ok(Self {
            keys: update(&self.keys, keys)?,
            values: update(&self.values, values)?,
        })
    }
    fn slice(&self, range: Range<usize>) -> Result<(Array, Array), CacheError> {
        Ok((
            slice(&self.keys, range.clone())?,
            slice(&self.values, range)?,
        ))
    }
}
fn invalid(message: &str) -> CacheError {
    CacheError::InvalidState(message.to_owned())
}
fn dimension(value: usize) -> Result<i32, CacheError> {
    i32::try_from(value).map_err(|_| invalid("cache dimension exceeds i32"))
}
fn shape(spec: &LayerCacheSpec, length: usize) -> Result<[i32; 4], CacheError> {
    if spec.batch_size == 0 || spec.kv_heads == 0 || spec.head_dim == 0 {
        return Err(invalid("cache dimensions must be positive"));
    }
    Ok([
        dimension(spec.batch_size)?,
        dimension(spec.kv_heads)?,
        dimension(length)?,
        dimension(spec.head_dim)?,
    ])
}
fn token_length(array: &Array) -> Result<usize, CacheError> {
    array
        .shape()
        .get(2)
        .copied()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| invalid("cache input must have a token axis"))
}
fn validate(spec: &LayerCacheSpec, keys: &Array, values: &Array) -> Result<usize, CacheError> {
    let length = token_length(keys)?;
    let expected = shape(spec, length)?;
    for array in [keys, values] {
        if array.shape() != expected {
            return Err(CacheError::ShapeMismatch {
                layer: 0,
                expected: expected.iter().map(|&n| n as usize).collect(),
                actual: array.shape().iter().map(|&n| n as usize).collect(),
            });
        }
        if array.dtype() != spec.dtype {
            return Err(invalid("cache input dtype differs from layout"));
        }
    }
    if length == 0 {
        return Err(invalid("cache update must contain tokens"));
    }
    Ok(length)
}
fn slice(array: &Array, range: Range<usize>) -> Result<Array, CacheError> {
    Ok(array.try_index((.., .., dimension(range.start)?..dimension(range.end)?, ..))?)
}
fn resolve_policy(
    spec: &LayerCacheSpec,
    policy: &CachePolicy,
) -> Result<Option<(usize, usize)>, CacheError> {
    let resolved = match policy {
        CachePolicy::Full => None,
        CachePolicy::ModelDefault => match spec.attention {
            crate::AttentionKind::Full => None,
            crate::AttentionKind::Sliding { window } => Some((window.get(), 0)),
        },
        CachePolicy::Rotating {
            capacity,
            keep_prefix,
        } => Some((capacity.get(), *keep_prefix)),
    };
    if let Some((capacity, keep)) = resolved {
        validate_rotation(capacity, keep)?;
    }
    shape(spec, 0)?;
    Ok(resolved)
}
fn validate_rotation(capacity: usize, keep: usize) -> Result<(), CacheError> {
    dimension(capacity)?;
    if capacity == 0 || keep >= capacity {
        return Err(CacheError::UnsupportedPolicy(
            "rotating capacity must exceed keep_prefix".to_owned(),
        ));
    }
    Ok(())
}
fn grown_capacity(current: usize, offset: usize, required: usize) -> Result<usize, CacheError> {
    dimension(required)?;
    let capacity = if required <= current {
        current
    } else {
        (offset + (required - offset).div_ceil(256) * 256).min(i32::MAX as usize)
    };
    dimension(capacity)?;
    Ok(capacity)
}

#[cfg(test)]
mod tests;

pub(crate) fn checked_capacity(
    prompt: usize,
    max_tokens: NonZeroUsize,
) -> Result<NonZeroUsize, crate::GenerationError> {
    let length = prompt
        .checked_add(max_tokens.get())
        .ok_or(crate::GenerationError::LengthOverflow)?;
    if length > i32::MAX as usize {
        return Err(crate::GenerationError::SequenceTooLong {
            length,
            limit: i32::MAX as usize,
        });
    }
    NonZeroUsize::new(length).ok_or(crate::GenerationError::LengthOverflow)
}
fn validate_prefix(
    cached: &[crate::TokenId],
    prompt: &[crate::TokenId],
) -> Result<usize, crate::GenerationError> {
    let matched = cached
        .iter()
        .zip(prompt)
        .take_while(|(a, b)| a == b)
        .count();
    if matched != cached.len() {
        return Err(crate::GenerationError::CachePrefixMismatch {
            matched,
            cached: cached.len(),
            prompt: prompt.len(),
        });
    }
    if prompt.len() == cached.len() {
        return Err(crate::GenerationError::NoUncachedTokens);
    }
    Ok(cached.len())
}
fn validate_policy(
    fingerprint: &LayerFingerprint,
    spec: &LayerCacheSpec,
    policy: &CachePolicy,
) -> Result<(), CacheError> {
    let expected = resolve_policy(spec, policy)?;
    let actual = match fingerprint.kind {
        CacheKind::Full => None,
        CacheKind::Rotating => Some((fingerprint.capacity, fingerprint.keep_prefix)),
    };
    if actual != expected {
        return Err(CacheError::PolicyMismatch);
    }
    Ok(())
}
