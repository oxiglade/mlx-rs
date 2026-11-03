pub use crate::error::SamplingError;
use std::{cell::RefCell, num::NonZeroUsize};

use crate::{GenerationError, RepetitionPenaltyOptions, TokenId};
use mlx_rs::{
    error::{ConversionError, Exception},
    ops::{self, indexing::TryIndexOp},
    random::RandomState,
    Array, Dtype,
};

/// Reusable temperature and probability-filter settings.
///
/// Defaults to greedy temperature zero, all probability filters disabled, and no seed.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SamplerOptions {
    /// Zero selects greedy argmax.
    pub temperature: f32,
    /// None disables nucleus filtering.
    pub top_p: Option<f32>,
    /// None disables top-k filtering.
    pub top_k: Option<NonZeroUsize>,
    /// None disables min-p filtering.
    pub min_p: Option<MinPOptions>,
    /// Seed for a private per-generation RNG. Greedy sampling constructs no RNG.
    pub seed: Option<u64>,
}
/// Minimum relative probability and retained support size.
#[derive(Debug, Clone)]
pub struct MinPOptions {
    /// Minimum probability relative to the most likely token.
    pub probability: f32,
    /// Minimum number of candidates retained by min-p.
    pub min_tokens_to_keep: NonZeroUsize,
}
impl Default for SamplerOptions {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: None,
            top_k: None,
            min_p: None,
            seed: None,
        }
    }
}

/// Presence or frequency penalty over a trailing token-history window.
///
/// Python's history rule starts with the final prompt token and adds accepted
/// generated tokens, excluding earlier prompt tokens and reused prefixes.
/// A context size of 20 matches Python's recipe; callers supply both fields.
#[derive(Debug, Clone)]
pub struct AdditivePenaltyOptions {
    /// Finite coefficient to subtract. Negative values reward tokens; zero is an identity.
    pub penalty: f32,
    /// Maximum number of history tokens considered by this penalty.
    pub context_size: NonZeroUsize,
}
impl AdditivePenaltyOptions {
    /// Rejects non-finite coefficients; every finite coefficient is admitted.
    pub fn validate(&self) -> Result<(), SamplingError> {
        if !self.penalty.is_finite() {
            return Err(SamplingError::InvalidAdditivePenalty(self.penalty));
        }
        Ok(())
    }
}

pub(crate) trait LogitsProcessor {
    fn process(
        &self,
        history: &[crate::TokenId],
        logits: mlx_rs::Array,
    ) -> Result<mlx_rs::Array, SamplingError>;
}
impl SamplerOptions {
    /// Validates sampler ranges against the runtime vocabulary.
    pub fn validate(&self, vocabulary_size: usize) -> Result<(), SamplingError> {
        validate_vocabulary(vocabulary_size)?;
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(SamplingError::InvalidTemperature(self.temperature));
        }
        if let Some(top_p) = self.top_p {
            if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
                return Err(SamplingError::InvalidProbability {
                    field: "top_p".into(),
                    value: top_p,
                });
            }
        }
        if let Some(top_k) = self.top_k {
            if top_k.get() > vocabulary_size {
                return Err(SamplingError::TopKExceedsVocabulary {
                    top_k: top_k.get(),
                    vocabulary_size,
                });
            }
        }
        if let Some(min_p) = &self.min_p {
            min_p.validate(vocabulary_size)?;
        }
        Ok(())
    }
}
impl MinPOptions {
    /// Validates the relative probability and retained support against the vocabulary.
    pub fn validate(&self, vocabulary_size: usize) -> Result<(), SamplingError> {
        validate_vocabulary(vocabulary_size)?;
        if !self.probability.is_finite() || !(0.0..=1.0).contains(&self.probability) {
            return Err(SamplingError::InvalidProbability {
                field: "min_p".into(),
                value: self.probability,
            });
        }
        if self.min_tokens_to_keep.get() > vocabulary_size {
            return Err(SamplingError::MinTokensToKeepExceedsVocabulary {
                min_tokens_to_keep: self.min_tokens_to_keep.get(),
                vocabulary_size,
            });
        }
        Ok(())
    }
}

fn validate_vocabulary(vocabulary_size: usize) -> Result<(), SamplingError> {
    if vocabulary_size == 0 {
        return Err(SamplingError::EmptyVocabulary);
    }
    Ok(())
}

#[cfg(test)]
mod tests;

impl LogitsProcessor for RepetitionPenaltyOptions {
    fn process(&self, history: &[TokenId], logits: Array) -> Result<Array, SamplingError> {
        if history.is_empty() || self.penalty == 1.0 {
            return Ok(logits);
        }
        self.process_indices(history_indices(history, self.context_size), logits)
    }
}
impl RepetitionPenaltyOptions {
    fn process_indices(&self, indices: Array, logits: Array) -> Result<Array, SamplingError> {
        if self.penalty == 1.0 {
            return Ok(logits);
        }
        let indices = indices.reshape(&[1, -1])?;
        let selected = logits.take_along_axis(&indices, -1)?;
        let penalty = scalar(self.penalty, logits.dtype())?;
        let selected = ops::select(
            selected.lt(scalar(0.0, logits.dtype())?)?,
            selected.multiply(&penalty)?,
            selected.divide(&penalty)?,
        )?;
        Ok(logits.put_along_axis(indices, selected, -1)?)
    }
}

struct PresencePenalty(AdditivePenaltyOptions);
impl LogitsProcessor for PresencePenalty {
    fn process(&self, history: &[TokenId], logits: Array) -> Result<Array, SamplingError> {
        if history.is_empty() || self.0.penalty == 0.0 {
            return Ok(logits);
        }
        self.process_indices(history_indices(history, self.0.context_size), logits)
    }
}
impl PresencePenalty {
    fn process_indices(&self, indices: Array, logits: Array) -> Result<Array, SamplingError> {
        if self.0.penalty == 0.0 {
            return Ok(logits);
        }
        let indices = indices.reshape(&[1, -1])?;
        let selected = logits.take_along_axis(&indices, -1)?;
        let selected = selected.subtract(scalar(self.0.penalty, logits.dtype())?)?;
        Ok(logits.put_along_axis(indices, selected, -1)?)
    }
}

struct FrequencyPenalty(AdditivePenaltyOptions);
impl LogitsProcessor for FrequencyPenalty {
    fn process(&self, history: &[TokenId], logits: Array) -> Result<Array, SamplingError> {
        if history.is_empty() || self.0.penalty == 0.0 {
            return Ok(logits);
        }
        self.process_indices(history_indices(history, self.0.context_size), logits)
    }
}
impl FrequencyPenalty {
    fn process_indices(&self, indices: Array, logits: Array) -> Result<Array, SamplingError> {
        if self.0.penalty == 0.0 {
            return Ok(logits);
        }
        let updates = ops::broadcast_to(
            scalar(-self.0.penalty, logits.dtype())?,
            &[indices.dim(0), 1, 1],
        )?;
        // Python's .at[].subtract accumulates duplicates in the logits dtype.
        Ok(ops::indexing::scatter_add_single(
            logits, indices, updates, 1,
        )?)
    }
}

fn history_indices(history: &[TokenId], context: NonZeroUsize) -> Array {
    let tokens: Vec<u32> = history[history.len().saturating_sub(context.get())..]
        .iter()
        .copied()
        .map(u32::from)
        .collect();
    Array::from_slice(&tokens, &[tokens.len() as i32])
}

fn scalar(value: f32, dtype: Dtype) -> Result<Array, Exception> {
    Array::from(value).as_dtype(dtype)
}

pub(crate) struct SamplingEngine {
    options: SamplerOptions,
    vocabulary_size: usize,
    repetition: Option<RepetitionPenaltyOptions>,
    presence: Option<PresencePenalty>,
    frequency: Option<FrequencyPenalty>,
}

pub(crate) struct PendingSample {
    pub(crate) token: Array,
    pub(crate) filtered_logprobs: Option<Array>,
    pub(crate) rng: Option<RandomState>,
}

impl SamplingEngine {
    pub(crate) fn new(
        options: SamplerOptions,
        vocabulary_size: usize,
        repetition: Option<RepetitionPenaltyOptions>,
        presence: Option<AdditivePenaltyOptions>,
        frequency: Option<AdditivePenaltyOptions>,
    ) -> Result<Self, SamplingError> {
        options.validate(vocabulary_size)?;
        if let Some(penalty) = &repetition {
            penalty.validate()?;
        }
        if let Some(penalty) = &presence {
            penalty.validate()?;
        }
        if let Some(penalty) = &frequency {
            penalty.validate()?;
        }
        Ok(Self {
            options,
            vocabulary_size,
            repetition,
            presence: presence.map(PresencePenalty),
            frequency: frequency.map(FrequencyPenalty),
        })
    }

    /// `rng` is absent at the first position; thereafter pass the last committed state.
    /// History starts at the final prompt token and contains only accepted decode inputs.
    /// Commit the returned RNG state only after the entire generation transaction succeeds.
    pub(crate) fn sample(
        &self,
        history: &[TokenId],
        logits: Array,
        rng: Option<&RandomState>,
        capture_logprobs: bool,
    ) -> Result<PendingSample, GenerationError> {
        self.sample_with_evaluator(history, logits, rng, capture_logprobs, &[], &mut |arrays| {
            mlx_rs::transforms::eval(arrays.iter().copied())
        })
    }

    /// Evaluation exceptions take precedence over value errors; never replay a failed graph.
    pub(crate) fn sample_with_evaluator(
        &self,
        history: &[TokenId],
        logits: Array,
        rng: Option<&RandomState>,
        capture_logprobs: bool,
        extra_outputs: &[&Array],
        evaluate: &mut impl FnMut(&[&Array]) -> Result<(), Exception>,
    ) -> Result<PendingSample, GenerationError> {
        self.prepare(history, None, logits, rng, capture_logprobs)?
            .evaluate(extra_outputs, evaluate)
    }

    pub(crate) fn prepare(
        &self,
        history: &[TokenId],
        previous: Option<&Array>,
        logits: Array,
        rng: Option<&RandomState>,
        capture_logprobs: bool,
    ) -> Result<SamplePlan, GenerationError> {
        self.validate_row(&logits)?;
        let mut checks = Vec::new();
        let sample = (|| -> Result<PendingSample, GenerationError> {
            checks.push(ValueCheck::new(&logits)?);
            let processed = self.process_ahead(history, previous, logits)?;
            if (previous.is_some() || !history.is_empty())
                && (self.repetition.as_ref().is_some_and(|p| p.penalty != 1.0)
                    || self.presence.as_ref().is_some_and(|p| p.0.penalty != 0.0)
                    || self.frequency.as_ref().is_some_and(|p| p.0.penalty != 0.0))
            {
                checks.push(ValueCheck::new(&processed)?);
            }
            let greedy = self.options.temperature == 0.0;
            let filtered = if !greedy || capture_logprobs {
                let logprobs = processed
                    .logsumexp_axis(-1, true)
                    .and_then(|normalizer| processed.subtract(normalizer))
                    .map_err(SamplingError::from)?;
                let filtered = if greedy {
                    logprobs
                } else {
                    self.filter(logprobs)?
                };
                checks.push(ValueCheck::new(&filtered)?);
                Some(filtered)
            } else {
                None
            };
            let (token, candidate_rng) = if greedy {
                (
                    ops::indexing::argmax_axis(&processed, -1, false)
                        .map_err(SamplingError::from)?,
                    None,
                )
            } else {
                let filtered = filtered.as_ref().ok_or(SamplingError::EmptySupport)?;
                let scaled = scalar(
                    (1.0 / f64::from(self.options.temperature)) as f32,
                    filtered.dtype(),
                )
                .and_then(|reciprocal| filtered.multiply(reciprocal))
                .map_err(SamplingError::from)?;
                checks.push(ValueCheck::new(&scaled)?);
                let mut candidate = match rng {
                    Some(state) => state.clone(),
                    None => match self.options.seed {
                        Some(seed) => RandomState::with_seed(seed),
                        None => RandomState::new(),
                    }
                    .map_err(SamplingError::from)?,
                };
                let key = candidate.next_key().map_err(SamplingError::from)?;
                let token = mlx_rs::random::categorical(&scaled, -1, None, &key)
                    .map_err(SamplingError::from)?;
                (token, Some(candidate))
            };
            Ok(PendingSample {
                token,
                filtered_logprobs: if capture_logprobs { filtered } else { None },
                rng: candidate_rng,
            })
        })();
        Ok(SamplePlan { sample, checks })
    }

    fn validate_row(&self, logits: &Array) -> Result<(), SamplingError> {
        if logits.shape().len() != 2
            || logits.shape()[0] != 1
            || usize::try_from(logits.shape()[1]).ok() != Some(self.vocabulary_size)
        {
            return Err(SamplingError::InvalidLogitsShape {
                shape: logits.shape().to_vec(),
                vocabulary_size: self.vocabulary_size,
            });
        }
        if !matches!(
            logits.dtype(),
            Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32
        ) {
            return Err(SamplingError::InvalidLogitsDtype(logits.dtype()));
        }
        Ok(())
    }

    fn process(&self, history: &[TokenId], mut logits: Array) -> Result<Array, SamplingError> {
        if let Some(processor) = &self.repetition {
            logits = processor.process(history, logits)?;
        }
        if let Some(processor) = &self.presence {
            logits = processor.process(history, logits)?;
        }
        if let Some(processor) = &self.frequency {
            logits = processor.process(history, logits)?;
        }
        Ok(logits)
    }

    fn process_ahead(
        &self,
        history: &[TokenId],
        previous: Option<&Array>,
        mut logits: Array,
    ) -> Result<Array, SamplingError> {
        let Some(previous) = previous else {
            return self.process(history, logits);
        };
        let indices = |context: NonZeroUsize| -> Result<Array, SamplingError> {
            let count = context.get().saturating_sub(1).min(history.len());
            let history = history_indices(&history[history.len() - count..], context);
            Ok(ops::concatenate(&[&history, &previous.reshape(&[1])?], 0)?)
        };
        if let Some(penalty) = &self.repetition {
            logits = penalty.process_indices(indices(penalty.context_size)?, logits)?;
        }
        if let Some(penalty) = &self.presence {
            logits = penalty.process_indices(indices(penalty.0.context_size)?, logits)?;
        }
        if let Some(penalty) = &self.frequency {
            logits = penalty.process_indices(indices(penalty.0.context_size)?, logits)?;
        }
        Ok(logits)
    }

    fn filter(&self, mut logprobs: Array) -> Result<Array, SamplingError> {
        if let Some(probability) = self.options.top_p {
            logprobs = top_p(&logprobs, probability)?;
        }
        if let Some(options) = &self.options.min_p {
            logprobs = min_p(&logprobs, options)?;
        }
        if let Some(count) = self.options.top_k {
            logprobs = top_k(&logprobs, count.get())?;
        }
        Ok(logprobs)
    }
}

pub(crate) struct SamplePlan {
    sample: Result<PendingSample, GenerationError>,
    checks: Vec<ValueCheck>,
}
impl SamplePlan {
    pub(crate) fn candidate(&self) -> Option<&PendingSample> {
        self.sample.as_ref().ok()
    }

    pub(crate) fn outputs<'a>(&'a self, extra: &[&'a Array]) -> Vec<&'a Array> {
        let mut outputs: Vec<_> = self.checks.iter().map(|check| &check.status).collect();
        if let Ok(sample) = &self.sample {
            outputs.push(&sample.token);
            if let Some(state) = &sample.rng {
                outputs.push(state.as_array());
            }
            if let Some(row) = &sample.filtered_logprobs {
                outputs.push(row);
            }
            outputs.extend_from_slice(extra);
        }
        outputs
    }

    pub(crate) fn submit(&self) -> Result<(), GenerationError> {
        let outputs = self.outputs(&[]);
        if !outputs.is_empty() {
            mlx_rs::transforms::async_eval(outputs)?;
        }
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<PendingSample, GenerationError> {
        self.evaluate(&[], &mut |arrays| {
            mlx_rs::transforms::eval(arrays.iter().copied())
        })
    }

    fn evaluate(
        self,
        extra: &[&Array],
        evaluate: &mut impl FnMut(&[&Array]) -> Result<(), Exception>,
    ) -> Result<PendingSample, GenerationError> {
        let outputs = self.outputs(extra);
        if !outputs.is_empty() {
            evaluate(&outputs)?;
        }
        for check in self.checks {
            check.validate()?;
        }
        self.sample
    }
}

type ValueClassifier = dyn FnMut(&Array) -> Result<Array, Exception>;

thread_local! {
    static VALUE_CLASSIFIER: RefCell<Option<(mlx_rs::Stream, Box<ValueClassifier>)>> = const { RefCell::new(None) };
}

struct ValueCheck {
    status: Array,
}
impl ValueCheck {
    fn new(row: &Array) -> Result<Self, SamplingError> {
        let stream = mlx_rs::Stream::thread_local_or_default();
        let status = VALUE_CLASSIFIER.with_borrow_mut(|cached| {
            // MLX's compile cache does not key on Rust's scoped stream.
            if cached
                .as_ref()
                .is_none_or(|(previous, _)| *previous != stream)
            {
                *cached = Some((
                    stream,
                    Box::new(mlx_rs::transforms::compile::compile(
                        |row: &Array| {
                            let invalid = row
                                .is_nan()?
                                .logical_or(row.eq(scalar(f32::INFINITY, row.dtype())?)?)?;
                            // Max preserves invalid-before-empty precedence: 2 > 1 > 0.
                            ops::select(
                                invalid,
                                Array::from(2i32),
                                row.is_finite()?.as_dtype(Dtype::Int32)?,
                            )?
                            .max(true)
                        },
                        true,
                    )),
                ));
            }
            cached.as_mut().unwrap().1(row)
        })?;
        Ok(Self { status })
    }
    fn validate(&self) -> Result<(), GenerationError> {
        match self
            .status
            .try_item_exact::<i32>()
            .map_err(conversion_error)?
        {
            2 => Err(SamplingError::InvalidLogitsValue.into()),
            0 => Err(SamplingError::EmptySupport.into()),
            _ => Ok(()),
        }
    }
}

fn conversion_error(error: ConversionError) -> GenerationError {
    match error {
        ConversionError::Exception(source) => GenerationError::Exception(source),
        other => SamplingError::Conversion(other).into(),
    }
}

fn top_p(logprobs: &Array, probability: f32) -> Result<Array, SamplingError> {
    if probability == 1.0 {
        return Ok(logprobs.clone());
    }
    let indices = ops::argsort_axis(logprobs, -1)?;
    let cumulative = logprobs
        .exp()?
        .take_along_axis(&indices, -1)?
        .cumsum(-1, false, true)?;
    let order = ops::arange::<_, u32>(0, logprobs.dim(1) as u32, None)?;
    let inverse = ops::zeros_like(&indices)?.put_along_axis(&indices, order, -1)?;
    let cumulative = cumulative.take_along_axis(inverse, -1)?;
    let keep = cumulative.gt(scalar(
        (1.0 - f64::from(probability)) as f32,
        logprobs.dtype(),
    )?)?;
    Ok(ops::select(
        keep,
        logprobs,
        scalar(f32::NEG_INFINITY, logprobs.dtype())?,
    )?)
}

fn min_p(logprobs: &Array, options: &MinPOptions) -> Result<Array, SamplingError> {
    if options.probability == 0.0 {
        return Ok(logprobs.clone());
    }
    let threshold = logprobs.max_axis(-1, true)?.add(scalar(
        f64::from(options.probability).ln() as f32,
        logprobs.dtype(),
    )?)?;
    let mut remove = logprobs.lt(threshold)?;
    let count = options.min_tokens_to_keep.get() as i32;
    if count > 1 {
        let indices = ops::argpartition_axis(logprobs, -count, -1)?.try_index((.., -count..))?;
        remove = remove.put_along_axis(indices, Array::from(false), -1)?;
    }
    Ok(ops::select(
        remove,
        scalar(f32::NEG_INFINITY, logprobs.dtype())?,
        logprobs,
    )?)
}

fn top_k(logprobs: &Array, count: usize) -> Result<Array, SamplingError> {
    if count == logprobs.dim(1) as usize {
        return Ok(logprobs.clone());
    }
    let count = count as i32;
    let indices =
        ops::argpartition_axis(logprobs.negative()?, count - 1, -1)?.try_index((.., count..))?;
    Ok(logprobs.put_along_axis(indices, scalar(f32::NEG_INFINITY, logprobs.dtype())?, -1)?)
}
