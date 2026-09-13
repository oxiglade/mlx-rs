pub(crate) mod gguf;
pub use crate::error::GenerationError;
use crate::{
    arch::DecoderModel, AdditivePenaltyOptions, Cache, CacheError, CacheOptions, CacheSnapshot,
    Config, LoadError, SamplerOptions, TokenId, Tokenizer,
};
use crate::{
    sampling::{PendingSample, SamplingEngine},
    stop::StopStringFilter,
    tokenizer::StreamingDecoder,
};
use mlx_rs::{ops::indexing::TryIndexOp, random::RandomState, Array};
use std::{num::NonZeroUsize, path::Path, rc::Rc};

/// A loaded decoder and its resolved tokenizer and configuration.
pub struct Model {
    decoder: Box<dyn DecoderModel>,
    tokenizer: Tokenizer,
    config: Config,
    identity: Rc<()>,
}
impl Model {
    #[cfg(feature = "oracle-hooks")]
    pub(crate) fn decoder_mut(&mut self) -> &mut dyn DecoderModel {
        self.decoder.as_mut()
    }

    /// Loads a local model with strict safetensors weight matching.
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let config_path = path.join("config.json");
        let bytes = std::fs::read(&config_path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                LoadError::MissingFile(config_path)
            } else {
                LoadError::Io(error)
            }
        })?;
        let raw = crate::config::RawConfig::from_bytes(&bytes)?;
        raw.resolve()?;
        let factory = crate::arch::factory(&raw.model_type)?;
        let tokenizer = Tokenizer::from_dir(path).map_err(|error| match error {
            crate::TokenizerError::Io(source) if source.kind() == std::io::ErrorKind::NotFound => {
                LoadError::MissingFile(path.join("tokenizer.json"))
            }
            crate::TokenizerError::MissingFile(missing) => LoadError::MissingFile(missing),
            other => LoadError::Tokenizer(other),
        })?;
        let parsed = factory.parse_config(&raw)?;
        let weights = crate::weights::WeightManifest::discover(path)?;
        let decoder = factory.build(parsed, &weights)?;
        let config = decoder.config().clone();
        Ok(Self {
            decoder,
            tokenizer,
            config,
            identity: Rc::new(()),
        })
    }
    /// Consumes a typed GGUF container with a separately constructed tokenizer.
    pub fn from_gguf(file: mlx_rs::io::GgufFile, tokenizer: Tokenizer) -> Result<Self, LoadError> {
        gguf::load(file, tokenizer)
    }
    /// Resolves an allowlisted Hub snapshot and loads it through the local loader.
    #[cfg(feature = "hf-hub")]
    pub fn from_hub(repo: &str, options: HubOptions) -> Result<Self, crate::HubError> {
        let _ = (repo, options);
        Err(
            LoadError::Config(crate::ConfigError::UnsupportedArchitecture(
                crate::NotYetImplemented("Hub loading").to_string(),
            ))
            .into(),
        )
    }
    /// Borrows the resolved model configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }
    /// Borrows the model-owned tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    /// Creates an empty cache for reuse within this loaded model instance.
    pub fn new_cache(&self, options: CacheOptions) -> Result<Cache, CacheError> {
        Cache::new_for_model(
            self.config.model_type.clone(),
            self.decoder.cache_layout(),
            &options,
            None,
            Rc::clone(&self.identity),
        )
    }
    /// Creates a synchronous single-request streaming generator.
    pub fn generate<'model>(
        &'model mut self,
        prompt: Prompt<'_>,
        options: GenerationOptions,
    ) -> Result<Generation<'model>, GenerationError> {
        self.start_generation(prompt, options, None)
    }
    /// Generates from the full intended prompt, including the cached prefix.
    ///
    /// Reuse requires this model instance, the same resolved cache policy, an exact
    /// token prefix, and at least one uncached prompt token. The prompt is encoded
    /// or copied during construction; it is not borrowed by the returned stream.
    pub fn generate_with_cache<'model>(
        &'model mut self,
        prompt: Prompt<'_>,
        options: GenerationOptions,
        cache: &'model mut Cache,
    ) -> Result<Generation<'model>, GenerationError> {
        self.start_generation(prompt, options, Some(cache))
    }

    fn start_generation<'a>(
        &'a mut self,
        prompt: Prompt<'_>,
        options: GenerationOptions,
        cache: Option<&'a mut Cache>,
    ) -> Result<Generation<'a>, GenerationError> {
        let vocabulary_size = self.config.dimensions.vocabulary_size;
        let sampling = SamplingEngine::new(
            options.sampling,
            vocabulary_size,
            options.repetition_penalty,
            options.presence_penalty,
            options.frequency_penalty,
        )?;
        let filter = StopStringFilter::new(options.stop.stop_strings)?;
        let mut stop_tokens = match options.stop.tokens {
            StopTokenPolicy::Tokenizer => self.tokenizer.eos_tokens().to_vec(),
            StopTokenPolicy::TokenizerPlus(mut ids) => {
                ids.extend_from_slice(self.tokenizer.eos_tokens());
                ids
            }
            StopTokenPolicy::Exact(ids) => ids,
        };
        stop_tokens.sort_unstable_by_key(|id| u32::from(*id));
        stop_tokens.dedup();
        if let Some(&token_id) = stop_tokens
            .iter()
            .find(|id| u32::from(**id) as usize >= vocabulary_size)
        {
            return Err(GenerationError::StopTokenOutOfRange {
                token_id,
                vocabulary_size,
            });
        }
        let prompt = match prompt {
            Prompt::Text(text) => self.tokenizer.encode_with_special_tokens(
                text,
                !self
                    .tokenizer
                    .bos_token()
                    .is_some_and(|bos| text.starts_with(bos)),
            )?,
            Prompt::Tokens(ids) => ids.to_vec(),
        };
        let last = *prompt.last().ok_or(GenerationError::EmptyPrompt)?;
        if let Some((index, &token_id)) = prompt
            .iter()
            .enumerate()
            .find(|(_, id)| u32::from(**id) as usize >= vocabulary_size)
        {
            return Err(GenerationError::PromptTokenOutOfRange {
                index,
                token_id,
                vocabulary_size,
            });
        }
        let cached = match &cache {
            Some(cache) => cache.validate_reuse(
                &self.identity,
                &self.config.model_type,
                self.decoder.cache_layout(),
                &options.cache,
                &prompt,
            )?,
            None => 0,
        };
        let capacity = crate::cache::checked_capacity(prompt.len(), options.max_tokens)?;
        let reserve = cache.as_ref().map(|_| capacity);
        let cache = match cache {
            Some(cache) => GenerationCache::Borrowed(cache),
            None => GenerationCache::Owned(Cache::new_for_model(
                self.config.model_type.clone(),
                self.decoder.cache_layout(),
                &options.cache,
                Some(capacity),
                Rc::clone(&self.identity),
            )?),
        };
        Ok(Generation {
            decoder: self.decoder.as_mut(),
            cache,
            prompt,
            cached,
            processed: 0,
            chunk: options.prefill_chunk_size,
            max_tokens: options.max_tokens,
            reserve,
            sampling,
            rng: None,
            history: vec![last],
            accepted: 0,
            text: Some(TextState {
                decoder: self.tokenizer.decode_stream(),
                filter,
            }),
            stop_tokens,
            state: GenerationState::Start,
            capture_logprobs: false,
            first_logprobs: None,
        })
    }
}
/// Text or already-tokenized input for one request.
pub enum Prompt<'a> {
    /// Text encoded with the tokenizer's post-processor adding special tokens,
    /// unless the original string starts with the configured BOS token content.
    /// The check uses `text.starts_with(bos_token)` without trimming or encoding
    /// first; when no BOS token is configured, special tokens are added.
    Text(&'a str),
    /// Token IDs supplied by the caller.
    Tokens(&'a [TokenId]),
}
/// Reusable limits, sampling, stopping, and cache options.
///
/// Defaults to 256 sampled tokens, prefill chunks of at most 2048, greedy sampling,
/// no penalties, tokenizer EOS stopping, no stop strings, and model-default caching.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GenerationOptions {
    /// Maximum number of sampled tokens.
    pub max_tokens: NonZeroUsize,
    /// Maximum number of prompt tokens per prefill step.
    pub prefill_chunk_size: NonZeroUsize,
    /// Temperature and support-filter settings.
    pub sampling: SamplerOptions,
    /// Optional history-dependent repetition penalty.
    pub repetition_penalty: Option<RepetitionPenaltyOptions>,
    /// Subtracts a coefficient once per distinct token in its history window.
    pub presence_penalty: Option<AdditivePenaltyOptions>,
    /// Subtracts a coefficient per token occurrence in its history window.
    pub frequency_penalty: Option<AdditivePenaltyOptions>,
    /// Stop-token sources and exact generated-text terminators.
    pub stop: StopPolicy,
    /// Per-request cache policy.
    pub cache: CacheOptions,
}
impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: NonZeroUsize::new(256).unwrap_or(NonZeroUsize::MIN),
            prefill_chunk_size: NonZeroUsize::new(2048).unwrap_or(NonZeroUsize::MIN),
            sampling: SamplerOptions::default(),
            repetition_penalty: None,
            presence_penalty: None,
            frequency_penalty: None,
            stop: StopPolicy::default(),
            cache: CacheOptions::default(),
        }
    }
}
/// Token and text conditions that finish generation with [`FinishReason::Stop`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StopPolicy {
    /// Source and precedence of stop tokens.
    pub tokens: StopTokenPolicy,
    /// Exact UTF-8 strings matched only against newly generated text.
    ///
    /// Empty strings are rejected. Matching holds back possible partial matches
    /// and excludes the earliest complete match and all following text. Matching
    /// uses exact UTF-8 without normalization; a match in the final decoder flush
    /// takes precedence over the token limit.
    pub stop_strings: Vec<String>,
}
impl Default for StopPolicy {
    fn default() -> Self {
        Self {
            tokens: StopTokenPolicy::Tokenizer,
            stop_strings: Vec::new(),
        }
    }
}
/// Selects the source of the effective stop-token set.
#[derive(Debug, Clone)]
pub enum StopTokenPolicy {
    /// Uses the tokenizer's resolved EOS set.
    Tokenizer,
    /// Extends the tokenizer's EOS set with explicit IDs.
    TokenizerPlus(Vec<TokenId>),
    /// Replaces the tokenizer's EOS set with explicit IDs.
    Exact(Vec<TokenId>),
}
/// History-dependent logits penalty settings.
///
/// Python's history rule starts with the final prompt token, then adds accepted
/// generated tokens. Earlier prompt tokens and reused prefixes are excluded.
/// A context size of 20 matches Python's recipe; no penalty is enabled by default.
#[derive(Debug, Clone)]
pub struct RepetitionPenaltyOptions {
    /// Positive finite multiplier applied to repeated tokens.
    pub penalty: f32,
    /// Maximum number of previous tokens considered.
    pub context_size: NonZeroUsize,
}
impl RepetitionPenaltyOptions {
    /// Validates the repetition penalty before generation.
    pub fn validate(&self) -> Result<(), crate::SamplingError> {
        if !self.penalty.is_finite() || self.penalty <= 0.0 {
            return Err(crate::SamplingError::InvalidRepetitionPenalty(self.penalty));
        }
        Ok(())
    }
}
/// Why the final successful event ended generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinishReason {
    /// A stop token or generated-text stop string was matched.
    Stop,
    /// The token limit was reached.
    Length,
}
/// A completed prefill boundary or sampled token and stable text delta.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GenerationEvent {
    /// Progress over newly prefilling tokens, with the final prompt token reserved.
    #[non_exhaustive]
    Prefill {
        /// Newly processed tokens committed to the cache; initially zero.
        processed: usize,
        /// Number of uncached prompt tokens in this request.
        total: usize,
    },
    /// A sampled token; the cache excludes this token until the next decode step.
    #[non_exhaustive]
    Token {
        /// Sampled token, including a final stop token.
        token_id: TokenId,
        /// Stable decoded text since the preceding event; it may be empty.
        text: String,
        /// Present only on the final successful event.
        finish_reason: Option<FinishReason>,
    },
}
enum GenerationCache<'a> {
    Owned(Cache),
    Borrowed(&'a mut Cache),
}
impl GenerationCache<'_> {
    fn get(&self) -> &Cache {
        match self {
            Self::Owned(cache) => cache,
            Self::Borrowed(cache) => cache,
        }
    }
    fn get_mut(&mut self) -> &mut Cache {
        match self {
            Self::Owned(cache) => cache,
            Self::Borrowed(cache) => cache,
        }
    }
}
enum GenerationState {
    Start,
    Prefill,
    FirstSample(Array),
    Decode(TokenId),
    Done,
}
struct TextState<'a> {
    decoder: StreamingDecoder<'a>,
    filter: StopStringFilter,
}
impl TextState<'_> {
    fn prepare(
        mut self,
        token: TokenId,
        stop: bool,
        length: bool,
    ) -> Result<(Option<Self>, String, Option<FinishReason>), GenerationError> {
        let delta = if stop {
            String::new()
        } else {
            self.decoder.step(token)?.unwrap_or_default()
        };
        let (mut text, matched) = self.filter.process(&delta);
        if matched {
            return Ok((None, text, Some(FinishReason::Stop)));
        }
        if stop || length {
            let (tail, matched) = self.filter.finish(&self.decoder.finish()?);
            text.push_str(&tail);
            return Ok((
                None,
                text,
                Some(if stop || matched {
                    FinishReason::Stop
                } else {
                    FinishReason::Length
                }),
            ));
        }
        Ok((Some(self), text, None))
    }
}
/// A synchronous stream fused after its final event or first error.
pub struct Generation<'model> {
    decoder: &'model mut dyn DecoderModel,
    cache: GenerationCache<'model>,
    prompt: Vec<TokenId>,
    cached: usize,
    processed: usize,
    chunk: NonZeroUsize,
    max_tokens: NonZeroUsize,
    reserve: Option<NonZeroUsize>,
    sampling: SamplingEngine,
    rng: Option<RandomState>,
    history: Vec<TokenId>,
    accepted: usize,
    text: Option<TextState<'model>>,
    stop_tokens: Vec<TokenId>,
    state: GenerationState,
    capture_logprobs: bool,
    first_logprobs: Option<Array>,
}
impl<'model> Generation<'model> {
    /// Borrows the cache at the most recent completed event boundary.
    pub fn cache(&self) -> &Cache {
        self.cache.get()
    }
    /// Captures cache state only, excluding RNG, detokenizer, and iterator state.
    pub fn snapshot(&mut self) -> Result<CacheSnapshot, CacheError> {
        self.cache.get_mut().snapshot()
    }

    #[cfg(feature = "oracle-hooks")]
    pub(crate) fn capture_first_logprobs(&mut self) {
        self.capture_logprobs = true;
    }
    #[cfg(feature = "oracle-hooks")]
    pub(crate) fn first_logprobs(&self) -> Option<&Array> {
        self.first_logprobs.as_ref()
    }

    fn advance(&mut self, state: GenerationState) -> Result<GenerationEvent, GenerationError> {
        let total = self.prompt.len() - self.cached;
        match state {
            GenerationState::Start => {
                self.state = GenerationState::Prefill;
                Ok(GenerationEvent::Prefill {
                    processed: 0,
                    total,
                })
            }
            GenerationState::Prefill => {
                let remaining = total - self.processed;
                let count = if remaining == 1 {
                    1
                } else {
                    self.chunk.get().min(remaining - 1)
                };
                let start = self.cached + self.processed;
                let ids = &self.prompt[start..start + count];
                let tokens = token_array(ids)?;
                let mut step = self.cache.get_mut().step_with_tokens(ids)?;
                if let Some(capacity) = self.reserve {
                    step.reserve(capacity)?;
                }
                let logits = self.decoder.forward(&tokens, &mut step)?;
                let row = if remaining == 1 {
                    Some(last_row(&logits)?)
                } else {
                    None
                };
                step.evaluate(&[row.as_ref().unwrap_or(&logits)])
                    .map_err(evaluation_error)?
                    .commit();
                self.reserve = None;
                self.processed += count;
                self.state = match row {
                    Some(row) => GenerationState::FirstSample(row),
                    None => GenerationState::Prefill,
                };
                Ok(GenerationEvent::Prefill {
                    processed: self.processed,
                    total,
                })
            }
            GenerationState::FirstSample(logits) => {
                let sample = self.sampling.sample(
                    &self.history,
                    logits,
                    self.rng.as_ref(),
                    self.capture_logprobs,
                )?;
                self.accept(sample)
            }
            GenerationState::Decode(previous) => {
                let tokens = token_array(&[previous])?;
                let mut step = self.cache.get_mut().step_with_tokens(&[previous])?;
                let logits = self.decoder.forward(&tokens, &mut step)?;
                let row = last_row(&logits)?;
                let sample = self
                    .sampling
                    .sample(&self.history, row, self.rng.as_ref(), false)?;
                let mut outputs = vec![&logits, &sample.token];
                if let Some(rng) = &sample.rng {
                    outputs.push(rng.as_array());
                }
                let guard = step.evaluate(&outputs).map_err(evaluation_error)?;
                let token = sample_token(&sample)?;
                let stop = self.stop_tokens.contains(&token);
                let text = self
                    .text
                    .take()
                    .expect("active generation has a text decoder");
                let (next_text, text, reason) =
                    text.prepare(token, stop, self.accepted + 1 == self.max_tokens.get())?;
                guard.commit();
                self.publish(sample, token, next_text, text, reason)
            }
            GenerationState::Done => unreachable!("done is handled by next"),
        }
    }

    fn accept(&mut self, sample: PendingSample) -> Result<GenerationEvent, GenerationError> {
        let token = sample_token(&sample)?;
        let stop = self.stop_tokens.contains(&token);
        let text = self
            .text
            .take()
            .expect("active generation has a text decoder");
        let (next_text, text, reason) =
            text.prepare(token, stop, self.accepted + 1 == self.max_tokens.get())?;
        self.publish(sample, token, next_text, text, reason)
    }

    fn publish(
        &mut self,
        sample: PendingSample,
        token: TokenId,
        next_text: Option<TextState<'model>>,
        text: String,
        reason: Option<FinishReason>,
    ) -> Result<GenerationEvent, GenerationError> {
        self.rng = sample.rng;
        if self.accepted == 0 {
            self.first_logprobs = sample.filtered_logprobs;
        }
        self.accepted += 1;
        self.history.push(token);
        self.text = next_text;
        self.state = if reason.is_some() {
            GenerationState::Done
        } else {
            GenerationState::Decode(token)
        };
        Ok(GenerationEvent::Token {
            token_id: token,
            text,
            finish_reason: reason,
        })
    }
}
fn token_array(ids: &[TokenId]) -> Result<Array, GenerationError> {
    let length = i32::try_from(ids.len()).map_err(|_| GenerationError::SequenceTooLong {
        length: ids.len(),
        limit: i32::MAX as usize,
    })?;
    let ids: Vec<u32> = ids.iter().copied().map(u32::from).collect();
    Ok(Array::from_slice(&ids, &[1, length]))
}
fn last_row(logits: &Array) -> Result<Array, GenerationError> {
    logits
        .try_index((.., -1, ..))
        .map_err(crate::InferenceError::from)
        .map_err(Into::into)
}
fn sample_token(sample: &PendingSample) -> Result<TokenId, GenerationError> {
    sample
        .token
        .try_item_exact::<u32>()
        .map(TokenId::from)
        .map_err(|error| match error {
            mlx_rs::error::ConversionError::Exception(source) => GenerationError::Exception(source),
            other => crate::SamplingError::Conversion(other).into(),
        })
}
fn evaluation_error(error: CacheError) -> GenerationError {
    match error {
        CacheError::Exception(source) => GenerationError::Exception(source),
        other => other.into(),
    }
}
impl Iterator for Generation<'_> {
    type Item = Result<GenerationEvent, GenerationError>;
    fn next(&mut self) -> Option<Self::Item> {
        let state = std::mem::replace(&mut self.state, GenerationState::Done);
        if matches!(state, GenerationState::Done) {
            return None;
        }
        let result = self.advance(state);
        if result.is_err() {
            self.text = None;
        }
        Some(result)
    }
}
impl std::iter::FusedIterator for Generation<'_> {}

/// Revision and local-cache controls for optional Hub loading.
#[cfg(feature = "hf-hub")]
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct HubOptions {
    /// Optional requested revision; immutable commit SHAs are recommended.
    pub revision: Option<String>,
    /// Requires all assets to be available without downloading.
    pub offline: bool,
    /// Optional cache directory override.
    pub cache_dir: Option<std::path::PathBuf>,
}

#[cfg(test)]
mod tests;
