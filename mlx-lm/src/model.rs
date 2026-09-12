pub use crate::error::GenerationError;
use crate::{
    arch::DecoderModel, AdditivePenaltyOptions, Cache, CacheError, CacheOptions, CacheSnapshot,
    Config, LoadError, SamplerOptions, TokenId, Tokenizer,
};
use std::{marker::PhantomData, num::NonZeroUsize, path::Path, rc::Rc};

/// A loaded decoder and its resolved tokenizer and configuration.
pub struct Model {
    decoder: Box<dyn DecoderModel>,
    tokenizer: Tokenizer,
    config: Config,
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
        })
    }
    /// Consumes a typed GGUF container with a separately constructed tokenizer.
    pub fn from_gguf(file: mlx_rs::io::GgufFile, tokenizer: Tokenizer) -> Result<Self, LoadError> {
        let _ = (file, tokenizer);
        Err(crate::WeightError::UnsupportedFormat(
            crate::NotYetImplemented("GGUF model loading").to_string(),
        )
        .into())
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
    ///
    /// Currently returns a not-yet-implemented cache policy error.
    pub fn new_cache(&self, options: CacheOptions) -> Result<Cache, CacheError> {
        let _ = options;
        Err(CacheError::UnsupportedPolicy(
            crate::NotYetImplemented("model cache construction").to_string(),
        ))
    }
    /// Creates a synchronous single-request streaming generator.
    ///
    /// Currently returns a not-yet-implemented inference error.
    pub fn generate<'model>(
        &'model mut self,
        prompt: Prompt<'_>,
        options: GenerationOptions,
    ) -> Result<Generation<'model>, GenerationError> {
        let _ = (&self.decoder, prompt, options);
        Err(crate::InferenceError::UnsupportedArchitecture(
            crate::NotYetImplemented("generation").to_string(),
        )
        .into())
    }
    /// Generates from the full intended prompt, including the cached prefix.
    ///
    /// Reuse requires this model instance, the same resolved cache policy, an exact
    /// token prefix, and at least one uncached prompt token. The prompt is encoded
    /// or copied during construction; it is not borrowed by the returned stream.
    /// Currently returns the same not-yet-implemented inference error as `generate`.
    pub fn generate_with_cache<'model>(
        &'model mut self,
        prompt: Prompt<'_>,
        options: GenerationOptions,
        cache: &'model mut Cache,
    ) -> Result<Generation<'model>, GenerationError> {
        let _ = (&self.decoder, prompt, options, cache);
        Err(crate::InferenceError::UnsupportedArchitecture(
            crate::NotYetImplemented("generation").to_string(),
        )
        .into())
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
/// A synchronous stream fused after its final event or first error.
pub struct Generation<'model> {
    _decoder: PhantomData<&'model mut dyn DecoderModel>,
    _thread_bound: PhantomData<Rc<()>>,
    cache: Cache,
    finished: bool,
}
impl Generation<'_> {
    /// Borrows the cache at the most recent completed event boundary.
    ///
    /// Generation construction is currently a placeholder, so no stream is available.
    pub fn cache(&self) -> &Cache {
        &self.cache
    }
    /// Captures cache state only, excluding RNG, detokenizer, and iterator state.
    ///
    /// Generation construction is currently a placeholder, so no stream is available.
    pub fn snapshot(&mut self) -> Result<CacheSnapshot, CacheError> {
        self.cache.snapshot()
    }
}
impl Iterator for Generation<'_> {
    type Item = Result<GenerationEvent, GenerationError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        self.finished = true;
        Some(Err(crate::InferenceError::UnsupportedArchitecture(
            crate::NotYetImplemented("generation step").to_string(),
        )
        .into()))
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
mod tests {
    use super::*;

    #[test]
    fn public_surface_defaults() {
        let options = GenerationOptions::default();
        assert_eq!(options.max_tokens.get(), 256);
        assert_eq!(options.prefill_chunk_size.get(), 2048);
        assert_eq!(options.sampling.temperature, 0.0);
        assert!(options.sampling.top_p.is_none());
        assert!(options.sampling.top_k.is_none());
        assert!(options.sampling.min_p.is_none());
        assert!(options.sampling.seed.is_none());
        assert!(options.repetition_penalty.is_none());
        assert!(options.presence_penalty.is_none());
        assert!(options.frequency_penalty.is_none());
        assert!(matches!(options.stop.tokens, StopTokenPolicy::Tokenizer));
        assert!(options.stop.stop_strings.is_empty());
        assert!(matches!(
            options.cache.policy,
            crate::CachePolicy::ModelDefault
        ));
        let stop = StopPolicy::default();
        assert!(matches!(stop.tokens, StopTokenPolicy::Tokenizer));
        assert!(stop.stop_strings.is_empty());
    }

    #[test]
    fn options_and_events_can_cross_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GenerationOptions>();
        assert_send_sync::<GenerationEvent>();
    }

    #[test]
    fn eos_precedence_generation_config_over_config_over_tokenizer() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures/llama-base");
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            std::fs::copy(fixture.join(name), directory.path().join(name))?;
        }
        let config = directory.path().join("config.json");
        let generation_config = directory.path().join("generation_config.json");
        std::fs::write(&config, br#"{"model_type":"llama","eos_token_id":9}"#)?;
        std::fs::write(&generation_config, br#"{"eos_token_id":[3,2]}"#)?;
        assert_eq!(
            Tokenizer::from_dir(directory.path())?.eos_tokens(),
            &[TokenId::from(2), TokenId::from(3)]
        );
        std::fs::remove_file(&generation_config)?;
        assert_eq!(
            Tokenizer::from_dir(directory.path())?.eos_tokens(),
            &[TokenId::from(9)]
        );
        std::fs::remove_file(&config)?;
        let tokenizer = Tokenizer::from_dir(directory.path())?;
        let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(
            directory.path().join("tokenizer_config.json"),
        )?)?;
        let eos = metadata["eos_token"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("fixture eos_token must be a string"))?;
        let expected = tokenizer.encode(eos)?;
        assert_eq!(expected.len(), 1);
        assert_eq!(tokenizer.eos_tokens(), expected);

        std::fs::write(&config, br#"{"model_type":"llama","eos_token_id":9}"#)?;
        std::fs::write(&generation_config, br#"{"eos_token_id":null}"#)?;
        // mlx_lm treats a falsy generation_config eos_token_id as absent and falls
        // through to config.json.
        assert_eq!(
            Tokenizer::from_dir(directory.path())?.eos_tokens(),
            &[TokenId::from(9)]
        );
        Ok(())
    }

    #[test]
    fn local_loading_matches_fixture_expectations() -> anyhow::Result<()> {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures");
        let mut count = 0;
        let mut not_run = 0;
        for entry in std::fs::read_dir(fixtures)? {
            let path = entry?.path();
            if !path.is_dir() {
                continue;
            }
            count += 1;
            let model = match Model::from_dir(&path) {
                Ok(model) => model,
                Err(LoadError::Weights(crate::WeightError::UnsupportedFormat(message)))
                | Err(LoadError::Config(crate::ConfigError::UnsupportedArchitecture(message)))
                    if message.ends_with("not yet implemented in this tranche") =>
                {
                    eprintln!(
                        "NOT RUN: Model::from_dir happy path for {}: {message}",
                        path.file_name().unwrap_or_default().to_string_lossy()
                    );
                    not_run += 1;
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            let config =
                crate::config::RawConfig::from_bytes(&std::fs::read(path.join("config.json"))?)?
                    .resolve()?;
            assert_eq!(model.config(), &config, "{}", path.display());
            let expectations: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path.join("expectations.json"))?)?;
            let mut expected: Vec<u32> =
                serde_json::from_value(expectations["tokenizer"]["eos_tokens"].clone())?;
            let mut actual: Vec<u32> = model
                .tokenizer()
                .eos_tokens()
                .iter()
                .copied()
                .map(u32::from)
                .collect();
            expected.sort_unstable();
            actual.sort_unstable();
            assert_eq!(actual, expected, "{}", path.display());
        }
        assert!(count > 0, "no fixture directories found");
        if not_run > 0 {
            eprintln!("NOT RUN: Model::from_dir happy path for {not_run}/{count} fixtures");
            anyhow::bail!(
                "NOT RUN: {not_run} fixtures still require architecture or loader implementation"
            );
        }
        Ok(())
    }

    #[test]
    fn local_loading_rejects_missing_and_invalid_config_before_allocation() -> Result<(), LoadError>
    {
        let directory = tempfile::tempdir()?;
        assert!(
            matches!(Model::from_dir(directory.path()), Err(LoadError::MissingFile(path)) if path == directory.path().join("config.json"))
        );
        let path = directory.path().join("config.json");
        std::fs::write(&path, b"{")?;
        assert!(matches!(
            Model::from_dir(directory.path()),
            Err(LoadError::Config(crate::ConfigError::Json(_)))
        ));
        std::fs::write(&path, b"{}")?;
        assert!(
            matches!(Model::from_dir(directory.path()), Err(LoadError::Config(crate::ConfigError::MissingField { field })) if field == "model_type")
        );
        std::fs::write(&path, br#"{"model_type":"unsupported_oracle_model"}"#)?;
        assert!(
            matches!(Model::from_dir(directory.path()), Err(LoadError::Config(crate::ConfigError::UnsupportedArchitecture(name))) if name == "unsupported_oracle_model")
        );
        let mut config: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../conformance/mlx-lm/fixtures/llama-base/config.json"
        ))
        .map_err(crate::ConfigError::from)?;
        config["num_attention_heads"] = serde_json::json!(0);
        std::fs::write(
            path,
            serde_json::to_vec(&config).map_err(crate::ConfigError::from)?,
        )?;
        assert!(
            matches!(Model::from_dir(directory.path()), Err(LoadError::Config(crate::ConfigError::InvalidNumericField { field, .. })) if field == "num_attention_heads")
        );
        Ok(())
    }

    #[test]
    fn local_loading_reads_tokenizer_metadata_before_building() -> Result<(), LoadError> {
        let directory = tempfile::tempdir()?;
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures/llama-base");
        std::fs::copy(
            fixture.join("config.json"),
            directory.path().join("config.json"),
        )?;
        assert!(
            matches!(Model::from_dir(directory.path()), Err(LoadError::MissingFile(path)) if path == directory.path().join("tokenizer.json"))
        );
        std::fs::copy(
            fixture.join("tokenizer.json"),
            directory.path().join("tokenizer.json"),
        )?;
        std::fs::write(directory.path().join("tokenizer_config.json"), b"{")?;
        assert!(matches!(
            Model::from_dir(directory.path()),
            Err(LoadError::Tokenizer(crate::TokenizerError::Json(_)))
        ));
        std::fs::copy(
            fixture.join("tokenizer_config.json"),
            directory.path().join("tokenizer_config.json"),
        )?;
        std::fs::write(
            directory.path().join("generation_config.json"),
            br#"{"eos_token_id":-1}"#,
        )?;
        assert!(matches!(
            Model::from_dir(directory.path()),
            Err(LoadError::Tokenizer(crate::TokenizerError::InvalidEos(_)))
        ));
        std::fs::write(directory.path().join("generation_config.json"), b"{")?;
        assert!(matches!(
            Model::from_dir(directory.path()),
            Err(LoadError::Tokenizer(crate::TokenizerError::Json(_)))
        ));
        Ok(())
    }

    #[test]
    fn repetition_penalty_range() -> Result<(), crate::SamplingError> {
        for penalty in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, 0.0] {
            let options = RepetitionPenaltyOptions {
                penalty,
                context_size: NonZeroUsize::MIN,
            };
            assert!(matches!(
                options.validate(),
                Err(crate::SamplingError::InvalidRepetitionPenalty(_))
            ));
        }
        for penalty in [f32::MIN_POSITIVE, 0.5, 1.0, 2.0, f32::MAX] {
            RepetitionPenaltyOptions {
                penalty,
                context_size: NonZeroUsize::MIN,
            }
            .validate()?;
        }
        Ok(())
    }
}
