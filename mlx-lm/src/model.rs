pub use crate::error::GenerationError;
use crate::{
    arch::DecoderModel, CacheOptions, Config, LoadError, SamplerOptions, TokenId, Tokenizer,
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
    /// Creates a synchronous single-request streaming generator.
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
}
/// Text or already-tokenized input for one request.
pub enum Prompt<'a> {
    /// Text to encode with the model tokenizer.
    Text(&'a str),
    /// Token IDs supplied by the caller.
    Tokens(&'a [TokenId]),
}
/// Reusable limits, sampling, stopping, and cache options.
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
    /// Source and precedence of stop tokens.
    pub stop_tokens: StopTokenPolicy,
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
            stop_tokens: StopTokenPolicy::Tokenizer,
            cache: CacheOptions::default(),
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
pub enum FinishReason {
    /// A stop token was sampled.
    Stop,
    /// The token limit was reached.
    Length,
}
/// One sampled token and its stable decoded text delta.
pub struct GenerationEvent {
    /// Sampled token, including a final stop token.
    pub token_id: TokenId,
    /// Stable decoded text since the preceding event; it may be empty.
    pub text: String,
    /// Present only on the final successful event.
    pub finish_reason: Option<FinishReason>,
}
/// A synchronous stream fused after its final event or first error.
pub struct Generation<'model> {
    _decoder: PhantomData<&'model mut dyn DecoderModel>,
    _thread_bound: PhantomData<Rc<()>>,
    finished: bool,
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
