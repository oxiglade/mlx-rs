use crate::{
    cache::{CacheStep, LayerCacheSpec},
    config::RawConfig,
    weights::{WeightDisposition, WeightManifest},
    Config, ConfigError, InferenceError, LoadError, WeightError,
};
use mlx_rs::{utils::StateProjection, Array};
pub(crate) mod gguf;
pub(crate) mod llama;
pub(crate) mod qwen3;

/// Architecture-owned interpretation and construction hooks.
pub(crate) trait ArchitectureFactory {
    /// Resolves architecture defaults and validates raw configuration.
    fn parse_config(&self, raw: &RawConfig) -> Result<ParsedArchitecture, ConfigError>;
    /// Resolves the bounded GGUF metadata profile against normalized checkpoint shapes.
    fn parse_gguf_config(
        &self,
        file: &mlx_rs::io::GgufFile,
        weights: &WeightManifest,
    ) -> Result<ParsedArchitecture, LoadError> {
        let raw = crate::model::gguf::raw_config(file, weights)?;
        Ok(self.parse_config(&raw)?)
    }
    /// Constructs a decoder with the manifest-selected parameter layout.
    fn build(
        &self,
        parsed: ParsedArchitecture,
        weights: &WeightManifest,
    ) -> Result<Box<dyn DecoderModel>, LoadError>;
    /// Maps a safetensors name to a parameter, approved ignore, or rejection.
    #[cfg(test)]
    fn map_safetensors_key(&self, external: &str) -> WeightDisposition;
    /// Maps a GGUF name to a canonical checkpoint key or rejection; GGUF has no ignore rule.
    fn map_gguf_key(&self, external: &str) -> WeightDisposition;
}
/// Shared execution's private boundary around architecture math.
pub(crate) trait DecoderModel {
    /// Borrows resolved architecture facts.
    fn config(&self) -> &Config;
    /// Borrows layer cache requirements in decoder order.
    fn cache_layout(&self) -> &[LayerCacheSpec];
    /// Constructs full-position logits against a staged cache transaction.
    fn forward(
        &mut self,
        tokens: &Array,
        cache: &mut CacheStep<'_>,
    ) -> Result<Array, InferenceError>;
    /// Projects every parameter into strict keyed mutable state.
    fn weight_projection(&mut self) -> Result<StateProjection<'_>, WeightError>;
}
/// Resolved public facts paired with architecture-private validated details.
pub(crate) enum ParsedArchitecture {
    Llama(llama::ParsedConfig),
    Qwen3(qwen3::ParsedConfig),
}
macro_rules! architectures {
    ($($module:ident::$factory:ident),+ $(,)?) => {
        pub(crate) fn factory(model_type: &str) -> Result<&'static dyn ArchitectureFactory, ConfigError> {
            match model_type {
                $(stringify!($module) => Ok(&$module::$factory),)+
                _ => Err(ConfigError::UnsupportedArchitecture(model_type.to_owned())),
            }
        }
    };
}
architectures! {
    llama::Factory,
    qwen3::Factory,
}
