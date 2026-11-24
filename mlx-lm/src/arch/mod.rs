use crate::{
    cache::{CacheStep, LayerCacheSpec},
    config::RawConfig,
    weights::{WeightDisposition, WeightManifest},
    Config, ConfigError, InferenceError, LoadError, WeightError,
};
use mlx_rs::{utils::StateProjection, Array};
mod activation;
mod attention;
pub(crate) mod gguf;
mod inkling2_loop;
pub(crate) mod llama;
pub(crate) mod qwen3;
mod qwen3_5_moe;

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
    /// Every staged K/V write must feed the logits: GPU decode submits only sampling roots.
    fn forward(
        &mut self,
        tokens: &Array,
        cache: &mut CacheStep<'_>,
    ) -> Result<Array, InferenceError>;
    /// Projects every parameter into strict keyed mutable state.
    fn weight_projection(&mut self) -> Result<StateProjection<'_>, WeightError>;
    /// The prompt token that stands for one row of an image, when the model reads images.
    fn image_token(&self) -> Option<crate::TokenId> {
        None
    }
    /// Embeds one image as the rows that replace its image tokens.
    fn encode_image(&mut self, _: &crate::Image<'_>) -> Result<Array, InferenceError> {
        Err(InferenceError::UnsupportedArchitecture("images".into()))
    }
    /// As [`Self::forward`], with `rows` replacing the image tokens in `tokens`, in order.
    fn forward_with_images(
        &mut self,
        _tokens: &Array,
        _rows: &Array,
        _cache: &mut CacheStep<'_>,
    ) -> Result<Array, InferenceError> {
        Err(InferenceError::UnsupportedArchitecture("images".into()))
    }
    /// Rows of one drafted verification step, the anchor included, when the checkpoint
    /// bundles a drafter.
    fn draft_block(&self) -> Option<usize> {
        None
    }
    /// As [`Self::forward`] or [`Self::forward_with_images`], with the hidden rows the
    /// drafter reads. Evaluate them with the step: later steps overwrite what they read.
    fn forward_taps(
        &mut self,
        _tokens: &Array,
        _rows: Option<&Array>,
        _cache: &mut CacheStep<'_>,
    ) -> Result<(Array, Array), InferenceError> {
        Err(InferenceError::UnsupportedArchitecture("drafting".into()))
    }
    /// Adds the taps of newly committed rows to the drafter's context.
    fn observe(&self, _context: &mut DraftContext, _taps: &Array) -> Result<(), InferenceError> {
        Err(InferenceError::UnsupportedArchitecture("drafting".into()))
    }
    /// Proposes the `count` tokens after `anchor`, the token that follows the context's rows.
    fn propose(
        &self,
        _context: &DraftContext,
        _anchor: crate::TokenId,
        _count: usize,
    ) -> Result<Array, InferenceError> {
        Err(InferenceError::UnsupportedArchitecture("drafting".into()))
    }
}
/// What a drafter keeps of one request's committed rows.
#[derive(Default)]
pub(crate) struct DraftContext {
    /// Each drafter layer's keys and values over the rows it still attends to.
    pub(crate) layers: Vec<(Array, Array)>,
    /// Rows absorbed, which is the position of the next anchor.
    pub(crate) seen: i32,
    /// The newest row's taps.
    pub(crate) last: Option<Array>,
}
/// Resolved public facts paired with architecture-private validated details.
pub(crate) enum ParsedArchitecture {
    Llama(llama::ParsedConfig),
    Qwen3(qwen3::ParsedConfig),
    Hybrid(Config),
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
    inkling2_loop::Factory,
    llama::Factory,
    qwen3::Factory,
    qwen3_5_moe::Factory,
}
