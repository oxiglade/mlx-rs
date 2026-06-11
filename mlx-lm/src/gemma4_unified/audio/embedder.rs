//! Gemma 4 Unified encoder-free audio embedder: `RMSNorm(no-scale) →
//! embedding_projection` (linear), mapping raw `[B, T, samples_per_token]`
//! frames into text-space soft tokens `[B, T, text_hidden]`.

use mlx_rs::builder::Builder;
use mlx_rs::macros::{ModuleParameters, Quantizable};
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::quantization::MaybeQuantized;
use mlx_rs::Array;

use crate::error::Error;
use crate::gemma4_unified::audio::config::AudioConfig;
use crate::nn::RmsNormNoScale;

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct AudioEmbedder {
    mm_norm: RmsNormNoScale,
    #[quantizable]
    #[param]
    pub embedding_projection: MaybeQuantized<nn::Linear>,
}

impl AudioEmbedder {
    pub fn new(cfg: &AudioConfig) -> Result<Self, Error> {
        Ok(Self {
            mm_norm: RmsNormNoScale::new(cfg.rms_norm_eps),
            embedding_projection: MaybeQuantized::Original(
                nn::LinearBuilder::new(cfg.audio_samples_per_token, cfg.output_proj_dims)
                    .bias(false)
                    .build()?,
            ),
        })
    }

    /// `frames` `[B, T, samples_per_token]` → soft tokens `[B, T, text_hidden]`.
    pub fn forward(&mut self, frames: &Array) -> Result<Array, Error> {
        let x = self.mm_norm.forward(frames)?;
        Ok(self.embedding_projection.forward(&x)?)
    }
}
