//! Gemma 4 Unified audio config (`audio_config`, `gemma4_unified_audio`).
//!
//! Encoder-free: raw 16 kHz waveform is chunked into `audio_samples_per_token`
//! (640 = 40 ms) frames, each frame projected straight into text space by an
//! RMSNorm(no-scale) → linear. No USM Conformer, no mel spectrogram.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    /// Raw samples per token (= projection input dim). 640 @ 16 kHz = 40 ms.
    pub audio_samples_per_token: i32,
    /// `embedding_projection` output width (= text hidden size).
    pub output_proj_dims: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}
