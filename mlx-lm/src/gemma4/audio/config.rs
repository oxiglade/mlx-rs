//! Gemma 4 audio config (`gemma4_audio`).

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_conv_kernel_size")]
    pub conv_kernel_size: i32,
    /// Final projection width onto `embed_audio`'s input (`None` ⇒ hidden_size).
    pub output_proj_dims: Option<i32>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_residual_weight")]
    pub residual_weight: f32,
    #[serde(default = "default_gradient_clipping")]
    pub gradient_clipping: f32,

    #[serde(default = "default_attention_chunk_size")]
    pub attention_chunk_size: i32,
    #[serde(default = "default_attention_context_left")]
    pub attention_context_left: i32,
    #[serde(default = "default_attention_context_right")]
    pub attention_context_right: i32,
    #[serde(default = "default_attention_logit_cap")]
    pub attention_logit_cap: f32,
    #[serde(default = "default_attention_invalid_logits_value")]
    pub attention_invalid_logits_value: f32,
}

impl AudioConfig {
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }

    /// `output_proj_dims` if set, else `hidden_size` — the `embed_audio` input.
    pub fn projector_input_dim(&self) -> i32 {
        self.output_proj_dims.unwrap_or(self.hidden_size)
    }

    pub fn max_past_horizon(&self) -> i32 {
        (self.attention_context_left - 1).max(0)
    }

    pub fn max_future_horizon(&self) -> i32 {
        self.attention_context_right
    }

    /// Context window per query block: chunk + past + future.
    pub fn context_size(&self) -> i32 {
        self.attention_chunk_size + self.max_past_horizon() + self.max_future_horizon()
    }
}

/// Subsample conv channels `[layer0, layer1]` (fixed by the architecture).
pub const SUBSAMPLING_CONV_CHANNELS: [i32; 2] = [128, 32];
/// Log-mel feature bins the subsample conv consumes.
pub const INPUT_FEAT_SIZE: i32 = 128;
/// Audio sample rate the front-end and encoder expect (16 kHz mono).
pub const AUDIO_SAMPLE_RATE: i32 = 16_000;

const fn default_hidden_size() -> i32 {
    1024
}
const fn default_num_hidden_layers() -> i32 {
    12
}
const fn default_num_attention_heads() -> i32 {
    8
}
const fn default_conv_kernel_size() -> i32 {
    5
}
const fn default_rms_norm_eps() -> f32 {
    1e-6
}
const fn default_residual_weight() -> f32 {
    0.5
}
const fn default_gradient_clipping() -> f32 {
    1e10
}
const fn default_attention_chunk_size() -> i32 {
    12
}
const fn default_attention_context_left() -> i32 {
    13
}
const fn default_attention_context_right() -> i32 {
    0
}
const fn default_attention_logit_cap() -> f32 {
    50.0
}
const fn default_attention_invalid_logits_value() -> f32 {
    -1e9
}
