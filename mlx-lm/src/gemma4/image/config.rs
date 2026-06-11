//! Gemma 4 vision config (`gemma4_vision`).

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_num_attention_heads")]
    pub num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    #[serde(default = "default_patch_size")]
    pub patch_size: i32,
    #[serde(default = "default_pooling_kernel_size")]
    pub pooling_kernel_size: i32,
    #[serde(default = "default_default_output_length")]
    pub default_output_length: i32,
    #[serde(default = "default_position_embedding_size")]
    pub position_embedding_size: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_standardize")]
    pub standardize: bool,
}

impl VisionConfig {
    /// Max patch count before pooling: `default_output_length · k²`.
    pub fn max_patches(&self) -> i32 {
        self.default_output_length * self.pooling_kernel_size * self.pooling_kernel_size
    }
}

const fn default_num_hidden_layers() -> i32 {
    27
}
const fn default_hidden_size() -> i32 {
    1152
}
const fn default_intermediate_size() -> i32 {
    4304
}
const fn default_num_attention_heads() -> i32 {
    16
}
const fn default_head_dim() -> i32 {
    72
}
const fn default_patch_size() -> i32 {
    16
}
const fn default_pooling_kernel_size() -> i32 {
    3
}
const fn default_default_output_length() -> i32 {
    280
}
const fn default_position_embedding_size() -> i32 {
    10240
}
const fn default_rms_norm_eps() -> f32 {
    1e-6
}
const fn default_rope_theta() -> f32 {
    100.0
}
const fn default_standardize() -> bool {
    true
}
