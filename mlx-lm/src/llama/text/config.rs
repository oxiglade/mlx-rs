//! Llama config body. `model_type` is the serde tag on
//! [`crate::config::Family`]; quantization lives on the outer
//! [`crate::config::ModelConfig`].

use std::collections::HashMap;

use serde::Deserialize;

use crate::family::EosSpec;
use crate::utils::rope::FloatOrString;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub num_key_value_heads: i32,
    pub max_position_embeddings: i32,
    pub rope_theta: f32,
    pub head_dim: i32,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    #[serde(default)]
    pub eos_token_id: Option<EosSpec>,
}

fn default_true() -> bool {
    true
}
