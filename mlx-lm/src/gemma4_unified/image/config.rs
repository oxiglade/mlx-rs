//! Gemma 4 Unified vision config (`vision_config`, `gemma4_unified_vision`).
//!
//! Encoder-free: a LayerNorm → dense → LayerNorm stack, factorised 2D
//! positional embedding, a final norm, then an RMSNorm(no-scale) → linear
//! projection into the text embedding space. No SigLIP tower.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    /// Vision hidden width (`patch_dense` output, `pos_embedding` width).
    pub mm_embed_dim: i32,
    /// Rows in the factorised 2D positional-embedding table `[N, 2, D]`.
    pub mm_posemb_size: i32,
    /// Merged-patch edge in pixels (`patch_size · pooling_kernel_size`).
    pub model_patch_size: i32,
    /// Teacher-patch edge in pixels.
    pub patch_size: i32,
    /// Square pooling window merging teacher patches into model patches.
    pub pooling_kernel_size: i32,
    /// Soft-token budget per image (max model patches after pooling).
    pub num_soft_tokens: i32,
    /// `embedding_projection` output width (= text hidden size).
    pub output_proj_dims: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

impl VisionConfig {
    /// Flattened merged-patch feature dim: `(model_patch_size)² · 3`.
    pub fn patch_dim(&self) -> i32 {
        self.model_patch_size * self.model_patch_size * 3
    }
}
