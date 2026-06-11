//! Gemma 4 Unified (`gemma4_unified`, gemma-4-12B) rewrite rules.
//!
//! The released checkpoint key layout already matches what `mlx_lm::load`
//! consumes — the gemma4 text loader strips `language_model.model.`→`model.`
//! and the VLM loader binds `vision_embedder.*` / `embed_vision.*` /
//! `embed_audio.*` directly. So this rewriter renames nothing; it only decides
//! per-tensor quantisation:
//!
//! - **Body** (true uniform q4): all projection `.weight`s — attention
//!   (`q/k/v/o_proj`), MLP (`gate/down/up_proj`), the tied `embed_tokens`
//!   table, and the encoder-free MM projections (`patch_dense`,
//!   `embed_vision`/`embed_audio.embedding_projection`).
//! - **Skip** (keep bf16): every norm (RMSNorm `*_layernorm`, `q/k_norm`,
//!   `model.norm`; vision LayerNorms `patch_ln1/2`, `pos_norm`), the
//!   `layer_scalar` per-layer scalars, the `pos_embedding` table, and all
//!   `.bias` tensors.
//!
//! Unlike mlx-community's mixed "4bit" (MLP pinned to 8-bit), this produces a
//! genuinely uniform q4 so decode bytes/token match a 6.5 GB LiteRT build.

use mlx_rs::Array;

use crate::plan::{QuantClass, RewriteOutput, Rewriter};
use crate::Result;

/// Gemma 4 Unified rewriter. `drop_mm` strips the encoder-free vision/audio
/// front-ends for a text-only convert; the default (`false`) keeps them for a
/// full-MM q4.
#[derive(Default)]
pub struct Gemma4UnifiedRewriter {
    pub drop_mm: bool,
}

impl Rewriter for Gemma4UnifiedRewriter {
    fn name(&self) -> &'static str {
        "gemma4_unified"
    }

    fn skip_source(&self, src_key: &str) -> bool {
        self.drop_mm && is_multimodal_key(src_key)
    }

    fn rewrite(&self, src_key: &str, src_tensor: Array) -> Result<RewriteOutput> {
        // Keys are already in the loader's expected form — no rename.
        Ok(vec![(src_key.to_owned(), src_tensor, classify(src_key))])
    }
}

/// Encoder-free vision/audio front-end keys, dropped when `drop_mm`.
fn is_multimodal_key(key: &str) -> bool {
    key.starts_with("vision_embedder.")
        || key.starts_with("embed_vision.")
        || key.starts_with("embed_audio.")
}

/// Per-tensor quant class. Body-quantise projection `.weight`s; skip norms,
/// scalars, the positional table, and biases.
fn classify(key: &str) -> QuantClass {
    if is_skip(key) {
        return QuantClass::Skip;
    }
    if key.ends_with(".weight") {
        return QuantClass::Body;
    }
    // layer_scalar, pos_embedding, anything non-.weight → keep as-is.
    QuantClass::Skip
}

/// Tensors that must stay in source dtype: every norm weight (RMSNorm +
/// vision LayerNorm), and all biases. `.bias` + the norm suffixes below cover
/// the full skip set; `layer_scalar` / `pos_embedding` fall through `classify`
/// as non-`.weight`.
fn is_skip(key: &str) -> bool {
    if key.ends_with(".bias") {
        return true;
    }
    const NORM_SUFFIXES: &[&str] = &[
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".pre_feedforward_layernorm.weight",
        ".post_feedforward_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        "model.norm.weight",
        // encoder-free vision LayerNorms (weight + bias) — keep bf16.
        ".patch_ln1.weight",
        ".patch_ln2.weight",
        ".pos_norm.weight",
    ];
    NORM_SUFFIXES.iter().any(|s| key.ends_with(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_weights_are_body_quantised() {
        for k in [
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.0.mlp.gate_proj.weight",
            "language_model.model.embed_tokens.weight",
            "vision_embedder.patch_dense.weight",
            "embed_vision.embedding_projection.weight",
            "embed_audio.embedding_projection.weight",
        ] {
            assert_eq!(classify(k), QuantClass::Body, "{k}");
        }
    }

    #[test]
    fn norms_scalars_biases_table_are_skipped() {
        for k in [
            "language_model.model.layers.0.input_layernorm.weight",
            "language_model.model.layers.0.self_attn.q_norm.weight",
            "language_model.model.norm.weight",
            "language_model.model.layers.0.layer_scalar",
            "vision_embedder.patch_ln1.weight",
            "vision_embedder.pos_norm.weight",
            "vision_embedder.pos_embedding",
            "vision_embedder.patch_dense.bias",
        ] {
            assert_eq!(classify(k), QuantClass::Skip, "{k}");
        }
    }

    #[test]
    fn drop_mm_skips_only_multimodal() {
        let r = Gemma4UnifiedRewriter { drop_mm: true };
        assert!(r.skip_source("vision_embedder.patch_dense.weight"));
        assert!(r.skip_source("embed_audio.embedding_projection.weight"));
        assert!(!r.skip_source("language_model.model.layers.0.mlp.gate_proj.weight"));
        // Default keeps multimodal.
        let keep = Gemma4UnifiedRewriter::default();
        assert!(!keep.skip_source("vision_embedder.patch_dense.weight"));
    }
}
