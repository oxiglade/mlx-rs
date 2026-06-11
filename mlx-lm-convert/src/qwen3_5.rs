//! Qwen 3.5 / 3.6 (dense + MoE) rewrite rules, including MTP.
//!
//! Translates the Qwen-released checkpoint key layout into the one that
//! `mlx_lm::load` (Rust) expects.
//!
//! Source convention (`Qwen/Qwen3.6-35B-A3B` bf16):
//!   `model.language_model.layers.N.…`           main decoder
//!   `model.language_model.embed_tokens.weight`  embedding
//!   `mtp.layers.0.…`                            MTP layer (no language_model prefix)
//!   `mtp.{fc,norm,pre_fc_norm_*}.weight`        MTP head plumbing
//!   `model.language_model.layers.N.mlp.experts.gate_up_proj`  [E, 2H, D] packed
//!   `model.language_model.layers.N.mlp.experts.down_proj`     [E, D, H]
//!
//! Destination convention (mlx-rs param walk):
//!   `language_model.model.layers.N.…`
//!   `language_model.model.embed_tokens.…`
//!   `language_model.mtp.layers.0.…`
//!   `language_model.mtp.{fc,norm,pre_fc_norm_*}.…`
//!   `…mlp.switch_mlp.{gate_proj,up_proj}.weight`  split [E, H, D] each
//!   `…mlp.switch_mlp.down_proj.weight`
//!
//! The bucket walks (`bucket_key` in `mlx-lm`) strip the
//! `language_model.` prefix and route the remainder into the
//! `LanguageModel<F>` param tree. MTP keys come along for the ride and
//! bind to the new `MtpHead` field.

use mlx_rs::ops::split_sections;
use mlx_rs::Array;

use crate::plan::{QuantClass, RewriteOutput, Rewriter};
use crate::{anyhow, Result};

/// Qwen 3.5 / 3.6 dense + MoE rewriter. The rules are identical for
/// dense and MoE; the MoE-specific `experts.*` keys simply don't appear
/// on a dense checkpoint, and the rule is a no-op there.
pub struct Qwen35Rewriter {
    /// When true, the source checkpoint's `vision_tower.*` and
    /// `model.visual.*` keys are skipped (use this for text-only
    /// conversion of a VLM checkpoint).
    pub drop_vision: bool,
}

impl Default for Qwen35Rewriter {
    fn default() -> Self {
        Self { drop_vision: true }
    }
}

impl Rewriter for Qwen35Rewriter {
    fn name(&self) -> &'static str {
        "qwen3_5"
    }

    fn skip_source(&self, src_key: &str) -> bool {
        self.drop_vision
            && (src_key.starts_with("vision_tower.") || src_key.starts_with("model.visual."))
    }

    fn rewrite(&self, src_key: &str, src_tensor: Array) -> Result<RewriteOutput> {
        let dst_prefix = sanitize_prefix(src_key);
        if let Some(out) = try_split_experts_gate_up(&dst_prefix, &src_tensor)? {
            return Ok(out);
        }
        if let Some(out) = try_rename_experts_down(&dst_prefix, src_tensor.clone()) {
            return Ok(out);
        }
        Ok(vec![(
            dst_prefix,
            src_tensor.clone(),
            classify_default(src_key),
        )])
    }
}

/// Apply the Qwen 3.5 sanitize-key rules. The HF release wraps the LM
/// under `model.language_model.…`; the mlx-rs loader expects
/// `language_model.model.…`. MTP keys are unprefixed in the source —
/// the loader's `bucket_key` requires a `language_model.` prefix to
/// route them into the LM walk, so we add it.
fn sanitize_prefix(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("model.language_model.") {
        return format!("language_model.model.{rest}");
    }
    if let Some(rest) = key.strip_prefix("model.visual.") {
        return format!("vision_tower.{rest}");
    }
    if let Some(rest) = key.strip_prefix("lm_head") {
        return format!("language_model.lm_head{rest}");
    }
    if let Some(rest) = key.strip_prefix("mtp.") {
        return format!("language_model.mtp.{rest}");
    }
    key.to_owned()
}

/// `…mlp.experts.gate_up_proj` is `[E, 2H, D]`. Split along axis 1 into
/// two `[E, H, D]` tensors and rename to the `switch_mlp.{gate,up}_proj`
/// pair the loader expects. Returns `None` if `key` isn't this exact
/// suffix.
fn try_split_experts_gate_up(dst_key: &str, tensor: &Array) -> Result<Option<RewriteOutput>> {
    let Some(stem) = dst_key.strip_suffix(".mlp.experts.gate_up_proj") else {
        return Ok(None);
    };
    let shape = tensor.shape();
    if shape.len() != 3 {
        return Err(anyhow!(
            "{dst_key}: expected 3-D [E, 2H, D] tensor, got {shape:?}"
        ));
    }
    if shape[1] % 2 != 0 {
        return Err(anyhow!(
            "{dst_key}: middle dim {} is not even, can't split into gate+up",
            shape[1]
        ));
    }
    let half = shape[1] / 2;
    let parts = split_sections(tensor, &[half], 1)?;
    if parts.len() != 2 {
        return Err(anyhow!(
            "{dst_key}: split_sections returned {} parts (expected 2)",
            parts.len()
        ));
    }
    let mut it = parts.into_iter();
    let gate = it.next().expect("split_sections returned 2 parts");
    let up = it.next().expect("split_sections returned 2 parts");
    Ok(Some(vec![
        (
            format!("{stem}.mlp.switch_mlp.gate_proj.weight"),
            gate,
            QuantClass::Body,
        ),
        (
            format!("{stem}.mlp.switch_mlp.up_proj.weight"),
            up,
            QuantClass::Body,
        ),
    ]))
}

/// `…mlp.experts.down_proj` → `…mlp.switch_mlp.down_proj.weight`. The
/// tensor is moved verbatim; only the key changes.
fn try_rename_experts_down(dst_key: &str, tensor: Array) -> Option<RewriteOutput> {
    let stem = dst_key.strip_suffix(".mlp.experts.down_proj")?;
    Some(vec![(
        format!("{stem}.mlp.switch_mlp.down_proj.weight"),
        tensor,
        QuantClass::Body,
    )])
}

/// Default per-tensor quantisation class for anything that didn't hit a
/// rename rule. Norms, biases, conv1d, embeds, MTP norms stay bf16;
/// the two pinned-to-q8 paths get [`QuantClass::Pinned`]; everything
/// else `*.weight` is body-class.
fn classify_default(src_key: &str) -> QuantClass {
    if is_norm_or_skip(src_key) {
        return QuantClass::Skip;
    }
    if src_key.ends_with("mlp.gate.weight") || src_key.ends_with("mlp.shared_expert_gate.weight") {
        return QuantClass::Pinned {
            group_size: 64,
            bits: 8,
        };
    }
    if src_key.ends_with(".weight") {
        return QuantClass::Body;
    }
    QuantClass::Skip
}

/// Suffixes that should never be quantised: layernorm scales, scalar
/// gates, conv1d kernels in GDN layers, biases. Embedding tables ARE
/// quantised (see the lmstudio-community Qwen3.6-35B-A3B-MLX-8bit
/// reference) — only their `.weight` carries the table, no special
/// handling needed.
fn is_norm_or_skip(key: &str) -> bool {
    const SKIP_SUFFIXES: &[&str] = &[
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        ".linear_attn.norm.weight",
        ".linear_attn.A_log",
        ".linear_attn.dt_bias",
        ".pre_fc_norm_hidden.weight",
        ".pre_fc_norm_embedding.weight",
        "model.norm.weight",
        "mtp.norm.weight",
        // mtp.fc is kept in full precision: vLLM's MTP loader expects
        // bf16 here and quantising it visibly degrades draft quality.
        "mtp.fc.weight",
        ".bias",
        ".conv1d.weight",
    ];
    SKIP_SUFFIXES.iter().any(|s| key.ends_with(s))
}
