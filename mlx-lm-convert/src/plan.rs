//! Per-model rewrite rules.
//!
//! A [`Rewriter`] decides three things for every tensor loaded from a
//! source safetensors shard:
//!
//! 1. The destination key name (rename + prefix rules).
//! 2. Whether to split a single tensor into multiple (e.g. the Qwen 3.6
//!    `experts.gate_up_proj` packed weight into separate `gate_proj` +
//!    `up_proj`).
//! 3. Whether to quantise the result, and at what bits/group_size — some
//!    tensors are pinned to a per-model override (Qwen 3.6 routers and
//!    shared-expert gates always at gs=64 bits=8).

use mlx_rs::Array;

use crate::Result;

/// One rewrite step's output: zero or more `(dst_key, tensor, quant_class)`
/// triples. Most rules produce exactly one; the gate-up split produces two.
pub type RewriteOutput = Vec<(String, Array, QuantClass)>;

/// What the quantisation pass should do to a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantClass {
    /// Skip quantisation — keep the tensor in its source dtype. Norms,
    /// embed_tokens, biases.
    Skip,
    /// Quantise at the user-chosen body bits/group_size.
    Body,
    /// Quantise at a per-model-pinned bits/group_size regardless of body
    /// settings. Qwen 3.6 router (`mlp.gate`) and `shared_expert_gate`
    /// always live at (gs=64, bits=8).
    Pinned { group_size: i32, bits: i32 },
}

/// Family-specific rewrite rules. Stateless — one shared impl per model
/// family, called for every key.
pub trait Rewriter: Send + Sync {
    /// Human-readable family name used in logs (`"qwen3_5"`,
    /// `"qwen3_5_moe"`).
    fn name(&self) -> &'static str;

    /// Rewrite one source key + tensor into one or more destination
    /// (key, tensor, quant_class) entries. Implementors should bail with
    /// [`anyhow::anyhow!`] on unexpected shapes rather than silently
    /// dropping tensors.
    fn rewrite(&self, src_key: &str, src_tensor: Array) -> Result<RewriteOutput>;

    /// Whether the source key should be dropped entirely (vision-tower
    /// weights on a text-only convert, for example). Default keeps
    /// everything.
    fn skip_source(&self, _src_key: &str) -> bool {
        false
    }
}
