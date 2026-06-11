//! Canonical per-layer attention input.

use mlx_rs::Array;

/// Per-layer attention input used by llama and qwen3.
pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
}
