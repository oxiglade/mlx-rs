//! Canonical model-level input.

use mlx_rs::Array;

/// Top-level model input. `mask` is `None` for models that build the
/// attention mask internally; llama/qwen3 pass `Some`.
pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}
