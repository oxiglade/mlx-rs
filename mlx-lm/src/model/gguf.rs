//! GGUF normalization for dense, biasless Llama and Qwen3 with full attention.
//!
//! Metadata parsing must follow the key order in the tranche 4 paper, section 2.4.
//! Shape-derived vocabulary, head width, and tying remain separate from metadata;
//! architecture validators must cross-check them before decoder allocation.

use crate::{LoadError, Model, TokenId, Tokenizer};
use mlx_rs::io::GgufFile;
use std::num::NonZeroUsize;

/// Prepares owned backing, canonical names and Llama rows before strict factory loading.
#[allow(dead_code)] // The public GGUF wrapper is wired by the tranche 4 model owner.
pub(crate) fn load(_file: GgufFile, _tokenizer: Tokenizer) -> Result<Model, LoadError> {
    Err(crate::WeightError::UnsupportedFormat(
        crate::NotYetImplemented("GGUF model loading").to_string(),
    )
    .into())
}

/// Exact `general.architecture` values admitted by the GGUF profile.
#[allow(dead_code)] // GGUF metadata parsing is tranche 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GgufArchitecture {
    /// Llama requires inverse Q/K row permutation.
    Llama,
    /// Qwen3 keeps Q/K rows and requires per-head Q/K norms.
    Qwen3,
}

/// Scaling admitted after rejecting orphan factors and unsupported scaling types.
#[allow(dead_code)] // GGUF metadata parsing is tranche 4.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GgufRopeScaling {
    /// Absent or explicit `none`, without a scaling factor.
    None,
    /// Explicit `linear` with a finite positive F32 factor.
    Linear {
        /// Checked scalar `A.rope.scaling.factor`.
        factor: f32,
    },
}

/// Typed metadata for the section 2.4 profile, before tensor-shape reconciliation.
///
/// Integers must be integral scalars excluding bool; real values must be F32
/// scalars. Length-one vectors do not satisfy either scalar requirement. Known
/// semantic alternatives are rejected unless causal is true, parallel residual
/// is false, and sliding window is zero. Tensor data layout and attention scale
/// must be absent. Unknown descriptive metadata may be present.
#[allow(dead_code)] // GGUF metadata parsing is tranche 4.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GgufMetadataProfile {
    /// Required `general.architecture` without aliases or tensor-name inference.
    pub(crate) architecture: GgufArchitecture,
    /// Required positive `A.block_count`.
    pub(crate) layer_count: NonZeroUsize,
    /// Required positive `A.embedding_length`.
    pub(crate) hidden_size: NonZeroUsize,
    /// Required positive scalar `A.feed_forward_length`.
    pub(crate) intermediate_size: NonZeroUsize,
    /// Required positive `A.attention.head_count`.
    pub(crate) attention_heads: NonZeroUsize,
    /// Required positive `A.attention.head_count_kv`; must divide query heads.
    pub(crate) kv_heads: NonZeroUsize,
    /// Required finite positive `A.attention.layer_norm_rms_epsilon`.
    pub(crate) rms_norm_epsilon: f32,
    /// Required finite positive `A.rope.freq_base`.
    pub(crate) rope_theta: f32,
    /// Optional positive `A.attention.key_length`, checked against Q/K/V rows.
    pub(crate) key_length: Option<NonZeroUsize>,
    /// Optional positive `A.attention.value_length`, checked against Q/K/V rows.
    pub(crate) value_length: Option<NonZeroUsize>,
    /// Optional `A.rope.dimension_count`; must equal the even derived head width.
    pub(crate) rope_dimensions: Option<NonZeroUsize>,
    /// Optional positive `A.context_length`, without an added generation cap.
    pub(crate) context_length: Option<NonZeroUsize>,
    /// Optional positive `A.vocab_size`, checked against embedding rows.
    pub(crate) vocabulary_size: Option<NonZeroUsize>,
    /// Explicitly bounded scaling; absence resolves to no scaling.
    pub(crate) rope_scaling: GgufRopeScaling,
    /// Optional positive `A.rope.scaling.original_context_length`, informative only.
    pub(crate) original_context_length: Option<NonZeroUsize>,
    /// Optional scalar bool `A.rope.scaling.finetuned`, informative only.
    pub(crate) rope_finetuned: Option<bool>,
    /// Embedded facts for consistency checks with the caller's tokenizer.
    pub(crate) tokenizer: GgufTokenizerMetadata,
    /// Optional integral `general.file_type`; never a per-matrix layout oracle.
    pub(crate) file_type: Option<i128>,
    /// Optional integral `general.quantization_version`, diagnostic only.
    pub(crate) quantization_version: Option<i128>,
}

/// Optional embedded facts cannot construct or override the supplied tokenizer.
#[allow(dead_code)] // GGUF tokenizer pairing is tranche 4.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GgufTokenizerMetadata {
    /// `tokenizer.ggml.tokens`, whose count cannot exceed embedding rows.
    pub(crate) tokens: Option<Vec<String>>,
    /// Nonnegative scalar `tokenizer.ggml.bos_token_id`, checked against the caller.
    pub(crate) bos_token_id: Option<TokenId>,
    /// Nonnegative scalar `tokenizer.ggml.eos_token_id`, checked against resolved EOS.
    pub(crate) eos_token_id: Option<TokenId>,
}
