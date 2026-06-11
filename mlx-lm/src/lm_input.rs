//! Model-facing input.
//!
//! [`crate::UserInputProcessor`] turns a [`crate::UserInput`] into an
//! [`LMInput`]; [`crate::LanguageModel::prepare`] consumes it to seed
//! the KV cache, then [`crate::LanguageModel::step`] consumes one token
//! id at a time.

use mlx_rs::Array;

/// Output of a [`crate::UserInputProcessor::prepare`] call.
///
/// `image` is independent of `text`: a VLM processor sets it to `Some`,
/// a text-only request leaves it `None`. Gated on the `image` feature.
#[derive(Debug)]
pub struct LMInput {
    pub text: Text,

    /// Pre-processed image tensor(s) for the vision tower. `None` for
    /// text-only requests or models that don't accept images.
    #[cfg(feature = "image")]
    pub image: Option<ProcessedImage>,

    /// Pre-processed log-mel audio for the audio tower. `None` for
    /// text/image-only requests or models that don't accept audio.
    #[cfg(feature = "audio")]
    pub audio: Option<ProcessedAudio>,
}

/// Tokenised text portion of an [`LMInput`].
#[derive(Debug)]
pub struct Text {
    /// `[1, S]` int32 token ids (batch dim always 1).
    pub tokens: Array,
    /// Optional `[1, S]` mask; `None` lets the model build its own.
    pub mask: Option<Array>,
}

/// Pre-processed image tensor(s), ready for the model's vision tower.
/// The processor handles per-family normalisation, patch packing, and
/// the temporal/height/width grid metadata.
#[cfg(feature = "image")]
#[derive(Debug)]
pub struct ProcessedImage {
    /// `[num_patches, feature_dim]` `f32` array. Patches are stacked
    /// across all images in the prompt; `grids` records the per-image
    /// `(t, h, w)` so the model can slice them apart.
    pub pixels: Array,

    /// One `[t, h, w]` patch-grid per image, in `UserInput::images` order.
    pub grids: Vec<[i32; 3]>,

    /// Per-patch 2D position ids `[num_images, num_patches, 2]` (`-1` padding)
    /// for encoder-free vision (gemma4_unified). `None` for tower-based
    /// families (qwen3.5, gemma4 SigLIP) that derive positions internally.
    pub position_ids: Option<Array>,
}

/// Pre-processed audio features for the model's audio path (one clip per turn).
/// `[1, T, feat]`: log-mel (`feat=128`) for the gemma4 USM tower, or raw frames
/// (`feat=audio_samples_per_token`) for the gemma4_unified encoder-free path.
#[cfg(feature = "audio")]
#[derive(Debug)]
pub struct ProcessedAudio {
    pub features: Array,
}

/// Result of [`crate::LanguageModel::prepare`]: logits to sample now, or
/// "cache primed, call `step`".
pub enum PrepareResult {
    Primed,
    Logits(Array),
}

/// One step's output from [`crate::LanguageModel::step`].
pub struct LMOutput {
    /// `[1, 1, vocab_size]` logits over the next token.
    pub logits: Array,
}
