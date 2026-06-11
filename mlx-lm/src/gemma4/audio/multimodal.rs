//! Stitch audio features into the text embedding sequence.

use mlx_rs::Array;

use crate::error::Error;
use crate::qwen3_5::image::multimodal::merge_input_ids_with_image_features;

/// Scatter `audio_features` `[N, hidden]` into the `audio_token_id` slots of
/// `inputs_embeds` `[B, S, hidden]`. Reuses the shared masked-scatter helper
/// (audio id covers both image/video args — there are no other special slots).
pub fn stitch_audio_features(
    audio_features: &Array,
    inputs_embeds: &Array,
    input_ids: &Array,
    audio_token_id: u32,
) -> Result<Array, Error> {
    merge_input_ids_with_image_features(
        audio_features,
        inputs_embeds,
        input_ids,
        audio_token_id,
        audio_token_id,
    )
}
