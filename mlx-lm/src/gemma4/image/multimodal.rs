//! Stitch vision features into the text embedding sequence.

use mlx_rs::Array;

use crate::error::Error;
use crate::qwen3_5::image::multimodal::merge_input_ids_with_image_features;

/// Scatter `image_features` `[N, hidden]` into the `image_token_id` slots of
/// `inputs_embeds` `[B, S, hidden]`. Gemma has no separate video token, so the
/// image id covers both image/video args of the shared merge helper.
pub fn stitch_image_features(
    image_features: &Array,
    inputs_embeds: &Array,
    input_ids: &Array,
    image_token_id: u32,
) -> Result<Array, Error> {
    merge_input_ids_with_image_features(
        image_features,
        inputs_embeds,
        input_ids,
        image_token_id,
        image_token_id,
    )
}
