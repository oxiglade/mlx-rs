//! Multimodal stitching helpers.
//!
//! - [`merge_input_ids_with_image_features`] replaces every image-token slot
//!   in a token-embedding sequence with the corresponding row of a vision-
//!   tower output.
//! - [`get_rope_index_single_batch`] computes the `[3, B, S]` `mrope` position
//!   ids the text decoder consumes when there are vision tokens in the prompt.
//!
//! [`get_rope_index_single_batch`] is B=1 by design; callers needing
//! multiple rows use [`get_rope_index_batched`].

use mlx_rs::{
    ops::{
        broadcast_to, concatenate_axis, cumsum, expand_dims, indexing::take_axis, maximum, r#where,
        reshape,
    },
    Array, Dtype,
};

use crate::error::Error;

/// Splice `image_features` into `inputs_embeds` at every position where
/// `input_ids` equals `image_token_id` (or `video_token_id`).
///
/// - `inputs_embeds`: `[B, S, hidden]` — typically the result of running
///   `embed_tokens(input_ids)`.
/// - `image_features`: `[N_image_tokens, hidden]` — the vision tower's
///   `merger` output for the whole batch concatenated along axis 0.
/// - `input_ids`: `[B, S]` `int32` token ids.
///
/// Returns the new `[B, S, hidden]` embedding sequence.
pub fn merge_input_ids_with_image_features(
    image_features: &Array,
    inputs_embeds: &Array,
    input_ids: &Array,
    image_token_id: u32,
    video_token_id: u32,
) -> Result<Array, Error> {
    let emb_shape = inputs_embeds.shape();
    if emb_shape.len() != 3 {
        return Err(Error::shape(
            "merge_input_ids_with_image_features: inputs_embeds must be [B, S, hidden]",
        ));
    }
    let hidden = emb_shape[2];

    let is_image = input_ids.eq(Array::from_int(image_token_id as i32))?;
    let is_video = input_ids.eq(Array::from_int(video_token_id as i32))?;
    let is_special = is_image.logical_or(&is_video)?;

    // Count image-token slots so we can sanity-check the vision-feature count.
    let n_special: i32 = is_special.sum(None)?.item::<i32>();
    if (n_special as i64) * (hidden as i64) != image_features.size() as i64 {
        return Err(Error::shape(format!(
            "merge_input_ids_with_image_features: image_features has {} elements but \
             input_ids contains {n_special} special positions × {hidden} hidden = {}",
            image_features.size(),
            (n_special as i64) * (hidden as i64),
        )));
    }
    if n_special == 0 {
        return Ok(inputs_embeds.clone());
    }

    // Flatten the embeddings to [B*S, hidden] and the mask to [B*S]. Scatter
    // the image features into the mask-true positions in row-major order.
    let bs = emb_shape[0] * emb_shape[1];
    let flat_embeds = reshape(inputs_embeds, &[bs, hidden])?;
    let flat_special = reshape(&is_special, &[bs])?;

    // Inclusive prefix sum -> at position `i` we have the 1-based count of
    // special tokens up to and including `i`. Subtract 1 to get the index of
    // the matching image-feature row; clamp negatives and mask non-special
    // positions back to the original embedding via `where`.
    let cum_special = cumsum(&flat_special.as_dtype(Dtype::Int32)?, 0, false, true)?;
    let one = Array::from_int(1);
    let scatter_idx = cum_special.subtract(&one)?;
    let zero_i = Array::from_int(0);
    let clamped = maximum(&scatter_idx, &zero_i)?;
    let gathered = take_axis(image_features, &clamped, 0)?;

    let mask_h = expand_dims(&flat_special, 1)?;
    let mask_h = broadcast_to(&mask_h, &[bs, hidden])?;
    let merged = r#where(
        &mask_h,
        &gathered.as_dtype(flat_embeds.dtype())?,
        &flat_embeds,
    )?;

    Ok(reshape(&merged, emb_shape)?)
}

/// Compute the `mrope_position_ids` for a single image-bearing prompt.
/// Returns `(t_pos, h_pos, w_pos, rope_delta)` where `rope_delta` is the
/// per-batch offset between the prompt length and the max position id (the
/// decode loop adds it to keep mrope coherent past the prompt).
///
/// Inputs (single sequence):
///
/// - `input_ids`: row-major Vec of token ids, length `S`.
/// - `image_grid_thw`: per-image `[t, h, w]` grids. Each image consumes
///   `t * h/merge * w/merge` positions in the sequence.
/// - `vision_start_token_id`: token id immediately preceding an image's
///   special-token block; defines where each image starts.
/// - `image_token_id` / `video_token_id`: ids of the placeholder tokens.
/// - `spatial_merge_size`: the vision tower's `spatial_merge_size`.
///
/// `position_ids[0]` is the temporal axis, `[1]` height, `[2]` width. Text
/// tokens advance all three uniformly; image tokens advance independently.
#[allow(
    clippy::type_complexity,
    reason = "rope-index tuple is the model's natural return shape"
)]
pub fn get_rope_index_single_batch(
    input_ids: &[i32],
    image_grid_thw: &[[i32; 3]],
    spatial_merge_size: i32,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>, i32), Error> {
    let s = input_ids.len();
    let mut t_pos = Vec::with_capacity(s);
    let mut h_pos = Vec::with_capacity(s);
    let mut w_pos = Vec::with_capacity(s);

    let mut next_pos: i32 = 0;
    let mut i = 0;
    let mut img_idx = 0;
    while i < s {
        let tok = input_ids[i] as u32;
        // Both branches advance one position uniformly across all 3 axes.
        t_pos.push(next_pos);
        h_pos.push(next_pos);
        w_pos.push(next_pos);
        i += 1;
        next_pos += 1;
        if tok == vision_start_token_id {
            // Read the per-axis grid for this image.
            if img_idx >= image_grid_thw.len() {
                return Err(Error::out_of_bounds(
                    "get_rope_index: vision_start token without matching grid_thw entry",
                ));
            }
            let [t, h, w] = image_grid_thw[img_idx];
            img_idx += 1;
            let merged_h = h / spatial_merge_size;
            let merged_w = w / spatial_merge_size;
            let total_image_tokens = (t as usize) * (merged_h as usize) * (merged_w as usize);
            if i + total_image_tokens > s {
                return Err(Error::out_of_bounds(format!(
                    "get_rope_index: not enough image-token slots: need {total_image_tokens}, have {}",
                    s - i
                )));
            }
            for ti in 0..t {
                for hi in 0..merged_h {
                    for wi in 0..merged_w {
                        // Verify each slot is actually an image / video token.
                        if (input_ids[i] as u32) != image_token_id
                            && (input_ids[i] as u32) != video_token_id
                        {
                            return Err(Error::shape(format!(
                                "get_rope_index: expected image token at index {i}, got {}",
                                input_ids[i]
                            )));
                        }
                        t_pos.push(next_pos + ti);
                        h_pos.push(next_pos + hi);
                        w_pos.push(next_pos + wi);
                        i += 1;
                    }
                }
            }
            // Bump next_pos past the image block.
            let max_axis = t.max(merged_h).max(merged_w);
            next_pos += max_axis;
        }
    }

    let max_pos = *t_pos.iter().max().unwrap_or(&0);
    let rope_delta = max_pos + 1 - (s as i32);
    Ok((t_pos, h_pos, w_pos, rope_delta))
}

/// Pack `[3, 1, S]` mrope position ids into an MLX `Array` for the LM.
pub fn pack_position_ids(t_pos: &[i32], h_pos: &[i32], w_pos: &[i32]) -> Result<Array, Error> {
    let s = t_pos.len() as i32;
    if h_pos.len() as i32 != s || w_pos.len() as i32 != s {
        return Err(Error::shape("pack_position_ids: axis length mismatch"));
    }
    let t_arr = Array::from_slice(t_pos, &[1, 1, s]);
    let h_arr = Array::from_slice(h_pos, &[1, 1, s]);
    let w_arr = Array::from_slice(w_pos, &[1, 1, s]);
    let stacked = concatenate_axis(&[t_arr, h_arr, w_arr], 0)?;
    Ok(stacked)
}

/// Multi-batch version of [`get_rope_index_single_batch`]. Each `batch_rows`
/// entry is one sequence's input ids; `image_grid_thw_per_row` is the per-row
/// list of image grids (rows without images pass `&[]`). All rows must share
/// the same sequence length `S`. Returns `([3, B, S]` ids, per-row deltas)`.
pub fn get_rope_index_batched(
    batch_rows: &[&[i32]],
    image_grid_thw_per_row: &[&[[i32; 3]]],
    spatial_merge_size: i32,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
) -> Result<(Array, Vec<i32>), Error> {
    if batch_rows.is_empty() {
        return Err(Error::config("get_rope_index_batched: batch_rows is empty"));
    }
    if batch_rows.len() != image_grid_thw_per_row.len() {
        return Err(Error::shape(format!(
            "get_rope_index_batched: batch_rows.len()={} but image_grid_thw_per_row.len()={}",
            batch_rows.len(),
            image_grid_thw_per_row.len()
        )));
    }
    let s = batch_rows[0].len();
    let mut t_all: Vec<i32> = Vec::with_capacity(batch_rows.len() * s);
    let mut h_all: Vec<i32> = Vec::with_capacity(batch_rows.len() * s);
    let mut w_all: Vec<i32> = Vec::with_capacity(batch_rows.len() * s);
    let mut deltas: Vec<i32> = Vec::with_capacity(batch_rows.len());
    for (ids, grids) in batch_rows.iter().zip(image_grid_thw_per_row.iter()) {
        if ids.len() != s {
            return Err(Error::shape(format!(
                "get_rope_index_batched: row length mismatch: expected {s}, got {}",
                ids.len()
            )));
        }
        let (t, h, w, delta) = get_rope_index_single_batch(
            ids,
            grids,
            spatial_merge_size,
            image_token_id,
            video_token_id,
            vision_start_token_id,
        )?;
        t_all.extend(t);
        h_all.extend(h);
        w_all.extend(w);
        deltas.push(delta);
    }
    let b = batch_rows.len() as i32;
    let s_i32 = s as i32;
    let t_arr = Array::from_slice(&t_all, &[1, b, s_i32]);
    let h_arr = Array::from_slice(&h_all, &[1, b, s_i32]);
    let w_arr = Array::from_slice(&w_all, &[1, b, s_i32]);
    let pos = concatenate_axis(&[t_arr, h_arr, w_arr], 0)?;
    Ok((pos, deltas))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;
    use mlx_rs::{random::uniform, transforms::eval};

    #[test]
    fn rope_index_text_only_advances_uniformly() {
        let (t, h, w, delta) = get_rope_index_single_batch(
            &[5, 6, 7, 8],
            &[],
            2,
            248_056, // image
            248_057, // video
            248_053, // vision_start
        )
        .unwrap();
        assert_eq!(t, vec![0, 1, 2, 3]);
        assert_eq!(h, vec![0, 1, 2, 3]);
        assert_eq!(w, vec![0, 1, 2, 3]);
        assert_eq!(delta, 0);
    }

    #[test]
    fn rope_index_single_image_consumes_grid() {
        // 2 text tokens, then vision_start, then 2*2*2 = 8 image tokens
        // (t=2, h=4 -> merged=2, w=4 -> merged=2), then 1 text token.
        let vs = 248_053_u32 as i32;
        let img = 248_056_u32 as i32;
        let mut ids = vec![10, 11, vs];
        ids.extend(std::iter::repeat_n(img, 8));
        ids.push(12);
        let (t, h, w, delta) =
            get_rope_index_single_batch(&ids, &[[2, 4, 4]], 2, 248_056, 248_057, 248_053).unwrap();
        assert_eq!(t.len(), ids.len());
        assert_eq!(t[0..2], [0, 1]);
        assert_eq!(t[2], 2);
        assert_eq!(h[2], 2);
        assert_eq!(w[2], 2);
        assert_eq!(t[3], 3); // first image patch
        assert_eq!(h[3], 3);
        assert_eq!(w[3], 3);
        assert_eq!(t[10], 4); // second t-slice last patch
        assert_eq!(h[10], 4);
        assert_eq!(w[10], 4);
        assert_eq!(t[11], 5); // text after image
        assert_eq!(delta, -6);
    }

    #[test]
    fn rope_index_batched_text_only_two_rows() {
        let row_a: Vec<i32> = vec![5, 6, 7, 8];
        let row_b: Vec<i32> = vec![10, 11, 12, 13];
        let (pos, deltas) =
            get_rope_index_batched(&[&row_a, &row_b], &[&[], &[]], 2, 248_056, 248_057, 248_053)
                .unwrap();
        assert_eq!(pos.shape(), &[3, 2, 4]);
        assert_eq!(deltas, vec![0, 0]);
    }

    #[test]
    fn rope_index_batched_mismatched_row_length_rejected() {
        let row_a: Vec<i32> = vec![1, 2, 3, 4];
        let row_b: Vec<i32> = vec![1, 2, 3];
        let err =
            get_rope_index_batched(&[&row_a, &row_b], &[&[], &[]], 2, 248_056, 248_057, 248_053)
                .unwrap_err();
        assert!(format!("{err}").contains("row length mismatch"));
    }

    #[test]
    fn pack_position_ids_shape() {
        let t = vec![0_i32, 1, 2];
        let h = vec![0_i32, 1, 2];
        let w = vec![0_i32, 1, 2];
        let pos = pack_position_ids(&t, &h, &w).unwrap();
        assert_eq!(pos.shape(), &[3, 1, 3]);
    }

    #[test]
    fn merge_input_ids_text_only_passthrough() {
        // No image tokens -> output equals input.
        let inputs_embeds = uniform::<_, f32>(0.0, 1.0, &[1, 4, 8], None).unwrap();
        let input_ids = Array::from_slice(&[10_i32, 11, 12, 13], &[1, 4]);
        let image_features = Array::from_slice::<f32>(&[], &[0, 8]);
        let out = merge_input_ids_with_image_features(
            &image_features,
            &inputs_embeds,
            &input_ids,
            99,
            100,
        )
        .unwrap();
        let diff = out
            .subtract(&inputs_embeds)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(diff < 1e-6);
    }

    fn flatten_f32(arr: &Array) -> Vec<f32> {
        let total: i32 = arr.shape().iter().product();
        let flat = reshape(arr, &[total]).unwrap();
        let evald = flat.add(Array::from_f32(0.0)).unwrap();
        eval([&evald]).unwrap();
        evald.as_slice::<f32>().to_vec()
    }

    #[test]
    fn merge_input_ids_replaces_image_slots() {
        // 4-token sequence with image_token at index 1 and 2; supply two
        // distinct feature rows and verify they land in the right slots.
        let hidden = 4;
        let inputs_embeds = uniform::<_, f32>(0.0, 1.0, &[1, 4, hidden], None).unwrap();
        let input_ids = Array::from_slice(&[10_i32, 99, 99, 13], &[1, 4]);
        let f0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let f1: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let mut data = f0.clone();
        data.extend(&f1);
        let image_features = Array::from_slice(&data, &[2, hidden]);
        let out = merge_input_ids_with_image_features(
            &image_features,
            &inputs_embeds,
            &input_ids,
            99,
            100,
        )
        .unwrap();
        let flat = flatten_f32(&out);
        for k in 0..hidden as usize {
            assert!(
                (flat[hidden as usize + k] - f0[k]).abs() < 1e-5,
                "slot1[{k}]={}",
                flat[hidden as usize + k]
            );
            assert!(
                (flat[2 * hidden as usize + k] - f1[k]).abs() < 1e-5,
                "slot2[{k}]={}",
                flat[2 * hidden as usize + k]
            );
        }
    }
}
