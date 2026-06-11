//! Centroid-masked sparse lm head (E2B/E4B `use_ordered_embeddings`): score
//! centroids → top-k clusters → exact logits for those tokens (via
//! `token_ordering`) → scatter into a −inf vocab vector.

use mlx_rs::builder::Builder;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::nn;
use mlx_rs::ops::{
    argpartition_axis, full, indexing::put_along_axis, indexing::take_axis, indexing::IndexOp,
    matmul, reshape, transpose_axes,
};
use mlx_rs::Array;

use crate::error::Error;

#[derive(Debug, ModuleParameters)]
pub struct MaskedEmbedder {
    /// `centroids.weight` `[num_centroids, draft_hidden]` (bias-free).
    #[param]
    pub centroids: nn::Linear,
    /// `token_ordering` `[vocab]` int — vocab permutation grouping tokens by
    /// cluster. Not a learned param (a registered buffer), so kept as a plain
    /// field hydrated by the loader.
    pub token_ordering: Array,
    num_centroids: i32,
    top_k: i32,
    vocab_size: i32,
    tokens_per_centroid: i32,
}

impl MaskedEmbedder {
    pub fn new(
        draft_hidden: i32,
        num_centroids: i32,
        top_k: i32,
        vocab_size: i32,
    ) -> Result<Self, Error> {
        Ok(Self {
            centroids: nn::LinearBuilder::new(draft_hidden, num_centroids)
                .bias(false)
                .build()?,
            token_ordering: Array::zeros::<i32>(&[vocab_size])?,
            num_centroids,
            top_k,
            vocab_size,
            tokens_per_centroid: vocab_size / num_centroids,
        })
    }

    /// Sparse logits `[1, L, vocab]`. `hidden` is `[1, L, draft_hidden]`;
    /// `lm_head_weight` is the tied embedding `[vocab, draft_hidden]`.
    pub fn forward(&self, hidden: &Array, lm_head_weight: &Array) -> Result<Array, Error> {
        let shape = hidden.shape();
        let (batch, seq_len) = (shape[0], shape[1]);
        let k = self.top_k;
        let vpc = self.tokens_per_centroid;

        // Top-k cluster indices via argpartition (last k slots, unordered ok).
        // Matmul (not Linear::forward) keeps this `&self`.
        let cw = self.centroids.weight.as_ref();
        let centroid_logits = matmul(hidden, &transpose_axes(cw, &[1, 0])?)?;
        let part = argpartition_axis(&centroid_logits, self.num_centroids - k, -1)?;
        let top_k_indices = part.index((.., .., (self.num_centroids - k)..));

        // Canonical token positions per cluster: [num_centroids, vpc].
        let canonical = reshape(&self.token_ordering, &[self.num_centroids, vpc])?;
        // Gather the selected clusters' rows → [1, L, k, vpc].
        let flat_idx = reshape(&top_k_indices, &[batch * seq_len * k])?;
        let selected_canonical = take_axis(&canonical, &flat_idx, 0)?; // [B*L*k, vpc]
        let selected_canonical = reshape(&selected_canonical, &[batch, seq_len, k, vpc])?;

        // Gather lm-head rows at those positions → [1, L, k*vpc, draft_hidden].
        let cand = k * vpc;
        let sel_flat = reshape(&selected_canonical, &[batch * seq_len * cand])?;
        let draft_hidden = shape[2];
        let selected_emb = take_axis(lm_head_weight, &sel_flat, 0)?; // [B*L*cand, H]
        let selected_emb = reshape(&selected_emb, &[batch, seq_len, cand, draft_hidden])?;

        // Dot products: [1, L, 1, H] @ [1, L, H, cand] → [1, L, cand].
        let h_row = reshape(hidden, &[batch, seq_len, 1, draft_hidden])?;
        let emb_t = transpose_axes(&selected_emb, &[0, 1, 3, 2])?;
        let selected_logits = reshape(&matmul(&h_row, &emb_t)?, &[batch, seq_len, cand])?;

        // Scatter into a low-filled [1, L, vocab]; masked tokens stay below any
        // real logit so they never win an argmax.
        let fill = Array::from_f32(-1.0e30);
        let output = full::<f32>(&[batch, seq_len, self.vocab_size], &fill)?
            .as_dtype(selected_logits.dtype())?;
        let scatter_idx = reshape(&selected_canonical, &[batch, seq_len, cand])?;
        Ok(put_along_axis(&output, &scatter_idx, &selected_logits, -1)?)
    }
}
