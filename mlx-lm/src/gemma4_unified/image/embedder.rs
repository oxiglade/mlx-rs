//! Gemma 4 Unified encoder-free vision embedder.
//!
//! Forward (merged patches `[B, N, patch_dim]` + position ids `[B, N, 2]`):
//! `patch_ln1 → patch_dense → patch_ln2 → + factorised-2D-posemb → pos_norm
//! → RMSNorm(no-scale) → embedding_projection`, yielding `[B, N, text_hidden]`.
//! Padding patches (position id `-1`) contribute a zero positional embedding
//! and are stripped by the caller before stitching.

use mlx_rs::builder::Builder;
use mlx_rs::macros::{ModuleParameters, Quantizable};
use mlx_rs::module::{Module, Param};
use mlx_rs::nn::{self, LayerNorm, LayerNormBuilder};
use mlx_rs::ops::indexing::take_axis;
use mlx_rs::ops::{expand_dims_axes, maximum};
use mlx_rs::quantization::MaybeQuantized;
use mlx_rs::{Array, Dtype};

use crate::error::Error;
use crate::gemma4_unified::image::config::VisionConfig;
use crate::nn::RmsNormNoScale;

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct VisionEmbedder {
    #[param]
    pub patch_ln1: LayerNorm,
    #[quantizable]
    #[param]
    pub patch_dense: MaybeQuantized<nn::Linear>,
    #[param]
    pub patch_ln2: LayerNorm,
    /// Factorised 2D positional embedding `[mm_posemb_size, 2, mm_embed_dim]`.
    #[param]
    pub pos_embedding: Param<Array>,
    #[param]
    pub pos_norm: LayerNorm,
    /// `embedding_projection`: RMSNorm(no-scale) → linear into text space.
    pub mm_norm: RmsNormNoScale,
    #[quantizable]
    #[param]
    pub embedding_projection: MaybeQuantized<nn::Linear>,
}

impl VisionEmbedder {
    pub fn new(cfg: &VisionConfig) -> Result<Self, Error> {
        let ln = |dim: i32| -> Result<LayerNorm, Error> {
            Ok(LayerNormBuilder::new(dim).eps(cfg.rms_norm_eps).build()?)
        };
        let linear = |inp: i32, out: i32| -> Result<MaybeQuantized<nn::Linear>, Error> {
            Ok(MaybeQuantized::Original(
                nn::LinearBuilder::new(inp, out).build()?,
            ))
        };
        Ok(Self {
            patch_ln1: ln(cfg.patch_dim())?,
            patch_dense: linear(cfg.patch_dim(), cfg.mm_embed_dim)?,
            patch_ln2: ln(cfg.mm_embed_dim)?,
            pos_embedding: Param::new(Array::zeros::<f32>(&[
                cfg.mm_posemb_size,
                2,
                cfg.mm_embed_dim,
            ])?),
            pos_norm: ln(cfg.mm_embed_dim)?,
            mm_norm: RmsNormNoScale::new(cfg.rms_norm_eps),
            embedding_projection: linear(cfg.mm_embed_dim, cfg.output_proj_dims)?,
        })
    }

    /// `pixel_values` `[B, N, patch_dim]`, `position_ids` `[B, N, 2]` (i32,
    /// `-1` for padding). Returns soft-token embeddings `[B, N, text_hidden]`.
    #[allow(non_snake_case, reason = "B/N/D mirror ML tensor axis names")]
    pub fn forward(&mut self, pixel_values: &Array, position_ids: &Array) -> Result<Array, Error> {
        let x = self.patch_ln1.forward(pixel_values)?;
        let x = self.patch_dense.forward(&x)?;
        let x = self.patch_ln2.forward(&x)?;

        let pos = self.positional_embedding(position_ids, x.dtype())?;
        let x = x.add(&pos)?;

        let x = self.pos_norm.forward(&x)?;
        let x = self.mm_norm.forward(&x)?;
        Ok(self.embedding_projection.forward(&x)?)
    }

    /// `(pos_embedding[clamp(ids,0), axis] * valid).sum(axis=-2)` — gather the
    /// row (axis 0) and column (axis 1) embeddings, zero out padding patches,
    /// and sum the two axes.
    fn positional_embedding(&self, position_ids: &Array, dtype: Dtype) -> Result<Array, Error> {
        let shape = position_ids.shape();
        let (b, n) = (shape[0], shape[1]);

        // Split the `[B, N, 2]` ids into row/col and clamp -1 → 0 for the gather.
        let ids = position_ids.reshape(&[b * n, 2])?;
        let rows = take_axis(&ids, Array::from_slice(&[0_i32], &[1]), 1)?.squeeze_axes(&[1])?;
        let cols = take_axis(&ids, Array::from_slice(&[1_i32], &[1]), 1)?.squeeze_axes(&[1])?;
        let zero = Array::from_int(0);
        let rows_c = maximum(&rows, &zero)?;
        let cols_c = maximum(&cols, &zero)?;

        // Row table = pos_embedding[:, 0, :], col table = pos_embedding[:, 1, :].
        let table = self.pos_embedding.as_ref();
        let row_table =
            take_axis(table, Array::from_slice(&[0_i32], &[1]), 1)?.squeeze_axes(&[1])?;
        let col_table =
            take_axis(table, Array::from_slice(&[1_i32], &[1]), 1)?.squeeze_axes(&[1])?;
        let row_emb = take_axis(&row_table, &rows_c, 0)?;
        let col_emb = take_axis(&col_table, &cols_c, 0)?;
        let emb = row_emb.add(&col_emb)?;

        // Zero padding patches (either axis == -1).
        let valid = rows.ge(&zero)?.logical_and(&cols.ge(&zero)?)?;
        let valid = expand_dims_axes(&valid.as_dtype(dtype)?, &[-1])?;
        let emb = emb.multiply(&valid)?;

        let d = table.shape()[2];
        Ok(emb.reshape(&[b, n, d])?.as_dtype(dtype)?)
    }
}
