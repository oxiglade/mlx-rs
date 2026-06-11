//! Qwen3-VL vision tower used unmodified by Qwen3.5.
//!
//! Components:
//!
//! - [`PatchEmbed`]: a Conv3d with `kernel = stride = [tps, ps, ps]`.
//! - [`PatchMerger`]: LayerNorm + Linear + GELU + Linear that maps
//!   `hidden * merge²` features to `out_hidden`.
//! - [`VisionAttention`] + [`VisionMlp`] + [`VisionBlock`]: the standard ViT
//!   block, with rotary embeddings on the 2-D image grid and a per-image
//!   SDPA driven by `cu_seqlens`.
//! - [`VisionModel`]: top level — patch embed, fast bilinear-interpolated
//!   learned positional embeddings, rotary tables computed from the
//!   `image_grid_thw`, ViT blocks, final merger.
//!
//! No deepstack injection — `deepstack_visual_indexes` is empty for Qwen3.5.

use mlx_rs::{
    builder::Builder,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::{
        arange, concatenate_axis, cos as cos_op, expand_dims, indexing::take_axis, linspace,
        minimum, outer, reshape, sin as sin_op, split, split_sections, transpose_axes,
    },
    quantization::MaybeQuantized,
    utils::SingleOrTriple,
    Array, Dtype,
};

use crate::error::Error;
use crate::qwen3_5::text::config::VisionConfig;

/// Rotary base for the vision tower's 2-D RoPE. Deliberately 10k — an
/// order of magnitude below the text decoder's `rope_theta`, matching the
/// upstream Qwen3-VL spec (the vision grid spans far fewer positions).
const VISION_ROPE_THETA: f32 = 10_000.0;

/// LayerNorm epsilon for every vision-tower norm (blocks + merger).
const VISION_LAYER_NORM_EPS: f32 = 1e-6;

/// Build the rotary frequency table `[seqlen, dim/2]` shared across the
/// height and width axes of the vision rotary embedding.
fn build_rotary_freqs(dim: i32, seqlen: i32, theta: f32) -> Result<Array, Error> {
    let half = dim / 2;
    let idx = arange::<_, f32>(0.0, dim as f32, 2.0)?;
    let exp = idx.multiply(Array::from_f32(1.0 / dim as f32))?;
    let inv_freq = Array::from_f32(theta).power(&exp)?.reciprocal()?;
    let seq = arange::<_, f32>(0.0, seqlen as f32, None)?;
    let outer_prod = outer(&seq, &inv_freq)?;
    // outer returns shape (seqlen, half); ensure it.
    debug_assert_eq!(outer_prod.shape(), &[seqlen, half]);
    Ok(outer_prod)
}

/// Apply the 2-D vision rotary embedding to `tensor`.
///
/// `tensor` has shape `[B, seq_len, num_heads, head_dim]`. `freqs` carries the
/// per-token angle for the full `head_dim` (already concatenated from the H and
/// W axis frequencies by [`rot_pos_emb`]). We take `cos = cos(freqs)`,
/// `sin = sin(freqs)`, broadcast across `B`/`H`, and apply
/// `x * cos + rotate_half(x) * sin`.
pub fn apply_vision_rope(tensor: &Array, freqs: &Array) -> Result<Array, Error> {
    let orig_dtype = tensor.dtype();
    let f32_freqs = freqs.as_dtype(Dtype::Float32)?;
    let cos = cos_op(&f32_freqs)?;
    let sin = sin_op(&f32_freqs)?;

    // cos/sin start as [S, D]. Insert axes to land at [1, S, 1, D] so we
    // broadcast against `tensor` of shape [B, S, num_heads, D].
    let cos_b = expand_dims(&expand_dims(&cos, 1)?, 0)?;
    let sin_b = expand_dims(&expand_dims(&sin, 1)?, 0)?;

    let x = tensor.as_dtype(Dtype::Float32)?;
    let lhs = x.multiply(&cos_b)?;
    let rh = rotate_half(&x)?;
    let rhs = rh.multiply(&sin_b)?;
    Ok(lhs.add(&rhs)?.as_dtype(orig_dtype)?)
}

fn rotate_half(x: &Array) -> Result<Array, Error> {
    let halves = split(x, 2, -1)?;
    let neg_x2 = halves[1].negative()?;
    Ok(concatenate_axis(&[&neg_x2, &halves[0]], -1)?)
}

/// Vision patch-embedding Conv3d. Maps `[B*P, C*tps*ps*ps]` flat patches into
/// `[B*P, hidden_size]` via a 3-D conv whose kernel + stride equal
/// `[tps, ps, ps]`.
#[derive(Debug, Clone, ModuleParameters)]
pub struct PatchEmbed {
    pub patch_size: i32,
    pub temporal_patch_size: i32,
    pub in_channels: i32,
    pub hidden_size: i32,

    #[param]
    pub proj: nn::Conv3d,
}

impl PatchEmbed {
    /// Build a freshly-initialised patch embedding.
    pub fn new(cfg: &VisionConfig) -> Result<Self, Error> {
        let proj = nn::Conv3dBuilder::new(
            cfg.in_channels,
            cfg.hidden_size,
            SingleOrTriple::Triple(cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size),
        )
        .bias(true)
        .stride(SingleOrTriple::Triple(
            cfg.temporal_patch_size,
            cfg.patch_size,
            cfg.patch_size,
        ))
        .build()?;
        Ok(Self {
            patch_size: cfg.patch_size,
            temporal_patch_size: cfg.temporal_patch_size,
            in_channels: cfg.in_channels,
            hidden_size: cfg.hidden_size,
            proj,
        })
    }

    /// Forward over a flat patch tensor of shape `[total_patches, C * tps *
    /// ps * ps]` (the image processor's `pixel_values` output).
    ///
    /// Internally reshapes to `[total_patches, C, tps, ps, ps]`, moves the
    /// channel axis last (NTHWC), runs Conv3d, and reshapes back to
    /// `[total_patches, hidden_size]`.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let shape = x.shape();
        let total = shape[0];
        let r = reshape(
            x,
            &[
                total,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ],
        )?;
        // moveaxis(1, 4): NCTHW -> NTHWC for the Conv3d NTHWC convention.
        let r = transpose_axes(&r, &[0, 2, 3, 4, 1])?;
        let y = self.proj.forward(&r)?;
        // Conv3d on a [1,1,1] grid with stride = kernel collapses TxHxW to 1x1x1.
        Ok(reshape(&y, &[total, self.hidden_size])?)
    }
}

/// Final patch merger: `LayerNorm + Linear + GELU + Linear`.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct PatchMerger {
    pub hidden_size: i32,
    pub use_postshuffle_norm: bool,

    #[param]
    pub norm: nn::LayerNorm,

    #[quantizable]
    #[param]
    pub linear_fc1: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    pub linear_fc2: MaybeQuantized<nn::Linear>,
}

impl PatchMerger {
    /// Build the merger. When `use_postshuffle_norm` is `false` (the Qwen3-VL
    /// default for the final merger) the LayerNorm runs at `hidden_size`
    /// width and the merger then reshapes to the `merge²` block; otherwise
    /// the LayerNorm runs after the reshape.
    pub fn new(cfg: &VisionConfig, use_postshuffle_norm: bool) -> Result<Self, Error> {
        let merge2 = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let hidden_size = cfg.hidden_size * merge2;
        let norm_width = if use_postshuffle_norm {
            hidden_size
        } else {
            cfg.hidden_size
        };
        let norm = nn::LayerNormBuilder::new(norm_width)
            .eps(VISION_LAYER_NORM_EPS)
            .build()?;
        let linear_fc1 = nn::LinearBuilder::new(hidden_size, hidden_size).build()?;
        let linear_fc2 = nn::LinearBuilder::new(hidden_size, cfg.out_hidden_size).build()?;
        Ok(Self {
            hidden_size,
            use_postshuffle_norm,
            norm,
            linear_fc1: MaybeQuantized::Original(linear_fc1),
            linear_fc2: MaybeQuantized::Original(linear_fc2),
        })
    }

    /// Run the merger on input shape `[total_tokens, hidden_size_in]` where
    /// `hidden_size_in == cfg.hidden_size` (pre-shuffle case) — the function
    /// reshapes to `[total_tokens / merge², hidden_size * merge²]` then
    /// applies the LN+MLP.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let y = if self.use_postshuffle_norm {
            self.norm.forward(&reshape(x, &[-1, self.hidden_size])?)?
        } else {
            let n = self.norm.forward(x)?;
            reshape(&n, &[-1, self.hidden_size])?
        };
        let h = self.linear_fc1.forward(&y)?;
        let h = nn::gelu(&h)?;
        Ok(self.linear_fc2.forward(&h)?)
    }
}

/// Vision-block MLP: `Linear(hidden, intermediate) + GELU(tanh) + Linear`.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct VisionMlp {
    #[quantizable]
    #[param]
    pub linear_fc1: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    pub linear_fc2: MaybeQuantized<nn::Linear>,
}

impl VisionMlp {
    pub fn new(cfg: &VisionConfig) -> Result<Self, Error> {
        let linear_fc1 = nn::LinearBuilder::new(cfg.hidden_size, cfg.intermediate_size).build()?;
        let linear_fc2 = nn::LinearBuilder::new(cfg.intermediate_size, cfg.hidden_size).build()?;
        Ok(Self {
            linear_fc1: MaybeQuantized::Original(linear_fc1),
            linear_fc2: MaybeQuantized::Original(linear_fc2),
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let h = self.linear_fc1.forward(x)?;
        let h = nn::gelu_approximate(&h)?;
        Ok(self.linear_fc2.forward(&h)?)
    }
}

/// Per-block multi-head attention with cu_seqlens-aware SDPA.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct VisionAttention {
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[quantizable]
    #[param]
    pub qkv: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub proj: MaybeQuantized<nn::Linear>,
}

impl VisionAttention {
    pub fn new(cfg: &VisionConfig) -> Result<Self, Error> {
        let dim = cfg.hidden_size;
        let num_heads = cfg.num_heads;
        let head_dim = dim / num_heads;
        let scale = (head_dim as f32).sqrt().recip();
        let qkv = nn::LinearBuilder::new(dim, dim * 3).build()?;
        let proj = nn::LinearBuilder::new(dim, dim).build()?;
        Ok(Self {
            num_heads,
            head_dim,
            scale,
            qkv: MaybeQuantized::Original(qkv),
            proj: MaybeQuantized::Original(proj),
        })
    }

    /// Forward over `x: [total_tokens, hidden]` (no batch dim — every image
    /// in the batch is concatenated and split via `cu_seqlens`).
    pub fn forward(
        &mut self,
        x: &Array,
        cu_seqlens: &[i32],
        rotary_freqs: &Array,
    ) -> Result<Array, Error> {
        let seq_len = x.shape()[0];
        let qkv = self.qkv.forward(x)?;
        let qkv = reshape(&qkv, &[seq_len, 3, self.num_heads, self.head_dim])?;
        // transpose 0/1: [3, seq_len, num_heads, head_dim]
        let qkv = transpose_axes(&qkv, &[1, 0, 2, 3])?;
        let parts = split(&qkv, 3, 0)?;
        let q = parts[0].squeeze_axes(&[0])?;
        let k = parts[1].squeeze_axes(&[0])?;
        let v = parts[2].squeeze_axes(&[0])?;

        // Apply rotary to q/k with a fake batch axis to reuse the same fn.
        let q4 = expand_dims(&q, 0)?;
        let k4 = expand_dims(&k, 0)?;
        let q4 = apply_vision_rope(&q4, rotary_freqs)?;
        let k4 = apply_vision_rope(&k4, rotary_freqs)?;
        let q = q4.squeeze_axes(&[0])?;
        let k = k4.squeeze_axes(&[0])?;

        // Reshape to [1, H, S, D] for SDPA; split by cu_seqlens, run per-image.
        let q = expand_dims(&transpose_axes(&q, &[1, 0, 2])?, 0)?;
        let k = expand_dims(&transpose_axes(&k, &[1, 0, 2])?, 0)?;
        let v = expand_dims(&transpose_axes(&v, &[1, 0, 2])?, 0)?;

        let split_indices: Vec<i32> = cu_seqlens
            .iter()
            .copied()
            .skip(1)
            .take(cu_seqlens.len().saturating_sub(2))
            .collect();

        let q_chunks = if split_indices.is_empty() {
            vec![q]
        } else {
            split_sections(&q, &split_indices, 2)?
        };
        let k_chunks = if split_indices.is_empty() {
            vec![k]
        } else {
            split_sections(&k, &split_indices, 2)?
        };
        let v_chunks = if split_indices.is_empty() {
            vec![v]
        } else {
            split_sections(&v, &split_indices, 2)?
        };

        let mut outs = Vec::with_capacity(q_chunks.len());
        for ((qc, kc), vc) in q_chunks.iter().zip(k_chunks.iter()).zip(v_chunks.iter()) {
            // At non-fused head dims (the ViT runs head_dim 72) MLX SDPA
            // falls back to the ops-composed path.
            let out = scaled_dot_product_attention(
                qc,
                kc,
                vc,
                self.scale,
                Option::<ScaledDotProductAttentionMask<'_>>::None,
                None,
            )?;
            outs.push(out);
        }
        let cat = concatenate_axis(&outs, 2)?;
        // [1, H, S, D] -> [S, H*D]
        let cat = transpose_axes(&cat, &[0, 2, 1, 3])?;
        let out = reshape(&cat, &[seq_len, -1])?;
        Ok(self.proj.forward(&out)?)
    }
}

/// One transformer block — pre-LN, attention, residual, pre-LN, MLP, residual.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct VisionBlock {
    #[param]
    pub norm1: nn::LayerNorm,
    #[param]
    pub norm2: nn::LayerNorm,
    #[quantizable]
    #[param]
    pub attn: VisionAttention,
    #[quantizable]
    #[param]
    pub mlp: VisionMlp,
}

impl VisionBlock {
    pub fn new(cfg: &VisionConfig) -> Result<Self, Error> {
        let norm1 = nn::LayerNormBuilder::new(cfg.hidden_size)
            .eps(VISION_LAYER_NORM_EPS)
            .build()?;
        let norm2 = nn::LayerNormBuilder::new(cfg.hidden_size)
            .eps(VISION_LAYER_NORM_EPS)
            .build()?;
        let attn = VisionAttention::new(cfg)?;
        let mlp = VisionMlp::new(cfg)?;
        Ok(Self {
            norm1,
            norm2,
            attn,
            mlp,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        cu_seqlens: &[i32],
        rotary_freqs: &Array,
    ) -> Result<Array, Error> {
        let r = self
            .attn
            .forward(&self.norm1.forward(x)?, cu_seqlens, rotary_freqs)?;
        let h = x.add(&r)?;
        let r = self.mlp.forward(&self.norm2.forward(&h)?)?;
        Ok(h.add(&r)?)
    }
}

/// Compute `cu_seqlens` (exclusive prefix sum) given a `grid_thw` matrix of
/// shape `[num_images, 3]` (t, h, w). Each image contributes `t` chunks of
/// length `h*w` to the sequence. `cu_seqlens` is the cumulative sum prefixed
/// with `0`, so adjacent differences give per-image attention block lengths.
pub fn build_cu_seqlens(grid_thw: &[[i32; 3]]) -> Vec<i32> {
    let mut out = Vec::with_capacity(grid_thw.len() + 1);
    let mut acc = 0_i32;
    out.push(0);
    for &[t, h, w] in grid_thw {
        let chunk = h * w;
        for _ in 0..t {
            acc += chunk;
            out.push(acc);
        }
    }
    out
}

/// Build the per-token `(row_idx, col_idx)` matrix and feed it through the
/// shared frequency table to materialise the rotary `[seq_len, head_dim]`
/// tensor. Matches `VisionModel.rot_pos_emb`.
fn rot_pos_emb(cfg: &VisionConfig, grid_thw: &[[i32; 3]], head_dim: i32) -> Result<Array, Error> {
    let merge_size = cfg.spatial_merge_size;
    let max_hw = grid_thw.iter().map(|g| g[1].max(g[2])).max().unwrap_or(1);
    let freq_table = build_rotary_freqs(head_dim / 2, max_hw, VISION_ROPE_THETA)?;

    // Precompute total i32 count to size `flat` exactly once and avoid
    // the intermediate Vec<Vec<[i32; 2]>> + per-t coords.clone().
    let total_pairs: usize = grid_thw
        .iter()
        .map(|&[t, h, w]| (t as usize) * (h as usize) * (w as usize))
        .sum();
    let mut flat: Vec<i32> = Vec::with_capacity(total_pairs * 2);
    for &[t, h, w] in grid_thw {
        let merged_h = h / merge_size;
        let merged_w = w / merge_size;
        // Build the coordinate sequence once per image (h*w*2 i32s),
        // then duplicate the chunk (t-1) more times via memcpy.
        let chunk_start = flat.len();
        for bh in 0..merged_h {
            for bw in 0..merged_w {
                for ih in 0..merge_size {
                    for iw in 0..merge_size {
                        flat.push(bh * merge_size + ih);
                        flat.push(bw * merge_size + iw);
                    }
                }
            }
        }
        let chunk_end = flat.len();
        for _ in 1..t {
            flat.extend_from_within(chunk_start..chunk_end);
        }
    }
    let total = (flat.len() / 2) as i32;
    let pos_array = Array::from_slice(&flat, &[total, 2]);

    // pos_ids[:, 0] -> row indices into freq_table
    let rows = take_axis(&pos_array, Array::from_slice(&[0_i32], &[1]), 1)?.squeeze_axes(&[1])?;
    let cols = take_axis(&pos_array, Array::from_slice(&[1_i32], &[1]), 1)?.squeeze_axes(&[1])?;

    let h_emb = take_axis(&freq_table, &rows, 0)?;
    let w_emb = take_axis(&freq_table, &cols, 0)?;
    // Bake the cos/sin head_dim tiling (×2) into the rotary table
    // directly, so the returned tensor's head_dim already matches the
    // tensor we're applying it to.
    let halves = concatenate_axis(&[h_emb, w_emb], -1)?;
    Ok(concatenate_axis(&[&halves, &halves], -1)?)
}

/// Linear interpolation of the learned positional embedding table across an
/// arbitrary image grid. Matches `VisionModel.fast_pos_embed_interpolate`.
fn fast_pos_embed_interpolate(
    cfg: &VisionConfig,
    pos_embed: &mut MaybeQuantized<nn::Embedding>,
    num_grid_per_side: i32,
    grid_thw: &[[i32; 3]],
) -> Result<Array, Error> {
    let mut outs: Vec<Array> = Vec::with_capacity(grid_thw.len());
    let merge = cfg.spatial_merge_size;
    // Loop-invariant scalar constants: built once, reused per image.
    let one_i = Array::from_int(1);
    let max_minus_one = Array::from_int(num_grid_per_side - 1);
    let grid_side = Array::from_int(num_grid_per_side);
    let one_f = Array::from_f32(1.0);
    for &[t, h, w] in grid_thw {
        let h_idxs = linspace::<f32, f32>(0.0, (num_grid_per_side - 1) as f32, h)?;
        let w_idxs = linspace::<f32, f32>(0.0, (num_grid_per_side - 1) as f32, w)?;

        let h_floor = h_idxs.as_dtype(Dtype::Int32)?;
        let w_floor = w_idxs.as_dtype(Dtype::Int32)?;
        let h_ceil = minimum(&h_floor.add(&one_i)?, &max_minus_one)?;
        let w_ceil = minimum(&w_floor.add(&one_i)?, &max_minus_one)?;
        let dh = h_idxs.subtract(&h_floor.as_dtype(Dtype::Float32)?)?;
        let dw = w_idxs.subtract(&w_floor.as_dtype(Dtype::Float32)?)?;

        let base_h = h_floor.multiply(&grid_side)?;
        let base_h_ceil = h_ceil.multiply(&grid_side)?;

        let base_h_b = expand_dims(&base_h, 1)?;
        let base_h_ceil_b = expand_dims(&base_h_ceil, 1)?;
        let w_floor_b = expand_dims(&w_floor, 0)?;
        let w_ceil_b = expand_dims(&w_ceil, 0)?;

        let idx00 = reshape(&base_h_b.add(&w_floor_b)?, &[-1])?;
        let idx01 = reshape(&base_h_b.add(&w_ceil_b)?, &[-1])?;
        let idx10 = reshape(&base_h_ceil_b.add(&w_floor_b)?, &[-1])?;
        let idx11 = reshape(&base_h_ceil_b.add(&w_ceil_b)?, &[-1])?;

        let dh_b = expand_dims(&dh, 1)?;
        let dw_b = expand_dims(&dw, 0)?;
        let w00 = reshape(
            &one_f.subtract(&dh_b)?.multiply(&one_f.subtract(&dw_b)?)?,
            &[-1],
        )?;
        let w01 = reshape(&one_f.subtract(&dh_b)?.multiply(&dw_b)?, &[-1])?;
        let w10 = reshape(&dh_b.multiply(&one_f.subtract(&dw_b)?)?, &[-1])?;
        let w11 = reshape(&dh_b.multiply(&dw_b)?, &[-1])?;

        let e00 = pos_embed.forward(&idx00)?;
        let e01 = pos_embed.forward(&idx01)?;
        let e10 = pos_embed.forward(&idx10)?;
        let e11 = pos_embed.forward(&idx11)?;

        let dtype = e00.dtype();
        let w00 = expand_dims(&w00.as_dtype(dtype)?, 1)?;
        let w01 = expand_dims(&w01.as_dtype(dtype)?, 1)?;
        let w10 = expand_dims(&w10.as_dtype(dtype)?, 1)?;
        let w11 = expand_dims(&w11.as_dtype(dtype)?, 1)?;

        let mut pe = e00.multiply(&w00)?;
        pe = pe.add(&e01.multiply(&w01)?)?;
        pe = pe.add(&e10.multiply(&w10)?)?;
        pe = pe.add(&e11.multiply(&w11)?)?;

        // pe shape [h*w, hidden]. Tile across temporal axis, then reshape and
        // permute to merge-block order.
        let pe_t = if t > 1 {
            let tiled: Vec<&Array> = (0..t).map(|_| &pe).collect();
            concatenate_axis(&tiled, 0)?
        } else {
            pe
        };
        let feature_dim = pe_t.shape()[1];
        let pe_t = reshape(&pe_t, &[t, h, w, feature_dim])?;
        let pe_t = reshape(&pe_t, &[t, h / merge, merge, w / merge, merge, feature_dim])?;
        let pe_t = transpose_axes(&pe_t, &[0, 1, 3, 2, 4, 5])?;
        let pe_t = reshape(&pe_t, &[-1, feature_dim])?;
        outs.push(pe_t);
    }
    if outs.len() == 1 {
        Ok(outs.into_iter().next().expect("len == 1 guard above"))
    } else {
        Ok(concatenate_axis(&outs, 0)?)
    }
}

/// The top-level Qwen3-VL vision tower.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct VisionModel {
    pub spatial_merge_size: i32,
    pub head_dim: i32,
    pub num_grid_per_side: i32,

    #[param]
    pub patch_embed: PatchEmbed,

    #[quantizable]
    #[param]
    pub pos_embed: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    pub blocks: Vec<VisionBlock>,

    #[quantizable]
    #[param]
    pub merger: PatchMerger,

    // Deepstack mergers are unused for qwen3.5 (`deepstack_visual_indexes` is
    // empty) but kept as a `Vec` so future configs flow through unchanged.
    #[quantizable]
    #[param]
    pub deepstack_merger_list: Vec<PatchMerger>,

    pub config: VisionConfig,
}

impl VisionModel {
    /// Build a freshly-initialised vision tower.
    pub fn new(cfg: &VisionConfig) -> Result<Self, Error> {
        // `cfg.model_type` is the typed [`VisionModelType`] enum, so any
        // unknown discriminant was already rejected at config deserialise.
        // No runtime branch needed — every variant uses the same tower
        // architecture and the shared `cfg.*` knobs drive the shape.
        let head_dim = cfg.hidden_size / cfg.num_heads;
        let num_grid_per_side = (cfg.num_position_embeddings as f32).sqrt() as i32;
        let patch_embed = PatchEmbed::new(cfg)?;
        let pos_embed = nn::Embedding::new(cfg.num_position_embeddings, cfg.hidden_size)?;
        let blocks = (0..cfg.depth)
            .map(|_| VisionBlock::new(cfg))
            .collect::<Result<Vec<_>, _>>()?;
        let merger = PatchMerger::new(cfg, false)?;
        let deepstack_merger_list = cfg
            .deepstack_visual_indexes
            .iter()
            .map(|_| PatchMerger::new(cfg, true))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            spatial_merge_size: cfg.spatial_merge_size,
            head_dim,
            num_grid_per_side,
            patch_embed,
            pos_embed: MaybeQuantized::Original(pos_embed),
            blocks,
            merger,
            deepstack_merger_list,
            config: cfg.clone(),
        })
    }

    /// Run the full vision tower.
    ///
    /// - `pixel_values`: shape `[total_patches, C * tps * ps * ps]` from the
    ///   image processor.
    /// - `grid_thw`: `[num_images, 3]` listing `[grid_t, grid_h, grid_w]`
    ///   for each image. `grid_t == 1` for still images; `> 1` for video.
    ///
    /// Returns the projected hidden states shaped `[total_merged_tokens,
    /// out_hidden_size]`.
    pub fn forward(&mut self, pixel_values: &Array, grid_thw: &[[i32; 3]]) -> Result<Array, Error> {
        let mut h = self.patch_embed.forward(pixel_values)?;
        let pos = fast_pos_embed_interpolate(
            &self.config,
            &mut self.pos_embed,
            self.num_grid_per_side,
            grid_thw,
        )?;
        h = h.add(&pos)?;
        let rotary = rot_pos_emb(&self.config, grid_thw, self.head_dim)?;
        let cu_seqlens = build_cu_seqlens(grid_thw);

        for blk in &mut self.blocks {
            h = blk.forward(&h, &cu_seqlens, &rotary)?;
        }
        self.merger.forward(&h)
    }

    /// Toggle training mode on every quantisable parameter.
    pub fn training_mode(&mut self, mode: bool) {
        self.pos_embed.training_mode(mode);
        for blk in &mut self.blocks {
            blk.attn.qkv.training_mode(mode);
            blk.attn.proj.training_mode(mode);
            blk.mlp.linear_fc1.training_mode(mode);
            blk.mlp.linear_fc2.training_mode(mode);
        }
        self.merger.linear_fc1.training_mode(mode);
        self.merger.linear_fc2.training_mode(mode);
        for m in &mut self.deepstack_merger_list {
            m.linear_fc1.training_mode(mode);
            m.linear_fc2.training_mode(mode);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;
    use mlx_rs::random::uniform;

    fn synthetic_vision_config() -> VisionConfig {
        let json = serde_json::json!({
            "model_type": "qwen3_5",
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "out_hidden_size": 32,
            "num_heads": 2,
            "patch_size": 4,
            "in_channels": 3,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn cu_seqlens_single_image() {
        let cu = build_cu_seqlens(&[[1, 4, 4]]);
        assert_eq!(cu, vec![0, 16]);
    }

    #[test]
    fn cu_seqlens_temporal_chunks() {
        let cu = build_cu_seqlens(&[[2, 4, 4]]);
        assert_eq!(cu, vec![0, 16, 32]);
    }

    #[test]
    fn cu_seqlens_multi_image() {
        let cu = build_cu_seqlens(&[[1, 4, 4], [1, 2, 2]]);
        assert_eq!(cu, vec![0, 16, 20]);
    }

    #[test]
    fn patch_embed_shape_round_trip() {
        let cfg = synthetic_vision_config();
        let mut pe = PatchEmbed::new(&cfg).unwrap();
        let total_patches = 8;
        let feat =
            (cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size) as usize;
        let x = uniform::<_, f32>(0.0, 1.0, &[total_patches, feat as i32], None).unwrap();
        let y = pe.forward(&x).unwrap();
        assert_eq!(y.shape(), &[total_patches, cfg.hidden_size]);
    }

    #[test]
    fn vision_model_runs_on_synthetic_input() {
        let cfg = synthetic_vision_config();
        let mut vm = VisionModel::new(&cfg).unwrap();
        let grid_thw = [[1_i32, 4, 4]];
        let total_patches = 16;
        let feat =
            (cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size) as usize;
        let pixel_values =
            uniform::<_, f32>(0.0, 1.0, &[total_patches, feat as i32], None).unwrap();
        let out = vm.forward(&pixel_values, &grid_thw).unwrap();
        // Merger collapses every merge² block to one out_hidden vector.
        assert_eq!(
            out.shape(),
            &[
                total_patches / (cfg.spatial_merge_size * cfg.spatial_merge_size),
                cfg.out_hidden_size,
            ]
        );
    }
}
