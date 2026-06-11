//! Gemma 4 SigLIP vision tower (`gemma4_vision`): Linear patch embed +
//! per-axis position table → bidirectional encoder (2-D RoPE, GeGLU) →
//! standardize → avg-pool to `default_output_length` soft tokens. Tower
//! weights are bf16; `EmbedVision` (projector) is quantized. B=1 single-image
//! path with no padding (grid fills `pooling_kernel_size`-divisible dims).

use mlx_rs::{
    builder::Builder,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    macros::{ModuleParameters, Quantizable},
    module::{Module, Param},
    nn,
    ops::{
        arange, concatenate_axis, cos as cos_op, expand_dims, indexing::take_axis,
        indexing::IndexOp, power, reshape, sin as sin_op, split, transpose_axes,
    },
    quantization::MaybeQuantized,
    Array, Dtype,
};

use crate::activations::{geglu, GegluCache};
use crate::error::Error;
use crate::gemma4::image::config::VisionConfig;
use crate::nn::RmsNormNoScale;

/// Spatial axes the 2-D RoPE splits `head_dim` across (height, width).
const ROPE_NDIM: i32 = 2;

/// Build a fresh `nn::RmsNorm` at `dim` width with the tower epsilon.
fn rms_norm(dim: i32, eps: f32) -> Result<nn::RmsNorm, Error> {
    Ok(nn::RmsNormBuilder::new(dim).eps(eps).build()?)
}

fn linear(inp: i32, out: i32) -> Result<MaybeQuantized<nn::Linear>, Error> {
    Ok(MaybeQuantized::Original(
        nn::LinearBuilder::new(inp, out).bias(false).build()?,
    ))
}

/// `[-x2, x1]` over the last axis.
fn rotate_half(x: &Array) -> Result<Array, Error> {
    let halves = split(x, 2, -1)?;
    let neg = halves[1].negative()?;
    Ok(concatenate_axis(&[&neg, &halves[0]], -1)?)
}

/// 2-D multidimensional RoPE: split `head_dim` into [`ROPE_NDIM`] equal parts,
/// rotate each by its own spatial position. `inputs` is `[B, L, H, head_dim]`;
/// `positions` is `[L, ROPE_NDIM]` (per-patch (x, y)). Computed in f32.
fn apply_vision_rope(inputs: &Array, positions: &Array, theta: f32) -> Result<Array, Error> {
    let orig = inputs.dtype();
    let head_dim = *inputs.shape().last().expect("inputs has a last axis");
    let channels_per_dim = 2 * (head_dim / (2 * ROPE_NDIM));
    let half_per_dim = channels_per_dim / 2;

    let exps = arange::<_, f32>(0.0, half_per_dim as f32, None)?
        .multiply(Array::from_f32(2.0 / channels_per_dim as f32))?;
    let timescale = power(Array::from_f32(theta), &exps)?;

    let x = inputs.as_dtype(Dtype::Float32)?;
    let mut parts: Vec<Array> = Vec::with_capacity(ROPE_NDIM as usize);
    for d in 0..ROPE_NDIM {
        let lo = d * channels_per_dim;
        let x_part = x.index((.., .., .., lo..lo + channels_per_dim));
        // positions[:, d] -> [L, 1], divide by per-dim timescale -> [L, half].
        let pos_d = positions.index((.., d..d + 1)).as_dtype(Dtype::Float32)?;
        let angle = pos_d.divide(&timescale)?;
        let cos = cos_op(&angle)?;
        let sin = sin_op(&angle)?;
        // tile to channels_per_dim, broadcast to [1, L, 1, channels_per_dim].
        let cos = concatenate_axis(&[&cos, &cos], -1)?;
        let sin = concatenate_axis(&[&sin, &sin], -1)?;
        let cos = expand_dims(&expand_dims(&cos, 0)?, 2)?;
        let sin = expand_dims(&expand_dims(&sin, 0)?, 2)?;
        let rot = x_part
            .multiply(&cos)?
            .add(&rotate_half(&x_part)?.multiply(&sin)?)?;
        parts.push(rot);
    }
    Ok(concatenate_axis(&parts, -1)?.as_dtype(orig)?)
}

/// Patchify pixels and add the per-axis learned position embedding.
#[derive(Debug, Clone, ModuleParameters)]
pub struct PatchEmbedder {
    patch_size: i32,

    #[param]
    pub input_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub position_embedding_table: Param<Array>,
}

impl PatchEmbedder {
    pub fn new(cfg: &VisionConfig) -> Result<Self, Error> {
        let patch_feats = 3 * cfg.patch_size * cfg.patch_size;
        Ok(Self {
            patch_size: cfg.patch_size,
            input_proj: linear(patch_feats, cfg.hidden_size)?,
            position_embedding_table: Param::new(Array::ones::<f32>(&[
                2,
                cfg.position_embedding_size,
                cfg.hidden_size,
            ])?),
        })
    }

    /// `pixel_values [B, C, H, W]` (channel-first, in [0, 1]) → patch tokens
    /// `[B, pH*pW, C*p*p]`, normalized to [-1, 1], then projected.
    fn patchify(&mut self, pixel_values: &Array) -> Result<Array, Error> {
        let shape = pixel_values.shape();
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let p = self.patch_size;
        let (ph, pw) = (h / p, w / p);
        let r = reshape(pixel_values, &[b, c, ph, p, pw, p])?;
        let r = transpose_axes(&r, &[0, 2, 4, 3, 5, 1])?;
        let r = reshape(&r, &[b, ph * pw, c * p * p])?;
        let dt = self.position_embedding_table.dtype();
        let r = r
            .subtract(Array::from_f32(0.5))?
            .multiply(Array::from_f32(2.0))?
            .as_dtype(dt)?;
        Ok(self.input_proj.forward(&r)?)
    }

    /// Sum the x-axis and y-axis position embeddings for the patch grid.
    /// `x_idx`/`y_idx` are `[pH*pW]` int32 column/row indices.
    fn position_embeddings(&self, x_idx: &Array, y_idx: &Array) -> Result<Array, Error> {
        let table_x = self.position_embedding_table.index(0);
        let table_y = self.position_embedding_table.index(1);
        let ex = take_axis(&table_x, x_idx, 0)?;
        let ey = take_axis(&table_y, y_idx, 0)?;
        Ok(ex.add(&ey)?)
    }

    pub fn forward(&mut self, pixel_values: &Array, grid: PatchGrid) -> Result<Array, Error> {
        let hidden = self.patchify(pixel_values)?;
        let pos = self
            .position_embeddings(&grid.x_idx, &grid.y_idx)?
            .as_dtype(hidden.dtype())?;
        Ok(hidden.add(&expand_dims(&pos, 0)?)?)
    }
}

/// Per-patch (x, y) coordinate indices for the grid; `[pH*pW]` each.
#[derive(Debug, Clone)]
pub struct PatchGrid {
    pub ph: i32,
    pub pw: i32,
    pub x_idx: Array,
    pub y_idx: Array,
    /// `[pH*pW, ROPE_NDIM]` (x, y) positions for the 2-D RoPE.
    pub positions: Array,
}

impl PatchGrid {
    /// Row-major grid: patch index `i = row*pW + col` → `(col, row)`.
    pub fn new(ph: i32, pw: i32) -> Self {
        let n = (ph * pw) as usize;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut pos = Vec::with_capacity(n * 2);
        for row in 0..ph {
            for col in 0..pw {
                xs.push(col);
                ys.push(row);
                pos.push(col);
                pos.push(row);
            }
        }
        let x_idx = Array::from_slice(&xs, &[ph * pw]);
        let y_idx = Array::from_slice(&ys, &[ph * pw]);
        let positions = Array::from_slice(&pos, &[ph * pw, ROPE_NDIM]);
        Self {
            ph,
            pw,
            x_idx,
            y_idx,
            positions,
        }
    }
}

/// Bidirectional multi-head attention with per-head q/k norm, param-free
/// v-norm, and 2-D RoPE. `scale = 1.0` (the q/k norms set the magnitude).
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct VisionAttention {
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rope_theta: f32,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: nn::RmsNorm,
    #[param]
    pub v_norm: RmsNormNoScale,
}

impl VisionAttention {
    pub fn new(cfg: &VisionConfig) -> Result<Self, Error> {
        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        Ok(Self {
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            rope_theta: cfg.rope_theta,
            q_proj: linear(h, cfg.num_attention_heads * hd)?,
            k_proj: linear(h, cfg.num_key_value_heads * hd)?,
            v_proj: linear(h, cfg.num_key_value_heads * hd)?,
            o_proj: linear(cfg.num_attention_heads * hd, h)?,
            q_norm: rms_norm(hd, cfg.rms_norm_eps)?,
            k_norm: rms_norm(hd, cfg.rms_norm_eps)?,
            v_norm: RmsNormNoScale::new(cfg.rms_norm_eps),
        })
    }

    pub fn forward(&mut self, x: &Array, grid: &PatchGrid) -> Result<Array, Error> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);

        let q = reshape(
            &self.q_proj.forward(x)?,
            &[b, l, self.num_heads, self.head_dim],
        )?;
        let k = reshape(
            &self.k_proj.forward(x)?,
            &[b, l, self.num_kv_heads, self.head_dim],
        )?;
        let v = reshape(
            &self.v_proj.forward(x)?,
            &[b, l, self.num_kv_heads, self.head_dim],
        )?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let v = self.v_norm.forward(&v)?;

        let q = apply_vision_rope(&q, &grid.positions, self.rope_theta)?;
        let k = apply_vision_rope(&k, &grid.positions, self.rope_theta)?;

        let q = transpose_axes(&q, &[0, 2, 1, 3])?;
        let k = transpose_axes(&k, &[0, 2, 1, 3])?;
        let v = transpose_axes(&v, &[0, 2, 1, 3])?;

        let out = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            1.0,
            Option::<ScaledDotProductAttentionMask<'_>>::None,
            None,
        )?;
        let out = transpose_axes(&out, &[0, 2, 1, 3])?;
        let out = reshape(&out, &[b, l, -1])?;
        Ok(self.o_proj.forward(&out)?)
    }
}

/// GeGLU MLP: `down(gelu(gate(x)) * up(x))`.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct VisionMlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,

    cache: GegluCache,
}

impl VisionMlp {
    pub fn new(cfg: &VisionConfig) -> Result<Self, Error> {
        Ok(Self {
            gate_proj: linear(cfg.hidden_size, cfg.intermediate_size)?,
            up_proj: linear(cfg.hidden_size, cfg.intermediate_size)?,
            down_proj: linear(cfg.intermediate_size, cfg.hidden_size)?,
            cache: GegluCache::default(),
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let act = geglu(&mut self.cache, &gate, &up)?;
        Ok(self.down_proj.forward(&act)?)
    }
}

/// Encoder layer with gemma sandwich norms (input/post-attn/pre-ff/post-ff).
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct VisionEncoderLayer {
    #[quantizable]
    #[param]
    pub self_attn: VisionAttention,
    #[quantizable]
    #[param]
    pub mlp: VisionMlp,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    pub post_feedforward_layernorm: nn::RmsNorm,
}

impl VisionEncoderLayer {
    pub fn new(cfg: &VisionConfig) -> Result<Self, Error> {
        let n = || rms_norm(cfg.hidden_size, cfg.rms_norm_eps);
        Ok(Self {
            self_attn: VisionAttention::new(cfg)?,
            mlp: VisionMlp::new(cfg)?,
            input_layernorm: n()?,
            post_attention_layernorm: n()?,
            pre_feedforward_layernorm: n()?,
            post_feedforward_layernorm: n()?,
        })
    }

    pub fn forward(&mut self, x: &Array, grid: &PatchGrid) -> Result<Array, Error> {
        let attn = self
            .self_attn
            .forward(&self.input_layernorm.forward(x)?, grid)?;
        let h = x.add(&self.post_attention_layernorm.forward(&attn)?)?;
        let ff = self
            .mlp
            .forward(&self.pre_feedforward_layernorm.forward(&h)?)?;
        Ok(h.add(&self.post_feedforward_layernorm.forward(&ff)?)?)
    }
}

/// Average-pool the `k×k` patch grid, scaled by `√hidden_size`, yielding
/// `(ph/k)·(pw/k)` soft tokens (the real count, no padding — the processor
/// resizes to a `pooling_kernel_size`-divisible grid so the
/// `[B, ph/k, k, pw/k, k, H]` reshape-mean is exact).
#[derive(Debug, Clone, ModuleParameters)]
pub struct VisionPooler {
    kernel: i32,
    root_hidden: f32,
}

impl VisionPooler {
    pub fn new(cfg: &VisionConfig) -> Self {
        Self {
            kernel: cfg.pooling_kernel_size,
            root_hidden: (cfg.hidden_size as f32).sqrt(),
        }
    }

    /// `states [B, ph*pw, H]` → `[B, (ph/k)*(pw/k), H]`.
    pub fn forward(&self, states: &Array, grid: &PatchGrid) -> Result<Array, Error> {
        let shape = states.shape();
        let (b, hidden) = (shape[0], shape[2]);
        let k = self.kernel;
        if grid.ph % k != 0 || grid.pw % k != 0 {
            return Err(Error::shape(format!(
                "VisionPooler: grid {}x{} not divisible by pooling_kernel_size {k}",
                grid.ph, grid.pw
            )));
        }
        let pooled_len = (grid.ph / k) * (grid.pw / k);
        let r = reshape(states, &[b, grid.ph / k, k, grid.pw / k, k, hidden])?;
        let r = r.mean_axes(&[2, 4], false)?;
        let r = reshape(&r, &[b, pooled_len, hidden])?;
        Ok(r.multiply(Array::from_f32(self.root_hidden).as_dtype(r.dtype())?)?)
    }
}

/// The full Gemma 4 vision tower.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct VisionModel {
    standardize: bool,

    #[param]
    pub patch_embedder: PatchEmbedder,
    #[quantizable]
    #[param]
    pub encoder: Vec<VisionEncoderLayer>,
    pub pooler: VisionPooler,

    #[param]
    pub std_bias: Param<Option<Array>>,
    #[param]
    pub std_scale: Param<Option<Array>>,
}

impl VisionModel {
    pub fn new(cfg: &VisionConfig) -> Result<Self, Error> {
        let encoder = (0..cfg.num_hidden_layers)
            .map(|_| VisionEncoderLayer::new(cfg))
            .collect::<Result<Vec<_>, _>>()?;
        let (std_bias, std_scale) = if cfg.standardize {
            (
                Param::new(Some(Array::zeros::<f32>(&[cfg.hidden_size])?)),
                Param::new(Some(Array::ones::<f32>(&[cfg.hidden_size])?)),
            )
        } else {
            (Param::new(None), Param::new(None))
        };
        Ok(Self {
            standardize: cfg.standardize,
            patch_embedder: PatchEmbedder::new(cfg)?,
            encoder,
            pooler: VisionPooler::new(cfg),
            std_bias,
            std_scale,
        })
    }

    /// `pixel_values [B, C, H, W]` (B=1) → `[B, (ph/k)*(pw/k), hidden_size]`.
    /// `H`/`W` must be `patch_size`-divisible. Standardize runs *after* the
    /// pooler's `√hidden` scaling (the bias is per-token, not per-patch).
    pub fn forward(&mut self, pixel_values: &Array, grid: PatchGrid) -> Result<Array, Error> {
        let mut h = self.patch_embedder.forward(pixel_values, grid.clone())?;
        for layer in &mut self.encoder {
            h = layer.forward(&h, &grid)?;
        }
        h = self.pooler.forward(&h, &grid)?;
        if self.standardize {
            let bias = self
                .std_bias
                .value
                .as_ref()
                .expect("standardize => std_bias");
            let scale = self
                .std_scale
                .value
                .as_ref()
                .expect("standardize => std_scale");
            let dt = h.dtype();
            h = h
                .subtract(&bias.as_dtype(dt)?)?
                .multiply(&scale.as_dtype(dt)?)?;
        }
        Ok(h)
    }
}

/// Projector `embed_vision`: param-free RMS norm → quantized Linear from the
/// vision hidden size to the text hidden size.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct EmbedVision {
    pub norm: RmsNormNoScale,
    #[quantizable]
    #[param]
    pub embedding_projection: MaybeQuantized<nn::Linear>,
}

impl EmbedVision {
    pub fn new(cfg: &VisionConfig, text_hidden_size: i32) -> Result<Self, Error> {
        Ok(Self {
            norm: RmsNormNoScale::new(cfg.rms_norm_eps),
            embedding_projection: linear(cfg.hidden_size, text_hidden_size)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let normed = self.norm.forward(x)?;
        Ok(self.embedding_projection.forward(&normed)?)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;
    use mlx_rs::random::uniform;

    fn synthetic_config() -> VisionConfig {
        let json = serde_json::json!({
            "model_type": "gemma4_vision",
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "patch_size": 4,
            "pooling_kernel_size": 2,
            "default_output_length": 4,
            "position_embedding_size": 16,
            "rms_norm_eps": 1e-6,
            "rope_theta": 100.0,
            "standardize": true,
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn patch_grid_row_major() {
        let g = PatchGrid::new(2, 3);
        assert_eq!(g.x_idx.as_slice::<i32>(), &[0, 1, 2, 0, 1, 2]);
        assert_eq!(g.y_idx.as_slice::<i32>(), &[0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn vision_model_shape_round_trip() {
        let cfg = synthetic_config();
        let mut vm = VisionModel::new(&cfg).unwrap();
        // 4x4 patch grid (16 patches) pools by 2x2 -> 2x2 = 4 tokens.
        let (ph, pw) = (4, 4);
        let (h, w) = (ph * cfg.patch_size, pw * cfg.patch_size);
        let pixels = uniform::<_, f32>(0.0, 1.0, &[1, 3, h, w], None).unwrap();
        let out = vm.forward(&pixels, PatchGrid::new(ph, pw)).unwrap();
        assert_eq!(
            out.shape(),
            &[1, cfg.default_output_length, cfg.hidden_size]
        );
    }

    #[test]
    fn embed_vision_projects_to_text_hidden() {
        let cfg = synthetic_config();
        let mut ev = EmbedVision::new(&cfg, 48).unwrap();
        let x = uniform::<_, f32>(0.0, 1.0, &[4, cfg.hidden_size], None).unwrap();
        let out = ev.forward(&x).unwrap();
        assert_eq!(out.shape(), &[4, 48]);
    }
}
