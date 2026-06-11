//! Qwen3.5 text decoder building blocks: full-attention layer
//! (`Qwen3_5Attention`) and the SwiGLU MLP.

use mlx_rs::{
    builder::Builder,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::{reshape, split, transpose_axes},
    quantization::MaybeQuantized,
    Array,
};

use super::config::TextConfig;
use super::rope::apply_multimodal_rotary_pos_emb;
use crate::activations::{attention_gate, swiglu, AttentionGateCache, SwigluCache};
use crate::cache::KeyValueCache;
use crate::error::Error;

/// SwiGLU feed-forward block: `down(silu(gate(x)) * up(x))`.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Mlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,

    /// Per-layer compiled-graph cache for [`swiglu`].
    swiglu_cache: SwigluCache,
}

impl Mlp {
    /// Build a freshly-initialised MLP with the given inner widths.
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Error> {
        let gate_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;
        Ok(Self {
            gate_proj: MaybeQuantized::Original(gate_proj),
            down_proj: MaybeQuantized::Original(down_proj),
            up_proj: MaybeQuantized::Original(up_proj),
            swiglu_cache: SwigluCache::default(),
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Error;

    /// SwiGLU forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
    fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = swiglu(&mut self.swiglu_cache, &gate, &up)?;
        Ok(self.down_proj.forward(&activated)?)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

/// Full-attention block used at every `full_attention_interval`-th layer.
///
/// Differences from a vanilla GQA attention:
///
/// - `q_proj` outputs `n_heads * head_dim * 2` features and is split into
///   queries and a per-head gate (`attn_output_gate`). The attention output is
///   element-wise multiplied by `sigmoid(gate)` before `o_proj`.
/// - Rotary embedding is the multimodal partial-rotary mrope from
///   [`super::rope::MultimodalRope`]; only the first
///   `head_dim * partial_rotary_factor` features get rotated. cos/sin
///   are precomputed once per forward at the decoder level and shared
///   across every layer — see
///   [`super::layer::Qwen35Decoder::forward_pre_and_post_norm`].
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

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

    /// Per-layer compiled-graph cache for [`attention_gate`].
    attention_gate_cache: AttentionGateCache,
}

impl Attention {
    /// Build a freshly-initialised attention block from a [`TextConfig`].
    pub fn new(cfg: &TextConfig) -> Result<Self, Error> {
        let dim = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        // q_proj has 2× the head_dim per head — half is queries, half is the gate.
        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim * 2)
            .bias(cfg.attention_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(cfg.attention_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(cfg.attention_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(cfg.attention_bias)
            .build()?;

        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(cfg.rms_norm_eps)
            .build()?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            q_norm,
            k_norm,
            attention_gate_cache: AttentionGateCache::default(),
        })
    }

    /// `[B, L, hidden] -> [B, L, hidden]`. Cache is generic so the
    /// adapter picks the backing (see `CacheOptions`).
    pub fn forward<C: KeyValueCache>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut C>,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array, Error> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];

        // Project once, then split queries from the gate.
        let qp = self.q_proj.forward(x)?;
        let qp = reshape(&qp, &[b, l, self.n_heads, 2 * self.head_dim])?;
        let qg = split(&qp, 2, -1)?;
        let queries = &qg[0];
        let gate = reshape(&qg[1], &[b, l, self.n_heads * self.head_dim])?;

        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // [B, L, H, D] -> [B, H, L, D] for q/k/v.
        let queries = self.q_norm.forward(queries)?;
        let queries = transpose_axes(&queries, &[0, 2, 1, 3])?;

        let keys = reshape(&keys, &[b, l, self.n_kv_heads, self.head_dim])?;
        let keys = self.k_norm.forward(&keys)?;
        let keys = transpose_axes(&keys, &[0, 2, 1, 3])?;

        let values = reshape(&values, &[b, l, self.n_kv_heads, self.head_dim])?;
        let values = transpose_axes(&values, &[0, 2, 1, 3])?;

        let (queries, keys) = apply_multimodal_rotary_pos_emb(&queries, &keys, cos, sin)?;

        // Cache owns kernel dispatch (steel prefill / fused qsdpa / SDPA).
        // No cache → bare SDPA, test paths only.
        let output = if let Some(cache) = cache {
            cache.attention(&queries, keys, values, self.scale, mask)?
        } else {
            match mask {
                Some(m) => scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    self.scale,
                    ScaledDotProductAttentionMask::Array(m),
                    None,
                )?,
                None if l > 1 => scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    self.scale,
                    ScaledDotProductAttentionMask::Causal,
                    None,
                )?,
                None => scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    self.scale,
                    Option::<ScaledDotProductAttentionMask<'_>>::None,
                    None,
                )?,
            }
        };
        let output = transpose_axes(&output, &[0, 2, 1, 3])?;
        let output = reshape(&output, &[b, l, -1])?;

        let gated = attention_gate(&mut self.attention_gate_cache, &output, &gate)?;
        Ok(self.o_proj.forward(&gated)?)
    }

    /// Toggle training mode on every quantisable parameter.
    pub fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        self.k_norm.training_mode(mode);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    #![allow(clippy::print_stdout, reason = "test code")]
    #![allow(clippy::print_stderr, reason = "test code")]
    use super::super::config::RopeParameters;
    use super::super::rope::MultimodalRope;
    use super::*;
    use crate::cache::{KVCache, KeyValueCache};
    use mlx_rs::ops::{arange, broadcast_to, expand_dims, reshape as reshape_op};
    use mlx_rs::{random::uniform, transforms::eval};

    /// Build a `MultimodalRope` matching `cfg` and materialise cos/sin
    /// shaped + dtype-cast the way `Attention::forward` expects: shape
    /// `[B, 1, S, rotary_dim]`, cast to `Float32` (the dtype the test
    /// inputs use). Production code does the same expand+cast inside
    /// `Qwen35Decoder::cos_sin_for_forward` once per forward — these
    /// tests call `Attention::forward` directly so they reproduce the
    /// prep locally.
    fn rope_cos_sin(cfg: &TextConfig, b: i32, offset: i32, l: i32) -> (Array, Array) {
        let head_dim = cfg.head_dim;
        let rotary_dim =
            (head_dim as f32 * cfg.rope_parameters.partial_rotary_factor).floor() as i32;
        let rope = MultimodalRope::new(
            rotary_dim,
            cfg.rope_parameters.rope_theta,
            &cfg.rope_parameters.mrope_section,
        )
        .unwrap();
        let range = arange::<_, i32>(offset, offset + l, None).unwrap();
        let range = reshape_op(&range, &[1, l]).unwrap();
        let pos = broadcast_to(&range, &[b, l]).unwrap();
        let (cos, sin) = rope.cos_sin(&pos).unwrap();
        // Match `Qwen35Decoder::cos_sin_for_forward`: unsqueeze the
        // per-head broadcast axis (axis=1). Test inputs are f32, which
        // is also `cos_sin`'s native dtype, so no `.as_dtype` cast
        // is needed here.
        let cos = expand_dims(&cos, 1).unwrap();
        let sin = expand_dims(&sin, 1).unwrap();
        (cos, sin)
    }

    fn synthetic_text_config() -> TextConfig {
        let json = serde_json::json!({
            "model_type": "qwen3_5_text",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "vocab_size": 100,
            "max_position_embeddings": 256,
            "layer_types": ["full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {
                "mrope_section": [2, 1, 1],
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "type": "default"
            }
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn unsupported_rope_type_rejected_at_deserialize() {
        let json = serde_json::json!({
            "mrope_section": [2, 1, 1],
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "type": "yarn"
        });
        let err = serde_json::from_value::<RopeParameters>(json).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("yarn"), "unexpected error: {msg}");
    }

    #[test]
    fn mlp_shape_round_trips() {
        let mut m = Mlp::new(32, 64).unwrap();
        let x = uniform::<_, f32>(0.0, 1.0, &[2, 4, 32], None).unwrap();
        let y = m.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 4, 32]);
    }

    #[test]
    fn attention_shape_round_trips_without_cache() {
        let cfg = synthetic_text_config();
        let mut a = Attention::new(&cfg).unwrap();
        let x = uniform::<_, f32>(0.0, 1.0, &[2, 4, cfg.hidden_size], None).unwrap();
        let (cos, sin) = rope_cos_sin(&cfg, 2, 0, 4);
        let y = a.forward::<KVCache>(&x, None, None, &cos, &sin).unwrap();
        assert_eq!(y.shape(), &[2, 4, cfg.hidden_size]);
    }

    #[test]
    fn attention_prefill_then_decode_extends_cache() {
        let cfg = synthetic_text_config();
        let mut a = Attention::new(&cfg).unwrap();

        let prompt = uniform::<_, f32>(0.0, 1.0, &[1, 5, cfg.hidden_size], None).unwrap();
        let (cos_p, sin_p) = rope_cos_sin(&cfg, 1, 0, 5);
        let mut cache = KVCache::new();
        let prefill_out = a
            .forward(&prompt, None, Some(&mut cache), &cos_p, &sin_p)
            .unwrap();
        eval([&prefill_out]).unwrap();
        assert_eq!(prefill_out.shape(), &[1, 5, cfg.hidden_size]);
        assert_eq!(cache.offset(), 5);

        let next = uniform::<_, f32>(0.0, 1.0, &[1, 1, cfg.hidden_size], None).unwrap();
        let (cos_d, sin_d) = rope_cos_sin(&cfg, 1, 5, 1);
        let decode_out = a
            .forward(&next, None, Some(&mut cache), &cos_d, &sin_d)
            .unwrap();
        eval([&decode_out]).unwrap();
        assert_eq!(decode_out.shape(), &[1, 1, cfg.hidden_size]);
        assert_eq!(cache.offset(), 6);
    }
}
