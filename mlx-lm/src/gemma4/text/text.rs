//! Gemma 4 dense text decoder: hybrid sliding/global attention, four
//! norms per layer, GeGLU MLP, embedding scaling, logit soft-capping,
//! tied embeddings.
//!
//! MoE expert routing, per-layer-input embeddings (E2B/E4B), and KV
//! sharing are deferred; each extends this base at its own consumer.

use std::collections::HashMap;

use mlx_rs::{
    builder::Builder,
    macros::{ModuleParameters, Quantizable},
    module::{Module, Param},
    nn,
    ops::{clip, expand_dims_axes, indexing::IndexOp, unflatten},
    quantization::MaybeQuantized,
    Array, Dtype,
};

use crate::activations::{
    geglu, gelu_approximate_in_dtype, logit_softcap, residual_add_scale, GegluCache,
    LogitSoftcapCache, ResidualAddScaleCache,
};
use crate::cache::KeyValueCache;
use crate::error::Error;
use crate::gemma4::text::config::{LayerKind, TextConfig};
use crate::gemma4::text::moe::{Experts, Router};
use crate::gemma4::text::rope::{build_layer_rope, LayerRope};
use crate::nn::{ModelInput, RmsNormNoScale};
use crate::utils::{create_attention_mask, AttentionMask};

/// fp16 max magnitude — residual sums are clipped to this before casting
/// back to fp16 to avoid overflow → inf.
const FP16_MAX: f32 = 65504.0;

/// Per-layer attention input. The dense base never sets `shared_kv` /
/// `offset` (always `None`); they are the seam the KV-sharing extension
/// consumes. Local to gemma4 so the shared [`crate::nn::AttentionInput`]
/// stays untouched by a gemma-only concern.
pub struct GemmaAttnInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
    pub shared_kv: Option<(Array, Array)>,
    pub offset: Option<i32>,
}

/// Hidden state + the layer's `(k, v)` (for downstream KV-shared layers)
/// + the pre-update offset.
pub struct AttentionOut {
    pub h: Array,
    pub shared_kv: (Array, Array),
    pub offset: i32,
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub layer_idx: i32,
    pub layer_kind: LayerKind,
    pub is_sliding: bool,
    pub use_k_eq_v: bool,
    /// `false` for KV-shared layers (reuse a prior layer's K/V).
    pub has_kv: bool,

    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    /// `None` on KV-shared layers; also `None` for v_proj when K == V.
    #[quantizable]
    #[param]
    pub k_proj: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    pub v_proj: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,

    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: Option<nn::RmsNorm>,
    #[param]
    pub v_norm: Option<RmsNormNoScale>,

    #[param]
    pub rope: LayerRope,
}

impl Attention {
    pub fn new(args: &TextConfig, layer_idx: i32) -> Result<Self, Error> {
        let layer_kind = args.layer_types_resolved()[layer_idx as usize];
        let is_sliding = matches!(layer_kind, LayerKind::SlidingAttention);

        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let head_dim = if matches!(layer_kind, LayerKind::FullAttention) {
            args.global_head_dim
        } else {
            args.head_dim
        };

        let first_kv_shared = args.num_hidden_layers - args.num_kv_shared_layers;
        let has_kv = layer_idx < first_kv_shared;

        let use_k_eq_v = args.attention_k_eq_v && !is_sliding;
        let n_kv_heads = match (use_k_eq_v, args.num_global_key_value_heads) {
            (true, Some(h)) => h,
            _ => args.num_key_value_heads,
        };

        let scale = 1.0_f32;

        let linear = |inp: i32, out: i32| -> Result<MaybeQuantized<nn::Linear>, Error> {
            Ok(MaybeQuantized::Original(
                nn::LinearBuilder::new(inp, out).bias(false).build()?,
            ))
        };
        let q_proj = linear(dim, n_heads * head_dim)?;
        let (k_proj, v_proj) = if has_kv {
            let k = linear(dim, n_kv_heads * head_dim)?;
            let v = if use_k_eq_v {
                None
            } else {
                Some(linear(dim, n_kv_heads * head_dim)?)
            };
            (Some(k), v)
        } else {
            (None, None)
        };
        let o_proj = linear(n_heads * head_dim, dim)?;

        let norm = |d: i32| -> Result<nn::RmsNorm, Error> {
            Ok(nn::RmsNormBuilder::new(d).eps(args.rms_norm_eps).build()?)
        };
        let q_norm = norm(head_dim)?;
        let (k_norm, v_norm) = if has_kv {
            (
                Some(norm(head_dim)?),
                Some(RmsNormNoScale::new(args.rms_norm_eps)),
            )
        } else {
            (None, None)
        };

        let rope = build_layer_rope(
            head_dim,
            layer_kind,
            args.rope_traditional,
            args.rope_parameters.as_ref(),
        )?;

        Ok(Self {
            layer_idx,
            layer_kind,
            is_sliding,
            use_k_eq_v,
            has_kv,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            rope,
        })
    }

    #[allow(
        non_snake_case,
        reason = "local bindings mirror ML tensor names (B, L)"
    )]
    pub fn attend<C: KeyValueCache>(
        &mut self,
        input: GemmaAttnInput<'_, C>,
    ) -> Result<AttentionOut, Error> {
        let GemmaAttnInput {
            x,
            mask,
            mut cache,
            shared_kv,
            offset,
        } = input;
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        // Pre-update offset: what RoPE applies to fresh queries. Kept on
        // device (0-D Array) so the dynamic-rope kernel cache stays warm.
        let pre_offset = match (offset, cache.as_ref()) {
            (Some(o), _) => o,
            (None, Some(c)) => c.offset(),
            (None, None) => 0,
        };
        let pre_offset_arr = Array::from_int(pre_offset);

        let queries = self
            .q_proj
            .forward(x)?
            .reshape(&[B, L, self.n_heads, self.head_dim])?;
        let mut queries = self.q_norm.forward(&queries)?;

        let (keys, values) = if let Some(kv) = shared_kv {
            kv
        } else {
            if !self.has_kv {
                return Err(Error::config(format!(
                    "gemma4: layer {} is KV-shared but no shared_kv supplied",
                    self.layer_idx
                )));
            }
            let keys = self
                .k_proj
                .as_mut()
                .expect("has_kv guarantees k_proj")
                .forward(x)?
                .reshape(&[B, L, self.n_kv_heads, self.head_dim])?;
            let k_for_attn = self
                .k_norm
                .as_mut()
                .expect("has_kv guarantees k_norm")
                .forward(&keys)?
                .transpose_axes(&[0, 2, 1, 3])?;
            let k_for_attn = self.rope.forward_dynamic(&k_for_attn, &pre_offset_arr)?;

            let values = if self.use_k_eq_v {
                keys.clone()
            } else {
                self.v_proj
                    .as_mut()
                    .expect("non-keqv layer has v_proj")
                    .forward(x)?
                    .reshape(&[B, L, self.n_kv_heads, self.head_dim])?
            };
            let v_for_attn = self
                .v_norm
                .as_mut()
                .expect("has_kv guarantees v_norm")
                .forward(&values)?
                .transpose_axes(&[0, 2, 1, 3])?;

            (k_for_attn, v_for_attn)
        };

        queries = queries.transpose_axes(&[0, 2, 1, 3])?;
        queries = self.rope.forward_dynamic(&queries, &pre_offset_arr)?;

        // Concat with cache, then attend. Downstream KV-shared layers reuse
        // `(k_full, v_full)`.
        let (k_full, v_full) = if let Some(cache) = cache.as_mut() {
            cache.update_and_fetch(keys, values)?
        } else {
            (keys, values)
        };
        let h = mlx_rs::fast::scaled_dot_product_attention(
            &queries,
            &k_full,
            &v_full,
            self.scale,
            mask.map(mlx_rs::fast::ScaledDotProductAttentionMask::Array),
            None,
        )?;

        let h = h.transpose_axes(&[0, 2, 1, 3])?.reshape(&[B, L, -1])?;
        let h = self.o_proj.forward(&h)?;

        Ok(AttentionOut {
            h,
            shared_kv: (k_full, v_full),
            offset: pre_offset,
        })
    }

    pub fn training_mode_set(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        if let Some(k) = self.k_proj.as_mut() {
            k.training_mode(mode);
        }
        if let Some(v) = self.v_proj.as_mut() {
            v.training_mode(mode);
        }
        self.o_proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        if let Some(k) = self.k_norm.as_mut() {
            k.training_mode(mode);
        }
        if let Some(v) = self.v_norm.as_mut() {
            v.training_mode(mode);
        }
    }
}

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
    geglu_cache: GegluCache,
}

impl Mlp {
    /// `intermediate_size` is the effective width (doubled for double-wide layers).
    pub fn new(args: &TextConfig, intermediate_size: i32) -> Result<Self, Error> {
        let linear = |inp: i32, out: i32| -> Result<MaybeQuantized<nn::Linear>, Error> {
            Ok(MaybeQuantized::Original(
                nn::LinearBuilder::new(inp, out).bias(false).build()?,
            ))
        };
        Ok(Self {
            gate_proj: linear(args.hidden_size, intermediate_size)?,
            down_proj: linear(intermediate_size, args.hidden_size)?,
            up_proj: linear(args.hidden_size, intermediate_size)?,
            geglu_cache: GegluCache::default(),
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Error;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = geglu(&mut self.geglu_cache, &gate, &up)?;
        Ok(self.down_proj.forward(&activated)?)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

/// fp16-safe additive residual: promote → add → clip → cast back. No-op
/// for non-fp16.
fn clip_residual(x: &Array, y: &Array) -> Result<Array, Error> {
    if x.dtype() != Dtype::Float16 {
        return Ok(x.add(y)?);
    }
    let xf = x.as_dtype(Dtype::Float32)?;
    let yf = y.as_dtype(Dtype::Float32)?;
    let sum = xf.add(&yf)?;
    Ok(clip(&sum, (-FP16_MAX, FP16_MAX))?.as_dtype(Dtype::Float16)?)
}

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct DecoderLayer {
    pub layer_idx: i32,
    pub layer_kind: LayerKind,
    /// `true` for the 26b-a4b MoE variant — gates `router`/`experts` and
    /// the dual-branch FFN. Dense layers keep all MoE fields `None`.
    pub enable_moe: bool,

    #[quantizable]
    #[param]
    pub self_attn: Attention,
    #[quantizable]
    #[param]
    pub mlp: Mlp,
    /// MoE router (`Some` iff `enable_moe`).
    #[quantizable]
    #[param]
    pub router: Option<Router>,
    /// MoE experts (`Some` iff `enable_moe`).
    #[quantizable]
    #[param]
    pub experts: Option<Experts>,

    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    pub post_feedforward_layernorm: nn::RmsNorm,
    /// Post-norm on the dense branch of a MoE layer (`Some` iff `enable_moe`).
    #[param]
    pub post_feedforward_layernorm_1: Option<nn::RmsNorm>,
    /// Pre-norm on the expert branch of a MoE layer (`Some` iff `enable_moe`).
    #[param]
    pub pre_feedforward_layernorm_2: Option<nn::RmsNorm>,
    /// Post-norm on the expert branch of a MoE layer (`Some` iff `enable_moe`).
    #[param]
    pub post_feedforward_layernorm_2: Option<nn::RmsNorm>,

    /// Per-layer-input gating (`Some` iff per-layer-input on):
    /// `h += post_norm(proj(gelu(gate(h)) * per_layer_input))`.
    #[quantizable]
    #[param]
    pub per_layer_input_gate: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    pub per_layer_projection: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    pub post_per_layer_input_norm: Option<nn::RmsNorm>,

    /// Multiplicative per-layer scalar on the residual stream.
    #[param]
    pub layer_scalar: Param<Array>,

    residual_scale_cache: ResidualAddScaleCache,
}

impl DecoderLayer {
    pub fn new(args: &TextConfig, layer_idx: i32) -> Result<Self, Error> {
        let layer_kind = args.layer_types_resolved()[layer_idx as usize];
        let norm = || -> Result<nn::RmsNorm, Error> {
            Ok(nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?)
        };

        let enable_moe = args.enable_moe_block;
        let (router, experts, post_ff_1, pre_ff_2, post_ff_2) = if enable_moe {
            let num_experts = args
                .num_experts
                .ok_or_else(|| Error::config("gemma4 MoE: num_experts missing"))?;
            let top_k = args
                .top_k_experts
                .ok_or_else(|| Error::config("gemma4 MoE: top_k_experts missing"))?;
            let moe_intermediate = args
                .moe_intermediate_size
                .ok_or_else(|| Error::config("gemma4 MoE: moe_intermediate_size missing"))?;
            (
                Some(Router::new(
                    args.hidden_size,
                    num_experts,
                    top_k,
                    args.rms_norm_eps,
                )?),
                Some(Experts::new(
                    args.hidden_size,
                    moe_intermediate,
                    num_experts,
                    top_k,
                )?),
                Some(norm()?),
                Some(norm()?),
                Some(norm()?),
            )
        } else {
            (None, None, None, None, None)
        };

        let pl = args.hidden_size_per_layer_input;
        let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) = if pl > 0 {
            let linear = |inp: i32, out: i32| -> Result<MaybeQuantized<nn::Linear>, Error> {
                Ok(MaybeQuantized::Original(
                    nn::LinearBuilder::new(inp, out).bias(false).build()?,
                ))
            };
            (
                Some(linear(args.hidden_size, pl)?),
                Some(linear(pl, args.hidden_size)?),
                Some(
                    nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                ),
            )
        } else {
            (None, None, None)
        };

        // Double-wide dense MLP on KV-shared layers (E2B).
        let first_kv_shared = args.num_hidden_layers - args.num_kv_shared_layers;
        let is_kv_shared = layer_idx >= first_kv_shared && args.num_kv_shared_layers > 0;
        let mlp_intermediate = if args.use_double_wide_mlp && is_kv_shared {
            args.intermediate_size * 2
        } else {
            args.intermediate_size
        };

        Ok(Self {
            layer_idx,
            layer_kind,
            enable_moe,
            self_attn: Attention::new(args, layer_idx)?,
            mlp: Mlp::new(args, mlp_intermediate)?,
            router,
            experts,
            input_layernorm: norm()?,
            post_attention_layernorm: norm()?,
            pre_feedforward_layernorm: norm()?,
            post_feedforward_layernorm: norm()?,
            post_feedforward_layernorm_1: post_ff_1,
            pre_feedforward_layernorm_2: pre_ff_2,
            post_feedforward_layernorm_2: post_ff_2,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            layer_scalar: Param::new(Array::ones::<f32>(&[1])?),
            residual_scale_cache: ResidualAddScaleCache::default(),
        })
    }

    /// `shared_kv`/`offset` feed the KV-sharing extension (E2B/E4B); the
    /// dense + MoE base always passes `None`. `per_layer_input` feeds the
    /// per-layer-input gating (E2B/E4B); `None` here. Returns the layer's
    /// `(k, v)` + pre-update offset so a downstream KV-shared layer can
    /// reuse them. These three args are the one forward-compat hook so the
    /// E2B/E4B follow-on is body-only.
    pub fn forward_layer<C: KeyValueCache>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut C>,
        shared_kv: Option<(Array, Array)>,
        offset: Option<i32>,
        per_layer_input: Option<&Array>,
    ) -> Result<AttentionOut, Error> {
        let h_pre = self.input_layernorm.forward(x)?;
        let AttentionOut {
            h,
            shared_kv: kv_out,
            offset: off_out,
        } = self.self_attn.attend(GemmaAttnInput {
            x: &h_pre,
            mask,
            cache,
            shared_kv,
            offset,
        })?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = clip_residual(x, &h)?;

        let ff_mid = if self.enable_moe {
            // Dense branch: post_1(MLP(pre_ff(h))).
            let h1 = self.pre_feedforward_layernorm.forward(&h)?;
            let h1 = self.mlp.forward(&h1)?;
            let h1 = self
                .post_feedforward_layernorm_1
                .as_mut()
                .expect("moe layer has post_ff_1")
                .forward(&h1)?;
            // Expert branch: post_2(Experts(pre_2(h), router(h))).
            let (idx, w) = self
                .router
                .as_mut()
                .expect("moe layer has router")
                .forward(&h)?;
            let h2 = self
                .pre_feedforward_layernorm_2
                .as_mut()
                .expect("moe layer has pre_ff_2")
                .forward(&h)?;
            let h2 = self
                .experts
                .as_mut()
                .expect("moe layer has experts")
                .forward(&h2, &idx, &w)?;
            let h2 = self
                .post_feedforward_layernorm_2
                .as_mut()
                .expect("moe layer has post_ff_2")
                .forward(&h2)?;
            h1.add(&h2)?
        } else {
            let mid = self.pre_feedforward_layernorm.forward(&h)?;
            self.mlp.forward(&mid)?
        };
        // Both branches share the final post-norm before the residual add.
        let ff_out = self.post_feedforward_layernorm.forward(&ff_mid)?;

        // Per-layer-input gating sits between the FFN residual and the
        // layer-scalar multiply, so it forces the unfused path.
        let pl_active = per_layer_input.is_some()
            && self.per_layer_input_gate.is_some()
            && self.per_layer_projection.is_some()
            && self.post_per_layer_input_norm.is_some();

        // bf16/fp32 + no per-layer gating: fuse `(h + ff_out) * layer_scalar`.
        if !pl_active && ff_out.dtype() != Dtype::Float16 {
            let out = residual_add_scale(
                &mut self.residual_scale_cache,
                &h,
                &ff_out,
                self.layer_scalar.as_ref(),
            )?;
            return Ok(AttentionOut {
                h: out,
                shared_kv: kv_out,
                offset: off_out,
            });
        }

        let mut h = clip_residual(&h, &ff_out)?;
        if let (Some(gate_l), Some(proj_l), Some(norm_l), Some(pl_in)) = (
            self.per_layer_input_gate.as_mut(),
            self.per_layer_projection.as_mut(),
            self.post_per_layer_input_norm.as_mut(),
            per_layer_input,
        ) {
            let g = gate_l.forward(&h)?;
            let g = gelu_approximate_in_dtype(&g)?;
            let g = g.multiply(pl_in)?;
            let g = proj_l.forward(&g)?;
            let g = norm_l.forward(&g)?;
            h = h.add(&g)?;
        }
        let out = h.multiply(self.layer_scalar.as_ref())?;
        Ok(AttentionOut {
            h: out,
            shared_kv: kv_out,
            offset: off_out,
        })
    }

    pub fn training_mode_set(&mut self, mode: bool) {
        self.self_attn.training_mode_set(mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
        self.pre_feedforward_layernorm.training_mode(mode);
        self.post_feedforward_layernorm.training_mode(mode);
    }
}

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Gemma4TextModel {
    pub vocab_size: i32,
    pub sliding_window_pattern: i32,
    pub embed_scale: f32,
    embed_scale_arr: std::sync::OnceLock<Array>,

    /// 0 disables the per-layer-input path.
    pub hidden_size_per_layer_input: i32,
    embed_tokens_per_layer_scale: f32,
    per_layer_input_scale: f32,
    per_layer_projection_scale: f32,
    embed_tokens_per_layer_scale_arr: std::sync::OnceLock<Array>,
    per_layer_input_scale_arr: std::sync::OnceLock<Array>,
    per_layer_projection_scale_arr: std::sync::OnceLock<Array>,
    /// `previous_kvs[j]` = the layer whose K/V layer `j` reuses (`== j` if not shared).
    pub previous_kvs: Vec<usize>,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub embed_tokens_per_layer: Option<MaybeQuantized<nn::Embedding>>,
    #[param]
    pub per_layer_model_projection: Option<nn::Linear>,
    #[param]
    pub per_layer_projection_norm: Option<nn::RmsNorm>,
    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

/// Map each KV-shared layer to the most-recent same-kind owned layer it
/// reuses K/V from. Identity for non-shared layers.
fn compute_previous_kvs(args: &TextConfig) -> Vec<usize> {
    let n = args.num_hidden_layers as usize;
    let mut previous_kvs: Vec<usize> = (0..n).collect();
    if args.num_kv_shared_layers <= 0 {
        return previous_kvs;
    }
    let first_kv_shared = (args.num_hidden_layers - args.num_kv_shared_layers) as usize;
    let layer_types = args.layer_types_resolved();
    let mut kvs_by_kind: HashMap<LayerKind, usize> = HashMap::new();
    for (i, k) in layer_types.iter().enumerate().take(first_kv_shared) {
        kvs_by_kind.insert(*k, i);
    }
    for (j, slot) in previous_kvs.iter_mut().enumerate().skip(first_kv_shared) {
        if let Some(&src) = kvs_by_kind.get(&layer_types[j]) {
            *slot = src;
        }
    }
    previous_kvs
}

impl Gemma4TextModel {
    pub fn new(args: &TextConfig) -> Result<Self, Error> {
        let layers = (0..args.num_hidden_layers)
            .map(|i| DecoderLayer::new(args, i))
            .collect::<Result<Vec<_>, _>>()?;

        let pl = args.hidden_size_per_layer_input;
        let (embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm) =
            if pl > 0 {
                let wide = args.num_hidden_layers * pl;
                (
                    Some(MaybeQuantized::Original(nn::Embedding::new(
                        args.vocab_size_per_layer_input,
                        wide,
                    )?)),
                    Some(
                        nn::LinearBuilder::new(args.hidden_size, wide)
                            .bias(false)
                            .build()?,
                    ),
                    Some(nn::RmsNormBuilder::new(pl).eps(args.rms_norm_eps).build()?),
                )
            } else {
                (None, None, None)
            };

        Ok(Self {
            vocab_size: args.vocab_size,
            sliding_window_pattern: args.effective_sliding_window_pattern(),
            embed_scale: (args.hidden_size as f32).sqrt(),
            embed_scale_arr: std::sync::OnceLock::new(),
            hidden_size_per_layer_input: pl,
            embed_tokens_per_layer_scale: if pl > 0 { (pl as f32).sqrt() } else { 0.0 },
            per_layer_input_scale: (2.0_f32).powf(-0.5),
            per_layer_projection_scale: (args.hidden_size as f32).powf(-0.5),
            embed_tokens_per_layer_scale_arr: std::sync::OnceLock::new(),
            per_layer_input_scale_arr: std::sync::OnceLock::new(),
            per_layer_projection_scale_arr: std::sync::OnceLock::new(),
            previous_kvs: compute_previous_kvs(args),
            embed_tokens: MaybeQuantized::Original(nn::Embedding::new(
                args.vocab_size,
                args.hidden_size,
            )?),
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

impl<C> Module<ModelInput<'_, C>> for Gemma4TextModel
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Error;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput { inputs, cache, .. } = input;
        let h = self.embed_scaled(inputs)?;
        self.forward_from_hidden(h, inputs, cache)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            layer.training_mode_set(mode);
        }
        self.norm.training_mode(mode);
    }
}

impl Gemma4TextModel {
    /// `embed_tokens(ids) * embed_scale`, scale staged in the embedding
    /// dtype so the multiply stays bf16/fp16.
    pub fn embed_scaled(&mut self, inputs: &Array) -> Result<Array, Error> {
        let h = self.embed_tokens.forward(inputs)?;
        let h_dtype = h.dtype();
        let scale = self.embed_scale_arr.get_or_init(|| {
            Array::from_f32(self.embed_scale)
                .as_dtype(h_dtype)
                .expect("embed_scale cast cannot fail")
        });
        Ok(h.multiply(scale)?)
    }

    /// Decoder body shared by token and pre-stitched-embedding entry points:
    /// per-layer inputs, masks, the KV-shared layer loop, final norm. `inputs`
    /// (token ids) still drives the per-layer-input embedding even when `h`
    /// comes from stitched multimodal embeddings.
    fn forward_from_hidden<C: KeyValueCache>(
        &mut self,
        mut h: Array,
        inputs: &Array,
        cache: &mut [Option<C>],
    ) -> Result<Array, Error> {
        let h_dtype = h.dtype();

        // Per-layer inputs `[B, L, num_layers, pl]`, sliced axis-2 per layer.
        let per_layer_inputs: Option<Vec<Array>> = if self.hidden_size_per_layer_input > 0 {
            let pl = self.hidden_size_per_layer_input;
            let n = self.layers.len() as i32;
            let etps = self.embed_tokens_per_layer_scale_arr.get_or_init(|| {
                Array::from_f32(self.embed_tokens_per_layer_scale)
                    .as_dtype(h_dtype)
                    .expect("etpl scale cast cannot fail")
            });
            let pps = self.per_layer_projection_scale_arr.get_or_init(|| {
                Array::from_f32(self.per_layer_projection_scale)
                    .as_dtype(h_dtype)
                    .expect("pl proj scale cast cannot fail")
            });
            let pis = self.per_layer_input_scale_arr.get_or_init(|| {
                Array::from_f32(self.per_layer_input_scale)
                    .as_dtype(h_dtype)
                    .expect("pl input scale cast cannot fail")
            });
            let etpl = self
                .embed_tokens_per_layer
                .as_mut()
                .expect("hidden_size_per_layer_input>0 has embed_tokens_per_layer");
            let pli = etpl.forward(inputs)?.multiply(etps)?;
            let pli = unflatten(&pli, -1, &[n, pl])?;

            let proj = self
                .per_layer_model_projection
                .as_mut()
                .expect("hidden_size_per_layer_input>0 has per_layer_model_projection");
            let pproj = proj.forward(&h)?.multiply(pps)?;
            let pproj = unflatten(&pproj, -1, &[n, pl])?;
            let pproj = self
                .per_layer_projection_norm
                .as_mut()
                .expect("hidden_size_per_layer_input>0 has per_layer_projection_norm")
                .forward(&pproj)?;

            let combined = pproj.add(&pli)?.multiply(pis)?;
            Some((0..n).map(|i| combined.index((.., .., i, ..))).collect())
        } else {
            None
        };

        // Per-layer-kind masks: full-attn uses the global cache slot, sliding
        // uses slot 0 (a Sliding cache whose max_size bounds the window).
        // `return_array=Some(true)` forces explicit Array masks (the sliding
        // window restriction needs the array form). `global_idx` and slot 0
        // are owned layers (shared `None` slots are the trailing layers).
        let pattern = self.sliding_window_pattern as usize;
        let global_idx = pattern.saturating_sub(1).min(cache.len().saturating_sub(1));
        let global_mask = mask_array(create_attention_mask(
            &h,
            &cache[global_idx..=global_idx],
            Some(true),
        )?)?;
        let sliding_mask = if pattern > 1 {
            mask_array(create_attention_mask(&h, &cache[0..1], Some(true))?)?
        } else {
            None
        };

        // `intermediates[i]` holds each layer's `(k, v, offset)` so a shared
        // layer (`previous_kvs[i] != i`) can reuse its source's. Split-borrow
        // avoids cloning the index table per step.
        let previous_kvs = self.previous_kvs.as_slice();
        let layers = &mut self.layers;
        let mut intermediates: Vec<Option<(Array, Array, i32)>> = vec![None; layers.len()];

        for (i, layer) in layers.iter_mut().enumerate() {
            let mask = match layer.layer_kind {
                LayerKind::FullAttention => global_mask.as_ref(),
                LayerKind::SlidingAttention => sliding_mask.as_ref(),
            };
            let cache_slot = cache.get_mut(i).and_then(|c| c.as_mut());

            let (shared_kv, offset_in) = if previous_kvs[i] != i {
                match &intermediates[previous_kvs[i]] {
                    Some((k, v, off)) => (Some((k.clone(), v.clone())), Some(*off)),
                    None => (None, None),
                }
            } else {
                (None, None)
            };
            let pli = per_layer_inputs.as_ref().map(|v| &v[i]);

            let out = layer.forward_layer(&h, mask, cache_slot, shared_kv, offset_in, pli)?;
            h = out.h;
            intermediates[i] = Some((out.shared_kv.0, out.shared_kv.1, out.offset));
        }

        Ok(self.norm.forward(&h)?)
    }

    /// Decode pre-stitched `inputs_embeds`, bypassing `embed_tokens`. Text rows
    /// are pre-scaled by `embed_scale`; stitched vision features are not.
    #[cfg(feature = "image")]
    pub fn forward_embeds<C: KeyValueCache>(
        &mut self,
        inputs_embeds: Array,
        inputs: &Array,
        cache: &mut [Option<C>],
    ) -> Result<Array, Error> {
        self.forward_from_hidden(inputs_embeds, inputs, cache)
    }

    /// Raw token embeddings (no `embed_scale`); the VLM adapter scales text
    /// rows itself before stitching vision features in.
    #[cfg(feature = "image")]
    pub fn embed_tokens(&mut self, inputs: &Array) -> Result<Array, Error> {
        Ok(self.embed_tokens.forward(inputs)?)
    }
}

/// Extract the `Array` from an [`AttentionMask`], expanding a 2-D
/// `[T, kT]` mask to 4-D `[1, 1, T, kT]` so it broadcasts against
/// `[B, H, T, kT]` in the non-fused SDPA path. `Causal`/`None` → `None`.
fn mask_array(mask: Option<AttentionMask>) -> Result<Option<Array>, Error> {
    match mask {
        Some(AttentionMask::Array(a)) => {
            let a = if a.shape().len() == 2 {
                expand_dims_axes(&a, &[0, 1])?
            } else {
                a
            };
            Ok(Some(a))
        }
        _ => Ok(None),
    }
}

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: TextConfig,
    pub final_logit_softcapping: Option<f32>,

    #[quantizable]
    #[param]
    pub model: Gemma4TextModel,
    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,

    softcap_cache: LogitSoftcapCache,
    softcap_array: std::sync::OnceLock<Array>,
}

impl Model {
    pub fn new(args: TextConfig) -> Result<Self, Error> {
        let final_logit_softcapping = if args.final_logit_softcapping > 0.0 {
            Some(args.final_logit_softcapping)
        } else {
            None
        };
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        };
        let model = Gemma4TextModel::new(&args)?;
        Ok(Self {
            args,
            final_logit_softcapping,
            model,
            lm_head,
            softcap_cache: LogitSoftcapCache::default(),
            softcap_array: std::sync::OnceLock::new(),
        })
    }

    /// LM head + final-logit softcap, shared by `forward` and `forward_embeds`.
    fn apply_head(&mut self, out: &Array) -> Result<Array, Error> {
        let mut logits = if let Some(lm) = self.lm_head.as_mut() {
            lm.forward(out)?
        } else {
            match &self.model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(out)?,
                MaybeQuantized::Quantized(qe) => qe.as_linear(out)?,
            }
        };
        if let Some(cap) = self.final_logit_softcapping {
            let logits_dtype = logits.dtype();
            let cap_arr = self.softcap_array.get_or_init(|| {
                Array::from_f32(cap)
                    .as_dtype(logits_dtype)
                    .expect("cap cast cannot fail")
            });
            logits = logit_softcap(&mut self.softcap_cache, &logits, cap_arr)?;
        }
        Ok(logits)
    }

    /// VLM decode/prefill over pre-stitched embeddings; `inputs` (token ids)
    /// still drives the per-layer-input embedding.
    #[cfg(feature = "image")]
    pub fn forward_embeds<C: KeyValueCache>(
        &mut self,
        inputs_embeds: Array,
        inputs: &Array,
        cache: &mut [Option<C>],
    ) -> Result<Array, Error> {
        let out = self.model.forward_embeds(inputs_embeds, inputs, cache)?;
        self.apply_head(&out)
    }

    /// As [`Self::forward_embeds`] but also returns the post-norm hidden, so a
    /// VLM prefill can seed the MTP drafter's `prev_hidden` anchor — image
    /// affects prefill only, MTP then decodes off the populated cache.
    #[cfg(feature = "image")]
    pub fn forward_embeds_hidden_and_logits<C: KeyValueCache>(
        &mut self,
        inputs_embeds: Array,
        inputs: &Array,
        cache: &mut [Option<C>],
    ) -> Result<(Array, Array), Error> {
        let hidden = self.model.forward_embeds(inputs_embeds, inputs, cache)?;
        let logits = self.apply_head(&hidden)?;
        Ok((hidden, logits))
    }
}

impl Model {
    /// Forward returning `(post-norm hidden, logits)` — the MTP drafter needs
    /// the last-position hidden for its concat input.
    pub fn forward_hidden_and_logits<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        cache: &mut [Option<C>],
    ) -> Result<(Array, Array), Error> {
        let h = self.model.embed_scaled(inputs)?;
        let hidden = self.model.forward_from_hidden(h, inputs, cache)?;
        let logits = self.apply_head(&hidden)?;
        Ok((hidden, logits))
    }

    /// `embed_scale`d token embedding — the MTP drafter's concat input half.
    pub fn embed_scaled_token(&mut self, inputs: &Array) -> Result<Array, Error> {
        self.model.embed_scaled(inputs)
    }
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Error;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward(input)?;
        self.apply_head(&out)
    }

    fn training_mode(&mut self, mode: bool) {
        <Gemma4TextModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm) = self.lm_head.as_mut() {
            lm.training_mode(mode);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;
    use crate::cache::CacheOptions;
    use crate::gemma4::text::cache::make_caches;
    use mlx_rs::transforms::eval;

    /// Small synthetic gemma4 config: 3 sliding + 1 global layer, even
    /// head dims so rope is happy.
    fn synthetic() -> TextConfig {
        let json = serde_json::json!({
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "head_dim": 8,
            "global_head_dim": 8,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 100,
            // ≥ the test's prefill length: a single forward never exceeds
            // the sliding window (the adapter caps prefill chunks at the
            // window via `effective_prefill_chunk_opt`).
            "sliding_window": 8,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "layer_types": [
                "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention"
            ],
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn attention_head_dim_per_layer_kind() {
        let cfg = synthetic();
        let sliding = Attention::new(&cfg, 0).unwrap();
        let global = Attention::new(&cfg, 3).unwrap();
        assert_eq!(sliding.layer_kind, LayerKind::SlidingAttention);
        assert_eq!(global.layer_kind, LayerKind::FullAttention);
        // Both head dims are 8 here; the dispatch picks the right source.
        assert_eq!(sliding.head_dim, cfg.head_dim);
        assert_eq!(global.head_dim, cfg.global_head_dim);
    }

    #[test]
    fn decoder_forward_shape_round_trips() {
        let cfg = synthetic();
        let vocab = cfg.vocab_size;
        let mut model = Model::new(cfg.clone()).unwrap();
        let mut caches = make_caches(&cfg, CacheOptions::default());

        // Prefill 5 tokens.
        let ids: Vec<i32> = (0..5).collect();
        let inputs = Array::from_slice(&ids, &[1, 5]);
        let logits = model
            .forward(ModelInput {
                inputs: &inputs,
                mask: None,
                cache: &mut caches,
            })
            .unwrap();
        eval([&logits]).unwrap();
        assert_eq!(logits.shape(), &[1, 5, vocab]);

        // Decode one more token.
        let next = Array::from_slice(&[7_i32], &[1, 1]);
        let logits2 = model
            .forward(ModelInput {
                inputs: &next,
                mask: None,
                cache: &mut caches,
            })
            .unwrap();
        eval([&logits2]).unwrap();
        assert_eq!(logits2.shape(), &[1, 1, vocab]);

        // Sliding slot (0) is windowed; global slot (3) is unbounded.
        assert_eq!(caches[0].as_ref().unwrap().max_size(), Some(8));
        assert_eq!(caches[3].as_ref().unwrap().max_size(), None);
        assert_eq!(caches[3].as_ref().unwrap().offset(), 6);
    }

    #[test]
    fn logit_softcap_bounds_output() {
        let cfg = synthetic();
        let mut model = Model::new(cfg.clone()).unwrap();
        let mut caches = make_caches(&cfg, CacheOptions::default());
        let inputs = Array::from_slice(&[1_i32, 2, 3], &[1, 3]);
        let logits = model
            .forward(ModelInput {
                inputs: &inputs,
                mask: None,
                cache: &mut caches,
            })
            .unwrap();
        eval([&logits]).unwrap();
        // final_logit_softcapping = 30.0 ⇒ |logits| < 30.
        let max_mag = logits.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max_mag < 30.0, "softcap did not bound logits: {max_mag}");
    }

    /// `synthetic()` + the MoE gate so every layer runs the dual branch.
    fn synthetic_moe() -> TextConfig {
        let json = serde_json::json!({
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "head_dim": 8,
            "global_head_dim": 8,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 100,
            "sliding_window": 8,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "layer_types": [
                "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention"
            ],
            "enable_moe_block": true,
            "num_experts": 8,
            "top_k_experts": 2,
            "moe_intermediate_size": 16,
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn moe_decoder_forward_shape_round_trips() {
        let cfg = synthetic_moe();
        let vocab = cfg.vocab_size;
        assert!(cfg.enable_moe_block);
        let mut model = Model::new(cfg.clone()).unwrap();
        let mut caches = make_caches(&cfg, CacheOptions::default());

        // Prefill 5 tokens through the dual-branch (dense + experts) FFN.
        let ids: Vec<i32> = (0..5).collect();
        let inputs = Array::from_slice(&ids, &[1, 5]);
        let logits = model
            .forward(ModelInput {
                inputs: &inputs,
                mask: None,
                cache: &mut caches,
            })
            .unwrap();
        eval([&logits]).unwrap();
        assert_eq!(logits.shape(), &[1, 5, vocab]);

        // Decode one more token.
        let next = Array::from_slice(&[7_i32], &[1, 1]);
        let logits2 = model
            .forward(ModelInput {
                inputs: &next,
                mask: None,
                cache: &mut caches,
            })
            .unwrap();
        eval([&logits2]).unwrap();
        assert_eq!(logits2.shape(), &[1, 1, vocab]);
    }

    /// `synthetic()` + per-layer-input + KV-sharing (last 2 of 4 shared).
    fn synthetic_e2b() -> TextConfig {
        let json = serde_json::json!({
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "head_dim": 8,
            "global_head_dim": 8,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 100,
            "sliding_window": 8,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            // full-attn at idx 1 (owned) so shared sliding layers 2,3 have a
            // same-kind source (layer 0).
            "layer_types": [
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention"
            ],
            "hidden_size_per_layer_input": 16,
            "vocab_size_per_layer_input": 100,
            "num_kv_shared_layers": 2,
            "use_double_wide_mlp": true,
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn e2b_decoder_forward_shape_round_trips() {
        let cfg = synthetic_e2b();
        let vocab = cfg.vocab_size;
        // Layers 0-1 own K/V; layers 2-3 are KV-shared (None cache slots).
        assert_eq!(cfg.num_kv_shared_layers, 2);
        let mut model = Model::new(cfg.clone()).unwrap();
        let mut caches = make_caches(&cfg, CacheOptions::default());
        assert!(caches[0].is_some());
        assert!(caches[2].is_none(), "KV-shared layer has no cache slot");

        let ids: Vec<i32> = (0..5).collect();
        let inputs = Array::from_slice(&ids, &[1, 5]);
        let logits = model
            .forward(ModelInput {
                inputs: &inputs,
                mask: None,
                cache: &mut caches,
            })
            .unwrap();
        eval([&logits]).unwrap();
        assert_eq!(logits.shape(), &[1, 5, vocab]);

        let next = Array::from_slice(&[7_i32], &[1, 1]);
        let logits2 = model
            .forward(ModelInput {
                inputs: &next,
                mask: None,
                cache: &mut caches,
            })
            .unwrap();
        eval([&logits2]).unwrap();
        assert_eq!(logits2.shape(), &[1, 1, vocab]);
    }

    #[test]
    fn kv_shared_layers_drop_own_kv() {
        let cfg = synthetic_e2b();
        // first_kv_shared = 4 - 2 = 2.
        let owned = Attention::new(&cfg, 0).unwrap();
        let shared = Attention::new(&cfg, 3).unwrap();
        assert!(owned.has_kv && owned.k_proj.is_some() && owned.k_norm.is_some());
        assert!(!shared.has_kv && shared.k_proj.is_none() && shared.v_norm.is_none());
    }
}
