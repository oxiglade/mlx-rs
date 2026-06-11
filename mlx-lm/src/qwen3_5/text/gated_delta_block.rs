//! [`GatedDeltaNet`] — the `linear_attention` decoder layer.
//!
//! Wraps the projector linears + causal Conv1d + sequential GDN scan
//! + gated RMSNorm + output projection.

// See `mlx_lm::activations` for the rationale on the fn-item-to-fn-pointer
// coercion required at the `Compile::compile_with_id` call site.
#![allow(
    trivial_casts,
    reason = "fn-item ZST → fn-pointer coercion for shared compile cache"
)]

use std::sync::OnceLock;

use crate::error::Error;
use mlx_rs::{
    builder::Builder,
    error::Exception,
    fast::{rms_norm, MetalKernel},
    macros::{ModuleParameters, Quantizable},
    module::{Module, Param},
    nn,
    ops::{
        concatenate_axis, expand_dims, indexing::IndexOp, ones, r#where, reshape, split_sections,
        zeros,
    },
    quantization::MaybeQuantized,
    random,
    transforms::compile::{shape::ThreeArgs, CallMut, Compile, Compiled},
    Array, Dtype,
};

use super::cache::LinearAttnCache;
use super::config::TextConfig;
use super::gated_delta::{
    gated_delta_update_metal, gated_delta_update_ops, make_gated_delta_kernel, ComputeGCache,
};

/// Process-wide compiled instance of the GDN scan kernel — built once
/// so all 24 `GatedDeltaNet` blocks share one handle.
fn gdn_kernel() -> Result<&'static MetalKernel, Error> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(k) = KERNEL.get() {
        return Ok(k);
    }
    // Build outside the closure (fallible); a lost init race just drops the spare.
    let built = make_gated_delta_kernel()?;
    let _ = KERNEL.set(built);
    Ok(KERNEL.get().expect("just set"))
}

/// Gated RMSNorm from `Qwen3_5RMSNormGated`. Returns
/// `silu(gate.astype(f32)) * rms_norm(hidden, weight)` cast back to `hidden.dtype`.
///
/// The trailing silu+multiply+cast triple runs through
/// `transforms::compile` as a shape-compiled closure. `weight_f32` is
/// the norm weight pre-cast to f32 (caller-cached) so this fn doesn't
/// pay the cast launch per call.
fn rms_norm_gated(
    cache: &mut PreciseSwigluCache,
    hidden: &Array,
    gate: &Array,
    weight_f32: &Array,
    eps: f32,
) -> Result<Array, Exception> {
    let normed = rms_norm(hidden.as_dtype(Dtype::Float32)?, Some(weight_f32), eps)?;
    precise_swiglu(cache, hidden, gate, &normed)
}

pub(crate) type PreciseSwigluCompiled = Compiled<
    fn((&Array, &Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
    ThreeArgs,
>;

#[derive(Default)]
pub(crate) struct PreciseSwigluCache(pub(crate) Option<PreciseSwigluCompiled>);

impl std::fmt::Debug for PreciseSwigluCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreciseSwigluCache")
            .field("filled", &self.0.is_some())
            .finish()
    }
}

/// `silu(gate.astype(f32)) * x.astype(f32)` cast back to `h.dtype`.
/// Caller-owned cache, same shape as [`crate::activations::swiglu`].
fn precise_swiglu(
    cache: &mut PreciseSwigluCache,
    h: &Array,
    gate: &Array,
    x: &Array,
) -> Result<Array, Exception> {
    let compiled = cache.0.get_or_insert_with(|| {
        Compile::<(&Array, &Array, &Array), Array, Exception>::compile(
            precise_swiglu_inner as fn((&Array, &Array, &Array)) -> Result<Array, Exception>,
            true,
        )
    });
    CallMut::call_mut(compiled, (h, gate, x))
}

fn precise_swiglu_inner((h, gate, x): (&Array, &Array, &Array)) -> Result<Array, Exception> {
    let gate_silu = nn::silu(gate.as_dtype(Dtype::Float32)?)?;
    let x_f32 = x.as_dtype(Dtype::Float32)?;
    gate_silu.multiply(&x_f32)?.as_dtype(h.dtype())
}

/// The Gated DeltaNet decoder block.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct GatedDeltaNet {
    pub num_v_heads: i32,
    pub num_k_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub key_dim: i32,
    pub value_dim: i32,
    pub conv_dim: i32,
    pub conv_kernel_size: i32,
    pub eps: f32,

    /// Causal 1-D depthwise conv on the projected `(q, k, v)`.
    #[param]
    pub conv1d: nn::Conv1d,

    /// `Linear(hidden, key_dim*2 + value_dim)`.
    #[quantizable]
    #[param]
    pub in_proj_qkv: MaybeQuantized<nn::Linear>,
    /// `Linear(hidden, value_dim)`.
    #[quantizable]
    #[param]
    pub in_proj_z: MaybeQuantized<nn::Linear>,
    /// `Linear(hidden, num_v_heads)`.
    #[quantizable]
    #[param]
    pub in_proj_b: MaybeQuantized<nn::Linear>,
    /// `Linear(hidden, num_v_heads)`.
    #[quantizable]
    #[param]
    pub in_proj_a: MaybeQuantized<nn::Linear>,

    /// Learned `[num_v_heads]` time-step bias inside `compute_g`.
    #[param]
    pub dt_bias: Param<Array>,
    /// Learned `[num_v_heads]` log-A used inside `compute_g`. Stored unscaled
    /// in fp32 (the cast_predicate keeps `A_log` out of the bf16 cast at load).
    #[param]
    pub a_log: Param<Array>,
    /// Learned `[head_v_dim]` RMS norm weight applied inside `Qwen3_5RMSNormGated`.
    #[param]
    pub norm_weight: Param<Array>,

    /// `Linear(value_dim, hidden_size)` — final out-projection.
    #[quantizable]
    #[param]
    pub out_proj: MaybeQuantized<nn::Linear>,

    /// Per-block compiled-graph caches for the GDN scan + gated RMS norm.
    precise_swiglu_cache: PreciseSwigluCache,
    compute_g_cache: ComputeGCache,
    /// Cached 0-D constants used per-call in the rms_norm + scale path.
    /// `Array::from_f32` allocates a fresh GPU array each call.
    q_inv_scale_sq: OnceLock<Array>,
    k_inv_scale: OnceLock<Array>,
    /// `norm_weight` cast to f32 once. `rms_norm_gated` needs the f32
    /// view every call; the original weight is bf16 (or whatever the
    /// loader stored). Caching avoids the per-call cast launch.
    norm_weight_f32: OnceLock<Array>,
}

impl GatedDeltaNet {
    /// Build a freshly-initialised block from a [`TextConfig`].
    pub fn new(cfg: &TextConfig) -> Result<Self, Error> {
        let hidden = cfg.hidden_size;
        let num_v_heads = cfg.linear_num_value_heads;
        let num_k_heads = cfg.linear_num_key_heads;
        if num_v_heads % num_k_heads != 0 {
            return Err(Error::config(format!(
                "GatedDeltaNet: linear_num_value_heads ({num_v_heads}) must be divisible by linear_num_key_heads ({num_k_heads})"
            )));
        }
        let head_k_dim = cfg.linear_key_head_dim;
        let head_v_dim = cfg.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let conv_kernel_size = cfg.linear_conv_kernel_dim;
        let eps = cfg.rms_norm_eps;

        // Depthwise conv: every group is a single channel, so the kernel
        // weight shape is `(conv_dim, kernel_size, 1)` — we build the layer
        // by hand because the upstream `Conv1dBuilder` always initialises
        // weights with `(out, k, in_channels)` and ignores `groups`.
        let conv1d = depthwise_conv1d(conv_dim, conv_kernel_size)?;

        let in_proj_qkv = nn::LinearBuilder::new(hidden, conv_dim)
            .bias(false)
            .build()?;
        let in_proj_z = nn::LinearBuilder::new(hidden, value_dim)
            .bias(false)
            .build()?;
        let in_proj_b = nn::LinearBuilder::new(hidden, num_v_heads)
            .bias(false)
            .build()?;
        let in_proj_a = nn::LinearBuilder::new(hidden, num_v_heads)
            .bias(false)
            .build()?;
        let out_proj = nn::LinearBuilder::new(value_dim, hidden)
            .bias(false)
            .build()?;

        let dt_bias = ones::<f32>(&[num_v_heads])?;
        let a_log_a = random::uniform::<_, f32>(0.0, 16.0, &[num_v_heads], None)?;
        let a_log = a_log_a.log()?;
        let norm_weight = ones::<f32>(&[head_v_dim])?;

        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size,
            eps,
            conv1d,
            in_proj_qkv: MaybeQuantized::Original(in_proj_qkv),
            in_proj_z: MaybeQuantized::Original(in_proj_z),
            in_proj_b: MaybeQuantized::Original(in_proj_b),
            in_proj_a: MaybeQuantized::Original(in_proj_a),
            dt_bias: Param::new(dt_bias),
            a_log: Param::new(a_log),
            norm_weight: Param::new(norm_weight),
            out_proj: MaybeQuantized::Original(out_proj),
            precise_swiglu_cache: PreciseSwigluCache::default(),
            compute_g_cache: ComputeGCache::default(),
            q_inv_scale_sq: OnceLock::new(),
            k_inv_scale: OnceLock::new(),
            norm_weight_f32: OnceLock::new(),
        })
    }

    /// Block forward.
    ///
    /// - `inputs`: `[B, S, hidden_size]`.
    /// - `mask`: optional `[B, S]` bool — `false` positions zero the conv input
    ///   and freeze the SSM state.
    /// - `cache`: per-layer cache slot. Mutated in place.
    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut LinearAttnCache>,
    ) -> Result<Array, Error> {
        let shape = inputs.shape();
        let b = shape[0];
        let s = shape[1];

        let mixed_qkv = self.in_proj_qkv.forward(inputs)?;
        let z = self.in_proj_z.forward(inputs)?;
        let z = reshape(&z, &[b, s, self.num_v_heads, self.head_v_dim])?;

        let b_arr = self.in_proj_b.forward(inputs)?;
        let a_arr = self.in_proj_a.forward(inputs)?;

        let mixed_qkv = match mask {
            Some(m) => {
                // mask: [B, S] -> [B, S, 1] to broadcast across conv_dim.
                let m_b = expand_dims(m, 2)?;
                let zero = Array::from_f32(0.0).as_dtype(mixed_qkv.dtype())?;
                r#where(&m_b, &mixed_qkv, &zero)?
            }
            None => mixed_qkv,
        };

        // Concatenate the cached conv history with the new tokens. The history
        // is the last `conv_kernel_size - 1` tokens of the prior conv_input.
        // `conv_state_owned` holds the zero-fill when no cache exists; the
        // borrow goes through `conv_state_ref` so cached histories pass to
        // `concatenate_axis` by reference (no per-step FFI refcount bump).
        let history_len = self.conv_kernel_size - 1;
        let conv_state_owned;
        let conv_state_ref: &Array = match cache.as_ref().and_then(|c| c.conv_state.as_ref()) {
            Some(cs) if cs.shape() == [b, history_len, self.conv_dim] => cs,
            _ => {
                conv_state_owned =
                    zeros::<f32>(&[b, history_len, self.conv_dim])?.as_dtype(mixed_qkv.dtype())?;
                &conv_state_owned
            }
        };
        let conv_input = concatenate_axis(&[conv_state_ref, &mixed_qkv], 1)?;

        let conv_out = self.conv1d.forward(&conv_input)?;
        let conv_out = nn::silu(conv_out)?;

        // Persist the new conv history tail. Range-index on axis 1
        // avoids the per-step `Vec<i32> + Array::from_slice` round-trip
        // that `take_axis` requires.
        if let Some(cache) = cache.as_deref_mut() {
            let total_len = history_len + s;
            let start = total_len - history_len;
            let new_history = conv_input.index((.., start..total_len, ..));
            cache.conv_state = Some(new_history);
        }

        // Split conv_out into q / k / v along the channel axis.
        let parts = split_sections(&conv_out, &[self.key_dim, 2 * self.key_dim], -1)?;
        let q = reshape(&parts[0], &[b, s, self.num_k_heads, self.head_k_dim])?;
        let k = reshape(&parts[1], &[b, s, self.num_k_heads, self.head_k_dim])?;
        let v = reshape(&parts[2], &[b, s, self.num_v_heads, self.head_v_dim])?;

        // Scale q/k with the head-dim power. The scale arrays are stored
        // in the input dtype so the multiply doesn't promote → no
        // trailing `as_dtype` cast launch.
        let head_k_dim = self.head_k_dim;
        let q_dtype = q.dtype();
        let q_scale = self.q_inv_scale_sq.get_or_init(|| {
            let inv = (head_k_dim as f32).powf(-0.5);
            Array::from_f32(inv * inv)
                .as_dtype(q_dtype)
                .expect("scale cast cannot fail")
        });
        let k_scale = self.k_inv_scale.get_or_init(|| {
            let inv = (head_k_dim as f32).powf(-0.5);
            Array::from_f32(inv)
                .as_dtype(q_dtype)
                .expect("scale cast cannot fail")
        });
        let q_normed = rms_norm(&q, None, 1e-6)?.multiply(q_scale)?;
        let k_normed = rms_norm(&k, None, 1e-6)?.multiply(k_scale)?;

        // Recurrent state carried across decode steps. `None` on the first
        // call (prefill) → the update path seeds zeros. Borrowed straight
        // from the cache slot — both update fns take `Option<&Array>`, so no
        // per-token clone of the per-layer state.
        let state_in = cache.as_ref().and_then(|c| c.recurrent_state.as_ref());
        // Kernel requires Dk to be a multiple of the SIMD width (32). The
        // ops path covers every other case including a non-None mask.
        let use_kernel = mask.is_none() && self.head_k_dim % 32 == 0;
        let (out, new_state) = if use_kernel {
            gated_delta_update_metal(
                gdn_kernel()?,
                &mut self.compute_g_cache,
                &q_normed,
                &k_normed,
                &v,
                &a_arr,
                &b_arr,
                &self.a_log.value,
                &self.dt_bias.value,
                state_in,
            )?
        } else {
            gated_delta_update_ops(
                &mut self.compute_g_cache,
                &q_normed,
                &k_normed,
                &v,
                &a_arr,
                &b_arr,
                &self.a_log.value,
                &self.dt_bias.value,
                state_in,
                mask,
            )?
        };

        if let Some(cache) = cache {
            cache.recurrent_state = Some(new_state);
            cache.offset += s;
        }

        // Apply the gated RMS norm with the per-head `z`. norm_weight
        // is loaded as bf16 (or model dtype); cast once.
        let weight_f32 = self.norm_weight_f32.get_or_init(|| {
            self.norm_weight
                .value
                .as_dtype(Dtype::Float32)
                .expect("norm_weight cast to f32 cannot fail")
        });
        let gated = rms_norm_gated(
            &mut self.precise_swiglu_cache,
            &out,
            &z,
            weight_f32,
            self.eps,
        )?;
        let flat = reshape(&gated, &[b, s, self.num_v_heads * self.head_v_dim])?;
        Ok(self.out_proj.forward(&flat)?)
    }

    /// Toggle training mode on every quantisable parameter.
    pub fn training_mode(&mut self, mode: bool) {
        self.in_proj_qkv.training_mode(mode);
        self.in_proj_z.training_mode(mode);
        self.in_proj_b.training_mode(mode);
        self.in_proj_a.training_mode(mode);
        self.out_proj.training_mode(mode);
    }
}

/// Build a depthwise Conv1d with weight shape `[conv_dim, kernel_size, 1]`
/// (Qwen3.5's depthwise convention).
fn depthwise_conv1d(conv_dim: i32, kernel_size: i32) -> Result<nn::Conv1d, Exception> {
    let scale = (1.0_f32 / (kernel_size as f32)).sqrt();
    let weight = random::uniform::<_, f32>(-scale, scale, &[conv_dim, kernel_size, 1], None)?;
    Ok(nn::Conv1d {
        weight: Param::new(weight),
        bias: Param::new(None),
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: conv_dim,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    #![allow(clippy::print_stdout, reason = "test code")]
    #![allow(clippy::print_stderr, reason = "test code")]
    use super::*;
    use mlx_rs::{random::uniform, transforms::eval};

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
            "layer_types": ["linear_attention"],
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
    fn block_shape_round_trips_without_cache() {
        let cfg = synthetic_text_config();
        let mut blk = GatedDeltaNet::new(&cfg).unwrap();
        let x = uniform::<_, f32>(0.0, 1.0, &[1, 5, cfg.hidden_size], None).unwrap();
        let y = blk.forward(&x, None, None).unwrap();
        eval([&y]).unwrap();
        assert_eq!(y.shape(), &[1, 5, cfg.hidden_size]);
    }

    /// `linear_key_head_dim = 32` so the kernel `dk_ok` gate is satisfied;
    /// the rest matches the synthetic text-only config used elsewhere.
    fn kernel_eligible_config() -> TextConfig {
        let json = serde_json::json!({
            "model_type": "qwen3_5_text",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": 100,
            "max_position_embeddings": 256,
            "layer_types": ["linear_attention"],
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {
                "mrope_section": [4, 2, 2],
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "type": "default"
            }
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn chained_kernel_blocks_match_ops_24_layers() {
        let cfg = kernel_eligible_config();
        let layers: usize = 24;

        let mut blocks: Vec<GatedDeltaNet> = (0..layers)
            .map(|_| GatedDeltaNet::new(&cfg).unwrap())
            .collect();
        let x = uniform::<_, f32>(0.0, 1.0, &[1, 4, cfg.hidden_size], None).unwrap();
        let mut out = x;
        for blk in &mut blocks {
            out = blk.forward(&out, None, None).unwrap();
        }
        eval([&out]).unwrap();
        assert_eq!(out.shape(), &[1, 4, cfg.hidden_size]);
    }

    #[test]
    fn block_prefill_then_decode_extends_cache_offset() {
        let cfg = synthetic_text_config();
        let mut blk = GatedDeltaNet::new(&cfg).unwrap();
        let prompt = uniform::<_, f32>(0.0, 1.0, &[1, 5, cfg.hidden_size], None).unwrap();
        let mut cache = LinearAttnCache::new();
        let prefill_out = blk.forward(&prompt, None, Some(&mut cache)).unwrap();
        eval([&prefill_out]).unwrap();
        assert_eq!(prefill_out.shape(), &[1, 5, cfg.hidden_size]);
        assert_eq!(cache.offset, 5);
        assert!(cache.conv_state.is_some());
        assert!(cache.recurrent_state.is_some());

        let next = uniform::<_, f32>(0.0, 1.0, &[1, 1, cfg.hidden_size], None).unwrap();
        let decode_out = blk.forward(&next, None, Some(&mut cache)).unwrap();
        eval([&decode_out]).unwrap();
        assert_eq!(decode_out.shape(), &[1, 1, cfg.hidden_size]);
        assert_eq!(cache.offset, 6);

        // Conv history must always have length conv_kernel_size - 1.
        let cs = cache.conv_state.as_ref().unwrap();
        assert_eq!(
            cs.shape(),
            &[1, cfg.linear_conv_kernel_dim - 1, blk.conv_dim]
        );
    }
}
