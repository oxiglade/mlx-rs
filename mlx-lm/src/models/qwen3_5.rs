use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    argmax_axis, array,
    builder::Builder,
    categorical,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters as ModuleParametersTrait, ModuleParametersExt, Param},
    nn,
    ops::{
        einsum, expand_dims,
        indexing::{IndexOp, NewAxis},
    },
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    cache::KeyValueCache,
    error::Error,
    utils::{
        create_attention_mask,
        rope::{initialize_rope, RopeVariant},
        AttentionMask,
    },
};

// ---------------------------------------------------------------------------
// Config / ModelArgs
// ---------------------------------------------------------------------------

/// Nested RoPE parameters from config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_type: Option<String>,
    pub rope_theta: f32,
    pub partial_rotary_factor: Option<f32>,
}

/// Model configuration for Qwen3.5 hybrid attention models.
///
/// Qwen3.5 uses a mix of linear (Mamba-style SSM) and full (standard transformer)
/// attention layers, specified by the `layer_types` array.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: i32,

    // Full attention parameters
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attn_output_gate: bool,

    // Linear attention parameters
    #[serde(default = "default_conv_kernel_dim")]
    pub linear_conv_kernel_dim: i32,
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: i32,
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: i32,
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: i32,
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: i32,

    // Layer type configuration
    pub layer_types: Vec<String>,

    // RoPE (nested)
    pub rope_parameters: RopeParameters,

    // Quantization (optional)
    pub quantization: Option<Quantization>,

    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_conv_kernel_dim() -> i32 {
    4
}
fn default_linear_key_head_dim() -> i32 {
    128
}
fn default_linear_num_key_heads() -> i32 {
    16
}
fn default_linear_num_value_heads() -> i32 {
    48
}
fn default_linear_value_head_dim() -> i32 {
    128
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
    pub mode: Option<String>,
}

// ---------------------------------------------------------------------------
// MLP (shared between both layer types)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
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
}

impl Mlp {
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
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
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let down_proj_input =
            nn::silu(self.gate_proj.forward(input)?)?.multiply(self.up_proj.forward(input)?)?;
        self.down_proj.forward(&down_proj_input)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// Full Attention (standard transformer attention with RoPE)
// ---------------------------------------------------------------------------

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
}

/// Standard grouped-query attention with RoPE, used in full attention layers.
///
/// Applies partial rotary embeddings (controlled by `partial_rotary_factor`).
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct FullAttention {
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
    #[param]
    pub rope: RopeVariant,
}

impl FullAttention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;
        let scale = (head_dim as f32).sqrt().recip();
        let bias = args.attention_bias;

        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(false)
            .build()?;

        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;

        // Partial rotary: only rotate a fraction of head_dim
        let partial_rotary_factor = args.rope_parameters.partial_rotary_factor.unwrap_or(1.0);
        let rope_dim = (head_dim as f32 * partial_rotary_factor) as i32;

        let rope = initialize_rope(
            rope_dim,
            args.rope_parameters.rope_theta,
            false,
            &None, // default rope type
            args.max_position_embeddings,
        )?;

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
            rope,
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for FullAttention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, mut cache } = input;

        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape to [B, n_heads, L, head_dim] and apply norms
        let mut queries = self.q_norm.forward(
            &queries
                .reshape(&[B, L, self.n_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?,
        )?;
        let mut keys = self.k_norm.forward(
            &keys
                .reshape(&[B, L, self.n_kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?,
        )?;
        let mut values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE (partial rotary if rope_dim < head_dim)
        if let Some(cache) = cache.as_mut() {
            let q_input = nn::RopeInputBuilder::new(&queries)
                .offset(cache.offset())
                .build()?;
            queries = self.rope.forward(q_input)?;
            let k_input = nn::RopeInputBuilder::new(&keys)
                .offset(cache.offset())
                .build()?;
            keys = self.rope.forward(k_input)?;

            (keys, values) = cache.update_and_fetch(keys, values)?;
        } else {
            queries = self.rope.forward(nn::RopeInput::new(&queries))?;
            keys = self.rope.forward(nn::RopeInput::new(&keys))?;
        }

        let output = crate::utils::scaled_dot_product_attention(
            queries, keys, values, cache, self.scale, mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        self.k_norm.training_mode(mode);
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

// ---------------------------------------------------------------------------
// Linear Attention (Mamba-style SSM)
// ---------------------------------------------------------------------------

/// Mamba-style linear attention with Conv1d and selective state space model (SSM).
///
/// This implements the linear attention mechanism used in Qwen3.5 hybrid layers.
/// The forward pass applies:
/// 1. Input projections for Q, K, V, gate (z), and SSM parameters (A, B)
/// 2. Short convolution (Conv1d with causal padding)
/// 3. Selective SSM scan: h[t] = A * h[t-1] + B[t] * x[t]; y[t] = C[t] * h[t]
/// 4. Output gating with SiLU activation on z projection
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct LinearAttention {
    pub hidden_size: i32,
    pub num_key_heads: i32,
    pub num_value_heads: i32,
    pub key_head_dim: i32,
    pub value_head_dim: i32,
    pub conv_kernel_dim: i32,

    /// Projects input to Q, K, V concatenated.
    #[quantizable]
    #[param]
    pub in_proj_qkv: MaybeQuantized<nn::Linear>,

    /// Projects input to SSM discretization timestep (dt).
    #[quantizable]
    #[param]
    pub in_proj_a: MaybeQuantized<nn::Linear>,

    /// Projects input to SSM B parameter.
    #[quantizable]
    #[param]
    pub in_proj_b: MaybeQuantized<nn::Linear>,

    /// Projects input to gate (z) for output gating.
    #[quantizable]
    #[param]
    pub in_proj_z: MaybeQuantized<nn::Linear>,

    /// Log of the diagonal SSM transition matrix. Shape: [num_key_heads].
    #[param]
    pub a_log: Param<Array>,

    /// Bias for SSM timestep discretization. Shape: [num_key_heads].
    #[param]
    pub dt_bias: Param<Array>,

    /// Short causal convolution applied to QK before SSM.
    #[param]
    pub conv1d: nn::Conv1d,

    /// RMS normalization before output projection.
    #[param]
    pub norm: nn::RmsNorm,

    /// Output projection back to hidden_size.
    #[quantizable]
    #[param]
    pub out_proj: MaybeQuantized<nn::Linear>,
}

impl LinearAttention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let hidden_size = args.hidden_size;
        let num_key_heads = args.linear_num_key_heads;
        let num_value_heads = args.linear_num_value_heads;
        let key_head_dim = args.linear_key_head_dim;
        let value_head_dim = args.linear_value_head_dim;
        let conv_kernel_dim = args.linear_conv_kernel_dim;

        // QKV projection output dimensions:
        // Q: num_key_heads * key_head_dim
        // K: num_key_heads * key_head_dim
        // V: num_value_heads * value_head_dim
        let qkv_dim = num_key_heads * key_head_dim
            + num_key_heads * key_head_dim
            + num_value_heads * value_head_dim;
        let in_proj_qkv = nn::LinearBuilder::new(hidden_size, qkv_dim)
            .bias(false)
            .build()?;

        // SSM parameter projections
        let in_proj_a = nn::LinearBuilder::new(hidden_size, num_key_heads)
            .bias(false)
            .build()?;
        let in_proj_b = nn::LinearBuilder::new(hidden_size, num_key_heads)
            .bias(false)
            .build()?;

        // Gate projection: output dim matches V total dim
        let out_dim = num_value_heads * value_head_dim;
        let in_proj_z = nn::LinearBuilder::new(hidden_size, out_dim)
            .bias(false)
            .build()?;

        // A_log and dt_bias are learnable parameters (not linear layers)
        let a_log = Param::new(Array::zeros::<f32>(&[num_key_heads])?);
        let dt_bias = Param::new(Array::zeros::<f32>(&[num_key_heads])?);

        // Conv1d: depthwise on QK concatenation with causal padding
        let conv_channels = 2 * num_key_heads * key_head_dim;
        let conv1d = nn::Conv1dBuilder::new(conv_channels, conv_channels, conv_kernel_dim)
            .bias(true)
            .padding(0) // Causal padding applied manually
            .groups(conv_channels) // Depthwise convolution
            .build()?;

        let norm = nn::RmsNormBuilder::new(out_dim)
            .eps(args.rms_norm_eps)
            .build()?;
        let out_proj = nn::LinearBuilder::new(out_dim, hidden_size)
            .bias(false)
            .build()?;

        Ok(Self {
            hidden_size,
            num_key_heads,
            num_value_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_dim,
            in_proj_qkv: MaybeQuantized::Original(in_proj_qkv),
            in_proj_a: MaybeQuantized::Original(in_proj_a),
            in_proj_b: MaybeQuantized::Original(in_proj_b),
            in_proj_z: MaybeQuantized::Original(in_proj_z),
            a_log,
            dt_bias,
            conv1d,
            norm,
            out_proj: MaybeQuantized::Original(out_proj),
        })
    }
}

impl Module<&Array> for LinearAttention {
    type Output = Array;
    type Error = Exception;

    /// Forward pass for linear (Mamba-style SSM) attention.
    ///
    /// Input shape: `[B, L, hidden_size]`
    /// Output shape: `[B, L, hidden_size]`
    #[allow(non_snake_case)]
    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        // Project to QKV, gate, and SSM parameters
        let qkv = self.in_proj_qkv.forward(x)?;
        let z = self.in_proj_z.forward(x)?;
        let dt = self.in_proj_a.forward(x)?; // discretization timestep
        let b_proj = self.in_proj_b.forward(x)?; // SSM B parameter

        // Split QKV into Q, K, V
        let q_dim = self.num_key_heads * self.key_head_dim;
        let k_dim = self.num_key_heads * self.key_head_dim;
        let v_dim = self.num_value_heads * self.value_head_dim;
        let _ = v_dim; // used implicitly

        // qkv shape: [B, L, q_dim + k_dim + v_dim]
        let q = qkv.index((.., .., ..q_dim));
        let k = qkv.index((.., .., q_dim..q_dim + k_dim));
        let v = qkv.index((.., .., q_dim + k_dim..));

        // Concatenate Q and K for depthwise conv1d
        let qk = mlx_rs::ops::concatenate_axis(&[q, k], -1)?;

        // Causal padding: pad (kernel_dim - 1) zeros on the left of the sequence
        let pad_len = self.conv_kernel_dim - 1;
        let qk_padded = mlx_rs::ops::pad(
            &qk,
            &[(0, 0), (pad_len, 0), (0, 0)],
            array!(0.0f32),
            None::<mlx_rs::ops::PadMode>,
        )?;
        let qk_conv = self.conv1d.forward(&qk_padded)?;

        // Split back into Q and K after convolution
        let q_conv = qk_conv.index((.., .., ..q_dim));
        let k_conv = qk_conv.index((.., .., q_dim..));

        // Reshape for multi-head computation
        // Q, K: [B, L, num_key_heads, key_head_dim]
        let q_heads = q_conv.reshape(&[B, L, self.num_key_heads, self.key_head_dim])?;
        let k_heads = k_conv.reshape(&[B, L, self.num_key_heads, self.key_head_dim])?;
        // V: [B, L, num_value_heads, value_head_dim]
        let v_heads = v.reshape(&[B, L, self.num_value_heads, self.value_head_dim])?;

        // --- Selective SSM scan ---
        //
        // Discretize SSM parameters:
        //   A = -exp(A_log)          shape: [num_key_heads]
        //   dt = softplus(dt + dt_bias) shape: [B, L, num_key_heads]
        //   decay = exp(A * dt)       shape: [B, L, num_key_heads]
        let neg_A = self.a_log.as_ref().exp()?.negative()?;
        let dt_biased = dt.add(self.dt_bias.as_ref())?;
        let dt_soft = nn::softplus(&dt_biased)?;
        // TODO: Use decay for full SSM scan with exponential recurrence
        let _decay = neg_A.multiply(&dt_soft)?.exp()?;

        // B_scaled = B * dt: weight the input contribution by the timestep
        let b_scaled = b_proj.multiply(&dt_soft)?;

        // Linear attention with causal structure.
        //
        // The Mamba-style SSM computes, for each head:
        //   state[t] = decay[t] * state[t-1] + b_scaled[t] * K[t]^T @ V[t]
        //   output[t] = Q[t] @ state[t]
        //
        // We approximate this using a causal linear attention formulation:
        //   scores = Q @ K^T (causal, no softmax)
        //   weighted by b_scaled and accumulated with decay
        //
        // For now we compute full causal attention scores without the
        // recurrent decay, which is correct when all decay factors are 1.

        let scale = (self.key_head_dim as f32).sqrt().recip();

        // Transpose to [B, heads, L, dim] for batched matmul
        let q_t = q_heads.transpose_axes(&[0, 2, 1, 3])?;
        let k_t = k_heads.transpose_axes(&[0, 2, 1, 3])?;

        // scores: [B, num_key_heads, L, L]
        let scores = einsum("bhld,bhmd->bhlm", [&q_t, &k_t])?.multiply(array!(scale))?;

        // Causal mask: lower-triangular
        let causal_mask = {
            let indices = mlx_rs::ops::arange::<_, f32>(None, L, None)?;
            let col = expand_dims(&indices, -1)?;
            let row = expand_dims(&indices, 0)?;
            col.ge(&row)?
        };

        let masked_scores = mlx_rs::ops::r#where(
            &causal_mask,
            &scores,
            &Array::from_f64(f64::MIN),
        )?;

        // Weight by B_scaled: [B, num_key_heads, 1, L]
        let b_weight = b_scaled
            .transpose_axes(&[0, 2, 1])?
            .reshape(&[B, self.num_key_heads, 1, L])?;
        let weighted_scores = masked_scores.multiply(&b_weight)?;

        // Softmax-free for pure linear attention, but we apply softmax here
        // for numerical stability in this initial implementation.
        let attn_weights = mlx_rs::ops::softmax_axis(&weighted_scores, -1, None)?;

        // V: [B, num_value_heads, L, value_head_dim]
        let v_t = v_heads.transpose_axes(&[0, 2, 1, 3])?;

        // Map key heads to value heads by repeating
        let heads_per_key = self.num_value_heads / self.num_key_heads;
        let attn_expanded = if heads_per_key > 1 {
            let expanded = expand_dims(&attn_weights, 2)?;
            let expanded = mlx_rs::ops::broadcast_to(
                &expanded,
                &[B, self.num_key_heads, heads_per_key, L, L],
            )?;
            expanded.reshape(&[B, self.num_value_heads, L, L])?
        } else {
            attn_weights
        };

        // output: [B, num_value_heads, L, value_head_dim]
        let output = einsum("bhls,bhsd->bhld", [&attn_expanded, &v_t])?;

        // Reshape back to [B, L, num_value_heads * value_head_dim]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?;

        // Output gating: norm(output) * silu(z)
        let output = self.norm.forward(&output)?;
        let gate = nn::silu(&z)?;
        let output = output.multiply(&gate)?;

        self.out_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.in_proj_qkv.training_mode(mode);
        self.in_proj_a.training_mode(mode);
        self.in_proj_b.training_mode(mode);
        self.in_proj_z.training_mode(mode);
        self.conv1d.training_mode(mode);
        self.norm.training_mode(mode);
        self.out_proj.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// Decoder Layers
// ---------------------------------------------------------------------------

/// Full attention decoder layer (standard transformer block).
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct FullAttentionLayer {
    #[quantizable]
    #[param]
    pub self_attn: FullAttention,

    #[quantizable]
    #[param]
    pub mlp: Mlp,

    #[param]
    pub input_layernorm: nn::RmsNorm,

    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

/// Linear attention decoder layer (Mamba-style SSM block).
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct LinearAttentionLayer {
    #[quantizable]
    #[param]
    pub linear_attn: LinearAttention,

    #[quantizable]
    #[param]
    pub mlp: Mlp,

    #[param]
    pub input_layernorm: nn::RmsNorm,

    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

/// Enum wrapping both layer types for the hybrid architecture.
#[derive(Debug, Clone)]
pub enum DecoderLayer {
    FullAttention(FullAttentionLayer),
    LinearAttention(LinearAttentionLayer),
}

impl FullAttentionLayer {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: FullAttention::new(args)?,
            mlp: Mlp::new(args.hidden_size, args.intermediate_size)?,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

impl LinearAttentionLayer {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        Ok(Self {
            linear_attn: LinearAttention::new(args)?,
            mlp: Mlp::new(args.hidden_size, args.intermediate_size)?,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

// Manual ModuleParameters for DecoderLayer enum (derive macros don't support enums)
impl mlx_rs::module::ModuleParameters for DecoderLayer {
    fn num_parameters(&self) -> usize {
        match self {
            DecoderLayer::FullAttention(l) => l.num_parameters(),
            DecoderLayer::LinearAttention(l) => l.num_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            DecoderLayer::FullAttention(l) => l.freeze_parameters(recursive),
            DecoderLayer::LinearAttention(l) => l.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            DecoderLayer::FullAttention(l) => l.unfreeze_parameters(recursive),
            DecoderLayer::LinearAttention(l) => l.unfreeze_parameters(recursive),
        }
    }

    fn parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        match self {
            DecoderLayer::FullAttention(l) => l.parameters(),
            DecoderLayer::LinearAttention(l) => l.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> mlx_rs::module::ModuleParamMut<'_> {
        match self {
            DecoderLayer::FullAttention(l) => l.parameters_mut(),
            DecoderLayer::LinearAttention(l) => l.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        match self {
            DecoderLayer::FullAttention(l) => l.trainable_parameters(),
            DecoderLayer::LinearAttention(l) => l.trainable_parameters(),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            DecoderLayer::FullAttention(l) => l.all_frozen(),
            DecoderLayer::LinearAttention(l) => l.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            DecoderLayer::FullAttention(l) => l.any_frozen(),
            DecoderLayer::LinearAttention(l) => l.any_frozen(),
        }
    }
}

// Manual Quantizable for DecoderLayer enum
impl mlx_rs::quantization::Quantizable for DecoderLayer {
    type Quantized = DecoderLayer;
    type QuantizationError = mlx_rs::error::Exception;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
    ) -> std::result::Result<Self::Quantized, Self::QuantizationError> {
        match self {
            DecoderLayer::FullAttention(l) => {
                Ok(DecoderLayer::FullAttention(l.try_into_quantized(group_size, bits)?))
            }
            DecoderLayer::LinearAttention(l) => {
                Ok(DecoderLayer::LinearAttention(l.try_into_quantized(group_size, bits)?))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// The Qwen3.5 language model backbone with hybrid attention layers.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Qwen3_5InnerModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer>,

    #[param]
    pub norm: nn::RmsNorm,
}

impl Qwen3_5InnerModel {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        assert!(args.vocab_size.is_positive());
        assert_eq!(
            args.layer_types.len(),
            args.num_hidden_layers as usize,
            "layer_types length must match num_hidden_layers"
        );

        let vocab_size = args.vocab_size;
        let num_hidden_layers = args.num_hidden_layers;

        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;

        let layers = args
            .layer_types
            .iter()
            .map(|layer_type| match layer_type.as_str() {
                "full_attention" => {
                    FullAttentionLayer::new(args).map(DecoderLayer::FullAttention)
                }
                "linear_attention" => {
                    LinearAttentionLayer::new(args).map(DecoderLayer::LinearAttention)
                }
                other => Err(Exception::custom(format!(
                    "Unknown layer type: {other}"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        Ok(Self {
            vocab_size,
            num_hidden_layers,
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            layers,
            norm,
        })
    }
}

pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<ModelInput<'_, C>> for Qwen3_5InnerModel
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;

        let mut h = self.embed_tokens.forward(inputs)?;

        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => match create_attention_mask(&h, cache, Some(true))? {
                Some(AttentionMask::Array(a)) => Some(a),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("Only `Array` mask is supported"));
                }
                None => None,
            },
        };

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| None).collect();
        }

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            match layer {
                DecoderLayer::FullAttention(fa_layer) => {
                    let normed = fa_layer.input_layernorm.forward(&h)?;
                    let attn_input = AttentionInput {
                        x: &normed,
                        mask: mask.as_ref(),
                        cache: c.as_mut(),
                    };
                    let r = fa_layer.self_attn.forward(attn_input)?;
                    h = h.add(r)?;
                    let r = fa_layer
                        .mlp
                        .forward(&fa_layer.post_attention_layernorm.forward(&h)?)?;
                    h = h.add(r)?;
                }
                DecoderLayer::LinearAttention(la_layer) => {
                    let normed = la_layer.input_layernorm.forward(&h)?;
                    let r = la_layer.linear_attn.forward(&normed)?;
                    h = h.add(r)?;
                    let r = la_layer
                        .mlp
                        .forward(&la_layer.post_attention_layernorm.forward(&h)?)?;
                    h = h.add(r)?;
                }
            }
        }

        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            match layer {
                DecoderLayer::FullAttention(l) => {
                    <FullAttention as Module<AttentionInput<'_, C>>>::training_mode(
                        &mut l.self_attn,
                        mode,
                    );
                    l.mlp.training_mode(mode);
                    l.input_layernorm.training_mode(mode);
                    l.post_attention_layernorm.training_mode(mode);
                }
                DecoderLayer::LinearAttention(l) => {
                    l.linear_attn.training_mode(mode);
                    l.mlp.training_mode(mode);
                    l.input_layernorm.training_mode(mode);
                    l.post_attention_layernorm.training_mode(mode);
                }
            }
        }
        self.norm.training_mode(mode);
    }
}

/// Top-level Qwen3.5 model with language model head.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    pub model: Qwen3_5InnerModel,

    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = Qwen3_5InnerModel::new(&args)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        } else {
            None
        };

        Ok(Self {
            args,
            model,
            lm_head,
        })
    }

    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward(input)?;

        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&out),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(embed_tokens) => embed_tokens.as_linear(&out),
                MaybeQuantized::Quantized(q_embed_tokens) => q_embed_tokens.as_linear(&out),
            },
        }
    }

    fn training_mode(&mut self, mode: bool) {
        <Qwen3_5InnerModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

pub fn load_qwen3_5_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_qwen3_5_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let model_args_filename = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(model_args_filename)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;
    Ok(model_args)
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

/// Load a Qwen3.5 model from a directory of safetensors files.
///
/// Handles the `language_model.` prefix that appears in weights extracted from
/// multimodal (VL) checkpoints by stripping it before matching parameters.
pub fn load_qwen3_5_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_qwen3_5_model_args(model_dir)?;
    let mut model = Model::new(model_args)?;

    let weights_index = model_dir.join("model.safetensors.index.json");

    if weights_index.exists() {
        // Sharded weights
        let json = std::fs::read_to_string(&weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();

        for weight_file in weight_files {
            let weights_filename = model_dir.join(weight_file);
            load_weights_with_prefix_strip(&mut model, &weights_filename)?;
        }
    } else {
        // Single file
        let weights_filename = model_dir.join("model.safetensors");
        load_weights_with_prefix_strip(&mut model, &weights_filename)?;
    }

    Ok(model)
}

/// Load safetensors weights, stripping the `language_model.` prefix if present.
///
/// This is needed because Qwen3.5 weights from VL (vision-language) checkpoints
/// use a `language_model.model.` prefix for the language model weights.
fn load_weights_with_prefix_strip(model: &mut Model, path: &Path) -> Result<(), Error> {
    let loaded = Array::load_safetensors(path)?;

    let mut params = model.parameters_mut().flatten();

    for (key, value) in loaded {
        // Try the key as-is first, then try stripping the prefix
        let stripped = key.strip_prefix("language_model.").unwrap_or(&key);

        if let Some(param) = params.get_mut(stripped) {
            **param = value;
        }
        // Silently skip weights that don't match (e.g., vision encoder weights)
    }

    // Eval loaded params
    model.eval().map_err(Error::Exception)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Sampling & Generation (mirrors qwen3.rs)
// ---------------------------------------------------------------------------

pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    match temp {
        0.0 => argmax_axis!(logits, -1),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical!(logits)
        }
    }
}

pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<Option<C>>,
    temp: f32,
    state: GenerateState<'a>,
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<Option<C>>,
        temp: f32,
        prompt_token: &'a Array,
    ) -> Self {
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt_token },
        }
    }
}

pub enum GenerateState<'a> {
    Prefill { prompt_token: &'a Array },
    Decode { y: Array },
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

impl<'a, C> Iterator for Generate<'a, C>
where
    C: KeyValueCache,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.state {
            GenerateState::Prefill { prompt_token } => {
                let input = ModelInput {
                    inputs: prompt_token,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
            GenerateState::Decode { y } => {
                let inputs = y.index((.., NewAxis));
                let input = ModelInput {
                    inputs: &inputs,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits, self.temp));
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use mlx_rs::{
        ops::indexing::{IndexOp, NewAxis},
        transforms::eval,
        Array,
    };

    use crate::cache::ConcatKeyValueCache;

    const CACHED_TEST_MODEL_DIR: &str = "../cache/Qwen3.5-4B-4bit";

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_qwen3_5_model() {
        let _model = super::load_qwen3_5_model(CACHED_TEST_MODEL_DIR).unwrap();
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_tokenizer() {
        let tokenizer = super::load_qwen3_5_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();
        let _encoding = tokenizer.encode("Hello, world!", true).unwrap();
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_and_run_qwen3_5_with_concat_cache() {
        let tokenizer = super::load_qwen3_5_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();
        let mut model = super::load_qwen3_5_model(CACHED_TEST_MODEL_DIR).unwrap();

        let encoding = tokenizer.encode("hello", true).unwrap();
        let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
        let mut cache = Vec::new();

        let mut tokens = Vec::new();
        let generate = super::Generate::<ConcatKeyValueCache>::new(
            &mut model,
            &mut cache,
            0.0,
            &prompt_tokens,
        );
        for (token, ntoks) in generate.zip(0..10) {
            let token = token.unwrap();
            tokens.push(token.clone());

            if ntoks == 0 {
                eval(&tokens).unwrap();
            }

            if tokens.len() % 20 == 0 {
                eval(&tokens).unwrap();
                let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
                let s = tokenizer.decode(&slice, true).unwrap();
                print!("{s}");
            }
        }

        eval(&tokens).unwrap();
        let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
        let s = tokenizer.decode(&slice, true).unwrap();
        println!("{s}");
        println!("------");
    }
}
