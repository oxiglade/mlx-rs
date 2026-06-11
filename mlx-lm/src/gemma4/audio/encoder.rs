//! Gemma 4 audio encoder (USM Conformer).
//!
//! Log-mel `[B,T,128]` → 2-layer Conv2d subsample → 12 Conformer blocks
//! (macaron FFN, chunked local attention with relative-position bias, depthwise
//! causal-conv GLU) → output projection → `[B,T',1536]`. `EmbedAudio` then
//! projects to the text hidden size. Attention runs in f32; all linears clamp
//! per `use_clipped_linears`. Single-clip path (B=1), no padding mask needed
//! when the whole clip is valid.

use mlx_rs::{
    builder::Builder,
    macros::{ModuleParameters, Quantizable},
    module::{Module, Param},
    nn,
    ops::{
        clip, concatenate_axis, conv_general, einsum, expand_dims, indexing::IndexOp, pad, reshape,
        softmax_axis, split, tanh as tanh_op, transpose_axes, tril,
    },
    quantization::MaybeQuantized,
    Array, Dtype,
};

use crate::error::Error;
use crate::gemma4::audio::clippable::ClippableLinear;
use crate::gemma4::audio::config::{AudioConfig, INPUT_FEAT_SIZE, SUBSAMPLING_CONV_CHANNELS};
use crate::nn::RmsNormNoScale;

/// RMSNorm epsilon-keyed `nn::RmsNorm` at `dim`.
fn rms_norm(dim: i32, eps: f32) -> Result<nn::RmsNorm, Error> {
    Ok(nn::RmsNormBuilder::new(dim).eps(eps).build()?)
}

/// `[B,T,F,C]` LayerNorm over the channel (last) axis, no bias (the checkpoint
/// carries only `norm.weight`). Built affine, then the bias param is nulled so
/// only `weight` binds.
fn channel_layer_norm(channels: i32, eps: f32) -> Result<nn::LayerNorm, Error> {
    let mut ln = nn::LayerNormBuilder::new(channels).eps(eps).build()?;
    ln.bias = Param::new(None);
    Ok(ln)
}

/// Conv2d k=3 s=2 (NHWC, weight `[out,3,3,in]`), no bias, padding done manually.
fn sscp_conv(inp: i32, out: i32) -> Result<nn::Conv2d, Error> {
    Ok(nn::Conv2dBuilder::new(inp, out, (3, 3))
        .stride((2, 2))
        .padding((0, 0))
        .bias(false)
        .build()?)
}

/// 2-layer Conv2d subsample (downsample mel time + freq) → flatten → project
/// to `hidden_size`. Channel-last throughout. No mask threading: B=1, the whole
/// clip is valid (single-utterance OCR/transcribe path).
#[derive(Debug, Clone, ModuleParameters)]
pub struct SubSampleConvProjection {
    #[param]
    pub layer0_conv: nn::Conv2d,
    #[param]
    pub layer0_norm: nn::LayerNorm,
    #[param]
    pub layer1_conv: nn::Conv2d,
    #[param]
    pub layer1_norm: nn::LayerNorm,
    #[param]
    pub input_proj_linear: nn::Linear,
}

impl SubSampleConvProjection {
    pub fn new(cfg: &AudioConfig) -> Result<Self, Error> {
        let [c0, c1] = SUBSAMPLING_CONV_CHANNELS;
        // freq after two stride-2 convs with pad 1: 128 → 64 → 32.
        let mut freq = INPUT_FEAT_SIZE;
        for _ in 0..2 {
            freq = (freq + 2 - 3) / 2 + 1;
        }
        let proj_in = freq * c1;
        Ok(Self {
            layer0_conv: sscp_conv(1, c0)?,
            layer0_norm: channel_layer_norm(c0, cfg.rms_norm_eps)?,
            layer1_conv: sscp_conv(c0, c1)?,
            layer1_norm: channel_layer_norm(c1, cfg.rms_norm_eps)?,
            input_proj_linear: nn::LinearBuilder::new(proj_in, cfg.hidden_size)
                .bias(false)
                .build()?,
        })
    }

    /// One SSCP block: symmetric pad (1,1) on T and F → Conv2d → LayerNorm → ReLU.
    fn block(conv: &mut nn::Conv2d, norm: &mut nn::LayerNorm, x: &Array) -> Result<Array, Error> {
        let padded = pad(x, &[(0, 0), (1, 1), (1, 1), (0, 0)][..], None, None)?;
        let y = conv.forward(&padded)?;
        let y = norm.forward(&y)?;
        Ok(nn::relu(&y)?)
    }

    /// `audio_mel [B,T,128]` → `[B,T',hidden_size]`.
    pub fn forward(&mut self, audio_mel: &Array) -> Result<Array, Error> {
        let x = expand_dims(audio_mel, -1)?; // [B,T,128,1]
        let x = Self::block(&mut self.layer0_conv, &mut self.layer0_norm, &x)?;
        let x = Self::block(&mut self.layer1_conv, &mut self.layer1_norm, &x)?;
        let shape = x.shape();
        let (b, t, f, c) = (shape[0], shape[1], shape[2], shape[3]);
        let x = reshape(&x, &[b, t, f * c])?;
        Ok(self.input_proj_linear.forward(&x)?)
    }
}

/// Macaron feed-forward: pre_norm → fc1 → silu → fc2 → post_norm → `+0.5·x`.
#[derive(Debug, Clone, ModuleParameters)]
pub struct ConformerFeedForward {
    grad_clip: f32,
    residual_weight: f32,

    #[param]
    pub pre_layer_norm: nn::RmsNorm,
    #[param]
    pub ffw_layer_1: ClippableLinear,
    #[param]
    pub ffw_layer_2: ClippableLinear,
    #[param]
    pub post_layer_norm: nn::RmsNorm,
}

impl ConformerFeedForward {
    pub fn new(cfg: &AudioConfig) -> Result<Self, Error> {
        let h = cfg.hidden_size;
        Ok(Self {
            grad_clip: cfg.gradient_clipping,
            residual_weight: cfg.residual_weight,
            pre_layer_norm: rms_norm(h, cfg.rms_norm_eps)?,
            ffw_layer_1: ClippableLinear::new(h, h * 4)?,
            ffw_layer_2: ClippableLinear::new(h * 4, h)?,
            post_layer_norm: rms_norm(h, cfg.rms_norm_eps)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let residual = x;
        let h = grad_clip(x, self.grad_clip)?;
        let h = self.pre_layer_norm.forward(&h)?;
        let h = self.ffw_layer_1.forward(&h)?;
        let h = nn::silu(&h)?;
        let h = self.ffw_layer_2.forward(&h)?;
        let h = grad_clip(&h, self.grad_clip)?;
        let h = self.post_layer_norm.forward(&h)?;
        Ok(residual
            .add(&h.multiply(Array::from_f32(self.residual_weight).as_dtype(h.dtype())?)?)?)
    }
}

/// Chunked local self-attention with relative-position bias, per-dim scaling,
/// and logit softcap. Computed in f32.
#[derive(Debug, Clone, ModuleParameters)]
pub struct AudioAttention {
    num_heads: i32,
    head_dim: i32,
    chunk_size: i32,
    max_past: i32,
    max_future: i32,
    context_size: i32,
    q_scale: f32,
    k_scale: f32,
    softcap: f32,
    invalid: f32,

    #[param]
    pub q_proj: ClippableLinear,
    #[param]
    pub k_proj: ClippableLinear,
    #[param]
    pub v_proj: ClippableLinear,
    #[param]
    pub post: ClippableLinear,
    #[param]
    pub relative_k_proj: nn::Linear,
    #[param]
    pub per_dim_scale: Param<Array>,
}

impl AudioAttention {
    pub fn new(cfg: &AudioConfig) -> Result<Self, Error> {
        let h = cfg.hidden_size;
        let hd = cfg.head_dim();
        Ok(Self {
            num_heads: cfg.num_attention_heads,
            head_dim: hd,
            chunk_size: cfg.attention_chunk_size,
            max_past: cfg.max_past_horizon(),
            max_future: cfg.max_future_horizon(),
            context_size: cfg.context_size(),
            q_scale: (hd as f32).powf(-0.5) / std::f32::consts::LN_2,
            k_scale: (1.0 + std::f32::consts::E).ln() / std::f32::consts::LN_2,
            softcap: cfg.attention_logit_cap,
            invalid: cfg.attention_invalid_logits_value,
            q_proj: ClippableLinear::new(h, h)?,
            k_proj: ClippableLinear::new(h, h)?,
            v_proj: ClippableLinear::new(h, h)?,
            post: ClippableLinear::new(h, h)?,
            relative_k_proj: nn::LinearBuilder::new(h, h).bias(false).build()?,
            per_dim_scale: Param::new(Array::zeros::<f32>(&[hd])?),
        })
    }

    /// Pad axis 1 by `(left, right)`.
    fn pad_t(x: &Array, left: i32, right: i32) -> Result<Array, Error> {
        let nd = x.shape().len();
        let mut widths = vec![(0, 0); nd];
        widths[1] = (left, right);
        Ok(pad(x, &widths[..], None, None)?)
    }

    /// `[B,T,...]` → `[B,U,chunk,...]` (pad T up to a chunk multiple).
    fn to_blocks(&self, x: &Array) -> Result<Array, Error> {
        let shape = x.shape();
        let (b, t) = (shape[0], shape[1]);
        let rest = &shape[2..];
        let u = (t + self.chunk_size - 1) / self.chunk_size;
        let pad_len = u * self.chunk_size - t;
        let x = if pad_len > 0 {
            Self::pad_t(x, 0, pad_len)?
        } else {
            x.clone()
        };
        let mut new_shape = vec![b, u, self.chunk_size];
        new_shape.extend_from_slice(rest);
        Ok(reshape(&x, &new_shape)?)
    }

    /// `[B,T,...]` → `[B,U,context,...]` sliding context windows.
    fn block_context(&self, x: &Array) -> Result<Array, Error> {
        let pad_left = self.max_past;
        let pad_right = self.max_future + self.chunk_size - 1;
        let x = Self::pad_t(x, pad_left, pad_right)?;
        let shape = x.shape();
        let (b, t_pad) = (shape[0], shape[1]);
        let rest = &shape[2..];
        let u = (t_pad - self.context_size) / self.chunk_size + 1;
        // Gather context windows: window i spans [i*chunk, i*chunk+context).
        let mut windows: Vec<Array> = Vec::with_capacity(u as usize);
        for i in 0..u {
            let start = i * self.chunk_size;
            windows.push(x.index((.., start..start + self.context_size)));
        }
        // Stack along a new axis-1 → [B,U,context,...].
        let stacked: Vec<Array> = windows
            .iter()
            .map(|w| expand_dims(w, 1))
            .collect::<Result<_, _>>()?;
        let refs: Vec<&Array> = stacked.iter().collect();
        let out = concatenate_axis(&refs, 1)?;
        let _ = (b, rest);
        Ok(out)
    }

    /// Sinusoidal relative-position table projected by `relative_k_proj`:
    /// `[max_span, num_heads, head_dim]`. `max_span = max_past + max_future + 1`.
    fn rel_pos_emb(&mut self, dtype: Dtype) -> Result<Array, Error> {
        let channels = self.num_heads * self.head_dim;
        let num_timescales = channels / 2;
        let min_ts = 1.0_f32;
        let max_ts = 10_000.0_f32;
        let log_inc = (max_ts / min_ts).ln() / ((num_timescales - 1).max(1) as f32);
        // positions descending: [max_past, ..., -max_future]
        let max_span = self.max_past + self.max_future + 1;
        let mut signal = Vec::with_capacity((max_span * channels) as usize);
        for p in 0..max_span {
            let pos = (self.max_past - p) as f32;
            // sin block then cos block (no interleave).
            for j in 0..num_timescales {
                let inv = min_ts * (-(j as f32) * log_inc).exp();
                signal.push((pos * inv).sin());
            }
            for j in 0..num_timescales {
                let inv = min_ts * (-(j as f32) * log_inc).exp();
                signal.push((pos * inv).cos());
            }
        }
        let emb = Array::from_slice(&signal, &[max_span, channels]).as_dtype(dtype)?;
        let proj = self.relative_k_proj.forward(&emb)?;
        Ok(reshape(&proj, &[max_span, self.num_heads, self.head_dim])?)
    }

    /// Transformer-XL relative shift on `[B,N,U,W,max_span]` → `[B,N,U,W,context]`.
    fn rel_shift(&self, term_bd: &Array, max_span: i32) -> Result<Array, Error> {
        let shape = term_bd.shape();
        let (b, n, u, w) = (shape[0], shape[1], shape[2], shape[3]);
        let c = self.context_size;
        let pad_amount = (c + 1) - max_span;
        let padded = pad(
            term_bd,
            &[(0, 0), (0, 0), (0, 0), (0, 0), (0, pad_amount)][..],
            None,
            None,
        )?;
        let flat = reshape(&padded, &[b, n, u, w * (c + 1)])?;
        let trimmed = flat.index((.., .., .., 0..w * c));
        Ok(reshape(&trimmed, &[b, n, u, w, c])?)
    }

    /// Build the `[context, chunk]` causal+validity mask (bool).
    fn causal_valid_mask(&self, dtype: Dtype) -> Result<Array, Error> {
        let c = self.context_size;
        let w = self.chunk_size;
        let upper_diag = self.max_past + self.max_future;
        // lower_causal = tril(ones[c,w]).T ; upper_causal = tril(ones[w,c], k=upper_diag)
        let lower = transpose_axes(&tril(Array::ones::<f32>(&[c, w])?, 0)?, &[1, 0])?;
        let upper = tril(Array::ones::<f32>(&[w, c])?, upper_diag)?;
        Ok(lower.multiply(&upper)?.as_dtype(dtype)?)
    }

    /// `hidden [B,T,hidden]` → `[B,T,hidden]`. No padding mask (B=1 valid clip).
    pub fn forward(&mut self, hidden: &Array) -> Result<Array, Error> {
        let shape = hidden.shape();
        let (b, t) = (shape[0], shape[1]);
        let qkv = |proj: &mut ClippableLinear, x: &Array| -> Result<Array, Error> {
            let y = proj.forward(x)?.as_dtype(Dtype::Float32)?;
            Ok(reshape(&y, &[b, t, self.num_heads, self.head_dim])?)
        };
        let q = qkv(&mut self.q_proj, hidden)?;
        let k = qkv(&mut self.k_proj, hidden)?;
        let v = qkv(&mut self.v_proj, hidden)?;

        let dim_scale = nn::softplus(&self.per_dim_scale.as_dtype(Dtype::Float32)?)?;
        let q_scale = dim_scale.multiply(Array::from_f32(self.q_scale))?;
        let q = q.multiply(&q_scale)?;
        let k = k.multiply(Array::from_f32(self.k_scale))?;

        let query_blocks = self.to_blocks(&q)?; // [B,U,W,N,H]
        let key_blocks = self.block_context(&k)?; // [B,U,C,N,H]
        let value_blocks = self.block_context(&v)?;
        let u = query_blocks.shape()[1];

        // term_ac = q·kᵀ : [B,N,U,W,C]
        let q_p = transpose_axes(&query_blocks, &[0, 3, 1, 2, 4])?; // [B,N,U,W,H]
        let k_p = transpose_axes(&key_blocks, &[0, 3, 1, 4, 2])?; // [B,N,U,H,C]
        let term_ac = einsum("bnuwh,bnuhc->bnuwc", [&q_p, &k_p])?;

        // term_bd = q·relposᵀ then relative shift.
        let max_span = self.max_past + self.max_future + 1;
        let rel = self.rel_pos_emb(Dtype::Float32)?; // [max_span,N,H]
        let rel_t = transpose_axes(&rel, &[1, 2, 0])?; // [N,H,max_span]
        let q_flat = reshape(
            &q_p,
            &[b, self.num_heads, u * self.chunk_size, self.head_dim],
        )?;
        let term_bd = einsum("bnxh,nhs->bnxs", [&q_flat, &rel_t])?;
        let term_bd = reshape(&term_bd, &[b, self.num_heads, u, self.chunk_size, max_span])?;
        let term_bd = self.rel_shift(&term_bd, max_span)?;

        let logits = term_ac.add(&term_bd)?;
        // softcap: tanh(logits/cap)·cap
        let logits = tanh_op(&logits.multiply(Array::from_f32(1.0 / self.softcap))?)?
            .multiply(Array::from_f32(self.softcap))?;
        // mask: causal_valid [W,C] broadcast to [1,1,1,W,C]
        let mask = self.causal_valid_mask(Dtype::Bool)?;
        let mask = expand_dims(&expand_dims(&expand_dims(&mask, 0)?, 0)?, 0)?;
        let invalid = Array::from_f32(self.invalid).as_dtype(Dtype::Float32)?;
        let logits = mlx_rs::ops::r#where(&mask, &logits, &invalid)?;

        let probs = softmax_axis(&logits, -1, None)?; // [B,N,U,W,C]
                                                      // context = Σ_c probs · value_blocks : [B,U,W,N,H]
        let context = einsum("bnuwc,bucnh->buwnh", [&probs, &value_blocks])?;
        let context = reshape(
            &context,
            &[b, u * self.chunk_size, self.num_heads, self.head_dim],
        )?;
        let context = context.index((.., 0..t));
        let context = reshape(&context, &[b, t, self.num_heads * self.head_dim])?;
        self.post.forward(&context.as_dtype(hidden.dtype())?)
    }
}

/// Depthwise causal-conv GLU block.
#[derive(Debug, Clone, ModuleParameters)]
pub struct ConformerLightConv1d {
    grad_clip: f32,
    causal_pad: i32,
    channels: i32,

    #[param]
    pub pre_layer_norm: nn::RmsNorm,
    #[param]
    pub linear_start: ClippableLinear,
    #[param]
    pub depthwise_conv1d: Param<Array>,
    #[param]
    pub conv_norm: nn::RmsNorm,
    #[param]
    pub linear_end: ClippableLinear,
}

impl ConformerLightConv1d {
    pub fn new(cfg: &AudioConfig) -> Result<Self, Error> {
        let h = cfg.hidden_size;
        Ok(Self {
            grad_clip: cfg.gradient_clipping,
            causal_pad: cfg.conv_kernel_size - 1,
            channels: h,
            pre_layer_norm: rms_norm(h, cfg.rms_norm_eps)?,
            linear_start: ClippableLinear::new(h, h * 2)?,
            depthwise_conv1d: Param::new(Array::zeros::<f32>(&[h, cfg.conv_kernel_size, 1])?),
            conv_norm: rms_norm(h, cfg.rms_norm_eps)?,
            linear_end: ClippableLinear::new(h, h)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let residual = x;
        let h = self.pre_layer_norm.forward(x)?;
        let h = self.linear_start.forward(&h)?;
        let halves = split(&h, 2, -1)?;
        let h = halves[0].multiply(&mlx_rs::ops::sigmoid(&halves[1])?)?;
        // causal left-pad on time then depthwise conv (groups = channels).
        let h = pad(&h, &[(0, 0), (self.causal_pad, 0), (0, 0)][..], None, None)?;
        let h = conv_general(
            &h,
            &self.depthwise_conv1d,
            &[1][..],
            &[0][..],
            None,
            None,
            self.channels,
            None,
        )?;
        let h = grad_clip(&h, self.grad_clip)?;
        let h = self.conv_norm.forward(&h)?;
        let h = nn::silu(&h)?;
        let h = self.linear_end.forward(&h)?;
        Ok(h.add(residual)?)
    }
}

/// One macaron Conformer block.
#[derive(Debug, Clone, ModuleParameters)]
pub struct ConformerBlock {
    grad_clip: f32,

    #[param]
    pub feed_forward1: ConformerFeedForward,
    #[param]
    pub self_attn: AudioAttention,
    #[param]
    pub lconv1d: ConformerLightConv1d,
    #[param]
    pub feed_forward2: ConformerFeedForward,
    #[param]
    pub norm_pre_attn: nn::RmsNorm,
    #[param]
    pub norm_post_attn: nn::RmsNorm,
    #[param]
    pub norm_out: nn::RmsNorm,
}

impl ConformerBlock {
    pub fn new(cfg: &AudioConfig) -> Result<Self, Error> {
        let h = cfg.hidden_size;
        Ok(Self {
            grad_clip: cfg.gradient_clipping,
            feed_forward1: ConformerFeedForward::new(cfg)?,
            self_attn: AudioAttention::new(cfg)?,
            lconv1d: ConformerLightConv1d::new(cfg)?,
            feed_forward2: ConformerFeedForward::new(cfg)?,
            norm_pre_attn: rms_norm(h, cfg.rms_norm_eps)?,
            norm_post_attn: rms_norm(h, cfg.rms_norm_eps)?,
            norm_out: rms_norm(h, cfg.rms_norm_eps)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let x = self.feed_forward1.forward(x)?;
        let residual = &x;
        let h = grad_clip(&x, self.grad_clip)?;
        let h = self.norm_pre_attn.forward(&h)?;
        let h = self.self_attn.forward(&h)?;
        let h = grad_clip(&h, self.grad_clip)?;
        let x = residual.add(&self.norm_post_attn.forward(&h)?)?;
        let x = self.lconv1d.forward(&x)?;
        let x = self.feed_forward2.forward(&x)?;
        let x = grad_clip(&x, self.grad_clip)?;
        Ok(self.norm_out.forward(&x)?)
    }
}

/// The full audio encoder.
#[derive(Debug, ModuleParameters)]
pub struct AudioEncoder {
    #[param]
    pub subsample_conv_projection: SubSampleConvProjection,
    #[param]
    pub layers: Vec<ConformerBlock>,
    #[param]
    pub output_proj: Option<nn::Linear>,
}

impl AudioEncoder {
    pub fn new(cfg: &AudioConfig) -> Result<Self, Error> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| ConformerBlock::new(cfg))
            .collect::<Result<Vec<_>, _>>()?;
        let output_proj = cfg
            .output_proj_dims
            .map(|d| nn::LinearBuilder::new(cfg.hidden_size, d).build())
            .transpose()?;
        Ok(Self {
            subsample_conv_projection: SubSampleConvProjection::new(cfg)?,
            layers,
            output_proj,
        })
    }

    /// `audio_mel [B,T,128]` → `[B,T',output_proj_dims]`.
    pub fn forward(&mut self, audio_mel: &Array) -> Result<Array, Error> {
        let mut h = self.subsample_conv_projection.forward(audio_mel)?;
        for block in &mut self.layers {
            h = block.forward(&h)?;
        }
        if let Some(proj) = self.output_proj.as_mut() {
            h = proj.forward(&h)?;
        }
        Ok(h)
    }
}

/// Projector `embed_audio`: param-free RMS norm → quantized Linear from the
/// audio output dim to the text hidden size.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct EmbedAudio {
    pub norm: RmsNormNoScale,
    #[quantizable]
    #[param]
    pub embedding_projection: MaybeQuantized<nn::Linear>,
}

impl EmbedAudio {
    pub fn new(cfg: &AudioConfig, text_hidden_size: i32) -> Result<Self, Error> {
        Ok(Self {
            norm: RmsNormNoScale::new(cfg.rms_norm_eps),
            embedding_projection: MaybeQuantized::Original(
                nn::LinearBuilder::new(cfg.projector_input_dim(), text_hidden_size)
                    .bias(false)
                    .build()?,
            ),
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let normed = self.norm.forward(x)?;
        Ok(self.embedding_projection.forward(&normed)?)
    }
}

/// Clamp to `±limit` (the encoder's `gradient_clipping` guard).
fn grad_clip(x: &Array, limit: f32) -> Result<Array, Error> {
    let lo = Array::from_f32(-limit).as_dtype(x.dtype())?;
    let hi = Array::from_f32(limit).as_dtype(x.dtype())?;
    Ok(clip(x, (&lo, &hi))?)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;
    use mlx_rs::random::uniform;

    fn synthetic_config() -> AudioConfig {
        let json = serde_json::json!({
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "conv_kernel_size": 5,
            "output_proj_dims": 48,
            "rms_norm_eps": 1e-6,
            "attention_chunk_size": 4,
            "attention_context_left": 5,
            "attention_context_right": 0,
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn encoder_shape_round_trip() {
        let cfg = synthetic_config();
        let mut enc = AudioEncoder::new(&cfg).unwrap();
        // 20 mel frames → subsample /4 → 5 tokens.
        let mel = uniform::<_, f32>(0.0, 1.0, &[1, 20, INPUT_FEAT_SIZE], None).unwrap();
        let out = enc.forward(&mel).unwrap();
        let s = out.shape();
        assert_eq!(s[0], 1);
        assert_eq!(s[2], 48);
        assert!(s[1] > 0);
    }

    #[test]
    fn embed_audio_projects_to_text_hidden() {
        let cfg = synthetic_config();
        let mut ea = EmbedAudio::new(&cfg, 64).unwrap();
        let x = uniform::<_, f32>(0.0, 1.0, &[5, 48], None).unwrap();
        let out = ea.forward(&x).unwrap();
        assert_eq!(out.shape(), &[5, 64]);
    }
}
