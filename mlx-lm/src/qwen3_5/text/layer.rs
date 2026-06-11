//! Decoder layer + top-level [`Qwen35Decoder`] / [`Qwen35Model`] wrappers.
//!
//! [`DecoderLayer`] is the per-layer dispatch: it owns either a
//! `linear_attn` Gated DeltaNet block or a `self_attn` full-attention
//! block based on the checkpoint's `layer_types`. Both code paths
//! share the same input / post-attention norms and the SwiGLU MLP.
//!
//! [`Qwen35Decoder`] holds the embeddings, the layer stack, and the
//! final `model.norm`. [`Qwen35Model`] wraps it with the LM head
//! (tied or untied) plus the optional MTP head and runs end-to-end
//! logits.

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::{arange, broadcast_to, concatenate_axis, expand_dims, reshape},
    quantization::{MaybeQuantized, Quantizable as QuantizableTrait},
    Array,
};

use crate::cache::KeyValueCache;
use crate::error::Error;

use super::cache::LayerCache;
use super::config::TextConfig;
use super::gated_delta_block::GatedDeltaNet;
use super::rope::MultimodalRope;
use super::text::{Attention, Mlp};

/// One Qwen3.5 decoder layer: either linear-attention (GDN) or full-attention.
/// Generic over the FFN `F`: defaults to dense [`Mlp`] for Qwen3.5/3.6
/// dense; the MoE variants alias `DecoderLayer<Qwen35MoeBlock>`.
///
/// `self_attn` / `linear_attn` are kept in `Option` fields rather than an
/// enum so the derived `ModuleParameters` / `Quantizable` walks both paths
/// in the weight loader. Exactly one is populated per layer.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct DecoderLayer<F = Mlp>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Error>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    pub is_linear: bool,

    #[quantizable]
    #[param]
    pub self_attn: Option<Attention>,

    #[quantizable]
    #[param]
    pub linear_attn: Option<GatedDeltaNet>,

    #[param]
    pub input_layernorm: nn::RmsNorm,

    #[param]
    pub post_attention_layernorm: nn::RmsNorm,

    #[quantizable]
    #[param]
    pub mlp: F,
}

impl<F> DecoderLayer<F>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Error>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    /// Build a layer of the right kind for the given index. `make_ffn`
    /// constructs the per-layer FFN (typically `Mlp::new` for dense or
    /// `Qwen35MoeBlock::new` for the MoE variant).
    pub fn new<MakeF>(cfg: &TextConfig, layer_idx: usize, make_ffn: MakeF) -> Result<Self, Error>
    where
        MakeF: FnOnce(&TextConfig) -> Result<F, Error>,
    {
        let is_linear = cfg.is_linear_layer(layer_idx);
        let (self_attn, linear_attn) = if is_linear {
            (None, Some(GatedDeltaNet::new(cfg)?))
        } else {
            (Some(Attention::new(cfg)?), None)
        };
        let input_layernorm = nn::RmsNormBuilder::new(cfg.hidden_size)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(cfg.hidden_size)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let mlp = make_ffn(cfg)?;
        Ok(Self {
            is_linear,
            self_attn,
            linear_attn,
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }

    /// Run the layer forward.
    ///
    /// `cos`/`sin` are precomputed once per forward at the decoder level
    /// and shared across every layer. The linear-attention branch (GDN)
    /// ignores them.
    pub fn forward(
        &mut self,
        x: &Array,
        full_attn_mask: Option<&Array>,
        ssm_mask: Option<&Array>,
        cache: Option<&mut LayerCache>,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array, Error> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = if self.is_linear {
            let blk = self
                .linear_attn
                .as_mut()
                .expect("linear_attn missing for linear layer");
            let cache = cache.map(|c| c.as_linear_attention_mut());
            blk.forward(&normed, ssm_mask, cache)?
        } else {
            let blk = self
                .self_attn
                .as_mut()
                .expect("self_attn missing for full-attn layer");
            let cache = cache.map(|c| c.as_full_attention_mut());
            blk.forward(&normed, full_attn_mask, cache, cos, sin)?
        };
        let h = x.add(&attn_out)?;
        let mlp_out = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&h)?)?;
        Ok(h.add(&mlp_out)?)
    }

    /// Toggle training mode on every parameter (attention/GDN, norms, FFN).
    pub fn training_mode(&mut self, mode: bool) {
        if let Some(blk) = self.self_attn.as_mut() {
            blk.training_mode(mode);
        }
        if let Some(blk) = self.linear_attn.as_mut() {
            blk.training_mode(mode);
        }
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

/// Source-compat alias for the dense (Mlp-FFN) layer.
pub type DenseDecoderLayer = DecoderLayer<Mlp>;

impl DecoderLayer<Mlp> {
    /// Convenience constructor for the dense (Mlp-FFN) layer; mirrors
    /// the pre-generic API.
    pub fn with_mlp(cfg: &TextConfig, layer_idx: usize) -> Result<Self, Error> {
        Self::new(cfg, layer_idx, |c| {
            Mlp::new(c.hidden_size, c.intermediate_size)
        })
    }
}

impl<F> DecoderLayer<F>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Error>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    /// Force a `self_attn` layer regardless of the `layer_types` pattern.
    /// MTP heads in Qwen 3.6 always use standard QKV attention even when
    /// the corresponding main-decoder position would have been a linear
    /// (Mamba-style) layer.
    pub fn new_self_attn<MakeF>(cfg: &TextConfig, make_ffn: MakeF) -> Result<Self, Error>
    where
        MakeF: FnOnce(&TextConfig) -> Result<F, Error>,
    {
        let input_layernorm = nn::RmsNormBuilder::new(cfg.hidden_size)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(cfg.hidden_size)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let mlp = make_ffn(cfg)?;
        Ok(Self {
            is_linear: false,
            self_attn: Some(Attention::new(cfg)?),
            linear_attn: None,
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }
}

/// Qwen 3.6 MTP head (Multi-Token Prediction). Takes the prior decoder's
/// last hidden state `h_t` plus the embedding of `token[t+1]`, normalises
/// them independently, concatenates, projects 2H→H, runs through one
/// self-attention decoder layer, and outputs logits for `token[t+2]` via
/// the shared `lm_head`.
///
/// Weight layout (verified against `Qwen/Qwen3.6-35B-A3B`):
///   `mtp.pre_fc_norm_hidden.weight`        [H]
///   `mtp.pre_fc_norm_embedding.weight`     [H]
///   `mtp.fc.weight`                        [H, 2H]
///   `mtp.layers.0.*`                       full self-attn DecoderLayer
///   `mtp.norm.weight`                      [H]
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct MtpHead<F = Mlp>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Error>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    #[param]
    pub pre_fc_norm_hidden: nn::RmsNorm,
    #[param]
    pub pre_fc_norm_embedding: nn::RmsNorm,
    #[quantizable]
    #[param]
    pub fc: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer<F>>,
    #[param]
    pub norm: nn::RmsNorm,

    /// Multimodal RoPE — same shape as the main decoder's. cos/sin is
    /// computed once per MTP call and threaded into each layer below.
    rope: MultimodalRope,
}

impl<F> MtpHead<F>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Error>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    pub fn new<MakeF>(cfg: &TextConfig, mut make_ffn: MakeF) -> Result<Self, Error>
    where
        MakeF: FnMut(&TextConfig) -> Result<F, Error>,
    {
        let h = cfg.hidden_size;
        let pre_fc_norm_hidden = nn::RmsNormBuilder::new(h).eps(cfg.rms_norm_eps).build()?;
        let pre_fc_norm_embedding = nn::RmsNormBuilder::new(h).eps(cfg.rms_norm_eps).build()?;
        let fc = nn::LinearBuilder::new(2 * h, h).bias(false).build()?;
        let n = cfg.mtp_num_hidden_layers.max(0) as usize;
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            layers.push(DecoderLayer::<F>::new_self_attn(cfg, &mut make_ffn)?);
        }
        let norm = nn::RmsNormBuilder::new(h).eps(cfg.rms_norm_eps).build()?;
        let rope = build_rope(cfg)?;
        Ok(Self {
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            fc: MaybeQuantized::Original(fc),
            layers,
            norm,
            rope,
        })
    }

    /// Run the MTP head. `h_t` is the main decoder's post-final-norm
    /// hidden at the last committed slot; `embed_next` is the
    /// embedding of the sampled `token[t+1]`. The head normalises
    /// both inputs via its own `pre_fc_norm_*` RMSNorms; the hidden
    /// is therefore normalised twice (model.norm, then
    /// pre_fc_norm_hidden), which matches what the Qwen 3.6 MTP head
    /// weights were trained against. Returns the post-norm hidden
    /// ready for the shared `lm_head`.
    pub fn forward(
        &mut self,
        h_t: &Array,
        embed_next: &Array,
        caches: &mut [LayerCache],
        position_ids: Option<&Array>,
    ) -> Result<Array, Error> {
        let h = self.run_inner(h_t, embed_next, caches, position_ids)?;
        Ok(self.norm.forward(&h)?)
    }

    /// Run the MTP head over a multi-token prompt segment purely to
    /// populate `caches`. Same compute path as [`Self::forward`] but
    /// the post-norm + lm_head projection is skipped — the call site
    /// only cares about advancing the MTP KV cache to match the main
    /// cache offset.
    ///
    /// Inputs are 3-D `[B, S, H]`: `h_full` is the main decoder's
    /// post-final-norm hidden at positions `0..S-1`, `embed_full` is
    /// the embedding of the tokens at positions `1..S` (the MTP head
    /// predicts the next-next token, so each position `i` of the
    /// segment consumes hidden[i] + embed[i+1]). After this call
    /// `caches[0].offset` advances by `S`.
    pub fn update_cache(
        &mut self,
        h_full: &Array,
        embed_full: &Array,
        caches: &mut [LayerCache],
    ) -> Result<(), Error> {
        self.run_inner(h_full, embed_full, caches, None)?;
        Ok(())
    }

    fn run_inner(
        &mut self,
        h_t: &Array,
        embed_next: &Array,
        caches: &mut [LayerCache],
        position_ids: Option<&Array>,
    ) -> Result<Array, Error> {
        let h_n = self.pre_fc_norm_hidden.forward(h_t)?;
        let e_n = self.pre_fc_norm_embedding.forward(embed_next)?;
        let combined = concatenate_axis(&[e_n, h_n], -1)?;
        let mut h = self.fc.forward(&combined)?;
        // cos/sin once per MTP call, shared across every layer.
        let (cos, sin) = self.cos_sin_for_run(&h, caches, position_ids)?;
        for (layer, cache) in self.layers.iter_mut().zip(caches.iter_mut()) {
            h = layer.forward(&h, None, None, Some(cache), &cos, &sin)?;
        }
        Ok(h)
    }

    /// Build cos/sin pre-cast + pre-shaped for the per-layer rope.
    /// See [`Qwen35Decoder::cos_sin_for_forward`] for the rationale.
    fn cos_sin_for_run(
        &self,
        h: &Array,
        caches: &[LayerCache],
        position_ids: Option<&Array>,
    ) -> Result<(Array, Array), Error> {
        let h_shape = h.shape();
        let b = h_shape[0];
        let l = h_shape[1];
        let owned_pos;
        let pos: &Array = if let Some(p) = position_ids {
            p
        } else {
            // MTP layers are always full-attention; the first cache
            // slot's KV offset drives the position window.
            let offset = caches.first().and_then(|c| c.kv_offset()).unwrap_or(0);
            let range = arange::<_, i32>(offset, offset + l, None)?;
            let range = reshape(&range, &[1, l])?;
            owned_pos = broadcast_to(&range, &[b, l])?;
            &owned_pos
        };
        let (cos, sin) = self.rope.cos_sin(pos)?;
        let dtype = h.dtype();
        let cos = expand_dims(&cos, 1)?.as_dtype(dtype)?;
        let sin = expand_dims(&sin, 1)?.as_dtype(dtype)?;
        Ok((cos, sin))
    }

    pub fn training_mode(&mut self, mode: bool) {
        self.pre_fc_norm_hidden.training_mode(mode);
        self.pre_fc_norm_embedding.training_mode(mode);
        self.fc.training_mode(mode);
        for layer in &mut self.layers {
            layer.training_mode(mode);
        }
        self.norm.training_mode(mode);
    }
}

/// Build the shared multimodal RoPE from `cfg`. Used by both the main
/// decoder and the MTP head so cos/sin runs once per forward, not
/// once per layer.
fn build_rope(cfg: &TextConfig) -> Result<MultimodalRope, Error> {
    let rotary_dim =
        (cfg.head_dim as f32 * cfg.rope_parameters.partial_rotary_factor).floor() as i32;
    // Unsupported rope variants (yarn / longrope) reject at
    // `config.json` deserialize via `QwenRopeType`, so the only
    // values reaching here are `default` / `mrope`.
    MultimodalRope::new(
        rotary_dim,
        cfg.rope_parameters.rope_theta,
        &cfg.rope_parameters.mrope_section,
    )
}

/// Hidden states returned by [`Qwen35Decoder::forward_pre_and_post_norm`].
///
/// `pre_norm` feeds the MTP head (which applies its own
/// `pre_fc_norm_hidden`); `post_norm` is what the lm_head projects.
#[derive(Debug)]
pub struct DecoderOutput {
    pub pre_norm: Array,
    pub post_norm: Array,
}

/// Top-level decoder model: embeddings + layers + final norm. Generic
/// over the FFN type (defaults to dense [`Mlp`]).
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Qwen35Decoder<F = Mlp>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Error>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer<F>>,

    #[param]
    pub norm: nn::RmsNorm,

    /// Multimodal rotary embedding. Stateless config (`inv_freq`,
    /// axis-selector masks) — built once per model load and shared
    /// across every layer's `Attention::forward`. Held here (not on
    /// `Attention`) so `cos_sin` runs once per forward, not 64×.
    rope: MultimodalRope,
}

impl<F> Qwen35Decoder<F>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Error>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    /// Build a freshly-initialised decoder. `make_ffn` is called once per
    /// layer to construct the FFN block (e.g. dense `Mlp::new` or MoE
    /// `Qwen35MoeBlock::new`).
    pub fn new<MakeF>(cfg: &TextConfig, mut make_ffn: MakeF) -> Result<Self, Error>
    where
        MakeF: FnMut(&TextConfig) -> Result<F, Error>,
    {
        let embed_tokens = nn::Embedding::new(cfg.vocab_size, cfg.hidden_size)?;
        let layers = (0..cfg.num_hidden_layers as usize)
            .map(|i| DecoderLayer::<F>::new(cfg, i, &mut make_ffn))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(cfg.hidden_size)
            .eps(cfg.rms_norm_eps)
            .build()?;
        let rope = build_rope(cfg)?;
        Ok(Self {
            vocab_size: cfg.vocab_size,
            num_hidden_layers: cfg.num_hidden_layers,
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            layers,
            norm,
            rope,
        })
    }

    /// Run the decoder over a batch of token ids.
    ///
    /// - `inputs`: `[B, S]` int32 token ids.
    /// - `caches`: one [`LayerCache`] per layer. Mutated in place.
    /// - `position_ids`: optional `[3, B, S]` mrope position ids; `None`
    ///   derives positions from the cache offset.
    pub fn forward(
        &mut self,
        inputs: Option<&Array>,
        caches: &mut [LayerCache],
        position_ids: Option<&Array>,
    ) -> Result<Array, Error> {
        Ok(self
            .forward_pre_and_post_norm(inputs, None, caches, position_ids)?
            .post_norm)
    }

    /// Multimodal forward over pre-stitched `[B, S, hidden]` embeddings
    /// (image features already spliced in) with explicit mrope position
    /// ids. The VLM prefill path's entry point.
    #[cfg(feature = "image")]
    pub fn forward_embeds(
        &mut self,
        inputs_embeds: Array,
        caches: &mut [LayerCache],
        position_ids: Option<&Array>,
    ) -> Result<Array, Error> {
        Ok(self
            .forward_pre_and_post_norm(None, Some(inputs_embeds), caches, position_ids)?
            .post_norm)
    }

    /// Like [`Self::forward`] but returns the pre-final-norm hidden
    /// state alongside the post-norm one. The MTP head consumes
    /// `pre_norm` (it applies its own pre_fc_norm_hidden); the lm_head
    /// projection uses `post_norm`. `inputs_embeds` (the VLM stitched
    /// embeddings) is taken by value so it moves into `h` with no clone.
    pub fn forward_pre_and_post_norm(
        &mut self,
        inputs: Option<&Array>,
        inputs_embeds: Option<Array>,
        caches: &mut [LayerCache],
        position_ids: Option<&Array>,
    ) -> Result<DecoderOutput, Error> {
        let mut h = if let Some(e) = inputs_embeds {
            e
        } else {
            let ids = inputs
                .ok_or_else(|| Error::config("Qwen35Decoder::forward: needs inputs or embeds"))?;
            self.embed_tokens.forward(ids)?
        };
        if caches.len() != self.layers.len() {
            return Err(Error::shape(format!(
                "Qwen35Decoder::forward: expected {} caches, got {}",
                self.layers.len(),
                caches.len()
            )));
        }
        // Hybrid mask scheme:
        //
        // - Full-attention layers use a causal mask of shape `[1, 1, T, kv]`
        //   built from the matching layer's KV-cache offset. When `T == 1`
        //   (decode) the mask is dropped — every full-attn block already
        //   relies on `fast::scaled_dot_product_attention` to handle the
        //   trivial 1-row case.
        // - Linear-attention layers don't need a mask in single-batch /
        //   non-chunked prefill, so we always pass `None`.
        let full_attn_mask = Self::build_full_attn_mask(&h, caches)?;
        let full_attn_mask_ref = full_attn_mask.as_ref();
        let ssm_mask_ref: Option<&Array> = None;
        // cos/sin computed ONCE here and threaded to every layer below.
        // Was previously recomputed inside each `Attention::forward` —
        // identical inputs, 64 redundant cos_sin / matmul / cos/sin ops
        // per token on Qwen 3.6-27B.
        let (cos, sin) = self.cos_sin_for_forward(&h, caches, position_ids)?;
        for (layer, cache) in self.layers.iter_mut().zip(caches.iter_mut()) {
            h = layer.forward(
                &h,
                full_attn_mask_ref,
                ssm_mask_ref,
                Some(cache),
                &cos,
                &sin,
            )?;
        }
        let post = self.norm.forward(&h)?;
        Ok(DecoderOutput {
            pre_norm: h,
            post_norm: post,
        })
    }

    /// Build `(cos, sin)` for this forward, fully prepared for every
    /// layer's `apply_multimodal_rotary_pos_emb`: shape
    /// `[B, 1, S, rotary_dim]` and cast to the input dtype. Doing the
    /// `expand_dims(axis=1) + as_dtype(h_dtype)` here means the per-layer
    /// rope call is just two `multiply` + add; both ops would otherwise
    /// fire 64× per token.
    ///
    /// Pulls the position-id offset from the first full-attention cache
    /// slot (matches [`Self::build_full_attn_mask`]) when `position_ids`
    /// is `None`; otherwise uses the supplied tensor directly.
    fn cos_sin_for_forward(
        &self,
        h: &Array,
        caches: &[LayerCache],
        position_ids: Option<&Array>,
    ) -> Result<(Array, Array), Error> {
        let h_shape = h.shape();
        let b = h_shape[0];
        let l = h_shape[1];
        let owned_pos;
        let pos: &Array = if let Some(p) = position_ids {
            p
        } else {
            // Derive [B, L] positions from the first full-attention
            // cache slot's offset. Linear-attn slots don't track an
            // SDPA offset, so they're skipped in `Self::find_full_attn_offset`.
            let offset = Self::find_full_attn_offset(caches);
            let range = arange::<_, i32>(offset, offset + l, None)?;
            let range = reshape(&range, &[1, l])?;
            owned_pos = broadcast_to(&range, &[b, l])?;
            &owned_pos
        };
        let (cos, sin) = self.rope.cos_sin(pos)?;
        let dtype = h.dtype();
        // Unsqueeze the per-head broadcast axis and cast once; downstream
        // `apply_multimodal_rotary_pos_emb` consumes the pair as-is.
        let cos = expand_dims(&cos, 1)?.as_dtype(dtype)?;
        let sin = expand_dims(&sin, 1)?.as_dtype(dtype)?;
        Ok((cos, sin))
    }

    /// Pick the offset of the first full-attention cache slot; 0 if
    /// none exists (pure linear-attn config, or no cache attached).
    fn find_full_attn_offset(caches: &[LayerCache]) -> i32 {
        caches.iter().find_map(|c| c.kv_offset()).unwrap_or(0)
    }

    /// Build the additive causal mask the full-attention layers need.
    ///
    /// Hybrid-cache variant of [`crate::utils::create_attention_mask`]:
    /// pulls the offset from the first full-attention slot (its KV cache
    /// tracks the prefill-vs-decode boundary) and emits a `[T, T + offset]`
    /// boolean matrix lifted to a 4-D broadcast-friendly shape.
    ///
    /// Returns `None` for decode steps (`T == 1`) where the implicit causal
    /// handling inside `fast::scaled_dot_product_attention` already covers
    /// the trivial mask shape.
    fn build_full_attn_mask(h: &Array, caches: &[LayerCache]) -> Result<Option<Array>, Error> {
        let shape = h.shape();
        let t = shape[1];
        if t <= 1 {
            return Ok(None);
        }

        // Find the first full-attention cache and read its offset.
        let offset = caches
            .iter()
            .find_map(|c| match c {
                LayerCache::FullAttention(kv) => Some(kv.offset()),
                LayerCache::LinearAttention(_) => None,
            })
            .unwrap_or(0);

        let total = offset + t;
        let rinds = arange::<_, i32>(0, total, None)?;
        let linds = arange::<_, i32>(offset, total, None)?;
        // [T, 1] >= [1, total]  -> [T, total]
        let linds_b = expand_dims(&linds, 1)?;
        let rinds_b = expand_dims(&rinds, 0)?;
        let mask = linds_b.ge(&rinds_b)?;
        // Lift to [1, 1, T, total] so it broadcasts against [B, H, T, total].
        let mask = expand_dims(&expand_dims(&mask, 0)?, 0)?;
        Ok(Some(broadcast_to(&mask, &[1, 1, t, total])?))
    }

    /// Toggle training mode on every quantisable parameter.
    pub fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            layer.training_mode(mode);
        }
        self.norm.training_mode(mode);
    }
}

/// Source-compat alias for the dense (Mlp-FFN) decoder.
pub type DenseQwen35Decoder = Qwen35Decoder<Mlp>;

impl Qwen35Decoder<Mlp> {
    /// Dense convenience constructor.
    pub fn with_mlp(cfg: &TextConfig) -> Result<Self, Error> {
        Self::new(cfg, |c| Mlp::new(c.hidden_size, c.intermediate_size))
    }
}

/// Qwen3.5 model: decoder stack + final norm + LM head + optional MTP
/// head. Generic over the FFN type (defaults to dense [`Mlp`]; the
/// MoE variant aliases this as [`super::moe::Qwen35MoeModel`]).
///
/// Named `Qwen35Model` (not `LanguageModel`) so the crate-root trait
/// [`crate::LanguageModel`] keeps its single meaning. This struct is
/// the concrete model body the qwen3_5 adapters wrap to implement
/// that trait.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Qwen35Model<F = Mlp>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Error>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    pub cfg: TextConfig,

    #[quantizable]
    #[param]
    pub model: Qwen35Decoder<F>,

    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,

    /// MTP head (Qwen 3.6 self-speculative decode). Present iff the
    /// checkpoint ships `mtp.*` weights AND `cfg.mtp_num_hidden_layers > 0`.
    #[quantizable]
    #[param]
    pub mtp: Option<MtpHead<F>>,
}

impl<F> Qwen35Model<F>
where
    F: for<'a> Module<&'a Array, Output = Array, Error = Error>
        + QuantizableTrait<Quantized = F, QuantizationError = Exception>
        + std::fmt::Debug,
{
    /// Build a freshly-initialised language model. `make_ffn` is called
    /// once per decoder layer (and once more per MTP layer when the
    /// config enables an MTP head) to construct the FFN block.
    pub fn new<MakeF>(cfg: TextConfig, mut make_ffn: MakeF) -> Result<Self, Error>
    where
        MakeF: FnMut(&TextConfig) -> Result<F, Error>,
    {
        let model = Qwen35Decoder::<F>::new(&cfg, &mut make_ffn)?;
        let lm_head = if !cfg.tie_word_embeddings {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(cfg.hidden_size, cfg.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        } else {
            None
        };
        let mtp = if cfg.mtp_num_hidden_layers > 0 {
            Some(MtpHead::<F>::new(&cfg, &mut make_ffn)?)
        } else {
            None
        };
        Ok(Self {
            cfg,
            model,
            lm_head,
            mtp,
        })
    }

    /// Run the model end-to-end. Returns `[B, S, vocab_size]` logits.
    pub fn forward(
        &mut self,
        inputs: Option<&Array>,
        caches: &mut [LayerCache],
        position_ids: Option<&Array>,
    ) -> Result<Array, Error> {
        let (_, logits) = self.forward_hidden_and_logits(inputs, caches, position_ids)?;
        Ok(logits)
    }

    /// Multimodal forward over pre-stitched `[B, S, hidden]` embeddings
    /// (image features already spliced in) with explicit mrope position
    /// ids. Returns `[B, S, vocab]` logits. The VLM prefill entry point.
    #[cfg(feature = "image")]
    pub fn forward_embeds(
        &mut self,
        inputs_embeds: Array,
        caches: &mut [LayerCache],
        position_ids: Option<&Array>,
    ) -> Result<Array, Error> {
        let hidden = self
            .model
            .forward_embeds(inputs_embeds, caches, position_ids)?;
        self.apply_lm_head(&hidden)
    }

    /// Like [`Self::forward`] but also returns the post-final-norm
    /// hidden state over the full sequence. The MTP head consumes
    /// this post-norm hidden and applies its own `pre_fc_norm_hidden`
    /// on top; the same post-norm is what the lm_head projects to
    /// logits, so both outputs share the normalised hidden the model
    /// was trained to produce. Hidden is `[B, S, H]` — callers slice
    /// to `[:, -1:]` for the next-token logits or to `[:, :-1]` for
    /// the MTP prime pass.
    pub fn forward_hidden_and_logits(
        &mut self,
        inputs: Option<&Array>,
        caches: &mut [LayerCache],
        position_ids: Option<&Array>,
    ) -> Result<(Array, Array), Error> {
        let DecoderOutput { post_norm, .. } =
            self.model
                .forward_pre_and_post_norm(inputs, None, caches, position_ids)?;
        let logits = self.apply_lm_head(&post_norm)?;
        Ok((post_norm, logits))
    }

    /// Project a hidden state to vocab logits via the LM head (tied
    /// embed lookup or untied `lm_head` linear, whichever the cfg
    /// selected).
    pub fn apply_lm_head(&mut self, hidden: &Array) -> Result<Array, Error> {
        if let Some(head) = self.lm_head.as_mut() {
            Ok(head.forward(hidden)?)
        } else {
            match &mut self.model.embed_tokens {
                MaybeQuantized::Original(e) => Ok(e.as_linear(hidden)?),
                MaybeQuantized::Quantized(q) => Ok(q.as_linear(hidden)?),
            }
        }
    }

    /// Embed token ids via the (possibly quantised) `embed_tokens` table.
    /// Needed by the MTP head — it consumes `embed(token_t+1)` as one of
    /// its two inputs.
    pub fn embed_tokens(&mut self, ids: &Array) -> Result<Array, Error> {
        match &mut self.model.embed_tokens {
            MaybeQuantized::Original(e) => Ok(e.forward(ids)?),
            MaybeQuantized::Quantized(q) => Ok(q.forward(ids)?),
        }
    }

    /// Toggle training mode on every quantisable parameter.
    pub fn training_mode(&mut self, mode: bool) {
        self.model.training_mode(mode);
        if let Some(head) = self.lm_head.as_mut() {
            head.training_mode(mode);
        }
    }
}

impl Qwen35Model<Mlp> {
    /// Dense convenience constructor; mirrors the pre-generic API.
    pub fn with_mlp(cfg: TextConfig) -> Result<Self, Error> {
        Self::new(cfg, |c| Mlp::new(c.hidden_size, c.intermediate_size))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    #![allow(clippy::print_stdout, reason = "test code")]
    #![allow(clippy::print_stderr, reason = "test code")]
    use super::super::config::ModelConfig;
    use super::*;
    use crate::cache::CacheOptions;
    use crate::qwen3_5::text::cache::make_caches;
    use mlx_rs::{transforms::eval, Array};

    fn synthetic_config(layer_types: Vec<&str>) -> ModelConfig {
        let layers: Vec<String> = layer_types.into_iter().map(String::from).collect();
        let n = layers.len() as i32;
        let json = serde_json::json!({
            "model_type": "qwen3_5",
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": n,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "rms_norm_eps": 1e-6,
                "vocab_size": 100,
                "max_position_embeddings": 256,
                "layer_types": layers,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "tie_word_embeddings": true,
                "rope_parameters": {
                    "mrope_section": [2, 1, 1],
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                    "type": "default"
                }
            },
            "vision_config": {
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 32,
                "out_hidden_size": 32,
                "num_heads": 2,
                "patch_size": 16,
                "in_channels": 3,
                "spatial_merge_size": 2
            }
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn hybrid_model_end_to_end_shape() {
        let cfg = synthetic_config(vec![
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]);
        let mut lm = Qwen35Model::with_mlp(cfg.text_config.clone()).unwrap();
        let mut caches = make_caches(&cfg, CacheOptions::default());

        // Token ids in [0, vocab_size).
        let ids: Vec<i32> = (0..5).collect();
        let inputs = Array::from_slice(&ids, &[1, 5]);
        let logits = lm.forward(Some(&inputs), &mut caches, None).unwrap();
        eval([&logits]).unwrap();
        assert_eq!(logits.shape(), &[1, 5, cfg.text_config.vocab_size]);

        // Decode one more token.
        let next = Array::from_slice(&[42_i32], &[1, 1]);
        let logits2 = lm.forward(Some(&next), &mut caches, None).unwrap();
        eval([&logits2]).unwrap();
        assert_eq!(logits2.shape(), &[1, 1, cfg.text_config.vocab_size]);

        // The full-attn layer's KV cache should be at offset 6 now.
        let LayerCache::FullAttention(fa) = &caches[3] else {
            panic!("expected FullAttention cache at index 3");
        };
        use crate::cache::KeyValueCache;
        assert_eq!(fa.offset(), 6);

        // The linear-attn layers should each be at offset 6 too.
        for (i, cache) in caches.iter().enumerate().take(3) {
            let LayerCache::LinearAttention(la) = cache else {
                panic!("expected LinearAttention cache at index {i}");
            };
            assert_eq!(la.offset, 6, "linear layer {i} offset");
        }
    }
}
