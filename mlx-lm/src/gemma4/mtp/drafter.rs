//! Gemma 4 MTP drafter: a 4-layer Q-only gemma4 stack borrowing the target's
//! per-type K/V. `pre_projection(concat) → layers → norm → lm head`; returns
//! draft logits + the `post_projection` hidden for the next step's concat.

use mlx_rs::builder::Builder;
use mlx_rs::macros::{ModuleParameters, Quantizable};
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops::{concatenate_axis, indexing::IndexOp};
use mlx_rs::quantization::MaybeQuantized;
use mlx_rs::Array;

use crate::error::Error;
use crate::gemma4::mtp::centroid::MaskedEmbedder;
use crate::gemma4::mtp::config::DrafterConfig;
use crate::gemma4::text::config::{LayerKind, TextConfig};
use crate::gemma4::text::text::DecoderLayer;

/// Target's per-type K/V borrowed read-only: `(global, sliding)`.
pub struct SharedKv {
    pub global: (Array, Array),
    pub sliding: (Array, Array),
}

#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Drafter {
    args: TextConfig,
    layer_kinds: Vec<LayerKind>,

    /// `embed_tokens` `[vocab, draft_hidden]` — the drafter's own table; tied
    /// to the lm head.
    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
    /// `pre_projection` `Linear(2*backbone → draft)`.
    #[quantizable]
    #[param]
    pub pre_projection: MaybeQuantized<nn::Linear>,
    /// `post_projection` `Linear(draft → backbone)`.
    #[quantizable]
    #[param]
    pub post_projection: MaybeQuantized<nn::Linear>,
    /// Sparse centroid lm head (E2B/E4B). `None` → dense tied head.
    #[param]
    pub masked_embedding: Option<MaskedEmbedder>,
}

impl Drafter {
    pub fn new(cfg: &DrafterConfig) -> Result<Self, Error> {
        let args = cfg.text_config.clone();
        let layer_kinds = args.layer_types_resolved();
        let layers = (0..args.num_hidden_layers)
            .map(|i| DecoderLayer::new(&args, i))
            .collect::<Result<Vec<_>, _>>()?;

        let linear = |inp: i32, out: i32| -> Result<MaybeQuantized<nn::Linear>, Error> {
            Ok(MaybeQuantized::Original(
                nn::LinearBuilder::new(inp, out).bias(false).build()?,
            ))
        };

        let masked_embedding = if cfg.use_ordered_embeddings {
            Some(MaskedEmbedder::new(
                args.hidden_size,
                cfg.num_centroids,
                cfg.centroid_intermediate_top_k,
                args.vocab_size,
            )?)
        } else {
            None
        };

        Ok(Self {
            embed_tokens: MaybeQuantized::Original(nn::Embedding::new(
                args.vocab_size,
                args.hidden_size,
            )?),
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            pre_projection: linear(2 * cfg.backbone_hidden_size, args.hidden_size)?,
            post_projection: linear(args.hidden_size, cfg.backbone_hidden_size)?,
            masked_embedding,
            args,
            layer_kinds,
        })
    }

    /// One draft step: `concat_input` `[B,L,2*backbone]` + borrowed K/V at
    /// constant `position` → `(logits [B,L,vocab], backbone_hidden)`.
    pub fn forward(
        &mut self,
        concat_input: &Array,
        shared_kv: &SharedKv,
        position: i32,
    ) -> Result<(Array, Array), Error> {
        let mut h = self.pre_projection.forward(concat_input)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let kv = match self.layer_kinds[i] {
                LayerKind::FullAttention => &shared_kv.global,
                LayerKind::SlidingAttention => &shared_kv.sliding,
            };
            // cache: None — read-only, never writes the target cache.
            let out = layer.forward_layer(
                &h,
                None,
                None::<&mut crate::gemma4::text::cache::LayerCache>,
                Some((kv.0.clone(), kv.1.clone())),
                Some(position),
                None,
            )?;
            h = out.h;
        }
        let hidden = self.norm.forward(&h)?;
        let backbone_hidden = self.post_projection.forward(&hidden)?;
        let logits = self.logits(&hidden)?;
        Ok((logits, backbone_hidden))
    }

    /// Draft logits over the vocab from the post-norm draft hidden.
    fn logits(&mut self, hidden: &Array) -> Result<Array, Error> {
        if let Some(me) = self.masked_embedding.as_ref() {
            let w = match &self.embed_tokens {
                MaybeQuantized::Original(e) => e.weight.as_ref().clone(),
                MaybeQuantized::Quantized(_) => {
                    return Err(Error::config(
                        "gemma4 drafter: centroid head needs an un-quantized tied embed",
                    ))
                }
            };
            return me.forward(hidden, &w);
        }
        Ok(match &self.embed_tokens {
            MaybeQuantized::Original(e) => e.as_linear(hidden)?,
            MaybeQuantized::Quantized(qe) => qe.as_linear(hidden)?,
        })
    }

    pub fn vocab_size(&self) -> i32 {
        self.args.vocab_size
    }
}

/// `concat(target_embed, prev_backbone_hidden)` → `[B, L, 2*backbone]`.
pub fn concat_input(target_embed: &Array, prev_backbone_hidden: &Array) -> Result<Array, Error> {
    Ok(concatenate_axis(&[target_embed, prev_backbone_hidden], -1)?)
}

/// Convenience: the last position slice `[B, 1, D]` of a `[B, L, D]` hidden.
pub fn last_pos(h: &Array) -> Result<Array, Error> {
    Ok(h.index((.., -1.., ..)))
}
