//! Qwen3.5 dense [`crate::LanguageModel`] adapter.
//!
//! The dense path: `LanguageModel<Mlp>` with a hybrid linear-attn +
//! full-attn cache stack. Drives prefill / decode by calling the
//! model's `forward` directly. The text-only adapter is built
//! standalone; the VLM path wraps this adapter with the vision tower
//! plus multimodal embedding stitch (see
//! [`crate::qwen3_5::image::adapter`]).

use std::path::Path;

use mlx_rs::{ops::indexing::IndexOp, Array};

use crate::cache::CacheOptions;
use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::family::LoadedContext;
use crate::language_model::{LanguageModel, TextOnlyProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult};
use crate::loader::resolve_bos_id;
use crate::qwen3_5::text::cache::{make_caches, LayerCache};
use crate::qwen3_5::text::config::ModelConfig;
use crate::qwen3_5::text::layer::Qwen35Model;
use crate::qwen3_5::text::text::Mlp;
use crate::qwen3_5::text::weights::load_language_model;
use crate::qwen3_5::text::{leftover_keys_error, load_common};

pub(crate) struct Qwen35DenseAdapter {
    pub(crate) model: Qwen35Model<Mlp>,
    pub(crate) cfg: ModelConfig,
    pub(crate) cache: Vec<LayerCache>,
    pub(crate) cache_options: CacheOptions,
    /// Cumulative position (prompt + decoded tokens). Only the
    /// multimodal `step` path reads it to build the `[3,1,1]` mrope
    /// position id; pure text never touches it.
    #[cfg(feature = "image")]
    pub(crate) cursor: i32,
    /// `Some` after a multimodal prefill: per-step decode position is
    /// `cursor + rope_delta` broadcast across the three mrope axes.
    #[cfg(feature = "image")]
    pub(crate) rope_delta: Option<i32>,
    pub(crate) vocab_size: i32,
}

impl Qwen35DenseAdapter {
    pub(crate) fn new(model: Qwen35Model<Mlp>, cfg: ModelConfig) -> Result<Self, Error> {
        let cache_options = CacheOptions::default();
        let cache = make_caches(&cfg, cache_options);
        let vocab_size = cfg.text_config.vocab_size;
        Ok(Self {
            model,
            cfg,
            cache,
            cache_options,
            #[cfg(feature = "image")]
            cursor: 0,
            #[cfg(feature = "image")]
            rope_delta: None,
            vocab_size,
        })
    }

    /// Multimodal prefill seed: pre-stitched embeddings + `[3,1,S]` mrope
    /// position ids + `rope_delta`. The VLM adapter is the only caller;
    /// records `cursor`/`rope_delta` so the subsequent `step` calls keep
    /// the rope advancing past the image block.
    #[cfg(feature = "image")]
    pub(crate) fn prefill_embeds(
        &mut self,
        inputs_embeds: Array,
        position_ids: Array,
        rope_delta: i32,
    ) -> Result<Array, Error> {
        let s = inputs_embeds.shape()[1];
        let logits =
            self.model
                .forward_embeds(inputs_embeds, &mut self.cache, Some(&position_ids))?;
        self.cursor = s;
        self.rope_delta = Some(rope_delta);
        Ok(logits.index((.., -1, ..)))
    }
}

impl LanguageModel for Qwen35DenseAdapter {
    fn reset(&mut self) {
        self.cache = make_caches(&self.cfg, self.cache_options);
        #[cfg(feature = "image")]
        {
            self.cursor = 0;
            self.rope_delta = None;
        }
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        let tokens = input.text.tokens;
        let shape = tokens.shape();
        debug_assert_eq!(shape[0], 1, "batch dim must be 1");
        let logits = self.model.forward(Some(&tokens), &mut self.cache, None)?;
        #[cfg(feature = "image")]
        {
            self.cursor = shape[1];
            self.rope_delta = None;
        }
        Ok(PrepareResult::Logits(logits.index((.., -1, ..))))
    }

    fn step(&mut self, last_token: &Array) -> Result<LMOutput, Error> {
        let inp = last_token.reshape(&[1, 1])?;

        // Multimodal decode needs an explicit `[3,1,1]` mrope position id so
        // the rope keeps advancing past the image block; pure text leaves
        // `rope_delta = None` and the model derives the position from the
        // cache offset internally. Without the `image` feature there is no
        // VLM path, so `pos` is always `None`.
        #[cfg(feature = "image")]
        let pos = self.rope_delta.map(|delta| {
            let p = self.cursor + delta;
            Array::from_slice(&[p, p, p], &[3, 1, 1])
        });
        #[cfg(not(feature = "image"))]
        let pos: Option<Array> = None;

        let logits = self
            .model
            .forward(Some(&inp), &mut self.cache, pos.as_ref())?;
        #[cfg(feature = "image")]
        {
            self.cursor += 1;
        }
        Ok(LMOutput {
            logits: logits.index((.., -1, ..)),
        })
    }

    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }

    fn prefill_chunk_size(&self) -> Option<i32> {
        // Qwen3.5 caches are unbounded; user cap wins.
        self.cache_options.max_prefill_chunk
    }

    fn prefill_chunk(&mut self, tokens: &Array) -> Result<(), Error> {
        let _ = self.model.forward(Some(tokens), &mut self.cache, None)?;
        Ok(())
    }

    fn set_cache_options(&mut self, options: CacheOptions) -> Result<(), Error> {
        self.cache = make_caches(&self.cfg, options);
        self.cache_options = options;
        Ok(())
    }
}

/// Load a qwen3_5 dense (text-only) checkpoint. Caller is the
/// family-level [`crate::qwen3_5::load_context`] dispatcher; it
/// guarantees the directory carries the dense weights only (no
/// `preprocessor_config.json`).
pub(crate) fn load_context_dense(
    cfg: &Config,
    env: &ModelConfig,
    dir: &Path,
) -> Result<LoadedContext, Error> {
    let (tokenizer, chat_template, eos_ids) = load_common(env, dir)?;
    let bos_id = resolve_bos_id(dir, &tokenizer);
    let (model, leftover) = load_language_model(cfg, env, dir)?;
    if !leftover.is_empty() {
        return Err(leftover_keys_error("dense", &leftover));
    }
    let dense = Qwen35DenseAdapter::new(model, env.clone())?;
    let processor = TextOnlyProcessor::new("qwen3_5", tokenizer, chat_template, bos_id);
    Ok((Box::new(dense), Box::new(processor), eos_ids))
}
