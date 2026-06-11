//! Gemma 4 dense [`crate::LanguageModel`] adapter.
//!
//! Gemma 4 uses a per-layer sliding/global cache enum
//! ([`crate::gemma4::text::cache::LayerCache`]) instead of the bare
//! [`crate::cache::KVCache`] used by llama / qwen3. The
//! `Vec<Option<LayerCache>>` slots are built up front by
//! [`crate::gemma4::text::cache::make_caches`].

use std::path::Path;

use mlx_rs::{module::Module, ops::indexing::IndexOp, Array};

use crate::cache::{effective_prefill_chunk_opt, CacheOptions};
use crate::chat_template::ChatTemplate;
use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::family::{EosSpec, LoadedContext};
use crate::gemma4::mtp::config::DrafterConfig;
use crate::gemma4::mtp::decode::{mtp_step, MtpContext};
use crate::gemma4::mtp::drafter::Drafter;
use crate::gemma4::mtp::weights::load_drafter;
use crate::gemma4::text::cache::{make_caches, LayerCache};
use crate::gemma4::text::config::{ModelConfig, TextConfig};
use crate::gemma4::text::text::Model;
use crate::gemma4::text::weights::load_model;
use crate::language_model::{LanguageModel, TextOnlyProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult};
use crate::loader::{load_tokenizer, resolve_bos_id};
use crate::nn::ModelInput;
use crate::sampler::SamplerState;

/// Upper bound on drafter depth γ (31B uses 8).
const MAX_DRAFT_DEPTH: u32 = 8;

pub(crate) struct Gemma4Adapter {
    model: Model,
    cache: Vec<Option<LayerCache>>,
    args: TextConfig,
    cache_options: CacheOptions,
    vocab_size: i32,
    /// MTP drafter + its state. `None` when no assistant checkpoint is loaded.
    drafter: Option<Drafter>,
    /// Target's last-position post-norm hidden (the drafter concat input).
    prev_hidden: Option<Array>,
    draft_depth: u32,
}

impl Gemma4Adapter {
    fn load(
        cfg: &Config,
        env: &ModelConfig,
        dir: &Path,
        draft_dir: Option<&Path>,
    ) -> Result<Self, Error> {
        let model = load_model(cfg, &env.text_config, dir)?;
        let args = model.args.clone();
        let vocab_size = args.vocab_size;
        let cache_options = CacheOptions::default();
        let cache = make_caches(&args, cache_options);

        let (drafter, draft_depth) = match draft_dir {
            Some(d) => {
                let dcfg = DrafterConfig::from_dir(d)?;
                let depth = dcfg.default_depth();
                (Some(load_drafter(&dcfg, d)?), depth)
            }
            None => (None, 0),
        };

        Ok(Self {
            model,
            cache,
            args,
            cache_options,
            vocab_size,
            drafter,
            prev_hidden: None,
            draft_depth,
        })
    }
}

impl Gemma4Adapter {
    /// Forward `inputs`, advancing the cache. When a drafter is loaded, also
    /// capture the last-position post-norm hidden as the next draft anchor.
    fn forward_capturing(&mut self, inputs: &Array) -> Result<Array, Error> {
        if self.drafter.is_some() {
            let (hidden, logits) = self
                .model
                .forward_hidden_and_logits(inputs, &mut self.cache)?;
            self.prev_hidden = Some(hidden.index((.., -1.., ..)));
            Ok(logits)
        } else {
            self.model.forward(ModelInput {
                inputs,
                mask: None,
                cache: &mut self.cache,
            })
        }
    }
}

impl LanguageModel for Gemma4Adapter {
    fn reset(&mut self) {
        self.cache = make_caches(&self.args, self.cache_options);
        self.prev_hidden = None;
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        let logits = self.forward_capturing(&input.text.tokens)?;
        Ok(PrepareResult::Logits(logits.index((.., -1, ..))))
    }

    fn step(&mut self, last_token: &Array) -> Result<LMOutput, Error> {
        let inp = last_token.reshape(&[1, 1])?;
        let logits = self.forward_capturing(&inp)?;
        Ok(LMOutput {
            logits: logits.index((.., -1, ..)),
        })
    }

    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }

    /// Gemma 4's sliding layers cap each forward at `sliding_window` K/V
    /// positions; combine with the user cap (which may narrow further but
    /// never exceed the window).
    fn prefill_chunk_size(&self) -> Option<i32> {
        effective_prefill_chunk_opt(&self.cache, self.cache_options.max_prefill_chunk)
    }

    fn prefill_chunk(&mut self, tokens: &Array) -> Result<(), Error> {
        let _ = self.forward_capturing(tokens)?;
        Ok(())
    }

    fn has_mtp(&self) -> bool {
        self.drafter.is_some()
    }

    fn set_mtp_depth(&mut self, n: u32) {
        self.draft_depth = n.clamp(1, MAX_DRAFT_DEPTH);
    }

    fn try_mtp_decode(
        &mut self,
        last_token: &Array,
        sampler: &mut SamplerState,
    ) -> Result<Option<(Vec<u32>, Array)>, Error> {
        let Some(drafter) = self.drafter.as_mut() else {
            return Ok(None);
        };
        let ctx = MtpContext {
            model: &mut self.model,
            cache: &mut self.cache,
            drafter,
            prev_hidden: &mut self.prev_hidden,
            depth: self.draft_depth,
            vocab_size: self.vocab_size,
        };
        mtp_step(ctx, last_token, sampler).map(Some)
    }

    fn set_cache_options(&mut self, options: CacheOptions) -> Result<(), Error> {
        self.cache = make_caches(&self.args, options);
        self.cache_options = options;
        Ok(())
    }
}

pub(crate) fn load_context(
    cfg: &Config,
    env: &ModelConfig,
    dir: &Path,
    draft_dir: Option<&Path>,
) -> Result<LoadedContext, Error> {
    let model = Gemma4Adapter::load(cfg, env, dir, draft_dir)?;
    let tokenizer = load_tokenizer(dir)?;
    let bos_id = resolve_bos_id(dir, &tokenizer);
    let chat_template = ChatTemplate::from_dir(dir)?;
    let eos_ids = EosSpec::to_vec(env.eos_token_id.as_ref());
    let processor = TextOnlyProcessor::new("gemma4", tokenizer, chat_template, bos_id);
    Ok((Box::new(model), Box::new(processor), eos_ids))
}
