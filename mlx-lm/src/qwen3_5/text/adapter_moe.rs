//! Qwen3.5-MoE (35B-A3B) [`crate::LanguageModel`] adapter.
//!
//! Same prefill / decode shape as the dense qwen3.5 adapter; the
//! only difference is the inner FFN type (`Qwen35MoeBlock`). No
//! multimodal path — MoE checkpoints are text-only.

use std::path::Path;

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{concatenate_axis, stack_axis};
use mlx_rs::Array;

use crate::cache::CacheOptions;
use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::family::LoadedContext;
use crate::language_model::{LanguageModel, TextOnlyProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult};
use crate::loader::resolve_bos_id;
use crate::qwen3_5::text::cache::{make_caches, make_mtp_caches, LayerCache};
use crate::qwen3_5::text::config::ModelConfig;
use crate::qwen3_5::text::layer::Qwen35Model;
use crate::qwen3_5::text::load_common;
use crate::qwen3_5::text::moe::{load_qwen3_5_moe_model, Qwen35MoeBlock};
use crate::sampler::SamplerState;
use crate::speculative::{
    accept_mask, draft_confidence, draft_gate_for, resample_on_reject, sample_draft, CacheSnapshot,
};

/// Upper bound on MTP draft depth. The walk-back algorithm is
/// depth-generic; this cap reflects the depth past which adding
/// drafts stops paying its cost on the per-call wall clock. On a
/// bandwidth-bound 35B MoE at 89% per-slot acceptance, depth-3
/// double-reject probability is `0.11² ≈ 1%` and triple-reject is
/// `0.11³ ≈ 0.1%`, so the expected cache-restore + re-prime cost
/// stays small. Past 3 the marginal D→D+1 acceptance ratio
/// (`accept²` for each added slot) makes the verify-forward cost
/// dominate.
pub const MAX_MTP_DEPTH: u32 = 3;

pub struct Qwen35MoeAdapter {
    model: Qwen35Model<Qwen35MoeBlock>,
    cfg: ModelConfig,
    cache: Vec<LayerCache>,
    /// Per-MTP-layer caches. Empty when the checkpoint has no MTP head.
    mtp_cache: Vec<LayerCache>,
    cache_options: CacheOptions,
    /// Post-final-norm hidden at the last decoded position, sliced
    /// to `[B=1, 1, hidden]`. Fed into the MTP head, which applies
    /// its own `pre_fc_norm_hidden` on top. `None` before the first
    /// prepare/step.
    prev_hidden: Option<Array>,
    vocab_size: i32,
    /// Number of tokens the MTP head drafts ahead per speculative
    /// call. Default 2 (best throughput/acceptance trade on the A3B
    /// MoE); override via [`Self::set_mtp_depth`]. Clamped to
    /// `1..=MAX_MTP_DEPTH`.
    mtp_depth: u32,
}

impl Qwen35MoeAdapter {
    pub fn load(cfg: &Config, env: &ModelConfig, dir: &Path) -> Result<Self, Error> {
        let model = load_qwen3_5_moe_model(cfg, env, dir)?;
        let cache_options = CacheOptions::default();
        let cache = make_caches(env, cache_options);
        let mtp_cache = make_mtp_caches(env, cache_options);
        let vocab_size = env.text_config.vocab_size;
        Ok(Self {
            model,
            cfg: env.clone(),
            cache,
            mtp_cache,
            cache_options,
            prev_hidden: None,
            vocab_size,
            mtp_depth: 2,
        })
    }

    /// Current MTP draft depth.
    pub fn mtp_depth(&self) -> u32 {
        self.mtp_depth
    }
}

impl LanguageModel for Qwen35MoeAdapter {
    fn reset(&mut self) {
        self.cache = make_caches(&self.cfg, self.cache_options);
        self.mtp_cache = make_mtp_caches(&self.cfg, self.cache_options);
        self.prev_hidden = None;
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        let tokens = input.text.tokens;
        let (hidden, logits) =
            self.model
                .forward_hidden_and_logits(Some(&tokens), &mut self.cache, None)?;
        prime_mtp_cache(&mut self.model, &tokens, &hidden, &mut self.mtp_cache)?;
        self.prev_hidden = Some(hidden.index((.., -1..)));
        Ok(PrepareResult::Logits(logits.index((.., -1, ..))))
    }

    fn step(&mut self, last_token: &Array) -> Result<LMOutput, Error> {
        let inp = last_token.reshape(&[1, 1])?;
        let prior_hidden = self.prev_hidden.clone();
        let (hidden, logits) =
            self.model
                .forward_hidden_and_logits(Some(&inp), &mut self.cache, None)?;
        // Keep the MTP cache offset in lockstep with the main cache so a
        // subsequent `try_mtp_decode` call sees matching RoPE positions.
        // No-op when the model ships no MTP head.
        if self.model.mtp.is_some() {
            if let Some(prior) = prior_hidden.as_ref() {
                let embed_next = self.model.embed_tokens(&inp)?;
                let mtp = self.model.mtp.as_mut().expect("checked mtp.is_some()");
                mtp.update_cache(prior, &embed_next, &mut self.mtp_cache)?;
            }
        }
        self.prev_hidden = Some(hidden.index((.., -1..)));
        Ok(LMOutput {
            logits: logits.index((.., -1, ..)),
        })
    }

    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }

    fn has_mtp(&self) -> bool {
        self.model.mtp.is_some()
    }

    fn try_mtp_decode(
        &mut self,
        last_token: &Array,
        sampler: &mut SamplerState,
    ) -> Result<Option<(Vec<u32>, Array)>, Error> {
        if self.model.mtp.is_none() {
            return Ok(None);
        }
        mtp_step(self, last_token, sampler).map(Some)
    }

    fn set_mtp_depth(&mut self, n: u32) {
        self.mtp_depth = n.clamp(1, MAX_MTP_DEPTH);
    }

    fn prefill_chunk_size(&self) -> Option<i32> {
        // Qwen3.5 caches are unbounded; user cap wins.
        self.cache_options.max_prefill_chunk
    }

    fn prefill_chunk(&mut self, tokens: &Array) -> Result<(), Error> {
        // MoE prefill chunks must also advance the MTP cache —
        // otherwise speculative decode after the final chunk sees
        // out-of-sync RoPE positions.
        let (hidden, _logits) =
            self.model
                .forward_hidden_and_logits(Some(tokens), &mut self.cache, None)?;
        prime_mtp_cache(&mut self.model, tokens, &hidden, &mut self.mtp_cache)?;
        // Track `prev_hidden` so the final `prepare` chunk's MTP step
        // has the right anchor.
        self.prev_hidden = Some(hidden.index((.., -1..)));
        Ok(())
    }

    fn set_cache_options(&mut self, options: CacheOptions) -> Result<(), Error> {
        self.cache = make_caches(&self.cfg, options);
        self.mtp_cache = make_mtp_caches(&self.cfg, options);
        self.cache_options = options;
        Ok(())
    }
}

/// One speculative MTP step.
///
/// Inputs:
/// - `last_token`: candidate for the next-to-commit slot. Its KV is
///   not yet in the cache.
/// - `self.prev_hidden`: post-final-norm hidden at the most-recently
///   committed cache slot. The Qwen 3.6 MTP head was trained against
///   the model's final-norm output and re-normalises via its own
///   `pre_fc_norm_hidden` on top.
/// - `sampler`: at `temperature == 0.0` the helpers below take the
///   greedy fast path (argmax draft, argmax-equality accept, argmax
///   resample). Above 0 they use Leviathan rejection sampling with a
///   shared union top-p mask between draft and verify distributions.
///
/// Algorithm:
/// 1. MTP forward on `prev_hidden` + embed(last_token) → draft logits.
///    Pick the draft token via [`sample_draft`].
/// 2. Snapshot caches.
/// 3. Two-token main forward `[last_token, draft]`. Returns logits at
///    both positions: `verify_logits[0]` (what comes after last_token)
///    and `verify_logits[1]` (what comes after draft).
/// 4. [`accept_draft`] tests `draft` against `verify_logits[0]`.
///    - Accept: emit `[last_token, draft]`. New `prev_hidden` is the
///      hidden at slot 1. Next pending sampled from `verify_logits[1]`.
///    - Reject: roll back caches, [`resample_on_reject`] picks the
///      corrected token from `verify_logits[0]`. Re-run a single
///      forward on `last_token` to commit just its slot. Emit
///      `[last_token]`. New `prev_hidden` is the hidden at the
///      committed slot. Next pending = corrected.
fn mtp_step(
    adapter: &mut Qwen35MoeAdapter,
    last_token: &Array,
    sampler: &mut SamplerState,
) -> Result<(Vec<u32>, Array), Error> {
    let depth = adapter.mtp_depth as usize;
    debug_assert!((1..=MAX_MTP_DEPTH as usize).contains(&depth));

    let prev_hidden = adapter
        .prev_hidden
        .clone()
        .ok_or_else(|| Error::Other("mtp_step: prev_hidden unset; call prepare first".into()))?;

    let last_token_2d = last_token.reshape(&[1, 1])?;
    // Host-read of `last_token` is deferred to after the verify forward
    // submission below so the GPU→host sync overlaps with the draft +
    // verify pipeline instead of blocking before it. `last_u32` is only
    // needed when building the `committed` return vec.

    // Snapshot caches BEFORE any draft so partial-reject can roll back
    // both main + mtp to the pre-step state and re-commit only the
    // accepted prefix. Snapshot clone is shared-ptr cheap (the Arrays
    // are `mlx::core::array` shared_ptr handles). The guard restores
    // both caches if dropped without `.commit()`, including the `?`
    // early-exit paths in the verify forward + accept_draft below.
    let mut main_guard = CacheSnapshot::new(&adapter.cache);
    let mut mtp_guard = CacheSnapshot::new(&adapter.mtp_cache);

    // Build the chained draft: drafts[i] predicts the token at slot
    // `last_token + i + 1`. Each MTP forward advances mtp_cache by 1
    // and produces both the draft's logits and the post-norm hidden
    // that feeds the NEXT chained MTP call as its `prev_hidden`.
    // Confidence gate: stop drafting at the first token the MTP head is unsure
    // about. On a fast target, verifying a low-confidence draft costs more than
    // a plain decode, so an empty draft falls back to an ordinary verified step.
    let mut draft_ids: Vec<Array> = Vec::with_capacity(depth);
    let mut draft_logits: Vec<Array> = Vec::with_capacity(depth);
    let mut prev_h = prev_hidden;
    let mut token_in = last_token_2d.clone();
    for d in 0..depth {
        let (logits_i, mtp_post_norm_i) = run_mtp(adapter, &prev_h, &token_in)?;
        let id_i = sample_draft(sampler, &logits_i)?;
        if draft_confidence(&logits_i, &id_i)? < draft_gate_for(d) {
            break;
        }
        token_in = id_i.reshape(&[1, 1])?;
        prev_h = mtp_post_norm_i;
        draft_ids.push(id_i);
        draft_logits.push(logits_i);
    }
    let depth = draft_ids.len();

    // Verify forward on [last_token, draft_0, .., draft_{depth-1}].
    // Main cache advances by depth+1.
    let mut verify_inputs: Vec<&Array> = Vec::with_capacity(depth + 1);
    verify_inputs.push(&last_token_2d);
    let draft_ids_2d: Vec<Array> = draft_ids
        .iter()
        .map(|d| d.reshape(&[1, 1]))
        .collect::<Result<_, _>>()?;
    for d in &draft_ids_2d {
        verify_inputs.push(d);
    }
    let verify_input = concatenate_axis(&verify_inputs, 1)?;
    let (verify_hidden, verify_logits) =
        adapter
            .model
            .forward_hidden_and_logits(Some(&verify_input), &mut adapter.cache, None)?;

    // Now sync `last_token` to host — verify forward has been submitted,
    // so this read overlaps with its dispatch instead of blocking it.
    let last_u32 = host_id_to_u32(last_token.item::<i32>(), adapter.vocab_size)?;

    // Gate truncated every draft: the verify forward was a plain decode of
    // `last_token`. The draft loop still ran one `run_mtp` before the gate
    // broke, advancing `mtp_cache` by the one committed token, so both caches
    // stay in lockstep — commit both and sample next from verify position 0.
    if depth == 0 {
        main_guard.commit();
        mtp_guard.commit();
        adapter.prev_hidden = Some(verify_hidden.index((.., -1..)));
        let next_pending = sampler.sample(&verify_logits.index((.., 0, ..)))?;
        return Ok((vec![last_u32], next_pending));
    }

    // Materialise host ids for every draft in one sync, instead of
    // re-syncing each `draft_ids[i]` individually inside the commit
    // loop below. Verify forward above already evaluated the chain,
    // so this stack is cheap. argmax returns u32; per-slot bounds
    // check mirrors `host_id_to_u32`.
    let draft_ids_stacked = stack_axis(&draft_ids, 0)?.reshape(&[depth as i32])?;
    // Stack per-level draft logits `[depth, vocab]` for the batched
    // accept decision below (one device pass, one host read).
    let draft_logits_stacked = stack_axis(&draft_logits, 0)?.reshape(&[depth as i32, -1])?;
    let vocab_u32 = u32::try_from(adapter.vocab_size).map_err(|_| {
        Error::Shape(format!(
            "mtp_step: vocab_size {} negative",
            adapter.vocab_size
        ))
    })?;
    let draft_ids_host: Vec<u32> = draft_ids_stacked
        .as_slice::<u32>()
        .iter()
        .map(|&id| {
            if id >= vocab_u32 {
                return Err(Error::Shape(format!(
                    "mtp_step: out-of-vocab id {id} (vocab = {vocab_u32})"
                )));
            }
            Ok(id)
        })
        .collect::<Result<_, _>>()?;

    // Walk-back accept: decide every level device-side in one pass + one
    // host read (no `.item()` between levels, so verify-forward compute
    // isn't serialized), then find the first level `k` that rejects.
    // `k == depth` means all-accept. Verify rows `0..depth` pair with
    // draft levels `0..depth`.
    let verify_levels = verify_logits
        .index((.., 0..depth as i32, ..))
        .reshape(&[depth as i32, -1])?;
    let accepts = accept_mask(
        sampler,
        &draft_ids_stacked,
        &draft_logits_stacked,
        &verify_levels,
    )?;
    let k = accepts.iter().position(|&a| !a).unwrap_or(depth);

    if k == depth {
        // All-accept: commit last_token + every draft.
        main_guard.commit();
        mtp_guard.commit();
        adapter.prev_hidden = Some(verify_hidden.index((.., -1..)));
        let next_logits = verify_logits.index((.., depth as i32, ..));
        let next_pending = sampler.sample(&next_logits)?;
        let mut committed = Vec::with_capacity(depth + 1);
        committed.push(last_u32);
        committed.extend_from_slice(&draft_ids_host);
        return Ok((committed, next_pending));
    }

    // Partial reject at level k (0 <= k < depth). Both caches are
    // currently over-committed: main by depth+1, mtp by depth. Roll
    // back to the pre-step snapshot via the guard, then re-commit
    // exactly the accepted prefix (k+1 tokens) on the main side, plus
    // matching MTP-cache priming so the next call's RoPE positions
    // line up.
    main_guard.rollback_into(&mut adapter.cache);
    mtp_guard.rollback_into(&mut adapter.mtp_cache);

    let corrected = {
        let verify_k = verify_logits.index((.., k as i32, ..));
        resample_on_reject(sampler, &draft_logits[k], &verify_k)?
    };

    // Re-commit the accepted prefix in one main forward.
    let mut accept_inputs: Vec<&Array> = Vec::with_capacity(k + 1);
    accept_inputs.push(&last_token_2d);
    for d in draft_ids_2d.iter().take(k) {
        accept_inputs.push(d);
    }
    let accept_tokens = concatenate_axis(&accept_inputs, 1)?;
    let (rehidden, _) =
        adapter
            .model
            .forward_hidden_and_logits(Some(&accept_tokens), &mut adapter.cache, None)?;
    // Re-prime MTP cache to match the new main cache offset. The
    // prime helper writes positions `1..k+1` (it skips position 0;
    // for k=0 the helper is a no-op since the accepted segment is
    // just `last_token` with no successor).
    prime_mtp_cache(
        &mut adapter.model,
        &accept_tokens,
        &rehidden,
        &mut adapter.mtp_cache,
    )?;
    adapter.prev_hidden = Some(rehidden.index((.., -1..)));

    let mut committed = Vec::with_capacity(k + 1);
    committed.push(last_u32);
    committed.extend_from_slice(&draft_ids_host[..k]);
    Ok((committed, corrected))
}

/// Run one MTP-head forward. Returns `(logits, mtp_post_norm)` — both
/// sliced to the last position. `logits` is `[1, vocab]` for sampling;
/// `mtp_post_norm` is `[1, 1, hidden]` for chained drafts (the next
/// MTP forward in a depth>1 chain consumes the prior level's post-norm
/// hidden as its `prev_hidden`).
fn run_mtp(
    adapter: &mut Qwen35MoeAdapter,
    prev_hidden: &Array,
    last_token_2d: &Array,
) -> Result<(Array, Array), Error> {
    let embed_next = adapter.model.embed_tokens(last_token_2d)?;
    let mtp = adapter
        .model
        .mtp
        .as_mut()
        .expect("run_mtp: caller checked mtp.is_some()");
    let mtp_hidden = mtp.forward(prev_hidden, &embed_next, &mut adapter.mtp_cache, None)?;
    let logits = adapter.model.apply_lm_head(&mtp_hidden)?;
    Ok((logits.index((.., -1, ..)), mtp_hidden.index((.., -1..))))
}

/// Populate `mtp_cache` so its offset matches the main cache after
/// prefill. Without this, the first `try_mtp_decode` call runs the
/// MTP attention block at RoPE position 0 while the main model is at
/// position `prompt_len` — the position-frequency mismatch collapses
/// MTP acceptance at long context. Mirrors what the standalone decode
/// loop would have done across the prompt one token at a time, but
/// folded into a single forward over the whole sequence.
///
/// `prompt_tokens` is `[1, N]`; `hidden_full` is the main model's
/// post-final-norm hidden over the same N positions. For N < 2 the
/// MTP head has nothing to predict from, so this is a no-op.
fn prime_mtp_cache(
    model: &mut Qwen35Model<Qwen35MoeBlock>,
    prompt_tokens: &Array,
    hidden_full: &Array,
    mtp_cache: &mut [LayerCache],
) -> Result<(), Error> {
    if model.mtp.is_none() {
        return Ok(());
    }
    let n = prompt_tokens.shape()[1];
    if n < 2 {
        return Ok(());
    }
    let next_tokens = prompt_tokens.index((.., 1..n));
    let next_embeds = model.embed_tokens(&next_tokens)?;
    let prime_hidden = hidden_full.index((.., ..n - 1));
    let mtp = model.mtp.as_mut().expect("checked mtp.is_some()");
    mtp.update_cache(&prime_hidden, &next_embeds, mtp_cache)?;
    Ok(())
}

fn host_id_to_u32(id: i32, vocab: i32) -> Result<u32, Error> {
    if id < 0 || id >= vocab {
        return Err(Error::Shape(format!(
            "mtp_step: out-of-vocab id {id} (vocab = {vocab})"
        )));
    }
    Ok(id as u32)
}

pub(crate) fn load_context_moe(
    cfg: &Config,
    env: &ModelConfig,
    dir: &Path,
) -> Result<LoadedContext, Error> {
    let model = Qwen35MoeAdapter::load(cfg, env, dir)?;
    let (tokenizer, chat_template, eos_ids) = load_common(env, dir)?;
    let bos_id = resolve_bos_id(dir, &tokenizer);
    let processor = TextOnlyProcessor::new("qwen3_5_moe", tokenizer, chat_template, bos_id);
    Ok((Box::new(model), Box::new(processor), eos_ids))
}
