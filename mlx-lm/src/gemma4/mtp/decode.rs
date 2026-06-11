//! Gemma 4 MTP speculative decode step.
//!
//! Snapshot → draft γ (read-only vs target K/V) → one verify forward → accept
//! prefix + bonus, or rollback + re-commit on partial reject.

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{concatenate_axis, stack_axis};
use mlx_rs::Array;

use crate::cache::KeyValueCache;
use crate::error::Error;
use crate::gemma4::mtp::drafter::{concat_input, last_pos, Drafter, SharedKv};
use crate::gemma4::text::cache::LayerCache;
use crate::gemma4::text::text::Model;
use crate::sampler::SamplerState;
use crate::speculative::{
    accept_mask, draft_confidence, draft_gate_for, resample_on_reject, sample_draft, CacheSnapshot,
};

/// Target's per-type shared K/V (last global + last sliding slot), read-only.
/// Resolves the two slot indices first so `current_kv()` (which dequantizes a
/// quantized cache) runs exactly twice, not once per owned layer.
pub fn shared_kv(cache: &[Option<LayerCache>]) -> Result<Option<SharedKv>, Error> {
    let mut global_idx: Option<usize> = None;
    let mut sliding_idx: Option<usize> = None;
    for (i, slot) in cache.iter().enumerate() {
        match slot {
            Some(LayerCache::Global(_)) => global_idx = Some(i),
            Some(LayerCache::Sliding(_)) => sliding_idx = Some(i),
            None => {}
        }
    }
    let (Some(gi), Some(si)) = (global_idx, sliding_idx) else {
        return Ok(None);
    };
    let (Some(global), Some(sliding)) = (cache[gi].as_ref(), cache[si].as_ref()) else {
        return Ok(None);
    };
    match (global.current_kv()?, sliding.current_kv()?) {
        (Some(global), Some(sliding)) => Ok(Some(SharedKv { global, sliding })),
        _ => Ok(None),
    }
}

/// Pieces of the gemma4 adapter the MTP step drives.
pub struct MtpContext<'a> {
    pub model: &'a mut Model,
    pub cache: &'a mut Vec<Option<LayerCache>>,
    pub drafter: &'a mut Drafter,
    pub prev_hidden: &'a mut Option<Array>,
    pub depth: u32,
    pub vocab_size: i32,
}

/// One speculative step. Returns `(committed_token_ids, next_pending_token)`.
pub fn mtp_step(
    ctx: MtpContext<'_>,
    last_token: &Array,
    sampler: &mut SamplerState,
) -> Result<(Vec<u32>, Array), Error> {
    let MtpContext {
        model,
        cache,
        drafter,
        prev_hidden,
        depth,
        vocab_size,
    } = ctx;
    let depth = depth as usize;

    let prev_h = prev_hidden
        .clone()
        .ok_or_else(|| Error::config("gemma4 mtp_step: prev_hidden unset; call prepare first"))?;

    let last_token_2d = last_token.reshape(&[1, 1])?;
    // Constant draft RoPE position: last committed token's index (`offset-1`),
    // read before the verify forward advances the cache.
    let offset = cache
        .iter()
        .flatten()
        .find_map(|s| matches!(s, LayerCache::Global(_)).then(|| s.offset()))
        .ok_or_else(|| Error::config("gemma4 mtp_step: no global cache slot"))?;
    let position = (offset - 1).max(0);

    let kv = shared_kv(cache)?
        .ok_or_else(|| Error::config("gemma4 mtp_step: target cache empty (no shared K/V)"))?;

    // Snapshot for partial-reject rollback.
    let mut guard = CacheSnapshot::new(cache);

    // Draft up to γ tokens read-only. Concat input = scaled-embed(token) ++
    // prev backbone hidden; next step's hidden is this step's post_projection.
    // A confidence gate stops drafting at the first token the drafter is unsure
    // about: on a fast (e.g. q4) target, verifying a low-confidence draft costs
    // more than a plain decode, so an empty draft → ordinary verified step.
    let mut draft_ids: Vec<Array> = Vec::with_capacity(depth);
    let mut draft_logits: Vec<Array> = Vec::with_capacity(depth);
    let mut prev_backbone = prev_h;
    let mut token_in = last_token_2d.clone();
    for d in 0..depth {
        let embed = model.embed_scaled_token(&token_in)?;
        let input = concat_input(&embed, &prev_backbone)?;
        let (logits_full, backbone) = drafter.forward(&input, &kv, position)?;
        let logits_i = logits_full.index((.., -1, ..));
        let id_i = sample_draft(sampler, &logits_i)?;
        if draft_confidence(&logits_i, &id_i)? < draft_gate_for(d) {
            break;
        }
        token_in = id_i.reshape(&[1, 1])?;
        prev_backbone = last_pos(&backbone)?;
        draft_ids.push(id_i);
        draft_logits.push(logits_i);
    }
    let depth = draft_ids.len();

    // Verify forward on [last_token, draft_0..draft_{γ-1}] (advances cache by γ+1).
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
    let (verify_hidden, verify_logits) = model.forward_hidden_and_logits(&verify_input, cache)?;

    let last_u32 = host_id(last_token.item::<i32>(), vocab_size)?;

    // Gate truncated every draft: the verify forward was a plain decode of
    // `last_token`. Commit it and sample the next token (no draft to accept).
    if depth == 0 {
        guard.commit();
        *prev_hidden = Some(verify_hidden.index((.., -1.., ..)));
        let next_pending = sampler.sample(&verify_logits.index((.., 0, ..)))?;
        return Ok((vec![last_u32], next_pending));
    }

    let draft_ids_stacked = stack_axis(&draft_ids, 0)?.reshape(&[depth as i32])?;
    let draft_logits_stacked = stack_axis(&draft_logits, 0)?.reshape(&[depth as i32, -1])?;
    let draft_ids_host: Vec<u32> = draft_ids_stacked
        .as_slice::<u32>()
        .iter()
        .map(|&id| host_u32(id, vocab_size))
        .collect::<Result<_, _>>()?;

    // Verify rows 0..depth pair with draft levels 0..depth.
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
        guard.commit();
        *prev_hidden = Some(verify_hidden.index((.., -1.., ..)));
        let next_logits = verify_logits.index((.., depth as i32, ..));
        let next_pending = sampler.sample(&next_logits)?;
        let mut committed = Vec::with_capacity(depth + 1);
        committed.push(last_u32);
        committed.extend_from_slice(&draft_ids_host);
        return Ok((committed, next_pending));
    }

    // Partial reject at k: roll back, re-commit the accepted prefix (k+1).
    guard.rollback_into(cache);
    let corrected = {
        let verify_k = verify_logits.index((.., k as i32, ..));
        resample_on_reject(sampler, &draft_logits[k], &verify_k)?
    };
    let mut accept_inputs: Vec<&Array> = Vec::with_capacity(k + 1);
    accept_inputs.push(&last_token_2d);
    for d in draft_ids_2d.iter().take(k) {
        accept_inputs.push(d);
    }
    let accept_tokens = concatenate_axis(&accept_inputs, 1)?;
    let (rehidden, _) = model.forward_hidden_and_logits(&accept_tokens, cache)?;
    *prev_hidden = Some(rehidden.index((.., -1.., ..)));

    let mut committed = Vec::with_capacity(k + 1);
    committed.push(last_u32);
    committed.extend_from_slice(&draft_ids_host[..k]);
    Ok((committed, corrected))
}

fn host_id(id: i32, vocab: i32) -> Result<u32, Error> {
    if id < 0 || id >= vocab {
        return Err(Error::shape(format!(
            "gemma4 mtp: out-of-vocab id {id} (vocab = {vocab})"
        )));
    }
    Ok(id as u32)
}

fn host_u32(id: u32, vocab: i32) -> Result<u32, Error> {
    if id >= vocab as u32 {
        return Err(Error::shape(format!(
            "gemma4 mtp: out-of-vocab id {id} (vocab = {vocab})"
        )));
    }
    Ok(id)
}
