//! Family-agnostic sampling helpers for speculative (draft/verify) decode.
//!
//! Shared by every family that runs a draft model or MTP head: the draft
//! token pick, the per-level accept decision, and the on-reject resample.
//! All operate purely on logits [`Array`]s + [`SamplerState`], so no family
//! state leaks in.

use mlx_rs::{
    ops::{
        argsort_axis, cumsum, exp, indexing::argmax_axis, indexing::take_along_axis, maximum,
        r#where, softmax_axis, sum_axis,
    },
    random::{categorical, uniform},
    Array,
};

use crate::error::Error;
use crate::sampler::{Sampler, SamplerState};

/// Vocab-positional top-p keep mask (`[1, vocab]` bool): slot `i` is `true`
/// iff token id `i` is in the smallest descending-probability set whose
/// preceding cumulative mass is below `p` — the same set
/// [`crate::sampler::top_p_sample`] keeps, indexed by vocab id.
pub fn top_p_keep_mask(logits: &Array, p: f32) -> Result<Array, Error> {
    let probs = softmax_axis(logits, -1, true)?;
    let order = argsort_axis(&probs.negative()?, -1)?;
    let sorted_probs = take_along_axis(&probs, &order, -1)?;
    let csum = cumsum(&sorted_probs, -1, false, false)?;
    let prev = csum.subtract(&sorted_probs)?;
    let keep_sorted = prev.lt(Array::from_f32(p))?;
    // argsort(order) is the inverse permutation: maps each vocab id to its
    // sort position, so the keep flags land back in vocab order.
    let inverse = argsort_axis(&order, -1)?;
    Ok(take_along_axis(&keep_sorted, &inverse, -1)?)
}

/// Pick the draft token. Greedy at `temperature == 0` (`argmax`);
/// categorical on the masked log-probs otherwise. The `[1]`-shape output is
/// what the rest of the speculative step expects.
pub fn sample_draft(sampler: &mut SamplerState, draft_logits: &Array) -> Result<Array, Error> {
    if matches!(sampler.sampler(), Sampler::Greedy) {
        return Ok(argmax_axis(draft_logits, -1, None)?.reshape(&[1])?);
    }
    let top_p_mask = match sampler.sampler().top_p() {
        Some(p) => Some(top_p_keep_mask(draft_logits, p)?),
        None => None,
    };
    let lp = sampler.masked_log_probs(draft_logits, top_p_mask.as_ref())?;
    Ok(categorical(&lp, None, None, None)?.reshape(&[1])?)
}

/// First-token draft-confidence gate. A step only speculates when the
/// drafter's top-token probability clears this; tuned for throughput, not
/// max accept (a longer-but-usually-accepted draft beats a short certain one).
pub const DRAFT_GATE_FIRST: f32 = 0.90;
/// Deep-position (≥ second draft token) gate. Tighter: a wrong deep draft
/// pays a larger recompute on reject, so only near-certain tokens extend it.
pub const DRAFT_GATE_DEEP: f32 = 0.999;

/// The drafter's own (temperature-1.0) probability for `draft_id` under
/// `draft_logits` `[1, vocab]` — the confidence the gate tests. One device
/// softmax + one host read.
pub fn draft_confidence(draft_logits: &Array, draft_id: &Array) -> Result<f32, Error> {
    let probs = softmax_axis(draft_logits, -1, true)?.reshape(&[-1])?;
    let id = draft_id.reshape(&[1])?;
    Ok(take_along_axis(&probs, &id, -1)?
        .reshape(&[1])?
        .item::<f32>())
}

/// Confidence threshold for draft position `depth_idx` (0 = first token).
pub fn draft_gate_for(depth_idx: usize) -> f32 {
    if depth_idx == 0 {
        DRAFT_GATE_FIRST
    } else {
        DRAFT_GATE_DEEP
    }
}

/// Per-level accept decisions for the whole draft batch in ONE device pass +
/// ONE host read, so verify-forward compute isn't serialized by a `.item()`
/// between levels. `draft_ids_stacked` is `[depth]`; `draft_logits` /
/// `verify_levels` are each `[depth, vocab]` (row `i` = level `i`).
///
/// Per-level acceptance: at `temperature == 0` argmax equality; above 0 the
/// Leviathan test over a shared union top-p mask, accept iff
/// `log p_verify(draft) - log p_draft(draft) >= log u`. The single
/// `uniform([depth])` draw has the same per-level acceptance probability as
/// depth-many scalar draws, but is not bit-identical for a fixed seed.
pub fn accept_mask(
    sampler: &mut SamplerState,
    draft_ids_stacked: &Array,
    draft_logits: &Array,
    verify_levels: &Array,
) -> Result<Vec<bool>, Error> {
    let accepts = if matches!(sampler.sampler(), Sampler::Greedy) {
        let verify_ids = argmax_axis(verify_levels, -1, None)?;
        verify_ids.eq(draft_ids_stacked)?
    } else {
        let keep_mask = match sampler.sampler().top_p() {
            Some(p) => {
                let draft_mask = top_p_keep_mask(draft_logits, p)?;
                let verify_mask = top_p_keep_mask(verify_levels, p)?;
                Some(draft_mask.logical_or(&verify_mask)?)
            }
            None => None,
        };
        let draft_lp = sampler.masked_log_probs(draft_logits, keep_mask.as_ref())?;
        let verify_lp = sampler.masked_log_probs(verify_levels, keep_mask.as_ref())?;
        let ids_2d = draft_ids_stacked.reshape(&[-1, 1])?;
        let lp_v = take_along_axis(&verify_lp, &ids_2d, -1)?.reshape(&[-1])?;
        let lp_d = take_along_axis(&draft_lp, &ids_2d, -1)?.reshape(&[-1])?;
        let log_ratio = lp_v.subtract(&lp_d)?;
        let depth = *log_ratio.shape().first().expect("log_ratio is [depth]");
        let u = uniform::<_, f32>(0.0_f32, 1.0_f32, &[depth], None)?;
        let log_u = u.log()?.as_dtype(log_ratio.dtype())?;
        log_ratio.ge(&log_u)?
    };
    Ok(accepts.as_slice::<bool>().to_vec())
}

/// Pick the corrected token at the rejected position. At `temperature == 0`
/// this is `argmax(verify_logits_i)`. Above 0 it is a categorical draw from
/// the Leviathan residual `max(0, exp(verify_lp) - exp(draft_lp))`, falling
/// back to the verify distribution when the residual sums to zero (the union
/// top-p mask collapsed the support).
pub fn resample_on_reject(
    sampler: &mut SamplerState,
    draft_logits: &Array,
    verify_logits_i: &Array,
) -> Result<Array, Error> {
    if matches!(sampler.sampler(), Sampler::Greedy) {
        return Ok(argmax_axis(verify_logits_i, -1, None)?.reshape(&[1])?);
    }
    let keep_mask = match sampler.sampler().top_p() {
        Some(p) => {
            let draft_mask = top_p_keep_mask(draft_logits, p)?;
            let verify_mask = top_p_keep_mask(verify_logits_i, p)?;
            Some(draft_mask.logical_or(&verify_mask)?)
        }
        None => None,
    };
    let draft_lp = sampler.masked_log_probs(draft_logits, keep_mask.as_ref())?;
    let verify_lp = sampler.masked_log_probs(verify_logits_i, keep_mask.as_ref())?;
    let p_v = exp(&verify_lp)?;
    let p_d = exp(&draft_lp)?;
    let zero = Array::from_f32(0.0).as_dtype(p_v.dtype())?;
    let residual = maximum(&p_v.subtract(&p_d)?, &zero)?;
    let z = sum_axis(&residual, -1, true)?;
    // Host sync on z: residual either has mass (sample from it) or sums to
    // zero (top-p mask collapsed the support — fall back to verify). The
    // branch is cheap and only fires on reject.
    let z_host = z.reshape(&[])?.item::<f32>();
    if z_host > 0.0 {
        let mask = residual.gt(&zero)?;
        let safe = r#where(&mask, &residual, &zero)?;
        let log_r = safe.log()?;
        Ok(categorical(&log_r, None, None, None)?.reshape(&[1])?)
    } else {
        Ok(categorical(&verify_lp, None, None, None)?.reshape(&[1])?)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    use super::*;

    #[test]
    fn keeps_only_top_token_at_half() {
        let logits = Array::from_slice(&[-10.0_f32, 5.0, -10.0], &[1, 3]);
        let mask = top_p_keep_mask(&logits, 0.5).unwrap();
        let m: &[bool] = mask.as_slice();
        assert_eq!(m, &[false, true, false]);
    }

    #[test]
    fn keeps_all_at_p_one() {
        let logits = Array::from_slice(&[0.1_f32, 0.5, 0.3, 0.05, 0.05], &[1, 5]);
        let mask = top_p_keep_mask(&logits, 1.0).unwrap();
        let m: &[bool] = mask.as_slice();
        assert_eq!(m, &[true, true, true, true, true]);
    }

    #[test]
    fn draft_confidence_is_softmax_prob_of_picked_id() {
        // Peaked logits: id 1 dominates, so its softmax prob ≈ 1.
        let logits = Array::from_slice(&[-10.0_f32, 10.0, -10.0], &[1, 3]);
        let high = draft_confidence(&logits, &Array::from_slice(&[1_u32], &[1])).unwrap();
        assert!(high > 0.99, "confident pick should clear the gate: {high}");
        let low = draft_confidence(&logits, &Array::from_slice(&[0_u32], &[1])).unwrap();
        assert!(low < 0.01, "unlikely pick should fail the gate: {low}");
    }

    #[test]
    fn gate_is_tighter_at_depth() {
        assert_eq!(draft_gate_for(0), DRAFT_GATE_FIRST);
        assert_eq!(draft_gate_for(1), DRAFT_GATE_DEEP);
        assert!(draft_gate_for(1) > draft_gate_for(0));
    }
}
