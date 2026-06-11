//! Fused MoE-router top-k selection kernel.
//!
//! Replaces `softmax(logits) → argpartition → take_along → renormalise`
//! (which lowers to a full merge-sort of all `E` expert scores) with a
//! single threadgroup-local kernel: one threadgroup per token row does
//! `k` rounds of argmax-and-mask over the `E` logits in shared memory,
//! then softmaxes the `k` kept logits. No global sort, one launch.
//!
//! Output weights are softmax over the kept logits, which equals the
//! renormalised top-k of the full softmax (softmax is monotonic, so the
//! kept set matches, and softmax-of-subset == renormalised-softmax-subset).
//!
//! Model-agnostic: takes raw `[B, L, E]` logits and `(num_experts, k)`.
//! Qwen3.6-MoE and Gemma 4 MoE both consume it (Gemma post-multiplies the
//! weights by its per-expert scale).

use mlx_rs::fast::{metal_kernel, MetalKernel, MetalKernelConfig};
use mlx_rs::Array;
use mlx_rs::Dtype;
use mlx_rs::Stream;

use crate::error::Error;

/// Max experts the kernel supports in one threadgroup pass (Gemma 4 MoE
/// ships `E = 128`, Qwen3.6-MoE `E = 256`); the threadgroup-memory scratch
/// is sized to this.
const MAX_EXPERTS: i32 = 1024;

/// One threadgroup per token row, this many threads per group. Each
/// thread reduces `ceil(E / TG)` logits into the group argmax.
const TG_THREADS: i32 = 256;

/// Build the router top-k kernel. Caller caches the handle (see the
/// per-family `router_topk_kernel` accessors); per-call recreation
/// recompiles needlessly.
pub(crate) fn make_router_topk_kernel() -> Result<MetalKernel, Error> {
    Ok(metal_kernel(
        // Bump `_vN` on any source change so mlx's name cache doesn't
        // serve a stale binary.
        "moe_router_topk_v1",
        &["logits"],
        &["indices", "weights"],
        ROUTER_TOPK_SOURCE,
        "",
        true,
        false,
    )?)
}

/// Top-k expert selection + renormalised softmax weights for the MoE
/// router. `logits` is `[B, L, E]`; returns `(indices [B,L,k] u32,
/// weights [B,L,k] same-dtype-as-logits)`.
pub(crate) fn router_topk(
    kernel: &MetalKernel,
    logits: &Array,
    num_experts: i32,
    k: i32,
) -> Result<(Array, Array), Error> {
    let shape = logits.shape();
    if shape.len() != 3 {
        return Err(Error::shape("router_topk: logits must be [B, L, E]"));
    }
    let b = shape[0];
    let l = shape[1];
    let e = shape[2];
    if e != num_experts {
        return Err(Error::shape(format!(
            "router_topk: logits last dim {e} != num_experts {num_experts}"
        )));
    }
    if e > MAX_EXPERTS {
        return Err(Error::shape(format!(
            "router_topk: E={e} exceeds MAX_EXPERTS={MAX_EXPERTS}"
        )));
    }
    if k <= 0 || k > e {
        return Err(Error::shape(format!(
            "router_topk: invalid k={k} for E={e}"
        )));
    }

    let dtype = logits.dtype();
    let config = MetalKernelConfig::new()
        .add_output([b, l, k], Dtype::Uint32)
        .add_output([b, l, k], dtype)
        .grid(TG_THREADS, 1, b * l)
        .thread_group(TG_THREADS, 1, 1)
        .add_template("InT", dtype)?
        .add_template("E", e)?
        .add_template("K", k)?
        .add_template("TG", TG_THREADS)?;

    let [indices, weights] = kernel.apply_array::<2>(&[logits], config, Stream::default())?;
    Ok((indices, weights))
}

// One threadgroup per token row. `tid` = thread_position_in_threadgroup.x
// in [0, TG). Each thread scans a strided slice of the E logits.
//
// Algorithm per row:
//   1. Load logits into threadgroup scratch `vals[E]` (f32).
//   2. K rounds: each thread finds its local max over its strided slice
//      of the live scratch; a threadgroup reduction picks the global
//      argmax; thread 0 records (index, value) and masks that slot to
//      -inf so the next round skips it.
//   3. Softmax the K kept values; thread 0 writes indices + weights.
//
// K (≤ ~16) and E (≤ 1024) are template constants. Round count is K, so
// the loop is bounded and fully unrollable.
const ROUTER_TOPK_SOURCE: &str = r#"
    uint row = thread_position_in_grid.z;
    uint tid = thread_position_in_threadgroup.x;

    threadgroup float vals[E];
    threadgroup float red_val[TG];
    threadgroup uint  red_idx[TG];
    threadgroup uint  kept_idx[K];
    threadgroup float kept_val[K];

    auto row_logits = logits + (ulong)row * E;

    // 1. Stage logits into shared scratch as f32.
    for (uint i = tid; i < E; i += TG) {
        vals[i] = static_cast<float>(row_logits[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. K rounds of argmax-and-mask.
    for (uint r = 0; r < K; ++r) {
        float best_v = -INFINITY;
        uint  best_i = 0;
        for (uint i = tid; i < E; i += TG) {
            float v = vals[i];
            // Tie-break on lowest index to match a stable selection.
            if (v > best_v || (v == best_v && i < best_i)) {
                best_v = v;
                best_i = i;
            }
        }
        red_val[tid] = best_v;
        red_idx[tid] = best_i;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree reduction over the TG partial maxima.
        for (uint s = TG / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float ov = red_val[tid + s];
                uint  oi = red_idx[tid + s];
                float cv = red_val[tid];
                uint  ci = red_idx[tid];
                if (ov > cv || (ov == cv && oi < ci)) {
                    red_val[tid] = ov;
                    red_idx[tid] = oi;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            kept_idx[r] = red_idx[0];
            kept_val[r] = red_val[0];
            vals[red_idx[0]] = -INFINITY;  // mask the winner for next round
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. Softmax over the K kept logits + write outputs (thread 0).
    if (tid == 0) {
        float m = -INFINITY;
        for (uint r = 0; r < K; ++r) m = max(m, kept_val[r]);
        float sum = 0.0f;
        for (uint r = 0; r < K; ++r) sum += exp(kept_val[r] - m);
        auto out_idx = indices + (ulong)row * K;
        auto out_w   = weights + (ulong)row * K;
        for (uint r = 0; r < K; ++r) {
            out_idx[r] = kept_idx[r];
            out_w[r]   = static_cast<InT>(exp(kept_val[r] - m) / sum);
        }
    }
"#;

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;
    use mlx_rs::ops::indexing::{take_along_axis, IndexOp};
    use mlx_rs::ops::{argpartition_axis, softmax_axis, sum_axis};
    use mlx_rs::random::uniform;
    use mlx_rs::transforms::eval;

    /// Kernel top-k indices + weights must match the sort-based reference
    /// (`softmax → argpartition → take_along → renormalise`) on random
    /// logits, for the production E=256 / k=8 shape.
    #[test]
    fn router_topk_matches_sort_reference() {
        let (e, k) = (256, 8);
        let logits = uniform::<_, f32>(-4.0, 4.0, &[1, 1, e], None).unwrap();

        // Reference path.
        let probs = softmax_axis(&logits, -1, true).unwrap();
        let part = argpartition_axis(&probs, -k, -1).unwrap();
        let ref_idx = part.index((.., .., (e - k)..e));
        let ref_probs = take_along_axis(&probs, &ref_idx, -1).unwrap();
        let denom = sum_axis(&ref_probs, -1, true).unwrap();
        let ref_w = ref_probs.divide(&denom).unwrap();

        // Kernel path.
        let kernel = make_router_topk_kernel().unwrap();
        let (k_idx, k_w) = router_topk(&kernel, &logits, e, k).unwrap();
        eval([&k_idx, &k_w]).unwrap();

        // Sets of selected experts must match (order differs: reference is
        // partition-order, kernel is descending-value order). Compare the
        // selected logit *values* sorted, plus the summed weights.
        let ref_sel: Vec<u32> = {
            let mut v = ref_idx.as_slice::<u32>().to_vec();
            v.sort_unstable();
            v
        };
        let ker_sel: Vec<u32> = {
            let mut v = k_idx.as_slice::<u32>().to_vec();
            v.sort_unstable();
            v
        };
        assert_eq!(ref_sel, ker_sel, "selected expert set diverged");

        // Weights: gather both by expert id, compare per expert.
        let ref_idx_h = ref_idx.as_slice::<u32>().to_vec();
        let ref_w_h = ref_w.as_slice::<f32>().to_vec();
        let ker_idx_h = k_idx.as_slice::<u32>().to_vec();
        let ker_w_h = k_w.as_slice::<f32>().to_vec();
        let mut max_diff = 0.0f32;
        for (ki, kw) in ker_idx_h.iter().zip(&ker_w_h) {
            let rpos = ref_idx_h.iter().position(|r| r == ki).unwrap();
            max_diff = max_diff.max((kw - ref_w_h[rpos]).abs());
        }
        assert!(max_diff < 1e-5, "router top-k weights diverge: {max_diff}");
    }
}
