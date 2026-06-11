//! [`RotatingKVCache`]: sliding-window KV cache. Backs Gemma 3/4
//! sliding-attention layers.
//!
//! 2× ring buffer over the rotating region: steady-state decode is one
//! `try_index_mut` write + one contiguous `index` view return, instead of
//! a per-step `concatenate_axis` of 3 slices × 2 (K and V) on each wrapped
//! decode step.
//!
//! Layout: `[B, H, keep + 2 * window, D]` where `window = max_size - keep`.
//!
//! - Slots `[0, keep)` are write-once head tokens (the `keep` prefix).
//! - Slots `[keep, keep + 2 * window)` are the rotating ring. Logical
//!   position `t` (post-keep) is at physical slot `keep + (t % (2 * window))`.
//! - When `write_head` (post-keep token count) reaches `2 * window`,
//!   compact: copy the last `window` rotating slots to the first `window`
//!   rotating slots, reset `write_head = window`. O(window) per compaction,
//!   amortised O(1) per token.
//!
//! At read time the logical window is the last `min(write_head, window)`
//! rotating tokens — always a single contiguous slice, no concat needed.

use mlx_rs::{
    error::Exception,
    ops::{
        concatenate_axis,
        indexing::{Ellipsis, IndexOp, TryIndexMutOp},
        zeros_dtype,
    },
    Array,
};

use super::trait_def::KeyValueCache;

/// Sliding-window KV cache. The oldest non-keep slot is overwritten on
/// append once the window is full. Implemented as a 2× ring buffer so the
/// decode-step cost is O(1) Array ops instead of a per-step concat.
#[derive(Debug, Clone)]
pub struct RotatingKVCache {
    keys: Option<Array>,
    values: Option<Array>,
    /// Real token count seen so far (monotonic; not bounded by `max_size`).
    offset: i32,
    /// Token count of writes into the rotating region (i.e. tokens past
    /// the `keep` prefix). Resets to `window` on each compaction.
    /// `0 <= write_head < 2 * window` between compactions.
    write_head: i32,
    /// Sliding-window capacity in tokens (`keep + window` ≤ `max_size`).
    max_size: i32,
    /// Number of head tokens pinned in the first `keep` slots.
    keep: i32,
}

impl RotatingKVCache {
    /// New empty cache. `max_size` is the sliding-window capacity in
    /// tokens; `keep` is the number of leading tokens never overwritten
    /// (must satisfy `0 <= keep < max_size`).
    pub fn new(max_size: i32, keep: i32) -> Self {
        assert!(max_size > 0, "max_size must be positive");
        assert!(
            keep >= 0 && keep < max_size,
            "keep must be in [0, max_size)"
        );
        Self {
            keys: None,
            values: None,
            offset: 0,
            write_head: 0,
            max_size,
            keep,
        }
    }

    /// Trim is only well-defined while the rotating region hasn't wrapped.
    pub fn is_trimmable(&self) -> bool {
        self.offset <= self.max_size
    }

    /// Drop the last `n` tokens (no-op once the ring has wrapped); returns
    /// the number actually trimmed.
    pub fn trim(&mut self, n: i32) -> i32 {
        if !self.is_trimmable() {
            return 0;
        }
        let trimmed = n.min(self.offset).max(0);
        self.offset -= trimmed;
        self.write_head = (self.offset - self.keep).max(0);
        trimmed
    }

    #[inline]
    fn window(&self) -> i32 {
        self.max_size - self.keep
    }

    /// Physical capacity: keep prefix + 2× rotating window.
    #[inline]
    fn physical_capacity(&self) -> i32 {
        self.keep + 2 * self.window()
    }

    fn alloc_like(template: &Array, capacity: i32) -> Result<Array, Exception> {
        let mut buf_shape = template.shape().to_vec();
        let t_axis = buf_shape.len() - 2;
        buf_shape[t_axis] = capacity;
        zeros_dtype(&buf_shape, template.dtype())
    }

    /// Snapshot the current logical window (keep prefix + rotating region
    /// in temporal order). Used by the prefill path so attention sees
    /// `old_window ++ new` without writing the new tokens through the ring
    /// first.
    fn snapshot_window(&self) -> Result<(Array, Array), Exception> {
        let buf_k = self.keys.as_ref().expect("snapshot: buffer exists");
        let buf_v = self.values.as_ref().expect("snapshot: buffer exists");
        let keep = self.keep;
        let visible_rot = self.write_head.min(self.window());
        let rot_start = keep + self.write_head - visible_rot;
        let rot_end = keep + self.write_head;
        let keep_filled = self.offset.min(keep);
        let rot_k = buf_k.index((Ellipsis, rot_start..rot_end, ..));
        let rot_v = buf_v.index((Ellipsis, rot_start..rot_end, ..));
        if keep_filled == 0 {
            Ok((rot_k, rot_v))
        } else {
            let head_k = buf_k.index((Ellipsis, 0..keep_filled, ..));
            let head_v = buf_v.index((Ellipsis, 0..keep_filled, ..));
            Ok((
                concatenate_axis(&[head_k, rot_k], -2)?,
                concatenate_axis(&[head_v, rot_v], -2)?,
            ))
        }
    }

    /// Write one token (`S == 1` slice) into the ring buffer. Shared by the
    /// decode and prefill paths so eviction semantics stay identical.
    fn write_one(&mut self, token_k: Array, token_v: Array) -> Result<(), Exception> {
        let keep = self.keep;
        let slot = if self.offset < keep {
            self.offset
        } else {
            if self.write_head >= 2 * self.window() {
                self.compact()?;
            }
            let slot = keep + self.write_head;
            self.write_head += 1;
            slot
        };
        let buf_k = self.keys.as_mut().expect("write_one: buffer exists");
        let buf_v = self.values.as_mut().expect("write_one: buffer exists");
        buf_k.try_index_mut((Ellipsis, slot..slot + 1, ..), token_k)?;
        buf_v.try_index_mut((Ellipsis, slot..slot + 1, ..), token_v)?;
        self.offset += 1;
        Ok(())
    }

    /// Copy the last `window` rotating slots to the first `window` rotating
    /// slots, then reset `write_head = window`. Called when `write_head`
    /// would otherwise exceed `2 * window`.
    fn compact(&mut self) -> Result<(), Exception> {
        let window = self.window();
        let keep = self.keep;
        let buf_k = self.keys.as_mut().expect("compact: buffer exists");
        let buf_v = self.values.as_mut().expect("compact: buffer exists");
        // Source: [keep + window, keep + 2*window). Dest: [keep, keep + window).
        let src_k = buf_k.index((Ellipsis, keep + window..keep + 2 * window, ..));
        let src_v = buf_v.index((Ellipsis, keep + window..keep + 2 * window, ..));
        buf_k.try_index_mut((Ellipsis, keep..keep + window, ..), src_k)?;
        buf_v.try_index_mut((Ellipsis, keep..keep + window, ..), src_v)?;
        self.write_head = window;
        Ok(())
    }
}

impl KeyValueCache for RotatingKVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        Some(self.max_size)
    }

    fn current_kv(&self) -> Result<Option<(Array, Array)>, Exception> {
        if self.offset == 0 || self.keys.is_none() {
            return Ok(None);
        }
        Ok(Some(self.snapshot_window()?))
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let key_shape = keys.shape();
        let s = key_shape[key_shape.len() - 2];
        let keep = self.keep;
        let window = self.window();

        // Allocate the 2× ring buffer on first append.
        if self.keys.is_none() {
            let cap = self.physical_capacity();
            self.keys = Some(Self::alloc_like(&keys, cap)?);
            self.values = Some(Self::alloc_like(&values, cap)?);
        }

        // Prefill (`S > 1`) after the cache already holds context: return
        // `pre_window ++ new` so every new token can attend to the full
        // sliding window of past context. Total cols =
        // `min(prev_offset, max_size) + S`, matching the sliding mask.
        if s > 1 && self.offset > 0 {
            let (old_k, old_v) = self.snapshot_window()?;
            for i in 0..s {
                let token_k = keys.index((Ellipsis, i..i + 1, ..));
                let token_v = values.index((Ellipsis, i..i + 1, ..));
                self.write_one(token_k, token_v)?;
            }
            return Ok((
                concatenate_axis(&[old_k, keys], -2)?,
                concatenate_axis(&[old_v, values], -2)?,
            ));
        }

        // Per-token write loop (single iteration on the S=1 decode path).
        for i in 0..s {
            let token_k = keys.index((Ellipsis, i..i + 1, ..));
            let token_v = values.index((Ellipsis, i..i + 1, ..));
            self.write_one(token_k, token_v)?;
        }

        // Return the populated buffer in temporal order: a single
        // contiguous rotating slice, plus the keep prefix when `keep > 0`.
        let buf_k = self.keys.as_ref().expect("alloc'd above");
        let buf_v = self.values.as_ref().expect("alloc'd above");
        let visible_rot = self.write_head.min(window);
        let rot_start = keep + self.write_head - visible_rot;
        let rot_end = keep + self.write_head;

        if keep == 0 {
            Ok((
                buf_k.index((Ellipsis, rot_start..rot_end, ..)),
                buf_v.index((Ellipsis, rot_start..rot_end, ..)),
            ))
        } else {
            let keep_filled = self.offset.min(keep);
            let head_k = buf_k.index((Ellipsis, 0..keep_filled, ..));
            let head_v = buf_v.index((Ellipsis, 0..keep_filled, ..));
            let rot_k = buf_k.index((Ellipsis, rot_start..rot_end, ..));
            let rot_v = buf_v.index((Ellipsis, rot_start..rot_end, ..));
            Ok((
                concatenate_axis(&[head_k, rot_k], -2)?,
                concatenate_axis(&[head_v, rot_v], -2)?,
            ))
        }
    }

    // `attention` uses the `KeyValueCache` default (update_and_fetch + SDPA);
    // the sliding-window restriction comes from the caller's mask plus the
    // bounded buffer this cache returns.
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;
    use mlx_rs::ops::arange;
    use mlx_rs::transforms::eval;

    const B: i32 = 1;
    const H: i32 = 1;
    const D: i32 = 4;

    /// One token `[B, H, 1, D]` whose values are all `t` (so positions are
    /// distinguishable when inspecting the returned window).
    fn token(t: i32) -> Array {
        let base = arange::<_, f32>(0.0, (D) as f32, None).unwrap();
        let val = base.add(Array::from_f32(t as f32 * 100.0)).unwrap();
        val.reshape(&[B, H, 1, D]).unwrap()
    }

    fn push(c: &mut RotatingKVCache, t: i32) -> (Array, Array) {
        let tk = token(t);
        let (k, v) = c.update_and_fetch(tk.clone(), tk).unwrap();
        eval([&k, &v]).unwrap();
        (k, v)
    }

    fn seq_len(a: &Array) -> i32 {
        let s = a.shape();
        s[s.len() - 2]
    }

    #[test]
    fn decode_window_caps_at_window_size() {
        let window = 4;
        let mut c = RotatingKVCache::new(window, 0);
        // Push well past 2*window to exercise compaction + wrap.
        for t in 0..(2 * window + 3) {
            let (k, _) = push(&mut c, t);
            let expected = (t + 1).min(window);
            assert_eq!(seq_len(&k), expected, "token {t}: window len");
        }
        assert_eq!(c.offset(), 2 * window + 3);
    }

    #[test]
    fn compaction_resets_write_head_to_window() {
        let window = 3;
        let mut c = RotatingKVCache::new(window, 0);
        for t in 0..(2 * window) {
            push(&mut c, t);
        }
        // write_head now == 2*window; the next write compacts first.
        push(&mut c, 2 * window);
        assert_eq!(c.write_head, window + 1);
    }

    #[test]
    fn offset_matches_tokens_seen() {
        let mut c = RotatingKVCache::new(8, 2);
        for t in 0..20 {
            push(&mut c, t);
        }
        assert_eq!(c.offset(), 20);
    }

    #[test]
    fn keep_prefix_is_pinned_in_window() {
        let window_cap = 5;
        let keep = 2;
        let mut c = RotatingKVCache::new(window_cap, keep);
        for t in 0..15 {
            let (k, _) = push(&mut c, t);
            eval([&k]).unwrap();
            // Window never exceeds max_size; once full it stays max_size.
            assert!(seq_len(&k) <= window_cap);
        }
        // After many tokens the returned window = keep prefix + rotating
        // tail, capped at max_size.
        let (k, _) = push(&mut c, 15);
        assert_eq!(seq_len(&k), window_cap);
    }

    #[test]
    fn prefill_returns_old_window_plus_new() {
        let window = 4;
        let mut c = RotatingKVCache::new(window, 0);
        // Seed 3 decode tokens.
        for t in 0..3 {
            push(&mut c, t);
        }
        // Prefill 2 tokens at once: returned width = min(offset, max) + S.
        let k0 = token(100);
        let k1 = token(101);
        let block = concatenate_axis(&[k0, k1], -2).unwrap();
        let (k, _) = c.update_and_fetch(block.clone(), block).unwrap();
        eval([&k]).unwrap();
        assert_eq!(seq_len(&k), 3 + 2);
        assert_eq!(c.offset(), 5);
    }

    #[test]
    fn trim_decrements_offset_while_unwrapped() {
        let mut c = RotatingKVCache::new(16, 0);
        for t in 0..5 {
            push(&mut c, t);
        }
        assert!(c.is_trimmable());
        let trimmed = c.trim(2);
        assert_eq!(trimmed, 2);
        assert_eq!(c.offset(), 3);
    }
}
