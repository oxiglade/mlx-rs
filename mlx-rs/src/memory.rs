//! MLX allocator introspection and control.
//!
//! Wraps `mlx_get_*_memory` / `mlx_set_*_limit` / `mlx_clear_cache` /
//! `mlx_reset_peak_memory` from mlx-c. All counters and limits are in
//! bytes.

/// Bytes currently allocated to live MLX arrays (excludes the reuse pool).
pub fn active_memory() -> usize {
    let mut n: usize = 0;
    unsafe {
        mlx_sys::mlx_get_active_memory(&mut n);
    }
    n
}

/// Bytes held in the buffer-reuse pool (not freed back to the driver).
pub fn cache_memory() -> usize {
    let mut n: usize = 0;
    unsafe {
        mlx_sys::mlx_get_cache_memory(&mut n);
    }
    n
}

/// Watermark of `active_memory` since process start or last `reset_peak`.
pub fn peak_memory() -> usize {
    let mut n: usize = 0;
    unsafe {
        mlx_sys::mlx_get_peak_memory(&mut n);
    }
    n
}

/// Reset the `peak_memory` watermark to current `active_memory`.
pub fn reset_peak_memory() {
    unsafe {
        mlx_sys::mlx_reset_peak_memory();
    }
}

/// Drain the buffer-reuse pool; next allocation re-fetches from driver.
pub fn clear_cache() {
    unsafe {
        mlx_sys::mlx_clear_cache();
    }
}

/// Cap retained reuse pool. `0` disables reuse (every buffer round-trips
/// the driver). Returns the previous limit.
pub fn set_cache_limit(limit: usize) -> usize {
    let mut prev: usize = 0;
    unsafe {
        mlx_sys::mlx_set_cache_limit(&mut prev, limit);
    }
    prev
}

/// Hard cap on total active MLX allocation. Allocations beyond the limit
/// will wait/swap/error per mlx-core policy. Returns previous limit.
pub fn set_memory_limit(limit: usize) -> usize {
    let mut prev: usize = 0;
    unsafe {
        mlx_sys::mlx_set_memory_limit(&mut prev, limit);
    }
    prev
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_counters_monotonic() {
        let _a = active_memory();
        let _c = cache_memory();
        let p0 = peak_memory();
        let _ = crate::Array::from(&[1.0_f32, 2.0, 3.0][..]);
        assert!(peak_memory() >= p0);
    }

    #[test]
    fn smoke_set_cache_limit_roundtrip() {
        let prev = set_cache_limit(1 << 20);
        let restored = set_cache_limit(prev);
        assert_eq!(restored, 1 << 20);
    }
}
