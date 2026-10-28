//! Process-global MLX allocator controls and observations.
//!
//! The state exposed here affects every thread and every evaluation in the process. These
//! functions do not synchronize a sequence of reads and writes for callers.

use crate::{error::Result, utils::guard::Guarded};

/// Clear cached allocator buffers.
///
/// This affects all threads and evaluations. It is unrelated to
/// [`transforms::compile::clear_cache`](crate::transforms::compile::clear_cache), which clears
/// compiled functions.
pub fn clear_cache() -> Result<()> {
    <()>::try_from_op(|_| unsafe { mlx_sys::mlx_clear_cache() })
}

/// Return the number of bytes in active use by MLX.
pub fn active_memory() -> Result<usize> {
    usize::try_from_op(|res| unsafe { mlx_sys::mlx_get_active_memory(res) })
}

/// Return the number of bytes held in MLX's allocator cache.
pub fn cache_memory() -> Result<usize> {
    usize::try_from_op(|res| unsafe { mlx_sys::mlx_get_cache_memory(res) })
}

/// Return the peak number of bytes used by MLX.
pub fn peak_memory() -> Result<usize> {
    usize::try_from_op(|res| unsafe { mlx_sys::mlx_get_peak_memory(res) })
}

/// Reset the process-global peak-memory counter for all threads and evaluations.
pub fn reset_peak_memory() -> Result<()> {
    <()>::try_from_op(|_| unsafe { mlx_sys::mlx_reset_peak_memory() })
}

/// Return the process-global memory limit in bytes.
pub fn memory_limit() -> Result<usize> {
    usize::try_from_op(|res| unsafe { mlx_sys::mlx_get_memory_limit(res) })
}

/// Set the process-global memory limit in bytes and return its previous value.
///
/// The new limit affects all threads and evaluations.
pub fn set_memory_limit(limit: usize) -> Result<usize> {
    usize::try_from_op(|res| unsafe { mlx_sys::mlx_set_memory_limit(res, limit) })
}

/// Set the process-global allocator-cache limit in bytes and return its previous value.
///
/// The new limit affects all threads and evaluations.
pub fn set_cache_limit(limit: usize) -> Result<usize> {
    usize::try_from_op(|res| unsafe { mlx_sys::mlx_set_cache_limit(res, limit) })
}

/// Set the process-global wired-memory limit in bytes and return its previous value.
///
/// The new limit affects all threads and evaluations. Wired-memory limits are useful only with
/// Metal on macOS 15 or later; other configurations may return an upstream error.
pub fn set_wired_limit(limit: usize) -> Result<usize> {
    usize::try_from_op(|res| unsafe { mlx_sys::mlx_set_wired_limit(res, limit) })
}
