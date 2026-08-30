//! Window functions for signal processing.

use crate::error::{Exception, Result};
use crate::utils::guard::Guarded;
use crate::{Array, Stream};

fn checked_size(size: usize) -> Result<i32> {
    i32::try_from(size).map_err(|_| Exception::custom("window size exceeds i32::MAX"))
}

/// Returns a Bartlett window of `size` samples.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{ops::windows::bartlett, Device, Dtype};
///
/// Device::set_default(&Device::cpu());
/// let window = bartlett(7).unwrap();
/// assert_eq!(window.shape(), &[7]);
/// assert_eq!(window.dtype(), Dtype::Float32);
/// ```
pub fn bartlett(size: usize) -> Result<Array> {
    let size = checked_size(size)?;
    let stream = Stream::task_local_or_default();
    Array::try_from_op(|res| unsafe { mlx_sys::mlx_bartlett(res, size, stream.as_ptr()) })
}

/// Returns a Blackman window of `size` samples.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{ops::windows::blackman, Device, Dtype};
///
/// Device::set_default(&Device::cpu());
/// let window = blackman(7).unwrap();
/// assert_eq!(window.shape(), &[7]);
/// assert_eq!(window.dtype(), Dtype::Float32);
/// ```
pub fn blackman(size: usize) -> Result<Array> {
    let size = checked_size(size)?;
    let stream = Stream::task_local_or_default();
    Array::try_from_op(|res| unsafe { mlx_sys::mlx_blackman(res, size, stream.as_ptr()) })
}

/// Returns a Hamming window of `size` samples.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{ops::windows::hamming, Device, Dtype};
///
/// Device::set_default(&Device::cpu());
/// let window = hamming(7).unwrap();
/// assert_eq!(window.shape(), &[7]);
/// assert_eq!(window.dtype(), Dtype::Float32);
/// ```
pub fn hamming(size: usize) -> Result<Array> {
    let size = checked_size(size)?;
    let stream = Stream::task_local_or_default();
    Array::try_from_op(|res| unsafe { mlx_sys::mlx_hamming(res, size, stream.as_ptr()) })
}

/// Returns a Hann window of `size` samples.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{ops::windows::hann, Device, Dtype};
///
/// Device::set_default(&Device::cpu());
/// let window = hann(7).unwrap();
/// assert_eq!(window.shape(), &[7]);
/// assert_eq!(window.dtype(), Dtype::Float32);
/// ```
pub fn hann(size: usize) -> Result<Array> {
    let size = checked_size(size)?;
    let stream = Stream::task_local_or_default();
    Array::try_from_op(|res| unsafe { mlx_sys::mlx_hanning(res, size, stream.as_ptr()) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_sizes_above_i32_max() {
        assert!(bartlett(usize::MAX).is_err());
        assert!(blackman(usize::MAX).is_err());
        assert!(hamming(usize::MAX).is_err());
        assert!(hann(usize::MAX).is_err());
    }
}
