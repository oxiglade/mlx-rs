use crate::error::{Exception, Result};
use crate::utils::guard::Guarded;
use crate::{Array, Stream};

fn checked_length(n: usize) -> Result<i32> {
    i32::try_from(n).map_err(|_| Exception::custom("FFT length exceeds i32::MAX"))
}

/// Returns the discrete Fourier transform sample frequencies for a transform of length `n`.
///
/// The sample spacing is `d`.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{fft::fftfreq, Device, Dtype};
///
/// Device::set_default(&Device::cpu());
/// let frequencies = fftfreq(4, 1.0).unwrap();
/// assert_eq!(frequencies.shape(), &[4]);
/// assert_eq!(frequencies.dtype(), Dtype::Float32);
/// ```
pub fn fftfreq(n: usize, d: f64) -> Result<Array> {
    let n = checked_length(n)?;
    let stream = Stream::task_local_or_default();
    Array::try_from_op(|res| unsafe { mlx_sys::mlx_fft_fftfreq(res, n, d, stream.as_ptr()) })
}

/// Returns the nonnegative discrete Fourier transform sample frequencies for a real transform.
///
/// The transform length is `n` and the sample spacing is `d`.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{fft::rfftfreq, Device, Dtype};
///
/// Device::set_default(&Device::cpu());
/// let frequencies = rfftfreq(4, 0.5).unwrap();
/// assert_eq!(frequencies.shape(), &[3]);
/// assert_eq!(frequencies.dtype(), Dtype::Float32);
/// ```
pub fn rfftfreq(n: usize, d: f64) -> Result<Array> {
    let n = checked_length(n)?;
    let stream = Stream::task_local_or_default();
    Array::try_from_op(|res| unsafe { mlx_sys::mlx_fft_rfftfreq(res, n, d, stream.as_ptr()) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_lengths_above_i32_max() {
        assert!(fftfreq(usize::MAX, 1.0).is_err());
        assert!(rfftfreq(usize::MAX, 1.0).is_err());
    }
}
