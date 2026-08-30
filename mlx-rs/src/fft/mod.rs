//! Fast Fourier Transform (FFT) and its inverse (IFFT) for one, two, and `N` dimensions.
//!
//! The legacy transform APIs provide checked, unchecked, and panicking variants, together with
//! per-operation stream variants. New APIs use one named `Result` path and follow the scoped or
//! current-thread default stream.
//!
//! The difference are explained below using `fftn` as an example:
//!
//! 1. `fftn_unchecked`/`fftn_device_unchecked`: This function is simply a wrapper around the C API
//!    and does not perform any checks on the input. It may panic or get an fatal error that cannot
//!    be caught by the rust runtime if the input is invalid.
//! 2. `try_fftn`/`try_fftn_device`: This function performs checks on the input and returns a
//!    `Result` instead of panicking.
//! 3. `fftn`/`fftn_device`: This function is a wrapper around `try_fftn` and unwraps the result. It
//!    panics if the input is invalid.
//!
//! The functions that contains `device` in their name are meant to be used with a user-specified
//! `StreamOrDevice`. If you don't care about the stream, you can use the functions without `device`
//! in their names. Please note that GPU device support is not yet implemented.
//!
//! # Examples
//!
//! ## One dimension
//!
//! ```rust
//! use mlx_rs::{
//!     Dtype, Array, StreamOrDevice, complex64, fft::*,
//!     test_utils::{assert_array_eq, tolerances},
//! };
//!
//! let src = [1.0f32, 2.0, 3.0, 4.0];
//! let mut array = Array::from_slice(&src[..], &[4]);
//!
//! let mut fft_result = fft(&array, 4, 0).unwrap();
//! assert_eq!(fft_result.dtype(), Dtype::Complex64);
//!
//! let expected = Array::from_slice(&[
//!     complex64::new(10.0, 0.0),
//!     complex64::new(-2.0, 2.0),
//!     complex64::new(-2.0, 0.0),
//!     complex64::new(-2.0, -2.0),
//! ], &[4]);
//! assert_array_eq(&fft_result, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
//!
//! let mut ifft_result = ifft(&fft_result, 4, 0).unwrap();
//! assert_eq!(ifft_result.dtype(), Dtype::Complex64);
//!
//! let expected = Array::from_slice(&[
//!    complex64::new(1.0, 0.0),
//!    complex64::new(2.0, 0.0),
//!    complex64::new(3.0, 0.0),
//!    complex64::new(4.0, 0.0),
//! ], &[4]);
//! assert_array_eq(ifft_result, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
//!
//! let mut rfft_result = rfft(&array, 4, 0).unwrap();
//! assert_eq!(rfft_result.dtype(), Dtype::Complex64);
//!
//! let expected = Array::from_slice(&[
//!    complex64::new(10.0, 0.0),
//!    complex64::new(-2.0, 2.0),
//!    complex64::new(-2.0, 0.0),
//! ], &[3]);
//! assert_array_eq(&rfft_result, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
//!
//! let mut irfft_result = irfft(&rfft_result, 4, 0).unwrap();
//! assert_eq!(irfft_result.dtype(), Dtype::Float32);
//! assert_array_eq(
//!     irfft_result,
//!     Array::from_slice(&src, &[4]),
//!     tolerances::EXACT.rtol,
//!     tolerances::EXACT.atol,
//! );
//!
//! // The original array is not modified
//! assert_array_eq(
//!     array,
//!     Array::from_slice(&src, &[4]),
//!     tolerances::EXACT.rtol,
//!     tolerances::EXACT.atol,
//! );
//! ```
//!
//! ## Two dimensions
//!
//! ```rust
//! use mlx_rs::{
//!     Dtype, Array, StreamOrDevice, complex64, fft::*,
//!     test_utils::{assert_array_eq, tolerances},
//! };
//!
//! let src = [1.0f32, 1.0, 1.0, 1.0];
//! let mut array = Array::from_slice(&src[..], &[2, 2]);
//!
//! let mut fft2_result = fft2(&array, None, None).unwrap();
//! assert_eq!(fft2_result.dtype(), Dtype::Complex64);
//! let expected = Array::from_slice(&[
//!     complex64::new(4.0, 0.0),
//!     complex64::new(0.0, 0.0),
//!     complex64::new(0.0, 0.0),
//!     complex64::new(0.0, 0.0),
//! ], &[2, 2]);
//! assert_array_eq(&fft2_result, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
//!
//! let mut ifft2_result = ifft2(&fft2_result, None, None).unwrap();
//! assert_eq!(ifft2_result.dtype(), Dtype::Complex64);
//!
//! let expected = Array::from_slice(&[
//!    complex64::new(1.0, 0.0),
//!    complex64::new(1.0, 0.0),
//!    complex64::new(1.0, 0.0),
//!    complex64::new(1.0, 0.0),
//! ], &[2, 2]);
//! assert_array_eq(ifft2_result, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
//!
//! let mut rfft2_result = rfft2(&array, None, None).unwrap();
//! assert_eq!(rfft2_result.dtype(), Dtype::Complex64);
//!
//! let expected = Array::from_slice(&[
//!     complex64::new(4.0, 0.0),
//!     complex64::new(0.0, 0.0),
//!     complex64::new(0.0, 0.0),
//!     complex64::new(0.0, 0.0),
//! ], &[2, 2]);
//! assert_array_eq(&rfft2_result, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
//!
//! let mut irfft2_result = irfft2(&rfft2_result, None, None).unwrap();
//! assert_eq!(irfft2_result.dtype(), Dtype::Float32);
//! assert_array_eq(
//!     irfft2_result,
//!     Array::from_slice(&src, &[2, 2]),
//!     tolerances::EXACT.rtol,
//!     tolerances::EXACT.atol,
//! );
//!
//! // The original array is not modified
//! assert_array_eq(
//!     array,
//!     Array::from_slice(&src, &[2, 2]),
//!     tolerances::EXACT.rtol,
//!     tolerances::EXACT.atol,
//! );
//! ```
//!
//! ## `N` dimensions
//!
//! ```rust
//! use mlx_rs::{
//!     Dtype, Array, StreamOrDevice, complex64, fft::*,
//!     test_utils::{assert_array_eq, tolerances},
//! };
//!
//! let mut array = Array::ones::<f32>(&[2, 2, 2]).unwrap();
//! let mut fftn_result = fftn(&array, None, None).unwrap();
//! assert_eq!(fftn_result.dtype(), Dtype::Complex64);
//!
//! let mut expected = [complex64::new(0.0, 0.0); 8];
//! expected[0] = complex64::new(8.0, 0.0);
//! assert_array_eq(
//!     &fftn_result,
//!     Array::from_slice(&expected, &[2, 2, 2]),
//!     tolerances::EXACT.rtol,
//!     tolerances::EXACT.atol,
//! );
//!
//! let mut ifftn_result = ifftn(&fftn_result, None, None).unwrap();
//! assert_eq!(ifftn_result.dtype(), Dtype::Complex64);
//!
//! let expected = [complex64::new(1.0, 0.0); 8];
//! assert_array_eq(
//!     ifftn_result,
//!     Array::from_slice(&expected, &[2, 2, 2]),
//!     tolerances::EXACT.rtol,
//!     tolerances::EXACT.atol,
//! );
//!
//! let mut rfftn_result = rfftn(&array, None, None).unwrap();
//! assert_eq!(rfftn_result.dtype(), Dtype::Complex64);
//!
//! let mut expected = [complex64::new(0.0, 0.0); 8];
//! expected[0] = complex64::new(8.0, 0.0);
//! assert_array_eq(
//!     &rfftn_result,
//!     Array::from_slice(&expected, &[2, 2, 2]),
//!     tolerances::EXACT.rtol,
//!     tolerances::EXACT.atol,
//! );
//!
//! let mut irfftn_result = irfftn(&rfftn_result, None, None).unwrap();
//! assert_eq!(irfftn_result.dtype(), Dtype::Float32);
//!
//! let expected = [1.0; 8];
//! assert_array_eq(
//!     irfftn_result,
//!     Array::from_slice(&expected, &[2, 2, 2]),
//!     tolerances::EXACT.rtol,
//!     tolerances::EXACT.atol,
//! );
//!
//! // The original array is not modified
//! assert_array_eq(
//!     array,
//!     Array::from_slice(&[1.0; 8], &[2, 2, 2]),
//!     tolerances::EXACT.rtol,
//!     tolerances::EXACT.atol,
//! );
//! ```

mod fftn;
mod frequencies;
mod rfftn;
mod shift;
mod utils;

pub use self::{fftn::*, frequencies::*, rfftn::*, shift::*};

/* -------------------------------------------------------------------------- */
/*                              Helper functions                              */
/* -------------------------------------------------------------------------- */
