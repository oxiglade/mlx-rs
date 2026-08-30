//! Fast Fourier Transform (FFT) and its inverse (IFFT) for one, two, and `N` dimensions.
//!
//! Unsuffixed functions are the canonical named `Result` paths and use the current thread's scoped
//! stream or runtime default. Select an explicit stream or device with [`crate::with_stream`] or
//! [`crate::with_device`]. The `_device` functions and generated operation macros are deprecated
//! forwarding shims.
//!
//! N-dimensional transforms accept [`FftnOptions`]. Its default selects every axis at the input
//! lengths, and correlated lengths and axes are validated before calling MLX.
//!
//! # Examples
//!
//! ## One dimension
//!
//! ```rust
//! use mlx_rs::{
//!     Dtype, Array, complex64, fft::*,
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
//!     Dtype, Array, complex64, fft::*,
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
//!     Dtype, Array, complex64, fft::*,
//!     test_utils::{assert_array_eq, tolerances},
//! };
//!
//! let mut array = Array::ones::<f32>(&[2, 2, 2]).unwrap();
//! let mut fftn_result = fftn(&array, FftnOptions::default()).unwrap();
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
//! let mut ifftn_result = ifftn(&fftn_result, FftnOptions::default()).unwrap();
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
//! let mut rfftn_result = rfftn(&array, FftnOptions::default()).unwrap();
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
//! let mut irfftn_result = irfftn(&rfftn_result, FftnOptions::default()).unwrap();
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
mod options;
mod rfftn;
mod shift;
mod utils;

pub use self::{fftn::*, frequencies::*, options::*, rfftn::*, shift::*};

/* -------------------------------------------------------------------------- */
/*                              Helper functions                              */
/* -------------------------------------------------------------------------- */
