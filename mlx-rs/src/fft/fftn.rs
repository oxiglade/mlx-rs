use mlx_internal_macros::{default_device, generate_macro};

use crate::{
    array::Array,
    error::Result,
    utils::{guard::Guarded, IntoOption},
    Stream,
};

use super::utils::{resolve_size_and_axis_unchecked, resolve_sizes_and_axes_unchecked};

/// `MLX_FFT_NORM_BACKWARD` from `mlx_fft_norm` — the historical default
/// for `mlx_fft_*` (mlx ≤ 0.30 had no `norm` arg). Keeps Rust API
/// surface stable across the mlx 0.31 binding bump.
const DEFAULT_NORM: u32 = mlx_sys::mlx_fft_norm__MLX_FFT_NORM_BACKWARD;

/// One dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n`. The default value is `a.shape[axis]`.
/// - `axis`: Axis along which to perform the FFT. The default is -1.
#[generate_macro(customize(root = "$crate::fft"))]
#[default_device]
pub fn fft_device(
    a: impl AsRef<Array>,
    #[optional] n: impl Into<Option<i32>>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let (n, axis) = resolve_size_and_axis_unchecked(a, n.into(), axis.into());
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_fft(
            res,
            a.as_ptr(),
            n,
            axis,
            DEFAULT_NORM,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Two dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Size of the transformed axes. The corresponding axes in the input are truncated or padded
/// with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
#[generate_macro(customize(root = "$crate::fft"))]
#[default_device]
pub fn fft2_device<'a>(
    a: impl AsRef<Array>,
    #[optional] s: impl IntoOption<&'a [i32]>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let axes = axes.into_option().unwrap_or(&[-2, -1]);
    let (s, axes) = resolve_sizes_and_axes_unchecked(a, s.into_option(), Some(axes));

    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_fft2(
            res,
            a.as_ptr(),
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            DEFAULT_NORM,
            stream.as_ref().as_ptr(),
        )
    })
}

/// n-dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
/// padded with zeros to match the sizes in `s`. The default value is the sizes of `a` along `axes`
/// if not specified.
/// - `axes`: Axes along which to perform the FFT. The default is `None` in which case the FFT is
/// over the last `len(s)` axes are or all axes if `s` is also `None`.
#[generate_macro(customize(root = "$crate::fft"))]
#[default_device]
pub fn fftn_device<'a>(
    a: impl AsRef<Array>,
    #[optional] s: impl IntoOption<&'a [i32]>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let (s, axes) = resolve_sizes_and_axes_unchecked(a, s.into_option(), axes.into_option());
    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_fftn(
            res,
            a.as_ptr(),
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            DEFAULT_NORM,
            stream.as_ref().as_ptr(),
        )
    })
}

/// One dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: Input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///  with zeros to match `n`. The default value is `a.shape[axis]` if not specified.
/// - `axis`: Axis along which to perform the FFT. The default is `-1` if not specified.
#[generate_macro(customize(root = "$crate::fft"))]
#[default_device]
pub fn ifft_device(
    a: impl AsRef<Array>,
    #[optional] n: impl Into<Option<i32>>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let (n, axis) = resolve_size_and_axis_unchecked(a, n.into(), axis.into());

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_ifft(
            res,
            a.as_ptr(),
            n,
            axis,
            DEFAULT_NORM,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Two dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Size of the transformed axes. The corresponding axes in the input are truncated or padded
/// with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
#[generate_macro(customize(root = "$crate::fft"))]
#[default_device]
pub fn ifft2_device<'a>(
    a: impl AsRef<Array>,
    #[optional] s: impl IntoOption<&'a [i32]>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let axes = axes.into_option().unwrap_or(&[-2, -1]);
    let (s, axes) = resolve_sizes_and_axes_unchecked(a, s.into_option(), Some(axes));

    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_ifft2(
            res,
            a.as_ptr(),
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            DEFAULT_NORM,
            stream.as_ref().as_ptr(),
        )
    })
}

/// n-dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Sizes of the transformed axes. The corresponding axes in the input are truncated or
/// padded with zeros to match the sizes in `s`. The default value is the sizes of `a` along `axes`
/// if not specified.
/// - `axes`: Axes along which to perform the FFT. The default is `None` in which case the FFT is
/// over the last `len(s)` axes are or all axes if `s` is also `None`.
#[generate_macro(customize(root = "$crate::fft"))]
#[default_device]
pub fn ifftn_device<'a>(
    a: impl AsRef<Array>,
    #[optional] s: impl IntoOption<&'a [i32]>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let (s, axes) = resolve_sizes_and_axes_unchecked(a, s.into_option(), axes.into_option());
    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_ifftn(
            res,
            a.as_ptr(),
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            DEFAULT_NORM,
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use crate::{complex64, fft::*, Array, Dtype};

    #[test]
    fn test_fft() {
        const FFT_DATA: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        const FFT_SHAPE: &[i32] = &[4];
        const FFT_EXPECTED: &[complex64; 4] = &[
            complex64::new(10.0, 0.0),
            complex64::new(-2.0, 2.0),
            complex64::new(-2.0, 0.0),
            complex64::new(-2.0, -2.0),
        ];

        let array = Array::from_slice(FFT_DATA, FFT_SHAPE);
        let fft = fft(&array, None, None).unwrap();

        assert_eq!(fft.dtype(), Dtype::Complex64);
        assert_eq!(fft.as_slice::<complex64>(), FFT_EXPECTED);

        let ifft = ifft(&fft, None, None).unwrap();

        assert_eq!(ifft.dtype(), Dtype::Complex64);
        assert_eq!(
            ifft.as_slice::<complex64>(),
            FFT_DATA
                .iter()
                .map(|&x| complex64::new(x, 0.0))
                .collect::<Vec<_>>()
        );

        // The original array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, FFT_DATA);
    }

    #[test]
    fn test_fft2() {
        const FFT2_DATA: &[f32] = &[1.0, 1.0, 1.0, 1.0];
        const FFT2_SHAPE: &[i32] = &[2, 2];
        const FFT2_EXPECTED: &[complex64; 4] = &[
            complex64::new(4.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
        ];

        let array = Array::from_slice(FFT2_DATA, FFT2_SHAPE);
        let fft2 = fft2(&array, None, None).unwrap();

        assert_eq!(fft2.dtype(), Dtype::Complex64);
        assert_eq!(fft2.as_slice::<complex64>(), FFT2_EXPECTED);

        let ifft2 = ifft2(&fft2, None, None).unwrap();

        assert_eq!(ifft2.dtype(), Dtype::Complex64);
        assert_eq!(
            ifft2.as_slice::<complex64>(),
            FFT2_DATA
                .iter()
                .map(|&x| complex64::new(x, 0.0))
                .collect::<Vec<_>>()
        );

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, FFT2_DATA);
    }

    #[test]
    fn test_fftn() {
        const FFTN_DATA: &[f32] = &[1.0; 8];
        const FFTN_SHAPE: &[i32] = &[2, 2, 2];
        const FFTN_EXPECTED: &[complex64; 8] = &[
            complex64::new(8.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
            complex64::new(0.0, 0.0),
        ];

        let array = Array::from_slice(FFTN_DATA, FFTN_SHAPE);
        let fftn = fftn(&array, None, None).unwrap();

        assert_eq!(fftn.dtype(), Dtype::Complex64);
        assert_eq!(fftn.as_slice::<complex64>(), FFTN_EXPECTED);

        let ifftn = ifftn(&fftn, FFTN_SHAPE, &[0, 1, 2]).unwrap();

        assert_eq!(ifftn.dtype(), Dtype::Complex64);
        assert_eq!(
            ifftn.as_slice::<complex64>(),
            FFTN_DATA
                .iter()
                .map(|&x| complex64::new(x, 0.0))
                .collect::<Vec<_>>()
        );

        // test that previous array is not modified and valid
        let data: &[f32] = array.as_slice();
        assert_eq!(data, FFTN_DATA);
    }
}
