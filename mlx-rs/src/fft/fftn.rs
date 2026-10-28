use mlx_internal_macros::generate_macro;

use crate::{
    array::Array,
    error::Result,
    utils::{guard::Guarded, IntoOption},
    with_stream, Stream,
};

use super::{
    utils::{
        legacy_fftn_options, resolve_lengths_and_axes, resolve_size_and_axis_unchecked,
        resolve_sizes_and_axes_unchecked,
    },
    FftnOptions,
};

/// One dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n`. The default value is `a.shape[axis]`.
/// - `axis`: Axis along which to perform the FFT. The default is -1.
pub fn fft(
    a: impl AsRef<Array>,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
) -> Result<Array> {
    let a = a.as_ref();
    let (n, axis) = resolve_size_and_axis_unchecked(a, n.into(), axis.into());
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_fft(
            res,
            a.as_ptr(),
            n,
            axis,
            mlx_sys::mlx_fft_norm__MLX_FFT_NORM_BACKWARD,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`fft`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fft"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `fft`"
)]
pub fn fft_device(
    a: impl AsRef<Array>,
    #[optional] n: impl Into<Option<i32>>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    with_stream(stream.as_ref(), || fft(a, n, axis))
}

/// Two dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Size of the transformed axes. The corresponding axes in the input are truncated or padded
///   with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
pub fn fft2<'a>(
    a: impl AsRef<Array>,
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
) -> Result<Array> {
    let a = a.as_ref();
    let axes = axes.into_option().unwrap_or(&[-2, -1]);
    let (s, axes) = resolve_sizes_and_axes_unchecked(a, s.into_option(), Some(axes));

    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();
    let stream = Stream::thread_local_or_default();

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_fft2(
            res,
            a.as_ptr(),
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            mlx_sys::mlx_fft_norm__MLX_FFT_NORM_BACKWARD,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`fft2`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fft"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `fft2`"
)]
pub fn fft2_device<'a>(
    a: impl AsRef<Array>,
    #[optional] s: impl IntoOption<&'a [i32]>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    with_stream(stream.as_ref(), || fft2(a, s, axes))
}

/// n-dimensional discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `options`: Transform lengths and axes. The default transforms all axes at their input sizes.
pub fn fftn(a: impl AsRef<Array>, options: FftnOptions) -> Result<Array> {
    let a = a.as_ref();
    let (s, axes) = resolve_lengths_and_axes(a.shape(), options.lengths.as_deref(), &options.axes)?;
    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();
    let stream = Stream::thread_local_or_default();

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_fftn(
            res,
            a.as_ptr(),
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            mlx_sys::mlx_fft_norm__MLX_FFT_NORM_BACKWARD,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`fftn`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fft"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `fftn` with `FftnOptions`"
)]
pub fn fftn_device<'a>(
    a: impl AsRef<Array>,
    #[optional] s: impl IntoOption<&'a [i32]>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let options = legacy_fftn_options(s.into_option(), axes.into_option())?;
    with_stream(stream.as_ref(), || fftn(a, options))
}

/// One dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: Input array.
/// - `n`: Size of the transformed axis. The corresponding axis in the input is truncated or padded
///   with zeros to match `n`. The default value is `a.shape[axis]` if not specified.
/// - `axis`: Axis along which to perform the FFT. The default is `-1` if not specified.
pub fn ifft(
    a: impl AsRef<Array>,
    n: impl Into<Option<i32>>,
    axis: impl Into<Option<i32>>,
) -> Result<Array> {
    let a = a.as_ref();
    let (n, axis) = resolve_size_and_axis_unchecked(a, n.into(), axis.into());
    let stream = Stream::thread_local_or_default();

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_ifft(
            res,
            a.as_ptr(),
            n,
            axis,
            mlx_sys::mlx_fft_norm__MLX_FFT_NORM_BACKWARD,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`ifft`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fft"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `ifft`"
)]
pub fn ifft_device(
    a: impl AsRef<Array>,
    #[optional] n: impl Into<Option<i32>>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    with_stream(stream.as_ref(), || ifft(a, n, axis))
}

/// Two dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `s`: Size of the transformed axes. The corresponding axes in the input are truncated or padded
///   with zeros to match `s`. The default value is the sizes of `a` along `axes`.
/// - `axes`: Axes along which to perform the FFT. The default is `[-2, -1]`.
pub fn ifft2<'a>(
    a: impl AsRef<Array>,
    s: impl IntoOption<&'a [i32]>,
    axes: impl IntoOption<&'a [i32]>,
) -> Result<Array> {
    let a = a.as_ref();
    let axes = axes.into_option().unwrap_or(&[-2, -1]);
    let (s, axes) = resolve_sizes_and_axes_unchecked(a, s.into_option(), Some(axes));

    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();
    let stream = Stream::thread_local_or_default();

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_ifft2(
            res,
            a.as_ptr(),
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            mlx_sys::mlx_fft_norm__MLX_FFT_NORM_BACKWARD,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`ifft2`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fft"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `ifft2`"
)]
pub fn ifft2_device<'a>(
    a: impl AsRef<Array>,
    #[optional] s: impl IntoOption<&'a [i32]>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    with_stream(stream.as_ref(), || ifft2(a, s, axes))
}

/// n-dimensional inverse discrete Fourier Transform.
///
/// # Params
///
/// - `a`: The input array.
/// - `options`: Transform lengths and axes. The default transforms all axes at their input sizes.
pub fn ifftn(a: impl AsRef<Array>, options: FftnOptions) -> Result<Array> {
    let a = a.as_ref();
    let (s, axes) = resolve_lengths_and_axes(a.shape(), options.lengths.as_deref(), &options.axes)?;
    let num_s = s.len();
    let num_axes = axes.len();

    let s_ptr = s.as_ptr();
    let axes_ptr = axes.as_ptr();
    let stream = Stream::thread_local_or_default();

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_ifftn(
            res,
            a.as_ptr(),
            s_ptr,
            num_s,
            axes_ptr,
            num_axes,
            mlx_sys::mlx_fft_norm__MLX_FFT_NORM_BACKWARD,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`ifftn`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fft"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `ifftn` with `FftnOptions`"
)]
pub fn ifftn_device<'a>(
    a: impl AsRef<Array>,
    #[optional] s: impl IntoOption<&'a [i32]>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let options = legacy_fftn_options(s.into_option(), axes.into_option())?;
    with_stream(stream.as_ref(), || ifftn(a, options))
}

#[cfg(test)]
mod tests {
    use crate::{
        complex64, fft::*, ops::indexing::TryIndexOp, test_utils::assert_array_eq,
        test_utils::tolerances, Array, Dtype, Stream,
    };

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
        assert_array_eq(
            &fft,
            Array::from_slice(FFT_EXPECTED, FFT_SHAPE),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        let ifft = ifft(&fft, None, None).unwrap();

        assert_eq!(ifft.dtype(), Dtype::Complex64);
        let expected = FFT_DATA
            .iter()
            .map(|&x| complex64::new(x, 0.0))
            .collect::<Vec<_>>();
        assert_array_eq(
            &ifft,
            Array::from_slice(&expected, FFT_SHAPE),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        assert_array_eq(
            array,
            Array::from_slice(FFT_DATA, FFT_SHAPE),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );
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
        assert_array_eq(
            &fft2,
            Array::from_slice(FFT2_EXPECTED, FFT2_SHAPE),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        let ifft2 = ifft2(&fft2, None, None).unwrap();

        assert_eq!(ifft2.dtype(), Dtype::Complex64);
        let expected = FFT2_DATA
            .iter()
            .map(|&x| complex64::new(x, 0.0))
            .collect::<Vec<_>>();
        assert_array_eq(
            &ifft2,
            Array::from_slice(&expected, FFT2_SHAPE),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        assert_array_eq(
            array,
            Array::from_slice(FFT2_DATA, FFT2_SHAPE),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );
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
        let fftn = fftn(&array, FftnOptions::default()).unwrap();

        assert_eq!(fftn.dtype(), Dtype::Complex64);
        assert_array_eq(
            &fftn,
            Array::from_slice(FFTN_EXPECTED, FFTN_SHAPE),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        let ifftn = ifftn(
            &fftn,
            FftnOptions {
                lengths: Some(FFTN_SHAPE.to_vec()),
                axes: [0, 1, 2].into(),
            },
        )
        .unwrap();

        assert_eq!(ifftn.dtype(), Dtype::Complex64);
        let expected = FFTN_DATA
            .iter()
            .map(|&x| complex64::new(x, 0.0))
            .collect::<Vec<_>>();
        assert_array_eq(
            &ifftn,
            Array::from_slice(&expected, FFTN_SHAPE),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        assert_array_eq(
            array,
            Array::from_slice(FFTN_DATA, FFTN_SHAPE),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );
    }

    #[test]
    fn asymmetric_multidimensional_transforms_use_backward_normalization() {
        let input2 = Array::from_slice(&[1.0_f32, 2.0, 3.0, 5.0, 7.0, 11.0], &[2, 3]);
        let spectrum2 = fft2(&input2, None, None).unwrap();
        assert_array_eq(
            spectrum2.try_index((0, 0)).unwrap(),
            Array::from(complex64::new(29.0, 0.0)),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );
        // The inverse of a complex spectrum is complex with ~zero imaginaries;
        // compare in the output's dtype rather than casting it away.
        let roundtrip2 = ifft2(&spectrum2, None, None).unwrap();
        let expected2 = Array::from_slice(
            &[1.0_f32, 2.0, 3.0, 5.0, 7.0, 11.0].map(|re| complex64::new(re, 0.0)),
            &[2, 3],
        );
        assert_array_eq(
            &roundtrip2,
            &expected2,
            tolerances::STANDARD.rtol,
            tolerances::STANDARD.atol,
        );

        let inputn = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
        let spectrumn = fftn(&inputn, FftnOptions::default()).unwrap();
        assert_array_eq(
            spectrumn.try_index((0, 0, 0)).unwrap(),
            Array::from(complex64::new(36.0, 0.0)),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );
        let roundtripn = ifftn(&spectrumn, FftnOptions::default()).unwrap();
        let expectedn = Array::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].map(|re| complex64::new(re, 0.0)),
            &[2, 2, 2],
        );
        assert_array_eq(
            &roundtripn,
            &expectedn,
            tolerances::STANDARD.rtol,
            tolerances::STANDARD.atol,
        );
    }

    #[allow(deprecated)]
    #[test]
    fn compatibility_functions_and_macros_forward_to_canonical_fft() {
        let input = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
        let stream = Stream::cpu();
        let canonical = fft(&input, None, None).unwrap();

        for compatibility in [
            fft_device(&input, None, None, &stream).unwrap(),
            fft!(&input).unwrap(),
            fft!(&input, stream = &stream).unwrap(),
        ] {
            assert_array_eq(
                &compatibility,
                &canonical,
                tolerances::EXACT.rtol,
                tolerances::EXACT.atol,
            );
        }

        let canonical_n = fftn(
            &input,
            FftnOptions {
                lengths: Some(vec![4]),
                axes: (-1).into(),
            },
        )
        .unwrap();
        let compatibility_n = fftn!(&input, s = &[4]).unwrap();
        assert_array_eq(
            compatibility_n,
            canonical_n,
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );
    }
}
