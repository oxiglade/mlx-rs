use mlx_internal_macros::generate_macro;
use smallvec::SmallVec;

use crate::{
    array::Array, constants::DEFAULT_STACK_VEC_LEN, error::Result, utils::guard::Guarded,
    utils::IntoOption, with_stream, Stream,
};

/// Resolve axes for shift operations - when None, returns all axes
fn resolve_axes(a: &Array, axes: Option<&[i32]>) -> SmallVec<[i32; DEFAULT_STACK_VEC_LEN]> {
    match axes {
        Some(axes) => SmallVec::from_slice(axes),
        None => (0..a.ndim() as i32).collect(),
    }
}

/// Shift the zero-frequency component to the center of the spectrum.
///
/// This function swaps half-spaces for all axes listed (defaults to all).
/// Note that `y[0]` is the Nyquist component only if `len(x)` is even.
///
/// # Params
///
/// - `a`: The input array.
/// - `axes`: Axes over which to shift. The default is `None` which shifts all axes.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, fft::*};
///
/// let a = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, -4.0, -3.0, -2.0, -1.0], &[9]);
/// let shifted = fftshift(&a, None).unwrap();
/// // shifted contains: [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
/// ```
pub fn fftshift<'a>(a: impl AsRef<Array>, axes: impl IntoOption<&'a [i32]>) -> Result<Array> {
    let a = a.as_ref();
    let axes = resolve_axes(a, axes.into_option());
    let stream = Stream::thread_local_or_default();

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_fftshift(
            res,
            a.as_ptr(),
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`fftshift`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fft"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `fftshift`"
)]
pub fn fftshift_device<'a>(
    a: impl AsRef<Array>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    with_stream(stream.as_ref(), || fftshift(a, axes))
}

/// The inverse of `fftshift`.
///
/// Although identical for even-length `x`, the functions differ by one sample for odd-length `x`.
///
/// # Params
///
/// - `a`: The input array.
/// - `axes`: Axes over which to calculate. The default is `None` which shifts all axes.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, fft::*};
///
/// let a = Array::from_slice(&[-4.0f32, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], &[9]);
/// let unshifted = ifftshift(&a, None).unwrap();
/// // unshifted contains: [0.0, 1.0, 2.0, 3.0, 4.0, -4.0, -3.0, -2.0, -1.0]
/// ```
pub fn ifftshift<'a>(a: impl AsRef<Array>, axes: impl IntoOption<&'a [i32]>) -> Result<Array> {
    let a = a.as_ref();
    let axes = resolve_axes(a, axes.into_option());
    let stream = Stream::thread_local_or_default();

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fft_ifftshift(
            res,
            a.as_ptr(),
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`ifftshift`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fft"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `ifftshift`"
)]
pub fn ifftshift_device<'a>(
    a: impl AsRef<Array>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    with_stream(stream.as_ref(), || ifftshift(a, axes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random;

    // Helper to check fftshift matches expected behavior
    fn check_fftshift(a: &Array, axes: Option<&[i32]>) {
        let shifted = fftshift(a, axes).unwrap();
        let unshifted = ifftshift(&shifted, axes).unwrap();
        assert!(
            unshifted
                .all_close(a, 1e-5, 1e-6, None)
                .unwrap()
                .item::<bool>(),
            "ifftshift(fftshift(x)) should equal x"
        );
    }

    #[test]
    fn test_fftshift_1d() {
        // Test 1D arrays (matches Python test)
        random::seed(42).unwrap();
        let r = random::uniform::<_, f32>(0.0, 1.0, &[100], None).unwrap();
        check_fftshift(&r, None);
    }

    #[test]
    fn test_fftshift_with_axes() {
        // Test with specific axis (matches Python test)
        random::seed(42).unwrap();
        let r = random::uniform::<_, f32>(0.0, 1.0, &[4, 6], None).unwrap();
        check_fftshift(&r, Some(&[0]));
        check_fftshift(&r, Some(&[1]));
        check_fftshift(&r, Some(&[0, 1]));
    }

    #[test]
    fn test_fftshift_negative_axes() {
        // Test with negative axes (matches Python test)
        random::seed(42).unwrap();
        let r = random::uniform::<_, f32>(0.0, 1.0, &[4, 6], None).unwrap();
        check_fftshift(&r, Some(&[-1]));
    }

    #[test]
    fn test_fftshift_odd_lengths() {
        // Test with odd lengths (matches Python test)
        random::seed(42).unwrap();
        let r = random::uniform::<_, f32>(0.0, 1.0, &[5, 7], None).unwrap();
        check_fftshift(&r, None);
        check_fftshift(&r, Some(&[0]));
    }

    #[test]
    fn test_ifftshift_1d() {
        // Test 1D arrays (matches Python test)
        random::seed(42).unwrap();
        let r = random::uniform::<_, f32>(0.0, 1.0, &[100], None).unwrap();

        let shifted = ifftshift(&r, None).unwrap();
        let unshifted = fftshift(&shifted, None).unwrap();
        assert!(
            unshifted
                .all_close(&r, 1e-5, 1e-6, None)
                .unwrap()
                .item::<bool>(),
            "fftshift(ifftshift(x)) should equal x"
        );
    }

    #[test]
    fn test_ifftshift_with_axes() {
        // Test with specific axis (matches Python test)
        random::seed(42).unwrap();
        let r = random::uniform::<_, f32>(0.0, 1.0, &[4, 6], None).unwrap();

        for axes in [&[0][..], &[1][..], &[0, 1][..]] {
            let shifted = ifftshift(&r, axes).unwrap();
            let unshifted = fftshift(&shifted, axes).unwrap();
            assert!(
                unshifted
                    .all_close(&r, 1e-5, 1e-6, None)
                    .unwrap()
                    .item::<bool>(),
                "fftshift(ifftshift(x)) should equal x for axes {:?}",
                axes
            );
        }
    }

    #[test]
    fn test_ifftshift_negative_axes() {
        // Test with negative axes (matches Python test)
        random::seed(42).unwrap();
        let r = random::uniform::<_, f32>(0.0, 1.0, &[4, 6], None).unwrap();

        let shifted = ifftshift(&r, &[-1]).unwrap();
        let unshifted = fftshift(&shifted, &[-1]).unwrap();
        assert!(unshifted
            .all_close(&r, 1e-5, 1e-6, None)
            .unwrap()
            .item::<bool>(),);
    }

    #[test]
    fn test_ifftshift_odd_lengths() {
        // Test with odd lengths (matches Python test)
        random::seed(42).unwrap();
        let r = random::uniform::<_, f32>(0.0, 1.0, &[5, 7], None).unwrap();

        let shifted = ifftshift(&r, None).unwrap();
        let unshifted = fftshift(&shifted, None).unwrap();
        assert!(unshifted
            .all_close(&r, 1e-5, 1e-6, None)
            .unwrap()
            .item::<bool>(),);

        let shifted = ifftshift(&r, &[0]).unwrap();
        let unshifted = fftshift(&shifted, &[0]).unwrap();
        assert!(unshifted
            .all_close(&r, 1e-5, 1e-6, None)
            .unwrap()
            .item::<bool>(),);
    }

    #[test]
    fn test_fftshift_empty_array() {
        // Test empty array (matches Python test)
        let x = Array::from_slice::<f32>(&[], &[0]);
        let shifted = fftshift(&x, None).unwrap();
        assert!(shifted.array_eq(&x, None).unwrap().item::<bool>());
    }
}
