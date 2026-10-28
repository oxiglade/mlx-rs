use mlx_internal_macros::generate_macro;

use crate::{error::Result, utils::guard::Guarded, Array, ArrayElement, Dtype, Stream};

impl Array {
    /// Convert an array to FP8 (E4M3) format.
    ///
    /// The input array must be a floating point type (float32, float16, or bfloat16).
    /// Values outside the representable range of FP8 E4M3 (-448 to 448) will be clipped.
    ///
    /// # Returns
    ///
    /// An array with dtype uint8 containing the FP8 E4M3 encoded values.
    pub fn to_fp8(&self) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_to_fp8(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Compatibility shim for [`to_fp8`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `to_fp8`"
    )]
    pub fn to_fp8_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.to_fp8())
    }

    /// Convert an FP8 (E4M3) encoded array back to a floating point type.
    ///
    /// The input array should be a uint8 array containing FP8 E4M3 encoded values.
    ///
    /// # Params
    ///
    /// - `dtype`: The target floating point dtype (float32, float16, or bfloat16)
    pub fn from_fp8(&self, dtype: Dtype) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_from_fp8(res, self.as_ptr(), dtype.into(), stream.as_ref().as_ptr())
        })
    }

    /// Compatibility shim for [`from_fp8`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `from_fp8`"
    )]
    pub fn from_fp8_device(&self, dtype: Dtype, stream: impl AsRef<Stream>) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.from_fp8(dtype))
    }

    /// Create a new array with the contents converted to the given [ArrayElement] type.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::{Array, Dtype};
    ///
    /// let array = Array::from_slice(&[1i16,2,3], &[3]);
    /// let mut new_array = array.as_type::<f32>().unwrap();
    ///
    /// assert_eq!(new_array.dtype(), Dtype::Float32);
    /// assert_eq!(new_array.shape(), &[3]);
    /// assert_eq!(new_array.item_size(), 4);
    /// assert_eq!(new_array.as_slice::<f32>(), &[1.0,2.0,3.0]);
    /// ```
    pub fn as_type<T: ArrayElement>(&self) -> Result<Array> {
        self.as_dtype(T::DTYPE)
    }

    /// Compatibility shim for [`as_type`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `as_type`"
    )]
    pub fn as_type_device<T: ArrayElement>(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.as_type::<T>())
    }

    /// Same as `as_type` but with a [`Dtype`] argument.
    pub fn as_dtype(&self, dtype: Dtype) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_astype(res, self.as_ptr(), dtype.into(), stream.as_ref().as_ptr())
        })
    }

    /// Compatibility shim for [`as_dtype`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `as_dtype`"
    )]
    pub fn as_dtype_device(&self, dtype: Dtype, stream: impl AsRef<Stream>) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.as_dtype(dtype))
    }

    /// View the array as a different type.
    ///
    /// The output array will change along the last axis if the input array's
    /// type and the output array's type do not have the same size.
    ///
    /// _Note: the view op does not imply that the input and output arrays share
    /// their underlying data. The view only guarantees that the binary
    /// representation of each element (or group of elements) is the same._
    ///
    pub fn view<T: ArrayElement>(&self) -> Result<Array> {
        self.view_dtype(T::DTYPE)
    }

    /// Compatibility shim for [`view`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `view`"
    )]
    pub fn view_device<T: ArrayElement>(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.view::<T>())
    }

    /// Same as `view` but with a [`Dtype`] argument.
    pub fn view_dtype(&self, dtype: Dtype) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_view(res, self.as_ptr(), dtype.into(), stream.as_ref().as_ptr())
        })
    }

    /// Compatibility shim for [`view_dtype`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `view_dtype`"
    )]
    pub fn view_dtype_device(&self, dtype: Dtype, stream: impl AsRef<Stream>) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.view_dtype(dtype))
    }
}

/// Convert an array to FP8 (E4M3) format.
///
/// See [`Array::to_fp8`] for more details.
pub fn to_fp8(a: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().to_fp8()
}

/// Compatibility shim for [`to_fp8`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `to_fp8`"
)]
pub fn to_fp8_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || to_fp8(a))
}

/// Convert an FP8 (E4M3) encoded array back to a floating point type.
///
/// See [`Array::from_fp8`] for more details.
pub fn from_fp8(a: impl AsRef<Array>, dtype: Dtype) -> Result<Array> {
    a.as_ref().from_fp8(dtype)
}

/// Compatibility shim for [`from_fp8`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `from_fp8`"
)]
pub fn from_fp8_device(
    a: impl AsRef<Array>,
    dtype: Dtype,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || from_fp8(a, dtype))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex64;
    use half::{bf16, f16};
    use pretty_assertions::assert_eq;

    macro_rules! test_as_type {
        ($src_type:ty, $src_val:expr, $dst_type:ty, $dst_val:expr, $len:expr) => {
            paste::paste! {
                #[test]
                fn [<test_as_type_ $src_type _ $dst_type>]() {
                    let array = Array::from_slice(&[$src_val; $len], &[$len as i32]);
                    let new_array = array.as_type::<$dst_type>().unwrap();

                    assert_eq!(new_array.dtype(), $dst_type::DTYPE);
                    assert_eq!(new_array.shape(), &[3]);
                    assert_eq!(new_array.item_size(), std::mem::size_of::<$dst_type>());
                    assert_eq!(new_array.as_slice::<$dst_type>(), &[$dst_val; $len]);
                }
            }
        };
    }

    test_as_type!(bool, true, i8, 1, 3);
    test_as_type!(bool, true, i16, 1, 3);
    test_as_type!(bool, true, i32, 1, 3);
    test_as_type!(bool, true, i64, 1, 3);
    test_as_type!(bool, true, u8, 1, 3);
    test_as_type!(bool, true, u16, 1, 3);
    test_as_type!(bool, true, u32, 1, 3);
    test_as_type!(bool, true, u64, 1, 3);
    test_as_type!(bool, true, f32, 1.0, 3);
    test_as_type!(bool, true, f16, f16::from_f32(1.0), 3);
    test_as_type!(bool, true, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(bool, true, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(i8, 1, bool, true, 3);
    test_as_type!(i8, 1, i16, 1, 3);
    test_as_type!(i8, 1, i32, 1, 3);
    test_as_type!(i8, 1, i64, 1, 3);
    test_as_type!(i8, 1, u8, 1, 3);
    test_as_type!(i8, 1, u16, 1, 3);
    test_as_type!(i8, 1, u32, 1, 3);
    test_as_type!(i8, 1, u64, 1, 3);
    test_as_type!(i8, 1, f32, 1.0, 3);
    test_as_type!(i8, 1, f16, f16::from_f32(1.0), 3);
    test_as_type!(i8, 1, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(i8, 1, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(i16, 1, bool, true, 3);
    test_as_type!(i16, 1, i8, 1, 3);
    test_as_type!(i16, 1, i32, 1, 3);
    test_as_type!(i16, 1, i64, 1, 3);
    test_as_type!(i16, 1, u8, 1, 3);
    test_as_type!(i16, 1, u16, 1, 3);
    test_as_type!(i16, 1, u32, 1, 3);
    test_as_type!(i16, 1, u64, 1, 3);
    test_as_type!(i16, 1, f32, 1.0, 3);
    test_as_type!(i16, 1, f16, f16::from_f32(1.0), 3);
    test_as_type!(i16, 1, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(i16, 1, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(i32, 1, bool, true, 3);
    test_as_type!(i32, 1, i8, 1, 3);
    test_as_type!(i32, 1, i16, 1, 3);
    test_as_type!(i32, 1, i64, 1, 3);
    test_as_type!(i32, 1, u8, 1, 3);
    test_as_type!(i32, 1, u16, 1, 3);
    test_as_type!(i32, 1, u32, 1, 3);
    test_as_type!(i32, 1, u64, 1, 3);
    test_as_type!(i32, 1, f32, 1.0, 3);
    test_as_type!(i32, 1, f16, f16::from_f32(1.0), 3);
    test_as_type!(i32, 1, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(i32, 1, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(i64, 1, bool, true, 3);
    test_as_type!(i64, 1, i8, 1, 3);
    test_as_type!(i64, 1, i16, 1, 3);
    test_as_type!(i64, 1, i32, 1, 3);
    test_as_type!(i64, 1, u8, 1, 3);
    test_as_type!(i64, 1, u16, 1, 3);
    test_as_type!(i64, 1, u32, 1, 3);
    test_as_type!(i64, 1, u64, 1, 3);
    test_as_type!(i64, 1, f32, 1.0, 3);
    test_as_type!(i64, 1, f16, f16::from_f32(1.0), 3);
    test_as_type!(i64, 1, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(i64, 1, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(u8, 1, bool, true, 3);
    test_as_type!(u8, 1, i8, 1, 3);
    test_as_type!(u8, 1, i16, 1, 3);
    test_as_type!(u8, 1, i32, 1, 3);
    test_as_type!(u8, 1, i64, 1, 3);
    test_as_type!(u8, 1, u16, 1, 3);
    test_as_type!(u8, 1, u32, 1, 3);
    test_as_type!(u8, 1, u64, 1, 3);
    test_as_type!(u8, 1, f32, 1.0, 3);
    test_as_type!(u8, 1, f16, f16::from_f32(1.0), 3);
    test_as_type!(u8, 1, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(u8, 1, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(u16, 1, bool, true, 3);
    test_as_type!(u16, 1, i8, 1, 3);
    test_as_type!(u16, 1, i16, 1, 3);
    test_as_type!(u16, 1, i32, 1, 3);
    test_as_type!(u16, 1, i64, 1, 3);
    test_as_type!(u16, 1, u8, 1, 3);
    test_as_type!(u16, 1, u32, 1, 3);
    test_as_type!(u16, 1, u64, 1, 3);
    test_as_type!(u16, 1, f32, 1.0, 3);
    test_as_type!(u16, 1, f16, f16::from_f32(1.0), 3);
    test_as_type!(u16, 1, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(u16, 1, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(u32, 1, bool, true, 3);
    test_as_type!(u32, 1, i8, 1, 3);
    test_as_type!(u32, 1, i16, 1, 3);
    test_as_type!(u32, 1, i32, 1, 3);
    test_as_type!(u32, 1, i64, 1, 3);
    test_as_type!(u32, 1, u8, 1, 3);
    test_as_type!(u32, 1, u16, 1, 3);
    test_as_type!(u32, 1, u64, 1, 3);
    test_as_type!(u32, 1, f32, 1.0, 3);
    test_as_type!(u32, 1, f16, f16::from_f32(1.0), 3);
    test_as_type!(u32, 1, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(u32, 1, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(u64, 1, bool, true, 3);
    test_as_type!(u64, 1, i8, 1, 3);
    test_as_type!(u64, 1, i16, 1, 3);
    test_as_type!(u64, 1, i32, 1, 3);
    test_as_type!(u64, 1, i64, 1, 3);
    test_as_type!(u64, 1, u8, 1, 3);
    test_as_type!(u64, 1, u16, 1, 3);
    test_as_type!(u64, 1, u32, 1, 3);
    test_as_type!(u64, 1, f32, 1.0, 3);
    test_as_type!(u64, 1, f16, f16::from_f32(1.0), 3);
    test_as_type!(u64, 1, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(u64, 1, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(f32, 1.0, bool, true, 3);
    test_as_type!(f32, 1.0, i8, 1, 3);
    test_as_type!(f32, 1.0, i16, 1, 3);
    test_as_type!(f32, 1.0, i32, 1, 3);
    test_as_type!(f32, 1.0, i64, 1, 3);
    test_as_type!(f32, 1.0, u8, 1, 3);
    test_as_type!(f32, 1.0, u16, 1, 3);
    test_as_type!(f32, 1.0, u32, 1, 3);
    test_as_type!(f32, 1.0, u64, 1, 3);
    test_as_type!(f32, 1.0, f16, f16::from_f32(1.0), 3);
    test_as_type!(f32, 1.0, bf16, bf16::from_f32(1.0), 3);
    test_as_type!(f32, 1.0, complex64, complex64::new(1.0, 0.0), 3);

    test_as_type!(f16, f16::from_f32(1.0), bool, true, 3);
    test_as_type!(f16, f16::from_f32(1.0), i8, 1, 3);
    test_as_type!(f16, f16::from_f32(1.0), i16, 1, 3);
    test_as_type!(f16, f16::from_f32(1.0), i32, 1, 3);
    test_as_type!(f16, f16::from_f32(1.0), i64, 1, 3);
    test_as_type!(f16, f16::from_f32(1.0), u8, 1, 3);
    test_as_type!(f16, f16::from_f32(1.0), u16, 1, 3);
    test_as_type!(f16, f16::from_f32(1.0), u32, 1, 3);
    test_as_type!(f16, f16::from_f32(1.0), u64, 1, 3);
    test_as_type!(f16, f16::from_f32(1.0), f32, 1.0, 3);
    test_as_type!(f16, f16::from_f32(1.0), bf16, bf16::from_f32(1.0), 3);
    test_as_type!(
        f16,
        f16::from_f32(1.0),
        complex64,
        complex64::new(1.0, 0.0),
        3
    );

    test_as_type!(bf16, bf16::from_f32(1.0), bool, true, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), i8, 1, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), i16, 1, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), i32, 1, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), i64, 1, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), u8, 1, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), u16, 1, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), u32, 1, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), u64, 1, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), f32, 1.0, 3);
    test_as_type!(bf16, bf16::from_f32(1.0), f16, f16::from_f32(1.0), 3);

    test_as_type!(complex64, complex64::new(1.0, 0.0), bool, true, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), i8, 1, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), i16, 1, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), i32, 1, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), i64, 1, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), u8, 1, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), u16, 1, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), u32, 1, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), u64, 1, 3);
    test_as_type!(complex64, complex64::new(1.0, 0.0), f32, 1.0, 3);
    test_as_type!(
        complex64,
        complex64::new(1.0, 0.0),
        f16,
        f16::from_f32(1.0),
        3
    );
    test_as_type!(
        complex64,
        complex64::new(1.0, 0.0),
        bf16,
        bf16::from_f32(1.0),
        3
    );

    #[test]
    fn test_view() {
        let array = Array::from_slice(&[1i16, 2, 3], &[3]);
        let new_array = array.view::<i8>().unwrap();

        assert_eq!(new_array.dtype(), Dtype::Int8);
        assert_eq!(new_array.shape(), &[6]);
        assert_eq!(new_array.item_size(), 1);
        assert_eq!(new_array.as_slice::<i8>(), &[1, 0, 2, 0, 3, 0]);
    }

    // The tests below are adapted from the C++ unit test `ops_tests.cpp/test fp8 conversion`
    #[test]
    fn test_fp8_conversion() {
        // Test round-trip for float32
        let input_f32 = Array::from_slice(&[-1.125f32, -1.0, 0.0, 1.0, 1.125, 4.5, 448.0], &[7]);
        let fp8 = input_f32.to_fp8().unwrap();
        assert_eq!(fp8.dtype(), Dtype::Uint8);
        let output_f32 = fp8.from_fp8(Dtype::Float32).unwrap();
        assert_eq!(output_f32.dtype(), Dtype::Float32);
        let data: &[f32] = output_f32.as_slice();
        assert_eq!(data, &[-1.125f32, -1.0, 0.0, 1.0, 1.125, 4.5, 448.0]);

        // Test round-trip for float16
        let input_f16 = Array::from_slice(
            &[
                f16::from_f32(-1.125),
                f16::from_f32(-1.0),
                f16::from_f32(0.0),
                f16::from_f32(1.0),
                f16::from_f32(1.125),
                f16::from_f32(4.5),
                f16::from_f32(448.0),
            ],
            &[7],
        );
        let fp8 = input_f16.to_fp8().unwrap();
        let output_f16 = fp8.from_fp8(Dtype::Float16).unwrap();
        assert_eq!(output_f16.dtype(), Dtype::Float16);
        let data: &[f16] = output_f16.as_slice();
        let expected_f16: Vec<f16> = vec![
            f16::from_f32(-1.125),
            f16::from_f32(-1.0),
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(1.125),
            f16::from_f32(4.5),
            f16::from_f32(448.0),
        ];
        assert_eq!(data, expected_f16.as_slice());

        // Test round-trip for bfloat16
        let input_bf16 = Array::from_slice(
            &[
                bf16::from_f32(-1.125),
                bf16::from_f32(-1.0),
                bf16::from_f32(0.0),
                bf16::from_f32(1.0),
                bf16::from_f32(1.125),
                bf16::from_f32(4.5),
                bf16::from_f32(448.0),
            ],
            &[7],
        );
        let fp8 = input_bf16.to_fp8().unwrap();
        let output_bf16 = fp8.from_fp8(Dtype::Bfloat16).unwrap();
        assert_eq!(output_bf16.dtype(), Dtype::Bfloat16);
        let data: &[bf16] = output_bf16.as_slice();
        let expected_bf16: Vec<bf16> = vec![
            bf16::from_f32(-1.125),
            bf16::from_f32(-1.0),
            bf16::from_f32(0.0),
            bf16::from_f32(1.0),
            bf16::from_f32(1.125),
            bf16::from_f32(4.5),
            bf16::from_f32(448.0),
        ];
        assert_eq!(data, expected_bf16.as_slice());

        // Test rounding - noisy input should round to expected values
        let noisy_in =
            Array::from_slice(&[-1.135f32, -1.01, 0.0001, 1.01, 1.135, 4.6, 447.0], &[7]);
        let expected = Array::from_slice(&[-1.125f32, -1.0, 0.0, 1.0, 1.125, 4.5, 448.0], &[7]);
        let fp8 = noisy_in.to_fp8().unwrap();
        let output = fp8.from_fp8(Dtype::Float32).unwrap();
        let output_data: &[f32] = output.as_slice();
        let expected_data: &[f32] = expected.as_slice();
        assert_eq!(output_data, expected_data);

        // Test overflow - values outside representable range get clamped
        let overflow_in = Array::from_slice(&[-600.0f32, 600.0], &[2]);
        let fp8 = overflow_in.to_fp8().unwrap();
        let output = fp8.from_fp8(Dtype::Float32).unwrap();
        let data: &[f32] = output.as_slice();
        assert_eq!(data, &[-448.0f32, 448.0]);
    }
}
