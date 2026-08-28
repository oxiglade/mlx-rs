//! Strict tensor assertions for this workspace's tests.
//!
//! This comparator is deliberately separate from the protected conformance-oracle comparator.
//! A runtime mismatch suspected to be an mlx-rs defect is isolated with an `#[ignore]` reason
//! until a fail-first defect fix. A wrong expectation is corrected only from an identified
//! source. Required numeric slack is added to the named tolerance table, never inline.

use crate::{complex64, Array, ArrayElement, Dtype};
use half::{bf16, f16};
use std::fmt::Debug;

/// A named relative/absolute tolerance pair.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    /// Relative tolerance multiplied by the absolute expected value.
    pub rtol: f64,
    /// Absolute tolerance added to the relative limit.
    pub atol: f64,
}

/// Named tolerances used by tensor assertions.
pub mod tolerances {
    use super::Tolerance;

    /// Exact comparison for integers and exactly reproducible floating-point results.
    pub const EXACT: Tolerance = Tolerance {
        rtol: 0.0,
        atol: 0.0,
    };

    /// The default tolerance used by MLX `all_close`.
    pub const MLX_DEFAULT: Tolerance = Tolerance {
        rtol: 1e-5,
        atol: 1e-8,
    };

    /// Tests that require `1e-5` for both tolerances.
    pub const STANDARD: Tolerance = Tolerance {
        rtol: 1e-5,
        atol: 1e-5,
    };

    /// References rounded to two decimal places.
    pub const ROUNDED_TWO_DECIMALS: Tolerance = Tolerance {
        rtol: 0.0,
        atol: 1e-2,
    };

    /// Seeded random statistics inherited from upstream tests with a two-percent relative bound.
    pub const RANDOM_STATISTIC: Tolerance = Tolerance {
        rtol: 2e-2,
        atol: 0.0,
    };
}

#[derive(Debug)]
enum TensorData {
    Bool(Vec<bool>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    C64(Vec<complex64>),
}

impl TensorData {
    fn len(&self) -> usize {
        match self {
            Self::Bool(values) => values.len(),
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::I8(values) => values.len(),
            Self::I16(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::F16(values) => values.len(),
            Self::BF16(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
            Self::C64(values) => values.len(),
        }
    }
}

#[derive(Debug)]
struct HostTensor {
    dtype: Dtype,
    shape: Vec<i32>,
    data: TensorData,
}

fn logical_offsets(shape: &[i32], strides: &[usize], count: usize) -> Result<Vec<usize>, String> {
    if shape.len() != strides.len() {
        return Err(format!(
            "shape rank {} does not match stride rank {}",
            shape.len(),
            strides.len()
        ));
    }
    let dimensions = shape
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension).map_err(|_| format!("negative dimension {dimension}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let logical_count = dimensions.iter().try_fold(1usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or_else(|| "logical element count overflow".to_owned())
    })?;
    if logical_count != count {
        return Err(format!(
            "shape contains {logical_count} elements but array size is {count}"
        ));
    }
    if count == 0 {
        return Ok(Vec::new());
    }

    (0..count)
        .map(|mut logical_index| {
            let mut offset = 0usize;
            for (&dimension, &stride) in dimensions.iter().zip(strides).rev() {
                let index = logical_index % dimension;
                logical_index /= dimension;
                offset = offset
                    .checked_add(
                        index
                            .checked_mul(stride)
                            .ok_or_else(|| "logical offset multiplication overflow".to_owned())?,
                    )
                    .ok_or_else(|| "logical offset overflow".to_owned())?;
            }
            Ok(offset)
        })
        .collect()
}

fn read_values<T: ArrayElement + Copy>(array: &Array, offsets: &[usize]) -> Result<Vec<T>, String> {
    let pointer = T::array_data(array);
    if pointer.is_null() && !offsets.is_empty() {
        return Err(format!("null data pointer for {:?}", array.dtype()));
    }
    unsafe { Ok(offsets.iter().map(|&offset| *pointer.add(offset)).collect()) }
}

fn observe(array: &Array) -> Result<HostTensor, String> {
    array.eval().map_err(|error| error.to_string())?;
    let dtype = array.dtype();
    let shape = array.shape().to_vec();
    let offsets = logical_offsets(&shape, array.strides(), array.size())?;
    let data = match dtype {
        Dtype::Bool => TensorData::Bool(read_values(array, &offsets)?),
        Dtype::Uint8 => TensorData::U8(read_values(array, &offsets)?),
        Dtype::Uint16 => TensorData::U16(read_values(array, &offsets)?),
        Dtype::Uint32 => TensorData::U32(read_values(array, &offsets)?),
        Dtype::Uint64 => TensorData::U64(read_values(array, &offsets)?),
        Dtype::Int8 => TensorData::I8(read_values(array, &offsets)?),
        Dtype::Int16 => TensorData::I16(read_values(array, &offsets)?),
        Dtype::Int32 => TensorData::I32(read_values(array, &offsets)?),
        Dtype::Int64 => TensorData::I64(read_values(array, &offsets)?),
        Dtype::Float16 => TensorData::F16(read_values(array, &offsets)?),
        Dtype::Bfloat16 => TensorData::BF16(read_values(array, &offsets)?),
        Dtype::Float32 => TensorData::F32(read_values(array, &offsets)?),
        Dtype::Float64 => TensorData::F64(read_values(array, &offsets)?),
        Dtype::Complex64 => TensorData::C64(read_values(array, &offsets)?),
    };
    Ok(HostTensor { dtype, shape, data })
}

trait ExactElement: Copy + Debug + PartialEq {
    fn bits(self) -> u64;
    fn error(self, other: Self) -> f64;
}

macro_rules! impl_exact_unsigned {
    ($($type:ty),+ $(,)?) => {
        $(impl ExactElement for $type {
            fn bits(self) -> u64 { self as u64 }
            fn error(self, other: Self) -> f64 { self.abs_diff(other) as f64 }
        })+
    };
}

macro_rules! impl_exact_signed {
    ($(($type:ty, $unsigned:ty)),+ $(,)?) => {
        $(impl ExactElement for $type {
            fn bits(self) -> u64 { self as $unsigned as u64 }
            fn error(self, other: Self) -> f64 { self.abs_diff(other) as f64 }
        })+
    };
}

impl ExactElement for bool {
    fn bits(self) -> u64 {
        u64::from(self)
    }

    fn error(self, other: Self) -> f64 {
        f64::from(u8::from(self) != u8::from(other))
    }
}

impl_exact_unsigned!(u8, u16, u32, u64);
impl_exact_signed!((i8, u8), (i16, u16), (i32, u32), (i64, u64));

fn compare_exact<T: ExactElement>(got: &[T], expected: &[T]) -> Result<(), String> {
    let mut first_bad = None;
    let mut max_error = 0.0f64;
    for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        let error = got.error(expected);
        max_error = max_error.max(error);
        if got != expected && first_bad.is_none() {
            first_bad = Some((index, got, expected));
        }
    }
    if let Some((index, got, expected)) = first_bad {
        Err(format!(
            "first bad index {index}: expected {expected:?} (bits 0x{:x}), got {got:?} (bits 0x{:x}); max error {max_error:e}",
            expected.bits(),
            got.bits()
        ))
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct FloatValue {
    value: f64,
    bits: u64,
}

fn compare_float_values(
    values: impl IntoIterator<Item = (String, FloatValue, FloatValue)>,
    rtol: f64,
    atol: f64,
) -> Result<(), String> {
    let mut first_bad = None;
    let mut max_error = 0.0f64;
    for (location, got, expected) in values {
        let error = if expected.value.is_nan() || got.value.is_nan() {
            if expected.value.is_nan() && got.value.is_nan() {
                0.0
            } else {
                f64::INFINITY
            }
        } else if expected.value.is_infinite() || got.value.is_infinite() {
            if expected.value == got.value {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            (expected.value - got.value).abs()
        };
        max_error = max_error.max(error);
        let limit = atol + rtol * expected.value.abs();
        let matches = if expected.value.is_nan() || got.value.is_nan() {
            expected.value.is_nan() && got.value.is_nan()
        } else if expected.value.is_infinite() || got.value.is_infinite() {
            expected.value == got.value
        } else {
            error <= limit
        };
        if !matches && first_bad.is_none() {
            first_bad = Some((location, got, expected, error, limit));
        }
    }
    if let Some((location, got, expected, error, limit)) = first_bad {
        Err(format!(
            "first bad {location}: expected {:?} (bits 0x{:x}), got {:?} (bits 0x{:x}), error {error:e}, limit {limit:e}; max error {max_error:e}",
            expected.value, expected.bits, got.value, got.bits
        ))
    } else {
        Ok(())
    }
}

fn floats<T: Copy>(
    got: &[T],
    expected: &[T],
    convert: impl Fn(T) -> FloatValue,
    rtol: f64,
    atol: f64,
) -> Result<(), String> {
    compare_float_values(
        got.iter()
            .zip(expected)
            .enumerate()
            .map(|(index, (&got, &expected))| {
                (format!("index {index}"), convert(got), convert(expected))
            }),
        rtol,
        atol,
    )
}

fn compare_tensor(
    got: &HostTensor,
    expected: &HostTensor,
    rtol: f64,
    atol: f64,
) -> Result<(), String> {
    if !rtol.is_finite() || !atol.is_finite() || rtol < 0.0 || atol < 0.0 {
        return Err(format!(
            "rtol and atol must be finite and non-negative, got rtol={rtol:?}, atol={atol:?}"
        ));
    }
    if got.dtype != expected.dtype {
        return Err(format!(
            "dtype mismatch: expected {:?}, got {:?}",
            expected.dtype, got.dtype
        ));
    }
    if got.shape != expected.shape {
        return Err(format!(
            "shape mismatch: expected {:?}, got {:?}",
            expected.shape, got.shape
        ));
    }
    if got.data.len() != expected.data.len() {
        return Err(format!(
            "size mismatch: expected {}, got {}",
            expected.data.len(),
            got.data.len()
        ));
    }

    match (&got.data, &expected.data) {
        (TensorData::Bool(got), TensorData::Bool(expected)) => compare_exact(got, expected),
        (TensorData::U8(got), TensorData::U8(expected)) => compare_exact(got, expected),
        (TensorData::U16(got), TensorData::U16(expected)) => compare_exact(got, expected),
        (TensorData::U32(got), TensorData::U32(expected)) => compare_exact(got, expected),
        (TensorData::U64(got), TensorData::U64(expected)) => compare_exact(got, expected),
        (TensorData::I8(got), TensorData::I8(expected)) => compare_exact(got, expected),
        (TensorData::I16(got), TensorData::I16(expected)) => compare_exact(got, expected),
        (TensorData::I32(got), TensorData::I32(expected)) => compare_exact(got, expected),
        (TensorData::I64(got), TensorData::I64(expected)) => compare_exact(got, expected),
        (TensorData::F16(got), TensorData::F16(expected)) => floats(
            got,
            expected,
            |value| FloatValue {
                value: value.to_f64(),
                bits: value.to_bits() as u64,
            },
            rtol,
            atol,
        ),
        (TensorData::BF16(got), TensorData::BF16(expected)) => floats(
            got,
            expected,
            |value| FloatValue {
                value: value.to_f64(),
                bits: value.to_bits() as u64,
            },
            rtol,
            atol,
        ),
        (TensorData::F32(got), TensorData::F32(expected)) => floats(
            got,
            expected,
            |value| FloatValue {
                value: value as f64,
                bits: value.to_bits() as u64,
            },
            rtol,
            atol,
        ),
        (TensorData::F64(got), TensorData::F64(expected)) => floats(
            got,
            expected,
            |value| FloatValue {
                value,
                bits: value.to_bits(),
            },
            rtol,
            atol,
        ),
        (TensorData::C64(got), TensorData::C64(expected)) => compare_float_values(
            got.iter()
                .zip(expected)
                .enumerate()
                .flat_map(|(index, (got, expected))| {
                    [("real", got.re, expected.re), ("imag", got.im, expected.im)].map(
                        move |(component, got, expected)| {
                            (
                                format!("index {index} {component}"),
                                FloatValue {
                                    value: got as f64,
                                    bits: got.to_bits() as u64,
                                },
                                FloatValue {
                                    value: expected as f64,
                                    bits: expected.to_bits() as u64,
                                },
                            )
                        },
                    )
                }),
            rtol,
            atol,
        ),
        _ => Err("dtype and host representation disagree".to_owned()),
    }
}

/// Asserts strict tensor equality with explicit relative and absolute tolerances.
///
/// Dtypes and shapes must always match. Integers and booleans compare exactly. Floating-point
/// values use `abs(got - expected) <= atol + rtol * abs(expected)`; NaNs match only NaNs,
/// infinities match only the same sign, and `-0.0` matches `+0.0`. Complex values compare each
/// component under the same rule. This implementation stays deliberately separate from the
/// protected conformance-oracle comparator.
///
/// # Panics
///
/// Panics on an observation error or mismatch. Value mismatch output includes the first bad
/// logical index, expected and actual values with their bits, and the maximum error.
pub fn assert_array_eq(got: impl AsRef<Array>, expected: impl AsRef<Array>, rtol: f64, atol: f64) {
    assert_array_eq_impl(got, expected, rtol, atol, None);
}

/// Asserts strict tensor equality with a context label included in panic diagnostics.
///
/// Comparison semantics are identical to [`assert_array_eq`].
pub fn assert_array_eq_with_context(
    got: impl AsRef<Array>,
    expected: impl AsRef<Array>,
    rtol: f64,
    atol: f64,
    context: &str,
) {
    assert_array_eq_impl(got, expected, rtol, atol, Some(context));
}

fn assert_array_eq_impl(
    got: impl AsRef<Array>,
    expected: impl AsRef<Array>,
    rtol: f64,
    atol: f64,
    context: Option<&str>,
) {
    let got = observe(got.as_ref()).unwrap_or_else(|error| {
        panic_with_context(context, &format!("failed to observe got tensor: {error}"))
    });
    let expected = observe(expected.as_ref()).unwrap_or_else(|error| {
        panic_with_context(
            context,
            &format!("failed to observe expected tensor: {error}"),
        )
    });
    if let Err(error) = compare_tensor(&got, &expected, rtol, atol) {
        panic_with_context(context, &format!("tensor mismatch: {error}"));
    }
}

fn panic_with_context(context: Option<&str>, message: &str) -> ! {
    match context {
        Some(context) => panic!("{context}: {message}"),
        None => panic!("{message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{assert_array_eq, compare_tensor, observe, tolerances, HostTensor, TensorData};
    use crate::{array, ops::broadcast_to, Array, Dtype};

    fn f32_tensor(dtype: Dtype, shape: &[i32], values: &[f32]) -> HostTensor {
        HostTensor {
            dtype,
            shape: shape.to_vec(),
            data: TensorData::F32(values.to_vec()),
        }
    }

    fn i64_tensor(shape: &[i32], values: &[i64]) -> HostTensor {
        HostTensor {
            dtype: Dtype::Int64,
            shape: shape.to_vec(),
            data: TensorData::I64(values.to_vec()),
        }
    }

    #[test]
    fn rejects_equal_values_with_different_dtypes() {
        let expected = f32_tensor(Dtype::Float32, &[1], &[1.0]);
        let got = f32_tensor(Dtype::Float16, &[1], &[1.0]);

        assert!(compare_tensor(&got, &expected, 0.0, 0.0).is_err());
    }

    #[test]
    fn rejects_equal_element_counts_with_different_shapes() {
        let expected = f32_tensor(Dtype::Float32, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let got = f32_tensor(Dtype::Float32, &[4], &[1.0, 2.0, 3.0, 4.0]);

        assert!(compare_tensor(&got, &expected, 0.0, 0.0).is_err());
    }

    #[test]
    fn observes_view_in_logical_order() {
        let array = array!([[1_u16, 2, 3], [4, 5, 6]]).transpose().unwrap();
        let expected = array!([[1_u16, 4], [2, 5], [3, 6]]);

        assert_eq!(observe(&array).unwrap().dtype, array.dtype());
        assert_array_eq(
            array,
            expected,
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );
    }

    #[test]
    fn observes_scalar_dtype_faithfully() {
        let array = array!(1);

        assert_eq!(observe(&array).unwrap().dtype, array.dtype());
    }

    #[test]
    fn observes_broadcast_in_logical_order() {
        let array = broadcast_to(&array!([true, false]), &[3, 2]).unwrap();
        let expected = array!([[true, false], [true, false], [true, false]]);

        assert_eq!(observe(&array).unwrap().dtype, array.dtype());
        assert_array_eq(
            array,
            expected,
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );
    }

    #[test]
    fn observes_empty_dtype_faithfully() {
        let array = Array::from_slice::<f32>(&[], &[0]);

        assert_eq!(observe(&array).unwrap().dtype, array.dtype());
    }

    #[test]
    fn rejects_value_just_beyond_atol() {
        let expected = f32_tensor(Dtype::Float32, &[1], &[0.0]);
        let got = f32_tensor(Dtype::Float32, &[1], &[0.100_000_01]);

        let error = compare_tensor(&got, &expected, 0.0, 0.1).unwrap_err();
        assert!(error.contains("first bad index 0"));
        assert!(error.contains("expected 0.0 (bits 0x0)"));
        assert!(error.contains("got 0.100000"));
        assert!(error.contains("bits 0x3dccccce"));
        assert!(error.contains("max error"));
    }

    #[test]
    fn rejects_value_just_beyond_rtol() {
        let expected = f32_tensor(Dtype::Float32, &[1], &[10.0]);
        let got = f32_tensor(Dtype::Float32, &[1], &[11.000_001]);

        assert!(compare_tensor(&got, &expected, 0.1, 0.0).is_err());
    }

    #[test]
    fn rejects_nan_against_finite() {
        let expected = f32_tensor(Dtype::Float32, &[1], &[1.0]);
        let got = f32_tensor(Dtype::Float32, &[1], &[f32::NAN]);

        assert!(compare_tensor(&got, &expected, 0.0, 0.0).is_err());
    }

    #[test]
    fn rejects_infinities_with_different_signs() {
        let expected = f32_tensor(Dtype::Float32, &[1], &[f32::INFINITY]);
        let got = f32_tensor(Dtype::Float32, &[1], &[f32::NEG_INFINITY]);

        assert!(compare_tensor(&got, &expected, 0.0, 0.0).is_err());
    }

    #[test]
    fn accepts_signed_zero_difference() {
        let expected = f32_tensor(Dtype::Float32, &[1], &[-0.0]);
        let got = f32_tensor(Dtype::Float32, &[1], &[0.0]);

        compare_tensor(&got, &expected, 0.0, 0.0).unwrap();
    }

    #[test]
    fn reports_error_for_large_adjacent_i64_values() {
        let expected = i64_tensor(&[1], &[9_007_199_254_740_993]);
        let got = i64_tensor(&[1], &[9_007_199_254_740_992]);

        let error = compare_tensor(&got, &expected, 0.0, 0.0).unwrap_err();
        assert!(error.contains("first bad index 0"));
        assert!(error.contains("expected 9007199254740993 (bits 0x20000000000001)"));
        assert!(error.contains("got 9007199254740992 (bits 0x20000000000000)"));
        assert!(error.contains("max error 1e0"));
    }
}
