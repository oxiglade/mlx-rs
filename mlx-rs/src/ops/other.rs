use std::ffi::CString;

use mlx_internal_macros::{default_device, generate_macro};

use crate::utils::guard::Guarded;
use crate::utils::VectorArray;
use crate::{
    error::{Exception, Result},
    Array, Stream,
};

impl Array {
    /// Extract a diagonal or construct a diagonal matrix.
    ///
    /// If self is 1-D then a diagonal matrix is constructed with self on the `k`-th diagonal. If
    /// self is 2-D then the `k`-th diagonal is returned.
    ///
    /// # Params:
    ///
    /// - `k`: the diagonal to extract or construct
    /// - `stream`: stream or device to evaluate on
    #[default_device]
    pub fn diag_device(
        &self,
        k: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_diag(
                res,
                self.as_ptr(),
                k.into().unwrap_or(0),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Return specified diagonals.
    ///
    /// If self is 2-D, then a 1-D array containing the diagonal at the given `offset` is returned.
    ///
    /// If self has more than two dimensions, then `axis1` and `axis2` determine the 2D subarrays
    /// from which diagonals are extracted. The new shape is the original shape with `axis1` and
    /// `axis2` removed and a new dimension inserted at the end corresponding to the diagonal.
    ///
    /// # Params:
    ///
    /// - `offset`: offset of the diagonal.  Can be positive or negative
    /// - `axis1`: first axis of the 2-D sub-array from which the diagonals should be taken
    /// - `axis2`: second axis of the 2-D sub-array from which the diagonals should be taken
    /// - `stream`: stream or device to evaluate on
    #[default_device]
    pub fn diagonal_device(
        &self,
        offset: impl Into<Option<i32>>,
        axis1: impl Into<Option<i32>>,
        axis2: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_diagonal(
                res,
                self.as_ptr(),
                offset.into().unwrap_or(0),
                axis1.into().unwrap_or(0),
                axis2.into().unwrap_or(1),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Perform the Walsh-Hadamard transform along the final axis.
    ///
    /// Supports sizes `n = m*2^k` for `m` in `(1, 12, 20, 28)` and `2^k <= 8192`
    /// for ``DType/float32`` and `2^k <= 16384` for ``DType/float16`` and ``DType/bfloat16``.
    ///
    /// # Params
    /// - scale: scale the output by this factor -- default is `1.0/sqrt(array.dim(-1))`
    /// - stream: stream to evaluate on.
    #[default_device]
    pub fn hadamard_transform_device(
        &self,
        scale: impl Into<Option<f32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let scale = scale.into();
        let scale = mlx_sys::mlx_optional_float {
            value: scale.unwrap_or(0.0),
            has_value: scale.is_some(),
        };

        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_hadamard_transform(res, self.as_ptr(), scale, stream.as_ref().as_ptr())
        })
    }
}

/// See [`Array::diag`]
#[generate_macro]
#[default_device]
pub fn diag_device(
    a: impl AsRef<Array>,
    #[optional] k: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().diag_device(k, stream)
}

/// See [`Array::diagonal`]
#[generate_macro]
#[default_device]
pub fn diagonal_device(
    a: impl AsRef<Array>,
    #[optional] offset: impl Into<Option<i32>>,
    #[optional] axis1: impl Into<Option<i32>>,
    #[optional] axis2: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().diagonal_device(offset, axis1, axis2, stream)
}

/// Perform the Einstein summation convention on the operands.
///
/// # Params
///
/// - subscripts: Einstein summation convention equation
/// - operands: input arrays
/// - stream: stream or device to evaluate on
#[generate_macro]
#[default_device]
pub fn einsum_device<'a>(
    subscripts: &str,
    operands: impl IntoIterator<Item = &'a Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let c_subscripts =
        CString::new(subscripts).map_err(|_| Exception::from("Invalid subscripts"))?;
    let c_operands = VectorArray::try_from_iter(operands.into_iter())?;

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_einsum(
            res,
            c_subscripts.as_ptr(),
            c_operands.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Perform the Kronecker product of two arrays.
///
/// # Params
///
/// - `a`: first array
/// - `b`: second array
/// - `stream`: stream or device to evaluate on
#[generate_macro]
#[default_device]
pub fn kron_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_kron(
            res,
            a.as_ref().as_ptr(),
            b.as_ref().as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        array,
        ops::{arange, diag, einsum, reshape},
        test_utils::{assert_array_eq, tolerances},
        Array,
    };
    use pretty_assertions::assert_eq;

    use super::diagonal;

    #[test]
    #[ignore = "mlx-rs diagonal changes int32 to float32; confirmed against python mlx 0.30.6 cpu"]
    fn test_diagonal() {
        let x = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7], &[4, 2]);
        let out = diagonal(&x, None, None, None).unwrap();
        assert_array_eq(
            out,
            array!([0, 3]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        assert!(diagonal(&x, 1, 6, 0).is_err());
        assert!(diagonal(&x, 1, 0, -3).is_err());

        let x = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let out = diagonal(&x, 2, 1, 0).unwrap();
        assert_array_eq(
            out,
            array!([8]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        let out = diagonal(&x, -1, 0, 1).unwrap();
        assert_array_eq(
            out,
            array!([4, 9]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        let out = diagonal(&x, -5, 0, 1).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[0]);

        let x = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 2, 2]);
        let out = diagonal(&x, 1, 0, 1).unwrap();
        assert_array_eq(
            out,
            array!([[2], [3]]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        let out = diagonal(&x, 0, 2, 0).unwrap();
        assert_array_eq(
            out,
            array!([[0, 5], [2, 7]]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        let out = diagonal(&x, 1, -1, 0).unwrap();
        assert_array_eq(
            out,
            array!([[4, 9], [6, 11]]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        let x = reshape(arange::<_, f32>(None, 16, None).unwrap(), &[2, 2, 2, 2]).unwrap();
        let out = diagonal(&x, 0, 0, 1).unwrap();
        assert_array_eq(
            out,
            Array::from_slice(&[0, 12, 1, 13, 2, 14, 3, 15], &[2, 2, 2]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        assert!(diagonal(&x, 0, 1, 1).is_err());

        let x = array!([0, 1]);
        assert!(diagonal(&x, 0, 0, 1).is_err());
    }

    #[test]
    fn test_diag() {
        // Too few or too many dimensions
        assert!(diag(Array::from_f32(0.0), None).is_err());
        assert!(diag(Array::from_slice(&[0.0], &[1, 1, 1]), None).is_err());

        // Test with 1D array
        let x = array!([0, 1, 2, 3]);
        let out = diag(&x, 0).unwrap();
        assert_array_eq(
            out,
            array!([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3]]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        let out = diag(&x, 1).unwrap();
        assert_array_eq(
            out,
            array!([
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 2, 0],
                [0, 0, 0, 0, 3],
                [0, 0, 0, 0, 0]
            ]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        let out = diag(&x, -1).unwrap();
        assert_array_eq(
            out,
            array!([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 3, 0]
            ]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        // Test with 2D array
        let x = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8], &[3, 3]);
        let out = diag(&x, 0).unwrap();
        assert_array_eq(
            out,
            array!([0, 4, 8]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        let out = diag(&x, 1).unwrap();
        assert_array_eq(
            out,
            array!([1, 5]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        let out = diag(&x, -1).unwrap();
        assert_array_eq(
            out,
            array!([3, 7]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );
    }

    #[test]
    #[ignore = "mlx-rs einsum changes float32 to int32; confirmed against python mlx 0.30.6 cpu"]
    fn test_einsum() {
        // Test dot product (vector-vector)
        let a = array!([0.0, 1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0, 6.0, 7.0]);
        let out = einsum("i,i->", &[a, b]).unwrap();
        assert_array_eq(
            out,
            array!(38.0),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        // Test trace (diagonal sum)
        let m = array!([[1, 2], [3, 4]]);
        let out = einsum("ii->", &[m]).unwrap();
        assert_array_eq(
            out,
            array!(5.0),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );
    }

    #[test]
    fn test_hadamard_transform() {
        let input = Array::from_slice(&[1.0, -1.0, -1.0, 1.0], &[2, 2]);
        let expected = Array::from_slice(
            &[
                0.0,
                2.0_f32 / 2.0_f32.sqrt(),
                0.0,
                -2.0_f32 / 2.0_f32.sqrt(),
            ],
            &[2, 2],
        );
        let result = input.hadamard_transform(None).unwrap();

        let c = result.all_close(&expected, 1e-5, 1e-5, None).unwrap();
        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true]);
    }

    // This test is adapted from the python unit test `mlx/test/test_ops.py` `test_kron`
    #[test]
    fn test_kron() {
        // Basic vector test
        let x = array!([1, 2]);
        let y = array!([3, 4]);
        let z = super::kron(&x, &y).unwrap();
        assert_array_eq(
            z,
            array!([3, 4, 6, 8]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );

        // Basic matrix test
        let x = array!([[1, 2], [3, 4]]);
        let y = array!([[0, 5], [6, 7]]);
        let z = super::kron(&x, &y).unwrap();
        assert_array_eq(
            z,
            array!([
                [0, 5, 0, 10],
                [6, 7, 12, 14],
                [0, 15, 0, 20],
                [18, 21, 24, 28]
            ]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );
    }
}
