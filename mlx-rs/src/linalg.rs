//! Linear algebra operations.

use crate::error::{Exception, Result};
use crate::utils::guard::Guarded;
use crate::utils::{IntoOption, VectorArray};
use crate::{Array, Stream};
use mlx_internal_macros::{default_device, generate_macro};
use std::f64;
use std::ffi::CString;

/// Order of the norm
///
/// See [`norm`] for more details.
#[derive(Debug, Clone, Copy)]
pub enum Ord<'a> {
    /// String representation of the order
    Str(&'a str),

    /// Order of the norm
    P(f64),
}

impl Default for Ord<'_> {
    fn default() -> Self {
        Ord::Str("fro")
    }
}

impl std::fmt::Display for Ord<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ord::Str(s) => write!(f, "{s}"),
            Ord::P(p) => write!(f, "{p}"),
        }
    }
}

impl<'a> From<&'a str> for Ord<'a> {
    fn from(value: &'a str) -> Self {
        Ord::Str(value)
    }
}

impl From<f64> for Ord<'_> {
    fn from(value: f64) -> Self {
        Ord::P(value)
    }
}

impl<'a> IntoOption<Ord<'a>> for &'a str {
    fn into_option(self) -> Option<Ord<'a>> {
        Some(Ord::Str(self))
    }
}

impl<'a> IntoOption<Ord<'a>> for f64 {
    fn into_option(self) -> Option<Ord<'a>> {
        Some(Ord::P(self))
    }
}

/// Compute p-norm of an [`Array`]
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn norm_device<'a>(
    array: impl AsRef<Array>,
    ord: f64,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let keep_dims = keep_dims.into().unwrap_or(false);

    match axes.into_option() {
        Some(axes) => Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_linalg_norm(
                res,
                array.as_ref().as_ptr(),
                ord,
                axes.as_ptr(),
                axes.len(),
                keep_dims,
                stream.as_ref().as_ptr(),
            )
        }),
        None => Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_linalg_norm(
                res,
                array.as_ref().as_ptr(),
                ord,
                std::ptr::null(),
                0,
                keep_dims,
                stream.as_ref().as_ptr(),
            )
        }),
    }
}

/// Matrix or vector norm.
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn norm_matrix_device<'a>(
    array: impl AsRef<Array>,
    ord: &'a str,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let ord = CString::new(ord).map_err(|e| Exception::custom(format!("{e}")))?;
    let keep_dims = keep_dims.into().unwrap_or(false);

    match axes.into_option() {
        Some(axes) => Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_linalg_norm_matrix(
                res,
                array.as_ref().as_ptr(),
                ord.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims,
                stream.as_ref().as_ptr(),
            )
        }),
        None => Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_linalg_norm_matrix(
                res,
                array.as_ref().as_ptr(),
                ord.as_ptr(),
                std::ptr::null(),
                0,
                keep_dims,
                stream.as_ref().as_ptr(),
            )
        }),
    }
}

/// Compute the L2 norm of an [`Array`]
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn norm_l2_device<'a>(
    array: impl AsRef<Array>,
    #[optional] axes: impl IntoOption<&'a [i32]>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let keep_dims = keep_dims.into().unwrap_or(false);

    match axes.into_option() {
        Some(axis) => Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_linalg_norm_l2(
                res,
                array.as_ref().as_ptr(),
                axis.as_ptr(),
                axis.len(),
                keep_dims,
                stream.as_ref().as_ptr(),
            )
        }),
        None => Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_linalg_norm_l2(
                res,
                array.as_ref().as_ptr(),
                std::ptr::null(),
                0,
                keep_dims,
                stream.as_ref().as_ptr(),
            )
        }),
    }
}

// TODO: Change the original `norm` function to use builder pattern
// /// Matrix or vector norm.
// ///
// /// For values of `ord < 1`, the result is, strictly speaking, not a
// /// mathematical norm, but it may still be useful for various numerical
// /// purposes.
// ///
// /// The following norms can be calculated:
// ///
// /// ord   | norm for matrices            | norm for vectors
// /// ----- | ---------------------------- | --------------------------
// /// None  | Frobenius norm               | 2-norm
// /// 'fro' | Frobenius norm               | --
// /// inf   | max(sum(abs(x), axis-1))     | max(abs(x))
// /// -inf  | min(sum(abs(x), axis-1))     | min(abs(x))
// /// 0     | --                           | sum(x !- 0)
// /// 1     | max(sum(abs(x), axis-0))     | as below
// /// -1    | min(sum(abs(x), axis-0))     | as below
// /// 2     | 2-norm (largest sing. value) | as below
// /// -2    | smallest singular value      | as below
// /// other | --                           | sum(abs(x)**ord)**(1./ord)
// ///
// /// > Nuclear norm and norms based on singular values are not yet implemented.
// ///
// /// The Frobenius norm is given by G. H. Golub and C. F. Van Loan, *Matrix Computations*,
// ///        Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15
// ///
// /// The nuclear norm is the sum of the singular values.
// ///
// /// Both the Frobenius and nuclear norm orders are only defined for
// /// matrices and produce a fatal error when `array.ndim != 2`
// ///
// /// # Params
// ///
// /// - `array`: input array
// /// - `ord`: order of the norm, see table
// /// - `axes`: axes that hold 2d matrices
// /// - `keep_dims`: if `true` the axes which are normed over are left in the result as dimensions
// ///   with size one
// #[generate_macro(customize(root = "$crate::linalg"))]
// #[default_device]
// pub fn norm_device<'a>(
//     array: impl AsRef<Array>,
//     #[optional] ord: impl IntoOption<Ord<'a>>,
//     #[optional] axes: impl IntoOption<&'a [i32]>,
//     #[optional] keep_dims: impl Into<Option<bool>>,
//     #[optional] stream: impl AsRef<Stream>,
// ) -> Result<Array> {
//     let ord = ord.into_option();
//     let axes = axes.into_option();
//     let keep_dims = keep_dims.into().unwrap_or(false);

//     match (ord, axes) {
//         // If axis and ord are both unspecified, computes the 2-norm of flatten(x).
//         (None, None) => {
//             let axes_ptr = std::ptr::null(); // mlx-c already handles the case where axes is null
//             Array::try_from_op(|res| unsafe {
//                 mlx_sys::mlx_linalg_norm(
//                     res,
//                     array.as_ref().as_ptr(),
//                     axes_ptr,
//                     0,
//                     keep_dims,
//                     stream.as_ref().as_ptr(),
//                 )
//             })
//         }
//         // If axis is not provided but ord is, then x must be either 1D or 2D.
//         //
//         // Frobenius norm is only supported for matrices
//         (Some(Ord::Str(ord)), None) => norm_ord_device(array, ord, axes, keep_dims, stream),
//         (Some(Ord::P(p)), None) => norm_p_device(array, p, axes, keep_dims, stream),
//         // If axis is provided, but ord is not, then the 2-norm (or Frobenius norm for matrices) is
//         // computed along the given axes. At most 2 axes can be specified.
//         (None, Some(axes)) => Array::try_from_op(|res| unsafe {
//             mlx_sys::mlx_linalg_norm(
//                 res,
//                 array.as_ref().as_ptr(),
//                 axes.as_ptr(),
//                 axes.len(),
//                 keep_dims,
//                 stream.as_ref().as_ptr(),
//             )
//         }),
//         // If both axis and ord are provided, then the corresponding matrix or vector
//         // norm is computed. At most 2 axes can be specified.
//         (Some(Ord::Str(ord)), Some(axes)) => norm_ord_device(array, ord, axes, keep_dims, stream),
//         (Some(Ord::P(p)), Some(axes)) => norm_p_device(array, p, axes, keep_dims, stream),
//     }
// }

/// The QR factorization of the input matrix. Returns an error if the input is not valid.
///
/// This function supports arrays with at least 2 dimensions. The matrices which are factorized are
/// assumed to be in the last two dimensions of the input.
///
/// Evaluation on the GPU is not yet implemented.
///
/// # Params
///
/// - `array`: input array
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, StreamOrDevice, linalg::*};
///
/// let a = Array::from_slice(&[2.0f32, 3.0, 1.0, 2.0], &[2, 2]);
///
/// let (q, r) = qr_device(&a, StreamOrDevice::cpu()).unwrap();
///
/// let q_expected = Array::from_slice(&[-0.894427, -0.447214, -0.447214, 0.894427], &[2, 2]);
/// let r_expected = Array::from_slice(&[-2.23607, -3.57771, 0.0, 0.447214], &[2, 2]);
///
/// assert!(q.all_close(&q_expected, None, None, None).unwrap().item::<bool>());
/// assert!(r.all_close(&r_expected, None, None, None).unwrap().item::<bool>());
/// ```
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn qr_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<(Array, Array)> {
    <(Array, Array)>::try_from_op(|(res_0, res_1)| unsafe {
        mlx_sys::mlx_linalg_qr(res_0, res_1, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// The Singular Value Decomposition (SVD) of the input matrix. Returns an error if the input is not
/// valid.
///
/// This function supports arrays with at least 2 dimensions. When the input has more than two
/// dimensions, the function iterates over all indices of the first a.ndim - 2 dimensions and for
/// each combination SVD is applied to the last two indices.
///
/// Evaluation on the GPU is not yet implemented.
///
/// # Params
///
/// - `array`: input array
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, StreamOrDevice, linalg::*};
///
/// let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
/// let (u, s, vt) = svd_device(&a, StreamOrDevice::cpu()).unwrap();
/// let u_expected = Array::from_slice(&[-0.404554, 0.914514, -0.914514, -0.404554], &[2, 2]);
/// let s_expected = Array::from_slice(&[5.46499, 0.365966], &[2]);
/// let vt_expected = Array::from_slice(&[-0.576048, -0.817416, -0.817415, 0.576048], &[2, 2]);
/// assert!(u.all_close(&u_expected, None, None, None).unwrap().item::<bool>());
/// assert!(s.all_close(&s_expected, None, None, None).unwrap().item::<bool>());
/// assert!(vt.all_close(&vt_expected, None, None, None).unwrap().item::<bool>());
/// ```
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn svd_device(
    array: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<(Array, Array, Array)> {
    let v = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_linalg_svd(res, array.as_ref().as_ptr(), true, stream.as_ref().as_ptr())
    })?;

    let [u, s, vt] = v.try_into_array::<3>()?;
    Ok((u, s, vt))
}

/// Compute the inverse of a square matrix. Returns an error if the input is not valid.
///
/// This function supports arrays with at least 2 dimensions. When the input has more than two
/// dimensions, the inverse is computed for each matrix in the last two dimensions of `a`.
///
/// Evaluation on the GPU is not yet implemented.
///
/// # Params
///
/// - `a`: input array
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, StreamOrDevice, linalg::*};
///
/// let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
/// let a_inv = inv_device(&a, StreamOrDevice::cpu()).unwrap();
/// let expected = Array::from_slice(&[-2.0, 1.0, 1.5, -0.5], &[2, 2]);
/// assert!(a_inv.all_close(&expected, None, None, None).unwrap().item::<bool>());
/// ```
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn inv_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_linalg_inv(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Compute the Cholesky decomposition of a real symmetric positive semi-definite matrix.
///
/// This function supports arrays with at least 2 dimensions. When the input has more than two
/// dimensions, the Cholesky decomposition is computed for each matrix in the last two dimensions of
/// `a`.
///
/// If the input matrix is not symmetric positive semi-definite, behaviour is undefined.
///
/// # Params
///
/// - `a`: input array
/// - `upper`: If `true`, return the upper triangular Cholesky factor. If `false`, return the lower
///   triangular Cholesky factor. Default: `false`.
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn cholesky_device(
    a: impl AsRef<Array>,
    #[optional] upper: Option<bool>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let upper = upper.unwrap_or(false);
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_linalg_cholesky(res, a.as_ref().as_ptr(), upper, stream.as_ref().as_ptr())
    })
}

/// Compute the inverse of a real symmetric positive semi-definite matrix using it’s Cholesky decomposition.
///
/// Please see the python documentation for more details.
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn cholesky_inv_device(
    a: impl AsRef<Array>,
    #[optional] upper: Option<bool>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let upper = upper.unwrap_or(false);
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_linalg_cholesky_inv(res, a.as_ref().as_ptr(), upper, stream.as_ref().as_ptr())
    })
}

/// Compute the cross product of two arrays along a specified axis.
///
/// The cross product is defined for arrays with size 2 or 3 in the specified axis. If the size is 2
/// then the third value is assumed to be zero.
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn cross_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] axis: Option<i32>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let axis = axis.unwrap_or(-1);
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_linalg_cross(
            res,
            a.as_ref().as_ptr(),
            b.as_ref().as_ptr(),
            axis,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compute the eigenvalues and eigenvectors of a complex Hermitian or real symmetric matrix.
///
/// This function supports arrays with at least 2 dimensions. When the input has more than two
/// dimensions, the eigenvalues and eigenvectors are computed for each matrix in the last two
/// dimensions.
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn eigh_device(
    a: impl AsRef<Array>,
    #[optional] uplo: Option<&str>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<(Array, Array)> {
    let a = a.as_ref();
    let uplo = CString::new(uplo.unwrap_or("L")).map_err(|e| Exception::custom(format!("{e}")))?;

    <(Array, Array) as Guarded>::try_from_op(|(res_0, res_1)| unsafe {
        mlx_sys::mlx_linalg_eigh(
            res_0,
            res_1,
            a.as_ptr(),
            uplo.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compute the eigenvalues of a complex Hermitian or real symmetric matrix.
///
/// This function supports arrays with at least 2 dimensions. When the input has more than two
/// dimensions, the eigenvalues are computed for each matrix in the last two dimensions.
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn eigvalsh_device(
    a: impl AsRef<Array>,
    #[optional] uplo: Option<&str>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let uplo = CString::new(uplo.unwrap_or("L")).map_err(|e| Exception::custom(format!("{e}")))?;
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_linalg_eigvalsh(res, a.as_ptr(), uplo.as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Compute the eigenvalues and eigenvectors of a square matrix.
///
/// This function supports arrays with at least 2 dimensions. When the input has more than two
/// dimensions, the eigenvalues and eigenvectors are computed for each matrix in the last two
/// dimensions.
///
/// Unlike [`eigh`], this function computes eigenvalues for general (not necessarily symmetric
/// or Hermitian) matrices. The eigenvalues and eigenvectors may be complex.
///
/// # Params
///
/// - `a`: Input array. Must be a square matrix.
///
/// # Returns
///
/// A tuple `(eigenvalues, eigenvectors)` where eigenvalues has shape `(..., N)` and
/// eigenvectors has shape `(..., N, N)`. The eigenvectors are stored as columns.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, linalg::*, StreamOrDevice};
///
/// let a = Array::from_slice(&[1.0f32, 1.0, 3.0, 4.0], &[2, 2]);
/// let (eigenvalues, eigenvectors) = eig_device(&a, StreamOrDevice::cpu()).unwrap();
/// // eigenvalues and eigenvectors are complex even for real input
/// ```
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn eig_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<(Array, Array)> {
    <(Array, Array) as Guarded>::try_from_op(|(res_0, res_1)| unsafe {
        mlx_sys::mlx_linalg_eig(res_0, res_1, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Compute the eigenvalues of a square matrix.
///
/// This function supports arrays with at least 2 dimensions. When the input has more than two
/// dimensions, the eigenvalues are computed for each matrix in the last two dimensions.
///
/// Unlike [`eigvalsh`], this function computes eigenvalues for general (not necessarily symmetric
/// or Hermitian) matrices. The eigenvalues may be complex.
///
/// # Params
///
/// - `a`: Input array. Must be a square matrix.
///
/// # Returns
///
/// An array of eigenvalues with shape `(..., N)`.
///
/// # Example
///
/// ```rust
/// use mlx_rs::{Array, linalg::*, StreamOrDevice};
///
/// let a = Array::from_slice(&[1.0f32, 1.0, 3.0, 4.0], &[2, 2]);
/// let eigenvalues = eigvals_device(&a, StreamOrDevice::cpu()).unwrap();
/// ```
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn eigvals_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_linalg_eigvals(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Compute the (Moore-Penrose) pseudo-inverse of a matrix.
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn pinv_device(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_linalg_pinv(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Compute the inverse of a triangular square matrix.
///
/// This function supports arrays with at least 2 dimensions. When the input has more than two
/// dimensions, the inverse is computed for each matrix in the last two dimensions of a.
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn tri_inv_device(
    a: impl AsRef<Array>,
    #[optional] upper: Option<bool>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let upper = upper.unwrap_or(false);
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_linalg_tri_inv(res, a.as_ref().as_ptr(), upper, stream.as_ref().as_ptr())
    })
}

/// Compute the LU factorization of the given matrix A.
///
/// Note, unlike the default behavior of scipy.linalg.lu, the pivots are
/// indices. To reconstruct the input use L[P, :] @ U for 2 dimensions or
/// mx.take_along_axis(L, P[..., None], axis=-2) @ U for more than 2 dimensions.
///
/// To construct the full permuation matrix do:
///
/// ```rust,ignore
/// // python
/// // P = mx.put_along_axis(mx.zeros_like(L), p[..., None], mx.array(1.0), axis=-1)
/// let p = mlx_rs::ops::put_along_axis(
///     mlx_rs::ops::zeros_like(&l),
///     p.index((Ellipsis, NewAxis)),
///     array!(1.0),
///     -1,
/// ).unwrap();
/// ```
///
/// # Params
///
/// - `a`: input array
/// - `stream`: stream to execute the operation
///
/// # Returns
///
/// The `p`, `L`, and `U` arrays, such that `A = L[P, :] @ U`
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn lu_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<(Array, Array, Array)> {
    let v = Vec::<Array>::try_from_op(|res| unsafe {
        mlx_sys::mlx_linalg_lu(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })?;
    let mut iter = v.into_iter();
    let p = iter.next().ok_or_else(|| Exception::custom("missing P"))?;
    let l = iter.next().ok_or_else(|| Exception::custom("missing L"))?;
    let u = iter.next().ok_or_else(|| Exception::custom("missing U"))?;
    Ok((p, l, u))
}

/// Computes a compact representation of the LU factorization.
///
/// # Params
///
/// - `a`: input array
/// - `stream`: stream to execute the operation
///
/// # Returns
///
/// The `LU` matrix and `pivots` array.
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn lu_factor_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<(Array, Array)> {
    <(Array, Array)>::try_from_op(|(res_0, res_1)| unsafe {
        mlx_sys::mlx_linalg_lu_factor(res_0, res_1, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Compute the solution to a system of linear equations `AX = B`
///
/// # Params
///
/// - `a`: input array
/// - `b`: input array
/// - `stream`: stream to execute the operation
///
/// # Returns
///
/// The unique solution to the system `AX = B`
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn solve_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_linalg_solve(
            res,
            a.as_ref().as_ptr(),
            b.as_ref().as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Computes the solution of a triangular system of linear equations `AX = B`
///
/// # Params
///
/// - `a`: input array
/// - `b`: input array
/// - `upper`: whether the matrix is upper triangular. Default: `false`
/// - `stream`: stream to execute the operation
///
/// # Returns
///
/// The unique solution to the system `AX = B`
#[generate_macro(customize(root = "$crate::linalg"))]
#[default_device]
pub fn solve_triangular_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] upper: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let upper = upper.into().unwrap_or(false);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_linalg_solve_triangular(
            res,
            a.as_ref().as_ptr(),
            b.as_ref().as_ptr(),
            upper,
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use crate::{
        array,
        ops::{eye, indexing::IndexOp, tril, triu},
        StreamOrDevice,
    };

    use super::*;

    // The tests below are adapted from the swift bindings tests
    // and they are not exhaustive. Additional tests should be added
    // to cover the error cases

    #[test]
    fn test_norm_no_axes() {
        let a = Array::from_iter(0..9, &[9]) - 4;
        let b = a.reshape(&[3, 3]).unwrap();

        assert_float_eq!(
            norm_l2(&a, None, None).unwrap().item::<f32>(),
            7.74597,
            abs <= 0.001
        );
        assert_float_eq!(
            norm_l2(&b, None, None).unwrap().item::<f32>(),
            7.74597,
            abs <= 0.001
        );

        assert_float_eq!(
            norm_matrix(&b, "fro", None, None).unwrap().item::<f32>(),
            7.74597,
            abs <= 0.001
        );

        assert_float_eq!(
            norm(&a, f64::INFINITY, None, None).unwrap().item::<f32>(),
            4.0,
            abs <= 0.001
        );
        assert_float_eq!(
            norm(&b, f64::INFINITY, None, None).unwrap().item::<f32>(),
            9.0,
            abs <= 0.001
        );

        assert_float_eq!(
            norm(&a, f64::NEG_INFINITY, None, None)
                .unwrap()
                .item::<f32>(),
            0.0,
            abs <= 0.001
        );
        assert_float_eq!(
            norm(&b, f64::NEG_INFINITY, None, None)
                .unwrap()
                .item::<f32>(),
            2.0,
            abs <= 0.001
        );

        assert_float_eq!(
            norm(&a, 1.0, None, None).unwrap().item::<f32>(),
            20.0,
            abs <= 0.001
        );
        assert_float_eq!(
            norm(&b, 1.0, None, None).unwrap().item::<f32>(),
            7.0,
            abs <= 0.001
        );

        assert_float_eq!(
            norm(&a, -1.0, None, None).unwrap().item::<f32>(),
            0.0,
            abs <= 0.001
        );
        assert_float_eq!(
            norm(&b, -1.0, None, None).unwrap().item::<f32>(),
            6.0,
            abs <= 0.001
        );
    }

    #[test]
    fn test_norm_axis() {
        let c = Array::from_slice(&[1, 2, 3, -1, 1, 4], &[2, 3]);

        let result = norm_l2(&c, &[0], None).unwrap();
        let expected = Array::from_slice(&[1.41421, 2.23607, 5.0], &[3]);
        assert!(result
            .all_close(&expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn test_norm_axes() {
        let m = Array::from_iter(0..8, &[2, 2, 2]);

        let result = norm_l2(&m, &[1, 2][..], None).unwrap();
        let expected = Array::from_slice(&[3.74166, 11.225], &[2]);
        assert!(result
            .all_close(&expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn test_qr() {
        let a = Array::from_slice(&[2.0f32, 3.0, 1.0, 2.0], &[2, 2]);

        let (q, r) = qr_device(&a, StreamOrDevice::cpu()).unwrap();

        let q_expected = Array::from_slice(&[-0.894427, -0.447214, -0.447214, 0.894427], &[2, 2]);
        let r_expected = Array::from_slice(&[-2.23607, -3.57771, 0.0, 0.447214], &[2, 2]);

        assert!(q
            .all_close(&q_expected, None, None, None)
            .unwrap()
            .item::<bool>());
        assert!(r
            .all_close(&r_expected, None, None, None)
            .unwrap()
            .item::<bool>());
    }

    // The tests below are adapted from the c++ tests

    #[test]
    fn test_svd() {
        // eval_gpu is not implemented yet.
        let stream = StreamOrDevice::cpu();

        // 0D and 1D returns error
        let a = Array::from_f32(0.0);
        assert!(svd_device(&a, &stream).is_err());

        let a = Array::from_slice(&[0.0, 1.0], &[2]);
        assert!(svd_device(&a, &stream).is_err());

        // Unsupported types returns error
        let a = Array::from_slice(&[0, 1], &[1, 2]);
        assert!(svd_device(&a, &stream).is_err());

        // TODO: wait for random
    }

    #[test]
    fn test_inv() {
        // eval_gpu is not implemented yet.
        let stream = StreamOrDevice::cpu();

        // 0D and 1D returns error
        let a = Array::from_f32(0.0);
        assert!(inv_device(&a, &stream).is_err());

        let a = Array::from_slice(&[0.0, 1.0], &[2]);
        assert!(inv_device(&a, &stream).is_err());

        // Unsupported types returns error
        let a = Array::from_slice(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        assert!(inv_device(&a, &stream).is_err());

        // TODO: wait for random
    }

    #[test]
    fn test_cholesky() {
        // eval_gpu is not implemented yet.
        let stream = StreamOrDevice::cpu();

        // 0D and 1D returns error
        let a = Array::from_f32(0.0);
        assert!(cholesky_device(&a, None, &stream).is_err());

        let a = Array::from_slice(&[0.0, 1.0], &[2]);
        assert!(cholesky_device(&a, None, &stream).is_err());

        // Unsupported types returns error
        let a = Array::from_slice(&[0, 1, 1, 2], &[2, 2]);
        assert!(cholesky_device(&a, None, &stream).is_err());

        // Non-square returns error
        let a = Array::from_slice(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        assert!(cholesky_device(&a, None, &stream).is_err());

        // TODO: wait for random
    }

    // The unit test below is adapted from the python unit test `test_linalg.py/test_lu`
    #[test]
    fn test_lu() {
        let scalar = array!(1.0);
        let result = lu_device(&scalar, StreamOrDevice::cpu());
        assert!(result.is_err());

        // # Test 3x3 matrix
        let a = array!([[3.0f32, 1.0, 2.0], [1.0, 8.0, 6.0], [9.0, 2.0, 5.0]]);
        let (p, l, u) = lu_device(&a, StreamOrDevice::cpu()).unwrap();
        let a_rec = l.index((p, ..)).matmul(u).unwrap();
        assert_array_all_close!(a, a_rec);
    }

    // The unit test below is adapted from the python unit test `test_linalg.py/test_lu_factor`
    #[test]
    fn test_lu_factor() {
        crate::random::seed(7).unwrap();

        // Test 3x3 matrix
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[5, 5], None).unwrap();
        let (lu, pivots) = lu_factor_device(&a, StreamOrDevice::cpu()).unwrap();
        let shape = a.shape();
        let n = shape[shape.len() - 1];

        let pivots: Vec<u32> = pivots.as_slice().to_vec();
        let mut perm: Vec<u32> = (0..n as u32).collect();
        for (i, p) in pivots.iter().enumerate() {
            perm.swap(i, *p as usize);
        }

        let l = tril(&lu, -1)
            .and_then(|l| l.add(eye::<f32>(n, None, None)?))
            .unwrap();
        let u = triu(&lu, None).unwrap();

        let lhs = l.matmul(&u).unwrap();
        let perm = Array::from_slice(&perm, &[n]);
        let rhs = a.index((perm, ..));
        assert_array_all_close!(lhs, rhs);
    }

    // The unit test below is adapted from the python unit test `test_linalg.py/test_solve`
    #[test]
    fn test_solve() {
        crate::random::seed(7).unwrap();

        // Test 3x3 matrix with 1D rhs
        let a = array!([[3.0f32, 1.0, 2.0], [1.0, 8.0, 6.0], [9.0, 2.0, 5.0]]);
        let b = array!([11.0f32, 35.0, 28.0]);

        let result = solve_device(&a, &b, StreamOrDevice::cpu()).unwrap();
        let expected = array!([1.0f32, 2.0, 3.0]);
        assert_array_all_close!(result, expected);
    }

    #[test]
    fn test_solve_triangular() {
        let a = array!([[4.0f32, 0.0, 0.0], [2.0, 3.0, 0.0], [1.0, -2.0, 5.0]]);
        let b = array!([8.0f32, 14.0, 3.0]);

        let result = solve_triangular_device(&a, &b, false, StreamOrDevice::cpu()).unwrap();
        let expected = array!([2.0f32, 3.333_333_3, 1.533_333_3]);
        assert_array_all_close!(result, expected);
    }

    // The tests below are adapted from the python unit test `test_linalg.py/test_eig`
    #[test]
    fn test_eig() {
        use crate::ops::expand_dims;

        // Helper to check eigenvalues and eigenvectors
        fn check_eigs_and_vecs(a: &Array) {
            let (eig_vals, eig_vecs) = eig_device(a, StreamOrDevice::cpu()).unwrap();

            // Check A @ eig_vecs == eig_vals * eig_vecs
            let lhs = a.matmul(&eig_vecs).unwrap();
            // eig_vals[..., None, :] * eig_vecs - broadcast eigenvalues
            // For a 1D eigenvalues array (n,), we need shape (1, n) to broadcast with eigenvectors (n, n)
            // For batched eigenvalues (..., n), we need shape (..., 1, n)
            let eig_vals_broadcast = expand_dims(&eig_vals, -2).unwrap();
            let rhs = eig_vals_broadcast.multiply(&eig_vecs).unwrap();
            assert!(
                lhs.all_close(&rhs, 1e-4, 1e-4, None)
                    .unwrap()
                    .item::<bool>(),
                "A @ eig_vecs should equal eig_vals * eig_vecs"
            );

            // Check eigvals returns same values
            let eig_vals_only = eigvals_device(a, StreamOrDevice::cpu()).unwrap();
            assert!(
                eig_vals
                    .all_close(&eig_vals_only, 1e-4, 1e-4, None)
                    .unwrap()
                    .item::<bool>(),
                "eigvals should return same eigenvalues as eig"
            );
        }

        // Test a simple 2x2 matrix
        let a = array!([[1.0f32, 1.0], [3.0, 4.0]]);
        check_eigs_and_vecs(&a);

        // Test complex eigenvalues (rotation-like matrix)
        let a = array!([[1.0f32, -1.0], [1.0, 1.0]]);
        check_eigs_and_vecs(&a);

        // Test a larger random matrix
        crate::random::seed(1).unwrap();
        let a = crate::random::normal::<f32>(&[5, 5], None, None, None).unwrap();
        check_eigs_and_vecs(&a);

        // Test with batched input
        let a = crate::random::normal::<f32>(&[3, 5, 5], None, None, None).unwrap();
        check_eigs_and_vecs(&a);
    }

    #[test]
    fn test_eig_errors() {
        // 1D array should fail
        let a = array!([1.0f32, 2.0]);
        assert!(eig_device(&a, StreamOrDevice::cpu()).is_err());
        assert!(eigvals_device(&a, StreamOrDevice::cpu()).is_err());

        // Non-square matrix should fail
        let a = array!([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        assert!(eig_device(&a, StreamOrDevice::cpu()).is_err());
        assert!(eigvals_device(&a, StreamOrDevice::cpu()).is_err());
    }
}
