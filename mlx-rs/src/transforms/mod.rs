//! Function transforms
//!
//! This mod provides functions for automatic differentiation and other
//! transformations on functions.
//!
//! **WARN**: Because function transforms including compilation works on
//! the computation graph, the user must ensure that all `Array`s are passed
//! as inputs to the function/closure. Closures with captured `Array`s may
//! not work as expected and may lead to undefined behavior.
//!
//! # Automatic Differentiation
//!
//! Automatic differentiation in MLX works on functions rather than on implicit
//! graphs.
//!
//! **NOTE**: If you are coming to MLX from PyTorch, you no longer need
//! functions like backward, zero_grad, and detach, or properties like
//! requires_grad.
//!
//! You can use the [`grad()`] and [`value_and_grad()`] function to compute
//! gradients of more complex functions. These functions compute the gradient
//! with respect to the first argument, in order to manually specify the the
//! argument to compute the gradient with respect to, use
//! [`grad_with_argnums()`] or [`value_and_grad_with_argnums()`].
//!
//! # Panics
//!
//! A panic in a Rust transform closure is caught before it reaches MLX. After MLX returns through
//! the C ABI, the panic resumes in Rust with its original payload.
//!
//! TODO: update the example once https://github.com/oxiglade/mlx-rs/pull/218 is merged
//!
//! ```rust,ignore
//! use mlx_rs::{Array, error::Result, transforms::grad};
//!
//! fn f(x: &Array) -> Result<Array> {
//!     x.square()
//! }
//!
//! fn calculate_grad(func: impl Fn(&Array) -> Result<Array>, arg: &Array) -> Result<Array> {
//!     grad(&func, &[0])(arg)
//! }
//!
//! let x = Array::from(1.5);
//!
//! let dfdx = calculate_grad(f, &x).unwrap();
//! assert_eq!(dfdx.item_exact::<f32>(), 2.0 * 1.5);
//!
//! let dfdx2 = calculate_grad(|args| calculate_grad(f, args), &x).unwrap();
//! assert_eq!(dfdx2.item_exact::<f32>(), 2.0);
//! ```

use mlx_sys::mlx_closure_value_and_grad;

use crate::{
    error::{get_and_clear_closure_error, Result},
    module::ModuleParamRef,
    utils::{guard::Guarded, Closure, VectorArray, SUCCESS},
    Array,
};

pub mod compile;
mod grad;
mod keyed_value_and_grad;
mod value_and_grad;

pub use grad::*;
pub use keyed_value_and_grad::*;
pub use value_and_grad::*;

/// Evaluate an iterator of [`Array`]s.
pub fn eval<'a>(outputs: impl IntoIterator<Item = &'a Array>) -> Result<()> {
    let vec = VectorArray::try_from_iter(outputs.into_iter())?;
    <() as Guarded>::try_from_op(|_| unsafe { mlx_sys::mlx_eval(vec.as_ptr()) })
}

/// Evaluate a module's parameters.
///
/// This is a convenience function that flattens the parameters and evaluates them.
pub fn eval_params(params: ModuleParamRef<'_>) -> Result<()> {
    eval(params.flatten().values().copied())
}

/// Asynchronously evaluate an iterator of [`Array`]s.
///
/// Please note that this is not a rust async function.
pub fn async_eval<'a>(outputs: impl IntoIterator<Item = &'a Array>) -> Result<()> {
    let vec = VectorArray::try_from_iter(outputs.into_iter())?;
    <() as Guarded>::try_from_op(|_| unsafe { mlx_sys::mlx_async_eval(vec.as_ptr()) })
}

/// Asynchronously evaluate a module's parameters.
///
/// This is a convenience function that flattens the parameters and evaluates them.
pub fn async_eval_params(params: ModuleParamRef<'_>) -> Result<()> {
    async_eval(params.flatten().values().copied())
}

#[inline]
fn jvp_inner(
    closure: Closure<'_>,
    primals: &[Array],
    tangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)> {
    let c_primals = VectorArray::try_from_iter(primals.iter())?;
    let c_tangents = VectorArray::try_from_iter(tangents.iter())?;

    <(Vec<Array>, Vec<Array>) as Guarded>::try_from_op(|(res_0, res_1)| unsafe {
        mlx_sys::mlx_jvp(
            res_0,
            res_1,
            closure.as_ptr(),
            c_primals.as_ptr(),
            c_tangents.as_ptr(),
        )
    })
    .map_err(|e| match get_and_clear_closure_error() {
        Some(err) => err,
        None => e,
    })
}

/// Compute the Jacobian-vector product.
///
/// This computes the product of the Jacobian of a function `f` evaluated at
/// `primals` with the `tangents`.
///
/// # Params:
///
/// - `f`: function which takes an array of `Array` and returns an array of
///   `Array`
/// - `primals`: array of `Array` at which to evaluate the Jacobian
/// - `tangents`: array of `Array` which are the "vector" in the Jacobian-vector
///   product.  The `tangents` should be the same in number, shape and type as
///   the inputs of `f`, e.g. the `primals`
///
/// # Returns:
///
/// Array of the Jacobian-vector products which is the same in number, shape and
/// type of the outputs of `f`
pub fn jvp<'a, F>(f: F, primals: &[Array], tangents: &[Array]) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    let closure = Closure::new(f);
    jvp_inner(closure, primals, tangents)
}

/// Similar to [`jvp`] but handles closures that can return an error.
pub fn fallible_jvp<'a, F>(
    f: F,
    primals: &[Array],
    tangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    let closure = Closure::new_fallible(f);
    jvp_inner(closure, primals, tangents)
}

#[inline]
fn vjp_inner(
    closure: Closure<'_>,
    primals: &[Array],
    cotangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)> {
    let c_primals = VectorArray::try_from_iter(primals.iter())?;
    let c_cotangents = VectorArray::try_from_iter(cotangents.iter())?;

    <(Vec<Array>, Vec<Array>) as Guarded>::try_from_op(|(res_0, res_1)| unsafe {
        mlx_sys::mlx_vjp(
            res_0,
            res_1,
            closure.as_ptr(),
            c_primals.as_ptr(),
            c_cotangents.as_ptr(),
        )
    })
    .map_err(|e| match get_and_clear_closure_error() {
        Some(err) => err,
        None => e,
    })
}

/// Compute the vector-Jacobian product.
///
/// Computes the product of the `cotangents` with the Jacobian of a function `f` evaluated at
/// `primals`.
///
/// # Params:
///
/// - f: function which takes an array of `Array` and returns an array of `Array`
/// - primals: array of `Array` at which to evaluate the Jacobian
/// - cotangents: array of `Array` which are the "vector" in the vector-Jacobian product. The
///   `cotangents` should be the same in number, shape and type as the outputs of `f`
///
/// # Returns:
///
/// array of the vector-Jacobian products which is the same in number, shape and type of the outputs
/// of `f`
pub fn vjp<'a, F>(f: F, primals: &[Array], cotangents: &[Array]) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    let closure = Closure::new(f);
    vjp_inner(closure, primals, cotangents)
}

/// Similar to [`vjp`] but handles closures that can return an error.
pub fn fallible_vjp<'a, F>(
    f: F,
    primals: &[Array],
    cotangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    let closure = Closure::new_fallible(f);
    vjp_inner(closure, primals, cotangents)
}

pub(crate) struct ClosureValueAndGrad {
    pub(crate) c_closure_value_and_grad: mlx_closure_value_and_grad,
}

impl ClosureValueAndGrad {
    pub fn as_ptr(&self) -> mlx_closure_value_and_grad {
        self.c_closure_value_and_grad
    }
}

impl Drop for ClosureValueAndGrad {
    fn drop(&mut self) {
        let status =
            unsafe { mlx_sys::mlx_closure_value_and_grad_free(self.c_closure_value_and_grad) };
        debug_assert_eq!(status, SUCCESS);
    }
}

fn value_and_gradient(
    value_and_grad: mlx_closure_value_and_grad,
    arrays: impl Iterator<Item = impl AsRef<Array>>,
) -> Result<(Vec<Array>, Vec<Array>)> {
    let input_vector = VectorArray::try_from_iter(arrays)?;

    <(Vec<Array>, Vec<Array>) as Guarded>::try_from_op(|(res_0, res_1)| unsafe {
        mlx_sys::mlx_closure_value_and_grad_apply(
            res_0,
            res_1,
            value_and_grad,
            input_vector.as_ptr(),
        )
    })
    .map_err(|e| match get_and_clear_closure_error() {
        Some(err) => err,
        None => e,
    })
}

#[cfg(test)]
mod tests {

    use crate::{
        array,
        transforms::{jvp, vjp},
        Array,
    };

    use super::*;

    // The unit tests below are adapted from the mlx c++ codebase

    #[test]
    fn test_jvp() {
        let f = |inputs: &[Array]| -> Vec<Array> { vec![&inputs[0] + &inputs[1]] };
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let (out, dout) = jvp(f, &[x, y], &[array!(1.0f32), array!(3.0f32)]).unwrap();
        assert_eq!(out[0].item_exact::<f32>(), 2.0f32);
        assert_eq!(dout[0].item_exact::<f32>(), 4.0f32);
    }

    #[test]
    fn test_jvp_with_error() {
        let f = |inputs: &[Array]| -> Result<Vec<Array>> {
            inputs[0].add(&inputs[1]).map(|res| vec![res])
        };

        // Success case
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let (out, dout) = fallible_jvp(f, &[x, y], &[array!(1.0f32), array!(3.0f32)]).unwrap();
        assert_eq!(out[0].item_exact::<f32>(), 2.0f32);
        assert_eq!(dout[0].item_exact::<f32>(), 4.0f32);

        // Error case
        // Use non-broadcastable shapes
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let result = fallible_jvp(f, &[a, b], &[array!(1.0f32), array!(3.0f32)]);
        assert!(result.is_err());

        // Check that the error is not just "mlx_closure returned a non-zero value"
        let err = result.unwrap_err();
        assert!(!err.what().contains("non-zero value"))
    }

    #[test]
    fn test_vjp() {
        let f = |inputs: &[Array]| -> Vec<Array> { vec![&inputs[0] + &inputs[1]] };
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let primals = vec![x, y];
        let cotangents = vec![array!(1.0f32)];
        let (out, dout) = vjp(f, &primals, &cotangents).unwrap();
        assert_eq!(out[0].item_exact::<f32>(), 2.0f32);
        assert_eq!(dout[0].item_exact::<f32>(), 1.0f32);
    }

    #[test]
    fn test_vjp_with_error() {
        let f = |inputs: &[Array]| -> Result<Vec<Array>> {
            inputs[0].add(&inputs[1]).map(|res| vec![res])
        };

        // Success case
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let primals = vec![x, y];
        let cotangents = vec![array!(1.0f32)];
        let (out, dout) = fallible_vjp(f, &primals, &cotangents).unwrap();
        assert_eq!(out[0].item_exact::<f32>(), 2.0f32);
        assert_eq!(dout[0].item_exact::<f32>(), 1.0f32);

        // Error case
        // Use non-broadcastable shapes
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let result = fallible_vjp(f, &[a, b], &[array!(1.0f32)]);
        assert!(result.is_err());

        // Check that the error is not just "mlx_closure returned a non-zero value"
        let err = result.unwrap_err();
        assert!(!err.what().contains("non-zero value"))
    }
}
