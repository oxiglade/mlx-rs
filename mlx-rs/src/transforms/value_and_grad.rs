use crate::{
    error::{Exception, Result},
    utils::{guard::Guarded, Closure, IntoOption},
    Array,
};

use super::{value_and_gradient, ClosureValueAndGrad};

fn build_value_and_gradient_inner<'a>(
    closure: Closure<'a>,
    argnums: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>)> + 'a {
    move |arrays: &[Array]| unsafe {
        let cvg = ClosureValueAndGrad::try_from_op(|res| {
            mlx_sys::mlx_value_and_grad(res, closure.as_ptr(), argnums.as_ptr(), argnums.len())
        })?;
        value_and_gradient(cvg.as_ptr(), arrays.iter())
    }
}

fn build_value_and_gradient<'a, F>(
    f: F,
    argnums: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>)> + 'a
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    let closure = Closure::new(f);
    build_value_and_gradient_inner(closure, argnums)
}

fn build_fallible_value_and_gradient<'a, F>(
    f: F,
    argnums: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>)> + 'a
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    let closure = Closure::new_fallible(f);
    build_value_and_gradient_inner(closure, argnums)
}

/// Trait for functions/closures that can be converted into a closure that computes the value and
/// gradient.
pub trait IntoValueAndGrad<'a, Err> {
    /// Convert the function/closure into a closure that computes the value and gradient.
    fn into_value_and_grad(
        self,
        argnums: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>)> + 'a;
}

impl<'a, F> IntoValueAndGrad<'a, ()> for F
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    // refining_impl_trait is fine here because we have restricted the Args and Output types
    // in the generics.
    #[allow(refining_impl_trait)]
    fn into_value_and_grad(
        self,
        argnums: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>)> + 'a {
        let argnums = argnums.into_option().unwrap_or(&[0]);
        build_value_and_gradient(self, argnums)
    }
}

impl<'a, F> IntoValueAndGrad<'a, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    #[allow(refining_impl_trait)]
    fn into_value_and_grad(
        self,
        argnums: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>)> + 'a {
        let argnums = argnums.into_option().unwrap_or(&[0]);
        build_fallible_value_and_gradient(self, argnums)
    }
}

/// Returns a function which computes the value and gradient of `f` with a
/// default argument number `&[0]`.
///
/// See also [`value_and_grad_with_arg_nums`] for a version that allows
/// specifying the argument numbers
pub fn value_and_grad<'a, F, Err>(
    f: F,
) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>)> + 'a
where
    F: IntoValueAndGrad<'a, Err> + 'a,
{
    f.into_value_and_grad(None)
}

/// Returns a function which computes the value and gradient of `f`.
///
/// See also [`value_and_grad`] for a version that uses the default argument
/// numbers `&[0]`.
pub fn value_and_grad_with_argnums<'a, F, Err>(
    f: F,
    argnums: impl IntoOption<&'a [i32]>,
) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>)> + 'a
where
    F: IntoValueAndGrad<'a, Err> + 'a,
{
    f.into_value_and_grad(argnums)
}

#[cfg(test)]
mod tests {

    use crate::{array, transforms::value_and_grad, Array};

    use super::*;

    // The unit tests below are adapted from the mlx c++ codebase
    #[test]
    fn test_value_and_grad() {
        let x = &[Array::from_f32(1.0)];
        let fun = |argin: &[Array]| -> Vec<Array> { vec![&argin[0] + 1.0] };
        let argnums = &[0];
        let (y, dfdx) = value_and_grad_with_argnums(fun, argnums)(x).unwrap();
        assert_eq!(y[0].item_exact::<f32>(), 2.0);
        assert_eq!(dfdx[0].item_exact::<f32>(), 1.0);

        let (y, dfdx) = value_and_grad(fun)(x).unwrap();
        assert_eq!(y[0].item_exact::<f32>(), 2.0);
        assert_eq!(dfdx[0].item_exact::<f32>(), 1.0);
    }

    #[test]
    fn test_value_and_grad_with_error() {
        let fun = |argin: &[Array]| -> Result<Vec<Array>> {
            argin[0].add(array!(1.0)).map(|res| vec![res])
        };

        // Success case
        let argnums = &[0];
        let x = array!(1.0f32);
        let y = array!(1.0f32);
        let args = &[x, y];
        let result = value_and_grad_with_argnums(fun, argnums)(args);
        assert!(result.is_ok());
        let result = value_and_grad(fun)(args);
        assert!(result.is_ok());

        // Error case
        // Use non-broadcastable shapes
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let args = &[a, b];
        let result = value_and_grad_with_argnums(fun, argnums)(args);
        assert!(result.is_err());
        let result = value_and_grad(fun)(args);
        assert!(result.is_err());

        // Check that the error is not just "mlx_closure returned a non-zero value"
        let err = result.unwrap_err();
        assert!(!err.what().contains("non-zero value"))
    }
}
