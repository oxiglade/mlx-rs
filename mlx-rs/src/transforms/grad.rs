use crate::{
    error::{Exception, Result},
    utils::{guard::Guarded, Closure, IntoOption},
    Array,
};

use super::{value_and_gradient, ClosureValueAndGrad};

/// Move the lone gradient out of a Vec returned by build_gradient when
/// the caller knows exactly one gradient was requested (argnums.len() == 1).
#[inline]
fn take_one_grad(mut grads: Vec<Array>) -> Result<Array> {
    if grads.len() != 1 {
        return Err(Exception::custom(format!(
            "grad: expected 1 gradient (argnums=[0]), got {}",
            grads.len()
        )));
    }
    Ok(grads.swap_remove(0))
}

#[inline]
fn build_gradient_inner<'a>(
    closure: Closure<'a>,
    argnums: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<Vec<Array>> + 'a {
    move |arrays: &[Array]| -> Result<Vec<Array>> {
        let cvg = ClosureValueAndGrad::try_from_op(|res| unsafe {
            mlx_sys::mlx_value_and_grad(res, closure.as_ptr(), argnums.as_ptr(), argnums.len())
        })?;
        let result = value_and_gradient(cvg.as_ptr(), arrays.iter())?;
        Ok(result.1)
    }
}

fn build_gradient<'a, F>(
    f: F,
    argnums: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<Vec<Array>> + 'a
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    let argnums = argnums.into_option().unwrap_or(&[0]);
    let closure = Closure::new(f);
    build_gradient_inner(closure, argnums)
}

fn build_fallible_gradient<'a, F>(
    f: F,
    argnums: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<Vec<Array>> + 'a
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    let closure = Closure::new_fallible(f);
    build_gradient_inner(closure, argnums)
}

/// Trait for functions/closures that can be converted into a closure that computes the gradient.
pub trait IntoGrad<'a, Args, Output, Err> {
    /// Convert the function/closure into a closure that computes the gradient.
    fn into_grad(
        self,
        argnums: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(Args) -> Result<Output> + 'a;
}

impl<'a, F> IntoGrad<'a, &[Array], Vec<Array>, ()> for F
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    // refining_impl_trait is fine here because we have restricted the Args and Output types
    // in the generics.
    #[allow(refining_impl_trait)]
    fn into_grad(
        self,
        argnums: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&[Array]) -> Result<Vec<Array>> + 'a {
        let argnums = argnums.into_option().unwrap_or(&[0]);
        build_gradient(self, argnums)
    }
}

impl<'a, F> IntoGrad<'a, &[Array], Vec<Array>, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    #[allow(refining_impl_trait)]
    fn into_grad(
        self,
        argnums: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&[Array]) -> Result<Vec<Array>> + 'a {
        let argnums = argnums.into_option().unwrap_or(&[0]);
        build_fallible_gradient(self, argnums)
    }
}

impl<'a, F> IntoGrad<'a, &Array, Array, ()> for F
where
    F: FnMut(&Array) -> Array + 'a,
{
    #[allow(refining_impl_trait)]
    fn into_grad(
        mut self,
        argnums: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&Array) -> Result<Array> + 'a {
        let f = move |args: &[Array]| -> Vec<Array> { vec![self(&args[0])] };
        let argnums = argnums.into_option().unwrap_or(&[0]);
        let mut g = build_gradient(f, argnums);
        move |args: &Array| -> Result<Array> {
            let args_clone = std::slice::from_ref(args);
            take_one_grad(g(args_clone)?)
        }
    }
}

impl<'a, F> IntoGrad<'a, &Array, Array, Exception> for F
where
    F: FnMut(&Array) -> Result<Array> + 'a,
{
    #[allow(refining_impl_trait)]
    fn into_grad(
        mut self,
        argnums: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&Array) -> Result<Array> + 'a {
        let f = move |args: &[Array]| -> Result<Vec<Array>> { self(&args[0]).map(|res| vec![res]) };
        let argnums = argnums.into_option().unwrap_or(&[0]);
        let mut g = build_fallible_gradient(f, argnums);
        move |args: &Array| -> Result<Array> {
            let args_clone = std::slice::from_ref(args);
            take_one_grad(g(args_clone)?)
        }
    }
}

impl<'a, F> IntoGrad<'a, &[Array], Array, ()> for F
where
    F: FnMut(&[Array]) -> Array + 'a,
{
    #[allow(refining_impl_trait)]
    fn into_grad(
        mut self,
        argnums: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&[Array]) -> Result<Array> + 'a {
        let f = move |args: &[Array]| -> Vec<Array> { vec![self(args)] };
        let argnums = argnums.into_option().unwrap_or(&[0]);
        let mut g = build_gradient(f, argnums);
        move |args: &[Array]| -> Result<Array> { take_one_grad(g(args)?) }
    }
}

impl<'a, F> IntoGrad<'a, &[Array], Array, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Array> + 'a,
{
    #[allow(refining_impl_trait)]
    fn into_grad(
        mut self,
        argnums: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&[Array]) -> Result<Array> + 'a {
        let f = move |args: &[Array]| -> Result<Vec<Array>> { self(args).map(|res| vec![res]) };
        let argnums = argnums.into_option().unwrap_or(&[0]);
        let mut g = build_fallible_gradient(f, argnums);
        move |args: &[Array]| -> Result<Array> { take_one_grad(g(args)?) }
    }
}

impl<'a, F> IntoGrad<'a, &Array, Vec<Array>, ()> for F
where
    F: FnMut(&Array) -> Vec<Array> + 'a,
{
    #[allow(refining_impl_trait)]
    fn into_grad(
        mut self,
        argnums: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&Array) -> Result<Vec<Array>> + 'a {
        let f = move |args: &[Array]| -> Vec<Array> { self(&args[0]) };
        let argnums = argnums.into_option().unwrap_or(&[0]);
        let mut g = build_gradient(f, argnums);
        move |args: &Array| -> Result<Vec<Array>> {
            let args_clone = std::slice::from_ref(args);
            let result = g(args_clone)?;
            Ok(result)
        }
    }
}

impl<'a, F> IntoGrad<'a, &Array, Vec<Array>, Exception> for F
where
    F: FnMut(&Array) -> Result<Vec<Array>> + 'a,
{
    #[allow(refining_impl_trait)]
    fn into_grad(
        mut self,
        argnums: impl IntoOption<&'a [i32]>,
    ) -> impl FnMut(&Array) -> Result<Vec<Array>> + 'a {
        let f = move |args: &[Array]| -> Result<Vec<Array>> { self(&args[0]) };
        let argnums = argnums.into_option().unwrap_or(&[0]);
        let mut g = build_fallible_gradient(f, argnums);
        move |args: &Array| -> Result<Vec<Array>> {
            let args_clone = std::slice::from_ref(args);
            let result = g(args_clone)?;
            Ok(result)
        }
    }
}

/// Returns a function which computes the gradient of `f` with the default
/// argument numbers `&[0]`.
///
/// See also [`grad_with_arg_nums`] for a version that allows specifying the
/// argument numbers
pub fn grad<'a, F, Args, Output, Err>(f: F) -> impl FnMut(Args) -> Result<Output> + 'a
where
    F: IntoGrad<'a, Args, Output, Err>,
{
    f.into_grad(None)
}

/// Returns a function which computes the gradient of `f`.
///
/// See also [`grad`] for a version that uses the default argument numbers
/// `&[0]`.
pub fn grad_with_argnums<'a, F, Args, Output, Err>(
    f: F,
    argnums: impl IntoOption<&'a [i32]>,
) -> impl FnMut(Args) -> Result<Output> + 'a
where
    F: IntoGrad<'a, Args, Output, Err>,
{
    f.into_grad(argnums)
}

#[cfg(test)]
mod tests {

    use crate::{
        transforms::{grad, grad_with_argnums, value_and_grad, value_and_grad_with_argnums},
        Array,
    };

    // The unit tests below are adapted from the mlx c++ codebase
    #[test]
    fn test_grad() {
        let x = &[Array::from_f32(1.0)];
        let fun = |argin: &[Array]| -> Vec<Array> { vec![&argin[0] + 1.0] };
        let argnums = &[0];

        // TODO: how to make this more "functional"?
        let grad_fn =
            move |args: &[Array]| -> Vec<Array> { grad_with_argnums(fun, argnums)(args).unwrap() };
        let (z, d2fdx2) = value_and_grad_with_argnums(grad_fn, argnums)(x).unwrap();

        assert_eq!(z[0].item::<f32>(), 1.0);
        assert_eq!(d2fdx2[0].item::<f32>(), 0.0);

        let grad_fn = move |args: &[Array]| -> Vec<Array> { grad(fun)(args).unwrap() };
        let (z, d2fdx2) = value_and_grad(grad_fn)(x).unwrap();

        assert_eq!(z[0].item::<f32>(), 1.0);
        assert_eq!(d2fdx2[0].item::<f32>(), 0.0);
    }
}
