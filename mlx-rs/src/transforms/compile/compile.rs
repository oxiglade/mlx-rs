//! Compilation of functions.

// TODO: there's plenty boilerplate code here but it's not clear how to reduce it

use std::marker::PhantomData;

use crate::{error::Exception, Array};

use super::{Closure, Compiled, CompiledState, Guarded, VectorArray};

/// Returns a compiled function that produces the same output as `f`.
///
/// Please refer to the [swift binding
/// documentation](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/compilation)
/// for more information.
pub fn compile<F, A, O, E>(
    f: F,
    shapeless: impl Into<Option<bool>>,
) -> impl for<'a> FnMut(F::Args<'a>) -> Result<O, Exception>
where
    F: Compile<A, O, E> + 'static,
{
    let shapeless = shapeless.into().unwrap_or(false);
    let mut compiled = f.compile(shapeless);
    move |args| compiled.call_mut(args)
}

/// A trait for functions that can be compiled.
///
/// # Generic parameters
///
/// - `A`: The type of the array arguments
/// - `O`: The type of the output
/// - `E`: The type of the error
pub trait Compile<A, O, E>: Sized {
    /// The type of the arguments that the returned closure takes.
    ///
    /// This is needed to relax the lifetime requirements of the returned
    /// closure. Otherwise, the arguments to the returned closure would have to
    /// live longer than the closure itself.
    type Args<'a>;

    /// Compiles the function.
    fn compile(self, shapeless: bool) -> impl for<'args> CallMut<Self::Args<'args>, O, E>;
}

impl<F> Compile<&[Array], Vec<Array>, ()> for F
where
    F: FnMut(&[Array]) -> Vec<Array> + 'static,
{
    type Args<'a> = &'a [Array];

    fn compile(
        self,
        shapeless: bool,
    ) -> impl for<'args> CallMut<Self::Args<'args>, Vec<Array>, ()> {
        let state = CompiledState::new(self, shapeless);
        Compiled::<F, _> {
            f_marker: PhantomData,
            state,
        }
    }
}

impl<F> Compile<&Array, Array, ()> for F
where
    F: FnMut(&Array) -> Array + 'static,
{
    type Args<'a> = &'a Array;

    fn compile(mut self, shapeless: bool) -> impl for<'args> CallMut<Self::Args<'args>, Array, ()> {
        let f = move |args: &[Array]| -> Vec<Array> {
            let result = (self)(&args[0]);
            vec![result]
        };
        let state = CompiledState::new(f, shapeless);
        Compiled::<F, _> {
            f_marker: PhantomData,
            state,
        }
    }
}

impl<F> Compile<(&Array, &Array), Array, ()> for F
where
    F: FnMut((&Array, &Array)) -> Array + 'static,
{
    type Args<'a> = (&'a Array, &'a Array);

    fn compile(mut self, shapeless: bool) -> impl for<'args> CallMut<Self::Args<'args>, Array, ()> {
        let f = move |args: &[Array]| -> Vec<Array> {
            let result = (self)((&args[0], &args[1]));
            vec![result]
        };
        let state = CompiledState::new(f, shapeless);
        Compiled::<F, _> {
            f_marker: PhantomData,
            state,
        }
    }
}

impl<F> Compile<(&Array, &Array, &Array), Array, ()> for F
where
    F: FnMut((&Array, &Array, &Array)) -> Array + 'static,
{
    type Args<'a> = (&'a Array, &'a Array, &'a Array);

    fn compile(mut self, shapeless: bool) -> impl for<'args> CallMut<Self::Args<'args>, Array, ()> {
        let f = move |args: &[Array]| -> Vec<Array> {
            let result = (self)((&args[0], &args[1], &args[2]));
            vec![result]
        };
        let state = CompiledState::new(f, shapeless);
        Compiled::<F, _> {
            f_marker: PhantomData,
            state,
        }
    }
}

impl<F> Compile<&[Array], Vec<Array>, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
{
    type Args<'a> = &'a [Array];

    fn compile(
        self,
        shapeless: bool,
    ) -> impl for<'args> CallMut<Self::Args<'args>, Vec<Array>, Exception> {
        let state = CompiledState::new(self, shapeless);
        Compiled::<F, _> {
            f_marker: PhantomData,
            state,
        }
    }
}

impl<F> Compile<&Array, Array, Exception> for F
where
    F: FnMut(&Array) -> Result<Array, Exception> + 'static,
{
    type Args<'a> = &'a Array;

    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl for<'args> CallMut<Self::Args<'args>, Array, Exception> {
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)(&args[0])?;
            Ok(vec![result])
        };
        let state = CompiledState::new(f, shapeless);
        Compiled::<F, _> {
            f_marker: PhantomData,
            state,
        }
    }
}

impl<F> Compile<(&Array, &Array), Array, Exception> for F
where
    F: FnMut((&Array, &Array)) -> Result<Array, Exception> + 'static,
{
    type Args<'a> = (&'a Array, &'a Array);

    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl for<'args> CallMut<Self::Args<'args>, Array, Exception> {
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)((&args[0], &args[1]))?;
            Ok(vec![result])
        };
        let state = CompiledState::new(f, shapeless);
        Compiled::<F, _> {
            f_marker: PhantomData,
            state,
        }
    }
}

impl<F> Compile<(&Array, &Array, &Array), Array, Exception> for F
where
    F: FnMut((&Array, &Array, &Array)) -> Result<Array, Exception> + 'static,
{
    type Args<'a> = (&'a Array, &'a Array, &'a Array);

    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl for<'args> CallMut<Self::Args<'args>, Array, Exception> {
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)((&args[0], &args[1], &args[2]))?;
            Ok(vec![result])
        };
        let state = CompiledState::new(f, shapeless);
        Compiled::<F, _> {
            f_marker: PhantomData,
            state,
        }
    }
}

/// A trait for a compiled function that can be called.
pub trait CallMut<A, O, E> {
    /// Calls the compiled function with the given arguments.
    fn call_mut(&mut self, args: A) -> Result<O, Exception>;
}

impl<'a, F, G> CallMut<&'a [Array], Vec<Array>, ()> for Compiled<F, G>
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception> {
        self.state.call_mut(args)
    }
}

impl<'a, F, G> CallMut<&'a Array, Array, ()> for Compiled<F, G>
where
    F: FnMut(&Array) -> Array + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: &Array) -> Result<Array, Exception> {
        let args = std::slice::from_ref(args);
        let result = self.state.call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<(&'a Array, &'a Array), Array, ()> for Compiled<F, G>
where
    F: FnMut((&Array, &Array)) -> Array + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: (&Array, &Array)) -> Result<Array, Exception> {
        let args = &[args.0, args.1];
        let result = self.state.call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<(&'a Array, &'a Array, &'a Array), Array, ()> for Compiled<F, G>
where
    F: FnMut((&Array, &Array, &Array)) -> Array + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: (&Array, &Array, &Array)) -> Result<Array, Exception> {
        // Is there any way to avoid this shallow clone?
        let args = &[args.0, args.1, args.2];
        let result = self.state.call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<&'a [Array], Vec<Array>, Exception> for Compiled<F, G>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception> {
        self.state.fallible_call_mut(args)
    }
}

impl<'a, F, G> CallMut<&'a Array, Array, Exception> for Compiled<F, G>
where
    F: FnMut(&Array) -> Result<Array, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: &Array) -> Result<Array, Exception> {
        let args = &[args];
        let result = self.state.fallible_call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<(&'a Array, &'a Array), Array, Exception> for Compiled<F, G>
where
    F: FnMut((&Array, &Array)) -> Result<Array, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: (&Array, &Array)) -> Result<Array, Exception> {
        let args = &[args.0, args.1];
        let result = self.state.fallible_call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<(&'a Array, &'a Array, &'a Array), Array, Exception> for Compiled<F, G>
where
    F: FnMut((&Array, &Array, &Array)) -> Result<Array, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: (&Array, &Array, &Array)) -> Result<Array, Exception> {
        let args = &[args.0, args.1, args.2];
        let result = self.state.fallible_call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

#[inline]
fn call_mut_inner(
    inner_closure: Closure,
    fun_id: usize,
    shapeless: bool,
    args: &[impl AsRef<Array>],
) -> crate::error::Result<Vec<Array>> {
    // note: this will use the cached compile (via the id)
    // but will be able to re-evaluate with fresh state if needed
    let compiled = Closure::try_from_op(|res| unsafe {
        let constants = &[];
        mlx_sys::mlx_detail_compile(
            res,
            inner_closure.as_ptr(),
            fun_id,
            shapeless,
            constants.as_ptr(),
            0,
        )
    })?;

    let inner_inputs_vector = VectorArray::try_from_iter(args.iter())?;

    // will compile the function (if needed) and evaluate the
    // compiled graph
    let result_vector = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_closure_apply(res, compiled.as_ptr(), inner_inputs_vector.as_ptr())
    })?;
    let result_plus_state_output: Vec<Array> = result_vector.try_into_values()?;

    let result_len = result_plus_state_output.len();
    Ok(result_plus_state_output
        .into_iter()
        .take(result_len)
        .collect())
}

impl<F> CompiledState<F> {
    fn call_mut(&mut self, args: &[impl AsRef<Array>]) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&[Array]) -> Vec<Array>,
    {
        let inner_closure = Closure::new(&mut self.f);

        call_mut_inner(inner_closure, self.id, self.shapeless, args)
    }

    fn fallible_call_mut(&mut self, args: &[impl AsRef<Array>]) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&[Array]) -> Result<Vec<Array>, Exception>,
    {
        let inner_closure = Closure::new_fallible(&mut self.f);

        call_mut_inner(inner_closure, self.id, self.shapeless, args)
    }
}

#[cfg(test)]
mod tests {
    use std::mem::ManuallyDrop;

    use crate::{
        array,
        error::Exception,
        ops::{multiply, ones},
        test_utils::{assert_array_eq, tolerances},
        Array,
    };

    use super::compile;

    fn example_fn_0(x: f32) -> f32 {
        x + 1.0
    }

    #[test]
    fn compile_ids_are_unique_for_live_instances_of_the_same_type() {
        let first = ManuallyDrop::new(super::CompiledState::new(example_fn_0, false));
        let second = ManuallyDrop::new(super::CompiledState::new(example_fn_0, false));

        assert_ne!(first.id, second.id);
    }

    #[test]
    fn cloned_compile_state_gets_a_fresh_monotonic_id() {
        let first = ManuallyDrop::new(super::CompiledState::new(example_fn_0, false));
        let cloned = ManuallyDrop::new(super::CompiledState::clone(&first));

        assert!(cloned.id > first.id);
    }

    #[test]
    fn test_compile() {
        // This unit test is modified from the mlx-swift codebase

        let f = |inputs: &[Array]| -> Vec<Array> { vec![&inputs[0] * &inputs[1]] };
        let mut compiled = compile(f, None);

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();

        let args = [i1, i2];

        // evaluate directly
        let r1 = f(&args).drain(0..1).next().unwrap();
        // evaluate compiled
        let r2 = compiled(&args).unwrap().drain(0..1).next().unwrap();

        assert_array_eq(&r1, &r2, tolerances::EXACT.rtol, tolerances::EXACT.atol);

        let r3 = compiled(&args).unwrap().drain(0..1).next().unwrap();
        assert_array_eq(&r1, &r3, tolerances::EXACT.rtol, tolerances::EXACT.atol);
    }

    #[test]
    fn test_compile_with_error() {
        let f = |inputs: &[Array]| -> Result<Vec<Array>, Exception> {
            multiply(&inputs[0], &inputs[1]).map(|x| vec![x])
        };

        // Success case
        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();
        let args = [i1, i2];

        // evaluate directly
        let r1 = f(&args).unwrap().drain(0..1).next().unwrap();

        // evaluate compiled
        let mut compiled = compile(f, None);
        let r2 = compiled(&args).unwrap().drain(0..1).next().unwrap();

        assert_array_eq(&r1, &r2, tolerances::EXACT.rtol, tolerances::EXACT.atol);

        let r3 = compiled(&args).unwrap().drain(0..1).next().unwrap();
        assert_array_eq(&r1, &r3, tolerances::EXACT.rtol, tolerances::EXACT.atol);

        // Error case
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let args = [a, b];

        // The cache is keyed by function pointer and argument shapes
        let c = array!([4.0, 5.0, 6.0]);
        let d = array!([7.0, 8.0]);
        let another_args = [c, d];

        // evaluate directly
        let result = f(&args);
        assert!(result.is_err());

        // evaluate compiled
        let mut compiled = compile(f, None);
        let result = compiled(&args);
        assert!(result.is_err());

        let result = compiled(&args);
        assert!(result.is_err());

        let result = compiled(&another_args);
        assert!(result.is_err());
    }

    #[test]
    fn test_compile_with_one_arg() {
        let f = |x: &Array| x * x;

        let i = ones::<f32>(&[20, 20]).unwrap();

        // evaluate directly
        let r1 = f(&i);

        // evaluate compiled
        let mut compiled = compile(f, None);
        let r2 = compiled(&i).unwrap();

        assert_array_eq(&r1, &r2, tolerances::EXACT.rtol, tolerances::EXACT.atol);

        let r3 = compiled(&i).unwrap();
        assert_array_eq(&r1, &r3, tolerances::EXACT.rtol, tolerances::EXACT.atol);
    }

    #[test]
    fn test_compile_with_two_args() {
        let f = |(x, y): (&Array, &Array)| x * y;

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();

        // evaluate directly
        let r1 = f((&i1, &i2));

        // evaluate compiled
        let mut compiled = compile(f, None);
        let r2 = compiled((&i1, &i2)).unwrap();

        assert_array_eq(&r1, &r2, tolerances::EXACT.rtol, tolerances::EXACT.atol);

        let r3 = compiled((&i1, &i2)).unwrap();
        assert_array_eq(&r1, &r3, tolerances::EXACT.rtol, tolerances::EXACT.atol);
    }

    #[test]
    fn test_compile_with_three_args() {
        let f = |(x, y, z): (&Array, &Array, &Array)| x * y * z;
        let mut compiled = compile(f, None);

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();
        let i3 = ones::<f32>(&[20, 20]).unwrap();

        // evaluate directly
        let r1 = f((&i1, &i2, &i3));

        // evaluate compiled
        let r2 = compiled((&i1, &i2, &i3)).unwrap();

        assert_array_eq(&r1, &r2, tolerances::EXACT.rtol, tolerances::EXACT.atol);

        let r3 = compiled((&i1, &i2, &i3)).unwrap();
        assert_array_eq(&r1, &r3, tolerances::EXACT.rtol, tolerances::EXACT.atol);
    }
}
