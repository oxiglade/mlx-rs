//! Compilation of functions.

use std::marker::PhantomData;

use crate::{error::Exception, Array};

use super::{next_compile_id, Closure, Compiled, CompiledState, Guarded, VectorArray};

/// Slice-based adapter MLX invokes internally (infallible path). `+ Send`
/// lets a `Compiled<F, G>` cross thread boundaries.
pub type BoxedSliceFn = Box<dyn FnMut(&[Array]) -> Vec<Array> + Send + 'static>;

/// Slice-based adapter MLX invokes internally (fallible path).
pub type BoxedSliceTryFn =
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>;

/// Returns a compiled function that produces the same output as `f`. The
/// underlying `mlx_closure` is built on the first call and reused.
///
/// See the [swift binding docs](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/compilation).
pub fn compile<F, A, O, E>(
    f: F,
    shapeless: impl Into<Option<bool>>,
) -> impl for<'a> FnMut(<F::Output as CallMut<O, E>>::Args<'a>) -> Result<O, Exception>
where
    F: Compile<A, O, E> + 'static + Copy,
    F::Output: CallMut<O, E>,
{
    let shapeless = shapeless.into().unwrap_or(false);
    let mut compiled = f.compile(shapeless);
    move |args| compiled.call_mut(args)
}

/// `A` argument-arity marker, `O` output, `E` error (`()` infallible, [`Exception`] otherwise).
pub trait Compile<A, O, E>: Sized {
    /// Concrete [`Compiled`] produced by [`Self::compile`].
    type Output: CallMut<O, E>;
    /// Compile the function. Allocates a fresh compile-cache id.
    fn compile(self, shapeless: bool) -> Self::Output;
    /// Like [`Self::compile`] but pins the compile-cache id to `id`
    /// instead of allocating a fresh one. Pass a stable per-operation id
    /// (see [`super::allocate_compile_id`]) so many module instances
    /// running the same activation reuse one compiled Metal kernel.
    fn compile_with_id(self, id: usize, shapeless: bool) -> Self::Output;
}

/// GAT on `Args<'a>` lets one long-lived [`Compiled`] accept borrows of any lifetime.
pub trait CallMut<O, E> {
    /// Input argument type, parameterised by borrow lifetime.
    type Args<'a>;
    /// Invoke the compiled function.
    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<O, Exception>;
}

/// Zero-sized markers discriminating per-arity [`CallMut`] impls.
pub mod shape {
    /// `&[Array]` inputs.
    #[derive(Debug, Clone, Copy)]
    pub struct ArraySlice;
    /// `&Array` input.
    #[derive(Debug, Clone, Copy)]
    pub struct OneArg;
    /// `(&Array, &Array)` inputs.
    #[derive(Debug, Clone, Copy)]
    pub struct TwoArgs;
    /// `(&Array, &Array, &Array)` inputs.
    #[derive(Debug, Clone, Copy)]
    pub struct ThreeArgs;
}

impl<F> Compile<&[Array], Vec<Array>, ()> for F
where
    F: FnMut(&[Array]) -> Vec<Array> + 'static,
{
    type Output = Compiled<F, F, shape::ArraySlice>;

    fn compile(self, shapeless: bool) -> Self::Output {
        self.compile_with_id(next_compile_id(), shapeless)
    }

    fn compile_with_id(self, id: usize, shapeless: bool) -> Self::Output {
        Compiled {
            shape: PhantomData,
            f_marker: PhantomData,
            state: CompiledState {
                f: self,
                shapeless,
                id,
                cached_compiled: None,
                cached_num_outputs: std::cell::Cell::new(None),
            },
        }
    }
}

impl<F> Compile<&Array, Array, ()> for F
where
    F: FnMut(&Array) -> Array + Send + 'static,
{
    type Output = Compiled<F, BoxedSliceFn, shape::OneArg>;

    fn compile(self, shapeless: bool) -> Self::Output {
        self.compile_with_id(next_compile_id(), shapeless)
    }

    fn compile_with_id(mut self, id: usize, shapeless: bool) -> Self::Output {
        let f: BoxedSliceFn = Box::new(move |args: &[Array]| vec![(self)(&args[0])]);
        Compiled {
            shape: PhantomData,
            f_marker: PhantomData,
            state: CompiledState {
                f,
                shapeless,
                id,
                cached_compiled: None,
                cached_num_outputs: std::cell::Cell::new(None),
            },
        }
    }
}

impl<F> Compile<(&Array, &Array), Array, ()> for F
where
    F: FnMut((&Array, &Array)) -> Array + Send + 'static,
{
    type Output = Compiled<F, BoxedSliceFn, shape::TwoArgs>;

    fn compile(self, shapeless: bool) -> Self::Output {
        self.compile_with_id(next_compile_id(), shapeless)
    }

    fn compile_with_id(mut self, id: usize, shapeless: bool) -> Self::Output {
        let f: BoxedSliceFn = Box::new(move |args: &[Array]| vec![(self)((&args[0], &args[1]))]);
        Compiled {
            shape: PhantomData,
            f_marker: PhantomData,
            state: CompiledState {
                f,
                shapeless,
                id,
                cached_compiled: None,
                cached_num_outputs: std::cell::Cell::new(None),
            },
        }
    }
}

impl<F> Compile<(&Array, &Array, &Array), Array, ()> for F
where
    F: FnMut((&Array, &Array, &Array)) -> Array + Send + 'static,
{
    type Output = Compiled<F, BoxedSliceFn, shape::ThreeArgs>;

    fn compile(self, shapeless: bool) -> Self::Output {
        self.compile_with_id(next_compile_id(), shapeless)
    }

    fn compile_with_id(mut self, id: usize, shapeless: bool) -> Self::Output {
        let f: BoxedSliceFn =
            Box::new(move |args: &[Array]| vec![(self)((&args[0], &args[1], &args[2]))]);
        Compiled {
            shape: PhantomData,
            f_marker: PhantomData,
            state: CompiledState {
                f,
                shapeless,
                id,
                cached_compiled: None,
                cached_num_outputs: std::cell::Cell::new(None),
            },
        }
    }
}

impl<F> Compile<&[Array], Vec<Array>, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static,
{
    type Output = Compiled<F, F, shape::ArraySlice>;

    fn compile(self, shapeless: bool) -> Self::Output {
        self.compile_with_id(next_compile_id(), shapeless)
    }

    fn compile_with_id(self, id: usize, shapeless: bool) -> Self::Output {
        Compiled {
            shape: PhantomData,
            f_marker: PhantomData,
            state: CompiledState {
                f: self,
                shapeless,
                id,
                cached_compiled: None,
                cached_num_outputs: std::cell::Cell::new(None),
            },
        }
    }
}

impl<F> Compile<&Array, Array, Exception> for F
where
    F: FnMut(&Array) -> Result<Array, Exception> + Send + 'static,
{
    type Output = Compiled<F, BoxedSliceTryFn, shape::OneArg>;

    fn compile(self, shapeless: bool) -> Self::Output {
        self.compile_with_id(next_compile_id(), shapeless)
    }

    fn compile_with_id(mut self, id: usize, shapeless: bool) -> Self::Output {
        let f: BoxedSliceTryFn = Box::new(move |args: &[Array]| Ok(vec![(self)(&args[0])?]));
        Compiled {
            shape: PhantomData,
            f_marker: PhantomData,
            state: CompiledState {
                f,
                shapeless,
                id,
                cached_compiled: None,
                cached_num_outputs: std::cell::Cell::new(None),
            },
        }
    }
}

impl<F> Compile<(&Array, &Array), Array, Exception> for F
where
    F: FnMut((&Array, &Array)) -> Result<Array, Exception> + Send + 'static,
{
    type Output = Compiled<F, BoxedSliceTryFn, shape::TwoArgs>;

    fn compile(self, shapeless: bool) -> Self::Output {
        self.compile_with_id(next_compile_id(), shapeless)
    }

    fn compile_with_id(mut self, id: usize, shapeless: bool) -> Self::Output {
        let f: BoxedSliceTryFn =
            Box::new(move |args: &[Array]| Ok(vec![(self)((&args[0], &args[1]))?]));
        Compiled {
            shape: PhantomData,
            f_marker: PhantomData,
            state: CompiledState {
                f,
                shapeless,
                id,
                cached_compiled: None,
                cached_num_outputs: std::cell::Cell::new(None),
            },
        }
    }
}

impl<F> Compile<(&Array, &Array, &Array), Array, Exception> for F
where
    F: FnMut((&Array, &Array, &Array)) -> Result<Array, Exception> + Send + 'static,
{
    type Output = Compiled<F, BoxedSliceTryFn, shape::ThreeArgs>;

    fn compile(self, shapeless: bool) -> Self::Output {
        self.compile_with_id(next_compile_id(), shapeless)
    }

    fn compile_with_id(mut self, id: usize, shapeless: bool) -> Self::Output {
        let f: BoxedSliceTryFn =
            Box::new(move |args: &[Array]| Ok(vec![(self)((&args[0], &args[1], &args[2]))?]));
        Compiled {
            shape: PhantomData,
            f_marker: PhantomData,
            state: CompiledState {
                f,
                shapeless,
                id,
                cached_compiled: None,
                cached_num_outputs: std::cell::Cell::new(None),
            },
        }
    }
}

impl<F, G> CallMut<Vec<Array>, ()> for Compiled<F, G, shape::ArraySlice>
where
    G: FnMut(&[Array]) -> Vec<Array> + 'static,
{
    type Args<'a> = &'a [Array];

    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<Vec<Array>, Exception> {
        self.state.call_mut_with(args)
    }
}

impl<F, G> CallMut<Array, ()> for Compiled<F, G, shape::OneArg>
where
    G: FnMut(&[Array]) -> Vec<Array> + 'static,
{
    type Args<'a> = &'a Array;

    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<Array, Exception> {
        self.state.call_mut_with_one(std::slice::from_ref(args))
    }
}

impl<F, G> CallMut<Array, ()> for Compiled<F, G, shape::TwoArgs>
where
    G: FnMut(&[Array]) -> Vec<Array> + 'static,
{
    type Args<'a> = (&'a Array, &'a Array);

    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<Array, Exception> {
        self.state.call_mut_with_one(&[args.0, args.1])
    }
}

impl<F, G> CallMut<Array, ()> for Compiled<F, G, shape::ThreeArgs>
where
    G: FnMut(&[Array]) -> Vec<Array> + 'static,
{
    type Args<'a> = (&'a Array, &'a Array, &'a Array);

    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<Array, Exception> {
        self.state.call_mut_with_one(&[args.0, args.1, args.2])
    }
}

impl<F, G> CallMut<Vec<Array>, Exception> for Compiled<F, G, shape::ArraySlice>
where
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
{
    type Args<'a> = &'a [Array];

    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<Vec<Array>, Exception> {
        self.state.fallible_call_mut_with(args)
    }
}

impl<F, G> CallMut<Array, Exception> for Compiled<F, G, shape::OneArg>
where
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
{
    type Args<'a> = &'a Array;

    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<Array, Exception> {
        self.state
            .fallible_call_mut_with_one(std::slice::from_ref(args))
    }
}

impl<F, G> CallMut<Array, Exception> for Compiled<F, G, shape::TwoArgs>
where
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
{
    type Args<'a> = (&'a Array, &'a Array);

    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<Array, Exception> {
        self.state.fallible_call_mut_with_one(&[args.0, args.1])
    }
}

impl<F, G> CallMut<Array, Exception> for Compiled<F, G, shape::ThreeArgs>
where
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
{
    type Args<'a> = (&'a Array, &'a Array, &'a Array);

    fn call_mut<'a>(&mut self, args: Self::Args<'a>) -> Result<Array, Exception> {
        self.state
            .fallible_call_mut_with_one(&[args.0, args.1, args.2])
    }
}

#[inline]
fn apply_compiled(
    compiled: &Closure<'_>,
    args: &[impl AsRef<Array>],
) -> Result<Vec<Array>, Exception> {
    let inner_inputs_vector = VectorArray::try_from_iter(args.iter())?;
    let result_vector = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_closure_apply(res, compiled.as_ptr(), inner_inputs_vector.as_ptr())
    })?;
    result_vector.try_into_values()
}

/// Single-output [`apply_compiled`]: reads the C-vector directly, no
/// `Vec` allocation.
#[inline]
fn apply_compiled_one(
    compiled: &Closure<'_>,
    args: &[impl AsRef<Array>],
) -> Result<Array, Exception> {
    let inner_inputs_vector = VectorArray::try_from_iter(args.iter())?;
    let result_vector = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_closure_apply(res, compiled.as_ptr(), inner_inputs_vector.as_ptr())
    })?;
    result_vector.try_into_one()
}

#[inline]
fn build_compiled(
    inner_closure: Closure<'_>,
    fun_id: usize,
    shapeless: bool,
) -> Result<Closure<'static>, Exception> {
    Closure::try_from_op(|res| unsafe {
        let constants: &[u64] = &[];
        mlx_sys::mlx_detail_compile(
            res,
            inner_closure.as_ptr(),
            fun_id,
            shapeless,
            constants.as_ptr(),
            0,
        )
    })
}

impl<F> CompiledState<F> {
    pub(super) fn call_mut_with(
        &mut self,
        args: &[impl AsRef<Array>],
    ) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&[Array]) -> Vec<Array> + 'static,
    {
        if let Some(compiled) = self.cached_compiled.as_ref() {
            return apply_compiled(compiled, args);
        }
        let inner_closure = Closure::new(&mut self.f);
        let compiled = build_compiled(inner_closure, self.id, self.shapeless)?;
        let result = apply_compiled(&compiled, args);
        self.cached_compiled = Some(compiled);
        result
    }

    pub(super) fn call_mut_with_one(
        &mut self,
        args: &[impl AsRef<Array>],
    ) -> Result<Array, Exception>
    where
        F: FnMut(&[Array]) -> Vec<Array> + 'static,
    {
        if let Some(compiled) = self.cached_compiled.as_ref() {
            return apply_compiled_one(compiled, args);
        }
        let inner_closure = Closure::new(&mut self.f);
        let compiled = build_compiled(inner_closure, self.id, self.shapeless)?;
        let result = apply_compiled_one(&compiled, args);
        self.cached_compiled = Some(compiled);
        result
    }

    pub(super) fn fallible_call_mut_with(
        &mut self,
        args: &[impl AsRef<Array>],
    ) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
    {
        if let Some(compiled) = self.cached_compiled.as_ref() {
            return apply_compiled(compiled, args);
        }
        let inner_closure = Closure::new_fallible(&mut self.f);
        let compiled = build_compiled(inner_closure, self.id, self.shapeless)?;
        let result = apply_compiled(&compiled, args);
        self.cached_compiled = Some(compiled);
        result
    }

    pub(super) fn fallible_call_mut_with_one(
        &mut self,
        args: &[impl AsRef<Array>],
    ) -> Result<Array, Exception>
    where
        F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
    {
        if let Some(compiled) = self.cached_compiled.as_ref() {
            return apply_compiled_one(compiled, args);
        }
        let inner_closure = Closure::new_fallible(&mut self.f);
        let compiled = build_compiled(inner_closure, self.id, self.shapeless)?;
        let result = apply_compiled_one(&compiled, args);
        self.cached_compiled = Some(compiled);
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        array,
        error::Exception,
        ops::{multiply, ones},
        Array,
    };

    use super::{compile, Compile};

    #[test]
    #[allow(
        trivial_casts,
        reason = "fn-item ZST → fn-pointer; required for distinct-compile-id regression"
    )]
    fn distinct_fn_pointers_get_distinct_compile_ids() {
        // Regression: the prior `type_id_to_usize<T>()` derived the cache id
        // from `TypeId::of::<T>()`. Two `fn` pointers cast to the same concrete
        // signature share one TypeId — so the second compile reused the first
        // compiled graph (e.g. an `attention_gate` returning
        // `sigmoid(output) * gate` after a `swiglu` warmed the same-signatured
        // slot). `next_compile_id()` must hand out distinct ids regardless of
        // source type.
        fn f0(x: &Array) -> Array {
            x.clone()
        }
        fn f1(x: &Array) -> Array {
            x.clone()
        }
        let c0 = (f0 as fn(&Array) -> Array).compile(false);
        let c1 = (f1 as fn(&Array) -> Array).compile(false);
        assert_ne!(c0.state.id, c1.state.id);
    }

    #[test]
    fn test_compile() {
        let f = |inputs: &[Array]| -> Vec<Array> { vec![&inputs[0] * &inputs[1]] };
        let mut compiled = compile(f, None);

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();

        let args = [i1, i2];

        let r1 = f(&args).drain(0..1).next().unwrap();
        let r2 = compiled(&args).unwrap().drain(0..1).next().unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled(&args).unwrap().drain(0..1).next().unwrap();
        assert_eq!(&r1, &r3);
    }

    #[test]
    fn test_compile_with_error() {
        let f = |inputs: &[Array]| -> Result<Vec<Array>, Exception> {
            multiply(&inputs[0], &inputs[1]).map(|x| vec![x])
        };

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();
        let args = [i1, i2];

        let r1 = f(&args).unwrap().drain(0..1).next().unwrap();

        let mut compiled = compile(f, None);
        let r2 = compiled(&args).unwrap().drain(0..1).next().unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled(&args).unwrap().drain(0..1).next().unwrap();
        assert_eq!(&r1, &r3);

        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let args = [a, b];

        let c = array!([4.0, 5.0, 6.0]);
        let d = array!([7.0, 8.0]);
        let another_args = [c, d];

        let result = f(&args);
        assert!(result.is_err());

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

        let r1 = f(&i);

        let mut compiled = compile(f, None);
        let r2 = compiled(&i).unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled(&i).unwrap();
        assert_eq!(&r1, &r3);
    }

    #[test]
    fn test_compile_with_two_args() {
        let f = |(x, y): (&Array, &Array)| x * y;

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();

        let r1 = f((&i1, &i2));

        let mut compiled = compile(f, None);
        let r2 = compiled((&i1, &i2)).unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled((&i1, &i2)).unwrap();
        assert_eq!(&r1, &r3);
    }

    #[test]
    fn test_compile_with_three_args() {
        let f = |(x, y, z): (&Array, &Array, &Array)| x * y * z;
        let mut compiled = compile(f, None);

        let i1 = ones::<f32>(&[20, 20]).unwrap();
        let i2 = ones::<f32>(&[20, 20]).unwrap();
        let i3 = ones::<f32>(&[20, 20]).unwrap();

        let r1 = f((&i1, &i2, &i3));

        let r2 = compiled((&i1, &i2, &i3)).unwrap();

        assert_eq!(&r1, &r2);

        let r3 = compiled((&i1, &i2, &i3)).unwrap();
        assert_eq!(&r1, &r3);
    }
}
