//! Compilation of functions with state.
//!
//! # Unit tests
//!
//! See `mlx-rs/mlx-tests/tests/test_compile.rs` for unit tests.

// TODO: there's plenty boilerplate code here but it's not clear how to reduce it

use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    rc::Rc,
};

use crate::{
    error::Exception,
    transforms::compile::{type_id_to_usize, CompiledState},
    utils::Updatable,
    Array,
};

use super::{update_by_replace_with_ref_to_new_array, Closure, Compiled, Guarded, VectorArray};

/// Similar to [`crate::transforms::compile`] but allows for functions that take
/// a mutable reference to a state `U`.
pub fn compile_with_state<F, U, A, O, E>(
    f: F,
    shapeless: impl Into<Option<bool>>,
) -> impl for<'a> FnMut(&mut U, F::Args<'a>) -> Result<O, Exception>
where
    F: CompileWithState<U, A, O, E> + Copy + 'static,
    U: Updatable,
{
    let shapeless = shapeless.into().unwrap_or(false);
    move |state, args| {
        let mut compiled = f.compile(shapeless);
        compiled.call_mut(state, args)
    }
}

/// A trait for functions that can be compiled with state.
///
/// This trait is used to compile a function that takes a mutable reference to a state
/// and some arguments and returns a result.
///
/// # Generic parameters
///
/// - `U`: The type of the state.
/// - `A`: The type of the arguments.
/// - `O`: The type of the output.
/// - `E`: The type of the exception.
pub trait CompileWithState<U, A, O, E> {
    /// The type of the arguments that the returned closure takes.
    ///
    /// This is needed to relax the lifetime requirements of the returned
    /// closure. Otherwise, the arguments to the returned closure would have to
    /// live longer than the closure itself.
    type Args<'a>;

    /// Compile the function.
    fn compile<'args>(self, shapeless: bool) -> impl CallMutWithState<U, Self::Args<'args>, O, E>;
}

impl<F, U> CompileWithState<U, &[Array], Vec<Array>, ()> for F
where
    F: FnMut(&mut U, &[Array]) -> Vec<Array> + 'static,
    U: Updatable,
{
    type Args<'a> = &'a [Array];

    fn compile<'args>(
        self,
        shapeless: bool,
    ) -> impl CallMutWithState<U, Self::Args<'args>, Vec<Array>, ()> {
        let id = type_id_to_usize(&self);
        let state = CompiledState {
            f: self,
            shapeless,
            id,
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F, U> CompileWithState<U, &Array, Array, ()> for F
where
    F: FnMut(&mut U, &Array) -> Array + 'static,
    U: Updatable,
{
    type Args<'a> = &'a Array;

    fn compile<'args>(
        mut self,
        shapeless: bool,
    ) -> impl CallMutWithState<U, Self::Args<'args>, Array, ()> {
        let id = type_id_to_usize(&self);
        let f = move |state: &mut U, args: &[Array]| -> Vec<Array> {
            let result = (self)(state, &args[0]);
            vec![result]
        };
        let state = CompiledState { f, shapeless, id };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F, U> CompileWithState<U, (&Array, &Array), Array, ()> for F
where
    F: FnMut(&mut U, (&Array, &Array)) -> Array + 'static,
    U: Updatable,
{
    type Args<'a> = (&'a Array, &'a Array);

    fn compile<'args>(
        mut self,
        shapeless: bool,
    ) -> impl CallMutWithState<U, Self::Args<'args>, Array, ()> {
        let id = type_id_to_usize(&self);
        let f = move |state: &mut U, args: &[Array]| -> Vec<Array> {
            let result = (self)(state, (&args[0], &args[1]));
            vec![result]
        };
        let state = CompiledState { f, shapeless, id };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F, U> CompileWithState<U, (&Array, &Array, &Array), Array, ()> for F
where
    F: FnMut(&mut U, (&Array, &Array, &Array)) -> Array + 'static,
    U: Updatable,
{
    type Args<'a> = (&'a Array, &'a Array, &'a Array);

    fn compile<'args>(
        mut self,
        shapeless: bool,
    ) -> impl CallMutWithState<U, Self::Args<'args>, Array, ()> {
        let id = type_id_to_usize(&self);
        let f = move |state: &mut U, args: &[Array]| -> Vec<Array> {
            let result = (self)(state, (&args[0], &args[1], &args[2]));
            vec![result]
        };
        let state = CompiledState { f, shapeless, id };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F, U> CompileWithState<U, &[Array], Vec<Array>, Exception> for F
where
    F: FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception> + 'static,
    U: Updatable,
{
    type Args<'a> = &'a [Array];

    fn compile<'args>(
        self,
        shapeless: bool,
    ) -> impl CallMutWithState<U, Self::Args<'args>, Vec<Array>, Exception> {
        let id = type_id_to_usize(&self);
        let state = CompiledState {
            f: self,
            shapeless,
            id,
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F, U> CompileWithState<U, &Array, Array, Exception> for F
where
    F: FnMut(&mut U, &Array) -> Result<Array, Exception> + 'static,
    U: Updatable,
{
    type Args<'a> = &'a Array;

    fn compile<'args>(
        mut self,
        shapeless: bool,
    ) -> impl CallMutWithState<U, Self::Args<'args>, Array, Exception> {
        let id = type_id_to_usize(&self);
        let f = move |state: &mut U, args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)(state, &args[0])?;
            Ok(vec![result])
        };
        let state = CompiledState { f, shapeless, id };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F, U> CompileWithState<U, (&Array, &Array), Array, Exception> for F
where
    F: FnMut(&mut U, (&Array, &Array)) -> Result<Array, Exception> + 'static,
    U: Updatable,
{
    type Args<'a> = (&'a Array, &'a Array);

    fn compile<'args>(
        mut self,
        shapeless: bool,
    ) -> impl CallMutWithState<U, Self::Args<'args>, Array, Exception> {
        let id = type_id_to_usize(&self);
        let f = move |state: &mut U, args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)(state, (&args[0], &args[1]))?;
            Ok(vec![result])
        };
        let state = CompiledState { f, shapeless, id };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F, U> CompileWithState<U, (&Array, &Array, &Array), Array, Exception> for F
where
    F: FnMut(&mut U, (&Array, &Array, &Array)) -> Result<Array, Exception> + 'static,
    U: Updatable,
{
    type Args<'a> = (&'a Array, &'a Array, &'a Array);

    fn compile<'args>(
        mut self,
        shapeless: bool,
    ) -> impl CallMutWithState<U, Self::Args<'args>, Array, Exception> {
        let id = type_id_to_usize(&self);
        let f = move |state: &mut U, args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)(state, (&args[0], &args[1], &args[2]))?;
            Ok(vec![result])
        };
        let state = CompiledState { f, shapeless, id };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

/// A trait for functions that can be called with state.
pub trait CallMutWithState<U, A, O, E> {
    /// Call the function with the given state and arguments.
    fn call_mut(&mut self, state: &mut U, args: A) -> Result<O, Exception>;
}

impl<U, F, G> CallMutWithState<U, &[Array], Vec<Array>, ()> for Compiled<F, G>
where
    F: FnMut(&mut U, &[Array]) -> Vec<Array>,
    G: FnMut(&mut U, &[Array]) -> Vec<Array>,
    U: Updatable,
{
    fn call_mut(&mut self, state: &mut U, args: &[Array]) -> Result<Vec<Array>, Exception> {
        self.state.retry_call_mut_with_state(state, args)
    }
}

impl<U, F, G> CallMutWithState<U, &Array, Array, ()> for Compiled<F, G>
where
    F: FnMut(&mut U, &Array) -> Array,
    G: FnMut(&mut U, &[Array]) -> Vec<Array>,
    U: Updatable,
{
    fn call_mut(&mut self, state: &mut U, args: &Array) -> Result<Array, Exception> {
        let args = std::slice::from_ref(args);
        let result = self.state.retry_call_mut_with_state(state, args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<U, F, G> CallMutWithState<U, (&Array, &Array), Array, ()> for Compiled<F, G>
where
    F: FnMut(&mut U, (&Array, &Array)) -> Array,
    G: FnMut(&mut U, &[Array]) -> Vec<Array>,
    U: Updatable,
{
    fn call_mut(&mut self, state: &mut U, args: (&Array, &Array)) -> Result<Array, Exception> {
        let args = &[args.0, args.1];
        let result = self.state.retry_call_mut_with_state(state, args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<U, F, G> CallMutWithState<U, (&Array, &Array, &Array), Array, ()> for Compiled<F, G>
where
    F: FnMut(&mut U, (&Array, &Array, &Array)) -> Array,
    G: FnMut(&mut U, &[Array]) -> Vec<Array>,
    U: Updatable,
{
    fn call_mut(
        &mut self,
        state: &mut U,
        args: (&Array, &Array, &Array),
    ) -> Result<Array, Exception> {
        let args = &[args.0, args.1, args.2];
        let result = self.state.retry_call_mut_with_state(state, args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<U, F, G> CallMutWithState<U, &[Array], Vec<Array>, Exception> for Compiled<F, G>
where
    F: FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception>,
    G: FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception>,
    U: Updatable,
{
    fn call_mut(&mut self, state: &mut U, args: &[Array]) -> Result<Vec<Array>, Exception> {
        self.state.retry_fallible_call_mut_with_state(state, args)
    }
}

impl<U, F, G> CallMutWithState<U, &Array, Array, Exception> for Compiled<F, G>
where
    F: FnMut(&mut U, &Array) -> Result<Array, Exception>,
    G: FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception>,
    U: Updatable,
{
    fn call_mut(&mut self, state: &mut U, args: &Array) -> Result<Array, Exception> {
        let args = std::slice::from_ref(args);
        let result = self.state.retry_fallible_call_mut_with_state(state, args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<U, F, G> CallMutWithState<U, (&Array, &Array), Array, Exception> for Compiled<F, G>
where
    F: FnMut(&mut U, (&Array, &Array)) -> Result<Array, Exception>,
    G: FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception>,
    U: Updatable,
{
    fn call_mut(&mut self, state: &mut U, args: (&Array, &Array)) -> Result<Array, Exception> {
        let args = &[args.0, args.1];
        let result = self.state.retry_fallible_call_mut_with_state(state, args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<U, F, G> CallMutWithState<U, (&Array, &Array, &Array), Array, Exception> for Compiled<F, G>
where
    F: FnMut(&mut U, (&Array, &Array, &Array)) -> Result<Array, Exception>,
    G: FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception>,
    U: Updatable,
{
    fn call_mut(
        &mut self,
        state: &mut U,
        args: (&Array, &Array, &Array),
    ) -> Result<Array, Exception> {
        let args = &[args.0, args.1, args.2];
        let result = self.state.retry_fallible_call_mut_with_state(state, args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

#[inline]
fn call_mut_with_state_inner<U>(
    inner_closure: Closure,
    fun_id: usize,
    shapeless: bool,
    state: Rc<RefCell<&mut U>>,
    args: &[impl AsRef<Array>],
    num_function_outputs: Rc<Cell<Option<usize>>>,
) -> crate::error::Result<Vec<Array>>
where
    U: Updatable,
{
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

    let inner_inputs_vector = {
        let borrow = state.borrow();
        VectorArray::try_from_iter(
            args.iter()
                .map(AsRef::as_ref)
                .chain(borrow.updatable_states()),
        )?
    };

    // will compile the function (if needed) and evaluate the
    // compiled graph
    let result_vector = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_closure_apply(res, compiled.as_ptr(), inner_inputs_vector.as_ptr())
    })?;

    let result_plus_state_output: Vec<Array> = result_vector.try_into_values()?;

    // The combined output layout is: [function_outputs..., state_arrays...]
    // We captured the function output count during tracing to know where to split.
    let num_fn_outputs = num_function_outputs.get().ok_or_else(|| {
        Exception::custom(
            "compile_with_state: internal error - function output count not captured during tracing"
        )
    })?;

    if num_fn_outputs > result_plus_state_output.len() {
        return Err(Exception::custom(format!(
            "compile_with_state: invalid output count - expected {} function outputs \
             but only got {} total outputs. This indicates an internal compilation error.",
            num_fn_outputs,
            result_plus_state_output.len()
        )));
    }

    let function_results = &result_plus_state_output[..num_fn_outputs];
    let state_outputs = &result_plus_state_output[num_fn_outputs..];

    // Update state arrays. MLX's compiler may prune unchanged arrays from output,
    // so zip() handles cases where fewer state arrays are returned than expected.
    for (s, new_values) in state
        .borrow_mut()
        .updatable_states_mut()
        .into_iter()
        .zip(state_outputs.iter())
    {
        update_by_replace_with_ref_to_new_array(s, new_values);
    }

    // Return only the function results (not the state arrays)
    Ok(function_results.to_vec())
}

fn state_snapshot(state: &impl Updatable) -> Vec<Array> {
    state
        .updatable_states()
        .into_iter()
        .map(Clone::clone)
        .collect()
}

fn restore_state(state: &mut impl Updatable, snapshot: &[Array]) {
    for (array, saved) in state.updatable_states_mut().into_iter().zip(snapshot) {
        update_by_replace_with_ref_to_new_array(array, saved);
    }
}

impl<F> CompiledState<F> {
    fn retry_call_mut_with_state<U>(
        &mut self,
        state: &mut U,
        args: &[impl AsRef<Array>],
    ) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&mut U, &[Array]) -> Vec<Array>,
        U: Updatable,
    {
        self.call_mut_with_state(state, args).or_else(|_e| {
            // Somehow the mlx_closure_apply may fail on the first call for
            // certain types of state with the error message:
            // "unordered_map::at: key not found", so we just try again.
            //
            // One type that is known to cause this is a tuple of
            // `Module` and `Optimizer` eg. `(<Module>, <Optimizer>)`
            self.call_mut_with_state(state, args)
        })
    }

    fn retry_fallible_call_mut_with_state<U>(
        &mut self,
        state: &mut U,
        args: &[impl AsRef<Array>],
    ) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception>,
        U: Updatable,
    {
        self.fallible_call_mut_with_state(state, args)
            .or_else(|_e| {
                // Somehow the mlx_closure_apply may fail on the first call for
                // certain types of state with the error message:
                // "unordered_map::at: key not found", so we just try again.
                //
                // One type that is known to cause this is a tuple of
                // `Module` and `Optimizer` eg. `(<Module>, <Optimizer>)`
                self.fallible_call_mut_with_state(state, args)
            })
    }

    fn call_mut_with_state<U>(
        &mut self,
        state: &mut U,
        args: &[impl AsRef<Array>],
    ) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&mut U, &[Array]) -> Vec<Array>,
        U: Updatable,
    {
        let args_len = args.len();
        let saved_state = state_snapshot(state);
        let state = Rc::new(RefCell::new(state));
        let f = &mut self.f;

        // Cell to capture the number of function outputs during tracing
        let num_function_outputs = Rc::new(Cell::new(None));
        let num_fn_outputs_clone = Rc::clone(&num_function_outputs);

        let state_clone = Rc::clone(&state);
        let inner = move |tracers: &[Array]| -> Vec<Array> {
            // put the tracers in their appropriate places:
            // - arguments to the function
            // - inner state

            let tracer_args = &tracers[..args_len];

            // save a snapshot of the inner state
            let saved_state_inputs = state_clone
                .borrow()
                .updatable_states()
                .into_iter()
                .map(|array| (*array).clone())
                .collect::<Vec<Array>>();

            // replace the inner state with the tracers
            for (s, tracer) in state_clone
                .borrow_mut()
                .updatable_states_mut()
                .into_iter()
                .zip(tracers.iter().skip(args_len))
            {
                update_by_replace_with_ref_to_new_array(s, tracer);
            }

            // call the function with the tracer arguments and the state holding tracers
            let mut result = (f)(*state_clone.borrow_mut(), tracer_args);

            // Capture function output count before appending state
            num_fn_outputs_clone.set(Some(result.len()));

            // recapture the state as it may have changed
            let mut state_output_tracers = state_clone
                .borrow()
                .updatable_states()
                .into_iter()
                .map(|array| (*array).clone())
                .collect::<Vec<Array>>();

            // put the original values back in the state
            for (s, saved) in state_clone
                .borrow_mut()
                .updatable_states_mut()
                .into_iter()
                .zip(saved_state_inputs)
            {
                update_by_replace_with_ref_to_new_array(s, &saved);
            }

            // return the result of the function and the state
            result.append(&mut state_output_tracers);

            result
        };

        let inner_closure = Closure::new(inner);
        let result = call_mut_with_state_inner(
            inner_closure,
            self.id,
            self.shapeless,
            Rc::clone(&state),
            args,
            num_function_outputs,
        );
        if result.is_err() {
            restore_state(*state.borrow_mut(), &saved_state);
        }
        result
    }

    fn fallible_call_mut_with_state<U>(
        &mut self,
        state: &mut U,
        args: &[impl AsRef<Array>],
    ) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception>,
        U: Updatable,
    {
        let args_len = args.len();
        let saved_state = state_snapshot(state);
        let state = Rc::new(RefCell::new(state));
        let f = &mut self.f;

        // Cell to capture the number of function outputs during tracing
        let num_function_outputs = Rc::new(Cell::new(None));
        let num_fn_outputs_clone = Rc::clone(&num_function_outputs);

        let state_clone = Rc::clone(&state);
        let inner = move |tracers: &[Array]| -> Result<Vec<Array>, Exception> {
            // put the tracers in their appropriate places:
            // - arguments to the function
            // - inner state

            let tracer_args = &tracers[..args_len];

            // save a snapshot of the inner state
            let saved_state_inputs = state_clone
                .borrow()
                .updatable_states()
                .into_iter()
                .map(|array| (*array).clone())
                .collect::<Vec<Array>>();

            // replace the inner state with the tracers
            for (s, tracer) in state_clone
                .borrow_mut()
                .updatable_states_mut()
                .into_iter()
                .zip(tracers.iter().skip(args_len))
            {
                update_by_replace_with_ref_to_new_array(s, tracer);
            }

            // call the function with the tracer arguments and the state holding tracers
            let mut result = (f)(*state_clone.borrow_mut(), tracer_args)?;

            // Capture function output count before appending state
            num_fn_outputs_clone.set(Some(result.len()));

            // recapture the state as it may have changed
            let mut state_output_tracers = state_clone
                .borrow()
                .updatable_states()
                .into_iter()
                .map(|array| (*array).clone())
                .collect::<Vec<Array>>();

            // put the original values back in the state
            for (s, saved) in state_clone
                .borrow_mut()
                .updatable_states_mut()
                .into_iter()
                .zip(saved_state_inputs)
            {
                update_by_replace_with_ref_to_new_array(s, &saved);
            }

            // return the result of the function and the state
            result.append(&mut state_output_tracers);

            Ok(result)
        };

        let inner_closure = Closure::new_fallible(inner);
        let result = call_mut_with_state_inner(
            inner_closure,
            self.id,
            self.shapeless,
            Rc::clone(&state),
            args,
            num_function_outputs,
        );
        if result.is_err() {
            restore_state(*state.borrow_mut(), &saved_state);
        }
        result
    }
}
