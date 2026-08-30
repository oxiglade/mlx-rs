//! Compilation of functions with state.
//!
//! # Unit tests
//!
//! See `mlx-rs/mlx-tests/tests/test_compile.rs` for unit tests.

// TODO: there's plenty boilerplate code here but it's not clear how to reduce it

use std::{
    cell::{Cell, RefCell},
    collections::BTreeSet,
    marker::PhantomData,
    rc::Rc,
};

use crate::{error::Exception, transforms::compile::CompiledState, utils::Updatable, Array};

use super::{Closure, Compiled, Guarded, VectorArray};

/// Similar to [`crate::transforms::compile`] but allows for functions that take
/// a mutable reference to a state `U`.
pub fn compile_with_state<F, U, A, O, E>(
    f: F,
    shapeless: impl Into<Option<bool>>,
) -> impl for<'a> FnMut(&mut U, F::Args<'a>) -> Result<O, Exception>
where
    F: CompileWithState<U, A, O, E> + 'static,
    U: Updatable,
{
    let shapeless = shapeless.into().unwrap_or(false);
    let mut compiled = f.compile(shapeless);
    move |state, args| compiled.call_mut(state, args)
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
    fn compile(
        self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, O, E>;
}

impl<F, U> CompileWithState<U, &[Array], Vec<Array>, ()> for F
where
    F: FnMut(&mut U, &[Array]) -> Vec<Array> + 'static,
    U: Updatable,
{
    type Args<'a> = &'a [Array];

    fn compile(
        self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, Vec<Array>, ()> {
        let state = CompiledState::new(self, shapeless);
        Compiled::<F, _> {
            f_marker: PhantomData,
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

    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, Array, ()> {
        let f = move |state: &mut U, args: &[Array]| -> Vec<Array> {
            let result = (self)(state, &args[0]);
            vec![result]
        };
        let state = CompiledState::new(f, shapeless);
        Compiled::<F, _> {
            f_marker: PhantomData,
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

    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, Array, ()> {
        let f = move |state: &mut U, args: &[Array]| -> Vec<Array> {
            let result = (self)(state, (&args[0], &args[1]));
            vec![result]
        };
        let state = CompiledState::new(f, shapeless);
        Compiled::<F, _> {
            f_marker: PhantomData,
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

    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, Array, ()> {
        let f = move |state: &mut U, args: &[Array]| -> Vec<Array> {
            let result = (self)(state, (&args[0], &args[1], &args[2]));
            vec![result]
        };
        let state = CompiledState::new(f, shapeless);
        Compiled::<F, _> {
            f_marker: PhantomData,
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

    fn compile(
        self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, Vec<Array>, Exception> {
        let state = CompiledState::new(self, shapeless);
        Compiled::<F, _> {
            f_marker: PhantomData,
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

    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, Array, Exception> {
        let f = move |state: &mut U, args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)(state, &args[0])?;
            Ok(vec![result])
        };
        let state = CompiledState::new(f, shapeless);
        Compiled::<F, _> {
            f_marker: PhantomData,
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

    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, Array, Exception> {
        let f = move |state: &mut U, args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)(state, (&args[0], &args[1]))?;
            Ok(vec![result])
        };
        let state = CompiledState::new(f, shapeless);
        Compiled::<F, _> {
            f_marker: PhantomData,
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

    fn compile(
        mut self,
        shapeless: bool,
    ) -> impl for<'args> CallMutWithState<U, Self::Args<'args>, Array, Exception> {
        let f = move |state: &mut U, args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)(state, (&args[0], &args[1], &args[2]))?;
            Ok(vec![result])
        };
        let state = CompiledState::new(f, shapeless);
        Compiled::<F, _> {
            f_marker: PhantomData,
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
        self.state.call_mut_with_state(state, args)
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
        let result = self.state.call_mut_with_state(state, args)?;
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
        let result = self.state.call_mut_with_state(state, args)?;
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
        let result = self.state.call_mut_with_state(state, args)?;
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
        self.state.fallible_call_mut_with_state(state, args)
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
        let result = self.state.fallible_call_mut_with_state(state, args)?;
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
        let result = self.state.fallible_call_mut_with_state(state, args)?;
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
        let result = self.state.fallible_call_mut_with_state(state, args)?;
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
    state_layout: Rc<RefCell<Option<Vec<(Rc<str>, crate::Dtype, Vec<i32>)>>>>,
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
        let state_inputs = borrow.updatable_state_snapshot();
        VectorArray::try_from_iter(
            args.iter()
                .map(AsRef::as_ref)
                .chain(state_inputs.iter().map(|(_, array)| array)),
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
    let expected_state_layout = state_layout.borrow().clone().ok_or_else(|| {
        Exception::custom(
            "compile_with_state: internal error - state layout not captured during tracing",
        )
    })?;
    validate_state_layout(&expected_state_layout, &**state.borrow(), "apply input")?;

    let expected_output_count = num_fn_outputs + expected_state_layout.len();
    if result_plus_state_output.len() != expected_output_count {
        return Err(Exception::custom(format!(
            "compile_with_state: invalid output count - expected {num_fn_outputs} function \
             outputs and {} state outputs, got {} total outputs",
            expected_state_layout.len(),
            result_plus_state_output.len()
        )));
    }

    let function_results = &result_plus_state_output[..num_fn_outputs];
    let state_outputs = &result_plus_state_output[num_fn_outputs..];

    let output_layout = expected_state_layout
        .iter()
        .enumerate()
        .map(|(index, (key, _, _))| {
            let array = &state_outputs[index];
            (key.clone(), array.dtype(), array.shape().to_vec())
        })
        .collect::<Vec<_>>();
    if output_layout != expected_state_layout {
        return Err(Exception::custom(format!(
            "compile_with_state: state output layout changed: expected \
             {expected_state_layout:?}, got {output_layout:?}"
        )));
    }

    let state_output = expected_state_layout
        .iter()
        .enumerate()
        .map(|(index, (key, _, _))| (key.clone(), state_outputs[index].clone()))
        .collect::<Vec<_>>();
    state
        .borrow_mut()
        .restore_updatable_state(state_output, false)
        .map_err(|error| {
            Exception::custom(format!(
                "compile_with_state: cannot apply compiled state: {error}"
            ))
        })?;

    // Return only the function results (not the state arrays)
    Ok(function_results.to_vec())
}

fn current_state_layout(state: &impl Updatable) -> Vec<(Rc<str>, crate::Dtype, Vec<i32>)> {
    state
        .updatable_state_snapshot()
        .into_iter()
        .map(|(key, array)| (key, array.dtype(), array.shape().to_vec()))
        .collect()
}

fn validate_state_layout(
    expected: &[(Rc<str>, crate::Dtype, Vec<i32>)],
    state: &impl Updatable,
    phase: &str,
) -> Result<(), Exception> {
    let actual = current_state_layout(state);
    if actual == expected {
        Ok(())
    } else {
        Err(Exception::custom(format!(
            "compile_with_state: state layout changed at {phase}: expected {expected:?}, got \
             {actual:?}"
        )))
    }
}

fn state_snapshot(state: &impl Updatable) -> Vec<(Rc<str>, Array)> {
    state.updatable_state_snapshot()
}

fn restore_state(
    state: &mut impl Updatable,
    snapshot: &[(Rc<str>, Array)],
    reset_new: bool,
) -> Result<(), Exception> {
    state
        .restore_updatable_state(snapshot.to_vec(), reset_new)
        .map_err(|error| {
            Exception::custom(format!("compile_with_state: cannot restore state: {error}"))
        })
}

fn state_grew(before: &[(Rc<str>, Array)], after: &[(Rc<str>, Array)]) -> bool {
    let before = before
        .iter()
        .map(|(key, _)| key.clone())
        .collect::<BTreeSet<_>>();
    let after = after
        .iter()
        .map(|(key, _)| key.clone())
        .collect::<BTreeSet<_>>();
    before.len() < after.len() && before.is_subset(&after)
}

fn is_recoverable_first_trace_growth(error: &Exception) -> bool {
    error.what().contains("unordered_map::at: key not found")
}

impl<F> CompiledState<F> {
    fn call_mut_with_state<U>(
        &mut self,
        state: &mut U,
        args: &[impl AsRef<Array>],
    ) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&mut U, &[Array]) -> Vec<Array>,
        U: Updatable,
    {
        self.call_mut_with_state_attempt(state, args, true)
    }

    fn call_mut_with_state_attempt<U>(
        &mut self,
        state: &mut U,
        args: &[impl AsRef<Array>],
        allow_growth_recovery: bool,
    ) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&mut U, &[Array]) -> Vec<Array>,
        U: Updatable,
    {
        let was_untraced = self.state_layout.is_none();
        if let Some(expected) = self.state_layout.as_deref() {
            validate_state_layout(expected, state, "call input")?;
        }
        let args_len = args.len();
        let saved_state = state_snapshot(state);
        let state = Rc::new(RefCell::new(state));
        let f = &mut self.f;

        // Cell to capture the number of function outputs during tracing
        let num_function_outputs = Rc::new(Cell::new(self.num_function_outputs));
        let num_fn_outputs_clone = Rc::clone(&num_function_outputs);
        let state_layout = Rc::new(RefCell::new(self.state_layout.clone()));
        let state_layout_clone = Rc::clone(&state_layout);

        let state_clone = Rc::clone(&state);
        let inner = move |tracers: &[Array]| -> Result<Vec<Array>, Exception> {
            // put the tracers in their appropriate places:
            // - arguments to the function
            // - inner state

            let tracer_args = &tracers[..args_len];

            // save a snapshot of the inner state
            let saved_state_inputs = state_snapshot(&**state_clone.borrow());

            // replace the inner state with the tracers
            if tracers.len() != args_len + saved_state_inputs.len() {
                return Err(Exception::custom(format!(
                    "compile_with_state: invalid tracer count - expected {} arguments and {} state inputs, got {} total inputs",
                    args_len,
                    saved_state_inputs.len(),
                    tracers.len()
                )));
            }
            let tracer_state = saved_state_inputs
                .iter()
                .enumerate()
                .map(|(index, (key, _))| (key.clone(), tracers[args_len + index].clone()))
                .collect::<Vec<_>>();
            state_clone
                .borrow_mut()
                .restore_updatable_state(tracer_state, false)
                .map_err(|error| {
                    Exception::custom(format!(
                        "compile_with_state: cannot install state tracers: {error}"
                    ))
                })?;

            // call the function with the tracer arguments and the state holding tracers
            let mut result = (f)(*state_clone.borrow_mut(), tracer_args);

            // Capture function output count before appending state
            num_fn_outputs_clone.set(Some(result.len()));

            // recapture the state as it may have changed
            let state_output = state_snapshot(&**state_clone.borrow());
            let mut state_output_tracers = state_output
                .iter()
                .map(|(_, array)| array.clone())
                .collect::<Vec<_>>();

            if state_layout_clone.borrow().is_none() {
                *state_layout_clone.borrow_mut() = Some(
                    state_output
                        .into_iter()
                        .map(|(key, array)| (key, array.dtype(), array.shape().to_vec()))
                        .collect(),
                );
            }

            // put the original values back in the state
            restore_state(*state_clone.borrow_mut(), &saved_state_inputs, true)?;

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
            Rc::clone(&num_function_outputs),
            Rc::clone(&state_layout),
        );
        self.num_function_outputs = num_function_outputs.get();
        self.state_layout = state_layout.borrow().clone();
        if let Err(error) = &result {
            let after_failure = state_snapshot(&**state.borrow());
            if allow_growth_recovery
                && was_untraced
                && state_grew(&saved_state, &after_failure)
                && is_recoverable_first_trace_growth(error)
            {
                restore_state(*state.borrow_mut(), &saved_state, true)?;
                let retry =
                    self.call_mut_with_state_attempt(&mut **state.borrow_mut(), args, false);
                if retry.is_err() {
                    restore_state(*state.borrow_mut(), &saved_state, false)?;
                    self.cache.erase(self.id);
                    self.num_function_outputs = None;
                    self.state_layout = None;
                }
                return retry;
            }
            restore_state(*state.borrow_mut(), &saved_state, false).map_err(|restore_error| {
                Exception::custom(format!(
                    "{}; transactional restore failed: {}",
                    error.what(),
                    restore_error.what()
                ))
            })?;
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
        self.fallible_call_mut_with_state_attempt(state, args, true)
    }

    fn fallible_call_mut_with_state_attempt<U>(
        &mut self,
        state: &mut U,
        args: &[impl AsRef<Array>],
        allow_growth_recovery: bool,
    ) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&mut U, &[Array]) -> Result<Vec<Array>, Exception>,
        U: Updatable,
    {
        let was_untraced = self.state_layout.is_none();
        if let Some(expected) = self.state_layout.as_deref() {
            validate_state_layout(expected, state, "call input")?;
        }
        let args_len = args.len();
        let saved_state = state_snapshot(state);
        let state = Rc::new(RefCell::new(state));
        let f = &mut self.f;

        // Cell to capture the number of function outputs during tracing
        let num_function_outputs = Rc::new(Cell::new(self.num_function_outputs));
        let num_fn_outputs_clone = Rc::clone(&num_function_outputs);
        let state_layout = Rc::new(RefCell::new(self.state_layout.clone()));
        let state_layout_clone = Rc::clone(&state_layout);

        let state_clone = Rc::clone(&state);
        let inner = move |tracers: &[Array]| -> Result<Vec<Array>, Exception> {
            // put the tracers in their appropriate places:
            // - arguments to the function
            // - inner state

            let tracer_args = &tracers[..args_len];

            // save a snapshot of the inner state
            let saved_state_inputs = state_snapshot(&**state_clone.borrow());

            // replace the inner state with the tracers
            if tracers.len() != args_len + saved_state_inputs.len() {
                return Err(Exception::custom(format!(
                    "compile_with_state: invalid tracer count - expected {} arguments and {} state inputs, got {} total inputs",
                    args_len,
                    saved_state_inputs.len(),
                    tracers.len()
                )));
            }
            let tracer_state = saved_state_inputs
                .iter()
                .enumerate()
                .map(|(index, (key, _))| (key.clone(), tracers[args_len + index].clone()))
                .collect::<Vec<_>>();
            state_clone
                .borrow_mut()
                .restore_updatable_state(tracer_state, false)
                .map_err(|error| {
                    Exception::custom(format!(
                        "compile_with_state: cannot install state tracers: {error}"
                    ))
                })?;

            // call the function with the tracer arguments and the state holding tracers
            let call_result = {
                let mut state = state_clone.borrow_mut();
                (f)(&mut **state, tracer_args)
            };
            let mut result = match call_result {
                Ok(result) => result,
                Err(error) => {
                    restore_state(*state_clone.borrow_mut(), &saved_state_inputs, false).map_err(
                        |restore_error| {
                            Exception::custom(format!(
                                "{}; transactional restore failed: {}",
                                error.what(),
                                restore_error.what()
                            ))
                        },
                    )?;
                    return Err(error);
                }
            };

            // Capture function output count before appending state
            num_fn_outputs_clone.set(Some(result.len()));

            // recapture the state as it may have changed
            let state_output = state_snapshot(&**state_clone.borrow());
            let mut state_output_tracers = state_output
                .iter()
                .map(|(_, array)| array.clone())
                .collect::<Vec<_>>();

            if state_layout_clone.borrow().is_none() {
                *state_layout_clone.borrow_mut() = Some(
                    state_output
                        .into_iter()
                        .map(|(key, array)| (key, array.dtype(), array.shape().to_vec()))
                        .collect(),
                );
            }

            // put the original values back in the state
            restore_state(*state_clone.borrow_mut(), &saved_state_inputs, true)?;

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
            Rc::clone(&num_function_outputs),
            Rc::clone(&state_layout),
        );
        self.num_function_outputs = num_function_outputs.get();
        self.state_layout = state_layout.borrow().clone();
        if let Err(error) = &result {
            let after_failure = state_snapshot(&**state.borrow());
            if allow_growth_recovery
                && was_untraced
                && state_grew(&saved_state, &after_failure)
                && is_recoverable_first_trace_growth(error)
            {
                restore_state(*state.borrow_mut(), &saved_state, true)?;
                let retry = self.fallible_call_mut_with_state_attempt(
                    &mut **state.borrow_mut(),
                    args,
                    false,
                );
                if retry.is_err() {
                    restore_state(*state.borrow_mut(), &saved_state, false)?;
                    self.cache.erase(self.id);
                    self.num_function_outputs = None;
                    self.state_layout = None;
                }
                return retry;
            }
            restore_state(*state.borrow_mut(), &saved_state, false).map_err(|restore_error| {
                Exception::custom(format!(
                    "{}; transactional restore failed: {}",
                    error.what(),
                    restore_error.what()
                ))
            })?;
        }
        result
    }
}
