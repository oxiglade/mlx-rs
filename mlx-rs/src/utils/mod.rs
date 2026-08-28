//! Utility functions and types.

use guard::Guarded;
use mlx_sys::mlx_vector_array;

use crate::error::{set_closure_error, set_closure_panic};
use crate::module::ModuleParameters;
use crate::{complex64, error::Exception, Array, FromNested};
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::{marker::PhantomData, rc::Rc};

/// Success status code from the c binding
pub(crate) const SUCCESS: i32 = 0;
pub(crate) const FAILURE: i32 = 1;

pub(crate) mod guard;
pub(crate) mod io;

pub(crate) fn resolve_index_signed_unchecked(index: i32, len: i32) -> i32 {
    if index < 0 {
        len.saturating_add(index)
    } else {
        index
    }
}

pub(crate) fn resolve_index_unchecked(index: i32, len: usize) -> usize {
    if index.is_negative() {
        (len as i32 + index) as usize
    } else {
        index as usize
    }
}

/// Helper method to convert an optional slice of axes to a Vec covering all axes.
pub(crate) fn axes_or_default_to_all<'a>(axes: impl IntoOption<&'a [i32]>, ndim: i32) -> Vec<i32> {
    match axes.into_option() {
        Some(axes) => axes.to_vec(),
        None => {
            let axes: Vec<i32> = (0..ndim).collect();
            axes
        }
    }
}

pub(crate) struct VectorArray {
    c_vec: mlx_sys::mlx_vector_array,
}

impl VectorArray {
    pub(crate) fn as_ptr(&self) -> mlx_sys::mlx_vector_array {
        self.c_vec
    }

    pub(crate) fn try_from_iter(
        iter: impl Iterator<Item = impl AsRef<Array>>,
    ) -> Result<Self, Exception> {
        VectorArray::try_from_op(|res| unsafe {
            let mut status = SUCCESS;
            for arr in iter {
                status = mlx_sys::mlx_vector_array_append_value(*res, arr.as_ref().as_ptr());
                if status != SUCCESS {
                    return status;
                }
            }
            status
        })
    }

    pub(crate) fn try_into_values<T>(self) -> Result<T, Exception>
    where
        T: FromIterator<Array>,
    {
        unsafe {
            let size = mlx_sys::mlx_vector_array_size(self.c_vec);
            (0..size)
                .map(|i| {
                    Array::try_from_op(|res| mlx_sys::mlx_vector_array_get(res, self.c_vec, i))
                })
                .collect::<Result<T, Exception>>()
        }
    }
}

impl Drop for VectorArray {
    fn drop(&mut self) {
        let status = unsafe { mlx_sys::mlx_vector_array_free(self.c_vec) };
        debug_assert_eq!(status, SUCCESS);
    }
}

/// A helper trait that is just like `Into<Option<T>>` but improves ergonomics by allowing
/// implicit conversion from &[T; N] to &[T].
pub trait IntoOption<T> {
    /// Convert into an [`Option`].
    fn into_option(self) -> Option<T>;
}

impl<T> IntoOption<T> for Option<T> {
    fn into_option(self) -> Option<T> {
        self
    }
}

impl<T> IntoOption<T> for T {
    fn into_option(self) -> Option<T> {
        Some(self)
    }
}

impl<'a, T, const N: usize> IntoOption<&'a [T]> for &'a [T; N] {
    fn into_option(self) -> Option<&'a [T]> {
        Some(self)
    }
}

impl<'a, T> IntoOption<&'a [T]> for &'a Vec<T> {
    fn into_option(self) -> Option<&'a [T]> {
        Some(self)
    }
}

/// A trait for a scalar or an array.
pub trait ScalarOrArray<'a> {
    /// The reference type of the array.
    type Array: AsRef<Array> + 'a;

    /// Convert to an owned or reference array.
    fn into_owned_or_ref_array(self) -> Self::Array;
}

impl ScalarOrArray<'_> for Array {
    type Array = Array;

    fn into_owned_or_ref_array(self) -> Array {
        self
    }
}

impl<'a> ScalarOrArray<'a> for &'a Array {
    type Array = &'a Array;

    // TODO: clippy would complain about `as_array`. Is there a better name?
    fn into_owned_or_ref_array(self) -> &'a Array {
        self
    }
}

impl ScalarOrArray<'static> for bool {
    type Array = Array;

    fn into_owned_or_ref_array(self) -> Array {
        Array::from_bool(self)
    }
}

impl ScalarOrArray<'static> for i32 {
    type Array = Array;

    fn into_owned_or_ref_array(self) -> Array {
        Array::from_int(self)
    }
}

impl ScalarOrArray<'static> for f32 {
    type Array = Array;

    fn into_owned_or_ref_array(self) -> Array {
        Array::from_f32(self)
    }
}

// TODO: this is bugged right now. See https://github.com/ml-explore/mlx/issues/1994
// impl ScalarOrArray<'static> for f64 {
//     type Array = Array;

//     fn into_owned_or_ref_array(self) -> Array {
//         Array::from_f64(self)
//     }
// }

impl ScalarOrArray<'static> for complex64 {
    type Array = Array;

    fn into_owned_or_ref_array(self) -> Array {
        Array::from_complex(self)
    }
}

impl<T> ScalarOrArray<'static> for T
where
    Array: FromNested<T>,
{
    type Array = Array;

    fn into_owned_or_ref_array(self) -> Array {
        Array::from_nested(self)
    }
}

#[derive(Debug)]
pub(crate) struct Closure<'a> {
    c_closure: mlx_sys::mlx_closure,
    lt_marker: PhantomData<&'a ()>,
}

impl<'a> Closure<'a> {
    pub(crate) fn as_ptr(&self) -> mlx_sys::mlx_closure {
        self.c_closure
    }

    pub(crate) fn new<F>(closure: F) -> Self
    where
        F: FnMut(&[Array]) -> Vec<Array> + 'a,
    {
        let c_closure = new_mlx_closure(closure);
        Self {
            c_closure,
            lt_marker: PhantomData,
        }
    }

    pub(crate) fn new_fallible<F>(closure: F) -> Self
    where
        F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
    {
        let c_closure = new_mlx_fallible_closure(closure);
        Self {
            c_closure,
            lt_marker: PhantomData,
        }
    }
}

impl Drop for Closure<'_> {
    fn drop(&mut self) {
        let status = unsafe { mlx_sys::mlx_closure_free(self.c_closure) };
        if !std::thread::panicking() {
            crate::error::resume_closure_panic();
        }
        debug_assert_eq!(status, SUCCESS);
    }
}

/// Helper method to create a mlx_closure from a Rust closure.
fn new_mlx_closure<'a, F>(closure: F) -> mlx_sys::mlx_closure
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    // Box the closure to keep it on the heap
    let boxed = Box::new(closure);

    // Create a raw pointer from the Box, transferring ownership to C
    let raw = Box::into_raw(boxed);
    let payload = raw as *mut std::ffi::c_void;

    unsafe {
        mlx_sys::mlx_closure_new_func_payload(
            Some(trampoline::<F>),
            payload,
            Some(closure_dtor::<F>),
        )
    }
}

fn new_mlx_fallible_closure<'a, F>(closure: F) -> mlx_sys::mlx_closure
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    let boxed = Box::new(closure);
    let raw = Box::into_raw(boxed);
    let payload = raw as *mut std::ffi::c_void;

    unsafe {
        mlx_sys::mlx_closure_new_func_payload(
            Some(trampoline_fallible::<F>),
            payload,
            Some(closure_dtor::<F>),
        )
    }
}

/// Function to create a new (+1 reference) mlx_vector_array from a vector of Array
fn new_mlx_vector_array(arrays: Vec<Array>) -> mlx_sys::mlx_vector_array {
    unsafe {
        let result = mlx_sys::mlx_vector_array_new();
        let ctx_ptrs: Vec<mlx_sys::mlx_array> = arrays.iter().map(|array| array.as_ptr()).collect();
        mlx_sys::mlx_vector_array_append_data(result, ctx_ptrs.as_ptr(), arrays.len());
        result
    }
}

fn mlx_vector_array_values(
    vector_array: mlx_sys::mlx_vector_array,
) -> Result<Vec<Array>, Exception> {
    unsafe {
        let size = mlx_sys::mlx_vector_array_size(vector_array);
        (0..size)
            .map(|index| {
                Array::try_from_op(|res| mlx_sys::mlx_vector_array_get(res, vector_array, index))
            })
            .collect()
    }
}

extern "C" fn trampoline<'a, F>(
    ret: *mut mlx_vector_array,
    vector_array: mlx_vector_array,
    payload: *mut std::ffi::c_void,
) -> i32
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let raw_closure: *mut F = payload as *mut _;
        let closure = &mut *raw_closure;
        let arrays = match mlx_vector_array_values(vector_array) {
            Ok(arrays) => arrays,
            Err(_) => return None,
        };
        let result = closure(&arrays);

        *ret = new_mlx_vector_array(result);
        Some(())
    }));

    match result {
        Ok(Some(())) => SUCCESS,
        Ok(None) => FAILURE,
        Err(payload) => {
            set_closure_panic(payload);
            FAILURE
        }
    }
}

extern "C" fn trampoline_fallible<'a, F>(
    ret: *mut mlx_vector_array,
    vector_array: mlx_vector_array,
    payload: *mut std::ffi::c_void,
) -> i32
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let raw_closure: *mut F = payload as *mut _;
        let closure = &mut *raw_closure;
        let arrays = match mlx_vector_array_values(vector_array) {
            Ok(arrays) => arrays,
            Err(e) => {
                set_closure_error(e);
                return FAILURE;
            }
        };
        let result = closure(&arrays);

        match result {
            Ok(result) => {
                *ret = new_mlx_vector_array(result);
                SUCCESS
            }
            Err(err) => {
                set_closure_error(err);
                FAILURE
            }
        }
    }));

    match result {
        Ok(status) => status,
        Err(payload) => {
            set_closure_panic(payload);
            FAILURE
        }
    }
}

// extern "C" fn noop_dtor(_data: *mut std::ffi::c_void) {}

extern "C" fn closure_dtor<F>(payload: *mut std::ffi::c_void) {
    if payload.is_null() {
        return;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(payload as *mut F));
    }));
    if let Err(payload) = result {
        set_closure_panic(payload);
    }
}

pub(crate) fn get_mut_or_insert_with<'a, T>(
    map: &'a mut HashMap<Rc<str>, T>,
    key: &Rc<str>,
    f: impl FnOnce() -> T,
) -> &'a mut T {
    if !map.contains_key(key) {
        map.insert(key.clone(), f());
    }

    map.get_mut(key).unwrap()
}

/// Helper trait for compiling a function that takes a Module and/or an Optimizer.
/// The implementation must ensure consistent ordering of the returned states.
///
/// This is automatically implemented for all types that implement ModuleParameters.
pub trait Updatable {
    /// Returns the number of updatable states.
    ///
    /// The number should be the same as calling `self.updatable_states().len()` but
    /// this method should be more efficient in general. The implementation should
    /// avoid iterating over the states if possible.
    fn updatable_states_len(&self) -> usize;

    /// Returns a list of references to the updatable states.
    ///
    /// The order of the states should be consistent across calls and should be the same as the
    /// order of the states returned by [`Updatable::updatable_states_mut`].
    fn updatable_states(&self) -> impl IntoIterator<Item = &Array>;

    /// Returns a list of mutable references to the updatable states.
    ///
    /// The order of the states should be consistent across calls and should be the same as the
    /// order of the states returned by [`Updatable::updatable_states`].
    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array>;
}

impl<T> Updatable for T
where
    T: ModuleParameters,
{
    fn updatable_states_len(&self) -> usize {
        self.num_parameters()
    }

    fn updatable_states(&self) -> impl IntoIterator<Item = &Array> {
        use itertools::Itertools;

        // TODO: should we change the parameter map to a BTreeMap because it is sorted?
        self.parameters()
            .flatten()
            .into_iter()
            .sorted_by(|a, b| a.0.cmp(&b.0))
            .map(|(_, v)| v)
    }

    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array> {
        use itertools::Itertools;

        self.parameters_mut()
            .flatten()
            .into_iter()
            .sorted_by(|a, b| a.0.cmp(&b.0))
            .map(|(_, v)| v)
    }
}

impl<T1, T2> Updatable for (T1, T2)
where
    T1: Updatable,
    T2: Updatable,
{
    fn updatable_states_len(&self) -> usize {
        let (a, b) = self;
        a.updatable_states_len() + b.updatable_states_len()
    }

    fn updatable_states(&self) -> impl IntoIterator<Item = &Array> {
        let (a, b) = self;
        let params = a.updatable_states();
        params.into_iter().chain(b.updatable_states())
    }

    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array> {
        let (a, b) = self;
        let params = a.updatable_states_mut();
        params.into_iter().chain(b.updatable_states_mut())
    }
}

impl Updatable for Vec<Array> {
    fn updatable_states_len(&self) -> usize {
        self.len()
    }

    fn updatable_states(&self) -> impl IntoIterator<Item = &Array> {
        self.iter()
    }

    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array> {
        self.iter_mut()
    }
}

/// Helper type to represent either a single value or a pair of values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingleOrPair<T = i32> {
    /// Single value.
    Single(T),

    /// Pair of values.
    Pair(T, T),
}

impl<T: Clone> SingleOrPair<T> {
    /// Returns the first value.
    pub fn first(&self) -> T {
        match self {
            SingleOrPair::Single(v) => v.clone(),
            SingleOrPair::Pair(v1, _) => v1.clone(),
        }
    }

    /// Returns the second value.
    pub fn second(&self) -> T {
        match self {
            SingleOrPair::Single(v) => v.clone(),
            SingleOrPair::Pair(_, v2) => v2.clone(),
        }
    }
}

impl<T> From<T> for SingleOrPair<T> {
    fn from(value: T) -> Self {
        SingleOrPair::Single(value)
    }
}

impl<T> From<(T, T)> for SingleOrPair<T> {
    fn from(value: (T, T)) -> Self {
        SingleOrPair::Pair(value.0, value.1)
    }
}

impl<T: Clone> From<SingleOrPair<T>> for (T, T) {
    fn from(value: SingleOrPair<T>) -> Self {
        match value {
            SingleOrPair::Single(v) => (v.clone(), v),
            SingleOrPair::Pair(v1, v2) => (v1, v2),
        }
    }
}

/// Helper type to represent either a single value or a triple of values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingleOrTriple<T = i32> {
    /// Single value.
    Single(T),

    /// Triple of values.
    Triple(T, T, T),
}

impl<T: Clone> SingleOrTriple<T> {
    /// Returns the first value.
    pub fn first(&self) -> T {
        match self {
            SingleOrTriple::Single(v) => v.clone(),
            SingleOrTriple::Triple(v1, _, _) => v1.clone(),
        }
    }

    /// Returns the second value.
    pub fn second(&self) -> T {
        match self {
            SingleOrTriple::Single(v) => v.clone(),
            SingleOrTriple::Triple(_, v2, _) => v2.clone(),
        }
    }

    /// Returns the third value.
    pub fn third(&self) -> T {
        match self {
            SingleOrTriple::Single(v) => v.clone(),
            SingleOrTriple::Triple(_, _, v3) => v3.clone(),
        }
    }
}

impl<T> From<T> for SingleOrTriple<T> {
    fn from(value: T) -> Self {
        SingleOrTriple::Single(value)
    }
}

impl<T> From<(T, T, T)> for SingleOrTriple<T> {
    fn from(value: (T, T, T)) -> Self {
        SingleOrTriple::Triple(value.0, value.1, value.2)
    }
}

impl<T: Clone> From<SingleOrTriple<T>> for (T, T, T) {
    fn from(value: SingleOrTriple<T>) -> Self {
        match value {
            SingleOrTriple::Single(v) => (v.clone(), v.clone(), v),
            SingleOrTriple::Triple(v1, v2, v3) => (v1, v2, v3),
        }
    }
}

/// Helper type to represent either a single value or a vector of values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SingleOrVec<T> {
    /// Single value.
    Single(T),

    /// Vector of values.
    Vec(Vec<T>),
}

impl<T> From<T> for SingleOrVec<T> {
    fn from(value: T) -> Self {
        SingleOrVec::Single(value)
    }
}

impl<T> From<Vec<T>> for SingleOrVec<T> {
    fn from(value: Vec<T>) -> Self {
        SingleOrVec::Vec(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;
    use std::process::Command;

    const PANIC_CHILD: &str = "MLX_RS_TRAMPOLINE_PANIC_CHILD";
    const DROP_DURING_UNWIND_CHILD: &str = "MLX_RS_DROP_DURING_UNWIND_CHILD";

    #[test]
    fn closure_trampoline_panics_resume_in_rust() {
        if std::env::var_os(PANIC_CHILD).is_some() {
            run_trampoline_panic_child();
            return;
        }

        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "utils::tests::closure_trampoline_panics_resume_in_rust",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(PANIC_CHILD, "1")
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "child status: {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn closure_drop_during_unwind_preserves_pending_panic() {
        if std::env::var_os(DROP_DURING_UNWIND_CHILD).is_some() {
            run_drop_during_unwind_child();
            return;
        }

        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "utils::tests::closure_drop_during_unwind_preserves_pending_panic",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(DROP_DURING_UNWIND_CHILD, "1")
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "child status: {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn run_drop_during_unwind_child() {
        let payload = catch_unwind(AssertUnwindSafe(|| {
            let _closure = Closure::new(|_| Vec::new());
            set_closure_panic(Box::new("pending closure panic"));
            panic!("unrelated unwind");
        }))
        .expect_err("the unrelated panic should remain catchable");
        assert_eq!(panic_message(payload).as_deref(), Some("unrelated unwind"));
        assert_captured_panic("pending closure panic");
    }

    fn run_trampoline_panic_child() {
        type Infallible = fn(&[Array]) -> Vec<Array>;
        let payload = Box::into_raw(Box::new(panic_infallible as Infallible)).cast();
        let input = unsafe { mlx_sys::mlx_vector_array_new() };
        let mut output = unsafe { mlx_sys::mlx_vector_array_new() };
        let status = trampoline::<Infallible>(&mut output, input, payload);
        assert_eq!(status, FAILURE);
        assert_captured_panic("infallible trampoline panic");
        closure_dtor::<Infallible>(payload);
        unsafe {
            mlx_sys::mlx_vector_array_free(input);
            mlx_sys::mlx_vector_array_free(output);
        }

        let payload = Box::into_raw(Box::new(PanicOnDrop)).cast();
        closure_dtor::<PanicOnDrop>(payload);
        assert_captured_panic("closure destructor panic");

        type Fallible = fn(&[Array]) -> Result<Vec<Array>, Exception>;
        let payload = Box::into_raw(Box::new(panic_fallible as Fallible)).cast();
        let input = unsafe { mlx_sys::mlx_vector_array_new() };
        let mut output = unsafe { mlx_sys::mlx_vector_array_new() };
        let status = trampoline_fallible::<Fallible>(&mut output, input, payload);
        assert_eq!(status, FAILURE);
        assert_captured_panic("fallible trampoline panic");
        closure_dtor::<Fallible>(payload);
        unsafe {
            mlx_sys::mlx_vector_array_free(input);
            mlx_sys::mlx_vector_array_free(output);
        }
    }

    fn panic_infallible(_: &[Array]) -> Vec<Array> {
        panic!("infallible trampoline panic")
    }

    fn panic_fallible(_: &[Array]) -> Result<Vec<Array>, Exception> {
        panic!("fallible trampoline panic")
    }

    struct PanicOnDrop;

    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("closure destructor panic")
        }
    }

    fn assert_captured_panic(expected: &str) {
        let payload = catch_unwind(crate::error::resume_closure_panic)
            .expect_err("the trampoline panic should resume in Rust");
        assert_eq!(panic_message(payload).as_deref(), Some(expected));
    }

    fn panic_message(payload: Box<dyn Any + Send>) -> Option<String> {
        payload
            .downcast_ref::<&'static str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
    }
}
