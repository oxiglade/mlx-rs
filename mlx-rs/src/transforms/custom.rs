//! Functions MLX builds from Rust closures: [`checkpoint`] and [`custom_vjp`].

use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

use mlx_sys::{mlx_closure_custom, mlx_vector_array};

use crate::{
    error::{get_and_clear_closure_error, set_closure_error, set_closure_panic, Result},
    utils::{guard::Guarded, Closure, VectorArray, FAILURE, SUCCESS},
    Array,
};

/// A function of arrays that MLX built from Rust closures, as [`checkpoint`] and [`custom_vjp`]
/// return it.
///
/// MLX holds the closures it was built from for as long as it needs them: as long as this value
/// lives, and as long as any graph traced through it does, since a backward pass over that graph
/// calls them again. They are therefore `'static`, and they should take every array they read as
/// an input: an array a closure captures is a constant to MLX's transforms.
#[derive(Debug)]
pub struct Function {
    closure: Closure<'static>,
}

impl Function {
    /// Calls the function on `inputs`. Under a transform ([`crate::transforms::vjp`],
    /// [`crate::transforms::value_and_grad`], ...) the call is traced like any other operation.
    ///
    /// An error a closure returns is returned here, or by the transform whose backward pass
    /// called it; a panic in a closure resumes in the caller with its payload.
    pub fn apply(&self, inputs: &[Array]) -> Result<Vec<Array>> {
        let inputs = VectorArray::try_from_iter(inputs.iter())?;
        <Vec<Array> as Guarded>::try_from_op(|res| unsafe {
            mlx_sys::mlx_closure_apply(res, self.closure.as_ptr(), inputs.as_ptr())
        })
        .map_err(|e| get_and_clear_closure_error().unwrap_or(e))
    }
}

/// `f`, with the arrays it computes dropped after the forward pass and computed again in the
/// backward pass: MLX's `checkpoint` (`mx.checkpoint`). The outputs and the gradients of every
/// input are those of `f`; what a backward pass keeps alive from the call is its inputs and
/// outputs, at the cost of running `f` again.
///
/// ```rust
/// use mlx_rs::{array, error::Result, transforms::{checkpoint, fallible_vjp}, Array};
///
/// let f = |x: &[Array]| -> Result<Vec<Array>> { Ok(vec![x[0].exp()?.multiply(&x[0])?]) };
/// let recomputed = checkpoint(f).unwrap();
///
/// let x = array!(1.5f32);
/// let (_, plain) = fallible_vjp(f, &[x.clone()], &[array!(1.0f32)]).unwrap();
/// let (_, again) =
///     fallible_vjp(|x: &[Array]| recomputed.apply(x), &[x], &[array!(1.0f32)]).unwrap();
/// assert_eq!(again[0].item_exact::<f32>(), plain[0].item_exact::<f32>());
/// ```
pub fn checkpoint<F>(f: F) -> Result<Function>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'static,
{
    let f = Closure::new_fallible(f);
    let closure = <Closure<'static> as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_checkpoint(res, f.as_ptr())
    })?;
    Ok(Function { closure })
}

/// `forward` with `vjp` as its backward pass: MLX's `custom_vjp` (`mx.custom_function` with a
/// `vjp`).
///
/// `vjp` is called with the primals (the inputs `forward` was called with), the cotangents of
/// its outputs, and the outputs themselves, and returns one cotangent per primal, each of its
/// primal's shape. MLX reads only the cotangents of the inputs a transform asked for.
///
/// ```rust
/// use mlx_rs::{array, error::Result, transforms::{custom_vjp, fallible_vjp}, Array};
///
/// // x * x, whose backward pass is written out as 2 x.
/// let square = custom_vjp(
///     |x: &[Array]| -> Result<Vec<Array>> { Ok(vec![x[0].multiply(&x[0])?]) },
///     |x: &[Array], cotangents: &[Array], _outputs: &[Array]| -> Result<Vec<Array>> {
///         Ok(vec![cotangents[0].multiply(&x[0])?.multiply(array!(2.0f32))?])
///     },
/// )
/// .unwrap();
/// let (_, grads) =
///     fallible_vjp(|x: &[Array]| square.apply(x), &[array!(3.0f32)], &[array!(1.0f32)]).unwrap();
/// assert_eq!(grads[0].item_exact::<f32>(), 6.0);
/// ```
pub fn custom_vjp<F, G>(forward: F, vjp: G) -> Result<Function>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'static,
    G: FnMut(&[Array], &[Array], &[Array]) -> Result<Vec<Array>> + 'static,
{
    let forward = Closure::new_fallible(forward);
    let vjp = ClosureCustom::new(vjp);
    let closure = <Closure<'static> as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_custom_vjp(res, forward.as_ptr(), vjp.c_closure)
    })?;
    Ok(Function { closure })
}

/// An `mlx_closure_custom` (primals, cotangents, outputs -> cotangents) that owns a Rust closure.
/// MLX copies it into the function it builds; dropping this releases only this handle.
struct ClosureCustom {
    c_closure: mlx_closure_custom,
}

impl ClosureCustom {
    fn new<G>(vjp: G) -> Self
    where
        G: FnMut(&[Array], &[Array], &[Array]) -> Result<Vec<Array>> + 'static,
    {
        let payload = Box::into_raw(Box::new(vjp)) as *mut c_void;
        let c_closure = unsafe {
            mlx_sys::mlx_closure_custom_new_func_payload(
                Some(trampoline_custom::<G>),
                payload,
                Some(drop_payload::<G>),
            )
        };
        Self { c_closure }
    }
}

impl Drop for ClosureCustom {
    fn drop(&mut self) {
        let status = unsafe { mlx_sys::mlx_closure_custom_free(self.c_closure) };
        debug_assert_eq!(status, SUCCESS);
    }
}

/// The arrays of a vector MLX owns, each with its own reference.
unsafe fn arrays(vector: mlx_vector_array) -> Result<Vec<Array>> {
    let size = mlx_sys::mlx_vector_array_size(vector);
    (0..size)
        .map(|i| Array::try_from_op(|res| mlx_sys::mlx_vector_array_get(res, vector, i)))
        .collect()
}

extern "C" fn trampoline_custom<G>(
    ret: *mut mlx_vector_array,
    primals: mlx_vector_array,
    cotangents: mlx_vector_array,
    outputs: mlx_vector_array,
    payload: *mut c_void,
) -> i32
where
    G: FnMut(&[Array], &[Array], &[Array]) -> Result<Vec<Array>>,
{
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let vjp = &mut *(payload as *mut G);
        let read =
            || -> Result<_> { Ok((arrays(primals)?, arrays(cotangents)?, arrays(outputs)?)) };
        let grads = match read().and_then(|(p, c, o)| vjp(&p, &c, &o)) {
            Ok(grads) => grads,
            Err(err) => {
                set_closure_error(err);
                return FAILURE;
            }
        };
        // mlx-c hands in an empty handle and frees whatever it holds after reading: fill it in
        // place rather than replace it.
        let ptrs: Vec<_> = grads.iter().map(|a| a.as_ptr()).collect();
        mlx_sys::mlx_vector_array_set_data(ret, ptrs.as_ptr(), ptrs.len())
    }));
    match result {
        Ok(status) => status,
        Err(payload) => {
            set_closure_panic(payload);
            FAILURE
        }
    }
}

extern "C" fn drop_payload<G>(payload: *mut c_void) {
    if payload.is_null() {
        return;
    }
    if let Err(panic) = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(payload as *mut G));
    })) {
        set_closure_panic(panic);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::rc::Rc;

    use crate::{
        array,
        error::{Exception, Result},
        transforms::{fallible_vjp, value_and_grad},
        Array,
    };

    use super::{checkpoint, custom_vjp};

    // Several intermediate arrays, so that recomputation has something to drop.
    fn f(x: &[Array]) -> Result<Vec<Array>> {
        let y = x[0].sin()?.multiply(x[0].exp()?)?;
        Ok(vec![y.multiply(&x[1])?.sum(None)?])
    }

    fn inputs() -> Vec<Array> {
        vec![array!([0.5f32, -1.0, 2.0]), array!([3.0f32, 0.25, -2.0])]
    }

    // Named invariant: a checkpointed function's outputs and gradients are the function's.
    #[test]
    fn test_checkpoint_gives_the_function_and_its_gradients() {
        let recomputed = checkpoint(f).unwrap();
        let cotangent = [array!(1.0f32)];
        let (plain_out, plain) = fallible_vjp(f, &inputs(), &cotangent).unwrap();
        let (out, grads) =
            fallible_vjp(|x: &[Array]| recomputed.apply(x), &inputs(), &cotangent).unwrap();
        assert_eq!(out[0].item_exact::<f32>(), plain_out[0].item_exact::<f32>());
        for (g, p) in grads.iter().zip(&plain) {
            assert_eq!(
                g.to_vec_exact::<f32>().unwrap(),
                p.to_vec_exact::<f32>().unwrap()
            );
        }
    }

    // Named invariant: checkpointing runs the function once forward and again backward.
    #[test]
    fn test_checkpoint_runs_the_function_again_in_the_backward_pass() {
        let calls = Rc::new(Cell::new(0));
        let counted = calls.clone();
        let recomputed = checkpoint(move |x: &[Array]| {
            counted.set(counted.get() + 1);
            f(x)
        })
        .unwrap();
        recomputed.apply(&inputs()).unwrap();
        assert_eq!(calls.get(), 1);
        calls.set(0);
        fallible_vjp(
            |x: &[Array]| recomputed.apply(x),
            &inputs(),
            &[array!(1.0f32)],
        )
        .unwrap();
        assert_eq!(calls.get(), 2);
    }

    // Named invariant: the gradient is the written-out backward pass's, here deliberately not the
    // forward pass's derivative, and the backward pass sees its primals, cotangents and outputs.
    #[test]
    fn test_custom_vjp_uses_the_written_backward_pass() {
        let seen = Rc::new(Cell::new(None));
        let record = seen.clone();
        let doubled = custom_vjp(
            |x: &[Array]| -> Result<Vec<Array>> { Ok(vec![x[0].multiply(array!(2.0f32))?]) },
            move |x: &[Array], cotangents: &[Array], outputs: &[Array]| -> Result<Vec<Array>> {
                record.set(Some((
                    x[0].item_exact::<f32>(),
                    cotangents[0].item_exact::<f32>(),
                    outputs[0].item_exact::<f32>(),
                )));
                Ok(vec![cotangents[0].multiply(array!(10.0f32))?])
            },
        )
        .unwrap();
        let (out, grads) = fallible_vjp(
            |x: &[Array]| doubled.apply(x),
            &[array!(3.0f32)],
            &[array!(0.5f32)],
        )
        .unwrap();
        assert_eq!(out[0].item_exact::<f32>(), 6.0);
        assert_eq!(grads[0].item_exact::<f32>(), 5.0);
        assert_eq!(seen.get(), Some((3.0, 0.5, 6.0)));
    }

    #[test]
    fn test_custom_vjp_under_value_and_grad() {
        let doubled = custom_vjp(
            |x: &[Array]| -> Result<Vec<Array>> { Ok(vec![x[0].multiply(array!(2.0f32))?]) },
            |_: &[Array], cotangents: &[Array], _: &[Array]| -> Result<Vec<Array>> {
                Ok(vec![cotangents[0].multiply(array!(2.0f32))?])
            },
        )
        .unwrap();
        let loss =
            |x: &[Array]| -> Result<Vec<Array>> { Ok(vec![doubled.apply(x)?[0].sum(None)?]) };
        let (value, grad) = value_and_grad(loss)(&[array!([1.0f32, 2.0])]).unwrap();
        assert_eq!(value[0].item_exact::<f32>(), 6.0);
        // The gradient of a sum is a broadcast: make it contiguous to read it.
        let grad = grad[0].contiguous().unwrap();
        assert_eq!(grad.to_vec_exact::<f32>().unwrap(), [2.0, 2.0]);
    }

    #[test]
    fn test_closure_errors_are_returned() {
        let failing = checkpoint(|_: &[Array]| -> Result<Vec<Array>> {
            Err(Exception::custom("forward failed"))
        })
        .unwrap();
        let err = failing.apply(&inputs()).unwrap_err();
        assert!(err.what().contains("forward failed"), "{err}");

        let backward_fails = custom_vjp(
            |x: &[Array]| -> Result<Vec<Array>> { Ok(vec![x[0].clone()]) },
            |_: &[Array], _: &[Array], _: &[Array]| -> Result<Vec<Array>> {
                Err(Exception::custom("backward failed"))
            },
        )
        .unwrap();
        let err = fallible_vjp(
            |x: &[Array]| backward_fails.apply(x),
            &[array!(1.0f32)],
            &[array!(1.0f32)],
        )
        .unwrap_err();
        assert!(err.what().contains("backward failed"), "{err}");
    }

    #[test]
    fn test_closure_panics_resume_with_their_payload() {
        let panics = custom_vjp(
            |x: &[Array]| -> Result<Vec<Array>> { Ok(vec![x[0].clone()]) },
            |_: &[Array], _: &[Array], _: &[Array]| -> Result<Vec<Array>> {
                panic!("backward panicked")
            },
        )
        .unwrap();
        let payload = catch_unwind(AssertUnwindSafe(|| {
            fallible_vjp(
                |x: &[Array]| panics.apply(x),
                &[array!(1.0f32)],
                &[array!(1.0f32)],
            )
        }))
        .unwrap_err();
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"backward panicked"));
        // The function stays usable after a panic in its backward pass.
        assert_eq!(
            panics.apply(&[array!(2.0f32)]).unwrap()[0].item_exact::<f32>(),
            2.0
        );
    }
}
