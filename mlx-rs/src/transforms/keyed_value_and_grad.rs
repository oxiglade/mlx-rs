use std::{collections::HashMap, rc::Rc};

use crate::{
    error::{Exception, Result},
    utils::{guard::Guarded, Closure},
    Array,
};

use super::{value_and_gradient, ClosureValueAndGrad};

/// Type alias for a hashmap of parameters.
pub type KeyedParameters<Arr> = HashMap<Rc<str>, Arr>;

/// Type alias for a hashmap of gradients.
pub type KeyedGrad = KeyedParameters<Array>;

macro_rules! keyed_value_and_grad {
    ($inner_ret:ty, $cls_new:ident, $f:ident, $args_ty:ty) => {
        move |parameters: KeyedParameters<Arr>,
              arrays: $args_ty|
              -> Result<(Vec<Array>, KeyedGrad)> {
            let (flattened_keys, flattened_values): (Vec<_>, Vec<_>) =
                parameters.into_iter().unzip();

            let inner = |flattened_arrays: &[Array]| -> $inner_ret {
                let parameters = flattened_keys
                    .iter()
                    .cloned()
                    .zip(flattened_arrays.iter().cloned())
                    .collect();
                ($f)(parameters, arrays.clone())
            };

            let argument_numbers = (0..flattened_values.len() as i32).collect::<Vec<_>>();

            let closure = Closure::$cls_new(inner);
            let cvg = ClosureValueAndGrad::try_from_op(|res| unsafe {
                mlx_sys::mlx_value_and_grad(
                    res,
                    closure.as_ptr(),
                    argument_numbers.as_ptr(),
                    argument_numbers.len(),
                )
            })?;

            let (value, grads) = value_and_gradient(cvg.as_ptr(), flattened_values.into_iter())?;

            let grads_map = flattened_keys.iter().cloned().zip(grads).collect();

            Ok((value, grads_map))
        }
    };
}

/// Similar to [`IntoValueAndGrad`] but for functions that take a hashmap of parameters.
pub trait IntoKeyedValueAndGrad<'a, Arr, Args, Err>
where
    Arr: AsRef<Array>,
    Args: Clone,
{
    /// Convert the function/closure into a closure that computes the value and gradient.
    fn into_keyed_value_and_grad(
        self,
    ) -> impl FnMut(KeyedParameters<Arr>, Args) -> Result<(Vec<Array>, KeyedGrad)> + 'a;
}

impl<'a, F, Arr, Args> IntoKeyedValueAndGrad<'a, Arr, Args, ()> for F
where
    F: FnMut(HashMap<Rc<str>, Array>, Args) -> Vec<Array> + 'a,
    Arr: AsRef<Array>,
    Args: Clone,
{
    fn into_keyed_value_and_grad(
        mut self,
    ) -> impl FnMut(KeyedParameters<Arr>, Args) -> Result<(Vec<Array>, KeyedGrad)> + 'a {
        keyed_value_and_grad!(Vec<Array>, new, self, Args)
    }
}

impl<'a, F, Arr, Args> IntoKeyedValueAndGrad<'a, Arr, Args, Exception> for F
where
    F: FnMut(HashMap<Rc<str>, Array>, Args) -> Result<Vec<Array>> + 'a,
    Arr: AsRef<Array>,
    Args: Clone,
{
    fn into_keyed_value_and_grad(
        mut self,
    ) -> impl FnMut(KeyedParameters<Arr>, Args) -> Result<(Vec<Array>, KeyedGrad)> + 'a {
        keyed_value_and_grad!(Result<Vec<Array>>, new_fallible, self, Args)
    }
}

/// Returns a function which computes the value and gradient of `f` with keyed parameters.
pub fn keyed_value_and_grad<'a, F, Arr, Args, Err>(
    f: F,
) -> impl FnMut(KeyedParameters<Arr>, Args) -> Result<(Vec<Array>, KeyedGrad)> + 'a
where
    F: IntoKeyedValueAndGrad<'a, Arr, Args, Err> + 'a,
    Arr: AsRef<Array>,
    Args: Clone,
{
    f.into_keyed_value_and_grad()
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, rc::Rc};

    use crate::{array, Array};

    use super::*;

    #[test]
    fn test_keyed_value_and_grad() {
        let f = |parameters: HashMap<Rc<str>, Array>, _: i32| -> Vec<Array> {
            vec![&parameters["x"] * &parameters["y"]]
        };

        let x = array!(1.5f32);
        let y = array!(2.0f32);
        let parameters = vec![("x", x), ("y", y)]
            .into_iter()
            .map(|(k, v)| (k.into(), v))
            .collect();

        let mut vg = keyed_value_and_grad(f);

        let (value, grad) = vg(parameters, 0).unwrap();

        assert_eq!(value[0].item_exact::<f32>(), 1.5 * 2.0);
        assert_eq!(grad["x"].item_exact::<f32>(), 2.0);
        assert_eq!(grad["y"].item_exact::<f32>(), 1.5);
    }
}
