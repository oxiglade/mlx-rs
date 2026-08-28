use crate::module::{update_parameters, ModuleParameters};
use crate::transforms::keyed_value_and_grad;
use crate::{error::Exception, Array};

use crate::module::FlattenedModuleParam;

fn trainable_params(model: &impl ModuleParameters) -> FlattenedModuleParam {
    model
        .trainable_parameters()
        .flatten()
        .into_iter()
        .map(|(k, v)| (k, v.clone()))
        .collect()
}

/// Helper trait for [`value_and_grad`]
pub trait IntoModuleValueAndGrad<'a, M, Args, Val, Err>
where
    M: ModuleParameters + 'a,
    Args: Clone,
{
    /// Computes the valud and gradient of the passed function `f(model, args)` with regard to the
    /// model's trainable parameters.
    fn into_module_value_and_grad(
        self,
    ) -> impl FnMut(&mut M, Args) -> Result<(Val, FlattenedModuleParam), Exception> + 'a;
}

impl<'a, F, M, Args> IntoModuleValueAndGrad<'a, M, Args, Vec<Array>, ()> for F
where
    M: ModuleParameters + 'a,
    F: FnMut(&mut M, Args) -> Vec<Array> + 'a,
    Args: Clone,
{
    fn into_module_value_and_grad(
        mut self,
    ) -> impl FnMut(&mut M, Args) -> Result<(Vec<Array>, FlattenedModuleParam), Exception> + 'a
    {
        move |model, arrays| {
            let trainable_parameters = trainable_params(model);
            let inner = |parameters: FlattenedModuleParam, arrays: Args| -> Vec<Array> {
                let flattened_parameters = parameters.into_iter();
                update_parameters(model, flattened_parameters);

                self(model, arrays)
            };
            let mut vg = keyed_value_and_grad(inner);

            let (v, g) = vg(trainable_parameters, arrays)?;
            Ok((v, g))
        }
    }
}

impl<'a, F, M, Args> IntoModuleValueAndGrad<'a, M, Args, Vec<Array>, Exception> for F
where
    M: ModuleParameters + 'a,
    F: FnMut(&mut M, Args) -> Result<Vec<Array>, Exception> + 'a,
    Args: Clone,
{
    fn into_module_value_and_grad(
        mut self,
    ) -> impl FnMut(&mut M, Args) -> Result<(Vec<Array>, FlattenedModuleParam), Exception> + 'a
    {
        move |model, arrays| {
            let trainable_parameters = trainable_params(model);
            let inner =
                |parameters: FlattenedModuleParam, arrays: Args| -> Result<Vec<Array>, Exception> {
                    let flattened_parameters = parameters.into_iter().map(|(k, v)| (k, v.clone()));
                    update_parameters(model, flattened_parameters);

                    self(model, arrays)
                };
            let mut vg = keyed_value_and_grad(inner);

            let (v, g) = vg(trainable_parameters, arrays)?;
            Ok((v, g))
        }
    }
}

impl<'a, F, M, Args> IntoModuleValueAndGrad<'a, M, Args, Array, ()> for F
where
    M: ModuleParameters + 'a,
    F: FnMut(&mut M, Args) -> Array + 'a,
    Args: Clone,
{
    fn into_module_value_and_grad(
        mut self,
    ) -> impl FnMut(&mut M, Args) -> Result<(Array, FlattenedModuleParam), Exception> + 'a {
        move |model, arrays| {
            let trainable_parameters = trainable_params(model);
            let inner = |parameters: FlattenedModuleParam, arrays: Args| -> Vec<Array> {
                let flattened_parameters = parameters.into_iter().map(|(k, v)| (k, v.clone()));
                update_parameters(model, flattened_parameters);

                vec![self(model, arrays)]
            };
            let mut vg = keyed_value_and_grad(inner);

            let (v, g) = vg(trainable_parameters, arrays)?;
            let v = v.into_iter().next().expect("Expected a single value");
            Ok((v, g))
        }
    }
}

impl<'a, F, M, Args> IntoModuleValueAndGrad<'a, M, Args, Array, Exception> for F
where
    M: ModuleParameters + 'a,
    F: FnMut(&mut M, Args) -> Result<Array, Exception> + 'a,
    Args: Clone,
{
    fn into_module_value_and_grad(
        mut self,
    ) -> impl FnMut(&mut M, Args) -> Result<(Array, FlattenedModuleParam), Exception> + 'a {
        move |model, arrays| {
            let trainable_parameters = trainable_params(model);
            let inner =
                |parameters: FlattenedModuleParam, arrays: Args| -> Result<Vec<Array>, Exception> {
                    let flattened_parameters = parameters.into_iter().map(|(k, v)| (k, v.clone()));
                    update_parameters(model, flattened_parameters);

                    self(model, arrays).map(|v| vec![v])
                };
            let mut vg = keyed_value_and_grad(inner);

            let (v, g) = vg(trainable_parameters, arrays)?;
            let v = v.into_iter().next().expect("Expected a single value");
            Ok((v, g))
        }
    }
}

/// Transform the passed function `f(model, args)` to a function that computes the gradients of `f`
/// with regard to the model's trainable parameters and also its value.
pub fn value_and_grad<'a, F, M, Args, Val, Err>(
    f: F,
) -> impl FnMut(&mut M, Args) -> Result<(Val, FlattenedModuleParam), Exception> + 'a
where
    M: ModuleParameters + 'a,
    F: IntoModuleValueAndGrad<'a, M, Args, Val, Err>,
    Args: Clone,
{
    f.into_module_value_and_grad()
}

#[cfg(test)]
mod tests {
    use crate::module::Module;
    use crate::{error::Exception, Array, Dtype};

    use crate::nn::{self, Linear};

    fn assert_finite_nonzero(value: impl AsRef<Array>) {
        let value = value.as_ref();
        assert_eq!(value.dtype(), Dtype::Float32);
        assert!(value.shape().is_empty());
        let value = value.item::<f32>();
        assert!(value.is_finite());
        assert_ne!(value, 0.0);
    }

    // The unit test below is adapted from `test_compiled_optimizer` in
    // `mlx/python/tests/test_optimizers.py``
    #[test]
    fn test_value_and_grad() {
        let mut model = Linear::new(2, 2).unwrap();
        let x = crate::random::uniform::<_, f32>(1.0, 2.0, &[2, 2], None).unwrap();

        let loss = |model: &mut Linear, x: &Array| -> Vec<Array> {
            vec![model.forward(x).unwrap().sum(None).unwrap()]
        };

        let mut vg = nn::value_and_grad(loss);
        let (v, g) = vg(&mut model, &x).unwrap();

        assert_finite_nonzero(v[0].sum(None).unwrap());
        assert_finite_nonzero(g["weight"].sum(None).unwrap());
        assert_finite_nonzero(g["bias"].sum(None).unwrap());
    }

    #[test]
    fn test_value_and_grad_with_unary_output() {
        let mut model = Linear::new(2, 2).unwrap();
        let x = crate::random::uniform::<_, f32>(1.0, 2.0, &[2, 2], None).unwrap();

        let loss = |model: &mut Linear, x: &Array| -> Array {
            model.forward(x).unwrap().sum(None).unwrap()
        };

        let mut vg = nn::value_and_grad(loss);
        let (v, g) = vg(&mut model, &x).unwrap();

        assert_finite_nonzero(v.sum(None).unwrap());
        assert_finite_nonzero(g["weight"].sum(None).unwrap());
        assert_finite_nonzero(g["bias"].sum(None).unwrap());
    }

    #[test]
    fn test_fallible_module_value_and_grad() {
        let mut model = Linear::new(2, 2).unwrap();
        let x = crate::random::uniform::<_, f32>(1.0, 2.0, &[2, 2], None).unwrap();

        let loss = |model: &mut Linear, x: &Array| -> Result<Vec<Array>, Exception> {
            Ok(vec![model.forward(x)?.sum(None)?])
        };

        let mut vg = nn::value_and_grad(loss);
        let (v, g) = vg(&mut model, &x).unwrap();

        assert_finite_nonzero(v[0].sum(None).unwrap());
        assert_finite_nonzero(g["weight"].sum(None).unwrap());
        assert_finite_nonzero(g["bias"].sum(None).unwrap());
    }

    #[test]
    fn test_value_and_grad_with_two_args() {
        let mut model = Linear::new(2, 2).unwrap();
        let x = crate::random::uniform::<_, f32>(1.0, 2.0, &[2, 2], None).unwrap();
        let y = crate::ops::ones::<f32>(x.shape()).unwrap();

        let loss =
            |model: &mut Linear, (x, y): (&Array, &Array)| -> Result<Vec<Array>, Exception> {
                model
                    .forward(x)?
                    .subtract(y)?
                    .square()?
                    .sum(None)
                    .map(|v| vec![v])
            };

        let mut vg = nn::value_and_grad(loss);
        let (v, g) = vg(&mut model, (&x, &y)).unwrap();

        assert_finite_nonzero(v[0].sum(None).unwrap());
        assert_finite_nonzero(g["weight"].sum(None).unwrap());
        assert_finite_nonzero(g["bias"].sum(None).unwrap());
    }

    #[test]
    fn test_value_and_grad_with_error() {
        let mut model = Linear::new(2, 2).unwrap();
        // Use a shape that is not compatible with the model
        let x = crate::random::uniform::<_, f32>(1.0, 2.0, &[3, 3], None).unwrap();

        let loss = |model: &mut Linear, x: &Array| -> Result<Vec<Array>, Exception> {
            Ok(vec![model.forward(x)?.sum(None)?])
        };

        let mut vg = nn::value_and_grad(loss);
        let result = vg(&mut model, &x);

        assert!(result.is_err());

        // Check that the error message is not just "mlx_closure returned a non-zero value"
        let err = result.unwrap_err();
        assert!(!err.what().contains("non-zero value"))
    }
}
