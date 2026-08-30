use std::borrow::Cow;

use crate::module::{Module, UnaryModule};
use crate::{error::Exception, Array};
use mlx_macros::ModuleParameters;

/// Marker trait for items that can be used in a `Sequential` module.
///
/// It is implemented for all types that implement [`Module`] and [`std::fmt::Debug`].
pub trait SequentialModuleItem: UnaryModule + std::fmt::Debug {}

impl<T> SequentialModuleItem for T where T: UnaryModule + std::fmt::Debug {}

/// A sequential layer.
///
/// It calls each layer in sequence.
#[derive(Debug, ModuleParameters)]
#[module(root = crate)]
pub struct Sequential<Err = Exception> {
    /// The layers to be called in sequence.
    #[param]
    pub layers: Vec<Box<dyn SequentialModuleItem<Error = Err>>>,
}

impl Module<&Array> for Sequential {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let mut x = Cow::Borrowed(x);

        for layer in &mut self.layers {
            x = Cow::Owned(layer.forward(x.as_ref())?);
        }

        match x {
            Cow::Owned(array) => Ok(array),
            Cow::Borrowed(array) => Ok(array.clone()),
        }
    }

    fn training_mode(&mut self, mode: bool) {
        self.layers
            .iter_mut()
            .for_each(|layer| layer.training_mode(mode));
    }
}

impl<Err> Default for Sequential<Err> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Err> Sequential<Err> {
    /// Creates a new [`Sequential`] module.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Appends a layer to the sequential module.
    pub fn append<M>(mut self, layer: M) -> Self
    where
        M: UnaryModule<Error = Err> + std::fmt::Debug + 'static,
    {
        self.layers.push(Box::new(layer));
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        array,
        builder::Builder,
        module::ModuleParameters,
        nn::{self, Linear},
        ops::zeros,
        optimizers::{Optimizer, Sgd},
        random::uniform,
        transforms::{eval, eval_params},
    };

    use crate::losses::{LossReduction, MseLossBuilder};

    use super::*;

    #[test]
    fn test_sequential_linear_param_len() {
        let model = Sequential::new()
            .append(Linear::new(2, 3).unwrap())
            .append(Linear::new(3, 1).unwrap());

        let params = model.parameters().flatten();
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_sequential_linear_param_update() {
        let mut model = Sequential::new()
            .append(Linear::new(2, 3).unwrap())
            .append(Linear::new(3, 1).unwrap());

        model
            .trainable_parameters()
            .flatten()
            .iter()
            .for_each(|(key, value)| {
                println!("{key}: {value:?}");
            });

        let mut params = model.parameters_mut().flatten();

        // Check that the initial weights are not all zeros
        assert!(
            params["layers.0.weight"]
                .abs()
                .unwrap()
                .sum(None)
                .unwrap()
                .item_exact::<f32>()
                - 0.0
                > 1e-6
        );

        // Update the weight with zeros
        let shape = params["layers.0.weight"].shape();
        let zeros = zeros::<f32>(shape).unwrap();
        let value_mut = params.get_mut("layers.0.weight").unwrap();
        **value_mut = zeros;

        // Check that the weight is now all zeros
        let first_layer = &model.layers[0];
        let linear_params = first_layer.parameters().flatten();
        let weight = linear_params["weight"];
        assert!(weight.abs().unwrap().sum(None).unwrap().item_exact::<f32>() - 0.0 < 1e-6);
    }

    #[test]
    fn test_sgd_update_sequential_linear_params() {
        let lr = 1e-2;
        let input_dim = 2;
        let hidden_dim = 3;
        let output_dim = 2;

        // Test using a simple linear equation
        let m = array!(0.25);
        let b = array!(0.75);

        let mut model = Sequential::new()
            .append(Linear::new(input_dim, hidden_dim).unwrap())
            .append(Linear::new(hidden_dim, output_dim).unwrap());

        let loss = MseLossBuilder::new()
            .reduction(LossReduction::Mean)
            .build()
            .unwrap();
        let loss_fn = |model: &mut Sequential, (x, y): (&Array, &Array)| {
            let y_pred = model.forward(x)?;
            loss.apply(&y_pred, y)
        };

        let mut lg = nn::value_and_grad(loss_fn);

        let mut optimizer = Sgd::new(lr);

        let mut losses = vec![];
        for _ in 0..100 {
            // Generate random data
            let x = uniform::<_, f32>(-5.0, 5.0, &[input_dim], None).unwrap();
            let y = &m * &x + &b;

            eval([&x, &y]).unwrap();

            // Compute the loss and gradients and update the model
            let (loss, grads) = lg(&mut model, (&x, &y)).unwrap();
            optimizer.update(&mut model, grads).unwrap();

            eval_params(model.parameters()).unwrap();

            losses.push(loss.item_exact::<f32>());
        }

        // Check that it converges
        assert!(
            losses[0] > losses[losses.len() - 1],
            "Not converging loss: {losses:?}"
        );
    }
}
