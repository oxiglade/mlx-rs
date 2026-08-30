//! Trait and implementations for optimizers.

#![deny(missing_docs)]

use std::{
    borrow::{Borrow, Cow},
    collections::HashMap,
    path::Path,
    rc::Rc,
};

use crate::{
    array,
    error::{IoError, StateProjectionError, UnflattenError},
    module::{FlattenedModuleParam, ModuleParameters},
    utils::{StateProjection, Updatable},
    Array,
};

mod adadelta;
mod adafactor;
mod adagrad;
mod adam;
mod adamax;
mod adamw;
mod lion;
mod rmsprop;
mod sgd;

pub use adadelta::*;
pub use adafactor::*;
pub use adagrad::*;
pub use adam::*;
pub use adamax::*;
pub use adamw::*;
use itertools::Itertools;
pub use lion::*;
pub use rmsprop::*;
pub use sgd::*;

// Unfortunate workaround to implement Updatable for mutable references of
// optimizers This is needed because of the orphan rule and lack of negative
// trait bound, otherwise we would need to implement Updatable for every
// `Module`
macro_rules! impl_updatable_for_mut_optimizer {
    ($optimizer:ty) => {
        impl Updatable for &'_ mut $optimizer {
            fn state_projection(&mut self) -> Result<StateProjection<'_>, StateProjectionError> {
                <$optimizer as Updatable>::state_projection(&mut **self)
            }
        }
    };
}
use impl_updatable_for_mut_optimizer;

macro_rules! optimizer_updatable_state_methods {
    () => {
        fn state_projection(&mut self) -> Result<StateProjection<'_>, StateProjectionError> {
            self.state_mut().state_projection()
        }
    };
}
use optimizer_updatable_state_methods;

/// Type alias for common optimizer state.
pub type State<T = Array> = HashMap<Rc<str>, T>;

/// Trait for optimizer states.
pub trait OptimizerState: Sized {
    /// Error type for unflatten.
    type UnflattenError: std::error::Error + Into<IoError>;

    /// Declare all required and optional optimizer-state slots.
    fn state_projection(&mut self) -> Result<StateProjection<'_>, StateProjectionError>;

    /// Flatten present state entries in stable key order.
    fn flatten(&mut self) -> Result<Vec<(Rc<str>, &Array)>, StateProjectionError> {
        Ok(self.state_projection()?.into_entries().collect())
    }

    /// Flatten mutable present state entries in stable key order.
    fn flatten_mut(&mut self) -> Result<Vec<(Rc<str>, &mut Array)>, StateProjectionError> {
        Ok(self.state_projection()?.into_entries_mut().collect())
    }

    /// Unflatten an iterator of key-value pairs into the optimizer state.
    fn unflatten<I, K>(input: I) -> Result<Self, Self::UnflattenError>
    where
        I: IntoIterator<Item = (K, Array)>,
        K: Ord + AsRef<str> + Into<Rc<str>>;

    /// Save the optimizer state to a safetensors file.
    fn save_safetensors(&mut self, path: impl AsRef<Path>) -> Result<(), IoError> {
        let state = self.flatten().map_err(IoError::StateProjection)?;
        Array::save_safetensors(state, None, path)
    }

    /// Load the optimizer state from a safetensors file.
    fn load_safetensors(&mut self, path: impl AsRef<Path>) -> Result<(), IoError> {
        let loaded = Array::load_safetensors(path)?;
        let unflattened = Self::unflatten(loaded).map_err(Into::into)?;

        *self = unflattened;

        Ok(())
    }
}

impl OptimizerState for State {
    type UnflattenError = std::convert::Infallible;

    fn state_projection(&mut self) -> Result<StateProjection<'_>, StateProjectionError> {
        let mut projection = StateProjection::new();
        for (key, value) in self {
            projection.required(key.clone(), value)?;
        }
        Ok(projection)
    }

    fn unflatten<I, K>(input: I) -> Result<Self, Self::UnflattenError>
    where
        Self: Sized,
        I: IntoIterator<Item = (K, Array)>,
        K: Ord + AsRef<str> + Into<Rc<str>>,
    {
        Ok(input.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }
}

impl OptimizerState for State<(Array, Array)> {
    type UnflattenError = UnflattenError;

    fn state_projection(&mut self) -> Result<StateProjection<'_>, StateProjectionError> {
        let mut projection = StateProjection::new();
        for (key, (first, second)) in self {
            projection.required(format!("{key}.0"), first)?;
            projection.required(format!("{key}.1"), second)?;
        }
        Ok(projection)
    }

    fn unflatten<I, K>(input: I) -> Result<Self, Self::UnflattenError>
    where
        Self: Sized,
        I: IntoIterator<Item = (K, Array)>,
        K: Ord + AsRef<str> + Into<Rc<str>>,
    {
        let mut state = State::new();
        let iter = input
            .into_iter()
            .sorted_by(|a, b| a.0.as_ref().cmp(b.0.as_ref()))
            .chunks(2);

        for mut chunk in &iter {
            let first = chunk.next().ok_or(UnflattenError::ExpectingNextPair)?;
            let second = chunk.next().ok_or(UnflattenError::ExpectingNextPair)?;

            // Check if the keys match up to the last dot and the suffix is 0 and 1 (should be already sorted)
            let first_key = first.0.as_ref();
            let second_key = second.0.as_ref();
            if !first_key.ends_with(".0") || !second_key.ends_with(".1") {
                return Err(UnflattenError::InvalidKey);
            }
            if first_key[..first_key.len() - 2] != second_key[..second_key.len() - 2] {
                return Err(UnflattenError::InvalidKey);
            }

            let key = &first_key[..first_key.len() - 2];
            let key: Rc<str> = Rc::from(key);
            state.insert(key, (first.1, second.1));
        }
        Ok(state)
    }
}

/// Trait for optimizers.
pub trait Optimizer: Updatable {
    /// State of the optimizer.
    type State: OptimizerState;

    /// Get the state of the optimizer.
    fn state(&self) -> &Self::State;

    /// Get the mutable state of the optimizer.
    fn state_mut(&mut self) -> &mut Self::State;

    /// Update a single parameter with the given gradient.
    ///
    /// The implementation should look up the state for the parameter using the key and update the
    /// state and the parameter accordingly. The key is provided instead of the state because it
    /// would otherwise create a mutable borrow conflict with the rest of the optimizer fields.
    fn update_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> crate::error::Result<()>;

    /// Apply the gradients to the parameters of the model and update the model with the new
    /// parameters.
    fn update<M>(
        &mut self,
        model: &mut M,
        gradients: impl Borrow<FlattenedModuleParam>,
    ) -> crate::error::Result<()>
    where
        M: ModuleParameters,
    {
        let mut parameters = model.parameters_mut().flatten();

        for (key, gradient) in gradients.borrow().iter() {
            if let Some(parameter) = parameters.get_mut(key) {
                self.update_single(key, gradient, parameter)?;
            }
        }

        Ok(())
    }
}

/// Type alias for clipped gradients that is returned by `clip_grad_norm`.
pub type MaybeClippedGrads<'a> = HashMap<Rc<str>, Cow<'a, Array>>;

/// Clips the global norm of the gradients
///
/// This function ensures that the global norm of the gradients does not exceed
/// `max_norm`. It scales down the gradients proportionally if their norm is
/// greater than `max_norm`.
pub fn clip_grad_norm(
    gradients: &FlattenedModuleParam,
    max_norm: f32,
) -> crate::error::Result<(MaybeClippedGrads<'_>, f32)> {
    let total_norm: f32 = gradients
        .values()
        .try_fold(array!(0.0), |acc, grad| acc.add(&grad.square()?.sum(None)?))?
        .sqrt()?
        .item_exact();
    let normalizer = array!(max_norm / (total_norm + 1e-6));

    let clipped_gradients: HashMap<_, _> = gradients
        .iter()
        .map(|(key, grad)| {
            let clipped_grad = if total_norm < max_norm {
                Cow::Borrowed(grad)
            } else {
                Cow::Owned(grad * &normalizer)
            };
            (key.clone(), clipped_grad)
        })
        .collect();
    Ok((clipped_gradients, total_norm))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        array,
        module::FlattenedModuleParam,
        test_utils::{assert_array_eq, tolerances},
        Array,
    };

    use super::clip_grad_norm;

    #[test]
    fn test_clip_grad_norm() {
        // Test with small gradients that do not require clipping
        let mut small_grads: FlattenedModuleParam = HashMap::new();
        small_grads.insert("first.a".into(), array!([0.1, 0.2]));
        small_grads.insert("first.b".into(), array!(0.1));
        small_grads.insert("second".into(), array!(0.3));

        let max_norm = 10.0;

        let (clipped_grads, _) = clip_grad_norm(&small_grads, max_norm).unwrap();
        for (key, value) in small_grads.iter() {
            assert_array_eq(
                &*clipped_grads[key],
                value,
                tolerances::EXACT.rtol,
                tolerances::EXACT.atol,
            );
        }

        // Test with large gradients that require clipping
        let mut large_grads: FlattenedModuleParam = HashMap::new();
        large_grads.insert("first.a".into(), array!([10.0, 20.0]));
        large_grads.insert("first.b".into(), array!(10.0));
        large_grads.insert("second".into(), array!(30.0));

        let max_norm = 1.0;

        let (clipped_grads, total_norm) = clip_grad_norm(&large_grads, max_norm).unwrap();
        let clipped_values: Vec<_> = clipped_grads.values().map(|v| v.as_ref()).collect();
        let norm_of_clipped = clipped_values
            .into_iter()
            .map(|g| g.square().unwrap().sum(None).unwrap())
            .sum::<Array>()
            .sqrt()
            .unwrap();

        float_eq::assert_float_eq!(norm_of_clipped.item_exact::<f32>(), max_norm, abs <= 1e-6);

        // Ensures that the scaling was done correctly
        let scale = max_norm / total_norm;
        let expected_grads: FlattenedModuleParam = large_grads
            .iter()
            .map(|(key, value)| (key.clone(), value * scale))
            .collect();
        for (key, value) in expected_grads.iter() {
            assert_array_eq(
                &*clipped_grads[key],
                value,
                tolerances::EXACT.rtol,
                tolerances::EXACT.atol,
            );
        }
    }
}
