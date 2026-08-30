use std::rc::Rc;

use crate::{
    array,
    ops::{sqrt, square},
    Array,
};
use mlx_internal_macros::{generate_builder, Buildable};

use crate::{error::RmsPropBuildError, utils::get_mut_or_insert_with};

use super::*;

generate_builder! {
    /// The RMSprop optimizer [1].
    ///
    /// [1]: Tieleman, T. and Hinton, G. 2012. Lecture 6.5-rmsprop, coursera: Neural networks for
    ///     machine learning
    #[derive(Debug, Clone, Buildable)]
    #[buildable(root = crate)]
    #[builder(
        build_with = build_rmdprop,
        err = RmsPropBuildError,
        root = crate
    )]
    pub struct RmsProp {
        /// Learning rate
        #[builder(ty_override = f32)]
        pub lr: Array,

        /// The smoothing constant. Default to [`RmsProp::DEFAULT_ALPHA`] if not specified.
        #[builder(optional, ty_override = f32, default = RmsProp::DEFAULT_ALPHA)]
        pub alpha: Array,

        /// The epsilon added to the denominator to improve numerical stability. Default to
        /// [`RmsProp::DEFAULT_EPSILON`] if not specified.
        #[builder(optional, ty_override = f32, default = RmsProp::DEFAULT_EPSILON)]
        pub epsilon: Array,

        /// Inner state
        #[builder(ignore)]
        pub state: State,
    }
}

fn build_rmdprop(builder: RmsPropBuilder) -> Result<RmsProp, RmsPropBuildError> {
    let lr = builder.lr;
    let alpha = builder.alpha;
    let epsilon = builder.epsilon;

    if alpha < 0.0 {
        return Err(RmsPropBuildError::NegativeAlpha);
    }

    if epsilon < 0.0 {
        return Err(RmsPropBuildError::NegativeEpsilon);
    }

    Ok(RmsProp {
        lr: array!(lr),
        alpha: array!(alpha),
        epsilon: array!(epsilon),
        state: State::new(),
    })
}

impl RmsProp {
    /// Default alpha if not specified.
    pub const DEFAULT_ALPHA: f32 = 0.99;

    /// Default epsilon if not specified.
    pub const DEFAULT_EPSILON: f32 = 1e-8;
}

impl Optimizer for RmsProp {
    type State = State;

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }

    fn update_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> crate::error::Result<()> {
        let state = get_mut_or_insert_with(&mut self.state, key, || array!(0.0));

        let lr = &self.lr;
        let alpha = &self.alpha;
        let eps = &self.epsilon;

        let one_minus_alpha = array!(1.0).subtract(alpha)?;
        let first_term = alpha.multiply(&*state)?;
        let second_term = one_minus_alpha.multiply(square(gradient)?)?;
        let v = first_term.add(&second_term)?;

        let num = lr.multiply(gradient)?;
        let den = sqrt(&v)?.add(eps)?;
        let new_param = parameter.subtract(num.divide(&den)?)?;

        *parameter = new_param;
        *state = v;

        Ok(())
    }
}

impl Updatable for RmsProp {
    optimizer_updatable_state_methods!();
}

impl_updatable_for_mut_optimizer!(RmsProp);
