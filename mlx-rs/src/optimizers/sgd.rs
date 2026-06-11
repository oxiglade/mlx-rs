use std::{borrow::Cow, rc::Rc};

use crate::{array, utils::get_mut_or_insert_with, Array};
use mlx_internal_macros::{generate_builder, Buildable};

use super::*;

generate_builder! {
    /// Stochastic gradient descent optimizer.
    #[derive(Debug, Clone, Buildable)]
    #[buildable(root = crate)]
    #[builder(
        build_with = build_sgd,
        root = crate
    )]
    pub struct Sgd {
        /// Learning rate
        pub lr: f32,

        /// Momentum strength. Default to [`Sgd::DEFAULT_MOMENTUM`] if not specified.
        #[builder(optional, default = Sgd::DEFAULT_MOMENTUM)]
        pub momentum: f32,

        /// Weight decay (L2 penalty). Default to [`Sgd::DEFAULT_WEIGHT_DECAY`] if not specified.
        #[builder(optional, default = Sgd::DEFAULT_WEIGHT_DECAY)]
        pub weight_decay: f32,

        /// Dampening for momentum. Default to [`Sgd::DEFAULT_DAMPENING`] if not specified.
        #[builder(optional, default = Sgd::DEFAULT_DAMPENING)]
        pub dampening: f32,

        /// Enables nesterov momentum. Default to [`Sgd::DEFAULT_NESTEROV`] if not specified.
        #[builder(optional, ty_override = bool, default = Sgd::DEFAULT_NESTEROV)]
        pub nesterov: bool,

        /// Inner state
        #[builder(ignore)]
        pub state: State,
    }
}

#[allow(
    clippy::unnecessary_wraps,
    reason = "Builder trait requires Result<Self, Self::Error>"
)]
fn build_sgd(builder: SgdBuilder) -> Result<Sgd, std::convert::Infallible> {
    let lr = builder.lr;
    let momentum = builder.momentum;
    let weight_decay = builder.weight_decay;
    let dampening = builder.dampening;
    let nesterov = builder.nesterov;

    Ok(Sgd {
        lr,
        momentum,
        weight_decay,
        dampening,
        nesterov,
        state: State::new(),
    })
}

impl Sgd {
    /// Default momentum if not specified.
    pub const DEFAULT_MOMENTUM: f32 = 0.0;

    /// Default weight decay if not specified.
    pub const DEFAULT_WEIGHT_DECAY: f32 = 0.0;

    /// Default dampening if not specified.
    pub const DEFAULT_DAMPENING: f32 = 0.0;

    /// Default nesterov if not specified.
    pub const DEFAULT_NESTEROV: bool = false;
}

impl Optimizer for Sgd {
    type State = State;

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }

    /// Apply SGD to a single parameter. Returns the updated parameter and the updated state.
    #[inline]
    fn update_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> crate::error::Result<()> {
        let state = get_mut_or_insert_with(&mut self.state, key, || array!(0.0));
        let mut gradient = Cow::Borrowed(gradient);

        if self.weight_decay != 0.0 {
            let weight_decay = array!(self.weight_decay);
            gradient = Cow::Owned(weight_decay.multiply(&*parameter)?.add(&*gradient)?);
        }

        if self.momentum <= 0.0 {
            let lr = array!(self.lr);
            *parameter = parameter.subtract(lr.multiply(gradient)?)?;
            return Ok(());
        }

        let mut v = &*state * self.momentum;

        if self.dampening > 0.0 {
            let dampening = array!(self.dampening);
            let one_minus_dampening = array!(1.0).subtract(dampening)?;
            v = v.add(&one_minus_dampening.multiply(&gradient)?)?;
        } else {
            v = v.add(&gradient)?;
        }

        if self.nesterov {
            let momentum = array!(self.momentum);
            let lr = array!(self.lr);
            let update = gradient.add(momentum.multiply(&v)?)?;
            *parameter = parameter.subtract(lr.multiply(&update)?)?;
            *state = v;
        } else {
            let update = &v;
            let lr = array!(self.lr);
            *parameter = parameter.subtract(lr.multiply(update)?)?;
            *state = v;
        }

        Ok(())
    }
}

impl Updatable for Sgd {
    fn updatable_states_len(&self) -> usize {
        self.state.len()
    }

    fn updatable_states(&self) -> impl IntoIterator<Item = &Array> {
        use itertools::Itertools;

        self.state
            .iter()
            .sorted_by(|a, b| a.0.cmp(b.0))
            .map(|(_, v)| v)
    }

    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array> {
        use itertools::Itertools;

        self.state
            .iter_mut()
            .sorted_by(|a, b| a.0.cmp(b.0))
            .map(|(_, v)| v)
    }
}

impl_updatable_for_mut_optimizer!(Sgd);
