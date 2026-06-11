use std::{convert::Infallible, rc::Rc};

use crate::{array, ops::square, utils::Updatable, Array};
use mlx_internal_macros::{generate_builder, Buildable};

use crate::utils::get_mut_or_insert_with;

use super::*;

generate_builder! {
    /// The Adagrad optimizer.
    ///
    /// Please refer to the original paper for more details:
    ///
    /// [1]: Duchi, J., Hazan, E. and Singer, Y., 2011. Adaptive subgradient methods for online
    ///     learning and stochastic optimization. JMLR 2011.
    #[derive(Debug, Clone, Buildable)]
    #[buildable(root = crate)]
    #[builder(
        build_with = build_adagrad,
        root = crate
    )]
    pub struct AdaGrad {
        /// Learning rate
        #[builder(ty_override = f32)]
        pub lr: Array,

        /// The epsilon added to the denominator to improve numerical stability. Default to
        /// [`AdaGrad::DEFAULT_EPS`].
        #[builder(optional, ty_override = f32, default = AdaGrad::DEFAULT_EPS)]
        pub eps: Array,

        /// Inner state
        #[builder(ignore)]
        pub state: State,
    }
}

/// Builds a new [`AdaGrad`].
#[allow(
    clippy::unnecessary_wraps,
    reason = "Builder trait requires Result<Self, Self::Error>"
)]
fn build_adagrad(builder: AdaGradBuilder) -> Result<AdaGrad, Infallible> {
    let eps = array!(builder.eps);

    Ok(AdaGrad {
        lr: array!(builder.lr),
        eps,
        state: State::new(),
    })
}

impl AdaGrad {
    /// Default value for `eps`.
    pub const DEFAULT_EPS: f32 = 1e-8;
}

impl Optimizer for AdaGrad {
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

        let v = state.add(square(gradient)?)?;

        let num = self.lr.multiply(gradient)?;
        let den = v.sqrt()?.add(&self.eps)?;
        let new_param = parameter.subtract(num.divide(&den)?)?;

        *state = v;
        *parameter = new_param;

        Ok(())
    }
}

impl Updatable for AdaGrad {
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

impl_updatable_for_mut_optimizer!(AdaGrad);
