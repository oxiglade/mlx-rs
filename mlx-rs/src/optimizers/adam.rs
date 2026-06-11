use std::convert::Infallible;

use mlx_internal_macros::{generate_builder, Buildable};

use crate::{array, utils::get_mut_or_insert_with};

use super::*;

/// `(f32, f32O)`. Type alias for betas in the Adam/AdamW/Adamax optimizer builders due to
/// limitation in the `generate_builder` macro
pub type Betas = (f32, f32); // The macro right now can't handle raw tuple types

generate_builder! {
    /// The Adam optimizer.
    ///
    /// Please refer to the original paper for more details:
    ///
    /// [1]: Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic optimization. ICLR 2015.
    #[derive(Debug, Clone, Buildable)]
    #[buildable(root = crate)]
    #[builder(
        build_with = build_adam,
        root = crate
    )]
    pub struct Adam {
        /// The learning rate
        #[builder(ty_override = f32)]
        pub lr: Array,

        /// The coefficients used for computing running averages of the gradient and its square
        ///
        /// Default to [`Adam::DEFAULT_BETAS`]
        #[builder(optional, ty_override = Betas, default = Adam::DEFAULT_BETAS)]
        pub betas: (Array, Array),

        /// The epsilon added to the denominator to improve numerical stability
        ///
        /// Default to [`Adam::DEFAULT_EPS`]
        #[builder(optional, ty_override = f32, default = Adam::DEFAULT_EPS)]
        pub eps: Array,

        /// Inner state
        #[builder(ignore)]
        pub state: State<(Array, Array)>,
    }
}

/// Builds a new [`Adam`].
#[allow(
    clippy::unnecessary_wraps,
    reason = "Builder trait requires Result<Self, Self::Error>"
)]
fn build_adam(builder: AdamBuilder) -> Result<Adam, Infallible> {
    let lr = array!(builder.lr);
    let betas = builder.betas;
    let eps = array!(builder.eps);

    Ok(Adam {
        lr,
        betas: (array!(betas.0), array!(betas.1)),
        eps,
        state: State::new(),
    })
}

impl Adam {
    /// Default values for `betas`
    pub const DEFAULT_BETAS: (f32, f32) = (0.9, 0.999);

    /// Default value for `eps`
    pub const DEFAULT_EPS: f32 = 1e-8;
}

impl Optimizer for Adam {
    type State = State<(Array, Array)>;

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
        let betas = &self.betas;
        let state = get_mut_or_insert_with(&mut self.state, key, || (array!(0.0), array!(0.0)));

        let (new_parameter, new_state) =
            adam_apply_single(&self.lr, betas, &self.eps, gradient, parameter, state)?;

        *state = new_state;
        *parameter = new_parameter;

        Ok(())
    }
}

// Returns (new_parameter, (new_m, new_v))
pub(super) fn adam_apply_single(
    lr: &Array,
    betas: &(Array, Array),
    eps: &Array,
    gradient: &Array,
    parameter: &Array,
    state: &(Array, Array),
) -> crate::error::Result<(Array, (Array, Array))> {
    let (b1, b2) = betas;
    let (m, v) = state;

    let one_minus_b1 = array!(1.0).subtract(b1)?;
    let one_minus_b2 = array!(1.0).subtract(b2)?;

    let new_m = b1.multiply(m)?.add(&one_minus_b1.multiply(gradient)?)?;
    let new_v = b2
        .multiply(v)?
        .add(&one_minus_b2.multiply(gradient.square()?)?)?;

    let new_parameter =
        parameter.subtract(&lr.multiply(&new_m.divide(&new_v.sqrt()?.add(eps)?)?)?)?;

    Ok((new_parameter, (new_m, new_v)))
}

impl Updatable for Adam {
    fn updatable_states_len(&self) -> usize {
        self.state.len() * 2
    }

    fn updatable_states(&self) -> impl IntoIterator<Item = &Array> {
        use itertools::Itertools;

        self.state
            .iter()
            .sorted_by(|a, b| a.0.cmp(b.0))
            .flat_map(|(_, (v, u))| vec![v, u])
    }

    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array> {
        use itertools::Itertools;

        self.state
            .iter_mut()
            .sorted_by(|a, b| a.0.cmp(b.0))
            .flat_map(|(_, (v, u))| vec![v, u])
    }
}

impl_updatable_for_mut_optimizer!(Adam);
