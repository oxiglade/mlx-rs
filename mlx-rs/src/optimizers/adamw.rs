use std::convert::Infallible;

use mlx_internal_macros::{generate_builder, Buildable};

use crate::{
    array,
    utils::{get_mut_or_insert_with, Updatable},
    Array,
};

use super::*;

generate_builder! {
    /// The AdamW optimizer [1].
    ///
    /// Following the above convention, in contrast with [1], we do not use bias
    /// correction in the first and second moments for AdamW. We update the weights
    /// with a `weightDecay` lambda value:
    ///
    /// [1]: Loshchilov, I. and Hutter, F., 2019. Decoupled weight decay regularization. ICLR 2019.
    #[derive(Debug, Clone, Buildable)]
    #[buildable(root = crate)]
    #[builder(
        build_with = build_adamw,
        root = crate
    )]
    pub struct AdamW {
        /// The learning rate.
        #[builder(ty_override = f32)]
        pub lr: Array,

        /// The coefficients used for computing running averages of the gradient and its square.
        ///
        /// Default to [`AdamW::DEFAULT_BETAS`].
        #[builder(optional, ty_override = Betas, default = AdamW::DEFAULT_BETAS)]
        pub betas: (Array, Array),

        /// The epsilon added to the denominator to improve numerical stability.
        ///
        /// Default to [`AdamW::DEFAULT_EPS`].
        #[builder(optional, ty_override = f32, default = AdamW::DEFAULT_EPS)]
        pub eps: Array,

        /// The weight decay
        ///
        /// Default to [`AdamW::DEFAULT_WEIGHT_DECAY`].
        #[builder(optional, ty_override = f32, default = AdamW::DEFAULT_WEIGHT_DECAY)]
        pub weight_decay: Array,

        /// Inner state.
        #[builder(ignore)]
        pub state: State<(Array, Array)>,
    }
}

/// Builds a new [`AdamW`] optimizer.
fn build_adamw(builder: AdamWBuilder) -> Result<AdamW, Infallible> {
    let lr = builder.lr;
    let betas = builder.betas;
    let eps = builder.eps;
    let weight_decay = builder.weight_decay;

    Ok(AdamW {
        lr: array!(lr),
        betas: (array!(betas.0), array!(betas.1)),
        eps: array!(eps),
        weight_decay: array!(weight_decay),
        state: State::new(),
    })
}

impl AdamW {
    /// Default value for `betas`.
    pub const DEFAULT_BETAS: (f32, f32) = super::Adam::DEFAULT_BETAS;

    /// Default value for `eps`.
    pub const DEFAULT_EPS: f32 = super::Adam::DEFAULT_EPS;

    /// Default value for `weight_decay`.
    pub const DEFAULT_WEIGHT_DECAY: f32 = 0.01;
}

impl Optimizer for AdamW {
    type State = State<(Array, Array)>;

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }

    fn update_single(
        &mut self,
        key: &std::rc::Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<(), crate::error::Exception> {
        let betas = &self.betas;
        let state = get_mut_or_insert_with(&mut self.state, key, || (array!(0.0), array!(0.0)));

        // SAFETY: These are all single-element arrays and won't panic.
        let one_minus_lr_wd = array!(1.0) - (&self.lr * &self.weight_decay);
        let decayed_parameter = &*parameter * &one_minus_lr_wd;

        let (new_parameter, new_states) = super::adam_apply_single(
            &self.lr,
            betas,
            &self.eps,
            gradient,
            &decayed_parameter,
            state,
        )?;

        *state = new_states;
        *parameter = new_parameter;

        Ok(())
    }
}

impl Updatable for AdamW {
    optimizer_updatable_state_methods!();
}

impl_updatable_for_mut_optimizer!(AdamW);
