use std::rc::Rc;

use crate::{
    array,
    ops::sqrt,
    utils::{get_mut_or_insert_with, Updatable},
    Array,
};
use mlx_internal_macros::{generate_builder, Buildable};

use crate::error::AdaDeltaBuildError;

use super::*;

generate_builder! {
    /// The AdaDelta optimizer with a learning rate.
    ///
    /// The default `rho` is `0.9`, matching Python MLX. Earlier mlx-rs releases used `0.99`.
    ///
    /// Please refer to the original paper for more details:
    ///
    /// [1]: Zeiler, M.D., 2012. ADADELTA: an adaptive learning rate method. arXiv preprint arXiv:1212.5701.
    #[derive(Debug, Clone, Buildable)]
    #[buildable(root = crate)]
    #[builder(
        build_with = build_adadelta,
        err = AdaDeltaBuildError,
        root = crate
    )]
    pub struct AdaDelta {
        /// The learning rate
        #[builder(ty_override = f32)]
        pub lr: Array,

        /// The coefficient used for computing a running average of squared gradients. Defaults to
        /// [`AdaDelta::DEFAULT_RHO`] (`0.9`, matching Python MLX).
        #[builder(optional, ty_override = f32, default = AdaDelta::DEFAULT_RHO)]
        pub rho: Array,

        /// The epsilon added to the denominator to improve numerical stability. Default to
        /// [`AdaDelta::DEFAULT_EPS`].
        #[builder(optional, ty_override = f32, default = AdaDelta::DEFAULT_EPS)]
        pub eps: Array,

        /// Inner state
        #[builder(ignore)]
        pub state: State<(Array, Array)>,
    }
}

/// Builds a new [`AdaDelta`] optimizer
fn build_adadelta(builder: AdaDeltaBuilder) -> Result<AdaDelta, AdaDeltaBuildError> {
    let rho = builder.rho;
    let eps = builder.eps;

    if rho < 0.0 {
        return Err(AdaDeltaBuildError::NegativeRho);
    }

    if eps < 0.0 {
        return Err(AdaDeltaBuildError::NegativeEps);
    }

    Ok(AdaDelta {
        lr: array!(builder.lr),
        rho: array!(rho),
        eps: array!(eps),
        state: State::new(),
    })
}

impl AdaDelta {
    /// Default value for `rho`, matching Python MLX.
    pub const DEFAULT_RHO: f32 = 0.9;

    /// Default value for `eps`
    pub const DEFAULT_EPS: f32 = 1e-6;
}

impl Optimizer for AdaDelta {
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
        let (v, u) = get_mut_or_insert_with(&mut self.state, key, || (array!(0.0), array!(0.0)));

        let one_minus_rho = array!(1.0).subtract(&self.rho)?;
        let first_term = self.rho.multiply(&v)?;
        let second_term = one_minus_rho.multiply(gradient.square()?)?;
        let v_new = first_term.add(&second_term)?;

        let num = sqrt(&u.add(&self.eps)?)?;
        let den = sqrt(&v_new.add(&self.eps)?)?;
        let d = num.divide(&den)?.multiply(gradient)?;
        let first_term = self.rho.multiply(&u)?;
        let second_term = one_minus_rho.multiply(d.square()?)?;
        let u_new = first_term.add(&second_term)?;

        let param_new = parameter.subtract(self.lr.multiply(d)?)?;

        *parameter = param_new;

        *v = v_new;
        *u = u_new;

        Ok(())
    }
}

impl Updatable for AdaDelta {
    fn updatable_states_len(&self) -> usize {
        self.state.len() * 2
    }

    fn updatable_states(&self) -> impl IntoIterator<Item = &Array> {
        use itertools::Itertools;

        self.state
            .iter()
            .sorted_by(|a, b| a.0.cmp(b.0))
            .flat_map(|(_, (v, u))| [v, u])
    }

    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array> {
        use itertools::Itertools;

        self.state
            .iter_mut()
            .sorted_by(|a, b| a.0.cmp(b.0))
            .flat_map(|(_, (v, u))| [v, u])
    }

    optimizer_updatable_state_methods!();
}

impl_updatable_for_mut_optimizer!(AdaDelta);

#[cfg(test)]
mod tests {
    use super::AdaDelta;

    #[test]
    fn default_rho_matches_python_mlx() {
        let optimizer = AdaDelta::new(0.1_f32).unwrap();

        assert_eq!(optimizer.rho.item::<f32>(), 0.9);
    }
}
