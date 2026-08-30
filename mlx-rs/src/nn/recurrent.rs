use std::sync::Arc;

use crate::{
    array,
    error::Exception,
    module::{Module, Param},
    ops::{
        addmm,
        indexing::{Ellipsis, IndexOp},
        matmul, sigmoid, split, stack_axis, tanh,
    },
    random::uniform,
    with_stream, Array, Stream,
};
use mlx_internal_macros::{generate_builder, Buildable, Builder};
use mlx_macros::ModuleParameters;

/// Type alias for the non-linearity function.
pub type NonLinearity = dyn Fn(&Array, &Stream) -> Result<Array, Exception>;

/// An Elman recurrent layer.
///
/// The input is a sequence of shape `NLD` or `LD` where:
///
/// * `N` is the optional batch dimension
/// * `L` is the sequence length
/// * `D` is the input's feature dimension
///
/// The hidden state `h` has shape `NH` or `H`, depending on
/// whether the input is batched or not. Returns the hidden state at each
/// time step, of shape `NLH` or `LH`.
#[derive(Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct Rnn {
    /// non-linearity function to use
    pub non_linearity: Arc<NonLinearity>,

    /// Wxh
    #[param]
    pub wxh: Param<Array>,

    /// Whh
    #[param]
    pub whh: Param<Array>,

    /// Bias. Enabled by default.
    #[param]
    pub bias: Param<Option<Array>>,
}

/// Builder for the [`Rnn`] module.
#[derive(Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_rnn,
    err = Exception,
)]
pub struct RnnBuilder {
    /// Dimension of the input, `D`.
    pub input_size: i32,

    /// Dimension of the hidden state, `H`.
    pub hidden_size: i32,

    /// non-linearity function to use. Default to `tanh` if not set.
    #[builder(optional, default = Rnn::DEFAULT_NONLINEARITY)]
    pub non_linearity: Option<Arc<NonLinearity>>,

    /// Bias. Default to [`Rnn::DEFAULT_BIAS`].
    #[builder(optional, default = Rnn::DEFAULT_BIAS)]
    pub bias: bool,
}

impl std::fmt::Debug for RnnBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("RnnBuilder")
            .field("bias", &self.bias)
            .finish()
    }
}

/// Build the [`Rnn`] module.
fn build_rnn(builder: RnnBuilder) -> Result<Rnn, Exception> {
    let input_size = builder.input_size;
    let hidden_size = builder.hidden_size;
    let non_linearity = builder
        .non_linearity
        .unwrap_or_else(|| Arc::new(|x, d| with_stream(d, || tanh(x))));

    let scale = 1.0 / (input_size as f32).sqrt();
    let wxh = uniform::<_, f32>(-scale, scale, &[hidden_size, input_size], None)?;
    let whh = uniform::<_, f32>(-scale, scale, &[hidden_size, hidden_size], None)?;
    let bias = if builder.bias {
        Some(uniform::<_, f32>(-scale, scale, &[hidden_size], None)?)
    } else {
        None
    };

    Ok(Rnn {
        non_linearity,
        wxh: Param::new(wxh),
        whh: Param::new(whh),
        bias: Param::new(bias),
    })
}

impl std::fmt::Debug for Rnn {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Rnn")
            .field("wxh", &self.wxh)
            .field("whh", &self.whh)
            .field("bias", &self.bias)
            .finish()
    }
}

impl Rnn {
    /// Default value for bias
    pub const DEFAULT_BIAS: bool = true;

    /// RnnBuilder::non_linearity is initialized with `None`, and the default non-linearity is `tanh` if not set.
    pub const DEFAULT_NONLINEARITY: Option<Arc<NonLinearity>> = None;

    /// Apply a single step of the RNN.
    pub fn step(&mut self, x: &Array, hidden: Option<&Array>) -> Result<Array, Exception> {
        let x = if let Some(bias) = &self.bias.value {
            addmm(bias, x, self.wxh.t(), None, None)?
        } else {
            matmul(x, self.wxh.t())?
        };

        let mut all_hidden = Vec::new();
        for index in 0..x.dim(-2) {
            let hidden = match hidden {
                Some(hidden_) => addmm(
                    x.index((Ellipsis, index, 0..)),
                    hidden_,
                    self.whh.t(),
                    None,
                    None,
                )?,
                None => x.index((Ellipsis, index, 0..)),
            };

            let hidden = (self.non_linearity)(&hidden, &Stream::default())?;
            all_hidden.push(hidden);
        }

        stack_axis(&all_hidden[..], -2)
    }
}

generate_builder! {
    /// Input for the RNN module.
    #[derive(Debug, Clone, Buildable)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct RnnInput<'a> {
        /// Input tensor
        pub x: &'a Array,

        /// Hidden state
        #[builder(optional, default = None)]
        pub hidden: Option<&'a Array>,
    }
}

impl<'a> From<&'a Array> for RnnInput<'a> {
    fn from(x: &'a Array) -> Self {
        RnnInput { x, hidden: None }
    }
}

impl<'a> From<(&'a Array,)> for RnnInput<'a> {
    fn from(input: (&'a Array,)) -> Self {
        RnnInput {
            x: input.0,
            hidden: None,
        }
    }
}

impl<'a> From<(&'a Array, &'a Array)> for RnnInput<'a> {
    fn from(input: (&'a Array, &'a Array)) -> Self {
        RnnInput {
            x: input.0,
            hidden: Some(input.1),
        }
    }
}

impl<'a> From<(&'a Array, Option<&'a Array>)> for RnnInput<'a> {
    fn from(input: (&'a Array, Option<&'a Array>)) -> Self {
        RnnInput {
            x: input.0,
            hidden: input.1,
        }
    }
}

impl<'a, Input> Module<Input> for Rnn
where
    Input: Into<RnnInput<'a>>,
{
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Input) -> Result<Array, Exception> {
        let input = input.into();
        self.step(input.x, input.hidden)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

/// A gated recurrent unit (GRU) RNN layer.
///
/// The input has shape `NLD` or `LD` where:
///
/// * `N` is the optional batch dimension
/// * `L` is the sequence length
/// * `D` is the input's feature dimension
///
/// The hidden state `h` has shape `NH` or `H`, depending on
/// whether the input is batched or not. Returns the hidden state at each
/// time step, of shape `NLH` or `LH`.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct Gru {
    /// Dimension of the hidden state, `H`
    pub hidden_size: i32,

    /// Wx
    #[param]
    pub wx: Param<Array>,

    /// Wh
    #[param]
    pub wh: Param<Array>,

    /// Bias. Enabled by default.
    #[param]
    pub bias: Param<Option<Array>>,

    /// bhn. Enabled by default.
    #[param]
    pub bhn: Param<Option<Array>>,
}

/// Builder for the [`Gru`] module.
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_gru,
    err = Exception,
)]
pub struct GruBuilder {
    /// Dimension of the input, `D`.
    pub input_size: i32,

    /// Dimension of the hidden state, `H`.
    pub hidden_size: i32,

    /// Bias. Default to [`Gru::DEFAULT_BIAS`].
    #[builder(optional, default = Gru::DEFAULT_BIAS)]
    pub bias: bool,
}

fn build_gru(builder: GruBuilder) -> Result<Gru, Exception> {
    let input_size = builder.input_size;
    let hidden_size = builder.hidden_size;

    let scale = 1.0 / f32::sqrt(hidden_size as f32);
    let wx = uniform::<_, f32>(-scale, scale, &[3 * hidden_size, input_size], None)?;
    let wh = uniform::<_, f32>(-scale, scale, &[3 * hidden_size, hidden_size], None)?;
    let (bias, bhn) = if builder.bias {
        let bias = uniform::<_, f32>(-scale, scale, &[3 * hidden_size], None)?;
        let bhn = uniform::<_, f32>(-scale, scale, &[hidden_size], None)?;
        (Some(bias), Some(bhn))
    } else {
        (None, None)
    };

    Ok(Gru {
        hidden_size,
        wx: Param::new(wx),
        wh: Param::new(wh),
        bias: Param::new(bias),
        bhn: Param::new(bhn),
    })
}

impl Gru {
    /// Enable `bias` and `bhn` by default
    pub const DEFAULT_BIAS: bool = true;

    /// Apply a single step of the GRU.
    pub fn step(&mut self, x: &Array, hidden: Option<&Array>) -> Result<Array, Exception> {
        let x = if let Some(b) = &self.bias.value {
            addmm(b, x, self.wx.t(), None, None)?
        } else {
            matmul(x, self.wx.t())?
        };

        let x_rz = x.index((Ellipsis, ..(-self.hidden_size)));
        let x_n = x.index((Ellipsis, (-self.hidden_size)..));

        let mut all_hidden = Vec::new();

        for index in 0..x.dim(-2) {
            let mut rz = x_rz.index((Ellipsis, index, ..));
            let mut h_proj_n = None;
            if let Some(hidden_) = hidden {
                let h_proj = matmul(hidden_, self.wh.t())?;
                let h_proj_rz = h_proj.index((Ellipsis, ..(-self.hidden_size)));
                h_proj_n = Some(h_proj.index((Ellipsis, (-self.hidden_size)..)));

                if let Some(bhn) = &self.bhn.value {
                    h_proj_n = h_proj_n
                        .map(|h_proj_n| h_proj_n.add(bhn))
                        // This is not matrix transpose, but from `Option<Result<_>>` to `Result<Option<_>>`
                        .transpose()?;
                }

                rz = rz.add(h_proj_rz)?;
            }

            rz = sigmoid(&rz)?;

            let parts = split(&rz, 2, -1)?;
            let r = &parts[0];
            let z = &parts[1];

            let mut n = x_n.index((Ellipsis, index, 0..));

            if let Some(h_proj_n) = h_proj_n {
                n = n.add(r.multiply(h_proj_n)?)?;
            }
            n = tanh(&n)?;

            let hidden = match hidden {
                Some(hidden) => array!(1.0)
                    .subtract(z)?
                    .multiply(&n)?
                    .add(z.multiply(hidden)?)?,
                None => array!(1.0).subtract(z)?.multiply(&n)?,
            };

            all_hidden.push(hidden);
        }

        stack_axis(&all_hidden[..], -2)
    }
}

/// Type alias for the input of the GRU module.
pub type GruInput<'a> = RnnInput<'a>;

/// Type alias for the builder of the input of the GRU module.
pub type GruInputBuilder<'a> = RnnInputBuilder<'a>;

impl<'a, Input> Module<Input> for Gru
where
    Input: Into<GruInput<'a>>,
{
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Input) -> Result<Array, Exception> {
        let input = input.into();
        self.step(input.x, input.hidden)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

/// A long short-term memory (LSTM) RNN layer.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct Lstm {
    /// Wx
    #[param]
    pub wx: Param<Array>,

    /// Wh
    #[param]
    pub wh: Param<Array>,

    /// Bias. Enabled by default.
    #[param]
    pub bias: Param<Option<Array>>,
}

/// Builder for the [`Lstm`] module.
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_lstm,
    err = Exception,
)]
pub struct LstmBuilder {
    /// Dimension of the input, `D`.
    pub input_size: i32,

    /// Dimension of the hidden state, `H`.
    pub hidden_size: i32,

    /// Bias. Default to [`Lstm::DEFAULT_BIAS`].
    #[builder(optional, default = Lstm::DEFAULT_BIAS)]
    pub bias: bool,
}

fn build_lstm(builder: LstmBuilder) -> Result<Lstm, Exception> {
    let input_size = builder.input_size;
    let hidden_size = builder.hidden_size;
    let scale = 1.0 / f32::sqrt(hidden_size as f32);
    let wx = uniform::<_, f32>(-scale, scale, &[4 * hidden_size, input_size], None)?;
    let wh = uniform::<_, f32>(-scale, scale, &[4 * hidden_size, hidden_size], None)?;
    let bias = if builder.bias {
        Some(uniform::<_, f32>(-scale, scale, &[4 * hidden_size], None)?)
    } else {
        None
    };

    Ok(Lstm {
        wx: Param::new(wx),
        wh: Param::new(wh),
        bias: Param::new(bias),
    })
}

generate_builder! {
    /// Input for the LSTM module.
    #[derive(Debug, Clone, Buildable)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct LstmInput<'a> {
        /// Input tensor
        pub x: &'a Array,

        /// Hidden state
        #[builder(optional, default = None)]
        pub hidden: Option<&'a Array>,

        /// Cell state
        #[builder(optional, default = None)]
        pub cell: Option<&'a Array>,
    }
}

impl<'a> From<&'a Array> for LstmInput<'a> {
    fn from(x: &'a Array) -> Self {
        LstmInput {
            x,
            hidden: None,
            cell: None,
        }
    }
}

impl<'a> From<(&'a Array,)> for LstmInput<'a> {
    fn from(input: (&'a Array,)) -> Self {
        LstmInput {
            x: input.0,
            hidden: None,
            cell: None,
        }
    }
}

impl<'a> From<(&'a Array, &'a Array)> for LstmInput<'a> {
    fn from(input: (&'a Array, &'a Array)) -> Self {
        LstmInput {
            x: input.0,
            hidden: Some(input.1),
            cell: None,
        }
    }
}

impl<'a> From<(&'a Array, &'a Array, &'a Array)> for LstmInput<'a> {
    fn from(input: (&'a Array, &'a Array, &'a Array)) -> Self {
        LstmInput {
            x: input.0,
            hidden: Some(input.1),
            cell: Some(input.2),
        }
    }
}

impl<'a> From<(&'a Array, Option<&'a Array>)> for LstmInput<'a> {
    fn from(input: (&'a Array, Option<&'a Array>)) -> Self {
        LstmInput {
            x: input.0,
            hidden: input.1,
            cell: None,
        }
    }
}

impl<'a> From<(&'a Array, Option<&'a Array>, Option<&'a Array>)> for LstmInput<'a> {
    fn from(input: (&'a Array, Option<&'a Array>, Option<&'a Array>)) -> Self {
        LstmInput {
            x: input.0,
            hidden: input.1,
            cell: input.2,
        }
    }
}

impl Lstm {
    /// Default value for `bias`
    pub const DEFAULT_BIAS: bool = true;

    /// Apply a single step of the LSTM.
    pub fn step(
        &mut self,
        x: &Array,
        hidden: Option<&Array>,
        cell: Option<&Array>,
    ) -> Result<(Array, Array), Exception> {
        let x = if let Some(b) = &self.bias.value {
            addmm(b, x, self.wx.t(), None, None)?
        } else {
            matmul(x, self.wx.t())?
        };

        let mut all_hidden = Vec::new();
        let mut all_cell = Vec::new();

        for index in 0..x.dim(-2) {
            let mut ifgo = x.index((Ellipsis, index, 0..));
            if let Some(hidden) = hidden {
                ifgo = addmm(&ifgo, hidden, self.wh.t(), None, None)?;
            }

            let pieces = split(&ifgo, 4, -1)?;

            let i = sigmoid(&pieces[0])?;
            let f = sigmoid(&pieces[1])?;
            let g = tanh(&pieces[2])?;
            let o = sigmoid(&pieces[3])?;

            let cell = match cell {
                Some(cell) => f.multiply(cell)?.add(i.multiply(&g)?)?,
                None => i.multiply(&g)?,
            };

            let hidden = o.multiply(tanh(&cell)?)?;

            all_hidden.push(hidden);
            all_cell.push(cell);
        }

        Ok((
            stack_axis(&all_hidden[..], -2)?,
            stack_axis(&all_cell[..], -2)?,
        ))
    }
}

impl<'a, Input> Module<Input> for Lstm
where
    Input: Into<LstmInput<'a>>,
{
    type Output = (Array, Array);
    type Error = Exception;

    fn forward(&mut self, input: Input) -> Result<(Array, Array), Exception> {
        let input = input.into();
        self.step(input.x, input.hidden, input.cell)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

// The uint tests below are ported from the python codebase
#[cfg(test)]
mod tests {
    use crate::{builder::Builder, ops::maximum, random::normal};

    use super::*;

    #[test]
    fn test_rnn() {
        let mut layer = Rnn::new(5, 12).unwrap();
        let inp = normal::<f32>(&[2, 25, 5], None, None, None).unwrap();

        let h_out = layer.forward(RnnInput::from(&inp)).unwrap();
        assert_eq!(h_out.shape(), &[2, 25, 12]);

        let nonlinearity = |x: &Array, d: &Stream| with_stream(d, || maximum(x, array!(0.0)));
        let mut layer = RnnBuilder::new(5, 12)
            .bias(false)
            .non_linearity(Arc::new(nonlinearity) as Arc<NonLinearity>)
            .build()
            .unwrap();

        let h_out = layer.forward(RnnInput::from(&inp)).unwrap();
        assert_eq!(h_out.shape(), &[2, 25, 12]);

        let inp = normal::<f32>(&[44, 5], None, None, None).unwrap();
        let h_out = layer.forward(RnnInput::from(&inp)).unwrap();
        assert_eq!(h_out.shape(), &[44, 12]);

        let hidden = h_out.index((-1, ..));
        let h_out = layer.forward(RnnInput::from((&inp, &hidden))).unwrap();
        assert_eq!(h_out.shape(), &[44, 12]);
    }

    #[test]
    fn test_gru() {
        let mut layer = Gru::new(5, 12).unwrap();
        let inp = normal::<f32>(&[2, 25, 5], None, None, None).unwrap();

        let h_out = layer.forward(GruInput::from(&inp)).unwrap();
        assert_eq!(h_out.shape(), &[2, 25, 12]);

        let hidden = h_out.index((.., -1, ..));
        let h_out = layer.forward(GruInput::from((&inp, &hidden))).unwrap();
        assert_eq!(h_out.shape(), &[2, 25, 12]);

        let inp = normal::<f32>(&[44, 5], None, None, None).unwrap();
        let h_out = layer.forward(GruInput::from(&inp)).unwrap();
        assert_eq!(h_out.shape(), &[44, 12]);

        let hidden = h_out.index((-1, ..));
        let h_out = layer.forward(GruInput::from((&inp, &hidden))).unwrap();
        assert_eq!(h_out.shape(), &[44, 12]);
    }

    #[test]
    fn test_lstm() {
        let mut layer = Lstm::new(5, 12).unwrap();
        let inp = normal::<f32>(&[2, 25, 5], None, None, None).unwrap();

        let (h_out, c_out) = layer.forward(LstmInput::from(&inp)).unwrap();
        assert_eq!(h_out.shape(), &[2, 25, 12]);
        assert_eq!(c_out.shape(), &[2, 25, 12]);

        let (h_out, c_out) = layer
            .step(
                &inp,
                Some(&h_out.index((.., -1, ..))),
                Some(&c_out.index((.., -1, ..))),
            )
            .unwrap();
        assert_eq!(h_out.shape(), &[2, 25, 12]);
        assert_eq!(c_out.shape(), &[2, 25, 12]);

        let inp = normal::<f32>(&[44, 5], None, None, None).unwrap();
        let (h_out, c_out) = layer.forward(LstmInput::from(&inp)).unwrap();
        assert_eq!(h_out.shape(), &[44, 12]);
        assert_eq!(c_out.shape(), &[44, 12]);

        let hidden = h_out.index((-1, ..));
        let cell = c_out.index((-1, ..));
        let (h_out, c_out) = layer
            .forward(LstmInput::from((&inp, &hidden, &cell)))
            .unwrap();
        assert_eq!(h_out.shape(), &[44, 12]);
        assert_eq!(c_out.shape(), &[44, 12]);
    }
}
