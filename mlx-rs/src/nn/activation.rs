use std::f32::consts::PI;

use crate::module::{Module, Param};
use crate::ops::logsumexp_axis;
use crate::{
    array,
    error::{Exception, Result},
    ops::{abs, exp, maximum, minimum, multiply, select},
    transforms::compile::compile,
    Array,
};
use mlx_internal_macros::{generate_builder, Buildable, Builder};
use mlx_macros::ModuleParameters;

/// Applies the element-wise sigmoid logistic sigmoid.
///
/// For details, please see
/// [this documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sigmoid.html)
///
/// This is:
///
/// ```rust, ignore
/// sigmoid(x)
/// ```
pub fn sigmoid(x: impl AsRef<Array>) -> Result<Array> {
    crate::ops::sigmoid(x.as_ref())
}

/// Applies the Rectified Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// maximum(x, 0)
/// ```
pub fn relu(x: impl AsRef<Array>) -> Result<Array> {
    crate::ops::maximum(x.as_ref(), &array!(0))
}

/// Applies the Leaky Rectified Linear Unit.
///
/// `neg_slope` is default to 0.01 if not provided.
///
/// This is:
///
/// ```rust, ignore
/// maximum(neg_slope * x, x)
/// ```
pub fn leaky_relu(x: impl AsRef<Array>, neg_slope: impl Into<Option<f32>>) -> Result<Array> {
    let neg_slope = array!(neg_slope.into().unwrap_or(0.01));
    // We have to use this indirection, otherwise the compiler cannot
    // infer the lifetime of the value returned by the closure properly
    compiled_leaky_relu(x.as_ref(), &neg_slope)
}

/// Applies the Log Softmax function.
///
/// This is:
///
/// ```rust, ignore
/// x - logsumexp_axis(x, axis, true)
/// ```
pub fn log_softmax(x: impl AsRef<Array>, axis: impl Into<Option<i32>>) -> Result<Array> {
    let x = x.as_ref();
    let axis = axis.into().unwrap_or(-1);
    x.subtract(logsumexp_axis(x, axis, true)?)
}

/// Applies the Exponential Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// select(x.gt(0), x, alpha * (exp(x) - 1))
/// ```
///
/// # Params
///
/// - `x`: The input array
/// - `alpha`: Default to 1.0 if not provided
pub fn elu(x: impl AsRef<Array>, alpha: impl Into<Option<f32>>) -> Result<Array> {
    let alpha = array!(alpha.into().unwrap_or(1.0));
    // We have to use this indirection, otherwise the compiler cannot
    // infer the lifetime of the value returned by the closure properly
    compiled_elu(x.as_ref(), &alpha)
}

/// Applies the Rectified Linear Unit 6.
///
/// This is:
///
/// ```rust, ignore
/// minimum(maximum(x, 0), 6)
/// ```
pub fn relu6(x: impl AsRef<Array>) -> Result<Array> {
    compiled_relu6(x.as_ref())
}

/// Applies the Exponential Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// logaddexp(x, 0)
/// ```
pub fn softplus(x: impl AsRef<Array>) -> Result<Array> {
    crate::ops::logaddexp(x.as_ref(), &array!(0))
}

/// Applies the Softsign function.
///
/// This is:
///
/// ```rust, ignore
/// x / (1 + abs(x))
/// ```
pub fn softsign(x: impl AsRef<Array>) -> Result<Array> {
    compiled_softsign(x.as_ref())
}

/// Applies the Continuously Differentiable Exponential Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// maximum(x, 0) + alpha * (exp(minimum(x, 0) / alpha) - 1)
/// ```
pub fn celu(x: impl AsRef<Array>, alpha: impl Into<Option<f32>>) -> Result<Array> {
    let alpha = array!(alpha.into().unwrap_or(1.0));
    // We have to use this indirection, otherwise the compiler cannot
    // infer the lifetime of the value returned by the closure properly
    compiled_celu(x.as_ref(), &alpha)
}

/// Applies the Sigmoid Linear Unit. Also known as Swish.
///
/// This is:
///
/// ```rust, ignore
/// x * sigmoid(x)
/// ```
pub fn silu(x: impl AsRef<Array>) -> Result<Array> {
    compiled_silu(x.as_ref())
}

/// Applies the Log Sigmoid function.
///
/// This is:
///
/// ```rust, ignore
/// -softplus(-x)
/// ```
pub fn log_sigmoid(x: impl AsRef<Array>) -> Result<Array> {
    compiled_log_sigmoid(x.as_ref())
}

/// Applies the Gaussian Error Linear Units function.
///
/// This is:
///
/// ```rust, ignore
/// x * (1 + erf(x / 2.sqrt())) / 2
/// ```
pub fn gelu(x: impl AsRef<Array>) -> Result<Array> {
    compiled_gelu(x.as_ref())
}

/// An approximation to Gaussian Error Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// 0.5 * x * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x ** 3)))
/// ```
pub fn gelu_approximate(x: impl AsRef<Array>) -> Result<Array> {
    compiled_gelu_approximate(x.as_ref())
}

/// A fast approximation to Gaussian Error Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// x * sigmoid(1.773 * x)
/// ```
pub fn gelu_fast_approximate(x: impl AsRef<Array>) -> Result<Array> {
    compiled_gelu_fast_approximate(x.as_ref())
}

/// Applies the gated linear unit function.
///
/// This function splits the `axis` dimension of the input into two halves
/// (`a` and `b`) and applies `a * sigmoid(b)`.
pub fn glu(x: impl AsRef<Array>, axis: impl Into<Option<i32>>) -> Result<Array> {
    let split = x.as_ref().split_equal(2, axis)?;
    let (a, b) = (&split[0], &split[1]);
    Ok(a * sigmoid(b)?)
}

/// Applies the Step Activation Function.
///
/// This function implements a binary step activation, where the output is set
/// to 1 if the input is greater than a specified threshold, and 0 otherwise.
///
/// This is:
///
/// ```rust, ignore
/// select(x.gt(threshold), 1, 0)
/// ```
pub fn step(x: impl AsRef<Array>, threshold: impl Into<Option<f32>>) -> Result<Array> {
    let threshold = array!(threshold.into().unwrap_or(0.0));
    crate::ops::select(&x.as_ref().gt(threshold)?, &array!(1), &array!(0))
}

/// Applies the Scaled Exponential Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// elu(x, 1.67326) * 1.0507
/// ```
pub fn selu(x: impl AsRef<Array>) -> Result<Array> {
    compiled_selu(x.as_ref())
}

/// Applies the element-wise parametric ReLU.
///
/// This is:
///
/// ```rust, ignore
/// maximum(0, x) + alpha * minimum(0, x)
/// ```
pub fn prelu(x: impl AsRef<Array>, alpha: impl AsRef<Array>) -> Result<Array> {
    compiled_prelu(x.as_ref(), alpha.as_ref())
}

/// Applies the Mish function, element-wise.
///
/// Mish: A Self Regularized Non-Monotonic Neural Activation Function.
///
/// Reference: [https://arxiv.org/abs/1908.08681](https://arxiv.org/abs/1908.08681)
///
/// This is:
///
/// ```rust, ignore
/// x * tanh(softplus(x))
/// ```
pub fn mish(x: impl AsRef<Array>) -> Result<Array> {
    compiled_mish(x.as_ref())
}

/// Applies the hardswish function, element-wise.
///
/// This is:
///
/// ```rust, ignore
/// x * minimum(maximum(x + 3, 0), 6) / 6
/// ```
pub fn hard_swish(x: impl AsRef<Array>) -> Result<Array> {
    compiled_hard_swish(x.as_ref())
}

generate_builder! {
    /// Applies the gated linear unit function.
    ///
    /// This splits the `axis` dimension of the input into two halves
    /// (`a` and `b`) and applies `a * sigmoid(b)`.
    #[derive(Debug, Clone, ModuleParameters, Buildable)]
    #[module(root = crate)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct Glu {
        /// The axis to split the input tensor. Default to [`Glu::DEFAULT_AXIS`] if not provided.
        #[builder(optional, default = Glu::DEFAULT_AXIS)]
        pub axis: i32,
    }
}

impl Glu {
    /// The default axis value.
    pub const DEFAULT_AXIS: i32 = -1;
}

impl Module<&Array> for Glu {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        glu(x, self.axis)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the element-wise logistic sigmoid.
///
/// For details, please see
/// [this documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sigmoid.html)
///
/// This is:
///
/// ```rust, ignore
/// sigmoid(x)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Sigmoid;

impl Module<&Array> for Sigmoid {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        sigmoid(x)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Mish function, element-wise.
///
/// Mish: A Self Regularized Non-Monotonic Neural Activation Function.
///
/// Reference: [https://arxiv.org/abs/1908.08681](https://arxiv.org/abs/1908.08681)
///
/// This is:
///
/// ```rust, ignore
/// x * tanh(softplus(x))
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Mish;

impl Module<&Array> for Mish {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        mish(x)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Rectified Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// maximum(x, 0)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Relu;

impl Module<&Array> for Relu {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        relu(x)
    }

    fn training_mode(&mut self, _: bool) {}
}

generate_builder! {
    /// Applies the Leaky Rectified Linear Unit.
    ///
    /// This is:
    ///
    /// ```rust, ignore
    /// maximum(neg_slope * x, x)
    /// ```
    #[derive(Debug, Clone, ModuleParameters, Buildable)]
    #[module(root = crate)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct LeakyRelu {
        /// The negative slope. Default to [`LeakyReLU::DEFAULT_NEG_SLOPE`] if not provided.
        #[builder(optional, default = LeakyRelu::DEFAULT_NEG_SLOPE)]
        pub neg_slope: f32,
    }
}

impl LeakyRelu {
    /// The default negative slope value.
    pub const DEFAULT_NEG_SLOPE: f32 = 0.01;
}

impl Module<&Array> for LeakyRelu {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        leaky_relu(x, self.neg_slope)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Rectified Linear Unit 6.
///
/// This is:
///
/// ```rust, ignore
/// minimum(&maximum(x, 0).unwrap(), 6).unwrap()
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Relu6;

impl Module<&Array> for Relu6 {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        relu6(x)
    }

    fn training_mode(&mut self, _: bool) {}
}

generate_builder! {
    /// Applies the Softmax function.
    ///
    /// This is:
    ///
    /// ```rust, ignore
    /// softmax(&x, None, None)
    /// ```
    #[derive(Debug, Clone, ModuleParameters, Buildable)]
    #[module(root = crate)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct Softmax {
        /// The axis to apply the softmax.
        #[builder(optional, default = Softmax::DEFAULT_AXIS)]
        pub axis: i32,
    }
}

impl Softmax {
    /// The default axis value.
    pub const DEFAULT_AXIS: i32 = -1;
}

impl Module<&Array> for Softmax {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        crate::ops::softmax_axis(x, self.axis, None)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Softplus function.
///
/// This is:
///
/// ```rust, ignore
/// logaddexp(x, 0)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Softplus;

impl Module<&Array> for Softplus {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        softplus(x)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Softsign function.
///
/// This is:
///
/// ```rust, ignore
/// x / (array!(1) + abs(x)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Softsign;

impl Module<&Array> for Softsign {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        softsign(x)
    }

    fn training_mode(&mut self, _: bool) {}
}

generate_builder! {
    /// Applies the Continuously Differentiable Exponential Linear Unit.
    ///
    /// This is:
    ///
    /// ```rust, ignore
    /// maximum(x, 0.0).unwrap()
    ///     + alpha * (exp(&(minimum(x, 0.0).unwrap() / alpha)) - 1)
    /// ```
    #[derive(Debug, Clone, ModuleParameters, Buildable)]
    #[module(root = crate)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct Celu {
        /// The alpha value. Default to [`Celu::DEFAULT_ALPHA`] if not provided.
        #[builder(optional, default = Celu::DEFAULT_ALPHA)]
        pub alpha: f32,
    }
}

impl Celu {
    /// The default alpha value.
    pub const DEFAULT_ALPHA: f32 = 1.0;
}

impl Module<&Array> for Celu {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        celu(x, self.alpha)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Sigmoid Linear Unit. Also known as Swish.
///
/// This is:
///
/// ```rust, ignore
/// x * sigmoid(x)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Silu;

impl Module<&Array> for Silu {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        silu(x)
    }

    fn training_mode(&mut self, _: bool) {}
}

generate_builder! {
    /// Applies the Log Softmax function.
    ///
    /// This is:
    ///
    /// ```rust, ignore
    /// x - logsumexp(x, axis, true)
    /// ```
    #[derive(Debug, Clone, ModuleParameters, Buildable)]
    #[module(root = crate)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct LogSoftmax {
        /// The axis value. Default to [`LogSoftmax::DEFAULT_AXIS`] if not provided.
        #[builder(optional, default = LogSoftmax::DEFAULT_AXIS)]
        pub axis: i32,
    }
}

impl LogSoftmax {
    /// The default axis value.
    pub const DEFAULT_AXIS: i32 = -1;
}

impl Module<&Array> for LogSoftmax {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        log_softmax(x, self.axis)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Log Sigmoid function.
///
/// This is:
///
/// ```rust, ignore
/// -softplus(-x)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct LogSigmoid;

impl Module<&Array> for LogSigmoid {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        log_sigmoid(x)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the element-wise parametric ReLU.
///
/// This is:
///
/// ```rust, ignore
/// maximum(0, x) + alpha * minimum(0, x)
/// ```
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct Prelu {
    /// The alpha value. See [`prelu`] for more details.
    #[param]
    #[builder(ignore)]
    pub weight: Param<Array>, // TODO: double check if this is trainable
}

/// The builder for the Prelu module.
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_prelu,
    default_infallible,
    err = Exception,
)]
pub struct PreluBuilder {
    /// The count. Default to [`Prelu::DEFAULT_COUNT`] if not provided.
    #[builder(optional, default = Prelu::DEFAULT_COUNT)]
    pub count: i32,

    /// The value. Default to [`Prelu::DEFAULT_VALUE`] if not provided.
    #[builder(optional, default = Prelu::DEFAULT_VALUE)]
    pub value: f32,
}

/// Builds the Prelu module.
fn build_prelu(builder: PreluBuilder) -> Result<Prelu> {
    let count = builder.count;
    let value = builder.value;
    let weight = Param::new(crate::ops::full::<f32>(&[count], &array!(value))?);
    Ok(Prelu { weight })
}

impl Prelu {
    /// The default count value.
    pub const DEFAULT_COUNT: i32 = 1;

    /// The default value.
    pub const DEFAULT_VALUE: f32 = 0.25;
}

impl Module<&Array> for Prelu {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        prelu(x, &self.weight)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Variants of Gaussian Error Linear Units function.
#[derive(Debug, Clone, Copy, Default)]
pub enum GeluApprox {
    /// Uses [`gelu`]
    #[default]
    None,

    /// Uses [`gelu_approximate`]
    Precise,

    /// Uses [`gelu_fast_approximate`]
    Fast,
}

generate_builder! {
    /// Applies the Gaussian Error Linear Units function.
    ///
    /// There are three variants:
    ///
    /// - `GeluApprox::None`: Uses [`gelu`]. This is the default.
    /// - `GeluApprox::Precise`: Uses [`gelu_approximate`]
    /// - `GeluApprox::Fast`: Uses [`gelu_fast_approximate`]
    #[derive(Debug, Clone, ModuleParameters, Buildable)]
    #[module(root = crate)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct Gelu {
        /// The approximation to use. Default to `GeluApprox::None` if not provided.
        #[builder(optional, default = GeluApprox::None)]
        pub approximate: GeluApprox,
    }
}

impl Module<&Array> for Gelu {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        match self.approximate {
            GeluApprox::None => gelu(x),
            GeluApprox::Precise => gelu_approximate(x),
            GeluApprox::Fast => gelu_fast_approximate(x),
        }
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the hyperbolic tangent function
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Tanh;

impl Module<&Array> for Tanh {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        crate::ops::tanh(x)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the hardswish function, element-wise
///
/// This is:
///
/// ```rust, ignore
/// x * minimum(maximum(x + 3, 0), 6) / 6
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct HardSwish;

impl Module<&Array> for HardSwish {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        hard_swish(x)
    }

    fn training_mode(&mut self, _: bool) {}
}

generate_builder! {
    /// Applies the Step Activation Function.
    ///
    /// This function implements a binary step activation, where the output is set
    /// to 1 if the input is greater than a specified threshold, and 0 otherwise.
    ///
    /// This is:
    ///
    /// ```rust, ignore
    /// select(x.gt(threshold), 1, 0)
    /// ```
    #[derive(Debug, Clone, ModuleParameters, Buildable)]
    #[module(root = crate)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct Step {
        /// The threshold value. Default to [`Step::DEFAULT_THRESHOLD`] if not provided.
        #[builder(optional, default = Step::DEFAULT_THRESHOLD)]
        pub threshold: f32,
    }
}

impl Step {
    /// The default threshold value.
    pub const DEFAULT_THRESHOLD: f32 = 0.0;
}

impl Module<&Array> for Step {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        step(x, self.threshold)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Scaled Exponential Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// elu(x, 1.67326) * 1.0507
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Selu;

impl Module<&Array> for Selu {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array> {
        selu(x)
    }

    fn training_mode(&mut self, _: bool) {}
}

/* -------------------------------------------------------------------------- */
/*                        Compiled activation functions                       */
/* -------------------------------------------------------------------------- */

#[inline]
fn compiled_leaky_relu(x: &Array, neg_slope: &Array) -> Result<Array> {
    let f = |(x_, neg_slope_): (&Array, &Array)| {
        // This will not panic because a scalar can always be broadcasted to any shape
        let a = multiply(neg_slope_, x_)?;
        maximum(&a, x_)
    };
    let mut compiled = compile(f, true);
    compiled((x, neg_slope))
}

#[inline]
fn compiled_elu(x: &Array, alpha: &Array) -> Result<Array> {
    let f = |(x_, alpha_): (&Array, &Array)| {
        select(&x_.gt(&array!(0.0))?, x_, alpha_ * (exp(x_)? - array!(1.0)))
    };
    let mut compiled = compile(f, true);
    compiled((x, alpha))
}

#[inline]
fn compiled_relu6(x: &Array) -> Result<Array> {
    let f = |x_: &Array| minimum(maximum(x_, &array!(0.0))?, &array!(6.0));
    let mut compiled = compile(f, true);
    compiled(x)
}

#[inline]
fn compiled_softsign(x: &Array) -> Result<Array> {
    let f = |x_: &Array| x_.divide(array!(1.0) + abs(x_)?);
    let mut compiled = compile(f, true);
    compiled(x)
}

#[inline]
fn compiled_celu(x: &Array, alpha: &Array) -> Result<Array> {
    let f = |(x_, alpha_): (&Array, &Array)| {
        maximum(x_, &array!(0.0))?
            .add(alpha_.multiply(exp(&(minimum(x_, &array!(0.0))? / alpha_))? - array!(1.0))?)
    };
    let mut compiled = compile(f, true);
    compiled((x, alpha))
}

#[inline]
fn compiled_silu(x: &Array) -> Result<Array> {
    let f = |x_: &Array| x_.multiply(sigmoid(x_)?);
    let mut compiled = compile(f, true);
    compiled(x)
}

#[inline]
fn compiled_log_sigmoid(x: &Array) -> Result<Array> {
    let f = |x_: &Array| Ok(-softplus(&(-x_))?);
    let mut compiled = compile(f, true);
    compiled(x)
}

#[inline]
fn compiled_gelu(x: &Array) -> Result<Array> {
    use crate::ops::erf;
    let f = |x_: &Array| {
        x_.multiply(array!(1) + erf(&(x_ / array!(2f32.sqrt())))?)?
            .divide(array!(2.0))
    };
    let mut compiled = compile(f, true);
    compiled(x)
}

#[inline]
fn compiled_gelu_approximate(x: &Array) -> Result<Array> {
    use crate::ops::{sqrt, tanh};

    let f = move |x_: &Array| {
        // 0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x ** 3)))
        array!(0.5).multiply(x_)?.multiply(
            array!(1.0).add(tanh(
                &(sqrt(&array!(2.0 / PI))?
                    .multiply(x_ + array!(0.044715).multiply(x_.power(&array!(3))?)?)?),
            )?)?,
        )
    };
    let mut compiled = compile(f, true);
    compiled(x)
}

#[inline]
fn compiled_gelu_fast_approximate(x: &Array) -> Result<Array> {
    let f = |x_: &Array| x_.multiply(sigmoid(&(array!(1.773) * x_))?);
    let mut compiled = compile(f, true);
    compiled(x)
}

#[inline]
fn compiled_selu(x: &Array) -> Result<Array> {
    let f = |x_: &Array| elu(x_, 1.67326)?.multiply(array!(1.0507));
    let mut compiled = compile(f, true);
    compiled(x)
}

#[inline]
fn compiled_prelu(x: &Array, alpha: &Array) -> Result<Array> {
    let f = |(x_, alpha_): (&Array, &Array)| {
        maximum(&array!(0.0), x_)?.add(alpha_ * minimum(&array!(0.0), x_)?)
    };
    let mut compiled = compile(f, true);
    compiled((x, alpha))
}

#[inline]
fn compiled_mish(x: &Array) -> Result<Array> {
    use crate::ops::tanh;

    let f = |x_: &Array| x_.multiply(tanh(&softplus(x_)?)?);
    let mut compiled = compile(f, true);
    compiled(x)
}

#[inline]
fn compiled_hard_swish(x: &Array) -> Result<Array> {
    let f = |x_: &Array| {
        let max_x_plus_3 = maximum(&(x_ + array!(3.0)), &array!(0.0))?;
        x_.multiply(minimum(&max_x_plus_3, &array!(6.0))?)?
            .divide(&array!(6.0))
    };
    let mut compiled = compile(f, true);
    compiled(x)
}

// The following tests are ported from the swift binding:
// mlx-swift/Tests/MLXTests/IntegrationTests.swift
#[cfg(test)]
mod tests {
    use crate::{builder::Builder, random::uniform, Dtype};
    use float_eq::assert_float_eq;

    use super::*;

    #[test]
    fn test_glu() {
        crate::random::seed(850).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.547_252_66,
            abs <= 0.010_945_053
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            140.096_68,
            abs <= 2.801_933_5
        );
        let result = Glu::new().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 8]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.333_276_75,
            abs <= 0.006_665_535
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            42.659_424,
            abs <= 0.853_188_46
        );
    }

    #[test]
    fn test_sigmoid() {
        crate::random::seed(589).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.529_697_9,
            abs <= 0.010_593_958
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            135.602_66,
            abs <= 2.712_053_3
        );
        let result = Sigmoid.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.627_014,
            abs <= 0.012_540_28
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            160.515_58,
            abs <= 3.210_311_7
        );
    }

    #[test]
    fn test_mish() {
        crate::random::seed(122).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.501_719_8,
            abs <= 0.010_034_395
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            128.440_26,
            abs <= 2.568_805_2
        );
        let result = Mish.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.395_375_73,
            abs <= 0.007_907_514
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            101.216_19,
            abs <= 2.024_323_7
        );
    }

    #[test]
    fn test_relu() {
        crate::random::seed(400).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.478_322_74,
            abs <= 0.009_566_455
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            122.450_62,
            abs <= 2.449_012_5
        );
        let result = Relu.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.478_322_74,
            abs <= 0.009_566_455
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            122.450_62,
            abs <= 2.449_012_5
        );
    }

    #[test]
    fn test_leaky_relu() {
        crate::random::seed(93).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.499_930_68,
            abs <= 0.009_998_614
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            127.982_254,
            abs <= 2.559_645_2
        );
        let result = LeakyRelu::new().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.499_930_68,
            abs <= 0.009_998_614
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            127.982_254,
            abs <= 2.559_645_2
        );
    }

    #[test]
    fn test_relu6() {
        crate::random::seed(379).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.493_258_66,
            abs <= 0.009_865_173
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            126.274_216,
            abs <= 2.525_484_3
        );
        let result = Relu6.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.493_258_66,
            abs <= 0.009_865_173
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            126.274_216,
            abs <= 2.525_484_3
        );
    }

    #[test]
    fn test_softmax() {
        crate::random::seed(853).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.514_396_3,
            abs <= 0.010_287_926_5
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            131.685_46,
            abs <= 2.633_709_2
        );
        let result = Softmax::new().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.062_499_996,
            abs <= 0.001_25
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            15.999_999,
            abs <= 0.32
        );
    }

    #[test]
    fn test_softplus() {
        crate::random::seed(118).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.498_981_42,
            abs <= 0.009_979_628
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            127.739_24,
            abs <= 2.554_784_8
        );
        let result = Softplus.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.982_857_76,
            abs <= 0.019_657_155
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            251.611_59,
            abs <= 5.032_232
        );
    }

    #[test]
    fn test_softsign() {
        crate::random::seed(37).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.506_551_27,
            abs <= 0.010_131_026
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            129.677_12,
            abs <= 2.593_542_6
        );
        let result = Softsign.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.314_089_83,
            abs <= 0.006_281_797
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            80.407,
            abs <= 1.608_14
        );
    }

    // The unit test below is adapted from the python binding:
    // mlx/python/tests/test_nn.py
    #[test]
    fn test_celu() {
        let x = array!([1.0, -1.0, 0.0]);
        let y = Celu::new().forward(&x).unwrap();
        let epsilon = array!(1e-4);
        let expected_y = array!([1.0, -0.6321, 0.0]);
        assert!(y
            .subtract(&expected_y)
            .unwrap()
            .abs()
            .unwrap()
            .lt(&epsilon)
            .unwrap()
            .all(None)
            .unwrap()
            .item_exact::<bool>());
        assert_eq!(y.shape(), &[3]);
        assert_eq!(y.dtype(), Dtype::Float32);

        let y = CeluBuilder::new()
            .alpha(1.1)
            .build()
            .unwrap()
            .forward(&x)
            .unwrap();
        let expected_y = array!([1.0, -0.6568, 0.0]);
        assert!(y
            .subtract(&expected_y)
            .unwrap()
            .abs()
            .unwrap()
            .lt(&epsilon)
            .unwrap()
            .all(None)
            .unwrap()
            .item_exact::<bool>());
        assert_eq!(y.shape(), &[3]);
        assert_eq!(y.dtype(), Dtype::Float32);
    }

    #[test]
    fn test_silu() {
        crate::random::seed(22).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.502_970_6,
            abs <= 0.010_059_412
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            128.760_47,
            abs <= 2.575_209_4
        );
        let result = Silu.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.331_970_93,
            abs <= 0.006_639_418_7
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            84.984_56,
            abs <= 1.699_691_2
        );
    }

    #[test]
    fn test_log_softmax() {
        crate::random::seed(199).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.527_843_7,
            abs <= 0.010_556_874
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            135.127_99,
            abs <= 2.702_559_7
        );
        let result = LogSoftmax::new().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            -2.810_954_6,
            abs <= 0.056_219_09
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            -719.604_4,
            abs <= 14.392_087
        );
    }

    #[test]
    fn test_log_sigmoid() {
        crate::random::seed(984).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.510_977_7,
            abs <= 0.010_219_553_5
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            130.810_29,
            abs <= 2.616_205_7
        );
        let result = LogSigmoid.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            -0.479_598_55,
            abs <= 0.009_591_971
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            -122.777_23,
            abs <= 2.455_544_5
        );
    }

    #[test]
    fn test_prelu() {
        crate::random::seed(993).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.496_651_44,
            abs <= 0.009_933_028
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            127.142_77,
            abs <= 2.542_855_3
        );
        let result = Prelu::new().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.496_651_44,
            abs <= 0.009_933_028
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            127.142_77,
            abs <= 2.542_855_3
        );
    }

    #[test]
    fn test_gelu() {
        crate::random::seed(189).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.492_950_32,
            abs <= 0.009_859_007
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            126.195_28,
            abs <= 2.523_905_8
        );
        let result = Gelu::new().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.365_638_38,
            abs <= 0.007_312_767_7
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            93.603_424,
            abs <= 1.872_068_5
        );
    }

    #[test]
    fn test_tanh() {
        crate::random::seed(735).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.474_122_7,
            abs <= 0.009_482_454_5
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            121.375_41,
            abs <= 2.427_508_4
        );
        let result = Tanh.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.413_079_68,
            abs <= 0.008_261_594
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            105.748_4,
            abs <= 2.114_968
        );
    }

    #[test]
    fn test_hardswish() {
        crate::random::seed(126).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.491_892_46,
            abs <= 0.009_837_849
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            125.924_47,
            abs <= 2.518_489_4
        );
        let result = HardSwish.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.299_602_24,
            abs <= 0.005_992_044_7
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            76.698_17,
            abs <= 1.533_963_4
        );
    }

    #[test]
    fn test_step() {
        crate::random::seed(490).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.479_360_64,
            abs <= 0.009_587_212_5
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            122.716_324,
            abs <= 2.454_326_4
        );
        let result = Step::new().forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Int32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            1.0,
            abs <= 0.02
        );
        assert_eq!(result.sum(None).unwrap().item_exact::<i32>(), 256);
    }

    #[test]
    fn test_selu() {
        crate::random::seed(215).unwrap();
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item_exact::<f32>(),
            0.493_026_8,
            abs <= 0.009_860_536
        );
        assert_float_eq!(
            a.sum(None).unwrap().item_exact::<f32>(),
            126.214_86,
            abs <= 2.524_297_2
        );
        let result = Selu.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.518_023_2,
            abs <= 0.010_360_463_5
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            132.613_94,
            abs <= 2.652_278_7
        );
    }
}
