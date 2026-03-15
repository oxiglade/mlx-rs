use std::{cell::RefCell, collections::HashMap};

use crate::{
    array,
    error::Exception,
    module::{Module, Param},
    ops::{
        arange, concatenate_axis, exp,
        indexing::{NewAxis, TryIndexOp},
        log,
    },
    Array, Dtype,
};
use mlx_internal_macros::{generate_builder, Buildable, Builder};
use mlx_macros::ModuleParameters;

/// Type alias for [`RotaryPositionalEncoding`].
pub type Rope = RotaryPositionalEncoding;

/// Type alias for [`RotaryPositionalEncodingBuilder`].
pub type RopeBuilder = RotaryPositionalEncodingBuilder;

generate_builder! {
    /// Implements the rotary positional encoding.
    ///
    /// The traditional implementation rotates consecutive pairs of elements in the
    /// feature dimension while the default implementation rotates pairs with
    /// stride half the feature dimensions for efficiency.
    ///
    /// For more details see _RoFormer: Enhanced Transformer with Rotary Position
    /// Embedding_ ([https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864))
    #[derive(Debug, Clone, ModuleParameters, Buildable)]
    #[module(root = crate)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct RotaryPositionalEncoding {
        /// The feature dimensions to be rotated. If the input feature is larger
        /// than dims then the rest is left unchanged
        pub dimensions: i32,

        /// If `true` choose the traditional implementation which is slightly
        /// less efficient
        #[builder(optional, default = RotaryPositionalEncoding::DEFAULT_TRADITIONAL)]
        pub traditional: bool,

        /// The base used to compute angular frequency for each dimension in the
        /// positional encodings
        #[builder(optional, default = RotaryPositionalEncoding::DEFAULT_BASE)]
        pub base: f32,

        /// scale used to scale the positions
        #[builder(optional, default = RotaryPositionalEncoding::DEFAULT_SCALE)]
        pub scale: f32,
    }
}

impl RotaryPositionalEncoding {
    /// Default value for `traditional` field.
    pub const DEFAULT_TRADITIONAL: bool = false;

    /// Default value for `base` field.
    pub const DEFAULT_BASE: f32 = 10_000.0;

    /// Default value for `scale` field.
    pub const DEFAULT_SCALE: f32 = 1.0;
}

generate_builder! {
    /// Input for the [`RotaryPositionalEncoding`] module.
    #[derive(Debug, Buildable, Clone)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct RopeInput<'a> {
        /// The input tensor.
        pub x: &'a Array,

        /// Offset
        #[builder(optional, default = RopeInput::DEFAULT_OFFSET)]
        pub offset: i32,
    }
}

impl RopeInput<'_> {
    /// Default value for `offset` field.
    pub const DEFAULT_OFFSET: i32 = 0;
}

impl<'a> From<&'a Array> for RopeInput<'a> {
    fn from(x: &'a Array) -> Self {
        RopeInput {
            x,
            offset: Self::DEFAULT_OFFSET,
        }
    }
}

impl<'a> From<(&'a Array,)> for RopeInput<'a> {
    fn from((x,): (&'a Array,)) -> Self {
        RopeInput {
            x,
            offset: Self::DEFAULT_OFFSET,
        }
    }
}

impl<'a> From<(&'a Array, i32)> for RopeInput<'a> {
    fn from((x, offset): (&'a Array, i32)) -> Self {
        RopeInput { x, offset }
    }
}

impl<'a, Input> Module<Input> for RotaryPositionalEncoding
where
    Input: Into<RopeInput<'a>>,
{
    type Error = Exception;

    type Output = Array;

    fn forward(&mut self, input: Input) -> Result<Self::Output, Self::Error> {
        let RopeInput { x, offset } = input.into();
        // Note: Do NOT reshape the input. The underlying fast::rope kernel
        // expects the input shape to be preserved. Reshaping [B, H, L, D] to
        // [B*H, L, D] causes incorrect behavior for multi-head attention.
        crate::fast::rope(
            x,
            self.dimensions,
            self.traditional,
            self.base,
            self.scale,
            offset,
            None,
        )
    }

    fn training_mode(&mut self, _mode: bool) {}
}

/// Type alias for [`SinusoidalPositionalEncoding`].
pub type Sinpe = SinusoidalPositionalEncoding;

/// Type alias for [`SinusoidalPositionalEncodingBuilder`].
pub type SinpeBuilder = SinusoidalPositionalEncodingBuilder;

/// Implements sinusoidal positional encoding.
///
/// For more details see the paper "Attention Is All You Need"
/// <https://arxiv.org/abs/1706.03762>.
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct SinusoidalPositionalEncoding {
    #[param]
    sigmas: Param<Array>,

    /// multiplicative scale for the embeddings.  Default is `sqrt(2/dimensions)`
    pub scale: f32,

    /// if `true` embed using `[cos(x), sin(x)]` instead of the reverse
    pub cosine_first: bool,
}

impl Sinpe {
    /// Default value for `cosine_first` field.
    pub const DEFAULT_COSINE_FIRST: bool = false;

    /// Default value for min frequency.
    pub const DEFAULT_MIN_FREQUENCY: f32 = 0.0001;

    /// Default value for max frequency.
    pub const DEFAULT_MAX_FREQUENCY: f32 = 1.0;

    /// Default value for full turns.
    pub const DEFAULT_FULL_TURNS: bool = false;
}

/// Builder for [`SinusoidalPositionalEncoding`].
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_sinpe,
    err = Exception,
)]
pub struct SinusoidalPositionalEncodingBuilder {
    dimensions: i32,

    #[builder(optional, default = Sinpe::DEFAULT_MIN_FREQUENCY)]
    min_frequency: f32,

    #[builder(optional, default = Sinpe::DEFAULT_MAX_FREQUENCY)]
    max_frequency: f32,

    #[builder(optional, default = None)]
    scale: Option<f32>,

    #[builder(optional, default = Sinpe::DEFAULT_COSINE_FIRST)]
    cosine_first: bool,

    #[builder(optional, default = Sinpe::DEFAULT_FULL_TURNS)]
    full_turns: bool,
}

fn build_sinpe(builder: SinpeBuilder) -> Result<SinusoidalPositionalEncoding, Exception> {
    let SinpeBuilder {
        dimensions,
        min_frequency,
        max_frequency,
        scale,
        cosine_first,
        full_turns,
    } = builder;

    let half_dim = dimensions / 2;
    let one_zero = array!(1.0)
        .subtract(Array::from_iter(0..half_dim, &[half_dim]).divide(array!(half_dim - 1))?)?;
    let min_frequency = log(array!(min_frequency))?;
    let max_frequency = log(array!(max_frequency))?;

    // SAFETY: max_frequency and min_frequency are scalars and operations with scalars won't throw
    let mut sigmas = exp(&one_zero * (&max_frequency - &min_frequency) + &min_frequency)?;
    if full_turns {
        // SAFETY: scalar array operation won't throw
        sigmas *= array!(2.0 * std::f32::consts::PI);
    }

    let scale = scale.unwrap_or_else(|| (2.0 / dimensions as f32).sqrt());

    Ok(SinusoidalPositionalEncoding {
        sigmas: Param::new(sigmas),
        scale,
        cosine_first,
    })
}

impl Module<&Array> for Sinpe {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let mut y = x
            .expand_dims_axes(&[-1])
            .and_then(|x| x.multiply(&self.sigmas))?;

        let cosy = y.cos()?;
        let siny = y.sin()?;

        if self.cosine_first {
            y = concatenate_axis(&[cosy, siny], -1)?;
        } else {
            y = concatenate_axis(&[siny, cosy], -1)?;
        }

        if self.scale != 1.0 {
            // SAFETY: multiplication with scalar won't throw
            y *= self.scale;
        }

        Ok(y)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct AlibiKey {
    q_seq_len: i32,
    k_seq_len: i32,
    num_heads: i32,
    offset: i32,
    dtype: Dtype,
}

thread_local! {
    static ALIBI_CACHE: RefCell<HashMap<AlibiKey, Array>> = RefCell::new(HashMap::new());
}

/// Attention with Linear Biases
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Alibi;

impl Alibi {
    fn slope(num_heads: i32) -> Result<Array, Exception> {
        let x = 2.0_f32.powi(8).powf(1.0 / num_heads as f32);
        array!(x)
            .power(&arange::<_, f32>(1, num_heads + 1, None)?)?
            .expand_dims_axes(&[-1, -2])
    }

    fn matrix(key: AlibiKey) -> Result<Array, Exception> {
        if let Some(value) = ALIBI_CACHE.with(|cache| cache.borrow().get(&key).cloned()) {
            return Ok(value);
        }

        let x1 = arange::<_, f32>(key.offset, key.q_seq_len, None)?;
        let x2 = arange::<_, f32>(0, key.k_seq_len, None)?;
        let distance_matrix = x1
            .try_index((.., NewAxis))?
            .subtract(x2.try_index((NewAxis, ..))?)?
            .expand_dims_axes(&[0, 1])?
            .abs()?
            .negative()?;

        let slope = Self::slope(key.num_heads)?;
        let mask = distance_matrix.multiply(&slope)?.as_dtype(key.dtype)?;

        ALIBI_CACHE.with(|cache| {
            cache.borrow_mut().insert(key, mask.clone());
        });

        Ok(mask)
    }
}

generate_builder! {
    /// Input for the [`Alibi`] module.
    #[derive(Debug, Clone, Buildable)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct AlibiInput<'a> {
        /// The attention scores.
        pub attention_scores: &'a Array,

        /// Offset
        #[builder(optional, default = AlibiInput::DEFAULT_OFFSET)]
        pub offset: i32,

        /// Mask
        #[builder(optional, default = None)]
        pub mask: Option<&'a Array>,
    }
}

impl AlibiInput<'_> {
    /// Default value for `offset` field.
    pub const DEFAULT_OFFSET: i32 = 0;
}

impl<'a> From<&'a Array> for AlibiInput<'a> {
    fn from(attention_scores: &'a Array) -> Self {
        AlibiInput {
            attention_scores,
            offset: Self::DEFAULT_OFFSET,
            mask: None,
        }
    }
}

impl<'a> From<(&'a Array,)> for AlibiInput<'a> {
    fn from((attention_scores,): (&'a Array,)) -> Self {
        AlibiInput {
            attention_scores,
            offset: Self::DEFAULT_OFFSET,
            mask: None,
        }
    }
}

impl<'a> From<(&'a Array, i32)> for AlibiInput<'a> {
    fn from((attention_scores, offset): (&'a Array, i32)) -> Self {
        AlibiInput {
            attention_scores,
            offset,
            mask: None,
        }
    }
}

impl<'a> From<(&'a Array, i32, &'a Array)> for AlibiInput<'a> {
    fn from((attention_scores, offset, mask): (&'a Array, i32, &'a Array)) -> Self {
        AlibiInput {
            attention_scores,
            offset,
            mask: Some(mask),
        }
    }
}

impl<'a> From<(&'a Array, i32, Option<&'a Array>)> for AlibiInput<'a> {
    fn from((attention_scores, offset, mask): (&'a Array, i32, Option<&'a Array>)) -> Self {
        AlibiInput {
            attention_scores,
            offset,
            mask,
        }
    }
}

impl<'a, Input> Module<Input> for Alibi
where
    Input: Into<AlibiInput<'a>>,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: Input) -> Result<Self::Output, Self::Error> {
        let AlibiInput {
            attention_scores,
            offset,
            mask,
        } = input.into();

        let key = AlibiKey {
            q_seq_len: attention_scores.dim(-2) + offset,
            k_seq_len: attention_scores.dim(-1),
            num_heads: attention_scores.dim(1),
            offset,
            dtype: attention_scores.dtype(),
        };

        let mut alibi_mask = Self::matrix(key)?;
        if let Some(mask) = mask {
            alibi_mask = alibi_mask.add(mask)?;
        }

        attention_scores.add(alibi_mask)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

#[allow(clippy::excessive_precision)]
#[cfg(test)]
mod tests {
    use crate::{module::Module, nn::AlibiInput, random::uniform, Dtype};
    use float_eq::assert_float_eq;

    use crate::nn::Rope;

    // The unit test below is adapted from the swift binding at:
    // mlx-swift/Tests/MLXTests/IntegrationTests.swift
    #[test]
    fn test_rope() {
        crate::random::seed(71).unwrap();
        let a = uniform::<_, f32>(0, 1, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item::<f32>(),
            0.5082664489746094,
            abs <= 0.010165328979492188
        );
        assert_float_eq!(
            a.sum(None).unwrap().item::<f32>(),
            130.1162109375,
            abs <= 2.60232421875
        );

        let mut rope = Rope::new(8);
        let result = rope.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.4562537670135498,
            abs <= 0.009125075340270997
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            116.80096435546875,
            abs <= 2.3360192871093752
        );
    }

    // The unit test below is adapted from the swift binding at:
    // mlx-swift/Tests/MLXTests/IntegrationTests.swift
    #[test]
    fn test_sinpe() {
        crate::random::seed(226).unwrap();
        let a = uniform::<_, f32>(0, 1, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None).unwrap().item::<f32>(),
            0.5026599168777466,
            abs <= 0.010053198337554931
        );
        assert_float_eq!(
            a.sum(None).unwrap().item::<f32>(),
            128.68093872070312,
            abs <= 2.5736187744140624
        );

        let mut sinpe = crate::nn::Sinpe::new(8).unwrap();
        let result = sinpe.forward(&a).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16, 8]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.2705308198928833,
            abs <= 0.005410616397857666
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            554.047119140625,
            abs <= 11.0809423828125
        );
    }

    // The unit test below is adapted from the python binding at:
    // mlx/python/tests/test_nn.py
    #[test]
    fn test_alibi() {
        let mut alibi = crate::nn::Alibi;
        let shape = [1, 8, 20, 20];
        let x = uniform::<_, f32>(0, 1, &shape, None).unwrap();
        let input = AlibiInput::from(&x);
        let y = alibi.forward(input).unwrap();
        assert_eq!(y.shape(), shape);
        assert_eq!(y.dtype(), Dtype::Float32);

        let x2 = x.as_dtype(Dtype::Float16).unwrap();
        let input = AlibiInput::from(&x2);
        let y = alibi.forward(input).unwrap();
        assert_eq!(y.dtype(), Dtype::Float16);
    }
}
