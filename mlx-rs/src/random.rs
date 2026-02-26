//! Collection of functions related to random number generation

use crate::ops::indexing::TryIndexOp;
use crate::utils::guard::Guarded;
use crate::utils::IntoOption;
use crate::{error::Result, Array, ArrayElement, Stream};
use mach_sys::mach_time;
use mlx_internal_macros::{default_device, generate_macro};
use parking_lot::Mutex;
use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::OnceLock;

static GLOBAL_STATE: OnceLock<Mutex<RandomState>> = OnceLock::new();

thread_local! {
    static TASK_LOCAL_STATE: RefCell<Option<RandomState>> = const { RefCell::new(None) };
}

/// Random state for reproducible random number generation.
///
/// This struct holds the PRNG state and can be used with compiled functions
/// to properly track random state across JIT compilation boundaries.
///
/// # Compilation Support
///
/// `RandomState` implements `Updatable`, making it compatible with
/// `compile_with_state`. This is the Rust equivalent of Python's
/// `@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)`.
///
/// # Example
///
/// ```rust,no_run
/// use mlx_rs::random::RandomState;
/// use mlx_rs::transforms::compile::compile_with_state;
/// use mlx_rs::random::categorical;
/// use mlx_rs::Array;
///
/// let mut state = RandomState::with_seed(42).unwrap();
/// let logits = Array::zeros::<f32>(&[1, 10]).unwrap();
/// let mut compiled = compile_with_state(
///     |state: &mut RandomState, x: &Array| {
///         let key = state.next_key()?;
///         categorical(x, None, None, Some(&key))
///     },
///     None
/// );
/// let result = compiled(&mut state, &logits).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RandomState {
    state: Array,
}

impl RandomState {
    /// Create a new random state with a time-based seed.
    pub fn new() -> Result<Self> {
        let now = unsafe { mach_time::mach_approximate_time() };
        Ok(Self { state: key(now)? })
    }

    /// Create a new random state from a specific seed.
    ///
    /// Use this for reproducible random number generation.
    pub fn with_seed(seed: u64) -> Result<Self> {
        Ok(Self { state: key(seed)? })
    }

    /// Create a random state from an existing key array.
    ///
    /// The key must be a valid PRNG key (typically created via `random::key()`).
    pub fn from_key(key: Array) -> Self {
        Self { state: key }
    }

    /// Get the next random key, advancing the state.
    ///
    /// This splits the current state into two keys: one becomes the new state,
    /// and the other is returned for use in random operations.
    pub fn next_key(&mut self) -> Result<Array> {
        let next = split(&self.state, 2)?;
        self.state = next.0;
        Ok(next.1)
    }

    /// Internal method for backward compatibility.
    fn next(&mut self) -> Result<Array> {
        self.next_key()
    }

    /// Reseed the random state.
    pub fn seed(&mut self, seed: u64) -> Result<()> {
        self.state = key(seed)?;
        Ok(())
    }

    /// Get a reference to the underlying state array.
    ///
    /// This is useful for inspection or manual state management.
    pub fn as_array(&self) -> &Array {
        &self.state
    }

    /// Get a mutable reference to the underlying state array.
    ///
    /// # Note
    ///
    /// Modifying the state array directly may break the PRNG invariants.
    /// Prefer using `seed()` or `next_key()` instead.
    pub fn as_array_mut(&mut self) -> &mut Array {
        &mut self.state
    }
}

impl Default for RandomState {
    /// Creates a new `RandomState` with a time-based seed.
    ///
    /// # Panics
    ///
    /// Panics if the underlying PRNG key creation fails, which should not
    /// occur under normal conditions.
    fn default() -> Self {
        Self::new().expect("Failed to create default RandomState")
    }
}

impl crate::utils::Updatable for RandomState {
    fn updatable_states_len(&self) -> usize {
        1
    }

    fn updatable_states(&self) -> impl IntoIterator<Item = &Array> {
        std::iter::once(&self.state)
    }

    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array> {
        std::iter::once(&mut self.state)
    }
}

fn global_state() -> &'static Mutex<RandomState> {
    GLOBAL_STATE.get_or_init(|| Mutex::new(RandomState::new().unwrap()))
}

/// Returns a key from the task-local state if it exists, otherwise
/// returns `None`
fn resolve_task_local_key() -> Option<Result<Array>> {
    TASK_LOCAL_STATE.with_borrow_mut(|state| state.as_mut().map(|s| s.next()))
}

fn resolve_global_key() -> Result<Array> {
    let mut state = global_state().lock();
    state.next()
}

/// Use given key or generate a new one if `None`.
fn resolve<'a>(key: impl Into<Option<&'a Array>>) -> Result<Cow<'a, Array>> {
    key.into().map_or_else(
        || {
            resolve_task_local_key()
                .unwrap_or_else(resolve_global_key)
                .map(Cow::Owned)
        },
        |k| Ok(Cow::Borrowed(k)),
    )
}

/// Use the given random state for the scope of `f`
pub fn with_random_state<F, T>(state: RandomState, f: F) -> T
where
    F: FnOnce() -> T,
{
    let prev_state = TASK_LOCAL_STATE.with_borrow_mut(|s| s.replace(state));

    let result = f();

    TASK_LOCAL_STATE.with_borrow_mut(|s| {
        *s = prev_state;
    });

    result
}

/// Seed the random number generator.
pub fn seed(seed: u64) -> Result<()> {
    let mut state = global_state().lock();
    state.seed(seed)
}

/// Get a PRNG key from a seed.
///
/// Return a value that can be used as a PRNG key.  All ``random::*``
/// functions take an optional key -- this will let you control the
/// random number generation.
pub fn key(seed: u64) -> Result<Array> {
    Array::try_from_op(|res| unsafe { mlx_sys::mlx_random_key(res, seed) })
}

/// Split a PRNG key into two keys and return a tuple.
#[default_device]
pub fn split_device(
    key: impl AsRef<Array>,
    num: i32,
    stream: impl AsRef<Stream>,
) -> Result<(Array, Array)> {
    let keys = Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_random_split_num(res, key.as_ref().as_ptr(), num, stream.as_ref().as_ptr())
    })?;

    Ok((keys.try_index(0)?, keys.try_index(1)?))
}

/// Generate uniformly distributed random numbers.
/// The values are sampled uniformly in the half-open interval `[lower, upper)`.
/// The lower and upper bound can be scalars or arrays and must be broadcastable to `shape`.
///
/// # Params
///
/// - `lower`: Lower bound of the distribution.
/// - `upper`: Upper bound of the distribution.
/// - `shape` (optional): Shape of the output. Default is `&[]`.
/// - `key` (optional): A PRNG key.
///
/// ```rust
/// let key = mlx_rs::random::key(0).unwrap();
///
/// // create an array of shape `[50]` type f32 values in the range [0, 10)
/// let array = mlx_rs::random::uniform::<_, f32>(0, 10, &[50], &key);
///
/// // same, but in range [0.5, 1)
/// let array = mlx_rs::random::uniform::<_, f32>(0.5f32, 1f32, &[50], &key);
/// ```
#[generate_macro(customize(root = "$crate::random"))]
#[default_device]
pub fn uniform_device<'a, E: Into<Array>, T: ArrayElement>(
    lower: E,
    upper: E,
    #[optional] shape: impl IntoOption<&'a [i32]>,
    #[optional] key: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let lb: Array = lower.into();
    let ub: Array = upper.into();
    let shape = shape.into_option().unwrap_or(&[]);
    let key = resolve(key)?;

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_random_uniform(
            res,
            lb.as_ptr(),
            ub.as_ptr(),
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            key.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Generate normally distributed random numbers.
///
/// Generate an array of random numbers using the optional shape. The result
/// will be of the given `T`. `T` must be a floating point type.
///
/// # Params
///
///  - shape: shape of the output, if `None` a single value is returned
///  - loc: mean of the distribution, default is `0.0`
///  - scale: standard deviation of the distribution, default is `1.0`
///  - key: PRNG key
///
/// # Example
///
/// ```rust
/// let key = mlx_rs::random::key(0).unwrap();
///
/// // generate a single f32 with normal distribution
/// let value = mlx_rs::random::normal::<f32>(None, None, None, &key).unwrap().item::<f32>();
///
/// // generate an array of f32 with normal distribution in shape [10, 5]
/// let array = mlx_rs::random::normal::<f32>(&[10, 5], None, None, &key);
/// ```
#[generate_macro(customize(root = "$crate::random"))]
#[default_device]
pub fn normal_device<'a, T: ArrayElement>(
    #[optional] shape: impl IntoOption<&'a [i32]>,
    #[optional] loc: impl Into<Option<f32>>,
    #[optional] scale: impl Into<Option<f32>>,
    #[optional] key: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let shape = shape.into_option().unwrap_or(&[]);
    let key = resolve(key)?;

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_random_normal(
            res,
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            loc.into().unwrap_or(0.0),
            scale.into().unwrap_or(1.0),
            key.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Generate jointly-normal random samples given a mean and covariance.
///
/// The matrix `covariance` must be positive semi-definite. The behavior is
/// undefined if it is not.  The only supported output type is f32.
///
/// # Params
/// - `mean`: array of shape `[..., n]`, the mean of the distribution.
/// - `covariance`: array  of shape `[..., n, n]`, the covariance matrix of the distribution. The batch shape `...` must be broadcast-compatible with that of `mean`.
/// - `shape`: The output shape must be broadcast-compatible with `&mean.shape[..mean.shape.len()-1]` and `&covariance.shape[..covariance.shape.len()-2]`. If empty, the result shape is determined by broadcasting the batch shapes of `mean` and `covariance`.
/// - `key`: PRNG key.
#[generate_macro(customize(root = "$crate::random"))]
#[default_device(device = "cpu")] // TODO: not supported on GPU yet
pub fn multivariate_normal_device<'a, T: ArrayElement>(
    mean: impl AsRef<Array>,
    covariance: impl AsRef<Array>,
    #[optional] shape: impl IntoOption<&'a [i32]>,
    #[optional] key: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let shape = shape.into_option().unwrap_or(&[]);
    let key = resolve(key)?;

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_random_multivariate_normal(
            res,
            mean.as_ref().as_ptr(),
            covariance.as_ref().as_ptr(),
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            key.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Generate random integers from the given interval (`lower:` and `upper:`).
///
/// The values are sampled with equal probability from the integers in
/// half-open interval `[lb, ub)`. The lower and upper bound can be
/// scalars or arrays and must be roadcastable to `shape`.
///
/// ```rust
/// use mlx_rs::{array, random};
///
/// let key = random::key(0).unwrap();
///
/// // generate an array of Int values, one in the range [0, 20) and one in the range [10, 100)
/// let array = random::randint::<_, i32>(array!([0, 20]), array!([10, 100]), None, &key);
/// ```
#[generate_macro(customize(root = "$crate::random"))]
#[default_device]
pub fn randint_device<'a, E: Into<Array>, T: ArrayElement>(
    lower: E,
    upper: E,
    #[optional] shape: impl IntoOption<&'a [i32]>,
    #[optional] key: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let lb: Array = lower.into();
    let ub: Array = upper.into();
    let shape = shape.into_option().unwrap_or(lb.shape());
    let key = resolve(key)?;

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_random_randint(
            res,
            lb.as_ptr(),
            ub.as_ptr(),
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            key.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Generate Bernoulli random values with a given `p` value.
///
/// The values are sampled from the bernoulli distribution with parameter
/// `p`. The parameter `p` must have a floating point type and
/// must be broadcastable to `shape`.
///
/// ```rust
/// use mlx_rs::{array, Array, random};
///
/// let key = random::key(0).unwrap();
///
/// // generate a single random Bool with p = 0.8
/// let p: Array = 0.8.into();
/// let value = random::bernoulli(&p, None, &key);
///
/// // generate an array of shape [50, 2] of random Bool with p = 0.8
/// let array = random::bernoulli(&p, &[50, 2], &key);
///
/// // generate an array of [3] Bool with the given p values
/// let array = random::bernoulli(&array!([0.1, 0.5, 0.8]), None, &key);
/// ```
#[generate_macro(customize(root = "$crate::random"))]
#[default_device]
pub fn bernoulli_device<'a>(
    #[optional] p: impl Into<Option<&'a Array>>,
    #[optional] shape: impl IntoOption<&'a [i32]>,
    #[optional] key: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let default_array = Array::from_f32(0.5);
    let p = p.into().unwrap_or(&default_array);

    let shape = shape.into_option().unwrap_or(p.shape());
    let key = resolve(key)?;

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_random_bernoulli(
            res,
            p.as_ptr(),
            shape.as_ptr(),
            shape.len(),
            key.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Generate values from a truncated normal distribution between `low` and `high`.
///
/// The values are sampled from the truncated normal distribution
/// on the domain `(lower, upper)`. The bounds `lower` and `upper`
/// can be scalars or arrays and must be broadcastable to `shape`.
///
/// ```rust
/// use mlx_rs::{array, random};
///
/// let key = random::key(0).unwrap();
///
/// // generate an array of two Float values, one in the range 0 ..< 10
/// // and one in the range 10 ..< 100
/// let value = random::truncated_normal::<_, f32>(array!([0, 10]), array!([10, 100]), None, &key);
/// ```
#[generate_macro(customize(root = "$crate::random"))]
#[default_device]
pub fn truncated_normal_device<'a, E: Into<Array>, T: ArrayElement>(
    lower: E,
    upper: E,
    #[optional] shape: impl IntoOption<&'a [i32]>,
    #[optional] key: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let lb: Array = lower.into();
    let ub: Array = upper.into();
    let shape = shape.into_option().unwrap_or(lb.shape());
    let key = resolve(key)?;

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_random_truncated_normal(
            res,
            lb.as_ptr(),
            ub.as_ptr(),
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            key.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Sample from the standard Gumbel distribution.
///
/// The values are sampled from a standard Gumbel distribution
/// which CDF `exp(-exp(-x))`.
///
/// ```rust
/// let key = mlx_rs::random::key(0).unwrap();
///
/// // generate a single Float with Gumbel distribution
/// let value = mlx_rs::random::gumbel::<f32>(None, &key).unwrap().item::<f32>();
///
/// // generate an array of Float with Gumbel distribution in shape [10, 5]
/// let array = mlx_rs::random::gumbel::<f32>(&[10, 5], &key);
/// ```
#[generate_macro(customize(root = "$crate::random"))]
#[default_device]
pub fn gumbel_device<'a, T: ArrayElement>(
    #[optional] shape: impl IntoOption<&'a [i32]>,
    #[optional] key: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let shape = shape.into_option().unwrap_or(&[]);
    let key = resolve(key)?;

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_random_gumbel(
            res,
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            key.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Shape or count for the categorical distribution.
#[derive(Debug, Clone, Copy)]
pub enum ShapeOrCount<'a> {
    /// Shape
    Shape(&'a [i32]),

    /// Count
    Count(i32),
}

/// Sample from a categorical distribution.
///
/// The values are sampled from the categorical distribution specified by
/// the unnormalized values in `logits`.   If the `shape` is not specified
/// the result shape will be the same shape as `logits` with the `axis`
/// dimension removed.
///
/// /// # Params
/// # Params
///
/// - `logits`: The *unnormalized* categorical distribution(s).
/// - `axis`(optional): The axis which specifies the distribution. Default is `-1`.
/// - `shape_or_count`(optional):
/// - - `Shape`: The shape of the output. This must be broadcast compatible with `logits.shape` with the `axis` dimension removed.
/// - - `Count`: The number of samples to draw from each of the categorical distributions in `logits`. The output will have the number of samples in the last dimension.
/// - `key` (optional): A PRNG key.
///
/// # Example
///
/// ```rust
/// let key = mlx_rs::random::key(0).unwrap();
///
/// let logits = mlx_rs::Array::zeros::<u32>(&[5, 20]).unwrap();
///
/// // produces Array of u32 shape &[5]
/// let result = mlx_rs::random::categorical(&logits, None, None, &key);
/// ```
#[generate_macro(customize(root = "$crate::random"))]
#[default_device]
pub fn categorical_device<'a>(
    logits: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] shape_or_count: impl Into<Option<ShapeOrCount<'a>>>,
    #[optional] key: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let axis = axis.into().unwrap_or(-1);
    let key = resolve(key)?;

    match shape_or_count.into() {
        Some(ShapeOrCount::Shape(shape)) => Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_random_categorical_shape(
                res,
                logits.as_ref().as_ptr(),
                axis,
                shape.as_ptr(),
                shape.len(),
                key.as_ptr(),
                stream.as_ref().as_ptr(),
            )
        }),
        Some(ShapeOrCount::Count(num_samples)) => Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_random_categorical_num_samples(
                res,
                logits.as_ref().as_ptr(),
                axis,
                num_samples,
                key.as_ptr(),
                stream.as_ref().as_ptr(),
            )
        }),
        None => Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_random_categorical(
                res,
                logits.as_ref().as_ptr(),
                axis,
                key.as_ptr(),
                stream.as_ref().as_ptr(),
            )
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{array, assert_array_eq};
    use float_eq::{assert_float_eq, float_eq};

    #[test]
    fn test_global_rng() {
        seed(3).unwrap();
        let a = uniform::<_, f32>(0, 1, None, None).unwrap();
        let b = uniform::<_, f32>(0, 1, None, None).unwrap();

        seed(3).unwrap();
        let x = uniform::<_, f32>(0, 1, None, None).unwrap();
        let y = uniform::<_, f32>(0, 1, None, None).unwrap();

        assert_array_eq!(a, x, 0.01);
        assert_array_eq!(b, y, 0.01);
    }

    #[test]
    fn test_key() {
        let k1 = key(0).unwrap();
        let k2 = key(0).unwrap();
        assert!(k1 == k2);

        let k2 = key(1).unwrap();
        assert!(k1 != k2);
    }

    #[test]
    fn test_split() {
        let key = key(0).unwrap();

        let (k1, k2) = split(&key, 2).unwrap();
        assert!(k1 != k2);

        let (r1, r2) = split(&key, 2).unwrap();
        assert!(r1 == k1);
        assert!(r2 == k2);
    }

    #[test]
    fn test_uniform_no_seed() {
        let value = uniform::<_, f32>(0, 10, &[3], None).unwrap();
        assert_eq!(value.shape(), &[3]);
    }

    #[test]
    fn test_uniform_single() {
        let key = key(0).unwrap();
        let value = uniform::<_, f32>(0, 10, None, Some(&key)).unwrap();
        float_eq!(value.item::<f32>(), 4.18, abs <= 0.01);
    }

    #[test]
    fn test_uniform_multiple() {
        let key = key(0).unwrap();
        let value = uniform::<_, f32>(0, 10, &[3], Some(&key)).unwrap();
        let expected = Array::from_slice(&[9.65, 3.14, 6.33], &[3]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_uniform_multiple_array() {
        let key = key(0).unwrap();
        let value = uniform::<_, f32>(&[0, 10], &[10, 100], &[2], Some(&key)).unwrap();
        let expected = Array::from_slice(&[2.16, 82.37], &[2]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_uniform_non_float() {
        let key = key(0).unwrap();
        let value = uniform::<_, i32>(&[0, 10], &[10, 100], &[2], Some(&key));
        assert!(value.is_err());
    }

    #[test]
    fn test_normal() {
        let key = key(0).unwrap();
        let value = normal::<f32>(None, None, None, &key).unwrap();
        float_eq!(value.item::<f32>(), -0.20, abs <= 0.01);
    }

    #[test]
    fn test_normal_non_float() {
        let key = key(0).unwrap();
        let value = normal::<i32>(None, None, None, &key);
        assert!(value.is_err());
    }

    #[test]
    fn test_multivariate_normal() {
        let key = key(0).unwrap();
        let mean = Array::from_slice(&[0.0, 0.0], &[2]);
        let covariance = Array::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

        let a = multivariate_normal::<f32>(&mean, &covariance, &[3], &key).unwrap();
        assert!(a.shape() == [3, 2]);
    }

    #[test]
    fn test_randint_single() {
        let key = key(0).unwrap();
        let value = randint::<_, i32>(0, 100, None, Some(&key)).unwrap();
        assert_eq!(value.item::<i32>(), 41);
    }

    #[test]
    fn test_randint_multiple() {
        let key = key(0).unwrap();
        let value =
            randint::<_, i32>(array!([0, 10]), array!([10, 100]), None, Some(&key)).unwrap();
        let expected = Array::from_slice(&[2, 82], &[2]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_randint_non_int() {
        let key = key(0).unwrap();
        let value = randint::<_, f32>(array!([0, 10]), array!([10, 100]), None, Some(&key));
        assert!(value.is_err());
    }

    #[test]
    fn test_bernoulli_single() {
        let key = key(0).unwrap();
        let value = bernoulli(None, None, &key).unwrap();
        assert!(value.item::<bool>());
    }

    #[test]
    fn test_bernoulli_multiple() {
        let key = key(0).unwrap();
        let value = bernoulli(None, &[4], &key).unwrap();
        let expected = Array::from_slice(&[false, true, false, true], &[4]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_bernoulli_p() {
        let key = key(0).unwrap();
        let p: Array = 0.8.into();
        let value = bernoulli(&p, &[4], &key).unwrap();
        let expected = Array::from_slice(&[false, true, true, true], &[4]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_bernoulli_p_array() {
        let key = key(0).unwrap();
        let value = bernoulli(&array!([0.1, 0.5, 0.8]), None, &key).unwrap();
        let expected = Array::from_slice(&[false, true, true], &[3]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_truncated_normal_single() {
        let key = key(0).unwrap();
        let value = truncated_normal::<_, f32>(0, 10, None, &key).unwrap();
        assert_array_eq!(value, Array::from_f32(0.55), 0.01);
    }

    #[test]
    fn test_truncated_normal_multiple() {
        let key = key(0).unwrap();
        let value = truncated_normal::<_, f32>(0.0, 0.5, &[3], &key).unwrap();
        let expected = Array::from_slice(&[0.48, 0.15, 0.30], &[3]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_truncated_normal_multiple_array() {
        let key = key(0).unwrap();
        let value =
            truncated_normal::<_, f32>(array!([0.0, 0.5]), array!([0.5, 1.0]), None, &key).unwrap();
        let expected = Array::from_slice(&[0.10, 0.88], &[2]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_gumbel() {
        let key = key(0).unwrap();
        let value = gumbel::<f32>(None, &key).unwrap();
        assert_array_eq!(value, Array::from_f32(0.13), 0.01);
    }

    #[test]
    fn test_logits() {
        let key = key(0).unwrap();
        let logits = Array::zeros::<u32>(&[5, 20]).unwrap();
        let result = categorical(&logits, None, None, &key).unwrap();

        assert_eq!(result.shape(), [5]);

        let expected = Array::from_slice(&[1, 1, 17, 17, 17], &[5]);
        assert_array_eq!(result, expected, 0.01);
    }

    #[test]
    fn test_logits_count() {
        let key = key(0).unwrap();
        let logits = Array::zeros::<u32>(&[5, 20]).unwrap();
        let result = categorical(&logits, None, ShapeOrCount::Count(2), &key).unwrap();

        assert_eq!(result.shape(), [5, 2]);

        let expected = Array::from_slice(&[16, 3, 14, 10, 17, 7, 6, 8, 12, 8], &[5, 2]);
        assert_array_eq!(result, expected, 0.01);
    }

    #[test]
    fn test_random_state_new() {
        let state = RandomState::new().unwrap();
        assert_eq!(state.as_array().shape(), &[2]);
    }

    #[test]
    fn test_random_state_with_seed_deterministic() {
        let s1 = RandomState::with_seed(42).unwrap();
        let s2 = RandomState::with_seed(42).unwrap();
        assert!(s1.as_array() == s2.as_array());
    }

    #[test]
    fn test_random_state_next_key_advances() {
        let mut state = RandomState::with_seed(0).unwrap();
        let k1 = state.next_key().unwrap();
        let k2 = state.next_key().unwrap();
        assert!(k1 != k2);
    }

    #[test]
    fn test_random_state_from_key_roundtrip() {
        let original = RandomState::with_seed(99).unwrap();
        let arr = original.as_array().clone();
        let restored = RandomState::from_key(arr);
        assert!(original.as_array() == restored.as_array());
    }

    #[test]
    fn test_random_state_updatable() {
        use crate::utils::Updatable;
        let state = RandomState::with_seed(0).unwrap();
        assert_eq!(state.updatable_states_len(), 1);
        assert_eq!(state.updatable_states().into_iter().count(), 1);
    }

    #[test]
    fn test_random_state_default() {
        let state = RandomState::default();
        assert_eq!(state.as_array().shape(), &[2]);
    }

    #[test]
    fn test_random_seed_same() {
        // Same random seed should produce the same results
        let seed = 23;
        let mut results = Vec::new();
        let f = || {
            uniform::<_, f32>(0.0, 1.0, &[10, 10], None)?
                .sum(None)?
                .try_item::<f32>()
        };
        for _ in 0..10 {
            let mut state = RandomState::new().unwrap();
            state.seed(seed).unwrap();
            let result = with_random_state(state, f).unwrap();
            results.push(result);
        }

        // Check that all results are the same within a small tolerance
        let first = results[0];
        for result in &results[1..] {
            assert_float_eq!(
                first,
                *result,
                abs <= 0.01,
                "Results should be equal for the same seed"
            );
        }
    }
}
