//! Fast implementations of commonly used multi-op functions.

use std::ffi::{CStr, CString};

use crate::error::{Exception, Result};
use crate::utils::guard::Guarded;
use crate::utils::{IntoOption, VectorArray};
use crate::{Array, Dtype, Stream};
use mlx_internal_macros::{default_device, generate_macro};

/// Optimized implementation of `NN.RoPE`.
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rope_device<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: i32,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rope(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset,
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or_else(|| mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Optimized implementation of `NN.RoPE` with dynamic (array) offset.
///
/// This variant allows specifying the offset as an array, enabling different
/// offsets for different positions in the input.
///
/// # Params
///
/// - `array`: Input array
/// - `dimensions`: The feature dimensions to apply rope to
/// - `traditional`: If true, uses the traditional rope implementation
/// - `base`: The base used to compute angular frequency for each dimension
/// - `scale`: The scale to apply to the positions
/// - `offset`: An array of position offsets
/// - `freqs`: Optional precomputed frequencies
/// - `stream`: Stream to evaluate on
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rope_dynamic_device<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: impl AsRef<Array>,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rope_dynamic(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset.as_ref().as_ptr(),
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or_else(|| mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

const DEFAULT_MASK_MODE: &CStr = c"";
const CAUSAL_MASK_MODE: &CStr = c"causal";

/// Mask modes for scaled dot product attention.
#[derive(Debug)]
pub enum ScaledDotProductAttentionMask<'a> {
    /// A single mask array
    Array(&'a Array),

    /// Causal masking (no explicit mask array needed)
    Causal,
}

impl<'a> From<&'a Array> for ScaledDotProductAttentionMask<'a> {
    fn from(mask: &'a Array) -> Self {
        ScaledDotProductAttentionMask::Array(mask)
    }
}

impl<'a> IntoOption<ScaledDotProductAttentionMask<'a>> for &'a Array {
    fn into_option(self) -> Option<ScaledDotProductAttentionMask<'a>> {
        Some(ScaledDotProductAttentionMask::Array(self))
    }
}

impl ScaledDotProductAttentionMask<'_> {
    fn as_mode_and_mask(&self) -> (&'static CStr, mlx_sys::mlx_array) {
        match self {
            ScaledDotProductAttentionMask::Array(mask) => (DEFAULT_MASK_MODE, mask.as_ptr()),
            ScaledDotProductAttentionMask::Causal => {
                (CAUSAL_MASK_MODE, unsafe { mlx_sys::mlx_array_new() })
            }
        }
    }
}

/// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
///
/// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
///
/// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
///
/// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
///
/// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn scaled_dot_product_attention_device<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    #[optional] mask: impl IntoOption<ScaledDotProductAttentionMask<'a>>,
    #[optional] sinks: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let (mask_mode, mask_arr) = mask.into_option().map_or_else(
        || (DEFAULT_MASK_MODE, unsafe { mlx_sys::mlx_array_new() }),
        |m| m.as_mode_and_mask(),
    );

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_scaled_dot_product_attention(
            res,
            queries.as_ref().as_ptr(),
            keys.as_ref().as_ptr(),
            values.as_ref().as_ptr(),
            scale,
            mask_mode.as_ptr(),
            mask_arr,
            sinks
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or_else(|| mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Root Mean Square normalization (RMS norm).
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///   with the same size as the last axis of `x`. If `None`, no scaling is applied.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rms_norm_device<'a>(
    #[named] x: impl AsRef<Array>,
    #[optional] weight: impl Into<Option<&'a Array>>,
    #[named] eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rms_norm(
            res,
            x.as_ref().as_ptr(),
            weight
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or_else(|| mlx_sys::mlx_array_new()),
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Layer normalization.
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///   with the same size as the last axis of `x`.  If not given no scaling will occur.
/// - bias: An additive offset to be added to the result. The `bias` should be one-dimensional
///   with the same size as the last axis of `x`.  It not given no offset will occur.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn layer_norm_device<'a>(
    #[named] x: impl AsRef<Array>,
    #[optional] weight: impl Into<Option<&'a Array>>,
    #[optional] bias: impl Into<Option<&'a Array>>,
    #[named] eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_layer_norm(
            res,
            x.as_ref().as_ptr(),
            weight
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or_else(|| mlx_sys::mlx_array_new()),
            bias.into()
                .map(|a| a.as_ptr())
                .unwrap_or_else(|| mlx_sys::mlx_array_new()),
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

/// A template argument for a [`MetalKernel`].
///
/// Forwarded into the kernel source as `template <typename InT, int Dk, ...>` so
/// the same kernel object can be reused with different element types or
/// compile-time-known integers.
#[derive(Debug, Clone)]
pub enum MetalKernelTemplateArg {
    /// A `mlx_dtype` template parameter (renders as the corresponding Metal type).
    Dtype(Dtype),
    /// An `int` template parameter.
    Int(i32),
    /// A `bool` template parameter.
    Bool(bool),
}

impl From<Dtype> for MetalKernelTemplateArg {
    fn from(value: Dtype) -> Self {
        MetalKernelTemplateArg::Dtype(value)
    }
}

impl From<i32> for MetalKernelTemplateArg {
    fn from(value: i32) -> Self {
        MetalKernelTemplateArg::Int(value)
    }
}

impl From<bool> for MetalKernelTemplateArg {
    fn from(value: bool) -> Self {
        MetalKernelTemplateArg::Bool(value)
    }
}

/// Builder describing how to launch a [`MetalKernel`].
///
/// Mirrors the Python `mx.fast.metal_kernel(...)(...)` call: declare each
/// output's shape and dtype, the dispatch grid, threadgroup, optional template
/// arguments, and an optional output initialiser.
#[derive(Debug, Clone, Default)]
pub struct MetalKernelConfig {
    outputs: Vec<(Vec<i32>, Dtype)>,
    grid: (i32, i32, i32),
    thread_group: (i32, i32, i32),
    template_args: Vec<(CString, MetalKernelTemplateArg)>,
    init_value: Option<f32>,
    verbose: bool,
}

impl MetalKernelConfig {
    /// Create an empty config. At least one output and a grid/thread_group must be set
    /// before passing to [`MetalKernel::apply`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an output buffer of the given `shape` and `dtype`. Outputs are returned
    /// from [`MetalKernel::apply`] in the order they were added, matching the
    /// `output_names` passed to [`MetalKernel::new`].
    pub fn add_output(mut self, shape: impl Into<Vec<i32>>, dtype: Dtype) -> Self {
        self.outputs.push((shape.into(), dtype));
        self
    }

    /// Set the launch grid (`grid1`, `grid2`, `grid3`). Matches the Python
    /// `grid=(g1, g2, g3)` argument.
    pub fn grid(mut self, grid1: i32, grid2: i32, grid3: i32) -> Self {
        self.grid = (grid1, grid2, grid3);
        self
    }

    /// Set the threadgroup size. Matches the Python `threadgroup=(t1, t2, t3)` argument.
    pub fn thread_group(mut self, t1: i32, t2: i32, t3: i32) -> Self {
        self.thread_group = (t1, t2, t3);
        self
    }

    /// Append a template argument. `name` becomes the template parameter inside the
    /// source, e.g. `add_template("Dk", 128_i32)` makes `Dk` available as a constant
    /// inside the kernel.
    pub fn add_template(
        mut self,
        name: &str,
        value: impl Into<MetalKernelTemplateArg>,
    ) -> Result<Self> {
        let cname =
            CString::new(name).map_err(|_| Exception::custom("template name contains a NUL"))?;
        self.template_args.push((cname, value.into()));
        Ok(self)
    }

    /// Pre-fill every output buffer with `value` before the kernel runs.
    pub fn init_value(mut self, value: f32) -> Self {
        self.init_value = Some(value);
        self
    }

    /// Print the generated Metal source on first launch — useful when debugging
    /// template expansion.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    fn into_raw(self) -> Result<RawMetalKernelConfig> {
        unsafe {
            let raw = mlx_sys::mlx_fast_metal_kernel_config_new();
            if raw.ctx.is_null() {
                return Err(Exception::custom(
                    "mlx_fast_metal_kernel_config_new returned a null context",
                ));
            }
            let guard = RawMetalKernelConfig { raw };

            for (shape, dtype) in &self.outputs {
                let status = mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                    guard.raw,
                    shape.as_ptr(),
                    shape.len(),
                    u32::from(*dtype),
                );
                if status != 0 {
                    return Err(Exception::custom("metal_kernel_config: add_output failed"));
                }
            }

            let (g1, g2, g3) = self.grid;
            if mlx_sys::mlx_fast_metal_kernel_config_set_grid(guard.raw, g1, g2, g3) != 0 {
                return Err(Exception::custom("metal_kernel_config: set_grid failed"));
            }

            let (t1, t2, t3) = self.thread_group;
            if mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(guard.raw, t1, t2, t3) != 0 {
                return Err(Exception::custom(
                    "metal_kernel_config: set_thread_group failed",
                ));
            }

            for (name, arg) in &self.template_args {
                let status = match arg {
                    MetalKernelTemplateArg::Dtype(dtype) => {
                        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                            guard.raw,
                            name.as_ptr(),
                            u32::from(*dtype),
                        )
                    }
                    MetalKernelTemplateArg::Int(v) => {
                        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                            guard.raw,
                            name.as_ptr(),
                            *v,
                        )
                    }
                    MetalKernelTemplateArg::Bool(v) => {
                        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                            guard.raw,
                            name.as_ptr(),
                            *v,
                        )
                    }
                };
                if status != 0 {
                    return Err(Exception::custom(
                        "metal_kernel_config: add_template_arg failed",
                    ));
                }
            }

            if let Some(value) = self.init_value {
                if mlx_sys::mlx_fast_metal_kernel_config_set_init_value(guard.raw, value) != 0 {
                    return Err(Exception::custom(
                        "metal_kernel_config: set_init_value failed",
                    ));
                }
            }

            if mlx_sys::mlx_fast_metal_kernel_config_set_verbose(guard.raw, self.verbose) != 0 {
                return Err(Exception::custom("metal_kernel_config: set_verbose failed"));
            }

            Ok(guard)
        }
    }
}

struct RawMetalKernelConfig {
    raw: mlx_sys::mlx_fast_metal_kernel_config,
}

impl std::fmt::Debug for RawMetalKernelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawMetalKernelConfig")
            .finish_non_exhaustive()
    }
}

impl Drop for RawMetalKernelConfig {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_fast_metal_kernel_config_free(self.raw) };
    }
}

/// A JIT-compiled Metal kernel produced from a source string.
///
/// Construct once with [`MetalKernel::new`], then launch any number of times with
/// [`MetalKernel::apply`], passing a fresh [`MetalKernelConfig`] each time. The
/// kernel is cached internally by `mlx`, so repeated launches with the same
/// template arguments do not recompile.
///
/// This mirrors `mlx.core.fast.metal_kernel` from the Python API.
pub struct MetalKernel {
    raw: mlx_sys::mlx_fast_metal_kernel,
}

impl std::fmt::Debug for MetalKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalKernel").finish_non_exhaustive()
    }
}

unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

impl MetalKernel {
    /// Compile a new Metal kernel.
    ///
    /// - `name`: unique name for the kernel (also used for the autogenerated `[[kernel]]` function).
    /// - `input_names` / `output_names`: parameter names referenced inside `source`. Order matches
    ///   the order of arrays passed to [`apply`](Self::apply) and the outputs declared on
    ///   [`MetalKernelConfig`].
    /// - `source`: the body of the kernel (no `[[kernel]]` wrapper — `mlx` synthesises one).
    /// - `header`: extra header text inserted before the kernel, e.g. helper functions.
    /// - `ensure_row_contiguous`: forces inputs to be made contiguous before launch.
    /// - `atomic_outputs`: marks outputs as `atomic<T>*` (required if multiple threads write the same slot).
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        header: &str,
        ensure_row_contiguous: bool,
        atomic_outputs: bool,
    ) -> Result<Self> {
        let cname = CString::new(name).map_err(|_| Exception::custom("name contains a NUL"))?;
        let csource =
            CString::new(source).map_err(|_| Exception::custom("source contains a NUL"))?;
        let cheader =
            CString::new(header).map_err(|_| Exception::custom("header contains a NUL"))?;

        let input_names_vec = StringVector::from_strs(input_names)?;
        let output_names_vec = StringVector::from_strs(output_names)?;

        let raw = unsafe {
            mlx_sys::mlx_fast_metal_kernel_new(
                cname.as_ptr(),
                input_names_vec.raw,
                output_names_vec.raw,
                csource.as_ptr(),
                cheader.as_ptr(),
                ensure_row_contiguous,
                atomic_outputs,
            )
        };
        if raw.ctx.is_null() {
            return Err(Exception::custom("mlx_fast_metal_kernel_new failed"));
        }
        Ok(Self { raw })
    }

    /// Launch the kernel on `inputs` using `config`, returning one [`Array`] per
    /// output declared on the config (in declaration order).
    pub fn apply(
        &self,
        inputs: &[impl AsRef<Array>],
        config: MetalKernelConfig,
        stream: impl AsRef<Stream>,
    ) -> Result<Vec<Array>> {
        let raw_config = config.into_raw()?;
        let input_vec = VectorArray::try_from_iter(inputs.iter())?;
        let stream_ptr = stream.as_ref().as_ptr();

        Vec::<Array>::try_from_op(|res| unsafe {
            mlx_sys::mlx_fast_metal_kernel_apply(
                res,
                self.raw,
                input_vec.as_ptr(),
                raw_config.raw,
                stream_ptr,
            )
        })
    }

    /// Fixed-output-count variant of [`Self::apply`] that drains the result
    /// into a stack `[Array; N]`, avoiding the `Vec<Array>` heap allocation
    /// on the hot path when the kernel's output count is statically known.
    /// Errors if the kernel returns a count other than `N`.
    pub fn apply_array<const N: usize>(
        &self,
        inputs: &[impl AsRef<Array>],
        config: MetalKernelConfig,
        stream: impl AsRef<Stream>,
    ) -> Result<[Array; N]> {
        let raw_config = config.into_raw()?;
        let input_vec = VectorArray::try_from_iter(inputs.iter())?;
        let stream_ptr = stream.as_ref().as_ptr();

        let out = VectorArray::try_from_op(|res| unsafe {
            mlx_sys::mlx_fast_metal_kernel_apply(
                res,
                self.raw,
                input_vec.as_ptr(),
                raw_config.raw,
                stream_ptr,
            )
        })?;
        out.try_into_array::<N>()
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_fast_metal_kernel_free(self.raw) };
    }
}

struct StringVector {
    raw: mlx_sys::mlx_vector_string,
}

impl std::fmt::Debug for StringVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StringVector").finish_non_exhaustive()
    }
}

impl StringVector {
    fn from_strs(values: &[&str]) -> Result<Self> {
        let cstrs: Vec<CString> = values
            .iter()
            .map(|s| {
                CString::new(*s).map_err(|_| Exception::custom("input/output name contains a NUL"))
            })
            .collect::<Result<_>>()?;
        let raw = unsafe { mlx_sys::mlx_vector_string_new() };
        for cs in &cstrs {
            let status = unsafe { mlx_sys::mlx_vector_string_append_value(raw, cs.as_ptr()) };
            if status != 0 {
                unsafe { mlx_sys::mlx_vector_string_free(raw) };
                return Err(Exception::custom("vector_string_append_value failed"));
            }
        }
        Ok(Self { raw })
    }
}

impl Drop for StringVector {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_vector_string_free(self.raw) };
    }
}

/// Convenience constructor mirroring `mx.fast.metal_kernel(...)` from the Python API.
pub fn metal_kernel(
    name: &str,
    input_names: &[&str],
    output_names: &[&str],
    source: &str,
    header: &str,
    ensure_row_contiguous: bool,
    atomic_outputs: bool,
) -> Result<MetalKernel> {
    MetalKernel::new(
        name,
        input_names,
        output_names,
        source,
        header,
        ensure_row_contiguous,
        atomic_outputs,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ops::indexing::{ArrayIndexOp, IndexOp},
        random::normal,
    };
    use float_eq::assert_float_eq;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_rope() {
        crate::random::seed(71).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let result = rope(a, 8, false, 10000., 1.0, 0, None).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.456_253_77,
            abs <= 0.009_125_075
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            116.800_964,
            abs <= 2.336_019_3
        );
    }

    // Test adapted from Python test_fast.py/test_rope - the Python test accepts both
    // int offset and array offset, which in C/Rust are separate functions
    #[test]
    fn test_rope_dynamic() {
        crate::random::seed(71).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        // Test with array offset - should produce similar results to int offset of 3
        let offset = crate::Array::from_int(3);
        let result = rope_dynamic(&a, 8, false, 10000., 1.0, &offset, None).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);

        // Compare with regular rope using int offset=3
        let result_int_offset = rope(&a, 8, false, 10000., 1.0, 3, None).unwrap();
        assert_eq!(result_int_offset.shape(), [2, 8, 16]);

        // The results should be close
        let diff = &result - &result_int_offset;
        let max_diff = diff.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max_diff < 1e-5, "Max difference was {}", max_diff);
    }

    #[test]
    fn test_rms_norm() {
        crate::random::seed(103).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let result = rms_norm(a, &weight, 1e-5).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.872_938_75,
            abs <= 0.017_458_774
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            223.472_32,
            abs <= 4.469_446
        );
    }

    #[test]
    fn rms_norm_without_weight_matches_manual_normalization() {
        let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 4]);
        let fused = rms_norm(&a, None, 1e-5).unwrap();

        let sq = a.multiply(&a).unwrap();
        let mean_sq = sq.mean_axes(&[-1], true).unwrap();
        let denom = crate::ops::rsqrt(mean_sq.add(Array::from_f32(1e-5)).unwrap()).unwrap();
        let expected = a.multiply(&denom).unwrap();

        let diff = fused.subtract(&expected).unwrap();
        let max = diff.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(
            max < 1e-5,
            "fused vs manual rms_norm(no weight) max_abs={max}"
        );
    }

    #[test]
    pub fn test_layer_norm_affine() {
        crate::random::seed(635).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let bias = Array::zeros::<f32>(&[16]).unwrap();
        let result = layer_norm(a, &weight, &bias, 1e-5).unwrap();
        let result = result.index((ArrayIndexOp::Ellipsis, 0));
        assert_eq!(result.shape(), [2, 8]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.290_990_38,
            abs <= 0.005_819_807_8
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            4.655_846,
            abs <= 0.093_116_924
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_fast_sdpa() {
        // This test just makes sure that `scaled_dot_product_attention` is callable
        // in the various cases, based on the Python test `test_fast_sdpa`.

        let Dk = 64;
        let scale = 1.0 / (Dk as f32).sqrt();
        for seq_len in [63, 129, 400] {
            for dtype in [crate::Dtype::Float32, crate::Dtype::Float16] {
                let B = 2;
                let H = 24;
                let q = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                let k = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                let v = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();

                let result = scaled_dot_product_attention(q, k, v, scale, None, None).unwrap();
                assert_eq!(result.shape(), [B, H, seq_len, Dk]);
                assert_eq!(result.dtype(), dtype);
            }
        }
    }

    // Test adapted from Python test `test_fast_sdpa.py/test_sdpa_attention_sinks`
    #[test]
    fn test_fast_sdpa_with_sinks() {
        let b = 2;
        let n_q = 8;
        let t_q = 128;
        let t_kv = 128;
        let d = 64;

        let q = normal::<f32>(&[b, n_q, t_q, d], None, None, None).unwrap();
        let k = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let v = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let scale = (d as f32).powf(-0.5);

        // Test with sinks parameter
        let sinks = normal::<f32>(&[n_q], None, None, None).unwrap() * 10.0;

        let result = scaled_dot_product_attention(&q, &k, &v, scale, None, &sinks).unwrap();
        assert_eq!(result.shape(), &[b, n_q, t_q, d]);
    }

    // Minimal smoke test that the `metal_kernel` plumbing compiles, links, and runs.
    // The kernel template-instantiates over the input dtype, returns one threadgroup
    // per output element, and computes `out = a * scale + b` so the test covers
    // template args (dtype + int), grid/threadgroup, multiple inputs and verifies
    // the result against the equivalent MLX op.
    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_kernel_axpy() {
        let n = 64;
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[n], None).unwrap();
        let b = crate::random::uniform::<_, f32>(0.0, 1.0, &[n], None).unwrap();

        let source = r#"
            uint gid = thread_position_in_grid.x;
            if (gid >= (uint)N) { return; }
            out[gid] = static_cast<T>(static_cast<float>(a[gid]) * Scale + static_cast<float>(b[gid]));
        "#;

        let kernel =
            metal_kernel("test_axpy", &["a", "b"], &["out"], source, "", true, false).unwrap();

        let config = MetalKernelConfig::new()
            .add_output([n], Dtype::Float32)
            .grid(n, 1, 1)
            .thread_group(32, 1, 1)
            .add_template("T", Dtype::Float32)
            .unwrap()
            .add_template("Scale", 2_i32)
            .unwrap()
            .add_template("N", n)
            .unwrap();

        let outs = kernel
            .apply(&[a.clone(), b.clone()], config, crate::Stream::default())
            .unwrap();
        assert_eq!(outs.len(), 1);
        let got = &outs[0];
        assert_eq!(got.shape(), &[n]);
        assert_eq!(got.dtype(), Dtype::Float32);

        let expected = (a * 2.0).add(&b).unwrap();
        let diff = (got - expected).abs().unwrap();
        let max_diff = diff.max(None).unwrap().item::<f32>();
        assert!(max_diff < 1e-5, "max diff {max_diff}");
    }
}
