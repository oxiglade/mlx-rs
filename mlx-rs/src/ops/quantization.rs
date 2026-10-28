use std::ffi::CStr;

use mlx_internal_macros::generate_macro;

use crate::{
    error::Result,
    utils::{guard::Guarded, VectorArray},
    Array, Stream,
};

/// Packed weight format. All variants support CPU and Metal in MLX 0.32.2.
///
/// Defaults are (group size, bits): affine (64, 4), mxfp4 (32, 4),
/// mxfp8 (32, 8), and nvfp4 (16, 4). Global scaling is not exposed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    /// Integer quantization with a scale and bias per group.
    #[default]
    Affine,
    /// E2M1 values with E8M0 scales.
    Mxfp4,
    /// E4M3 values with E8M0 scales.
    Mxfp8,
    /// E2M1 values with E4M3 scales, without a global scale.
    Nvfp4,
}

/// Invalid quantization configuration or an upstream operation failure.
#[derive(Debug, thiserror::Error)]
pub enum QuantizationError {
    /// The mode does not support this group size and bit width.
    #[error("unsupported {mode:?} quantization: group size {group_size}, bits {bits}")]
    InvalidConfiguration {
        /// Requested format.
        mode: QuantizationMode,
        /// Resolved group size.
        group_size: i32,
        /// Resolved bit width.
        bits: i32,
    },
    /// Diagnostic from MLX.
    #[error(transparent)]
    Exception(#[from] crate::error::Exception),
}

impl From<QuantizationError> for crate::error::Exception {
    fn from(error: QuantizationError) -> Self {
        match error {
            QuantizationError::Exception(error) => error,
            error => Self::custom(error.to_string()),
        }
    }
}

impl QuantizationMode {
    fn validate(
        self,
        group_size: Option<i32>,
        bits: Option<i32>,
    ) -> std::result::Result<(), QuantizationError> {
        let (default_group, default_bits) = match self {
            Self::Affine => (64, 4),
            Self::Mxfp4 => (32, 4),
            Self::Mxfp8 => (32, 8),
            Self::Nvfp4 => (16, 4),
        };
        let group_size = group_size.unwrap_or(default_group);
        let bits = bits.unwrap_or(default_bits);
        let valid = match self {
            Self::Affine => {
                matches!(group_size, 32 | 64 | 128) && matches!(bits, 2 | 3 | 4 | 5 | 6 | 8)
            }
            _ => group_size == default_group && bits == default_bits,
        };
        if !valid {
            return Err(QuantizationError::InvalidConfiguration {
                mode: self,
                group_size,
                bits,
            });
        }
        Ok(())
    }

    fn as_c_str(self) -> &'static CStr {
        match self {
            Self::Affine => c"affine",
            Self::Mxfp4 => c"mxfp4",
            Self::Mxfp8 => c"mxfp8",
            Self::Nvfp4 => c"nvfp4",
        }
    }
}

/// Configuration for [`quantize`]. `None` selects the mode's default.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct QuantizeOptions {
    /// Quantization format.
    pub mode: QuantizationMode,
    /// Elements sharing a scale: affine 32/64/128, MX 32, NV 16.
    pub group_size: Option<i32>,
    /// Bits per element: affine 2/3/4/5/6/8, mxfp8 8, other modes 4.
    pub bits: Option<i32>,
}

/// Configuration for [`dequantize`]. `None` selects the upstream mode default.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DequantizeOptions {
    /// Quantization format used to pack the weights.
    pub mode: QuantizationMode,
    /// Elements sharing a scale; see [`QuantizeOptions`].
    pub group_size: Option<i32>,
    /// Bits per packed element; see [`QuantizeOptions`].
    pub bits: Option<i32>,
}

/// Configuration for [`quantized_matmul`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedMatmulOptions {
    /// Quantization format used to pack the weights.
    pub mode: QuantizationMode,
    /// Elements sharing a scale; `None` selects the mode's default.
    pub group_size: Option<i32>,
    /// Bits per packed element; `None` selects the mode's default.
    pub bits: Option<i32>,
    /// Transpose the packed weight matrix. Defaults to false.
    pub transpose: bool,
}

/// Configuration for [`gather_qmm`].
#[derive(Debug, Clone, Copy)]
pub struct GatherQmmOptions<'a> {
    /// Quantization format used to pack the weights.
    pub mode: QuantizationMode,
    /// Elements sharing a scale; `None` selects the mode's default.
    pub group_size: Option<i32>,
    /// Bits per packed element; `None` selects the mode's default.
    pub bits: Option<i32>,
    /// Transpose the packed weight matrix. Defaults to true.
    pub transpose: bool,
    /// Indices into the input batch dimensions.
    pub lhs_indices: Option<&'a Array>,
    /// Indices into the weight batch dimensions.
    pub rhs_indices: Option<&'a Array>,
    /// Whether the supplied indices are sorted. Defaults to false.
    pub sorted_indices: bool,
}

impl Default for GatherQmmOptions<'_> {
    fn default() -> Self {
        Self {
            mode: QuantizationMode::default(),
            group_size: None,
            bits: None,
            transpose: true,
            lhs_indices: None,
            rhs_indices: None,
            sorted_indices: false,
        }
    }
}

const DEFAULT_GROUP_SIZE: i32 = 64;
const DEFAULT_BITS: i32 = 4;

/// Helper to convert Option<i32> to mlx_optional_int
fn optional_int(value: Option<i32>, default: i32) -> mlx_sys::mlx_optional_int {
    mlx_sys::mlx_optional_int {
        value: value.unwrap_or(default),
        has_value: value.is_some(),
    }
}

/// Helper to create a "no value" optional dtype
fn optional_dtype_none() -> mlx_sys::mlx_optional_dtype {
    mlx_sys::mlx_optional_dtype {
        value: mlx_sys::mlx_dtype__MLX_FLOAT32, // default value, ignored when has_value is false
        has_value: false,
    }
}

/// Quantize weights into packed values, scales, and optional biases.
///
/// Biases are present only for [`QuantizationMode::Affine`]. Inputs must have at
/// least two dimensions and a last dimension divisible by the group size.
/// Invalid mode/group/bit combinations return [`QuantizationError::InvalidConfiguration`].
/// Shapes and dtypes are checked upstream.
///
/// ```
/// use mlx_rs::{Array, ops::{self, QuantizationMode, QuantizeOptions, DequantizeOptions}};
/// # fn main() -> Result<(), ops::QuantizationError> {
/// let weights = Array::ones::<f32>(&[32, 64])?;
/// let mode = QuantizationMode::Mxfp4;
/// let (packed, scales, biases) = ops::quantize(
///     &weights, QuantizeOptions { mode, ..Default::default() },
/// )?;
/// assert!(biases.is_none());
/// let restored = ops::dequantize(
///     &packed, &scales, biases.as_ref(),
///     DequantizeOptions { mode, ..Default::default() },
/// )?;
/// assert_eq!(restored.shape(), weights.shape());
/// # Ok(())
/// # }
/// ```
pub fn quantize(
    w: impl AsRef<Array>,
    options: QuantizeOptions,
) -> std::result::Result<(Array, Array, Option<Array>), QuantizationError> {
    options.mode.validate(options.group_size, options.bits)?;
    let stream = Stream::thread_local_or_default();
    let group_size = optional_int(options.group_size, DEFAULT_GROUP_SIZE);
    let bits = optional_int(options.bits, DEFAULT_BITS);
    let global_scale = unsafe { Array::from_ptr(mlx_sys::mlx_array_new()) };

    let result = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_quantize(
            res,
            w.as_ref().as_ptr(),
            group_size,
            bits,
            options.mode.as_c_str().as_ptr(),
            global_scale.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })?;

    let arrays: Vec<Array> = result.try_into_values()?;
    let expected = if options.mode == QuantizationMode::Affine {
        3
    } else {
        2
    };
    if arrays.len() != expected {
        return Err(crate::error::Exception::custom(format!(
            "Expected {expected} arrays from quantize, got {}",
            arrays.len()
        ))
        .into());
    }
    let mut iter = arrays.into_iter();
    Ok((iter.next().unwrap(), iter.next().unwrap(), iter.next()))
}

/// Compatibility shim for [`quantize`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `quantize`"
)]
pub fn quantize_device(
    w: impl AsRef<Array>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<(Array, Array, Array)> {
    crate::with_stream(stream.as_ref(), || {
        let (weights, scales, biases) = quantize(
            w,
            QuantizeOptions {
                group_size: group_size.into(),
                bits: bits.into(),
                ..Default::default()
            },
        )?;
        Ok((
            weights,
            scales,
            biases.expect("affine quantization returns biases"),
        ))
    })
}

/// Multiply `x` by packed weights. Pass biases for affine weights and `None`
/// for floating-point modes; options must match the weights' quantization.
pub fn quantized_matmul<'a>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: impl Into<Option<&'a Array>>,
    options: QuantizedMatmulOptions,
) -> std::result::Result<Array, QuantizationError> {
    options.mode.validate(options.group_size, options.bits)?;
    let stream = Stream::thread_local_or_default();
    let absent = unsafe { Array::from_ptr(mlx_sys::mlx_array_new()) };
    let transpose = options.transpose;
    let group_size = optional_int(options.group_size, DEFAULT_GROUP_SIZE);
    let bits = optional_int(options.bits, DEFAULT_BITS);

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_quantized_matmul(
            res,
            x.as_ref().as_ptr(),
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases.into().map(|a| a.as_ptr()).unwrap_or(absent.as_ptr()),
            transpose,
            group_size,
            bits,
            options.mode.as_c_str().as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
    .map_err(Into::into)
}

/// Compatibility shim for [`quantized_matmul`].
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `quantized_matmul`"
)]
pub fn quantized_matmul_device<'a>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    #[optional] biases: impl Into<Option<&'a Array>>,
    #[optional] transpose: impl Into<Option<bool>>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || {
        quantized_matmul(
            x,
            w,
            scales,
            biases,
            QuantizedMatmulOptions {
                transpose: transpose.into().unwrap_or(false),
                group_size: group_size.into(),
                bits: bits.into(),
                ..Default::default()
            },
        )
        .map_err(Into::into)
    })
}

/// Reconstruct weights using the same mode, group size, and bits used to quantize them.
/// Pass biases for affine weights and `None` for the floating-point modes.
/// Floating-point modes dequantize to bfloat16; affine uses the scale/bias dtype.
pub fn dequantize<'a>(
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: impl Into<Option<&'a Array>>,
    options: DequantizeOptions,
) -> std::result::Result<Array, QuantizationError> {
    options.mode.validate(options.group_size, options.bits)?;
    let stream = Stream::thread_local_or_default();
    let absent = unsafe { Array::from_ptr(mlx_sys::mlx_array_new()) };
    let group_size = optional_int(options.group_size, DEFAULT_GROUP_SIZE);
    let bits = optional_int(options.bits, DEFAULT_BITS);
    let global_scale = unsafe { Array::from_ptr(mlx_sys::mlx_array_new()) };

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_dequantize(
            res,
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases.into().map(|a| a.as_ptr()).unwrap_or(absent.as_ptr()),
            group_size,
            bits,
            options.mode.as_c_str().as_ptr(),
            global_scale.as_ptr(),
            optional_dtype_none(),
            stream.as_ref().as_ptr(),
        )
    })
    .map_err(Into::into)
}

/// Compatibility shim for [`dequantize`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `dequantize`"
)]
pub fn dequantize_device<'a>(
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    #[optional] biases: impl Into<Option<&'a Array>>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || {
        dequantize(
            w,
            scales,
            biases,
            DequantizeOptions {
                group_size: group_size.into(),
                bits: bits.into(),
                ..Default::default()
            },
        )
        .map_err(Into::into)
    })
}

/// Multiply selected input and packed-weight batches.
/// Pass affine biases or `None` for floating-point modes. Quantization options
/// must match the packed weights; indices select the respective batch dimensions.
pub fn gather_qmm<'b>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: impl Into<Option<&'b Array>>,
    options: GatherQmmOptions<'_>,
) -> std::result::Result<Array, QuantizationError> {
    options.mode.validate(options.group_size, options.bits)?;
    let stream = Stream::thread_local_or_default();
    let absent = unsafe { Array::from_ptr(mlx_sys::mlx_array_new()) };
    let transpose = options.transpose;
    let group_size = optional_int(options.group_size, DEFAULT_GROUP_SIZE);
    let bits = optional_int(options.bits, DEFAULT_BITS);
    let sorted = options.sorted_indices;

    unsafe {
        let biases_ptr = biases.into().map(|a| a.as_ptr()).unwrap_or(absent.as_ptr());
        let lhs_ptr = options
            .lhs_indices
            .map(|i| i.as_ptr())
            .unwrap_or(absent.as_ptr());
        let rhs_ptr = options
            .rhs_indices
            .map(|i| i.as_ptr())
            .unwrap_or(absent.as_ptr());

        <Array as Guarded>::try_from_op(|res| {
            mlx_sys::mlx_gather_qmm(
                res,
                x.as_ref().as_ptr(),
                w.as_ref().as_ptr(),
                scales.as_ref().as_ptr(),
                biases_ptr,
                lhs_ptr,
                rhs_ptr,
                transpose,
                group_size,
                bits,
                options.mode.as_c_str().as_ptr(),
                sorted,
                stream.as_ref().as_ptr(),
            )
        })
        .map_err(Into::into)
    }
}

/// Compatibility shim for [`gather_qmm`].
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `gather_qmm`"
)]
pub fn gather_qmm_device<'b, 'lhs, 'rhs>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    #[optional] biases: impl Into<Option<&'b Array>>,
    #[optional] lhs_indices: impl Into<Option<&'lhs Array>>,
    #[optional] rhs_indices: impl Into<Option<&'rhs Array>>,
    #[optional] transpose: impl Into<Option<bool>>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] sorted_indices: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || {
        gather_qmm(
            x,
            w,
            scales,
            biases,
            GatherQmmOptions {
                lhs_indices: lhs_indices.into(),
                rhs_indices: rhs_indices.into(),
                transpose: transpose.into().unwrap_or(true),
                group_size: group_size.into(),
                bits: bits.into(),
                sorted_indices: sorted_indices.into().unwrap_or(false),
                ..Default::default()
            },
        )
        .map_err(Into::into)
    })
}

/// Quantized matrix multiplication with quantization of both inputs.
///
/// Performs matrix multiplication where `x` is dynamically quantized and `w` is pre-quantized.
/// This function supports `nvfp4` and `mxfp8` quantization modes.
///
/// Note: This function is only supported on GPU with the CUDA backend (Linux with NVIDIA GPU).
/// It is not available on macOS.
///
/// # Params
///
/// - `x`: Input matrix to be dynamically quantized
/// - `w`: Pre-quantized weight matrix
/// - `w_scales`: Optional scales for the quantized weights (required if `w` is already quantized)
/// - `group_size`: The quantization group size (default depends on mode: 16 for nvfp4, 32 for mxfp8)
/// - `bits`: The number of bits per element (default depends on mode: 4 for nvfp4, 8 for mxfp8)
/// - `mode`: Quantization mode - either "nvfp4" or "mxfp8" (default: "nvfp4")
#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn qqmm<'a>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    w_scales: impl Into<Option<&'a Array>>,
    group_size: impl Into<Option<i32>>,
    bits: impl Into<Option<i32>>,
    mode: impl Into<Option<&'a str>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let mode_str = mode.into().unwrap_or("nvfp4");
    let mode_cstr = std::ffi::CString::new(mode_str).expect("Invalid mode string");

    // Defaults depend on mode
    let (default_group_size, default_bits) = match mode_str {
        "nvfp4" => (16, 4),
        "mxfp8" => (32, 8),
        _ => (16, 4), // fallback to nvfp4 defaults
    };

    let group_size = optional_int(group_size.into(), default_group_size);
    let bits = optional_int(bits.into(), default_bits);
    let global_scale_x = unsafe { Array::from_ptr(mlx_sys::mlx_array_new()) };
    let global_scale_w = unsafe { Array::from_ptr(mlx_sys::mlx_array_new()) };

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_qqmm(
            res,
            x.as_ref().as_ptr(),
            w.as_ref().as_ptr(),
            w_scales
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            group_size,
            bits,
            mode_cstr.as_ptr(),
            global_scale_x.as_ptr(),
            global_scale_w.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`qqmm`].
#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `qqmm`"
)]
pub fn qqmm_device<'a>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    #[optional] w_scales: impl Into<Option<&'a Array>>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] mode: impl Into<Option<&'a str>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || {
        qqmm(x, w, w_scales, group_size, bits, mode)
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        ops::{dequantize, expand_dims, quantize, quantized_matmul},
        random, Array,
    };

    #[test]
    fn invalid_quantization_options_do_not_reach_ffi() {
        use super::{QuantizationError, QuantizationMode, QuantizeOptions, QuantizedMatmulOptions};
        let weights = Array::ones::<f32>(&[32, 64]).unwrap();
        for mode in [
            QuantizationMode::Affine,
            QuantizationMode::Mxfp4,
            QuantizationMode::Mxfp8,
            QuantizationMode::Nvfp4,
        ] {
            for (group_size, bits) in [(Some(0), None), (None, Some(0)), (Some(7), Some(7))] {
                assert!(matches!(
                    quantize(
                        &weights,
                        QuantizeOptions {
                            mode,
                            group_size,
                            bits
                        }
                    ),
                    Err(QuantizationError::InvalidConfiguration { .. })
                ));
                assert!(matches!(
                    quantized_matmul(
                        &weights,
                        &weights,
                        &weights,
                        None,
                        QuantizedMatmulOptions {
                            mode,
                            group_size,
                            bits,
                            ..Default::default()
                        }
                    ),
                    Err(QuantizationError::InvalidConfiguration { .. })
                ));
            }
        }
    }

    #[test]
    fn test_quantize_dequantize() {
        let x1 = Array::ones::<f32>(&[128, 1]).unwrap();
        let x2 = expand_dims(Array::arange::<_, f32>(0, 512, None).unwrap(), 0).unwrap();
        let x = x1 * x2;

        for i in [2, 4, 8].iter() {
            let el_per_int = 32 / i;
            let (x_q, scales, biases) = quantize(
                &x,
                crate::ops::QuantizeOptions {
                    group_size: Some(128),
                    bits: Some(*i),
                    ..Default::default()
                },
            )
            .unwrap();
            let biases = biases.expect("affine quantization returns biases");
            assert_eq!(x_q.shape(), [128, 512 / el_per_int]);
            assert_eq!(scales.shape(), [128, 4]);
            assert_eq!(biases.shape(), [128, 4]);

            let x_hat = dequantize(
                &x_q,
                &scales,
                &biases,
                crate::ops::DequantizeOptions {
                    group_size: Some(128),
                    bits: Some(*i),
                    ..Default::default()
                },
            )
            .unwrap();
            let max_diff = ((&x - &x_hat).abs().unwrap().max(None).unwrap()).item_exact::<f32>();
            assert!(max_diff <= 127.0 / (1 << i) as f32);
        }
    }

    // Test adapted from Python test `test_quantized.py/test_qmm`
    #[test]
    fn test_quantized_matmul() {
        random::seed(0).unwrap();

        let group_size = 64;
        let bits = 4;
        let m = 32;
        let n = 128;
        let k = 128;

        let scale = 1.0 / (k as f32).sqrt();
        let x = random::normal::<f32>(&[m, k], None, None, None).unwrap() * scale;
        let w = random::normal::<f32>(&[k, n], None, None, None).unwrap() * scale;

        let (w_q, scales, biases) = quantize(
            &w,
            crate::ops::QuantizeOptions {
                group_size: Some(group_size),
                bits: Some(bits),
                ..Default::default()
            },
        )
        .unwrap();
        let biases = biases.expect("affine quantization returns biases");
        let w_hat = dequantize(
            &w_q,
            &scales,
            &biases,
            crate::ops::DequantizeOptions {
                group_size: Some(group_size),
                bits: Some(bits),
                ..Default::default()
            },
        )
        .unwrap();

        // Test with biases
        let y_q = quantized_matmul(
            &x,
            &w_q,
            &scales,
            &biases,
            crate::ops::QuantizedMatmulOptions {
                transpose: false,
                group_size: Some(group_size),
                bits: Some(bits),
                ..Default::default()
            },
        )
        .unwrap();
        let y_hat = x.matmul(&w_hat).unwrap();

        assert_eq!(y_q.shape(), y_hat.shape());
        let max_diff = ((&y_q - &y_hat).abs().unwrap().max(None).unwrap()).item_exact::<f32>();
        assert!(max_diff < 1e-3, "max_diff: {}", max_diff);
    }

    // Test adapted from Python test `test_quantized.py/test_gather_qmm`
    #[test]
    fn test_gather_qmm() {
        use crate::ops::{gather_mm, gather_qmm, swap_axes};

        random::seed(0).unwrap();

        let group_size = 64;
        let bits = 4;

        // Helper to quantize with transpose option
        fn quantize_with_transpose(
            w: &Array,
            transpose: bool,
            group_size: i32,
            bits: i32,
        ) -> (Array, Array, Array, Array) {
            let (w_q, scales, biases) = quantize(
                w,
                crate::ops::QuantizeOptions {
                    group_size: Some(group_size),
                    bits: Some(bits),
                    ..Default::default()
                },
            )
            .unwrap();
            let biases = biases.expect("affine quantization returns biases");
            let mut w_hat = dequantize(
                &w_q,
                &scales,
                &biases,
                crate::ops::DequantizeOptions {
                    group_size: Some(group_size),
                    bits: Some(bits),
                    ..Default::default()
                },
            )
            .unwrap();
            if transpose {
                w_hat = swap_axes(&w_hat, -1, -2).unwrap();
            }
            (w_hat, w_q, scales, biases)
        }

        // Test case 1: batch_A=(1,), lhs_indices=(0,), batch_B=(3,), rhs_indices=(2, 1)
        let m = 32;
        let n = 64;
        let k = 64;

        let x = random::normal::<f32>(&[1, m, k], None, None, None).unwrap();
        let w = random::normal::<f32>(&[3, n, k], None, None, None).unwrap(); // transpose=true shape
        let (w_hat, w_q, scales, biases) = quantize_with_transpose(&w, true, group_size, bits);

        let lhs_indices = Array::from_slice(&[0u32], &[1]);
        let rhs_indices = Array::from_slice(&[2u32, 1], &[2]);

        // Compare gather_mm on dequantized weights vs gather_qmm
        let c1 = gather_mm(&x, &w_hat, &lhs_indices, &rhs_indices, None).unwrap();
        let c2 = gather_qmm(
            &x,
            &w_q,
            &scales,
            &biases,
            crate::ops::GatherQmmOptions {
                lhs_indices: Some(&lhs_indices),
                rhs_indices: Some(&rhs_indices),
                transpose: true,
                group_size: Some(group_size),
                bits: Some(bits),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            c1.all_close(&c2, 1e-4, 1e-4, None).unwrap(),
            "gather_qmm test case 1 failed"
        );

        // Test case 2: batch_A=(5,), lhs_indices=(0, 2), batch_B=(3,), rhs_indices=(2, 1)
        let x = random::normal::<f32>(&[5, m, k], None, None, None).unwrap();
        let lhs_indices = Array::from_slice(&[0u32, 2], &[2]);

        let c1 = gather_mm(&x, &w_hat, &lhs_indices, &rhs_indices, None).unwrap();
        let c2 = gather_qmm(
            &x,
            &w_q,
            &scales,
            &biases,
            crate::ops::GatherQmmOptions {
                lhs_indices: Some(&lhs_indices),
                rhs_indices: Some(&rhs_indices),
                transpose: true,
                group_size: Some(group_size),
                bits: Some(bits),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            c1.all_close(&c2, 1e-4, 1e-4, None).unwrap(),
            "gather_qmm test case 2 failed"
        );
    }
}
