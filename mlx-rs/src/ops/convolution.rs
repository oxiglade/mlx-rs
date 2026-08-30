use crate::error::Result;
use crate::utils::guard::Guarded;
use crate::utils::IntoOption;
use crate::{Array, Stream};
use mlx_internal_macros::{default_device, generate_macro};

/// General convolution over an input with several channels returning an error if the inputs are invalid.
///
/// - Only 1d and 2d convolutions are supported at the moment
/// - the default `groups: 1` is currently supported
///
/// # Params
///
/// - array: Input array of shape `&[N, ..., C_in]`
/// - weight: Weight array of shape `&[C_out, ..., C_in]`
/// - strides: The kernel strides. All dimensions get the same stride if only one number is specified.
/// - padding: The input padding. All dimensions get the same padding if only one number is specified.
/// - kernel_dilation: The kernel dilation. All dimensions get the same dilation if only one number is specified.
/// - input_dilation: The input dilation. All dimensions get the same dilation if only one number is specified.
/// - groups: Input feature groups
/// - flip: Flip the order in which the spatial dimensions of the weights are processed.
///   Performs the cross-correlation operator when `flip` is `false` and the convolution
///   operator otherwise.
#[generate_macro]
#[default_device]
#[allow(clippy::too_many_arguments)]
pub fn conv_general_device<'a>(
    array: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] strides: impl IntoOption<&'a [i32]>,
    #[optional] padding: impl IntoOption<&'a [i32]>,
    #[optional] kernel_dilation: impl IntoOption<&'a [i32]>,
    #[optional] input_dilation: impl IntoOption<&'a [i32]>,
    #[optional] groups: impl Into<Option<i32>>,
    #[optional] flip: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let strides = strides.into_option().unwrap_or(&[1]);
    let padding = padding.into_option().unwrap_or(&[0]);
    let kernel_dilation = kernel_dilation.into_option().unwrap_or(&[1]);
    let input_dilation = input_dilation.into_option().unwrap_or(&[1]);
    let groups = groups.into().unwrap_or(1);
    let flip = flip.into().unwrap_or(false);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_conv_general(
            res,
            array.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            strides.as_ptr(),
            strides.len(),
            padding.as_ptr(),
            padding.len(),
            padding.as_ptr(),
            padding.len(),
            kernel_dilation.as_ptr(),
            kernel_dilation.len(),
            input_dilation.as_ptr(),
            input_dilation.len(),
            groups,
            flip,
            stream.as_ref().as_ptr(),
        )
    })
}

/// 1D convolution over an input with several channels returning an error if the inputs are invalid.
///
/// Only the default `groups=1` is currently supported.
///
/// # Params
///
/// - array: input array of shape `&[N, H, C_in]`
/// - weight: weight array of shape `&[C_out, H, C_in]`
/// - stride: kernel stride. Default to 1 if not specified.
/// - padding: input padding. Default to 0 if not specified.
/// - dilation: kernel dilation. Default to 1 if not specified.
/// - groups: input feature groups. Default to 1 if not specified.
#[generate_macro]
#[default_device]
pub fn conv1d_device(
    array: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride: impl Into<Option<i32>>,
    #[optional] padding: impl Into<Option<i32>>,
    #[optional] dilation: impl Into<Option<i32>>,
    #[optional] groups: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stride = stride.into().unwrap_or(1);
    let padding = padding.into().unwrap_or(0);
    let dilation = dilation.into().unwrap_or(1);
    let groups = groups.into().unwrap_or(1);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_conv1d(
            res,
            array.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            stride,
            padding,
            dilation,
            groups,
            stream.as_ref().as_ptr(),
        )
    })
}

/// 2D convolution over an input with several channels returning an error if the inputs are invalid.
///
/// Only the default `groups=1` is currently supported.
///
/// # Params
///
/// - array: input array of shape `[N, H, W, C_in]`
/// - weight: weight array of shape `[C_out, H, W, C_in]`
/// - stride: kernel stride. Default to (1, 1) if not specified.
/// - padding: input padding. Default to (0, 0) if not specified.
/// - dilation: kernel dilation. Default to (1, 1) if not specified.
/// - groups: input feature groups. Default to 1 if not specified.
#[generate_macro]
#[default_device]
pub fn conv2d_device(
    array: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride: impl Into<Option<(i32, i32)>>,
    #[optional] padding: impl Into<Option<(i32, i32)>>,
    #[optional] dilation: impl Into<Option<(i32, i32)>>,
    #[optional] groups: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stride = stride.into().unwrap_or((1, 1));
    let padding = padding.into().unwrap_or((0, 0));
    let dilation = dilation.into().unwrap_or((1, 1));
    let groups = groups.into().unwrap_or(1);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_conv2d(
            res,
            array.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            stride.0,
            stride.1,
            padding.0,
            padding.1,
            dilation.0,
            dilation.1,
            groups,
            stream.as_ref().as_ptr(),
        )
    })
}

/// 3D convolution over an input with several channels.
///
/// Only the default `groups=1` is currently supported.
#[generate_macro]
#[default_device]
pub fn conv3d_device(
    array: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride: impl Into<Option<(i32, i32, i32)>>,
    #[optional] padding: impl Into<Option<(i32, i32, i32)>>,
    #[optional] dilation: impl Into<Option<(i32, i32, i32)>>,
    #[optional] groups: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stride = stride.into().unwrap_or((1, 1, 1));
    let padding = padding.into().unwrap_or((0, 0, 0));
    let dilation = dilation.into().unwrap_or((1, 1, 1));
    let groups = groups.into().unwrap_or(1);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_conv3d(
            res,
            array.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            stride.0,
            stride.1,
            stride.2,
            padding.0,
            padding.1,
            padding.2,
            dilation.0,
            dilation.1,
            dilation.2,
            groups,
            stream.as_ref().as_ptr(),
        )
    })
}

/// 1D transposed convolution over an input with several channels.
///
/// Only the default `groups=1` is currently supported.
///
/// # Params
///
/// - array: input array of shape `[N, H, C_in]`
/// - weight: weight array of shape `[C_out, H, C_in]`
/// - stride: kernel stride. Default to 1 if not specified.
/// - padding: input padding. Default to 0 if not specified.
/// - dilation: kernel dilation. Default to 1 if not specified.
/// - groups: input feature groups. Default to 1 if not specified.
/// - stream: stream or device to evaluate on.
#[allow(clippy::too_many_arguments)]
#[generate_macro]
#[default_device]
pub fn conv_transpose1d_device(
    array: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride: impl Into<Option<i32>>,
    #[optional] padding: impl Into<Option<i32>>,
    #[optional] dilation: impl Into<Option<i32>>,
    #[optional] output_padding: impl Into<Option<i32>>,
    #[optional] groups: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stride = stride.into().unwrap_or(1);
    let padding = padding.into().unwrap_or(0);
    let dilation = dilation.into().unwrap_or(1);
    let output_padding = output_padding.into().unwrap_or(0);
    let groups = groups.into().unwrap_or(1);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_conv_transpose1d(
            res,
            array.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            stride,
            padding,
            dilation,
            output_padding,
            groups,
            stream.as_ref().as_ptr(),
        )
    })
}

/// 2D transposed convolution over an input with several channels.
///
/// Only the default `groups=1` is currently supported.
///
/// The numeric parameters may be given as single values:
///
/// # Params
/// - array: input array of shape `[N, H, W, C_in]`
/// - weight: weight array of shape `[C_out, H, W, C_in]`
/// - stride: kernel stride. Default to (1, 1) if not specified.
/// - padding: input padding. Default to (0, 0) if not specified.
/// - dilation: kernel dilation. Default to (1, 1) if not specified.
/// - groups: input feature groups. Default to 1 if not specified.
/// - stream: stream or device to evaluate on.
#[allow(clippy::too_many_arguments)]
#[generate_macro]
#[default_device]
pub fn conv_transpose2d_device(
    array: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride: impl Into<Option<(i32, i32)>>,
    #[optional] padding: impl Into<Option<(i32, i32)>>,
    #[optional] dilation: impl Into<Option<(i32, i32)>>,
    #[optional] output_padding: impl Into<Option<(i32, i32)>>,
    #[optional] groups: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stride = stride.into().unwrap_or((1, 1));
    let padding = padding.into().unwrap_or((0, 0));
    let dilation = dilation.into().unwrap_or((1, 1));
    let output_padding = output_padding.into().unwrap_or((0, 0));
    let groups = groups.into().unwrap_or(1);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_conv_transpose2d(
            res,
            array.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            stride.0,
            stride.1,
            padding.0,
            padding.1,
            dilation.0,
            dilation.1,
            output_padding.0,
            output_padding.1,
            groups,
            stream.as_ref().as_ptr(),
        )
    })
}

/// 3D transposed convolution over an input with several channels.
///
/// Only the default `groups=1` is currently supported.
///
/// The numeric parameters may be given as single values:
///
/// # Params
/// - array: input array of shape `[N, D, H, W, C_in]`
/// - weight: weight array of shape `[C_out, D, H, W, C_in]`
/// - stride: kernel stride. Default to (1, 1, 1) if not specified.
/// - padding: input padding. Default to (0, 0, 0) if not specified.
/// - dilation: kernel dilation. Default to (1, 1, 1) if not specified.
/// - groups: input feature groups. Default to 1 if not specified.
/// - stream: stream or device to evaluate on.
#[allow(clippy::too_many_arguments)]
#[generate_macro]
#[default_device]
pub fn conv_transpose3d_device(
    array: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride: impl Into<Option<(i32, i32, i32)>>,
    #[optional] padding: impl Into<Option<(i32, i32, i32)>>,
    #[optional] dilation: impl Into<Option<(i32, i32, i32)>>,
    #[optional] output_padding: impl Into<Option<(i32, i32, i32)>>,
    #[optional] groups: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stride = stride.into().unwrap_or((1, 1, 1));
    let padding = padding.into().unwrap_or((0, 0, 0));
    let dilation = dilation.into().unwrap_or((1, 1, 1));
    let output_padding = output_padding.into().unwrap_or((0, 0, 0));
    let groups = groups.into().unwrap_or(1);

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_conv_transpose3d(
            res,
            array.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            stride.0,
            stride.1,
            stride.2,
            padding.0,
            padding.1,
            padding.2,
            dilation.0,
            dilation.1,
            dilation.2,
            output_padding.0,
            output_padding.1,
            output_padding.2,
            groups,
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_array_eq, tolerances};
    use pretty_assertions::assert_eq;

    #[test]
    fn test_conv1d_complex_device() {
        // Define a 1D input with two channels
        let input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let input_array = Array::from_slice(&input_data, &[1, 5, 2]);

        // Define a 1D kernel with two input channels and two output channels
        let weight_data = [0.5, 0.0, -0.5, 1.0, 0.0, 1.5, 2.0, 0.0, -2.0, 1.5, 0.0, 1.0];
        let weight_array = Array::from_slice(&weight_data, &[2, 3, 2]);

        let result = conv1d(
            &input_array,
            &weight_array,
            Some(1), // stride
            Some(0), // padding
            Some(1), // dilation
            Some(1), // groups
        )
        .unwrap();

        let expected_output = [12.0, 8.0, 17.0, 13.0, 22.0, 18.0];
        assert_eq!(result.shape(), &[1, 3, 2]);
        assert_eq!(result.as_slice::<f32>(), &expected_output);
    }

    #[test]
    fn test_conv_transpose1d() {
        // Single channel input
        let input = Array::from_slice(&[1.0, 2.0, 3.0], &[1, 3, 1]);
        // Single input/output channel kernel
        let weights = Array::from_slice(&[1.0, 0.5], &[1, 2, 1]);

        let result = conv_transpose1d(
            &input,
            &weights,
            Some(1), // stride
            Some(0), // padding
            Some(1), // dilation
            None,    // output padding
            Some(1), // groups
        )
        .unwrap();

        let expected = [1.0, 2.5, 4.0, 1.5];
        assert_eq!(result.shape(), &[1, 4, 1]);
        assert_eq!(result.as_slice::<f32>(), &expected);
    }

    #[test]
    fn test_conv2d() {
        // Define a 2x2 input with one channel (grayscale image or similar)
        let input_data = [1.0, 2.0, 3.0, 4.0];
        let input_shape = [1, 2, 2, 1]; // [N, H, W, C]
        let input_array = Array::from_slice(&input_data, &input_shape);

        // Define a 2x2 kernel with one input channel and one output channel
        let weight_data = [1.0, 0.0, 0.0, 1.0];
        let weight_shape = [1, 2, 2, 1]; // [C_out, H_k, W_k, C_in]
        let weight_array = Array::from_slice(&weight_data, &weight_shape);

        // Perform the convolution with no padding and stride of 1
        let result = conv2d(
            &input_array,
            &weight_array,
            Some((1, 1)), // stride
            Some((0, 0)), // padding
            Some((1, 1)), // dilation
            Some(1),      // groups
        )
        .unwrap();

        // Expected result is the convolution of a 2x2 filter over a 2x2 input with valid padding, resulting in a single output value
        let expected_output = 1.0 * 1.0 + 2.0 * 0.0 + 3.0 * 0.0 + 4.0 * 1.0; // = 1*1 + 4*1 = 5
        assert_eq!(result.as_slice::<f32>(), &[expected_output]);
    }

    #[test]
    fn test_conv_transpose2d() {
        // 2x2 single channel input
        let input = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
        // 2x2 single channel kernel (identity-like)
        let weights = Array::from_slice(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2, 1]);

        let result = conv_transpose2d(
            &input,
            &weights,
            Some((1, 1)), // stride
            Some((0, 0)), // padding
            Some((1, 1)), // dilation
            None,         // output padding
            Some(1),      // groups
        )
        .unwrap();

        let expected = [1.0, 2.0, 0.0, 3.0, 5.0, 2.0, 0.0, 3.0, 4.0];
        assert_eq!(result.shape(), &[1, 3, 3, 1]);
        assert_eq!(result.as_slice::<f32>(), &expected);
    }

    #[test]
    fn test_conv3d() {
        // Define a 2x2x2 input with one channel
        let input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input_shape = [1, 2, 2, 2, 1]; // [N, D, H, W, C]
        let input_array = Array::from_slice(&input_data, &input_shape);

        // Define a 2x2x2 kernel with one input channel and one output channel
        let weight_data = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let weight_shape = [1, 2, 2, 2, 1]; // [C_out, D_k, H_k, W_k, C_in]
        let weight_array = Array::from_slice(&weight_data, &weight_shape);

        // Perform the convolution with no padding and stride of 1
        let result = conv3d(
            &input_array,
            &weight_array,
            Some((1, 1, 1)), // stride
            Some((0, 0, 0)), // padding
            Some((1, 1, 1)), // dilation
            Some(1),         // groups
        )
        .unwrap();

        // Expected result is the convolution of a 2x2x2 filter over a 2x2x2 input with valid padding, resulting in a single output value
        let expected_output = 1.0 * 1.0
            + 2.0 * 0.0
            + 3.0 * 0.0
            + 4.0 * 1.0
            + 5.0 * 0.0
            + 6.0 * 1.0
            + 7.0 * 1.0
            + 8.0 * 0.0; // = 1*1 + 4*1 + 6*1 + 7*1 = 18
        assert_array_eq(
            result,
            Array::from_slice(&[expected_output], &[1, 1, 1, 1, 1]),
            tolerances::EXACT.rtol,
            tolerances::EXACT.atol,
        );
    }

    #[test]
    fn test_conv_transpose3d() {
        // 2x2x2 single channel input
        let input = Array::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 2, 2, 1]);
        // 2x2x2 single channel kernel
        let weights =
            Array::from_slice(&[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], &[1, 2, 2, 2, 1]);

        let result = conv_transpose3d(
            &input,
            &weights,
            Some((1, 1, 1)), // stride
            Some((0, 0, 0)), // padding
            Some((1, 1, 1)), // dilation
            None,            // output padding
            Some(1),         // groups
        )
        .unwrap();

        assert_eq!(result.shape(), &[1, 3, 3, 3, 1]);
    }

    #[test]
    fn test_conv_wrong_dimensions() {
        let input_data = [1.0, 2.0, 3.0, 4.0];
        let input_shape = [1, 2, 2, 1]; // [N, H, W, C]
        let input_array = Array::from_slice(&input_data, &input_shape);

        let weight_data = [1.0, 0.0, 0.0, 1.0];
        let weight_shape = [1, 2, 2]; // [C_out, H_k, W_k]
        let weight_array = Array::from_slice(&weight_data, &weight_shape);

        let result = conv2d(
            &input_array,
            &weight_array,
            Some((1, 1)), // stride
            Some((0, 0)), // padding
            Some((1, 1)), // dilation
            Some(1),      // groups
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_conv_invalid_group_size() {
        let input_data = [1.0, 2.0, 3.0, 4.0];
        let input_shape = [1, 2, 2, 1]; // [N, H, W, C]
        let input_array = Array::from_slice(&input_data, &input_shape);

        let weight_data = [1.0, 0.0, 0.0, 1.0];
        let weight_shape = [1, 2, 2, 1]; // [C_out, H_k, W_k, C_in]
        let weight_array = Array::from_slice(&weight_data, &weight_shape);

        let result = conv2d(
            &input_array,
            &weight_array,
            Some((1, 1)), // stride
            Some((0, 0)), // padding
            Some((1, 1)), // dilation
            Some(2),      // groups
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_conv_non_float() {
        let input_data = [1, 2, 3, 4];
        let input_shape = [1, 2, 2, 1]; // [N, H, W, C]
        let input_array = Array::from_slice(&input_data, &input_shape);

        let weight_data = [1, 0, 0, 1];
        let weight_shape = [1, 2, 2, 1]; // [C_out, H_k, W_k, C_in]
        let weight_array = Array::from_slice(&weight_data, &weight_shape);

        let result = conv2d(
            &input_array,
            &weight_array,
            Some((1, 1)), // stride
            Some((0, 0)), // padding
            Some((1, 1)), // dilation
            Some(1),      // groups
        );

        assert!(result.is_err());
    }
}
