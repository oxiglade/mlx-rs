use crate::error::Result;
use crate::utils::guard::Guarded;
use crate::{Array, Stream};
use mlx_internal_macros::{default_device, generate_macro};

fn optional_dtype_none() -> mlx_sys::mlx_optional_dtype {
    mlx_sys::mlx_optional_dtype {
        value: mlx_sys::mlx_dtype__MLX_FLOAT32,
        has_value: false,
    }
}

impl Array {
    /// Return the cumulative maximum of the elements along the given axis returning an error if the inputs are invalid.
    ///
    /// # Params
    ///
    /// - axis: Optional axis to compute the cumulative maximum over. If unspecified the cumulative maximum of the flattened array is returned.
    /// - reverse: If true, the cumulative maximum is computed in reverse - defaults to false if unspecified.
    /// - inclusive: If true, the i-th element of the output includes the i-th element of the input - defaults to true if unspecified.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [[5, 8], [5, 9]] -- cumulative max along the columns
    /// let result = array.cummax(0, None, None).unwrap();
    /// ```
    #[default_device]
    pub fn cummax_device(
        &self,
        axis: impl Into<Option<i32>>,
        reverse: impl Into<Option<bool>>,
        inclusive: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let stream = stream.as_ref();

        match axis.into() {
            Some(axis) => Array::try_from_op(|res| unsafe {
                mlx_sys::mlx_cummax_axis(
                    res,
                    self.as_ptr(),
                    axis,
                    reverse.into().unwrap_or(false),
                    inclusive.into().unwrap_or(true),
                    stream.as_ptr(),
                )
            }),
            None => Array::try_from_op(|res| unsafe {
                mlx_sys::mlx_cummax(
                    res,
                    self.as_ptr(),
                    reverse.into().unwrap_or(false),
                    inclusive.into().unwrap_or(true),
                    stream.as_ptr(),
                )
            }),
        }
    }

    /// Return the cumulative minimum of the elements along the given axis returning an error if the inputs are invalid.
    ///
    /// # Params
    ///
    /// - axis: Optional axis to compute the cumulative minimum over. If unspecified the cumulative maximum of the flattened array is returned.
    /// - reverse: If true, the cumulative minimum is computed in reverse - defaults to false if unspecified.
    /// - inclusive: If true, the i-th element of the output includes the i-th element of the input - defaults to true if unspecified.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [[5, 8], [4, 8]] -- cumulative min along the columns
    /// let result = array.cummin(0, None, None).unwrap();
    /// ```
    #[default_device]
    pub fn cummin_device(
        &self,
        axis: impl Into<Option<i32>>,
        reverse: impl Into<Option<bool>>,
        inclusive: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let stream = stream.as_ref();

        match axis.into() {
            Some(axis) => Array::try_from_op(|res| unsafe {
                mlx_sys::mlx_cummin_axis(
                    res,
                    self.as_ptr(),
                    axis,
                    reverse.into().unwrap_or(false),
                    inclusive.into().unwrap_or(true),
                    stream.as_ptr(),
                )
            }),
            None => Array::try_from_op(|res| unsafe {
                mlx_sys::mlx_cummin(
                    res,
                    self.as_ptr(),
                    reverse.into().unwrap_or(false),
                    inclusive.into().unwrap_or(true),
                    stream.as_ptr(),
                )
            }),
        }
    }

    /// Return the cumulative product of the elements along the given axis returning an error if the inputs are invalid.
    ///
    /// # Params
    ///
    /// - axis: Optional axis to compute the cumulative product over. If unspecified the cumulative maximum of the flattened array is returned.
    /// - reverse: If true, the cumulative product is computed in reverse - defaults to false if unspecified.
    /// - inclusive: If true, the i-th element of the output includes the i-th element of the input - defaults to true if unspecified.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [[5, 8], [20, 72]] -- cumulative min along the columns
    /// let result = array.cumprod(0, None, None).unwrap();
    /// ```
    #[default_device]
    pub fn cumprod_device(
        &self,
        axis: impl Into<Option<i32>>,
        reverse: impl Into<Option<bool>>,
        inclusive: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let stream = stream.as_ref();

        match axis.into() {
            Some(axis) => Array::try_from_op(|res| unsafe {
                mlx_sys::mlx_cumprod_axis(
                    res,
                    self.as_ptr(),
                    axis,
                    reverse.into().unwrap_or(false),
                    inclusive.into().unwrap_or(true),
                    optional_dtype_none(),
                    stream.as_ptr(),
                )
            }),
            None => Array::try_from_op(|res| unsafe {
                mlx_sys::mlx_cumprod(
                    res,
                    self.as_ptr(),
                    reverse.into().unwrap_or(false),
                    inclusive.into().unwrap_or(true),
                    optional_dtype_none(),
                    stream.as_ptr(),
                )
            }),
        }
    }

    /// Return the cumulative sum of the elements along the given axis returning an error if the inputs are invalid.
    ///
    /// # Params
    ///
    /// - axis: Optional axis to compute the cumulative sum over. If unspecified the cumulative maximum of the flattened array is returned.
    /// - reverse: If true, the cumulative sum is computed in reverse - defaults to false if unspecified.
    /// - inclusive: If true, the i-th element of the output includes the i-th element of the input - defaults to true if unspecified.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [[5, 8], [9, 17]] -- cumulative min along the columns
    /// let result = array.cumsum(0, None, None).unwrap();
    /// ```
    #[default_device]
    pub fn cumsum_device(
        &self,
        axis: impl Into<Option<i32>>,
        reverse: impl Into<Option<bool>>,
        inclusive: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let stream = stream.as_ref();

        match axis.into() {
            Some(axis) => Array::try_from_op(|res| unsafe {
                mlx_sys::mlx_cumsum_axis(
                    res,
                    self.as_ptr(),
                    axis,
                    reverse.into().unwrap_or(false),
                    inclusive.into().unwrap_or(true),
                    optional_dtype_none(),
                    stream.as_ptr(),
                )
            }),
            None => Array::try_from_op(|res| unsafe {
                mlx_sys::mlx_cumsum(
                    res,
                    self.as_ptr(),
                    reverse.into().unwrap_or(false),
                    inclusive.into().unwrap_or(true),
                    optional_dtype_none(),
                    stream.as_ptr(),
                )
            }),
        }
    }
}

/// See [`Array::cummax`]
#[generate_macro]
#[default_device]
pub fn cummax_device(
    a: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] reverse: impl Into<Option<bool>>,
    #[optional] inclusive: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().cummax_device(axis, reverse, inclusive, stream)
}

/// See [`Array::cummin`]
#[generate_macro]
#[default_device]
pub fn cummin_device(
    a: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] reverse: impl Into<Option<bool>>,
    #[optional] inclusive: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().cummin_device(axis, reverse, inclusive, stream)
}

/// See [`Array::cumprod`]
#[generate_macro]
#[default_device]
pub fn cumprod_device(
    a: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] reverse: impl Into<Option<bool>>,
    #[optional] inclusive: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().cumprod_device(axis, reverse, inclusive, stream)
}

/// See [`Array::cumsum`]
#[generate_macro]
#[default_device]
pub fn cumsum_device(
    a: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] reverse: impl Into<Option<bool>>,
    #[optional] inclusive: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    a.as_ref().cumsum_device(axis, reverse, inclusive, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_cummax() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        let result = array.cummax(0, None, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 5, 9]);

        let result = array.cummax(1, None, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 4, 9]);

        let result = array.cummax(None, None, None).unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 8, 9]);

        let result = array.cummax(0, Some(true), None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 9, 4, 9]);

        let result = array.cummax(0, None, Some(true)).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 5, 9]);
    }

    #[test]
    fn test_cummax_out_of_bounds() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.cummax(2, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cummin() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        let result = array.cummin(0, None, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 4, 8]);

        let result = array.cummin(1, None, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 5, 4, 4]);

        let result = array.cummin(None, None, None).unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.as_slice::<i32>(), &[5, 5, 4, 4]);

        let result = array.cummin(0, Some(true), None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[4, 8, 4, 9]);

        let result = array.cummin(0, None, Some(true)).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 4, 8]);
    }

    #[test]
    fn test_cummin_out_of_bounds() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.cummin(2, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cumprod() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        let result = array.cumprod(0, None, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 20, 72]);

        let result = array.cumprod(1, None, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 40, 4, 36]);

        let result = array.cumprod(None, None, None).unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.as_slice::<i32>(), &[5, 40, 160, 1440]);

        let result = array.cumprod(0, Some(true), None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[20, 72, 4, 9]);

        let result = array.cumprod(0, None, Some(true)).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 20, 72]);
    }

    #[test]
    fn test_cumprod_out_of_bounds() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.cumprod(2, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cumsum() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);

        let result = array.cumsum(0, None, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 9, 17]);

        let result = array.cumsum(1, None, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 13, 4, 13]);

        let result = array.cumsum(None, None, None).unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.as_slice::<i32>(), &[5, 13, 17, 26]);

        let result = array.cumsum(0, Some(true), None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[9, 17, 4, 9]);

        let result = array.cumsum(0, None, Some(true)).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i32>(), &[5, 8, 9, 17]);
    }

    #[test]
    fn test_cumsum_out_of_bounds() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.cumsum(2, None, None);
        assert!(result.is_err());
    }
}
