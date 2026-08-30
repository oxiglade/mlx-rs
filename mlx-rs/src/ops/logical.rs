use crate::array::Array;
use crate::error::Result;
use crate::utils::guard::Guarded;
use crate::Stream;
use mlx_internal_macros::generate_macro;

impl Array {
    /// Element-wise equality returning an error if the arrays are not broadcastable.
    ///
    /// Equality comparison on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to compare
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.eq(&b).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    pub fn eq(&self, other: impl AsRef<Array>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_equal(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`eq`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `eq`"
    )]
    pub fn eq_device(&self, other: impl AsRef<Array>, stream: impl AsRef<Stream>) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.eq(other))
    }

    /// Element-wise less than or equal returning an error if the arrays are not broadcastable.
    ///
    /// Less than or equal on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to compare
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.le(&b).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    pub fn le(&self, other: impl AsRef<Array>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_less_equal(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`le`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `le`"
    )]
    pub fn le_device(&self, other: impl AsRef<Array>, stream: impl AsRef<Stream>) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.le(other))
    }

    /// Element-wise greater than or equal returning an error if the arrays are not broadcastable.
    ///
    /// Greater than or equal on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to compare
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.ge(&b).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    pub fn ge(&self, other: impl AsRef<Array>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_greater_equal(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`ge`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `ge`"
    )]
    pub fn ge_device(&self, other: impl AsRef<Array>, stream: impl AsRef<Stream>) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.ge(other))
    }

    /// Element-wise not equal returning an error if the arrays are not broadcastable.
    ///
    /// Not equal on two arrays with
    /// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to compare
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.ne(&b).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [false, false, false]
    /// ```
    pub fn ne(&self, other: impl AsRef<Array>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_not_equal(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`ne`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `ne`"
    )]
    pub fn ne_device(&self, other: impl AsRef<Array>, stream: impl AsRef<Stream>) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.ne(other))
    }

    /// Element-wise less than returning an error if the arrays are not broadcastable.
    ///
    /// Less than on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to compare
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.lt(&b).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [false, false, false]
    /// ```
    pub fn lt(&self, other: impl AsRef<Array>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_less(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`lt`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `lt`"
    )]
    pub fn lt_device(&self, other: impl AsRef<Array>, stream: impl AsRef<Stream>) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.lt(other))
    }

    /// Element-wise greater than returning an error if the arrays are not broadcastable.
    ///
    /// Greater than on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to compare
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[1, 2, 3], &[3]);
    /// let b = Array::from_slice(&[1, 2, 3], &[3]);
    /// let mut c = a.gt(&b).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [false, false, false]
    /// ```
    pub fn gt(&self, other: impl AsRef<Array>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_greater(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`gt`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `gt`"
    )]
    pub fn gt_device(&self, other: impl AsRef<Array>, stream: impl AsRef<Stream>) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.gt(other))
    }

    /// Element-wise logical and returning an error if the arrays are not broadcastable.
    ///
    /// Logical and on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to compare
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[true, false, true], &[3]);
    /// let b = Array::from_slice(&[true, true, false], &[3]);
    /// let mut c = a.logical_and(&b).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, false, false]
    /// ```
    pub fn logical_and(&self, other: impl AsRef<Array>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_logical_and(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`logical_and`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `logical_and`"
    )]
    pub fn logical_and_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.logical_and(other))
    }

    /// Element-wise logical or returning an error if the arrays are not broadcastable.
    ///
    /// Logical or on two arrays with [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    ///
    /// # Params
    ///
    /// - other: array to compare
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[true, false, true], &[3]);
    /// let b = Array::from_slice(&[true, true, false], &[3]);
    /// let mut c = a.logical_or(&b).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true, true, true]
    /// ```
    pub fn logical_or(&self, other: impl AsRef<Array>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_logical_or(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`logical_or`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `logical_or`"
    )]
    pub fn logical_or_device(
        &self,
        other: impl AsRef<Array>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.logical_or(other))
    }

    /// Unary element-wise logical not.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a: Array = false.into();
    /// let mut b = a.logical_not().unwrap();
    ///
    /// let b_data: &[bool] = b.as_slice();
    /// // b_data == [true]
    /// ```
    pub fn logical_not(&self) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_logical_not(res, self.as_ptr(), stream.as_ref().as_ptr())
        })
    }

    /// Compatibility shim for [`logical_not`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `logical_not`"
    )]
    pub fn logical_not_device(&self, stream: impl AsRef<Stream>) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.logical_not())
    }

    /// Approximate comparison of two arrays returning an error if the inputs aren't valid.
    ///
    /// The arrays are considered equal if:
    ///
    /// ```text
    /// all(abs(a - b) <= (atol + rtol * abs(b)))
    /// ```
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - rtol: relative tolerance = defaults to 1e-5 when None
    /// - atol: absolute tolerance - defaults to 1e-8 when None
    /// - equal_nan: whether to consider NaNs equal -- default is false when None
    ///
    /// # Example
    ///
    /// ```rust
    /// use num_traits::Pow;
    /// use mlx_rs::array;
    /// let a = array!([0., 1., 2., 3.]).sqrt().unwrap();
    /// let b = array!([0., 1., 2., 3.]).power(array!(0.5)).unwrap();
    /// let mut c = a.all_close(&b, None, None, None).unwrap();
    ///
    /// let c_data: &[bool] = c.as_slice();
    /// // c_data == [true]
    /// ```
    pub fn all_close(
        &self,
        other: impl AsRef<Array>,
        rtol: impl Into<Option<f64>>,
        atol: impl Into<Option<f64>>,
        equal_nan: impl Into<Option<bool>>,
    ) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_allclose(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                rtol.into().unwrap_or(1e-5),
                atol.into().unwrap_or(1e-8),
                equal_nan.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`all_close`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `all_close`"
    )]
    pub fn all_close_device(
        &self,
        other: impl AsRef<Array>,
        rtol: impl Into<Option<f64>>,
        atol: impl Into<Option<f64>>,
        equal_nan: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || {
            self.all_close(other, rtol, atol, equal_nan)
        })
    }

    /// Returns a boolean array where two arrays are element-wise equal within a tolerance returning an error if the arrays are not broadcastable.
    ///
    /// Infinite values are considered equal if they have the same sign, NaN values are not equal unless
    /// `equalNAN` is `true`.
    ///
    /// Two values are considered close if:
    ///
    /// ```text
    /// abs(a - b) <= (atol + rtol * abs(b))
    /// ```
    ///
    /// Unlike [self.array_eq] this function supports [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting).
    pub fn is_close(
        &self,
        other: impl AsRef<Array>,
        rtol: impl Into<Option<f64>>,
        atol: impl Into<Option<f64>>,
        equal_nan: impl Into<Option<bool>>,
    ) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_isclose(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                rtol.into().unwrap_or(1e-5),
                atol.into().unwrap_or(1e-8),
                equal_nan.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`is_close`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `is_close`"
    )]
    pub fn is_close_device(
        &self,
        other: impl AsRef<Array>,
        rtol: impl Into<Option<f64>>,
        atol: impl Into<Option<f64>>,
        equal_nan: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || {
            self.is_close(other, rtol, atol, equal_nan)
        })
    }

    /// Array equality check.
    ///
    /// Compare two arrays for equality. Returns `true` iff the arrays have
    /// the same shape and their values are equal. The arrays need not have
    /// the same type to be considered equal.
    ///
    /// # Params
    ///
    /// - other: array to compare
    /// - equal_nan: whether to consider NaNs equal -- default is false when None
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[0, 1, 2, 3], &[4]);
    /// let b = Array::from_slice(&[0., 1., 2., 3.], &[4]);
    ///
    /// let c = a.array_eq(&b, None);
    /// // c == [true]
    /// ```
    pub fn array_eq(
        &self,
        other: impl AsRef<Array>,
        equal_nan: impl Into<Option<bool>>,
    ) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_array_equal(
                res,
                self.as_ptr(),
                other.as_ref().as_ptr(),
                equal_nan.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`array_eq`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `array_eq`"
    )]
    pub fn array_eq_device(
        &self,
        other: impl AsRef<Array>,
        equal_nan: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.array_eq(other, equal_nan))
    }

    /// An `or` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over -- defaults to all axes if not provided
    /// - keep_dims: if `true` keep reduced axis as singleton dimension -- defaults to false if not provided
    ///
    ///  # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    ///
    /// let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
    ///
    /// // will produce a scalar Array with true -- some of the values are non-zero
    /// let all = array.any(None).unwrap();
    ///
    /// // produces an Array([true, true, true, true]) -- all rows have non-zeros
    /// let all_rows = array.any_axes(&[0], None).unwrap();
    /// ```
    pub fn any_axes(&self, axes: &[i32], keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_any_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`any_axes`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `any_axes`"
    )]
    pub fn any_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.any_axes(axes, keep_dims))
    }

    /// Similar to [`any_axes`] but defaults to all axes.
    pub fn any_axis(&self, axis: i32, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_any_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`any_axis`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `any_axis`"
    )]
    pub fn any_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.any_axis(axis, keep_dims))
    }

    /// Similar to [`any_axes`] but defaults to all axes.
    pub fn any(&self, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_any(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`any`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `any`"
    )]
    pub fn any_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.any(keep_dims))
    }
}

/// See [`Array::any`]
pub fn any_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().any_axes(axes, keep_dims)
}

/// Compatibility shim for [`any_axes`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `any_axes`"
)]
pub fn any_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || any_axes(array, axes, keep_dims))
}

/// See [`Array::any`]
pub fn any_axis(
    array: impl AsRef<Array>,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().any_axis(axis, keep_dims)
}

/// Compatibility shim for [`any_axis`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `any_axis`"
)]
pub fn any_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || any_axis(array, axis, keep_dims))
}

/// See [`Array::any`]
pub fn any(array: impl AsRef<Array>, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
    array.as_ref().any(keep_dims)
}

/// Compatibility shim for [`any`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `any`"
)]
pub fn any_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || any(array, keep_dims))
}

/// See [`Array::logical_and`]
pub fn logical_and(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().logical_and(b)
}

/// Compatibility shim for [`logical_and`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `logical_and`"
)]
pub fn logical_and_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || logical_and(a, b))
}

/// See [`Array::logical_or`]
pub fn logical_or(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().logical_or(b)
}

/// Compatibility shim for [`logical_or`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `logical_or`"
)]
pub fn logical_or_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || logical_or(a, b))
}

/// See [`Array::logical_not`]
pub fn logical_not(a: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().logical_not()
}

/// Compatibility shim for [`logical_not`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `logical_not`"
)]
pub fn logical_not_device(
    a: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || logical_not(a))
}

/// See [`Array::all_close`]
pub fn all_close(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    rtol: impl Into<Option<f64>>,
    atol: impl Into<Option<f64>>,
    equal_nan: impl Into<Option<bool>>,
) -> Result<Array> {
    a.as_ref().all_close(b, rtol, atol, equal_nan)
}

/// Compatibility shim for [`all_close`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `all_close`"
)]
pub fn all_close_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] rtol: impl Into<Option<f64>>,
    #[optional] atol: impl Into<Option<f64>>,
    #[optional] equal_nan: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || all_close(a, b, rtol, atol, equal_nan))
}

/// See [`Array::is_close`]
pub fn is_close(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    rtol: impl Into<Option<f64>>,
    atol: impl Into<Option<f64>>,
    equal_nan: impl Into<Option<bool>>,
) -> Result<Array> {
    a.as_ref().is_close(b, rtol, atol, equal_nan)
}

/// Compatibility shim for [`is_close`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `is_close`"
)]
pub fn is_close_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] rtol: impl Into<Option<f64>>,
    #[optional] atol: impl Into<Option<f64>>,
    #[optional] equal_nan: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || is_close(a, b, rtol, atol, equal_nan))
}

/// See [`Array::array_eq`]
pub fn array_eq(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    equal_nan: impl Into<Option<bool>>,
) -> Result<Array> {
    a.as_ref().array_eq(b, equal_nan)
}

/// Compatibility shim for [`array_eq`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `array_eq`"
)]
pub fn array_eq_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] equal_nan: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || array_eq(a, b, equal_nan))
}

/// See [`Array::eq`]
pub fn eq(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().eq(b)
}

/// Compatibility shim for [`eq`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `eq`"
)]
pub fn eq_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || eq(a, b))
}

/// See [`Array::le`]
pub fn le(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().le(b)
}

/// Compatibility shim for [`le`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `le`"
)]
pub fn le_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || le(a, b))
}

/// See [`Array::ge`]
pub fn ge(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().ge(b)
}

/// Compatibility shim for [`ge`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `ge`"
)]
pub fn ge_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || ge(a, b))
}

/// See [`Array::ne`]
pub fn ne(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().ne(b)
}

/// Compatibility shim for [`ne`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `ne`"
)]
pub fn ne_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || ne(a, b))
}

/// See [`Array::lt`]
pub fn lt(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().lt(b)
}

/// Compatibility shim for [`lt`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `lt`"
)]
pub fn lt_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || lt(a, b))
}

/// See [`Array::gt`]
pub fn gt(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    a.as_ref().gt(b)
}

/// Compatibility shim for [`gt`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `gt`"
)]
pub fn gt_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || gt(a, b))
}
// TODO: check if the functions below could throw an exception.

/// Return a boolean array indicating which elements are NaN.
pub fn is_nan(array: impl AsRef<Array>) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_isnan(res, array.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Compatibility shim for [`is_nan`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `is_nan`"
)]
pub fn is_nan_device(
    array: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || is_nan(array))
}

/// Return a boolean array indicating which elements are +/- inifnity.
pub fn is_inf(array: impl AsRef<Array>) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_isinf(res, array.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Compatibility shim for [`is_inf`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `is_inf`"
)]
pub fn is_inf_device(
    array: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || is_inf(array))
}

/// Return a boolean array indicating which elements are positive infinity.
pub fn is_pos_inf(array: impl AsRef<Array>) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_isposinf(res, array.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Compatibility shim for [`is_pos_inf`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `is_pos_inf`"
)]
pub fn is_pos_inf_device(
    array: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || is_pos_inf(array))
}

/// Return a boolean array indicating which elements are negative infinity.
pub fn is_neg_inf(array: impl AsRef<Array>) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_isneginf(res, array.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Compatibility shim for [`is_neg_inf`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `is_neg_inf`"
)]
pub fn is_neg_inf_device(
    array: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || is_neg_inf(array))
}

/// Select from `a` or `b` according to `condition` returning an error if the arrays are not
/// broadcastable.
///
/// The condition and input arrays must be the same shape or
/// [broadcasting](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/broadcasting)
/// with each another.
///
/// # Params
///
/// - condition: condition array
/// - a: input selected from where condition is non-zero or `true`
/// - b: input selected from where condition is zero or `false`
pub fn r#where(
    condition: impl AsRef<Array>,
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_where(
            res,
            condition.as_ref().as_ptr(),
            a.as_ref().as_ptr(),
            b.as_ref().as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`r#where`].
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `r#where`"
)]
pub fn r#where_device(
    condition: impl AsRef<Array>,
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || r#where(condition, a, b))
}

/// Alias for [`r#where`]
pub fn which(
    condition: impl AsRef<Array>,
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
) -> Result<Array> {
    r#where(condition, a, b)
}

/// Compatibility shim for [`which`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `which`"
)]
pub fn which_device(
    condition: impl AsRef<Array>,
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || which(condition, a, b))
}

#[cfg(test)]
mod tests {
    use crate::{array, Dtype};

    use super::*;

    #[test]
    fn test_eq() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let c = a.eq(&b).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true, true, true]);

        // check a and b are not modified
        let a_data: &[i32] = a.as_slice();
        assert_eq!(a_data, [1, 2, 3]);

        let b_data: &[i32] = b.as_slice();
        assert_eq!(b_data, [1, 2, 3]);
    }

    #[test]
    fn test_eq_invalid_broadcast() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3, 4], &[4]);
        let c = a.eq(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_le() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let c = a.le(&b).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true, true, true]);

        // check a and b are not modified
        let a_data: &[i32] = a.as_slice();
        assert_eq!(a_data, [1, 2, 3]);

        let b_data: &[i32] = b.as_slice();
        assert_eq!(b_data, [1, 2, 3]);
    }

    #[test]
    fn test_le_invalid_broadcast() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3, 4], &[4]);
        let c = a.le(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_ge() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let c = a.ge(&b).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true, true, true]);

        // check a and b are not modified
        let a_data: &[i32] = a.as_slice();
        assert_eq!(a_data, [1, 2, 3]);

        let b_data: &[i32] = b.as_slice();
        assert_eq!(b_data, [1, 2, 3]);
    }

    #[test]
    fn test_ge_invalid_broadcast() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3, 4], &[4]);
        let c = a.ge(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_ne() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let c = a.ne(&b).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [false, false, false]);

        // check a and b are not modified
        let a_data: &[i32] = a.as_slice();
        assert_eq!(a_data, [1, 2, 3]);

        let b_data: &[i32] = b.as_slice();
        assert_eq!(b_data, [1, 2, 3]);
    }

    #[test]
    fn test_ne_invalid_broadcast() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3, 4], &[4]);
        let c = a.ne(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_lt() {
        let a = Array::from_slice(&[1, 0, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let c = a.lt(&b).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [false, true, false]);

        // check a and b are not modified
        let a_data: &[i32] = a.as_slice();
        assert_eq!(a_data, [1, 0, 3]);

        let b_data: &[i32] = b.as_slice();
        assert_eq!(b_data, [1, 2, 3]);
    }

    #[test]
    fn test_lt_invalid_broadcast() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3, 4], &[4]);
        let c = a.lt(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_gt() {
        let a = Array::from_slice(&[1, 4, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3], &[3]);
        let c = a.gt(&b).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [false, true, false]);

        // check a and b are not modified
        let a_data: &[i32] = a.as_slice();
        assert_eq!(a_data, [1, 4, 3]);

        let b_data: &[i32] = b.as_slice();
        assert_eq!(b_data, [1, 2, 3]);
    }

    #[test]
    fn test_gt_invalid_broadcast() {
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[1, 2, 3, 4], &[4]);
        let c = a.gt(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_logical_and() {
        let a = Array::from_slice(&[true, false, true], &[3]);
        let b = Array::from_slice(&[true, true, false], &[3]);
        let c = a.logical_and(&b).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true, false, false]);

        // check a and b are not modified
        let a_data: &[bool] = a.as_slice();
        assert_eq!(a_data, [true, false, true]);

        let b_data: &[bool] = b.as_slice();
        assert_eq!(b_data, [true, true, false]);
    }

    #[test]
    fn test_logical_and_invalid_broadcast() {
        let a = Array::from_slice(&[true, false, true], &[3]);
        let b = Array::from_slice(&[true, true, false, true], &[4]);
        let c = a.logical_and(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_logical_or() {
        let a = Array::from_slice(&[true, false, true], &[3]);
        let b = Array::from_slice(&[true, true, false], &[3]);
        let c = a.logical_or(&b).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true, true, true]);

        // check a and b are not modified
        let a_data: &[bool] = a.as_slice();
        assert_eq!(a_data, [true, false, true]);

        let b_data: &[bool] = b.as_slice();
        assert_eq!(b_data, [true, true, false]);
    }

    #[test]
    fn test_logical_or_invalid_broadcast() {
        let a = Array::from_slice(&[true, false, true], &[3]);
        let b = Array::from_slice(&[true, true, false, true], &[4]);
        let c = a.logical_or(&b);
        assert!(c.is_err());
    }

    #[test]
    fn test_all_close() {
        let a = Array::from_slice(&[0., 1., 2., 3.], &[4]).sqrt().unwrap();
        let b = Array::from_slice(&[0., 1., 2., 3.], &[4])
            .power(array!(0.5))
            .unwrap();
        let c = a.all_close(&b, 1e-5, None, None).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true]);
    }

    #[test]
    fn test_all_close_invalid_broadcast() {
        let a = Array::from_slice(&[0., 1., 2., 3.], &[4]);
        let b = Array::from_slice(&[0., 1., 2., 3., 4.], &[5]);
        let c = a.all_close(&b, 1e-5, None, None);
        assert!(c.is_err());
    }

    #[test]
    fn test_is_close_false() {
        let a = Array::from_slice(&[1., 2., 3.], &[3]);
        let b = Array::from_slice(&[1.1, 2.2, 3.3], &[3]);
        let c = a.is_close(&b, None, None, false).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [false, false, false]);
    }

    #[test]
    fn test_is_close_true() {
        let a = Array::from_slice(&[1., 2., 3.], &[3]);
        let b = Array::from_slice(&[1.1, 2.2, 3.3], &[3]);
        let c = a.is_close(&b, 0.1, 0.2, true).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true, true, true]);
    }

    #[test]
    fn test_is_close_invalid_broadcast() {
        let a = Array::from_slice(&[1., 2., 3.], &[3]);
        let b = Array::from_slice(&[1.1, 2.2, 3.3, 4.4], &[4]);
        let c = a.is_close(&b, None, None, false);
        assert!(c.is_err());
    }

    #[test]
    fn test_array_eq() {
        let a = Array::from_slice(&[0, 1, 2, 3], &[4]);
        let b = Array::from_slice(&[0., 1., 2., 3.], &[4]);
        let c = a.array_eq(&b, None).unwrap();

        let c_data: &[bool] = c.as_slice();
        assert_eq!(c_data, [true]);
    }

    #[test]
    fn test_any() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let all = array.any_axes(&[0][..], None).unwrap();

        let results: &[bool] = all.as_slice();
        assert_eq!(results, &[true, true, true, true]);
    }

    #[test]
    fn test_any_empty_axes() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let all = array.any_axes(&[][..], None).unwrap();

        let results: &[bool] = all.as_slice();
        assert_eq!(
            results,
            &[false, true, true, true, true, true, true, true, true, true, true, true]
        );
    }

    #[test]
    fn test_any_out_of_bounds() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[12]);
        let result = array.any_axes(&[1][..], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_any_duplicate_axes() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let result = array.any_axes(&[0, 0][..], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_which() {
        let condition = Array::from_slice(&[true, false, true], &[3]);
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[4, 5, 6], &[3]);
        let c = which(&condition, &a, &b).unwrap();

        let c_data: &[i32] = c.as_slice();
        assert_eq!(c_data, [1, 5, 3]);
    }

    #[test]
    fn test_which_invalid_broadcast() {
        let condition = Array::from_slice(&[true, false, true], &[3]);
        let a = Array::from_slice(&[1, 2, 3], &[3]);
        let b = Array::from_slice(&[4, 5, 6, 7], &[4]);
        let c = which(&condition, &a, &b);
        assert!(c.is_err());
    }

    // The unit tests below are adapted from the mlx c++ codebase

    #[test]
    fn test_unary_logical_not() {
        let x = array!(false);
        assert!(logical_not(&x).unwrap().item::<bool>());

        let x = array!(1.0);
        let y = logical_not(&x).unwrap();
        assert_eq!(y.dtype(), Dtype::Bool);
        assert!(!y.item::<bool>());

        let x = array!(0);
        let y = logical_not(&x).unwrap();
        assert_eq!(y.dtype(), Dtype::Bool);
        assert!(y.item::<bool>());
    }

    #[test]
    fn test_unary_logical_and() {
        let x = array!(true);
        let y = array!(true);
        assert!(logical_and(&x, &y).unwrap().item::<bool>());

        let x = array!(1.0);
        let y = array!(1.0);
        let z = logical_and(&x, &y).unwrap();
        assert_eq!(z.dtype(), Dtype::Bool);
        assert!(z.item::<bool>());

        let x = array!(0);
        let y = array!(1.0);
        let z = logical_and(&x, &y).unwrap();
        assert_eq!(z.dtype(), Dtype::Bool);
        assert!(!z.item::<bool>());
    }

    #[test]
    fn test_unary_logical_or() {
        let a = array!(false);
        let b = array!(false);
        assert!(!logical_or(&a, &b).unwrap().item::<bool>());

        let a = array!(1.0);
        let b = array!(1.0);
        let c = logical_or(&a, &b).unwrap();
        assert_eq!(c.dtype(), Dtype::Bool);
        assert!(c.item::<bool>());

        let a = array!(0);
        let b = array!(1.0);
        let c = logical_or(&a, &b).unwrap();
        assert_eq!(c.dtype(), Dtype::Bool);
        assert!(c.item::<bool>());
    }
}
