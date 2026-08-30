use crate::array::Array;
use crate::error::Result;
use crate::utils::axes_or_default_to_all;
use crate::utils::guard::Guarded;
use crate::Stream;
use mlx_internal_macros::{default_device, generate_macro};

impl Array {
    /// An `and` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: The axes to reduce over -- defaults to all axes if not provided
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let a = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
    /// let mut b = a.all_axes(&[0], None).unwrap();
    ///
    /// let results: &[bool] = b.as_slice();
    /// // results == [false, true, true, true]
    /// ```
    #[default_device]
    pub fn all_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_all_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::all_axes`] but only reduces over a single axis.
    #[default_device]
    pub fn all_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_all_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::all_axes`] but reduces over all axes.
    #[default_device]
    pub fn all_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_all(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// A `product` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [20, 72]
    /// let result = array.prod_axes(&[0], None).unwrap();
    /// ```
    #[default_device]
    pub fn prod_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_prod_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::prod_axes`] but only reduces over a single axis.
    #[default_device]
    pub fn prod_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_prod_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::prod_axes`] but reduces over all axes.
    #[default_device]
    pub fn prod_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_prod(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// A `max` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [5, 9]
    /// let result = array.max_axes(&[0], None).unwrap();
    /// ```
    #[default_device]
    pub fn max_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_max_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::max_axes`] but only reduces over a single axis.
    #[default_device]
    pub fn max_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_max_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::max_axes`] but reduces over all axes.
    #[default_device]
    pub fn max_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_max(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Sum reduce the array over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: if `true`, keep the reduces axes as singleton dimensions
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [9, 17]
    /// let result = array.sum_axes(&[0], None).unwrap();
    /// ```
    #[default_device]
    pub fn sum_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_sum_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::sum_axes`] but only reduces over a single axis.
    #[default_device]
    pub fn sum_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_sum_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::sum_axes`] but reduces over all axes.
    #[default_device]
    pub fn sum_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_sum(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// A `mean` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [4.5, 8.5]
    /// let result = array.mean_axes(&[0], None).unwrap();
    /// ```
    #[default_device]
    pub fn mean_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        let axes = axes_or_default_to_all(axes, self.ndim() as i32);
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_mean_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::mean_axes`] but only reduces over a single axis.
    #[default_device]
    pub fn mean_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_mean_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::mean_axes`] but reduces over all axes.
    #[default_device]
    pub fn mean_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_mean(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// A `min` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlx_rs::Array;
    /// let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
    ///
    /// // result is [4, 8]
    /// let result = array.min_axes(&[0], None).unwrap();
    /// ```
    #[default_device]
    pub fn min_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_min_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::min_axes`] but only reduces over a single axis.
    #[default_device]
    pub fn min_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_min_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::min_axes`] but reduces over all axes.
    #[default_device]
    pub fn min_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_min(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compute the variance(s) over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: if `true`, keep the reduces axes as singleton dimensions
    /// - ddof: the divisor to compute the variance is `N - ddof`
    #[default_device]
    pub fn var_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        ddof: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_var_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                ddof.into().unwrap_or(0),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::var_axes`] but only reduces over a single axis.
    #[default_device]
    pub fn var_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        ddof: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_var_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                ddof.into().unwrap_or(0),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::var_axes`] but reduces over all axes.
    #[default_device]
    pub fn var_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        ddof: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_var(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                ddof.into().unwrap_or(0),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compute the median over the given axes.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    #[default_device]
    pub fn median_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_median_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::median_axes`] but only reduces over a single axis.
    #[default_device]
    pub fn median_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_median_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compute the median over all axes.
    #[default_device]
    pub fn median_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_median(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// A `log-sum-exp` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// The log-sum-exp reduction is a numerically stable version of using the individual operations.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    #[default_device]
    pub fn logsumexp_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_logsumexp_axes(
                res,
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::logsumexp_axes`] but only reduces over a single axis.
    #[default_device]
    pub fn logsumexp_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_logsumexp_axis(
                res,
                self.as_ptr(),
                axis,
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Similar to [`Array::logsumexp_axes`] but reduces over all axes.
    #[default_device]
    pub fn logsumexp_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_logsumexp(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }
}

/// See [`Array::all_axes`]
#[generate_macro]
#[default_device]
pub fn all_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().all_axes_device(axes, keep_dims, stream)
}

/// See [`Array::all_axis`]
#[generate_macro]
#[default_device]
pub fn all_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().all_axis_device(axis, keep_dims, stream)
}

/// See [`Array::all`]
#[generate_macro]
#[default_device]
pub fn all_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().all_device(keep_dims, stream)
}

/// See [`Array::prod_axes`]
#[generate_macro]
#[default_device]
pub fn prod_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().prod_axes_device(axes, keep_dims, stream)
}

/// See [`Array::prod_axis`]
#[generate_macro]
#[default_device]
pub fn prod_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().prod_axis_device(axis, keep_dims, stream)
}

/// See [`Array::prod`]
#[generate_macro]
#[default_device]
pub fn prod_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().prod_device(keep_dims, stream)
}

/// See [`Array::max_axes`]
#[generate_macro]
#[default_device]
pub fn max_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().max_axes_device(axes, keep_dims, stream)
}

/// See [`Array::max_axis`]
#[generate_macro]
#[default_device]
pub fn max_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().max_axis_device(axis, keep_dims, stream)
}

/// See [`Array::max`]
#[generate_macro]
#[default_device]
pub fn max_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().max_device(keep_dims, stream)
}

/// Compute the standard deviation(s) over the given axes.
///
/// # Params
///
/// - `a`: Input array
/// - `axes`: Optional axis or axes to reduce over. If unspecified this defaults to reducing over
///   the entire array.
/// - `keep_dims`: Keep reduced axes as singleton dimensions, defaults to False.
/// - `ddof`: The divisor to compute the variance is `N - ddof`, defaults to `0`.
#[generate_macro]
#[default_device]
pub fn std_axes_device(
    a: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let keep_dims = keep_dims.into().unwrap_or(false);
    let ddof = ddof.into().unwrap_or(0);
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_std_axes(
            res,
            a.as_ptr(),
            axes.as_ptr(),
            axes.len(),
            keep_dims,
            ddof,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Similar to [`std_axes`] but only reduces over a single axis.
#[generate_macro]
#[default_device]
pub fn std_axis_device(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let keep_dims = keep_dims.into().unwrap_or(false);
    let ddof = ddof.into().unwrap_or(0);
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_std_axis(
            res,
            a.as_ptr(),
            axis,
            keep_dims,
            ddof,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Similar to [`std_axes`] but reduces over all axes.
#[generate_macro]
#[default_device]
pub fn std_device(
    a: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let keep_dims = keep_dims.into().unwrap_or(false);
    let ddof = ddof.into().unwrap_or(0);
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_std(res, a.as_ptr(), keep_dims, ddof, stream.as_ref().as_ptr())
    })
}

/// See [`Array::sum_axes`]
#[generate_macro]
#[default_device]
pub fn sum_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().sum_axes_device(axes, keep_dims, stream)
}

/// See [`Array::sum_axis`]
#[generate_macro]
#[default_device]
pub fn sum_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().sum_axis_device(axis, keep_dims, stream)
}

/// See [`Array::sum`]
#[generate_macro]
#[default_device]
pub fn sum_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().sum_device(keep_dims, stream)
}

/// See [`Array::mean_axes`]
#[generate_macro]
#[default_device]
pub fn mean_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().mean_axes_device(axes, keep_dims, stream)
}

/// See [`Array::mean_axis`]
#[generate_macro]
#[default_device]
pub fn mean_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().mean_axis_device(axis, keep_dims, stream)
}

/// See [`Array::mean`]
#[generate_macro]
#[default_device]
pub fn mean_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().mean_device(keep_dims, stream)
}

/// See [`Array::min`]
#[generate_macro]
#[default_device]
pub fn min_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().min_axes_device(axes, keep_dims, stream)
}

/// See [`Array::min_axis`]
#[generate_macro]
#[default_device]
pub fn min_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().min_axis_device(axis, keep_dims, stream)
}

/// See [`Array::min`]
#[generate_macro]
#[default_device]
pub fn min_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().min_device(keep_dims, stream)
}

/// See [`Array::var_axes`]
#[generate_macro]
#[default_device]
pub fn var_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array
        .as_ref()
        .var_axes_device(axes, keep_dims, ddof, stream)
}

/// See [`Array::var_axis`]
#[generate_macro]
#[default_device]
pub fn var_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array
        .as_ref()
        .var_axis_device(axis, keep_dims, ddof, stream)
}

/// See [`Array::var`]
#[generate_macro]
#[default_device]
pub fn var_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().var_device(keep_dims, ddof, stream)
}

/// See [`Array::median_axes`]
#[generate_macro]
#[default_device]
pub fn median_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().median_axes_device(axes, keep_dims, stream)
}

/// See [`Array::median_axis`]
#[generate_macro]
#[default_device]
pub fn median_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().median_axis_device(axis, keep_dims, stream)
}

/// See [`Array::median`]
#[generate_macro]
#[default_device]
pub fn median_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().median_device(keep_dims, stream)
}

/// See [`Array::logsumexp_axes`]
#[generate_macro]
#[default_device]
pub fn logsumexp_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array
        .as_ref()
        .logsumexp_axes_device(axes, keep_dims, stream)
}

/// See [`Array::logsumexp_axis`]
#[generate_macro]
#[default_device]
pub fn logsumexp_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array
        .as_ref()
        .logsumexp_axis_device(axis, keep_dims, stream)
}

/// See [`Array::logsumexp`]
#[generate_macro]
#[default_device]
pub fn logsumexp_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    array.as_ref().logsumexp_device(keep_dims, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_all() {
        let array = Array::from_slice(&[true, false, true, false], &[2, 2]);

        assert_eq!(array.all(None).unwrap().item::<bool>(), false);
        assert_eq!(array.all(true).unwrap().shape(), &[1, 1]);
        assert_eq!(array.all_axes(&[0, 1], None).unwrap().item::<bool>(), false);

        let result = array.all_axis(0, None).unwrap();
        assert_eq!(result.as_slice::<bool>(), &[true, false]);

        let result = array.all_axis(1, None).unwrap();
        assert_eq!(result.as_slice::<bool>(), &[false, false]);
    }

    #[test]
    fn test_all_empty_axes() {
        let array = Array::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], &[3, 4]);
        let all = array.all_axes(&[], None).unwrap();

        let results: &[bool] = all.as_slice();
        assert_eq!(
            results,
            &[false, true, true, true, true, true, true, true, true, true, true, true]
        );
    }

    #[test]
    fn test_prod() {
        let x = Array::from_slice(&[1, 2, 3, 3], &[2, 2]);
        assert_eq!(x.prod(None).unwrap().item::<i32>(), 18);

        let y = x.prod(true).unwrap();
        assert_eq!(y.item::<i32>(), 18);
        assert_eq!(y.shape(), &[1, 1]);

        let result = x.prod_axis(0, None).unwrap();
        assert_eq!(result.as_slice::<i32>(), &[3, 6]);

        let result = x.prod_axis(1, None).unwrap();
        assert_eq!(result.as_slice::<i32>(), &[2, 9])
    }

    #[test]
    fn test_prod_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.prod_axes(&[], None).unwrap();

        let results: &[i32] = result.as_slice();
        assert_eq!(results, &[5, 8, 4, 9]);
    }

    #[test]
    fn test_max() {
        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(x.max(None).unwrap().item::<i32>(), 4);
        let y = x.max(true).unwrap();
        assert_eq!(y.item::<i32>(), 4);
        assert_eq!(y.shape(), &[1, 1]);

        let result = x.max_axis(0, None).unwrap();
        assert_eq!(result.as_slice::<i32>(), &[3, 4]);

        let result = x.max_axis(1, None).unwrap();
        assert_eq!(result.as_slice::<i32>(), &[2, 4]);
    }

    #[test]
    fn test_max_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.max_axes(&[], None).unwrap();

        let results: &[i32] = result.as_slice();
        assert_eq!(results, &[5, 8, 4, 9]);
    }

    #[test]
    fn test_sum() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.sum_axis(0, None).unwrap();

        let results: &[i32] = result.as_slice();
        assert_eq!(results, &[9, 17]);
    }

    #[test]
    fn test_sum_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.sum_axes(&[], None).unwrap();

        let results: &[i32] = result.as_slice();
        assert_eq!(results, &[5, 8, 4, 9]);
    }

    #[test]
    fn test_mean() {
        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(x.mean(None).unwrap().item::<f32>(), 2.5);
        let y = x.mean(true).unwrap();
        assert_eq!(y.item::<f32>(), 2.5);
        assert_eq!(y.shape(), &[1, 1]);

        let result = x.mean_axis(0, None).unwrap();
        assert_eq!(result.as_slice::<f32>(), &[2.0, 3.0]);

        let result = x.mean_axis(1, None).unwrap();
        assert_eq!(result.as_slice::<f32>(), &[1.5, 3.5]);
    }

    #[test]
    fn test_mean_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.mean_axes(&[], None).unwrap();

        let results: &[f32] = result.as_slice();
        assert_eq!(results, &[5.0, 8.0, 4.0, 9.0]);
    }

    #[test]
    fn test_mean_out_of_bounds() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.mean_axis(2, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_min() {
        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(x.min(None).unwrap().item::<i32>(), 1);
        let y = x.min(true).unwrap();
        assert_eq!(y.item::<i32>(), 1);
        assert_eq!(y.shape(), &[1, 1]);

        let result = x.min_axis(0, None).unwrap();
        assert_eq!(result.as_slice::<i32>(), &[1, 2]);

        let result = x.min_axis(1, None).unwrap();
        assert_eq!(result.as_slice::<i32>(), &[1, 3]);
    }

    #[test]
    fn test_min_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.min_axes(&[], None).unwrap();

        let results: &[i32] = result.as_slice();
        assert_eq!(results, &[5, 8, 4, 9]);
    }

    #[test]
    fn test_var() {
        let x = Array::from_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(x.var(None, None).unwrap().item::<f32>(), 1.25);
        let y = x.var(true, None).unwrap();
        assert_eq!(y.item::<f32>(), 1.25);
        assert_eq!(y.shape(), &[1, 1]);

        let result = x.var_axis(0, None, None).unwrap();
        assert_eq!(result.as_slice::<f32>(), &[1.0, 1.0]);

        let result = x.var_axis(1, None, None).unwrap();
        assert_eq!(result.as_slice::<f32>(), &[0.25, 0.25]);

        let x = Array::from_slice(&[1.0, 2.0], &[2]);
        let out = x.var(None, Some(3)).unwrap();
        assert_eq!(out.item::<f32>(), f32::INFINITY);
    }

    #[test]
    fn test_var_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.var_axes(&[], None, 0).unwrap();

        let results: &[f32] = result.as_slice();
        assert_eq!(results, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_log_sum_exp() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.logsumexp_axis(0, None).unwrap();

        let results: &[f32] = result.as_slice();
        assert_eq!(results, &[5.3132615, 9.313262]);
    }

    #[test]
    fn test_log_sum_exp_empty_axes() {
        let array = Array::from_slice(&[5, 8, 4, 9], &[2, 2]);
        let result = array.logsumexp_axes(&[], None).unwrap();

        let results: &[f32] = result.as_slice();
        assert_eq!(results, &[5.0, 8.0, 4.0, 9.0]);
    }

    // Tests adapted from Python test `test_ops.py/test_median`
    #[test]
    fn test_median() {
        // Test basic median over all elements (odd count)
        let x = Array::from_slice(&[0, 1, 2, 3, 4], &[5]);
        let out = x.median(None).unwrap();
        assert_eq!(out.shape(), &[] as &[i32]);
        assert_eq!(out.item::<i32>(), 2);

        // Test keepdims
        let out = x.median(true).unwrap();
        assert_eq!(out.shape(), &[1]);

        // Test median with even count (should be average of two middle values)
        let x = Array::from_slice(&[0, 1, 2, 3, 4, 5], &[6]);
        let out = x.median(None).unwrap();
        assert!((out.item::<f32>() - 2.5).abs() < 1e-5);

        // Test median over specific axes
        use crate::random;
        random::seed(0).unwrap();
        let x = random::normal::<f32>(&[5, 5, 5, 5], None, None, None).unwrap();

        let out = x.median_axes(&[0, 2], true).unwrap();
        assert_eq!(out.shape(), &[1, 5, 1, 5]);

        let out = x.median_axes(&[1, 3], true).unwrap();
        assert_eq!(out.shape(), &[5, 1, 5, 1]);

        // Test single axis
        let x = Array::from_slice(&[1, 5, 2, 4, 3, 6], &[2, 3]);
        let out = x.median_axis(0, None).unwrap();
        assert_eq!(out.shape(), &[3]);

        let out = x.median_axis(1, None).unwrap();
        assert_eq!(out.shape(), &[2]);
    }
}
