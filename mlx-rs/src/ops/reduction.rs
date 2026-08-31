use crate::array::Array;
use crate::error::Result;
use crate::utils::axes_or_default_to_all;
use crate::utils::guard::Guarded;
use crate::{Axes, Stream};
use mlx_internal_macros::generate_macro;

/// Axis selection and dimension retention for [`Array::count_nonzero`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CountNonzeroOptions {
    /// Axes to reduce.
    pub axes: Axes,

    /// Keep reduced axes as singleton dimensions.
    pub keep_dims: bool,
}

static EMPTY_AXES_DUMMY: [i32; 1] = [0];

impl Array {
    /// Count values that compare unequal to zero.
    ///
    /// The result is the sum of an `i32` nonzero mask. An explicitly empty axis list returns that
    /// elementwise mask instead of reducing it.
    ///
    /// ```rust
    /// use mlx_rs::{array, ops::CountNonzeroOptions, Axes, Dtype};
    ///
    /// let input = array!([[0, 2], [3, 0]]);
    /// let output = input
    ///     .count_nonzero(CountNonzeroOptions {
    ///         axes: Axes::Axis(0),
    ///         keep_dims: false,
    ///     })
    ///     .unwrap();
    /// assert_eq!(output.dtype(), Dtype::Int32);
    /// ```
    pub fn count_nonzero(&self, options: CountNonzeroOptions) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        match options.axes {
            Axes::All => Array::try_from_op(|res| unsafe {
                mlx_sys::mlx_count_nonzero(
                    res,
                    self.as_ptr(),
                    options.keep_dims,
                    stream.as_ref().as_ptr(),
                )
            }),
            Axes::Axis(axis) => Array::try_from_op(|res| unsafe {
                mlx_sys::mlx_count_nonzero_axis(
                    res,
                    self.as_ptr(),
                    axis,
                    options.keep_dims,
                    stream.as_ref().as_ptr(),
                )
            }),
            Axes::Axes(axes) => {
                let axes_ptr = if axes.is_empty() {
                    EMPTY_AXES_DUMMY.as_ptr()
                } else {
                    axes.as_ptr()
                };
                Array::try_from_op(|res| unsafe {
                    mlx_sys::mlx_count_nonzero_axes(
                        res,
                        self.as_ptr(),
                        axes_ptr,
                        axes.len(),
                        options.keep_dims,
                        stream.as_ref().as_ptr(),
                    )
                })
            }
        }
    }

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
    pub fn all_axes(&self, axes: &[i32], keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`all_axes`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `all_axes`"
    )]
    pub fn all_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.all_axes(axes, keep_dims))
    }

    /// Similar to [`Array::all_axes`] but only reduces over a single axis.
    pub fn all_axis(&self, axis: i32, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`all_axis`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `all_axis`"
    )]
    pub fn all_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.all_axis(axis, keep_dims))
    }

    /// Similar to [`Array::all_axes`] but reduces over all axes.
    pub fn all(&self, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_all(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`all`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `all`"
    )]
    pub fn all_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.all(keep_dims))
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
    pub fn prod_axes(&self, axes: &[i32], keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`prod_axes`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `prod_axes`"
    )]
    pub fn prod_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.prod_axes(axes, keep_dims))
    }

    /// Similar to [`Array::prod_axes`] but only reduces over a single axis.
    pub fn prod_axis(&self, axis: i32, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`prod_axis`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `prod_axis`"
    )]
    pub fn prod_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.prod_axis(axis, keep_dims))
    }

    /// Similar to [`Array::prod_axes`] but reduces over all axes.
    pub fn prod(&self, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_prod(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`prod`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `prod`"
    )]
    pub fn prod_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.prod(keep_dims))
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
    pub fn max_axes(&self, axes: &[i32], keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`max_axes`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `max_axes`"
    )]
    pub fn max_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.max_axes(axes, keep_dims))
    }

    /// Similar to [`Array::max_axes`] but only reduces over a single axis.
    pub fn max_axis(&self, axis: i32, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`max_axis`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `max_axis`"
    )]
    pub fn max_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.max_axis(axis, keep_dims))
    }

    /// Similar to [`Array::max_axes`] but reduces over all axes.
    pub fn max(&self, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_max(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`max`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `max`"
    )]
    pub fn max_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.max(keep_dims))
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
    pub fn sum_axes(&self, axes: &[i32], keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`sum_axes`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `sum_axes`"
    )]
    pub fn sum_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.sum_axes(axes, keep_dims))
    }

    /// Similar to [`Array::sum_axes`] but only reduces over a single axis.
    pub fn sum_axis(&self, axis: i32, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`sum_axis`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `sum_axis`"
    )]
    pub fn sum_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.sum_axis(axis, keep_dims))
    }

    /// Similar to [`Array::sum_axes`] but reduces over all axes.
    pub fn sum(&self, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_sum(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`sum`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `sum`"
    )]
    pub fn sum_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.sum(keep_dims))
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
    pub fn mean_axes(&self, axes: &[i32], keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`mean_axes`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `mean_axes`"
    )]
    pub fn mean_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.mean_axes(axes, keep_dims))
    }

    /// Similar to [`Array::mean_axes`] but only reduces over a single axis.
    pub fn mean_axis(&self, axis: i32, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`mean_axis`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `mean_axis`"
    )]
    pub fn mean_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.mean_axis(axis, keep_dims))
    }

    /// Similar to [`Array::mean_axes`] but reduces over all axes.
    pub fn mean(&self, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_mean(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`mean`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `mean`"
    )]
    pub fn mean_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.mean(keep_dims))
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
    pub fn min_axes(&self, axes: &[i32], keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`min_axes`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `min_axes`"
    )]
    pub fn min_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.min_axes(axes, keep_dims))
    }

    /// Similar to [`Array::min_axes`] but only reduces over a single axis.
    pub fn min_axis(&self, axis: i32, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`min_axis`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `min_axis`"
    )]
    pub fn min_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.min_axis(axis, keep_dims))
    }

    /// Similar to [`Array::min_axes`] but reduces over all axes.
    pub fn min(&self, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_min(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`min`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `min`"
    )]
    pub fn min_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.min(keep_dims))
    }

    /// Compute the variance(s) over the given axes returning an error if the axes are invalid.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: if `true`, keep the reduces axes as singleton dimensions
    /// - ddof: the divisor to compute the variance is `N - ddof`
    pub fn var_axes(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        ddof: impl Into<Option<i32>>,
    ) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`var_axes`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `var_axes`"
    )]
    pub fn var_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        ddof: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.var_axes(axes, keep_dims, ddof))
    }

    /// Similar to [`Array::var_axes`] but only reduces over a single axis.
    pub fn var_axis(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        ddof: impl Into<Option<i32>>,
    ) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`var_axis`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `var_axis`"
    )]
    pub fn var_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        ddof: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.var_axis(axis, keep_dims, ddof))
    }

    /// Similar to [`Array::var_axes`] but reduces over all axes.
    pub fn var(
        &self,
        keep_dims: impl Into<Option<bool>>,
        ddof: impl Into<Option<i32>>,
    ) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`var`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `var`"
    )]
    pub fn var_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        ddof: impl Into<Option<i32>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.var(keep_dims, ddof))
    }

    /// Compute the median over the given axes.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    pub fn median_axes(&self, axes: &[i32], keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`median_axes`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `median_axes`"
    )]
    pub fn median_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.median_axes(axes, keep_dims))
    }

    /// Similar to [`Array::median_axes`] but only reduces over a single axis.
    pub fn median_axis(&self, axis: i32, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`median_axis`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `median_axis`"
    )]
    pub fn median_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.median_axis(axis, keep_dims))
    }

    /// Compute the median over all axes.
    pub fn median(&self, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_median(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`median`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `median`"
    )]
    pub fn median_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.median(keep_dims))
    }

    /// A `log-sum-exp` reduction over the given axes returning an error if the axes are invalid.
    ///
    /// The log-sum-exp reduction is a numerically stable version of using the individual operations.
    ///
    /// # Params
    ///
    /// - axes: axes to reduce over
    /// - keep_dims: Whether to keep the reduced dimensions -- defaults to false if not provided
    pub fn logsumexp_axes(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
    ) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`logsumexp_axes`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `logsumexp_axes`"
    )]
    pub fn logsumexp_axes_device(
        &self,
        axes: &[i32],
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.logsumexp_axes(axes, keep_dims))
    }

    /// Similar to [`Array::logsumexp_axes`] but only reduces over a single axis.
    pub fn logsumexp_axis(&self, axis: i32, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
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

    /// Compatibility shim for [`logsumexp_axis`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `logsumexp_axis`"
    )]
    pub fn logsumexp_axis_device(
        &self,
        axis: i32,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.logsumexp_axis(axis, keep_dims))
    }

    /// Similar to [`Array::logsumexp_axes`] but reduces over all axes.
    pub fn logsumexp(&self, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
        let stream = Stream::thread_local_or_default();
        Array::try_from_op(|res| unsafe {
            mlx_sys::mlx_logsumexp(
                res,
                self.as_ptr(),
                keep_dims.into().unwrap_or(false),
                stream.as_ref().as_ptr(),
            )
        })
    }

    /// Compatibility shim for [`logsumexp`].
    #[deprecated(
        since = "0.26.0",
        note = "use `with_stream` or `with_device` around `logsumexp`"
    )]
    pub fn logsumexp_device(
        &self,
        keep_dims: impl Into<Option<bool>>,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        crate::with_stream(stream.as_ref(), || self.logsumexp(keep_dims))
    }
}

/// See [`Array::all_axes`]
pub fn all_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().all_axes(axes, keep_dims)
}

/// Compatibility shim for [`all_axes`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `all_axes`"
)]
pub fn all_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || all_axes(array, axes, keep_dims))
}

/// See [`Array::all_axis`]
pub fn all_axis(
    array: impl AsRef<Array>,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().all_axis(axis, keep_dims)
}

/// Compatibility shim for [`all_axis`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `all_axis`"
)]
pub fn all_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || all_axis(array, axis, keep_dims))
}

/// See [`Array::all`]
pub fn all(array: impl AsRef<Array>, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
    array.as_ref().all(keep_dims)
}

/// Compatibility shim for [`all`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `all`"
)]
pub fn all_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || all(array, keep_dims))
}

/// See [`Array::prod_axes`]
pub fn prod_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().prod_axes(axes, keep_dims)
}

/// Compatibility shim for [`prod_axes`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `prod_axes`"
)]
pub fn prod_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || prod_axes(array, axes, keep_dims))
}

/// See [`Array::prod_axis`]
pub fn prod_axis(
    array: impl AsRef<Array>,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().prod_axis(axis, keep_dims)
}

/// Compatibility shim for [`prod_axis`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `prod_axis`"
)]
pub fn prod_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || prod_axis(array, axis, keep_dims))
}

/// See [`Array::prod`]
pub fn prod(array: impl AsRef<Array>, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
    array.as_ref().prod(keep_dims)
}

/// Compatibility shim for [`prod`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `prod`"
)]
pub fn prod_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || prod(array, keep_dims))
}

/// See [`Array::max_axes`]
pub fn max_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().max_axes(axes, keep_dims)
}

/// Compatibility shim for [`max_axes`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `max_axes`"
)]
pub fn max_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || max_axes(array, axes, keep_dims))
}

/// See [`Array::max_axis`]
pub fn max_axis(
    array: impl AsRef<Array>,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().max_axis(axis, keep_dims)
}

/// Compatibility shim for [`max_axis`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `max_axis`"
)]
pub fn max_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || max_axis(array, axis, keep_dims))
}

/// See [`Array::max`]
pub fn max(array: impl AsRef<Array>, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
    array.as_ref().max(keep_dims)
}

/// Compatibility shim for [`max`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `max`"
)]
pub fn max_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || max(array, keep_dims))
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
pub fn std_axes(
    a: impl AsRef<Array>,
    axes: &[i32],
    keep_dims: impl Into<Option<bool>>,
    ddof: impl Into<Option<i32>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
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

/// Compatibility shim for [`std_axes`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `std_axes`"
)]
pub fn std_axes_device(
    a: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || std_axes(a, axes, keep_dims, ddof))
}

/// Similar to [`std_axes`] but only reduces over a single axis.
pub fn std_axis(
    a: impl AsRef<Array>,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
    ddof: impl Into<Option<i32>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
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

/// Compatibility shim for [`std_axis`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `std_axis`"
)]
pub fn std_axis_device(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || std_axis(a, axis, keep_dims, ddof))
}

/// Similar to [`std_axes`] but reduces over all axes.
pub fn std(
    a: impl AsRef<Array>,
    keep_dims: impl Into<Option<bool>>,
    ddof: impl Into<Option<i32>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let a = a.as_ref();
    let keep_dims = keep_dims.into().unwrap_or(false);
    let ddof = ddof.into().unwrap_or(0);
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_std(res, a.as_ptr(), keep_dims, ddof, stream.as_ref().as_ptr())
    })
}

/// Compatibility shim for [`std`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `std`"
)]
pub fn std_device(
    a: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || std(a, keep_dims, ddof))
}

/// See [`Array::sum_axes`]
pub fn sum_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().sum_axes(axes, keep_dims)
}

/// Compatibility shim for [`sum_axes`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `sum_axes`"
)]
pub fn sum_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || sum_axes(array, axes, keep_dims))
}

/// See [`Array::sum_axis`]
pub fn sum_axis(
    array: impl AsRef<Array>,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().sum_axis(axis, keep_dims)
}

/// Compatibility shim for [`sum_axis`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `sum_axis`"
)]
pub fn sum_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || sum_axis(array, axis, keep_dims))
}

/// See [`Array::sum`]
pub fn sum(array: impl AsRef<Array>, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
    array.as_ref().sum(keep_dims)
}

/// Compatibility shim for [`sum`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `sum`"
)]
pub fn sum_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || sum(array, keep_dims))
}

/// See [`Array::mean_axes`]
pub fn mean_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().mean_axes(axes, keep_dims)
}

/// Compatibility shim for [`mean_axes`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `mean_axes`"
)]
pub fn mean_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || mean_axes(array, axes, keep_dims))
}

/// See [`Array::mean_axis`]
pub fn mean_axis(
    array: impl AsRef<Array>,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().mean_axis(axis, keep_dims)
}

/// Compatibility shim for [`mean_axis`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `mean_axis`"
)]
pub fn mean_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || mean_axis(array, axis, keep_dims))
}

/// See [`Array::mean`]
pub fn mean(array: impl AsRef<Array>, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
    array.as_ref().mean(keep_dims)
}

/// Compatibility shim for [`mean`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `mean`"
)]
pub fn mean_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || mean(array, keep_dims))
}

/// See [`Array::min`]
pub fn min_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().min_axes(axes, keep_dims)
}

/// Compatibility shim for [`min_axes`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `min_axes`"
)]
pub fn min_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || min_axes(array, axes, keep_dims))
}

/// See [`Array::min_axis`]
pub fn min_axis(
    array: impl AsRef<Array>,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().min_axis(axis, keep_dims)
}

/// Compatibility shim for [`min_axis`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `min_axis`"
)]
pub fn min_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || min_axis(array, axis, keep_dims))
}

/// See [`Array::min`]
pub fn min(array: impl AsRef<Array>, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
    array.as_ref().min(keep_dims)
}

/// Compatibility shim for [`min`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `min`"
)]
pub fn min_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || min(array, keep_dims))
}

/// See [`Array::var_axes`]
pub fn var_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    keep_dims: impl Into<Option<bool>>,
    ddof: impl Into<Option<i32>>,
) -> Result<Array> {
    array.as_ref().var_axes(axes, keep_dims, ddof)
}

/// Compatibility shim for [`var_axes`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `var_axes`"
)]
pub fn var_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || var_axes(array, axes, keep_dims, ddof))
}

/// See [`Array::var_axis`]
pub fn var_axis(
    array: impl AsRef<Array>,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
    ddof: impl Into<Option<i32>>,
) -> Result<Array> {
    array.as_ref().var_axis(axis, keep_dims, ddof)
}

/// Compatibility shim for [`var_axis`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `var_axis`"
)]
pub fn var_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || var_axis(array, axis, keep_dims, ddof))
}

/// See [`Array::var`]
pub fn var(
    array: impl AsRef<Array>,
    keep_dims: impl Into<Option<bool>>,
    ddof: impl Into<Option<i32>>,
) -> Result<Array> {
    array.as_ref().var(keep_dims, ddof)
}

/// Compatibility shim for [`var`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `var`"
)]
pub fn var_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || var(array, keep_dims, ddof))
}

/// See [`Array::median_axes`]
pub fn median_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().median_axes(axes, keep_dims)
}

/// Compatibility shim for [`median_axes`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `median_axes`"
)]
pub fn median_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || median_axes(array, axes, keep_dims))
}

/// See [`Array::median_axis`]
pub fn median_axis(
    array: impl AsRef<Array>,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().median_axis(axis, keep_dims)
}

/// Compatibility shim for [`median_axis`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `median_axis`"
)]
pub fn median_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || median_axis(array, axis, keep_dims))
}

/// See [`Array::median`]
pub fn median(array: impl AsRef<Array>, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
    array.as_ref().median(keep_dims)
}

/// Compatibility shim for [`median`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `median`"
)]
pub fn median_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || median(array, keep_dims))
}

/// See [`Array::logsumexp_axes`]
pub fn logsumexp_axes(
    array: impl AsRef<Array>,
    axes: &[i32],
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().logsumexp_axes(axes, keep_dims)
}

/// Compatibility shim for [`logsumexp_axes`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `logsumexp_axes`"
)]
pub fn logsumexp_axes_device(
    array: impl AsRef<Array>,
    axes: &[i32],
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || logsumexp_axes(array, axes, keep_dims))
}

/// See [`Array::logsumexp_axis`]
pub fn logsumexp_axis(
    array: impl AsRef<Array>,
    axis: i32,
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    array.as_ref().logsumexp_axis(axis, keep_dims)
}

/// Compatibility shim for [`logsumexp_axis`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `logsumexp_axis`"
)]
pub fn logsumexp_axis_device(
    array: impl AsRef<Array>,
    axis: i32,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || logsumexp_axis(array, axis, keep_dims))
}

/// See [`Array::logsumexp`]
pub fn logsumexp(array: impl AsRef<Array>, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
    array.as_ref().logsumexp(keep_dims)
}

/// Compatibility shim for [`logsumexp`].
#[generate_macro(customize(forwarding_shim = true))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `logsumexp`"
)]
pub fn logsumexp_device(
    array: impl AsRef<Array>,
    #[optional] keep_dims: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || logsumexp(array, keep_dims))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_all() {
        let array = Array::from_slice(&[true, false, true, false], &[2, 2]);

        assert_eq!(array.all(None).unwrap().item_exact::<bool>(), false);
        assert_eq!(array.all(true).unwrap().shape(), &[1, 1]);
        assert_eq!(
            array.all_axes(&[0, 1], None).unwrap().item_exact::<bool>(),
            false
        );

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
        assert_eq!(x.prod(None).unwrap().item_exact::<i32>(), 18);

        let y = x.prod(true).unwrap();
        assert_eq!(y.item_exact::<i32>(), 18);
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
        assert_eq!(x.max(None).unwrap().item_exact::<i32>(), 4);
        let y = x.max(true).unwrap();
        assert_eq!(y.item_exact::<i32>(), 4);
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
        assert_eq!(x.mean(None).unwrap().item_exact::<f32>(), 2.5);
        let y = x.mean(true).unwrap();
        assert_eq!(y.item_exact::<f32>(), 2.5);
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
        assert_eq!(x.min(None).unwrap().item_exact::<i32>(), 1);
        let y = x.min(true).unwrap();
        assert_eq!(y.item_exact::<i32>(), 1);
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
        assert_eq!(x.var(None, None).unwrap().item_exact::<f32>(), 1.25);
        let y = x.var(true, None).unwrap();
        assert_eq!(y.item_exact::<f32>(), 1.25);
        assert_eq!(y.shape(), &[1, 1]);

        let result = x.var_axis(0, None, None).unwrap();
        assert_eq!(result.as_slice::<f32>(), &[1.0, 1.0]);

        let result = x.var_axis(1, None, None).unwrap();
        assert_eq!(result.as_slice::<f32>(), &[0.25, 0.25]);

        let x = Array::from_slice(&[1.0, 2.0], &[2]);
        let out = x.var(None, Some(3)).unwrap();
        assert_eq!(out.item_exact::<f32>(), f32::INFINITY);
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
        assert_eq!(out.item_exact::<f32>(), 2.0);

        // Test keepdims
        let out = x.median(true).unwrap();
        assert_eq!(out.shape(), &[1]);

        // Test median with even count (should be average of two middle values)
        let x = Array::from_slice(&[0, 1, 2, 3, 4, 5], &[6]);
        let out = x.median(None).unwrap();
        assert!((out.item_exact::<f32>() - 2.5).abs() < 1e-5);

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
