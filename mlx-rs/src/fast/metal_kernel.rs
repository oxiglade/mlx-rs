use std::ffi::CString;

use crate::error::{Exception, Result};
use crate::utils::{guard::Guarded, SUCCESS};
use crate::{Array, Dtype, Stream};

/// A compile-time template parameter for a [`MetalKernel`].
///
/// Each argument is emitted as a template parameter of the generated `[[kernel]]` function, so
/// the kernel source can refer to it by name (for example `T`, `N` or `USE_BIAS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateArg {
    /// A type parameter, emitted as the matching Metal type (for example `float`).
    Dtype(Dtype),
    /// An `int` parameter.
    Int(i32),
    /// A `bool` parameter.
    Bool(bool),
}

/// Construction options for [`MetalKernel::new`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetalKernelOptions {
    /// Metal source placed before the generated kernel, such as helper functions or includes.
    pub header: String,
    /// Copy inputs to row-contiguous layout only when `true`. Defaults to `false`; Python
    /// defaults to `true`. When `false`, the source must handle the inputs' actual layout
    /// (for example through `_strides` and `elem_to_loc`).
    pub ensure_row_contiguous: bool,
    /// Declare outputs as `device atomic<T>*` so the kernel can use Metal atomics on them.
    pub atomic_outputs: bool,
}

/// The shape and dtype of one [`MetalKernel`] output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalKernelOutput {
    /// Output shape.
    pub shape: Vec<i32>,
    /// Output element type.
    pub dtype: Dtype,
}

/// Independent per-call options for [`MetalKernel::apply`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetalKernelApplyOptions {
    /// Template parameters, in declaration order.
    pub template_args: Vec<(String, TemplateArg)>,
    /// Fill every output with this value before the kernel runs. When absent, outputs start
    /// uninitialized and the kernel must write every element.
    pub init_value: Option<f32>,
    /// Print the generated Metal source on every call to `apply`.
    pub verbose: bool,
}

/// A custom Metal kernel, compiled from source on first use and cached by MLX.
///
/// The kernel body is written against the names given at construction: each input `a` is a
/// `const device T* a`, or `const constant T* a` when it has fewer than eight elements.
/// Non-scalar inputs also expose `a_shape`, `a_strides` and `a_ndim` when referenced in the
/// source. A 0-d input is a `const constant T& a` and has no shape, strides or ndim arguments.
/// Each output is a `device T*` (or an atomic pointer with `atomic_outputs`), and Metal
/// attributes such as `thread_position_in_grid` are declared automatically when referenced.
///
/// Kernels only run on the GPU. [`MetalKernel::apply`] uses the stream selected by
/// [`crate::with_stream`] or [`crate::with_device`], falling back to the GPU outside a scope,
/// even when the default device is the CPU. Selecting a CPU scope returns an error.
///
/// ```rust
/// use mlx_rs::{fast::{MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput}, Array, Dtype};
///
/// let kernel = MetalKernel::new(
///     "double",
///     &["a"],
///     &["out"],
///     "uint i = thread_position_in_grid.x; out[i] = 2 * a[i];",
///     MetalKernelOptions::default(),
/// )?;
/// let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
/// // SAFETY: Four threads read four f32 inputs and initialize distinct f32 outputs.
/// let out = unsafe {
///     kernel.apply(
///         &[&a],
///         &[MetalKernelOutput { shape: vec![4], dtype: Dtype::Float32 }],
///         (4, 1, 1),
///         (4, 1, 1),
///         MetalKernelApplyOptions::default(),
///     )?
/// };
/// assert_eq!(out[0].as_slice::<f32>(), &[2.0, 4.0, 6.0, 8.0]);
/// # Ok::<(), mlx_rs::error::Exception>(())
/// ```
pub struct MetalKernel {
    raw: mlx_sys::mlx_fast_metal_kernel,
}

impl MetalKernel {
    /// Builds a kernel from its name, input and output names, and Metal body `source`.
    ///
    /// At least one output name is required. Compilation is deferred until the outputs are
    /// evaluated, so source errors surface from [`Array::eval`], not [`apply`](Self::apply).
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        options: MetalKernelOptions,
    ) -> Result<Self> {
        crate::error::INIT_ERR_HANDLER.call_once(crate::error::setup_mlx_error_handler);
        let name = c_string(name)?;
        let source = c_string(source)?;
        let header = c_string(&options.header)?;
        let input_names = StringVector::new(input_names)?;
        let output_names = StringVector::new(output_names)?;
        let raw = unsafe {
            mlx_sys::mlx_fast_metal_kernel_new(
                name.as_ptr(),
                input_names.raw,
                output_names.raw,
                source.as_ptr(),
                header.as_ptr(),
                options.ensure_row_contiguous,
                options.atomic_outputs,
            )
        };
        if raw.ctx.is_null() {
            return Err(crate::error::exception_from_status(
                SUCCESS + 1,
                "creating a Metal kernel",
            ));
        }
        Ok(Self { raw })
    }

    /// Runs the kernel over `grid` threads in groups of `thread_group`, returning one array per
    /// entry of `outputs`.
    ///
    /// `inputs` and `outputs` must match the counts given at construction. Like other
    /// operations the result is lazy; the kernel compiles and runs when the outputs are
    /// evaluated. Source and threadgroup errors surface from [`Array::eval`]. The kernel and
    /// option values may be dropped before evaluating the outputs.
    ///
    /// # Safety
    ///
    /// For the given inputs, outputs, template arguments and launch configuration, the kernel
    /// must only access memory inside the input and output buffers and must never write an input.
    /// It must write every output element before it is read unless `init_value` is set, only
    /// write values valid for the output dtype, and have no unsynchronized conflicting accesses.
    /// These requirements apply throughout deferred execution.
    pub unsafe fn apply(
        &self,
        inputs: &[impl AsRef<Array>],
        outputs: &[MetalKernelOutput],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        options: MetalKernelApplyOptions,
    ) -> Result<Vec<Array>> {
        crate::error::INIT_ERR_HANDLER.call_once(crate::error::setup_mlx_error_handler);
        let config = KernelConfig::new()?;
        for output in outputs {
            config.check(
                unsafe {
                    mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                        config.raw,
                        output.shape.as_ptr(),
                        output.shape.len(),
                        output.dtype.into(),
                    )
                },
                "adding a Metal kernel output",
            )?;
        }
        config.check(
            unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_set_grid(config.raw, grid.0, grid.1, grid.2)
            },
            "setting the Metal kernel grid",
        )?;
        config.check(
            unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(
                    config.raw,
                    thread_group.0,
                    thread_group.1,
                    thread_group.2,
                )
            },
            "setting the Metal kernel thread group",
        )?;
        for (name, arg) in &options.template_args {
            let name = c_string(name)?;
            let status = unsafe {
                match *arg {
                    TemplateArg::Dtype(dtype) => {
                        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                            config.raw,
                            name.as_ptr(),
                            dtype.into(),
                        )
                    }
                    TemplateArg::Int(value) => {
                        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                            config.raw,
                            name.as_ptr(),
                            value,
                        )
                    }
                    TemplateArg::Bool(value) => {
                        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                            config.raw,
                            name.as_ptr(),
                            value,
                        )
                    }
                }
            };
            config.check(status, "adding a Metal kernel template argument")?;
        }
        if let Some(value) = options.init_value {
            config.check(
                unsafe { mlx_sys::mlx_fast_metal_kernel_config_set_init_value(config.raw, value) },
                "setting the Metal kernel init value",
            )?;
        }
        config.check(
            unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_set_verbose(config.raw, options.verbose)
            },
            "setting Metal kernel verbosity",
        )?;

        let inputs = crate::utils::VectorArray::try_from_iter(inputs.iter())?;
        let stream = Stream::thread_local_or_gpu();
        Vec::<Array>::try_from_op(|res| unsafe {
            mlx_sys::mlx_fast_metal_kernel_apply(
                res,
                self.raw,
                inputs.as_ptr(),
                config.raw,
                stream.as_ref().as_ptr(),
            )
        })
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_fast_metal_kernel_free(self.raw) };
    }
}

impl std::fmt::Debug for MetalKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalKernel").finish_non_exhaustive()
    }
}

struct KernelConfig {
    raw: mlx_sys::mlx_fast_metal_kernel_config,
}

impl KernelConfig {
    fn new() -> Result<Self> {
        let raw = unsafe { mlx_sys::mlx_fast_metal_kernel_config_new() };
        if raw.ctx.is_null() {
            return Err(crate::error::exception_from_status(
                SUCCESS + 1,
                "creating a Metal kernel config",
            ));
        }
        Ok(Self { raw })
    }

    fn check(&self, status: i32, operation: &str) -> Result<()> {
        if status == SUCCESS {
            Ok(())
        } else {
            Err(crate::error::exception_from_status(status, operation))
        }
    }
}

impl Drop for KernelConfig {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_fast_metal_kernel_config_free(self.raw) };
    }
}

struct StringVector {
    raw: mlx_sys::mlx_vector_string,
}

impl StringVector {
    fn new(values: &[&str]) -> Result<Self> {
        let owned = values
            .iter()
            .map(|value| c_string(value))
            .collect::<Result<Vec<_>>>()?;
        let mut pointers: Vec<_> = owned.iter().map(|value| value.as_ptr()).collect();
        // mlx-c copies each string into the vector, so `owned` only needs to outlive this call.
        let raw =
            unsafe { mlx_sys::mlx_vector_string_new_data(pointers.as_mut_ptr(), pointers.len()) };
        if raw.ctx.is_null() {
            return Err(crate::error::exception_from_status(
                SUCCESS + 1,
                "creating an MLX string vector",
            ));
        }
        Ok(Self { raw })
    }
}

impl Drop for StringVector {
    fn drop(&mut self) {
        let _ = unsafe { mlx_sys::mlx_vector_string_free(self.raw) };
    }
}

fn c_string(value: &str) -> Result<CString> {
    CString::new(value).map_err(|error| Exception::custom(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_metal_kernel_template_args_and_shapes() {
        let kernel = MetalKernel::new(
            "scale_shift",
            &["a", "b"],
            &["out1", "out2"],
            "uint elem = thread_position_in_grid.x;
             T v = a[elem] * T(SCALE);
             out1[elem] = SHIFT ? v + b[0] : v;
             out2[elem] = a_shape[0];",
            MetalKernelOptions::default(),
        )
        .unwrap();
        let a = &crate::ops::arange::<_, f32>(None, 8, None).unwrap() / 4.0;
        let b = Array::from_slice(&[0.5_f32], &[1]);
        // SAFETY: Eight threads read valid inputs and initialize distinct outputs of matching dtypes.
        let outputs = crate::with_device(crate::Device::gpu(), || unsafe {
            kernel.apply(
                &[&a, &b],
                &[
                    MetalKernelOutput {
                        shape: vec![8],
                        dtype: Dtype::Float32,
                    },
                    MetalKernelOutput {
                        shape: vec![8],
                        dtype: Dtype::Int32,
                    },
                ],
                (8, 1, 1),
                (4, 1, 1),
                MetalKernelApplyOptions {
                    template_args: vec![
                        ("T".into(), TemplateArg::Dtype(Dtype::Float32)),
                        ("SCALE".into(), TemplateArg::Int(3)),
                        ("SHIFT".into(), TemplateArg::Bool(true)),
                    ],
                    ..Default::default()
                },
            )
        })
        .unwrap();

        assert_eq!(
            outputs[0].to_vec_exact::<f32>().unwrap(),
            [0.5, 1.25, 2.0, 2.75, 3.5, 4.25, 5.0, 5.75]
        );
        assert_eq!(outputs[1].to_vec_exact::<i32>().unwrap(), [8; 8]);
    }

    #[test]
    fn test_metal_kernel_init_value_fills_unwritten_outputs() {
        let kernel = MetalKernel::new(
            "touch_half",
            &["a"],
            &["out"],
            "uint e = thread_position_in_grid.x; if (e < 2) out[e] = a[e];",
            MetalKernelOptions::default(),
        )
        .unwrap();
        let a = Array::from_slice(&[0.0_f32, 0.25, 0.5, 0.75], &[4]);
        // SAFETY: Threads 0 and 1 copy valid f32 elements; init_value initializes the rest.
        let out = crate::with_device(crate::Device::gpu(), || unsafe {
            kernel.apply(
                &[&a],
                &[MetalKernelOutput {
                    shape: vec![4],
                    dtype: Dtype::Float32,
                }],
                (4, 1, 1),
                (4, 1, 1),
                MetalKernelApplyOptions {
                    init_value: Some(-1.0),
                    ..Default::default()
                },
            )
        })
        .unwrap();
        assert_eq!(
            out[0].to_vec_exact::<f32>().unwrap(),
            [0.0, 0.25, -1.0, -1.0]
        );
    }

    #[test]
    fn test_metal_kernel_rejects_invalid_arguments() {
        let error = MetalKernel::new("none", &["a"], &[], "", MetalKernelOptions::default())
            .expect_err("a kernel needs at least one output");
        assert!(error.what().contains("at least one output"));
        assert!(MetalKernel::new("nul", &["a\0"], &["out"], "", Default::default()).is_err());

        let kernel = MetalKernel::new(
            "copy",
            &["a"],
            &["out"],
            "out[0] = a[0];",
            Default::default(),
        )
        .unwrap();
        let output = [MetalKernelOutput {
            shape: vec![1],
            dtype: Dtype::Float32,
        }];
        let a = Array::from_slice(&[1.0_f32], &[1]);
        let no_inputs: [&Array; 0] = [];
        // SAFETY: The input count mismatch is rejected before any kernel can execute.
        let error = unsafe {
            kernel.apply(
                &no_inputs,
                &output,
                (1, 1, 1),
                (1, 1, 1),
                Default::default(),
            )
        }
        .expect_err("input count must match construction");
        assert!(error.what().contains("`inputs`"));

        let error = crate::with_device(crate::Device::cpu(), || {
            // SAFETY: One thread copies one f32 between single-element buffers.
            unsafe { kernel.apply(&[&a], &output, (1, 1, 1), (1, 1, 1), Default::default()) }
        })
        .expect_err("custom Metal kernels only run on the GPU");
        assert!(error.what().contains("[metal_kernel]"));
    }

    #[test]
    fn test_metal_kernel_gpu_fallback_and_scopes() {
        let kernel = MetalKernel::new(
            "stream_copy",
            &["a"],
            &["out"],
            "out[0] = a[0];",
            Default::default(),
        )
        .unwrap();
        let a = Array::from_slice(&[3.0_f32], &[1]);
        let output = [MetalKernelOutput {
            shape: vec![1],
            dtype: Dtype::Float32,
        }];
        let apply = || {
            // SAFETY: One thread copies one f32 between single-element buffers.
            unsafe { kernel.apply(&[&a], &output, (1, 1, 1), (1, 1, 1), Default::default()) }
        };
        assert_eq!(apply().unwrap()[0].to_vec_exact::<f32>().unwrap(), [3.0]);
        let previous = crate::Device::default();
        crate::Device::set_default(&crate::Device::cpu());
        let fallback = apply();
        let gpu_scope = crate::with_device(crate::Device::gpu(), apply);
        let cpu_scope = crate::with_device(crate::Device::cpu(), apply);
        let gpu_stream = crate::with_stream(&Stream::gpu(), apply);
        let cpu_stream = crate::with_stream(&Stream::cpu(), apply);
        crate::Device::set_default(&previous);
        for outputs in [fallback, gpu_scope, gpu_stream] {
            assert_eq!(outputs.unwrap()[0].to_vec_exact::<f32>().unwrap(), [3.0]);
        }
        assert!(cpu_scope.is_err());
        assert!(cpu_stream.is_err());
    }

    #[test]
    fn test_metal_kernel_layout_copies_are_opt_in() {
        let a = Array::from_slice(&[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
        let transposed = a.transpose().unwrap();
        for (options, expected) in [
            (
                MetalKernelOptions::default(),
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            ),
            (
                MetalKernelOptions {
                    ensure_row_contiguous: true,
                    ..Default::default()
                },
                [0.0, 3.0, 1.0, 4.0, 2.0, 5.0],
            ),
        ] {
            let kernel = MetalKernel::new(
                "physical_copy",
                &["a"],
                &["out"],
                "uint i = thread_position_in_grid.x; out[i] = a[i];",
                options,
            )
            .unwrap();
            // SAFETY: Six threads copy distinct f32 elements within both six-element buffers.
            let outputs = unsafe {
                kernel.apply(
                    &[&transposed],
                    &[MetalKernelOutput {
                        shape: vec![6],
                        dtype: Dtype::Float32,
                    }],
                    (6, 1, 1),
                    (6, 1, 1),
                    Default::default(),
                )
            }
            .unwrap();
            assert_eq!(outputs[0].to_vec_exact::<f32>().unwrap(), expected);
        }
    }

    #[test]
    fn test_metal_kernel_errors_are_deferred() {
        for (name, source, size, group) in [
            ("bad_source", "out[0] = a[0] +;", 1, 1),
            (
                "bad_group",
                "uint i = thread_position_in_grid.x; out[i] = a[i];",
                4096,
                4096,
            ),
        ] {
            let kernel =
                MetalKernel::new(name, &["a"], &["out"], source, Default::default()).unwrap();
            let input = Array::from_slice(&vec![1.0_f32; size as usize], &[size]);
            // SAFETY: Invalid source cannot execute; the copy case has one f32 per grid thread.
            let outputs = unsafe {
                kernel.apply(
                    &[&input],
                    &[MetalKernelOutput {
                        shape: vec![size],
                        dtype: Dtype::Float32,
                    }],
                    (size, 1, 1),
                    (group, 1, 1),
                    Default::default(),
                )
            }
            .expect("source and threadgroup errors are deferred");
            assert!(outputs[0].eval().is_err());
        }
        let kernel = MetalKernel::new(
            "after_error",
            &["a"],
            &["out"],
            "out[0] = a;",
            Default::default(),
        )
        .unwrap();
        let input = Array::from_f32(2.0);
        // SAFETY: One thread reads a scalar and initializes the single f32 output.
        let outputs = unsafe {
            kernel.apply(
                &[&input],
                &[MetalKernelOutput {
                    shape: vec![1],
                    dtype: Dtype::Float32,
                }],
                (1, 1, 1),
                (1, 1, 1),
                Default::default(),
            )
        }
        .unwrap();
        assert_eq!(outputs[0].to_vec_exact::<f32>().unwrap(), [2.0]);
    }
}
