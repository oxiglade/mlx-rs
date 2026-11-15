use crate::error::{Exception, Result};
use crate::utils::{guard::Guarded, SUCCESS};
use crate::{Array, Dtype, Stream};

use super::custom_kernel::{c_string, StringVector, TemplateArg};

/// Construction options for [`CudaKernel::new`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CudaKernelOptions {
    /// CUDA source placed before the generated kernel, such as helper functions or includes.
    pub header: String,
    /// Copy inputs to row-contiguous layout only when `true`. Defaults to `false`; Python
    /// defaults to `true`. When `false`, the source must handle the inputs' actual layout
    /// (for example through `_strides` and `elem_to_loc`).
    pub ensure_row_contiguous: bool,
    /// Bytes of dynamic shared memory per block, available to the source as `extern __shared__`.
    pub shared_memory: u32,
}

/// The shape and dtype of one [`CudaKernel`] output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaKernelOutput {
    /// Output shape.
    pub shape: Vec<i32>,
    /// Output element type.
    pub dtype: Dtype,
}

/// Independent per-call options for [`CudaKernel::apply`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CudaKernelApplyOptions {
    /// Compile-time parameters, in declaration order.
    pub template_args: Vec<(String, TemplateArg)>,
    /// Fill every output with this value before the kernel runs. When absent, outputs start
    /// uninitialized and the kernel must write every element.
    pub init_value: Option<f32>,
    /// Print the generated CUDA source on every call to `apply`.
    pub verbose: bool,
}

/// A custom CUDA kernel, compiled from source with NVRTC on first use and cached by MLX.
///
/// `source` is the body of a generated `__global__` function in `namespace mlx::core::cu`,
/// after `header`, with MLX's device utilities included and `cooperative_groups` available
/// as `cg`. Each input `a` is a `const T* a`, including 0-d inputs. Non-scalar inputs also
/// expose `a_shape`, `a_strides` and `a_ndim` when referenced in the source. Each output is a
/// `T*`. Template arguments become `using` aliases and `constexpr` constants in the body.
///
/// `grid` counts threads, as with `MetalKernel`: MLX launches `ceil(grid / thread_group)` blocks of `thread_group` threads in each dimension.
///
/// Kernels only run on the GPU. [`CudaKernel::apply`] uses the stream selected by
/// [`crate::with_stream`] or [`crate::with_device`], falling back to the GPU outside a scope,
/// even when the default device is the CPU. Selecting a CPU scope returns an error.
///
/// ```rust
/// use mlx_rs::{fast::{CudaKernel, CudaKernelApplyOptions, CudaKernelOptions, CudaKernelOutput}, Array, Dtype};
///
/// let kernel = CudaKernel::new(
///     "double",
///     &["a"],
///     &["out"],
///     "auto i = blockIdx.x * blockDim.x + threadIdx.x; out[i] = 2 * a[i];",
///     CudaKernelOptions::default(),
/// )?;
/// let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
/// // SAFETY: Four threads read four f32 inputs and initialize distinct f32 outputs.
/// let out = unsafe {
///     kernel.apply(
///         &[&a],
///         &[CudaKernelOutput { shape: vec![4], dtype: Dtype::Float32 }],
///         (4, 1, 1),
///         (4, 1, 1),
///         CudaKernelApplyOptions::default(),
///     )?
/// };
/// assert_eq!(out[0].as_slice::<f32>(), &[2.0, 4.0, 6.0, 8.0]);
/// # Ok::<(), mlx_rs::error::Exception>(())
/// ```
pub struct CudaKernel {
    raw: mlx_sys::mlx_fast_cuda_kernel,
}

impl CudaKernel {
    /// Builds a kernel from its name, input and output names, and CUDA body `source`.
    ///
    /// At least one output name is required. Compilation is deferred until the outputs are
    /// evaluated, so source errors surface from [`Array::eval`], not [`apply`](Self::apply).
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        options: CudaKernelOptions,
    ) -> Result<Self> {
        crate::error::ensure_mlx_error_handler();
        let name = c_string(name)?;
        let source = c_string(source)?;
        let header = c_string(&options.header)?;
        let shared_memory = i32::try_from(options.shared_memory).map_err(|_| {
            Exception::custom(format!(
                "shared_memory {} exceeds i32::MAX bytes",
                options.shared_memory
            ))
        })?;
        let input_names = StringVector::new(input_names)?;
        let output_names = StringVector::new(output_names)?;
        let raw = unsafe {
            mlx_sys::mlx_fast_cuda_kernel_new(
                name.as_ptr(),
                input_names.raw,
                output_names.raw,
                source.as_ptr(),
                header.as_ptr(),
                options.ensure_row_contiguous,
                shared_memory,
            )
        };
        if raw.ctx.is_null() {
            return Err(crate::error::exception_from_status(
                SUCCESS + 1,
                "creating a CUDA kernel",
            ));
        }
        Ok(Self { raw })
    }

    /// Runs the kernel over `grid` threads in blocks of `thread_group`, returning one array per
    /// entry of `outputs`.
    ///
    /// `inputs` and `outputs` must match the counts given at construction. Like other
    /// operations the result is lazy; the kernel compiles and runs when the outputs are
    /// evaluated. Source and launch errors surface from [`Array::eval`]. The kernel and
    /// option values may be dropped before evaluating the outputs.
    ///
    /// # Safety
    ///
    /// For the given inputs, outputs, template arguments and launch configuration, the kernel
    /// must only access memory inside the input and output buffers and its shared memory, and
    /// must never write an input. It must write every output element before it is read unless
    /// `init_value` is set, only write values valid for the output dtype, and have no
    /// unsynchronized conflicting accesses. These requirements apply throughout deferred
    /// execution.
    pub unsafe fn apply(
        &self,
        inputs: &[impl AsRef<Array>],
        outputs: &[CudaKernelOutput],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        options: CudaKernelApplyOptions,
    ) -> Result<Vec<Array>> {
        crate::error::ensure_mlx_error_handler();
        let config = KernelConfig::new()?;
        for output in outputs {
            config.check(
                unsafe {
                    mlx_sys::mlx_fast_cuda_kernel_config_add_output_arg(
                        config.raw,
                        output.shape.as_ptr(),
                        output.shape.len(),
                        output.dtype.into(),
                    )
                },
                "adding a CUDA kernel output",
            )?;
        }
        config.check(
            unsafe {
                mlx_sys::mlx_fast_cuda_kernel_config_set_grid(config.raw, grid.0, grid.1, grid.2)
            },
            "setting the CUDA kernel grid",
        )?;
        config.check(
            unsafe {
                mlx_sys::mlx_fast_cuda_kernel_config_set_thread_group(
                    config.raw,
                    thread_group.0,
                    thread_group.1,
                    thread_group.2,
                )
            },
            "setting the CUDA kernel thread group",
        )?;
        for (name, arg) in &options.template_args {
            let name = c_string(name)?;
            let status = unsafe {
                match *arg {
                    TemplateArg::Dtype(dtype) => {
                        mlx_sys::mlx_fast_cuda_kernel_config_add_template_arg_dtype(
                            config.raw,
                            name.as_ptr(),
                            dtype.into(),
                        )
                    }
                    TemplateArg::Int(value) => {
                        mlx_sys::mlx_fast_cuda_kernel_config_add_template_arg_int(
                            config.raw,
                            name.as_ptr(),
                            value,
                        )
                    }
                    TemplateArg::Bool(value) => {
                        mlx_sys::mlx_fast_cuda_kernel_config_add_template_arg_bool(
                            config.raw,
                            name.as_ptr(),
                            value,
                        )
                    }
                }
            };
            config.check(status, "adding a CUDA kernel template argument")?;
        }
        if let Some(value) = options.init_value {
            config.check(
                unsafe { mlx_sys::mlx_fast_cuda_kernel_config_set_init_value(config.raw, value) },
                "setting the CUDA kernel init value",
            )?;
        }
        config.check(
            unsafe {
                mlx_sys::mlx_fast_cuda_kernel_config_set_verbose(config.raw, options.verbose)
            },
            "setting CUDA kernel verbosity",
        )?;

        let inputs = crate::utils::VectorArray::try_from_iter(inputs.iter())?;
        let stream = Stream::thread_local_or_gpu();
        Vec::<Array>::try_from_op(|res| unsafe {
            mlx_sys::mlx_fast_cuda_kernel_apply(
                res,
                self.raw,
                inputs.as_ptr(),
                config.raw,
                stream.as_ref().as_ptr(),
            )
        })
    }
}

impl Drop for CudaKernel {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_fast_cuda_kernel_free(self.raw) };
    }
}

impl std::fmt::Debug for CudaKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaKernel").finish_non_exhaustive()
    }
}

struct KernelConfig {
    raw: mlx_sys::mlx_fast_cuda_kernel_config,
}

impl KernelConfig {
    fn new() -> Result<Self> {
        let raw = unsafe { mlx_sys::mlx_fast_cuda_kernel_config_new() };
        if raw.ctx.is_null() {
            return Err(crate::error::exception_from_status(
                SUCCESS + 1,
                "creating a CUDA kernel config",
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
        unsafe { mlx_sys::mlx_fast_cuda_kernel_config_free(self.raw) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_cuda_kernel_template_args_and_shapes() {
        let kernel = CudaKernel::new(
            "scale_shift",
            &["a", "b"],
            &["out1", "out2"],
            "auto elem = blockIdx.x * blockDim.x + threadIdx.x;
             T v = a[elem] * T(SCALE);
             out1[elem] = SHIFT ? v + b[0] : v;
             out2[elem] = a_shape[0];",
            CudaKernelOptions::default(),
        )
        .unwrap();
        let a = &crate::ops::arange::<_, f32>(None, 8, None).unwrap() / 4.0;
        let b = Array::from_slice(&[0.5_f32], &[1]);
        // SAFETY: Eight threads read valid inputs and initialize distinct outputs of matching dtypes.
        let outputs = crate::with_device(crate::Device::gpu(), || unsafe {
            kernel.apply(
                &[&a, &b],
                &[
                    CudaKernelOutput {
                        shape: vec![8],
                        dtype: Dtype::Float32,
                    },
                    CudaKernelOutput {
                        shape: vec![8],
                        dtype: Dtype::Int32,
                    },
                ],
                (8, 1, 1),
                (4, 1, 1),
                CudaKernelApplyOptions {
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
    fn test_cuda_kernel_init_value_fills_unwritten_outputs() {
        let kernel = CudaKernel::new(
            "touch_half",
            &["a"],
            &["out"],
            "auto e = blockIdx.x * blockDim.x + threadIdx.x; if (e < 2) out[e] = a[e];",
            CudaKernelOptions::default(),
        )
        .unwrap();
        let a = Array::from_slice(&[0.0_f32, 0.25, 0.5, 0.75], &[4]);
        // SAFETY: Threads 0 and 1 copy valid f32 elements; init_value initializes the rest.
        let out = crate::with_device(crate::Device::gpu(), || unsafe {
            kernel.apply(
                &[&a],
                &[CudaKernelOutput {
                    shape: vec![4],
                    dtype: Dtype::Float32,
                }],
                (4, 1, 1),
                (4, 1, 1),
                CudaKernelApplyOptions {
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
    fn test_cuda_kernel_shared_memory() {
        let kernel = CudaKernel::new(
            "reverse",
            &["a"],
            &["out"],
            "extern __shared__ float tile[];
             auto i = threadIdx.x;
             tile[i] = a[i];
             __syncthreads();
             out[i] = tile[blockDim.x - 1 - i];",
            CudaKernelOptions {
                shared_memory: 4 * 4,
                ..Default::default()
            },
        )
        .unwrap();
        let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
        // SAFETY: One block of four threads stays within four f32 elements of shared memory,
        // synchronizes before reading it, and initializes distinct outputs.
        let out = unsafe {
            kernel.apply(
                &[&a],
                &[CudaKernelOutput {
                    shape: vec![4],
                    dtype: Dtype::Float32,
                }],
                (4, 1, 1),
                (4, 1, 1),
                Default::default(),
            )
        }
        .unwrap();
        assert_eq!(out[0].to_vec_exact::<f32>().unwrap(), [4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_cuda_kernel_rejects_invalid_arguments() {
        let error = CudaKernel::new("none", &["a"], &[], "", CudaKernelOptions::default())
            .expect_err("a kernel needs at least one output");
        assert!(error.what().contains("at least one output"));
        assert!(CudaKernel::new("nul", &["a\0"], &["out"], "", Default::default()).is_err());
        let error = CudaKernel::new(
            "huge",
            &["a"],
            &["out"],
            "",
            CudaKernelOptions {
                shared_memory: u32::MAX,
                ..Default::default()
            },
        )
        .expect_err("shared memory must fit in an i32");
        assert!(error.what().contains("shared_memory"));

        let kernel = CudaKernel::new(
            "copy",
            &["a"],
            &["out"],
            "out[0] = a[0];",
            Default::default(),
        )
        .unwrap();
        let output = [CudaKernelOutput {
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
        .expect_err("custom CUDA kernels only run on the GPU");
        assert!(error.what().contains("[custom_kernel]"));
    }

    #[test]
    fn test_cuda_kernel_gpu_fallback_and_scopes() {
        let kernel = CudaKernel::new(
            "stream_copy",
            &["a"],
            &["out"],
            "out[0] = a[0];",
            Default::default(),
        )
        .unwrap();
        let a = Array::from_slice(&[3.0_f32], &[1]);
        let output = [CudaKernelOutput {
            shape: vec![1],
            dtype: Dtype::Float32,
        }];
        let apply = || {
            // SAFETY: One thread copies one f32 between single-element buffers.
            unsafe { kernel.apply(&[&a], &output, (1, 1, 1), (1, 1, 1), Default::default()) }
        };
        let fallback = apply();
        let gpu_scope = crate::with_device(crate::Device::gpu(), apply);
        let cpu_scope = crate::with_device(crate::Device::cpu(), apply);
        let gpu_stream = crate::with_stream(&Stream::gpu(), apply);
        let cpu_stream = crate::with_stream(&Stream::cpu(), apply);
        for outputs in [fallback, gpu_scope, gpu_stream] {
            assert_eq!(outputs.unwrap()[0].to_vec_exact::<f32>().unwrap(), [3.0]);
        }
        assert!(cpu_scope.is_err());
        assert!(cpu_stream.is_err());
    }

    #[test]
    fn test_cuda_kernel_layout_copies_are_opt_in() {
        let a = Array::from_slice(&[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
        let transposed = a.transpose().unwrap();
        for (options, expected) in [
            (CudaKernelOptions::default(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            (
                CudaKernelOptions {
                    ensure_row_contiguous: true,
                    ..Default::default()
                },
                [0.0, 3.0, 1.0, 4.0, 2.0, 5.0],
            ),
        ] {
            let kernel = CudaKernel::new(
                "physical_copy",
                &["a"],
                &["out"],
                "auto i = blockIdx.x * blockDim.x + threadIdx.x; out[i] = a[i];",
                options,
            )
            .unwrap();
            // SAFETY: Six threads copy distinct f32 elements within both six-element buffers.
            let outputs = unsafe {
                kernel.apply(
                    &[&transposed],
                    &[CudaKernelOutput {
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
    fn test_cuda_kernel_errors_are_deferred() {
        let kernel = CudaKernel::new(
            "bad_source",
            &["a"],
            &["out"],
            "out[0] = a[0] +;",
            Default::default(),
        )
        .unwrap();
        let input = Array::from_slice(&[1.0_f32], &[1]);
        // SAFETY: Invalid source cannot execute.
        let outputs = unsafe {
            kernel.apply(
                &[&input],
                &[CudaKernelOutput {
                    shape: vec![1],
                    dtype: Dtype::Float32,
                }],
                (1, 1, 1),
                (1, 1, 1),
                Default::default(),
            )
        }
        .expect("source errors are deferred");
        assert!(outputs[0].eval().is_err());

        let kernel = CudaKernel::new(
            "after_error",
            &["a"],
            &["out"],
            "out[0] = a[0];",
            Default::default(),
        )
        .unwrap();
        let input = Array::from_f32(2.0);
        // SAFETY: One thread reads the scalar through its pointer and initializes the single
        // f32 output.
        let outputs = unsafe {
            kernel.apply(
                &[&input],
                &[CudaKernelOutput {
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
