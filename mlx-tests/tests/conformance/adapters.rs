use super::oracle::{dtype_from_name, mlx_error, Arg, Args, Case, ScalarValue};
use mlx_rs::{fft, ops, Array, StreamOrDevice};
use safetensors::SafeTensors;

pub(super) const ADAPTERS: &[&str] = &[
    "ops.add.array_array",
    "array.add.array",
    "operator.add.array",
    "ops.subtract.array_array",
    "array.multiply.array",
    "operator.divide.array",
    "operator.neg.array",
    "array.negative",
    "array.abs",
    "ops.exp.array",
    "ops.where.ternary",
    "ops.divmod.array_array",
    "operator.add.bool_rhs",
    "operator.add.i32_rhs",
    "operator.add.f32_rhs",
    "operator.add.complex64_rhs",
    "ops.add.explicit_cpu",
    "ops.maximum.array_array",
    "ops.power.array_array",
    "ops.reshape",
    "array.reshape",
    "array.transpose",
    "ops.transpose.axes",
    "ops.split.parts",
    "ops.split.sections",
    "ops.broadcast_arrays",
    "ops.expand_dims.axis",
    "ops.concatenate",
    "ops.sum.axis_optional",
    "ops.sum.axes",
    "array.sum.all_optional",
    "ops.take.array_indices",
    "ops.take_axis.array_indices_axis",
    "array.as_dtype",
    "ops.exp.explicit_cpu",
    "array.reshape.explicit_cpu",
    "ops.divmod.explicit_cpu",
    "ops.windows.bartlett",
    "ops.windows.blackman",
    "ops.windows.hamming",
    "ops.windows.hann",
    "fft.fftfreq",
    "fft.rfftfreq",
];

pub(super) fn dispatch(case: &Case, safe: &SafeTensors<'_>) -> Result<Vec<Array>, String> {
    let mut args = Args::new(case, safe);
    let outputs = match case.rust_call.as_str() {
        "ops.add.array_array" => {
            let a = args.tensor("input0")?;
            let b = args.tensor("input1")?;
            args.execution()?;
            vec![mlx_error(ops::add(&a, &b))?]
        }
        "array.add.array" => {
            let a = args.tensor("input0")?;
            let b = args.tensor("input1")?;
            args.execution()?;
            vec![mlx_error(a.add(&b))?]
        }
        "operator.add.array" => {
            let a = args.tensor("input0")?;
            let b = args.tensor("input1")?;
            args.execution()?;
            vec![&a + &b]
        }
        "ops.subtract.array_array" => {
            let a = args.tensor("input0")?;
            let b = args.tensor("input1")?;
            args.execution()?;
            vec![mlx_error(ops::subtract(&a, &b))?]
        }
        "array.multiply.array" => {
            let a = args.tensor("input0")?;
            let b = args.tensor("input1")?;
            args.execution()?;
            vec![mlx_error(a.multiply(&b))?]
        }
        "operator.divide.array" => {
            let a = args.tensor("input0")?;
            let b = args.tensor("input1")?;
            args.execution()?;
            vec![&a / &b]
        }
        "operator.neg.array" => {
            let a = args.tensor("input0")?;
            args.execution()?;
            vec![-&a]
        }
        "array.negative" => {
            let a = args.tensor("input0")?;
            args.execution()?;
            vec![mlx_error(a.negative())?]
        }
        "array.abs" => {
            let a = args.tensor("input0")?;
            args.execution()?;
            vec![mlx_error(a.abs())?]
        }
        "ops.exp.array" => {
            let a = args.tensor("input0")?;
            args.execution()?;
            vec![mlx_error(ops::exp(&a))?]
        }
        "ops.where.ternary" => {
            let c = args.tensor("input0")?;
            let a = args.tensor("input1")?;
            let b = args.tensor("input2")?;
            args.execution()?;
            vec![mlx_error(ops::r#where(&c, &a, &b))?]
        }
        "ops.divmod.array_array" => {
            let a = args.tensor("input0")?;
            let b = args.tensor("input1")?;
            args.execution()?;
            let (q, r) = mlx_error(ops::divmod(&a, &b))?;
            vec![q, r]
        }
        "operator.add.bool_rhs" => {
            let a = args.tensor("input0")?;
            let ScalarValue::Bool(rhs) = args.scalar("rhs")? else {
                return Err("rhs scalar type mismatch".into());
            };
            args.execution()?;
            vec![&a + rhs]
        }
        "operator.add.i32_rhs" => {
            let a = args.tensor("input0")?;
            let ScalarValue::I32(rhs) = args.scalar("rhs")? else {
                return Err("rhs scalar type mismatch".into());
            };
            args.execution()?;
            vec![&a + rhs]
        }
        "operator.add.f32_rhs" => {
            let a = args.tensor("input0")?;
            let ScalarValue::F32(rhs) = args.scalar("rhs")? else {
                return Err("rhs scalar type mismatch".into());
            };
            args.execution()?;
            vec![&a + rhs]
        }
        "operator.add.complex64_rhs" => {
            let a = args.tensor("input0")?;
            let ScalarValue::C64(rhs) = args.scalar("rhs")? else {
                return Err("rhs scalar type mismatch".into());
            };
            args.execution()?;
            vec![&a + rhs]
        }
        "ops.add.explicit_cpu" => {
            let a = args.tensor("input0")?;
            let b = args.tensor("input1")?;
            args.execution()?;
            vec![mlx_error(ops::add_device(&a, &b, StreamOrDevice::cpu()))?]
        }
        "ops.maximum.array_array" => {
            let a = args.tensor("input0")?;
            let b = args.tensor("input1")?;
            args.execution()?;
            vec![mlx_error(ops::maximum(&a, &b))?]
        }
        "ops.power.array_array" => {
            let a = args.tensor("input0")?;
            let b = args.tensor("input1")?;
            args.execution()?;
            vec![mlx_error(ops::power(&a, &b))?]
        }
        "ops.reshape" => {
            let a = args.tensor("input0")?;
            let shape = args.shape("shape")?;
            args.execution()?;
            vec![mlx_error(ops::reshape(&a, &shape))?]
        }
        "array.reshape" => {
            let a = args.tensor("input0")?;
            let shape = args.shape("shape")?;
            args.execution()?;
            vec![mlx_error(a.reshape(&shape))?]
        }
        "array.transpose" => {
            let a = args.tensor("input0")?;
            args.execution()?;
            vec![mlx_error(a.transpose())?]
        }
        "ops.transpose.axes" => {
            let a = args.tensor("input0")?;
            let axes = args.axes("axes")?;
            args.execution()?;
            vec![mlx_error(ops::transpose_axes(&a, &axes))?]
        }
        "ops.split.parts" => {
            let a = args.tensor("input0")?;
            let ScalarValue::I32(parts) = args.scalar("parts")? else {
                return Err("parts scalar type mismatch".into());
            };
            let axis = args.optional_axis("axis")?;
            args.execution()?;
            mlx_error(ops::split(&a, parts, axis))?
        }
        "ops.split.sections" => {
            let a = args.tensor("input0")?;
            let indices = args.axes("indices")?;
            let axis = args.optional_axis("axis")?;
            args.execution()?;
            mlx_error(ops::split_sections(&a, &indices, axis))?
        }
        "ops.broadcast_arrays" => {
            let mut arrays = Vec::new();
            let mut index = 0;
            while case
                .args
                .iter()
                .any(|arg| arg.name() == format!("input{index}"))
            {
                arrays.push(args.tensor(&format!("input{index}"))?);
                index += 1;
            }
            args.execution()?;
            mlx_error(ops::broadcast_arrays(&arrays))?
        }
        "ops.expand_dims.axis" => {
            let a = args.tensor("input0")?;
            let axis = args.axis("axis")?;
            args.execution()?;
            vec![mlx_error(ops::expand_dims(&a, axis))?]
        }
        "ops.concatenate" => {
            let mut arrays = Vec::new();
            let mut index = 0;
            while case
                .args
                .iter()
                .any(|arg| arg.name() == format!("input{index}"))
            {
                arrays.push(args.tensor(&format!("input{index}"))?);
                index += 1;
            }
            let axis = args.axis("axis")?;
            args.execution()?;
            vec![mlx_error(ops::concatenate_axis(&arrays, axis))?]
        }
        "ops.sum.axis_optional" => {
            let a = args.tensor("input0")?;
            let axis = args.optional_axis("axis")?;
            let keep = args.optional_bool("keepdims")?;
            args.execution()?;
            vec![match axis {
                Some(axis) => mlx_error(ops::sum_axis(&a, axis, keep))?,
                None => mlx_error(ops::sum(&a, keep))?,
            }]
        }
        "ops.sum.axes" => {
            let a = args.tensor("input0")?;
            let axes = args.axes("axes")?;
            let keep = args.optional_bool("keepdims")?;
            args.execution()?;
            vec![mlx_error(ops::sum_axes(&a, &axes, keep))?]
        }
        "array.sum.all_optional" => {
            let a = args.tensor("input0")?;
            let keep = args.optional_bool("keepdims")?;
            args.execution()?;
            vec![mlx_error(a.sum(keep))?]
        }
        "ops.take.array_indices" => {
            let a = args.tensor("input0")?;
            let indices = args.tensor("input1")?;
            args.execution()?;
            vec![mlx_error(ops::indexing::take(&a, &indices))?]
        }
        "ops.take_axis.array_indices_axis" => {
            let a = args.tensor("input0")?;
            let indices = args.tensor("input1")?;
            let axis = args.axis("axis")?;
            args.execution()?;
            vec![mlx_error(ops::indexing::take_axis(&a, &indices, axis))?]
        }
        "array.as_dtype" => {
            let a = args.tensor("input0")?;
            let dtype = match args.take("dtype")? {
                Arg::Dtype { value, .. } => dtype_from_name(value)?,
                _ => return Err("dtype is not a dtype argument".into()),
            };
            args.execution()?;
            vec![mlx_error(a.as_dtype(dtype))?]
        }
        "ops.exp.explicit_cpu" => {
            let a = args.tensor("input0")?;
            args.execution()?;
            vec![mlx_error(ops::exp_device(&a, StreamOrDevice::cpu()))?]
        }
        "array.reshape.explicit_cpu" => {
            let a = args.tensor("input0")?;
            let shape = args.shape("shape")?;
            args.execution()?;
            vec![mlx_error(a.reshape_device(&shape, StreamOrDevice::cpu()))?]
        }
        "ops.divmod.explicit_cpu" => {
            let a = args.tensor("input0")?;
            let b = args.tensor("input1")?;
            args.execution()?;
            let (q, r) = mlx_error(ops::divmod_device(&a, &b, StreamOrDevice::cpu()))?;
            vec![q, r]
        }
        "ops.windows.bartlett" => {
            let ScalarValue::I32(size) = args.scalar("size")? else {
                return Err("size scalar type mismatch".into());
            };
            args.execution()?;
            vec![mlx_error(ops::windows::bartlett(
                usize::try_from(size).map_err(|_| "size must be nonnegative")?,
            ))?]
        }
        "ops.windows.blackman" => {
            let ScalarValue::I32(size) = args.scalar("size")? else {
                return Err("size scalar type mismatch".into());
            };
            args.execution()?;
            vec![mlx_error(ops::windows::blackman(
                usize::try_from(size).map_err(|_| "size must be nonnegative")?,
            ))?]
        }
        "ops.windows.hamming" => {
            let ScalarValue::I32(size) = args.scalar("size")? else {
                return Err("size scalar type mismatch".into());
            };
            args.execution()?;
            vec![mlx_error(ops::windows::hamming(
                usize::try_from(size).map_err(|_| "size must be nonnegative")?,
            ))?]
        }
        "ops.windows.hann" => {
            let ScalarValue::I32(size) = args.scalar("size")? else {
                return Err("size scalar type mismatch".into());
            };
            args.execution()?;
            vec![mlx_error(ops::windows::hann(
                usize::try_from(size).map_err(|_| "size must be nonnegative")?,
            ))?]
        }
        "fft.fftfreq" => {
            let ScalarValue::I32(n) = args.scalar("n")? else {
                return Err("n scalar type mismatch".into());
            };
            let ScalarValue::F32(d) = args.scalar("d")? else {
                return Err("d scalar type mismatch".into());
            };
            args.execution()?;
            vec![mlx_error(fft::fftfreq(
                usize::try_from(n).map_err(|_| "n must be nonnegative")?,
                f64::from(d),
            ))?]
        }
        "fft.rfftfreq" => {
            let ScalarValue::I32(n) = args.scalar("n")? else {
                return Err("n scalar type mismatch".into());
            };
            let ScalarValue::F32(d) = args.scalar("d")? else {
                return Err("d scalar type mismatch".into());
            };
            args.execution()?;
            vec![mlx_error(fft::rfftfreq(
                usize::try_from(n).map_err(|_| "n must be nonnegative")?,
                f64::from(d),
            ))?]
        }
        other => return Err(format!("missing adapter {other}")),
    };
    args.finish()?;
    Ok(outputs)
}
