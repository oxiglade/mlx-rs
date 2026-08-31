use super::oracle::{
    dtype_from_name, mlx_error, Arg, Args, Case, ExecutionTarget, GgufCase, GgufKind,
    GgufObservation, GgufObservedMetadata, GgufRecipe, IndexRecipe, ScalarValue, UpdateModeRecipe,
};
use half::{bf16, f16};
use mlx_rs::{
    io::{GgufError, GgufFile, GgufMetadataValue},
    linalg,
    ops::{
        self,
        indexing::{
            Ellipsis, IndexUpdateError, IntoStrideBy, NewAxis, TryIndexMutOp, TryIndexUpdateOp,
            UpdateMode,
        },
        CountNonzeroOptions, LinspaceOptions, LogCumsumExpOptions, TraceOptions,
    },
    with_device, with_stream, Array, Axes, Device, Dtype, Stream,
};
use safetensors::SafeTensors;
use std::{collections::BTreeMap, path::Path};

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
    "array.count_nonzero.all",
    "array.count_nonzero.axis",
    "array.count_nonzero.axes",
    "array.count_nonzero.explicit_cpu",
    "array.diff",
    "array.flip.all",
    "array.flip.axis",
    "array.flip.axes",
    "linalg.det",
    "linalg.det.gpu",
    "linalg.slogdet",
    "ops.linspace.f32",
    "ops.linspace.f64",
    "ops.linspace.i32",
    "array.logcumsumexp",
    "ops.logical_xor",
    "array.search_sorted.left",
    "array.search_sorted.right",
    "array.search_sorted.explicit_gpu",
    "array.trace.default",
    "array.trace.options",
    "array.trace.dtype",
    "array.trunc",
    "array.unstack",
    "ops.vecdot",
    "array.try_index_update",
    "array.try_index_update.source_unchanged",
    "array.try_index_mut.compatibility",
    "gguf.load",
    "gguf.load_error",
    "gguf.absence",
    "gguf.wrong_kind",
    "gguf.prevalidation",
    "gguf.construct",
];

fn index_update_error(result: Result<Array, IndexUpdateError>) -> Result<Array, String> {
    result.map_err(|error| match error {
        IndexUpdateError::ZeroStride { .. } => format!("[index_update:zero_stride] {error}"),
        IndexUpdateError::Exception(_) => format!("[index_update:exception] {error}"),
    })
}

fn update_mode(mode: UpdateModeRecipe) -> UpdateMode {
    match mode {
        UpdateModeRecipe::Replace => UpdateMode::Replace,
        UpdateModeRecipe::Add => UpdateMode::Add,
        UpdateModeRecipe::Min => UpdateMode::Min,
        UpdateModeRecipe::Max => UpdateMode::Max,
        UpdateModeRecipe::Product => UpdateMode::Product,
    }
}

fn dispatch_index_update(case: &Case, args: &mut Args<'_>) -> Result<Vec<Array>, String> {
    let source = args.tensor("input0")?;
    let update = args.tensor("input1")?;
    let index = args.index("index")?;
    let mode = update_mode(args.update_mode("mode")?);
    args.execution()?;

    if case.rust_call == "array.try_index_mut.compatibility" {
        let mut result = source.clone();
        result
            .try_index_mut(1..4, &update)
            .map_err(|error| error.to_string())?;
        return Ok(vec![result]);
    }

    let result = match index {
        IndexRecipe::PositiveSlice => source.try_index_update(1..4, &update, mode),
        IndexRecipe::NegativeStride => source.try_index_update((4..1).stride_by(-1), &update, mode),
        IndexRecipe::Advanced | IndexRecipe::DuplicateAdvanced => {
            let indices = args.tensor("input2")?;
            source.try_index_update(&indices, &update, mode)
        }
        IndexRecipe::Tuple2d => source.try_index_update((1..3, 0..2), &update, mode),
        IndexRecipe::NegativeIndex => source.try_index_update(-1, &update, mode),
        IndexRecipe::EllipsisNewAxis => {
            source.try_index_update((Ellipsis, NewAxis, 1..3), &update, mode)
        }
        IndexRecipe::TupleColumns => source.try_index_update((.., 1..3), &update, mode),
        IndexRecipe::Full => source.try_index_update(.., &update, mode),
        IndexRecipe::Empty => source.try_index_update(3..3, &update, mode),
        IndexRecipe::Clipped => source.try_index_update(-100..100, &update, mode),
        IndexRecipe::Noop => source.try_index_update(100..200, &update, mode),
        IndexRecipe::NegativeBounds => source.try_index_update(-4..-1, &update, mode),
        IndexRecipe::AdvancedTuple => {
            let indices = args.tensor("input2")?;
            source.try_index_update((&indices, -1), &update, mode)
        }
        IndexRecipe::ZeroStride => source.try_index_update((..).stride_by(0), &update, mode),
    };
    let result = index_update_error(result)?;
    if case.rust_call == "array.try_index_update.source_unchanged" {
        let ScalarValue::Bool(return_source) = args.scalar("return_source")? else {
            return Err("return_source scalar type mismatch".into());
        };
        if !return_source {
            return Err("source_unchanged adapter requires return_source".into());
        }
        Ok(vec![result, source])
    } else {
        Ok(vec![result])
    }
}

fn gguf_error(error: GgufError) -> (String, Option<(GgufKind, GgufKind)>) {
    let variant = match error {
        GgufError::NotFile => "not_file",
        GgufError::InvalidPathUtf8 => "invalid_path_utf8",
        GgufError::UnsupportedExtension => "unsupported_extension",
        GgufError::InteriorNul => "interior_nul",
        GgufError::InvalidUtf8 => "invalid_utf8",
        GgufError::ArrayKeyAlreadyExists { .. } => "array_key_already_exists",
        GgufError::MetadataKeyAlreadyExists { .. } => "metadata_key_already_exists",
        GgufError::WrongMetadataKind {
            expected, actual, ..
        } => {
            let kind = |value| match value {
                mlx_rs::io::GgufMetadataKind::Array => GgufKind::Array,
                mlx_rs::io::GgufMetadataKind::String => GgufKind::String,
                mlx_rs::io::GgufMetadataKind::Strings => GgufKind::Strings,
            };
            return (
                "wrong_metadata_kind".into(),
                Some((kind(expected), kind(actual))),
            );
        }
        GgufError::UnsupportedTensorDtype { .. } => "unsupported_tensor_dtype",
        GgufError::UnsupportedMetadataArrayDtype { .. } => "unsupported_metadata_array_dtype",
        GgufError::InvalidMetadataArrayRank { .. } => "invalid_metadata_array_rank",
        GgufError::EmptyMetadataArray => "empty_metadata_array",
        GgufError::Exception(_) => "exception",
        _ => "unknown",
    };
    (variant.into(), None)
}

fn empty_gguf_observation() -> GgufObservation {
    GgufObservation {
        array_keys: Vec::new(),
        arrays: BTreeMap::new(),
        metadata: BTreeMap::new(),
        array_absent: None,
        metadata_absent: None,
        errors: Vec::new(),
        error_kinds: Vec::new(),
        dequantized: None,
    }
}

fn array_for_dtype(dtype: Dtype) -> Array {
    match dtype {
        Dtype::Bool => Array::from_bool(true),
        Dtype::Uint8 => Array::from_slice(&[1_u8], &[1]),
        Dtype::Uint16 => Array::from_slice(&[1_u16], &[1]),
        Dtype::Uint32 => Array::from_slice(&[1_u32], &[1]),
        Dtype::Uint64 => Array::from_slice(&[1_u64], &[1]),
        Dtype::Int8 => Array::from_slice(&[1_i8], &[1]),
        Dtype::Int16 => Array::from_slice(&[1_i16], &[1]),
        Dtype::Int32 => Array::from_slice(&[1_i32], &[1]),
        Dtype::Int64 => Array::from_slice(&[1_i64], &[1]),
        Dtype::Float16 => Array::from_slice(&[f16::from_f32(1.0)], &[1]),
        Dtype::Float32 => Array::from_f32(1.0),
        Dtype::Float64 => Array::from_f64(1.0),
        Dtype::Bfloat16 => Array::from_slice(&[bf16::from_f32(1.0)], &[1]),
        Dtype::Complex64 => Array::from_complex(mlx_rs::complex64::new(1.0, 0.0)),
    }
}

pub(super) fn dispatch_gguf(root: &Path, case: &GgufCase) -> Result<GgufObservation, String> {
    let mut observed = empty_gguf_observation();
    match &case.recipe {
        GgufRecipe::Load {
            path,
            execution,
            dequantize,
        } => {
            let path = root.join(path);
            let loaded = match execution {
                ExecutionTarget::DefaultCpu => GgufFile::load(&path),
                ExecutionTarget::ExplicitCpu => {
                    with_device(Device::cpu(), || GgufFile::load(&path))
                }
                ExecutionTarget::ExplicitGpu => {
                    return Err("GGUF adapter does not support explicit GPU execution".into())
                }
            };
            if case.rust_call == "gguf.load_error" {
                let error = loaded.expect_err("error load case must fail");
                let (variant, kinds) = gguf_error(error);
                observed.errors.push(variant);
                if let Some(kinds) = kinds {
                    observed.error_kinds.push(kinds);
                }
                return Ok(observed);
            }
            let file = loaded.map_err(|error| error.to_string())?;
            observed.array_keys = file.array_keys().map_err(|error| error.to_string())?;
            observed.arrays = file
                .arrays()
                .map_err(|error| error.to_string())?
                .into_iter()
                .collect();
            for expected in &case.expected.metadata {
                let value = file
                    .get_metadata(&expected.key)
                    .map_err(|error| error.to_string())?
                    .ok_or_else(|| format!("missing metadata {}", expected.key))?;
                let value = match value {
                    GgufMetadataValue::Array(value) => GgufObservedMetadata::Array(value),
                    GgufMetadataValue::String(value) => GgufObservedMetadata::String(value),
                    GgufMetadataValue::Strings(value) => GgufObservedMetadata::Strings(value),
                };
                observed.metadata.insert(expected.key.clone(), value);
            }
            if let Some(dequantize) = dequantize {
                let weight = observed
                    .arrays
                    .get("quantized.weight")
                    .ok_or("missing quantized weight")?;
                let scales = observed
                    .arrays
                    .get("quantized.scales")
                    .ok_or("missing quantized scales")?;
                let biases = observed
                    .arrays
                    .get("quantized.biases")
                    .ok_or("missing quantized biases")?;
                observed.dequantized = Some(
                    ops::dequantize(
                        weight,
                        scales,
                        biases,
                        dequantize.group_size,
                        dequantize.bits,
                    )
                    .map_err(|error| error.to_string())?,
                );
            }
        }
        GgufRecipe::Absence {
            path,
            array_key,
            metadata_key,
        } => {
            let file = GgufFile::load(root.join(path)).map_err(|error| error.to_string())?;
            observed.array_absent = Some(
                file.get_array(array_key)
                    .map_err(|error| error.to_string())?
                    .is_none(),
            );
            observed.metadata_absent = Some(
                file.get_metadata(metadata_key)
                    .map_err(|error| error.to_string())?
                    .is_none(),
            );
        }
        GgufRecipe::WrongKind {
            path,
            key,
            requested,
        } => {
            let file = GgufFile::load(root.join(path)).map_err(|error| error.to_string())?;
            let result = match requested {
                GgufKind::Array => file.get_metadata_array(key).map(|_| ()),
                GgufKind::String => file.get_metadata_string(key).map(|_| ()),
                GgufKind::Strings => file.get_metadata_strings(key).map(|_| ()),
            };
            let error = result.expect_err("wrong kind must fail");
            let (variant, kinds) = gguf_error(error);
            observed.errors.push(variant);
            if let Some(kinds) = kinds {
                observed.error_kinds.push(kinds);
            }
        }
        GgufRecipe::TensorRejects { accepted, dtypes } => {
            let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
            let output = directory.path().join("accepted.gguf");
            let mut accepted_file = GgufFile::new().map_err(|error| error.to_string())?;
            for (index, dtype) in accepted.iter().enumerate() {
                let dtype = dtype_from_name(dtype)?;
                accepted_file
                    .insert_array(format!("accepted.{index}"), &array_for_dtype(dtype))
                    .map_err(|error| error.to_string())?;
            }
            accepted_file
                .save(&output)
                .map_err(|error| error.to_string())?;
            let loaded = GgufFile::load(output).map_err(|error| error.to_string())?;
            let keys = loaded.array_keys().map_err(|error| error.to_string())?;
            if keys.len() != accepted.len() {
                return Err("accepted tensor dtype count differs".into());
            }
            for (index, expected) in accepted.iter().enumerate() {
                let key = format!("accepted.{index}");
                let array = loaded
                    .get_array(&key)
                    .map_err(|error| error.to_string())?
                    .ok_or_else(|| format!("accepted tensor {key} is absent"))?;
                if array.dtype() != dtype_from_name(expected)? {
                    return Err(format!("accepted tensor {key} dtype differs"));
                }
            }
            for (index, dtype) in dtypes.iter().enumerate() {
                let dtype = dtype_from_name(dtype)?;
                let mut file = GgufFile::new().map_err(|error| error.to_string())?;
                let array = array_for_dtype(dtype);
                let error = file
                    .insert_array(format!("rejected.{index}"), &array)
                    .expect_err("rejected tensor dtype must fail");
                observed.errors.push(gguf_error(error).0);
            }
        }
        GgufRecipe::MetadataRejects {
            accepted,
            dtypes,
            ranks,
            empty,
        } => {
            let mut file = GgufFile::new().map_err(|error| error.to_string())?;
            for (index, dtype) in accepted.iter().enumerate() {
                let dtype = dtype_from_name(dtype)?;
                let base = array_for_dtype(dtype);
                let scalar = base.reshape(&[]).map_err(|error| error.to_string())?;
                let vector = base.reshape(&[1]).map_err(|error| error.to_string())?;
                file.insert_metadata(format!("accepted.scalar.{index}"), scalar)
                    .map_err(|error| error.to_string())?;
                file.insert_metadata(format!("accepted.vector.{index}"), vector)
                    .map_err(|error| error.to_string())?;
            }
            for (index, dtype) in dtypes.iter().enumerate() {
                let error = file
                    .insert_metadata(
                        format!("dtype.{index}"),
                        array_for_dtype(dtype_from_name(dtype)?),
                    )
                    .expect_err("rejected metadata dtype must fail");
                observed.errors.push(gguf_error(error).0);
            }
            for rank in ranks {
                let shape = vec![1; *rank];
                let error = file
                    .insert_metadata("rank", Array::from_slice(&[1_i32], &shape))
                    .expect_err("rejected metadata rank must fail");
                observed.errors.push(gguf_error(error).0);
            }
            if *empty {
                let error = file
                    .insert_metadata("empty", Array::from_slice::<i32>(&[], &[0]))
                    .expect_err("empty metadata must fail");
                observed.errors.push(gguf_error(error).0);
            }
            let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
            let output = directory.path().join("metadata-accepted.gguf");
            file.save(&output).map_err(|error| error.to_string())?;
            let loaded = GgufFile::load(output).map_err(|error| error.to_string())?;
            for (index, expected) in accepted.iter().enumerate() {
                let expected_dtype = dtype_from_name(expected)?;
                for (shape_name, expected_shape) in [("scalar", &[][..]), ("vector", &[1][..])] {
                    let key = format!("accepted.{shape_name}.{index}");
                    let array = loaded
                        .get_metadata_array(&key)
                        .map_err(|error| error.to_string())?
                        .ok_or_else(|| format!("accepted metadata {key} is absent"))?;
                    if array.dtype() != expected_dtype || array.shape() != expected_shape {
                        return Err(format!("accepted metadata {key} declaration differs"));
                    }
                }
            }
        }
        GgufRecipe::ConstructSave {
            path,
            same_spelling,
            metadata_value,
            non_contiguous_shape,
        } => {
            let &[rows, columns] = non_contiguous_shape.as_slice() else {
                return Err("construct_save requires a rank-two shape".into());
            };
            if rows <= 0 || columns <= 0 {
                return Err("construct_save dimensions must be positive".into());
            }
            let values = (0..rows * columns)
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            let source = Array::from_slice(&values, &[rows, columns]);
            let transposed = ops::transpose(&source).map_err(|error| error.to_string())?;
            let mut file = GgufFile::new().map_err(|error| error.to_string())?;
            file.insert_array(same_spelling, &transposed)
                .map_err(|error| error.to_string())?;
            file.insert_metadata(same_spelling, metadata_value.as_str())
                .map_err(|error| error.to_string())?;
            let duplicate = file
                .insert_array(same_spelling, &transposed)
                .expect_err("duplicate array must fail");
            observed.errors.push(gguf_error(duplicate).0);
            let duplicate = file
                .insert_metadata(same_spelling, vec!["different".to_owned()])
                .expect_err("duplicate metadata must fail");
            observed.errors.push(gguf_error(duplicate).0);
            let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
            let output_name = Path::new(path)
                .file_name()
                .ok_or("construct_save path has no file name")?;
            let output = directory.path().join(output_name);
            file.save(&output).map_err(|error| error.to_string())?;
            let loaded = GgufFile::load(output).map_err(|error| error.to_string())?;
            observed.array_keys = loaded.array_keys().map_err(|error| error.to_string())?;
            observed.arrays = loaded
                .arrays()
                .map_err(|error| error.to_string())?
                .into_iter()
                .collect();
            let metadata = loaded
                .get_metadata_string(same_spelling)
                .map_err(|error| error.to_string())?
                .ok_or("constructed metadata is absent")?;
            observed.metadata.insert(
                same_spelling.clone(),
                GgufObservedMetadata::String(metadata),
            );
        }
    }
    Ok(observed)
}

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
            vec![mlx_error(ops::select(&c, &a, &b))?]
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
            vec![mlx_error(with_device(Device::cpu(), || ops::add(&a, &b)))?]
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
            mlx_error(ops::split_equal(&a, parts, axis))?
        }
        "ops.split.sections" => {
            let a = args.tensor("input0")?;
            let indices = args.axes("indices")?;
            let axis = args.optional_axis("axis")?;
            args.execution()?;
            mlx_error(ops::split_at_indices(&a, &indices, axis))?
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
            vec![mlx_error(ops::concatenate(&arrays, axis))?]
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
            vec![mlx_error(with_device(Device::cpu(), || ops::exp(&a)))?]
        }
        "array.reshape.explicit_cpu" => {
            let a = args.tensor("input0")?;
            let shape = args.shape("shape")?;
            args.execution()?;
            vec![mlx_error(with_device(Device::cpu(), || a.reshape(&shape)))?]
        }
        "ops.divmod.explicit_cpu" => {
            let a = args.tensor("input0")?;
            let b = args.tensor("input1")?;
            args.execution()?;
            let (q, r) = mlx_error(with_device(Device::cpu(), || ops::divmod(&a, &b)))?;
            vec![q, r]
        }
        "array.count_nonzero.all" => {
            let input = args.tensor("input0")?;
            let keep_dims = args.optional_bool("keepdims")?.unwrap_or(false);
            args.execution()?;
            vec![mlx_error(input.count_nonzero(CountNonzeroOptions {
                axes: Axes::All,
                keep_dims,
            }))?]
        }
        "array.count_nonzero.axis" => {
            let input = args.tensor("input0")?;
            let axis = args.axis("axis")?;
            let keep_dims = args.optional_bool("keepdims")?.unwrap_or(false);
            args.execution()?;
            vec![mlx_error(input.count_nonzero(CountNonzeroOptions {
                axes: Axes::Axis(axis),
                keep_dims,
            }))?]
        }
        "array.count_nonzero.axes" => {
            let input = args.tensor("input0")?;
            let axes = args.axes("axes")?;
            let keep_dims = args.optional_bool("keepdims")?.unwrap_or(false);
            args.execution()?;
            vec![mlx_error(input.count_nonzero(CountNonzeroOptions {
                axes: Axes::Axes(axes),
                keep_dims,
            }))?]
        }
        "array.count_nonzero.explicit_cpu" => {
            let input = args.tensor("input0")?;
            let keep_dims = args.optional_bool("keepdims")?.unwrap_or(false);
            args.execution()?;
            vec![mlx_error(with_stream(&Stream::cpu(), || {
                input.count_nonzero(CountNonzeroOptions {
                    axes: Axes::All,
                    keep_dims,
                })
            }))?]
        }
        "array.diff" => {
            let input = args.tensor("input0")?;
            let ScalarValue::I32(n) = args.scalar("n")? else {
                return Err("n scalar type mismatch".into());
            };
            let axis = args.axis("axis")?;
            args.execution()?;
            vec![mlx_error(input.diff(n, axis))?]
        }
        "array.flip.all" => {
            let input = args.tensor("input0")?;
            args.execution()?;
            vec![mlx_error(input.flip(Axes::All))?]
        }
        "array.flip.axis" => {
            let input = args.tensor("input0")?;
            let axis = args.axis("axis")?;
            args.execution()?;
            vec![mlx_error(input.flip(Axes::Axis(axis)))?]
        }
        "array.flip.axes" => {
            let input = args.tensor("input0")?;
            let axes = args.axes("axes")?;
            args.execution()?;
            vec![mlx_error(input.flip(Axes::Axes(axes)))?]
        }
        "linalg.det" => {
            let input = args.tensor("input0")?;
            args.execution()?;
            vec![mlx_error(linalg::det(&input))?]
        }
        "linalg.det.gpu" => {
            let input = args.tensor("input0")?;
            args.execution()?;
            vec![mlx_error(with_stream(&Stream::gpu(), || {
                linalg::det(&input)
            }))?]
        }
        "linalg.slogdet" => {
            let input = args.tensor("input0")?;
            args.execution()?;
            let result = mlx_error(linalg::slogdet(&input))?;
            vec![result.sign, result.log_abs_det]
        }
        "ops.linspace.f32" => {
            let ScalarValue::F64(start) = args.scalar("start")? else {
                return Err("start scalar type mismatch".into());
            };
            let ScalarValue::F64(stop) = args.scalar("stop")? else {
                return Err("stop scalar type mismatch".into());
            };
            let ScalarValue::I32(count) = args.scalar("count")? else {
                return Err("count scalar type mismatch".into());
            };
            let ScalarValue::Bool(endpoint) = args.scalar("endpoint")? else {
                return Err("endpoint scalar type mismatch".into());
            };
            let dtype = match args.take("dtype")? {
                Arg::Dtype { value, .. } => dtype_from_name(value)?,
                _ => return Err("dtype is not a dtype argument".into()),
            };
            if dtype != Dtype::Float32 {
                return Err("linspace dtype does not match adapter".into());
            }
            args.execution()?;
            vec![mlx_error(ops::linspace::<_, f32>(
                start,
                stop,
                LinspaceOptions { count, endpoint },
            ))?]
        }
        "ops.linspace.f64" => {
            let ScalarValue::F64(start) = args.scalar("start")? else {
                return Err("start scalar type mismatch".into());
            };
            let ScalarValue::F64(stop) = args.scalar("stop")? else {
                return Err("stop scalar type mismatch".into());
            };
            let ScalarValue::I32(count) = args.scalar("count")? else {
                return Err("count scalar type mismatch".into());
            };
            let ScalarValue::Bool(endpoint) = args.scalar("endpoint")? else {
                return Err("endpoint scalar type mismatch".into());
            };
            let dtype = match args.take("dtype")? {
                Arg::Dtype { value, .. } => dtype_from_name(value)?,
                _ => return Err("dtype is not a dtype argument".into()),
            };
            if dtype != Dtype::Float64 {
                return Err("linspace dtype does not match adapter".into());
            }
            args.execution()?;
            vec![mlx_error(ops::linspace::<_, f64>(
                start,
                stop,
                LinspaceOptions { count, endpoint },
            ))?]
        }
        "ops.linspace.i32" => {
            let ScalarValue::F64(start) = args.scalar("start")? else {
                return Err("start scalar type mismatch".into());
            };
            let ScalarValue::F64(stop) = args.scalar("stop")? else {
                return Err("stop scalar type mismatch".into());
            };
            let ScalarValue::I32(count) = args.scalar("count")? else {
                return Err("count scalar type mismatch".into());
            };
            let ScalarValue::Bool(endpoint) = args.scalar("endpoint")? else {
                return Err("endpoint scalar type mismatch".into());
            };
            let dtype = match args.take("dtype")? {
                Arg::Dtype { value, .. } => dtype_from_name(value)?,
                _ => return Err("dtype is not a dtype argument".into()),
            };
            if dtype != Dtype::Int32 {
                return Err("linspace dtype does not match adapter".into());
            }
            args.execution()?;
            vec![mlx_error(ops::linspace::<_, i32>(
                start,
                stop,
                LinspaceOptions { count, endpoint },
            ))?]
        }
        "array.logcumsumexp" => {
            let input = args.tensor("input0")?;
            let axis = args.optional_axis("axis")?;
            let ScalarValue::Bool(reverse) = args.scalar("reverse")? else {
                return Err("reverse scalar type mismatch".into());
            };
            let ScalarValue::Bool(inclusive) = args.scalar("inclusive")? else {
                return Err("inclusive scalar type mismatch".into());
            };
            args.execution()?;
            vec![mlx_error(input.logcumsumexp(LogCumsumExpOptions {
                axis,
                reverse,
                inclusive,
            }))?]
        }
        "ops.logical_xor" => {
            let lhs = args.tensor("input0")?;
            let rhs = args.tensor("input1")?;
            args.execution()?;
            vec![mlx_error(ops::logical_xor(&lhs, &rhs))?]
        }
        "array.search_sorted.left" | "array.search_sorted.right" => {
            let sequence = args.tensor("input0")?;
            let values = args.tensor("input1")?;
            let ScalarValue::Bool(right) = args.scalar("right")? else {
                return Err("right scalar type mismatch".into());
            };
            args.execution()?;
            let side = if case.rust_call.ends_with(".left") {
                ops::SearchSide::Left
            } else {
                ops::SearchSide::Right
            };
            if right != matches!(side, ops::SearchSide::Right) {
                return Err("search side does not match adapter".into());
            }
            vec![mlx_error(sequence.search_sorted(&values, side))?]
        }
        "array.search_sorted.explicit_gpu" => {
            let sequence = args.tensor("input0")?;
            let values = args.tensor("input1")?;
            let ScalarValue::Bool(right) = args.scalar("right")? else {
                return Err("right scalar type mismatch".into());
            };
            if right {
                return Err("GPU search adapter expected the left side".into());
            }
            args.execution()?;
            vec![mlx_error(with_stream(&Stream::gpu(), || {
                sequence.search_sorted(&values, ops::SearchSide::Left)
            }))?]
        }
        "array.trace.default" => {
            let input = args.tensor("input0")?;
            args.execution()?;
            vec![mlx_error(input.trace(TraceOptions::default()))?]
        }
        "array.trace.options" => {
            let input = args.tensor("input0")?;
            let ScalarValue::I32(offset) = args.scalar("offset")? else {
                return Err("offset scalar type mismatch".into());
            };
            let axis1 = args.axis("axis1")?;
            let axis2 = args.axis("axis2")?;
            args.execution()?;
            vec![mlx_error(input.trace(TraceOptions {
                offset,
                axis1,
                axis2,
                dtype: None,
            }))?]
        }
        "array.trace.dtype" => {
            let input = args.tensor("input0")?;
            let ScalarValue::I32(offset) = args.scalar("offset")? else {
                return Err("offset scalar type mismatch".into());
            };
            let axis1 = args.axis("axis1")?;
            let axis2 = args.axis("axis2")?;
            let dtype = match args.take("dtype")? {
                Arg::Dtype { value, .. } => dtype_from_name(value)?,
                _ => return Err("dtype is not a dtype argument".into()),
            };
            args.execution()?;
            vec![mlx_error(input.trace(TraceOptions {
                offset,
                axis1,
                axis2,
                dtype: Some(dtype),
            }))?]
        }
        "array.trunc" => {
            let input = args.tensor("input0")?;
            args.execution()?;
            vec![mlx_error(input.trunc())?]
        }
        "array.unstack" => {
            let input = args.tensor("input0")?;
            let axis = args.axis("axis")?;
            args.execution()?;
            mlx_error(input.unstack(axis))?
        }
        "ops.vecdot" => {
            let lhs = args.tensor("input0")?;
            let rhs = args.tensor("input1")?;
            let axis = args.axis("axis")?;
            args.execution()?;
            vec![mlx_error(ops::vecdot(&lhs, &rhs, axis))?]
        }
        "array.try_index_update"
        | "array.try_index_update.source_unchanged"
        | "array.try_index_mut.compatibility" => dispatch_index_update(case, &mut args)?,
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
            vec![mlx_error(mlx_rs::fft::fftfreq(
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
            vec![mlx_error(mlx_rs::fft::rfftfreq(
                usize::try_from(n).map_err(|_| "n must be nonnegative")?,
                f64::from(d),
            ))?]
        }
        other => return Err(format!("missing adapter {other}")),
    };
    args.finish()?;
    Ok(outputs)
}
