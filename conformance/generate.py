#!/usr/bin/env python3
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import struct
import sys
import tempfile
from pathlib import Path

EXPECTED_PYTHON = (3, 12, 14)
EXPECTED_ARCH = "arm64"
EXPECTED_MLX = "0.30.6"
EXPECTED_NUMPY = "2.2.6"
CORPUS_SEED = "mlx-rs-committed-cpu-ops-v1"
SUITES = ("arithmetic", "dtypes", "errors", "execution", "reductions", "shapes")
ROOT = Path(__file__).resolve().parent


def check_environment():
    if sys.version_info[:3] != EXPECTED_PYTHON:
        raise SystemExit(f"requires Python 3.12.14, got {platform.python_version()}")
    if platform.machine() != EXPECTED_ARCH:
        raise SystemExit(f"requires arm64, got {platform.machine()}")
    import mlx.core as mx
    import numpy as np

    if importlib.metadata.version("mlx") != EXPECTED_MLX:
        raise SystemExit(f"requires mlx {EXPECTED_MLX}")
    if importlib.metadata.version("mlx-metal") != EXPECTED_MLX:
        raise SystemExit(f"requires mlx-metal {EXPECTED_MLX}")
    if np.__version__ != EXPECTED_NUMPY:
        raise SystemExit(f"requires numpy {EXPECTED_NUMPY}, got {np.__version__}")
    return mx, np


def arg_tensor(name, ref, encoding=None, imag_ref=None):
    value = {"name": name, "kind": "tensor", "ref": ref}
    if encoding:
        value["encoding"] = encoding
        value["imag_ref"] = imag_ref
    return value


def arg_scalar(name, scalar_type, value=None, bits=None, real_bits=None, imag_bits=None):
    result = {"name": name, "kind": "scalar", "type": scalar_type}
    if value is not None:
        result["value"] = value
    if bits is not None:
        result["bits"] = bits
    if real_bits is not None:
        result["real_bits"] = real_bits
        result["imag_bits"] = imag_bits
    return result


def execution(explicit=False):
    return {"name": "execution", "kind": "execution", "target": "explicit_cpu" if explicit else "default_cpu"}


def source(dtype, shape, values=None, distribution="small_integers"):
    item = {"dtype": dtype, "shape": shape}
    if values is None:
        item["random"] = distribution
    else:
        item["values"] = values
    return item


def case(case_id, suite, semantic_op, rust_call, inputs, extra_args=None, policy="exact_numeric", explicit=False):
    return {
        "id": case_id,
        "suite": suite,
        "semantic_op": semantic_op,
        "recipe": semantic_op,
        "rust_call": rust_call,
        "sources": inputs,
        "extra_args": extra_args or [],
        "policy": policy,
        "execution": execution(explicit),
    }


def axes(values):
    return {"name": "axes", "kind": "axes", "values": values}


def axis(value):
    return {"name": "axis", "kind": "optional_axis", "value": value}


def shape(values):
    return {"name": "shape", "kind": "shape", "values": values}


def keepdims(value):
    return {"name": "keepdims", "kind": "optional_bool", "value": value}


def dtype_arg(value):
    return {"name": "dtype", "kind": "dtype", "value": value}


def data_movement_policy(op, inputs):
    data_movement_ops = {
        "reshape", "transpose", "transpose_axes", "expand_dims", "split", "split_sections",
        "concatenate", "broadcast_arrays",
    }
    float_dtypes = {"F16", "BF16", "F32", "F64", "C64"}
    if op in data_movement_ops and any(item["dtype"] in float_dtypes for item in inputs):
        return "exact_bits"
    return "exact_numeric"


def build_specs():
    specs = []
    add = lambda i, op, call, ins, **kw: specs.append(case(f"arithmetic.{i:03d}", "arithmetic", op, call, ins, **kw))
    add(1, "add", "ops.add.array_array", [source("F32", [5]), source("F32", [5])], policy="elementwise_float")
    add(2, "add", "array.add.array", [source("I32", []), source("I32", [])])
    add(3, "add", "operator.add.array", [source("I16", [2, 3]), source("I16", [2, 3])])
    add(4, "subtract", "ops.subtract.array_array", [source("F32", [2, 3]), source("F32", [2, 3])], policy="elementwise_float")
    add(5, "multiply", "array.multiply.array", [source("F32", [2, 3]), source("F32", [3])], policy="elementwise_float")
    add(6, "divide", "operator.divide.array", [source("F32", [4], [2.0, -4.0, 9.0, 1.0]), source("F32", [4], [2.0, 2.0, 3.0, 4.0])], policy="elementwise_float")
    add(7, "negative", "operator.neg.array", [source("I64", [4], [1, -2, 3, -4])])
    add(8, "abs", "array.abs", [source("F32", [5], [-2.5, -0.0, 0.0, 3.25, 8.0])], policy="exact_bits")
    add(9, "exp", "ops.exp.array", [source("F32", [4], [-1.0, 0.0, 0.5, 2.0])], policy="elementwise_float")
    add(10, "where", "ops.where.ternary", [source("BOOL", [2, 2], [True, False, False, True]), source("F32", [2, 2]), source("F32", [2, 2])], policy="exact_bits")
    add(11, "divmod", "ops.divmod.array_array", [source("I32", [5], [9, -9, 10, 7, 0]), source("I32", [5], [4, 4, 3, 2, 5])])
    add(12, "add_scalar", "operator.add.bool_rhs", [source("BOOL", [4], [True, False, True, False])], extra_args=[arg_scalar("rhs", "bool", value=True)])
    add(13, "add_scalar", "operator.add.i32_rhs", [source("I32", [4], [-2, 0, 4, 9])], extra_args=[arg_scalar("rhs", "i32", value=-7)])
    add(14, "add_scalar", "operator.add.f32_rhs", [source("F32", [4], [-2.0, 0.0, 4.0, 9.0])], extra_args=[arg_scalar("rhs", "f32", bits="0x3f400000")], policy="exact_bits")
    add(15, "add_scalar", "operator.add.complex64_rhs", [source("C64", [3], [[1.0, 2.0], [-3.0, 0.5], [0.0, -1.0]])], extra_args=[arg_scalar("rhs", "complex64", real_bits="0x3f000000", imag_bits="0xbf800000")], policy="elementwise_float")
    add(16, "add", "ops.add.explicit_cpu", [source("F32", [2, 3]), source("F32", [3])], policy="elementwise_float", explicit=True)
    add(17, "maximum", "ops.maximum.array_array", [source("F32", [5], [-1.0, 2.0, float("nan"), 4.0, -0.0]), source("F32", [5], [0.0, 1.0, 3.0, float("nan"), 0.0])], policy="elementwise_float")
    add(18, "power", "ops.power.array_array", [source("F32", [4], [1.0, 2.0, 4.0, 9.0]), source("F32", [4], [2.0, 3.0, 0.5, -1.0])], policy="elementwise_float")
    add(19, "add", "ops.add.array_array", [source("F32", [5], [float("nan"), float("inf"), -float("inf"), 1.0, -1.0]), source("F32", [5], [0.0, 1.0, -1.0, float("inf"), -float("inf")])], policy="elementwise_float")
    add(20, "negative", "operator.neg.array", [source("F32", [2], [0.0, -0.0])], policy="exact_bits")
    add(21, "add", "ops.add.array_array", [source("F32", [0], []), source("F32", [0], [])], policy="exact_bits")
    add(22, "add", "array.add.array", [source("U64", [], [9]), source("U64", [], [11])])
    add(23, "subtract", "ops.subtract.array_array", [source("I32", [3, 5]), source("I32", [3, 5])])
    add(24, "multiply", "array.multiply.array", [source("F64", [2, 3]), source("F64", [1, 3])], policy="elementwise_float")

    all_dtypes = ["BOOL", "U8", "U16", "U32", "U64", "I8", "I16", "I32", "I64", "F16", "BF16", "F32", "F64", "C64"]
    for index, dtype in enumerate(all_dtypes, 1):
        values = [[1.0, -0.5], [0.0, 2.0], [-3.0, 1.0]] if dtype == "C64" else None
        policy = "low_precision_float" if dtype in ("F16", "BF16") else "elementwise_float" if dtype in ("F32", "F64", "C64") else "exact_numeric"
        specs.append(case(f"dtypes.{index:03d}", "dtypes", "add_zero", "ops.add.array_array", [source(dtype, [3], values), source(dtype, [], [[0.0, 0.0]] if dtype == "C64" else [0])], policy=policy))
    promotions = [
        (15, "I64", "F16", [1, -2, 300], [0.5, 1.0, -2.0], "F16"),
        (16, "I64", "F32", [1, -2, 300], [0.5, 1.0, -2.0], "F32"),
        (17, "U64", "I64", [1, 2, 3], [-2, 1, 5], "F32"),
        (18, "F16", "BF16", [1.0, -2.0, 0.5], [0.5, 1.0, -2.0], "F32"),
        (19, "I64", "F32", [16777217, -16777217, 3], [0.0, 0.0, 0.25], "F32"),
        (20, "I64", "F32", [16777217, 16777219, 16777221], [1.0, -1.0, 3.0], "F32"),
    ]
    for index, lhs, rhs, lv, rv, result in promotions:
        policy = "low_precision_float" if result in ("F16", "BF16") else "exact_bits"
        specs.append(case(f"dtypes.{index:03d}", "dtypes", "add", "ops.add.array_array", [source(lhs, [3], lv), source(rhs, [3], rv)], policy=policy))
    specs.append(case("dtypes.021", "dtypes", "astype", "array.as_dtype", [source("I32", [3], [1, -2, 3])], extra_args=[dtype_arg("F32")], policy="exact_bits"))

    shape_specs = [
        (1, "reshape", "ops.reshape", [source("I32", [2, 3])], [shape([3, 2])]),
        (2, "reshape", "array.reshape", [source("F32", [6])], [shape([2, 3])]),
        (3, "transpose", "array.transpose", [source("I16", [2, 3])], []),
        (4, "transpose_axes", "ops.transpose.axes", [source("F32", [2, 3, 4])], [axes([2, 0, 1])]),
        (5, "transpose_axes", "ops.transpose.axes", [source("I32", [2, 3])], [axes([-1, -2])]),
        (6, "split", "ops.split.parts", [source("F32", [3, 6])], [arg_scalar("parts", "i32", value=3), axis(1)]),
        (7, "split", "ops.split.parts", [source("I32", [8])], [arg_scalar("parts", "i32", value=4), axis(None)]),
        (8, "split_sections", "ops.split.sections", [source("F32", [2, 7])], [{"name": "indices", "kind": "axes", "values": [1, 5]}, axis(-1)]),
        (9, "broadcast_arrays", "ops.broadcast_arrays", [source("F32", [2, 1]), source("F32", [1, 3]), source("F32", [])], []),
        (10, "reshape", "array.reshape", [source("F32", [])], [shape([1, 1])]),
        (11, "reshape", "ops.reshape", [source("I32", [0, 3], [])], [shape([3, 0])]),
        (12, "transpose", "array.transpose", [source("F64", [2, 5])], []),
        (13, "expand_dims", "ops.expand_dims.axis", [source("U8", [2, 3])], [{"name": "axis", "kind": "axis", "value": -1}]),
        (14, "concatenate", "ops.concatenate", [source("I32", [2, 2]), source("I32", [1, 2])], [{"name": "axis", "kind": "axis", "value": 0}]),
    ]
    for index, op, call, inputs, extra in shape_specs:
        specs.append(case(f"shapes.{index:03d}", "shapes", op, call, inputs, extra_args=extra, policy=data_movement_policy(op, inputs)))

    reduction_specs = [
        (1, "sum", "ops.sum.axis_optional", [source("F32", [2, 3])], [axis(None), keepdims(None)], "reduction_float"),
        (2, "sum", "ops.sum.axis_optional", [source("I32", [2, 3])], [axis(0), keepdims(False)], "exact_numeric"),
        (3, "sum", "ops.sum.axis_optional", [source("F32", [2, 3, 4])], [axis(-1), keepdims(False)], "reduction_float"),
        (4, "sum_axes", "ops.sum.axes", [source("F32", [2, 3, 4])], [axes([0, 2]), keepdims(False)], "reduction_float"),
        (5, "sum_axes", "ops.sum.axes", [source("I64", [2, 3, 4])], [axes([-1, 0]), keepdims(True)], "exact_numeric"),
        (6, "sum", "array.sum.all_optional", [source("F16", [4, 5])], [keepdims(None)], "reduction_float"),
        (7, "sum", "array.sum.all_optional", [source("BF16", [4, 5])], [keepdims(False)], "reduction_float"),
        (8, "sum", "ops.sum.axis_optional", [source("F32", [7, 9], distribution="normal")], [axis(1), keepdims(True)], "reduction_float"),
        (9, "sum", "ops.sum.axis_optional", [source("F32", [0, 3], [])], [axis(0), keepdims(False)], "reduction_float"),
        (10, "sum", "array.sum.all_optional", [source("U32", [2, 3], [1, 2, 3, 4, 5, 6])], [keepdims(True)], "exact_numeric"),
    ]
    for index, op, call, inputs, extra, policy in reduction_specs:
        specs.append(case(f"reductions.{index:03d}", "reductions", op, call, inputs, extra_args=extra, policy=policy))

    execution_specs = [
        (1, "add", "ops.add.array_array", False),
        (2, "add", "ops.add.explicit_cpu", True),
        (3, "exp", "ops.exp.array", False),
        (4, "exp", "ops.exp.explicit_cpu", True),
        (5, "reshape", "array.reshape", False),
        (6, "reshape", "array.reshape.explicit_cpu", True),
        (7, "divmod", "ops.divmod.array_array", False),
        (8, "divmod", "ops.divmod.explicit_cpu", True),
    ]
    for index, op, call, explicit in execution_specs:
        inputs = [source("F32", [2, 3])] if op in ("exp", "reshape") else [source("I32", [6], [8, 9, 10, 11, 12, 13]), source("I32", [6], [3, 4, 3, 5, 5, 2])]
        extra = [shape([3, 2])] if op == "reshape" else []
        policy = "elementwise_float" if op == "exp" else data_movement_policy(op, inputs)
        specs.append(case(f"execution.{index:03d}", "execution", op, call, inputs, extra_args=extra, policy=policy, explicit=explicit))

    errors = [
        case("errors.001", "errors", "add", "ops.add.array_array", [source("F32", [2]), source("F32", [3])]),
        case("errors.002", "errors", "reshape", "ops.reshape", [source("F32", [6])], extra_args=[shape([4, 2])]),
        case("errors.003", "errors", "take", "ops.take_axis.array_indices_axis", [source("I32", [3], [10, 20, 30]), source("I32", [2], [0, 5])], extra_args=[{"name": "axis", "kind": "axis", "value": 2}]),
        case("errors.004", "errors", "negative", "array.negative", [source("BOOL", [2], [True, False])]),
    ]
    errors[0]["error"] = ("invoke_or_eval", "incompatible broadcast shapes", "ValueError", "arithmetic.001")
    errors[1]["error"] = ("invoke_or_eval", "reshape changes element count", "ValueError", "shapes.001")
    errors[2]["error"] = ("invoke_or_eval", "take axis exceeds input rank", "ValueError", "arithmetic.001")
    errors[3]["error"] = ("invoke_or_eval", "boolean negation is unsupported", "ValueError", "arithmetic.007")
    specs.extend(errors)
    return sorted(specs, key=lambda item: item["id"])


def dtype_objects(mx):
    return {
        "BOOL": mx.bool_, "U8": mx.uint8, "U16": mx.uint16, "U32": mx.uint32, "U64": mx.uint64,
        "I8": mx.int8, "I16": mx.int16, "I32": mx.int32, "I64": mx.int64,
        "F16": mx.float16, "BF16": mx.bfloat16, "F32": mx.float32, "F64": mx.float64, "C64": mx.complex64,
    }


def make_values(np, spec, rng):
    count = 1
    for dim in spec["shape"]:
        count *= dim
    if "values" in spec:
        values = spec["values"]
    elif spec["random"] == "normal":
        values = rng.standard_normal(count).tolist()
    else:
        values = rng.integers(-4, 6, size=count).tolist()
    if spec["dtype"] == "BOOL" and "values" not in spec:
        values = [bool(v & 1) for v in values]
    if spec["dtype"].startswith("U") and "values" not in spec:
        values = [abs(v) for v in values]
    if spec["dtype"] == "C64" and "values" not in spec:
        values = [[float(v), float(-v) / 2] for v in values]
    return values


def mlx_array(mx, np, spec, rng):
    values = make_values(np, spec, rng)
    if spec["dtype"] == "C64":
        values = [complex(v[0], v[1]) for v in values]
    return mx.array(values, dtype=dtype_objects(mx)[spec["dtype"]]).reshape(spec["shape"])


def scalar_array(mx, arg):
    scalar_type = arg["type"]
    if scalar_type == "bool":
        return mx.array(arg["value"], dtype=mx.bool_)
    if scalar_type == "i32":
        return mx.array(arg["value"], dtype=mx.int32)
    if scalar_type == "f32":
        value = struct.unpack("<f", struct.pack("<I", int(arg["bits"], 16)))[0]
        return mx.array(value, dtype=mx.float32)
    real = struct.unpack("<f", struct.pack("<I", int(arg["real_bits"], 16)))[0]
    imag = struct.unpack("<f", struct.pack("<I", int(arg["imag_bits"], 16)))[0]
    return mx.array(complex(real, imag), dtype=mx.complex64)


def call_recipe(mx, op, arrays, extra, explicit):
    stream = mx.cpu if explicit else None
    kwargs = {} if stream is None else {"stream": stream}
    by_name = {item["name"]: item for item in extra}
    if op in ("add", "add_zero"):
        return [mx.add(arrays[0], arrays[1], **kwargs)]
    if op == "add_scalar":
        return [mx.add(arrays[0], scalar_array(mx, by_name["rhs"]), **kwargs)]
    if op == "subtract":
        return [mx.subtract(arrays[0], arrays[1], **kwargs)]
    if op == "multiply":
        return [mx.multiply(arrays[0], arrays[1], **kwargs)]
    if op == "divide":
        return [mx.divide(arrays[0], arrays[1], **kwargs)]
    if op == "negative":
        return [mx.negative(arrays[0], **kwargs)]
    if op == "abs":
        return [mx.abs(arrays[0], **kwargs)]
    if op == "exp":
        return [mx.exp(arrays[0], **kwargs)]
    if op == "where":
        return [mx.where(arrays[0], arrays[1], arrays[2], **kwargs)]
    if op == "divmod":
        return list(mx.divmod(arrays[0], arrays[1], **kwargs))
    if op == "maximum":
        return [mx.maximum(arrays[0], arrays[1], **kwargs)]
    if op == "power":
        return [mx.power(arrays[0], arrays[1], **kwargs)]
    if op == "reshape":
        return [mx.reshape(arrays[0], by_name["shape"]["values"], **kwargs)]
    if op == "transpose":
        return [mx.transpose(arrays[0], **kwargs)]
    if op == "transpose_axes":
        return [mx.transpose(arrays[0], by_name["axes"]["values"], **kwargs)]
    if op == "split":
        return list(mx.split(arrays[0], by_name["parts"]["value"], axis=by_name["axis"]["value"] or 0, **kwargs))
    if op == "split_sections":
        return list(mx.split(arrays[0], by_name["indices"]["values"], axis=by_name["axis"]["value"], **kwargs))
    if op == "broadcast_arrays":
        return list(mx.broadcast_arrays(*arrays, **kwargs))
    if op == "expand_dims":
        return [mx.expand_dims(arrays[0], by_name["axis"]["value"], **kwargs)]
    if op == "concatenate":
        return [mx.concatenate(arrays, axis=by_name["axis"]["value"], **kwargs)]
    if op in ("sum", "sum_axes"):
        selected_axis = by_name.get("axis", by_name.get("axes", {"value": None})).get("value", by_name.get("axes", {}).get("values"))
        return [mx.sum(arrays[0], axis=selected_axis, keepdims=by_name["keepdims"]["value"] or False, **kwargs)]
    if op == "take":
        axis_arg = by_name.get("axis")
        axis_value = axis_arg["value"] if axis_arg else None
        return [mx.take(arrays[0], arrays[1], axis=axis_value, **kwargs)]
    if op == "astype":
        return [arrays[0].astype(dtype_objects(mx)[by_name["dtype"]["value"]], **kwargs)]
    raise ValueError(f"unknown recipe {op}")


def dtype_name(array):
    text = str(array.dtype).rsplit(".", 1)[-1]
    return {"bool": "BOOL", "uint8": "U8", "uint16": "U16", "uint32": "U32", "uint64": "U64", "int8": "I8", "int16": "I16", "int32": "I32", "int64": "I64", "float16": "F16", "bfloat16": "BF16", "float32": "F32", "float64": "F64", "complex64": "C64"}[text]


def tensor_ref(fixtures, key, array):
    if dtype_name(array) == "C64":
        real_key, imag_key = f"{key}.real", f"{key}.imag"
        fixtures[real_key] = array.real.astype(array.real.dtype)
        fixtures[imag_key] = array.imag.astype(array.imag.dtype)
        return real_key, "complex64_split", imag_key
    fixtures[key] = array
    return key, None, None


def tensor_bytes(mx, np, array):
    dtype = dtype_name(array)
    if dtype == "BF16":
        values = np.array(mx.view(array, mx.uint16), dtype=np.uint16)
    elif dtype == "F16":
        values = np.array(array, dtype=np.float16)
    elif dtype == "BOOL":
        values = np.array(array, dtype=np.bool_).astype(np.uint8)
    else:
        values = np.array(array)
    return values.tobytes(order="C")


def write_safetensors(path, tensors, mx, np):
    header = {}
    data = []
    offset = 0
    for name, array in sorted(tensors.items()):
        encoded = tensor_bytes(mx, np, array)
        end = offset + len(encoded)
        header[name] = {
            "dtype": dtype_name(array),
            "shape": list(array.shape),
            "data_offsets": [offset, end],
        }
        data.append(encoded)
        offset = end
    encoded_header = json.dumps(header, separators=(",", ":")).encode()
    encoded_header += b" " * (-len(encoded_header) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + b"".join(data))


def numpy_agrees(np, arrays, outputs, op, extra):
    if any(dtype_name(value) == "BF16" for value in arrays + outputs):
        return False
    values = [np.asarray(value) for value in arrays]
    by_name = {item["name"]: item for item in extra}
    try:
        if op in ("add", "add_zero", "add_scalar"):
            rhs = values[1] if len(values) > 1 else np.asarray(scalar_array_value(by_name["rhs"]), dtype=values[0].dtype)
            result = [np.add(values[0], rhs)]
        elif op == "subtract": result = [np.subtract(values[0], values[1])]
        elif op == "multiply": result = [np.multiply(values[0], values[1])]
        elif op == "divide": result = [np.divide(values[0], values[1])]
        elif op == "negative": result = [np.negative(values[0])]
        elif op == "abs": result = [np.abs(values[0])]
        elif op == "exp": result = [np.exp(values[0])]
        elif op == "where": result = [np.where(values[0], values[1], values[2])]
        elif op == "divmod": result = list(np.divmod(values[0], values[1]))
        elif op == "maximum": result = [np.maximum(values[0], values[1])]
        elif op == "power": result = [np.power(values[0], values[1])]
        elif op == "reshape": result = [np.reshape(values[0], by_name["shape"]["values"])]
        elif op == "transpose": result = [np.transpose(values[0])]
        elif op == "transpose_axes": result = [np.transpose(values[0], by_name["axes"]["values"])]
        elif op == "split": result = list(np.split(values[0], by_name["parts"]["value"], axis=by_name["axis"]["value"] or 0))
        elif op == "split_sections": result = list(np.split(values[0], by_name["indices"]["values"], axis=by_name["axis"]["value"]))
        elif op == "broadcast_arrays": result = list(np.broadcast_arrays(*values))
        elif op == "expand_dims": result = [np.expand_dims(values[0], by_name["axis"]["value"])]
        elif op == "concatenate": result = [np.concatenate(values, axis=by_name["axis"]["value"])]
        elif op in ("sum", "sum_axes"):
            selected_axis = by_name.get("axis", by_name.get("axes", {"value": None})).get("value", by_name.get("axes", {}).get("values"))
            result = [np.sum(values[0], axis=selected_axis, keepdims=by_name["keepdims"]["value"] or False)]
        elif op == "take": result = [np.take(values[0], values[1])]
        elif op == "astype": result = [values[0].astype({"F32": np.float32}[by_name["dtype"]["value"]])]
        else: return False
    except Exception:
        return False
    if len(result) != len(outputs):
        return False
    for lhs, rhs in zip(result, outputs):
        expected = np.asarray(rhs)
        got = np.asarray(lhs)
        if got.dtype != expected.dtype or got.shape != expected.shape:
            return False
        if not np.array_equal(got, expected, equal_nan=True):
            return False
    return True


def scalar_array_value(arg):
    if arg["type"] in ("bool", "i32"):
        return arg["value"]
    if arg["type"] == "f32":
        return struct.unpack("<f", struct.pack("<I", int(arg["bits"], 16)))[0]
    real = struct.unpack("<f", struct.pack("<I", int(arg["real_bits"], 16)))[0]
    imag = struct.unpack("<f", struct.pack("<I", int(arg["imag_bits"], 16)))[0]
    return complex(real, imag)


def generate_tree(target, mx, np):
    (target / "suites").mkdir(parents=True)
    (target / "fixtures").mkdir()
    specs = build_specs()
    old_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        with mx.stream(mx.cpu):
            for suite in SUITES:
                fixtures = {}
                cases = []
                for spec in (item for item in specs if item["suite"] == suite):
                    seed_bytes = hashlib.sha256((CORPUS_SEED + "\0" + spec["id"]).encode()).digest()
                    rng = np.random.Generator(np.random.PCG64(int.from_bytes(seed_bytes[:16], "little")))
                    arrays = []
                    args = []
                    for index, input_spec in enumerate(spec["sources"]):
                        array = mlx_array(mx, np, input_spec, rng)
                        key = f"{spec['id']}.input.{index}"
                        ref, encoding, imag_ref = tensor_ref(fixtures, key, array)
                        args.append(arg_tensor(f"input{index}", ref, encoding, imag_ref))
                        arrays.append(array)
                    args.extend(spec["extra_args"])
                    args.append(spec["execution"])
                    record = {"id": spec["id"], "semantic_op": spec["semantic_op"], "recipe": spec["recipe"], "rust_call": spec["rust_call"], "args": args}
                    if "error" in spec:
                        stage, reason, exception_type, control = spec["error"]
                        invoked = False
                        try:
                            outputs = call_recipe(mx, spec["recipe"], arrays, spec["extra_args"], spec["execution"]["target"] == "explicit_cpu")
                            invoked = True
                            for output in outputs:
                                mx.eval(output)
                        except Exception as error:
                            if stage == "eval_only" and not invoked:
                                raise RuntimeError(f"{spec['id']} did not reach evaluation") from error
                            record["expected"] = {"status": "error", "allowed_stage": stage, "reason": reason, "python_exception": {"module": type(error).__module__, "type": type(error).__name__}, "control_case_id": control, "diagnostic": str(error)}
                        else:
                            raise RuntimeError(f"{spec['id']} did not raise")
                    else:
                        outputs = call_recipe(mx, spec["recipe"], arrays, spec["extra_args"], spec["execution"]["target"] == "explicit_cpu")
                        for output in outputs:
                            mx.eval(output)
                        provenance = "numpy_corroborated" if numpy_agrees(np, arrays, outputs, spec["recipe"], spec["extra_args"]) else "mlx_python"
                        expected_outputs = []
                        for index, output in enumerate(outputs):
                            key = f"{spec['id']}.output.{index}"
                            ref, encoding, imag_ref = tensor_ref(fixtures, key, output)
                            item = {"name": f"output{index}", "ref": ref, "dtype": dtype_name(output), "shape": list(output.shape), "policy": spec["policy"]}
                            if encoding:
                                item["encoding"] = encoding
                                item["imag_ref"] = imag_ref
                            expected_outputs.append(item)
                        record["expected"] = {"status": "success", "provenance": provenance, "outputs": expected_outputs}
                    cases.append(record)
                fixture_name = f"{suite}.safetensors"
                write_safetensors(target / "fixtures" / fixture_name, fixtures, mx, np)
                suite_doc = {"schema_version": 1, "name": suite, "fixture": f"fixtures/{fixture_name}", "cases": cases}
                (target / "suites" / f"{suite}.json").write_text(json.dumps(suite_doc, indent=2, sort_keys=False, allow_nan=False) + "\n")
    finally:
        mx.set_default_device(old_device)

    generator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    policies = {
        "exact_bits": {"kind": "float", "atol": 0.0, "rtol": 0.0, "nan_equal": True, "infinity_sign": True, "signed_zero": True, "complex": "componentwise"},
        "exact_numeric": {"kind": "exact_numeric"},
        "elementwise_float": {"kind": "float", "atol": 1e-6, "rtol": 1e-5, "nan_equal": True, "infinity_sign": True, "signed_zero": False, "complex": "componentwise"},
        "low_precision_float": {"kind": "float", "atol": 0.008, "rtol": 0.008, "nan_equal": True, "infinity_sign": True, "signed_zero": False, "complex": "componentwise"},
        "reduction_float": {"kind": "float", "atol": 2e-5, "rtol": 2e-5, "nan_equal": True, "infinity_sign": True, "signed_zero": False, "complex": "componentwise"},
    }
    corpus = {
        "schema_version": 1,
        "corpus_seed": CORPUS_SEED,
        "rng": {"algorithm": "numpy.PCG64", "case_seed_hash": "sha256(corpus_seed || NUL || case_id)", "seed_bytes": 16, "byte_order": "little"},
        "canonical_device": "cpu",
        "generator_digest": f"sha256:{generator_digest}",
        "environment": {"python": "3.12.14", "architecture": EXPECTED_ARCH, "mlx_package": EXPECTED_MLX, "mlx_metal_package": EXPECTED_MLX, "mlx_runtime": EXPECTED_MLX, "numpy": EXPECTED_NUMPY},
        "tolerance_policies": policies,
        "suites": [f"suites/{name}.json" for name in SUITES],
    }
    (target / "corpus.json").write_text(json.dumps(corpus, indent=2) + "\n")
    qualification = {
        "schema_version": 1,
        "mutations": [
            {"id": "dtype_changed_values_equal", "base_case_id": "arithmetic.001", "kind": "dtype_changed_values_equal", "expected_class": "dtype"},
            {"id": "shape_changed_same_count", "base_case_id": "arithmetic.001", "kind": "shape_changed_same_count", "expected_class": "shape"},
            {"id": "output_removed", "base_case_id": "arithmetic.011", "kind": "output_removed", "expected_class": "output_count"},
            {"id": "output_added", "base_case_id": "arithmetic.001", "kind": "output_added", "expected_class": "output_count"},
            {"id": "output_reordered", "base_case_id": "arithmetic.011", "kind": "output_reordered", "expected_class": "output_order"},
            {"id": "beyond_absolute", "base_case_id": "arithmetic.001", "kind": "beyond_absolute", "expected_class": "value_absolute"},
            {"id": "beyond_relative", "base_case_id": "arithmetic.001", "kind": "beyond_relative", "expected_class": "value_relative"},
            {"id": "nan_vs_finite", "base_case_id": "arithmetic.019", "kind": "nan_vs_finite", "expected_class": "nan"},
            {"id": "positive_inf_vs_negative_inf", "base_case_id": "arithmetic.019", "kind": "positive_inf_vs_negative_inf", "expected_class": "infinity_sign"},
            {"id": "swapped_subtraction", "base_case_id": "arithmetic.004", "kind": "swapped_subtraction", "expected_class": "value"},
            {"id": "wrong_axis", "base_case_id": "reductions.003", "kind": "wrong_axis", "expected_class": "shape"},
            {"id": "error_to_valid", "base_case_id": "errors.001", "kind": "error_to_valid", "expected_class": "expected_error"},
            {"id": "f16_decoder", "base_case_id": "dtypes.010", "kind": "f16_decoder", "expected_class": "decoder_f16"},
            {"id": "bf16_decoder", "base_case_id": "dtypes.011", "kind": "bf16_decoder", "expected_class": "decoder_bf16"},
            {"id": "empty_tensor", "base_case_id": "arithmetic.021", "kind": "empty_tensor", "expected_class": "empty"},
            {"id": "endianness", "base_case_id": "dtypes.008", "kind": "endianness", "expected_class": "endianness"},
        ],
    }
    (target / "qualification.json").write_text(json.dumps(qualification, indent=2) + "\n")


def tree_hash(path):
    digest = hashlib.sha256()
    for item in sorted(entry for entry in path.rglob("*") if entry.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


def main():
    mx, np = check_environment()
    first = Path(tempfile.mkdtemp(prefix="mlx-rs-conformance-a-"))
    second = Path(tempfile.mkdtemp(prefix="mlx-rs-conformance-b-"))
    try:
        generate_tree(first, mx, np)
        generate_tree(second, mx, np)
        first_hash = tree_hash(first)
        second_hash = tree_hash(second)
        if first_hash != second_hash:
            raise SystemExit(f"generation is not reproducible: {first_hash} != {second_hash}")
        for name in ("corpus.json", "qualification.json", "suites", "fixtures"):
            destination = ROOT / name
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
            os.replace(first / name, destination)
        print(first_hash)
    finally:
        shutil.rmtree(first, ignore_errors=True)
        shutil.rmtree(second, ignore_errors=True)


if __name__ == "__main__":
    main()
