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
EXPECTED_MLX = "0.32.2"
EXPECTED_NUMPY = "2.2.6"
CORPUS_SEED = "mlx-rs-committed-cpu-ops-v1"
OP_SUITES = ("arithmetic", "dtypes", "errors", "execution", "fast", "indexing", "math", "reductions", "shapes", "signal")
SUITES = OP_SUITES + ("gguf",)
ROOT = Path(__file__).resolve().parent

GGUF_VALUE_TYPES = {
    "U8": 0,
    "I8": 1,
    "U16": 2,
    "I16": 3,
    "U32": 4,
    "I32": 5,
    "F32": 6,
    "BOOL": 7,
    "STRING": 8,
    "ARRAY": 9,
    "U64": 10,
    "I64": 11,
    "F64": 12,
}
GGUF_TENSOR_TYPES = {"F32": 0, "F16": 1, "Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q8_0": 8}
GGUF_PACK_FORMATS = {
    "U8": "<B",
    "I8": "<b",
    "U16": "<H",
    "I16": "<h",
    "U32": "<I",
    "I32": "<i",
    "F32": "<f",
    "BOOL": "<?",
    "U64": "<Q",
    "I64": "<q",
    "F64": "<d",
}


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


def execution(explicit=False, target=None):
    return {"name": "execution", "kind": "execution", "target": target or ("explicit_cpu" if explicit else "default_cpu")}


def source(dtype, shape, values=None, distribution="small_integers"):
    item = {"dtype": dtype, "shape": shape}
    if values is None:
        item["random"] = distribution
    else:
        item["values"] = values
    return item


def source_f32_bits(shape, bits):
    return {"dtype": "F32", "shape": shape, "bits": [f"0x{value:08x}" for value in bits]}


def case(case_id, suite, semantic_op, rust_call, inputs, extra_args=None, policy="exact_numeric", explicit=False, execution_target=None):
    return {
        "id": case_id,
        "suite": suite,
        "semantic_op": semantic_op,
        "recipe": semantic_op,
        "rust_call": rust_call,
        "sources": inputs,
        "extra_args": extra_args or [],
        "policy": policy,
        "execution": execution(explicit, execution_target),
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


def index_recipe(value):
    return {"name": "index", "kind": "index", "value": value}


def update_mode(value):
    return {"name": "mode", "kind": "update_mode", "value": value}


def f64_arg(name, value):
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    return arg_scalar(name, "f64", bits=f"0x{bits:016x}")


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

    payload = [0x80000000, 0x7FC01234, 0x3F800000, 0xC0200000, 0x00000000, 0x40A00000]
    specs.extend([
        case(
            "shapes.015", "shapes", "contiguous", "array.contiguous.transpose",
            [source_f32_bits([2, 3], payload)],
            extra_args=[arg_scalar("view", "i32", value=0), arg_scalar("allow_col_major", "bool", value=False)],
            policy="exact_bits",
        ),
        case(
            "shapes.016", "shapes", "contiguous", "array.contiguous.slice",
            [source_f32_bits([8], [0x3F000000, 0x80000000, 0x40000000, 0x7FC01234, 0x40400000, 0x3F800000, 0x40800000, 0x00000000])],
            extra_args=[arg_scalar("view", "i32", value=1), arg_scalar("allow_col_major", "bool", value=False)],
            policy="exact_bits",
        ),
        case(
            "shapes.017", "shapes", "contiguous", "array.contiguous.broadcast",
            [source_f32_bits([1, 3], [0x80000000, 0x7FC01234, 0x3F800000])],
            extra_args=[arg_scalar("view", "i32", value=2), shape([2, 3]), arg_scalar("allow_col_major", "bool", value=False)],
            policy="exact_bits",
        ),
        case(
            "shapes.018", "shapes", "contiguous", "array.contiguous.options.explicit_cpu",
            [source_f32_bits([2, 3], payload)],
            extra_args=[arg_scalar("view", "i32", value=0), arg_scalar("allow_col_major", "bool", value=True)],
            policy="exact_bits", explicit=True,
        ),
    ])

    fast_specs = [
        case(
            "fast.001", "fast", "rms_norm", "fast.rms_norm.none",
            [source("F32", [2, 4], [1.0] * 8)],
            extra_args=[arg_scalar("has_weight", "bool", value=False), arg_scalar("eps", "f32", bits="0x00000000")],
            policy="reduction_float",
        ),
        case(
            "fast.002", "fast", "rms_norm", "fast.rms_norm.weighted",
            [source("F32", [2, 3], [1.0, 2.0, 3.0, -4.0, 5.0, -6.0]), source("F32", [3], [0.5, 2.0, -1.0])],
            extra_args=[arg_scalar("has_weight", "bool", value=True), arg_scalar("eps", "f32", bits="0x3e800000")],
            policy="reduction_float",
        ),
        case(
            "fast.003", "fast", "rms_norm", "fast.rms_norm.weighted",
            [source("F32", [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), source("F32", [2], [1.0, 2.0])],
            extra_args=[arg_scalar("has_weight", "bool", value=True), arg_scalar("eps", "f32", bits="0x3727c5ac")],
        ),
        case(
            "fast.004", "fast", "rms_norm", "fast.rms_norm.weighted",
            [source("F32", [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), source("F32", [1, 3], [1.0, 2.0, 3.0])],
            extra_args=[arg_scalar("has_weight", "bool", value=True), arg_scalar("eps", "f32", bits="0x3727c5ac")],
        ),
        case(
            "fast.005", "fast", "rms_norm", "fast.rms_norm.explicit_cpu",
            [source("F32", [1, 4], [1.0, -2.0, 3.0, -4.0])],
            extra_args=[arg_scalar("has_weight", "bool", value=False), arg_scalar("eps", "f32", bits="0x3a83126f")],
            policy="reduction_float", explicit=True,
        ),
    ]
    for error_case in fast_specs[2:4]:
        error_case["error"] = ("invoke_only", "RMSNorm weight shape is invalid", "ValueError", "fast.002")
        error_case["rust_diagnostic"] = "[rms_norm]"
    specs.extend(fast_specs)

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

    signal_index = 1
    for op in ("bartlett", "blackman", "hamming", "hann"):
        for size in (1, 2, 7):
            specs.append(case(
                f"signal.{signal_index:03d}",
                "signal",
                op,
                f"ops.windows.{op}",
                [],
                extra_args=[arg_scalar("size", "i32", value=size)],
                policy="exact_bits",
            ))
            signal_index += 1
    for op in ("fftfreq", "rfftfreq"):
        for n in (5, 6):
            for d_bits in ("0x3f800000", "0x3f000000"):
                specs.append(case(
                    f"signal.{signal_index:03d}",
                    "signal",
                    op,
                    f"fft.{op}",
                    [],
                    extra_args=[
                        arg_scalar("n", "i32", value=n),
                        arg_scalar("d", "f32", bits=d_bits),
                    ],
                    policy="exact_bits",
                ))
                signal_index += 1

    math_specs = []
    math_add = lambda op, call, ins, extra=None, policy="exact_numeric", explicit=False: math_specs.append(
        (op, call, ins, extra or [], policy, explicit)
    )
    keep_false = keepdims(False)
    keep_true = keepdims(True)
    count_values = [0, 1, -2, 0, 3, 0]
    math_add("count_nonzero", "array.count_nonzero.all", [source("I32", [2, 3], count_values)], [keep_false])
    math_add("count_nonzero", "array.count_nonzero.all", [source("C64", [3], [[0, 0], [0, 2], [float("nan"), 0]])], [keep_true])
    math_add("count_nonzero", "array.count_nonzero.axis", [source("BOOL", [2, 3], [True, False, True, False, False, True])], [{"name": "axis", "kind": "axis", "value": -1}, keep_false])
    math_add("count_nonzero", "array.count_nonzero.axes", [source("F32", [2, 3, 2], [0, 1, 2, 0, 0, 3, 4, 5, 0, 0, float("nan"), 0])], [axes([0, -1]), keep_true])
    math_add("count_nonzero", "array.count_nonzero.axes", [source("I16", [2, 2], [0, 1, 2, 0])], [axes([]), keep_false])
    math_add("count_nonzero", "array.count_nonzero.axis", [source("F32", [0, 3], [])], [{"name": "axis", "kind": "axis", "value": 0}, keep_false])
    math_add("count_nonzero", "array.count_nonzero.explicit_cpu", [source("U8", [4], [0, 1, 0, 2])], [keep_false], explicit=True)

    for n, selected_axis in [(0, -1), (1, 0), (2, -1), (5, -1)]:
        math_add("diff", "array.diff", [source("I32" if n != 2 else "F32", [2, 4], list(range(8)))], [arg_scalar("n", "i32", value=n), {"name": "axis", "kind": "axis", "value": selected_axis}], "exact_numeric" if n != 2 else "exact_bits")
    math_add("diff", "array.diff", [source("C64", [4], [[1, 2], [3, -1], [0, 4], [-2, 1]])], [arg_scalar("n", "i32", value=1), {"name": "axis", "kind": "axis", "value": -1}], "exact_bits")

    math_add("flip", "array.flip.all", [source("I32", [2, 3], [1, 2, 3, 4, 5, 6])], policy="exact_bits")
    math_add("flip", "array.flip.axis", [source("F32", [2, 3], list(range(6)))], [{"name": "axis", "kind": "axis", "value": -1}], "exact_bits")
    math_add("flip", "array.flip.axes", [source("C64", [2, 2, 2], [[v, -v] for v in range(8)])], [axes([0, -1])], "exact_bits")
    math_add("flip", "array.flip.axes", [source("I16", [2, 3], list(range(6)))], [axes([])], "exact_bits")
    math_add("flip", "array.flip.axes", [source("I32", [2, 3], list(range(6)))], [axes([1, 1])], "exact_bits")
    math_add("flip", "array.flip.all", [source("F32", [0, 3], [])], policy="exact_bits")

    for dtype, shape_, values in [
        ("F32", [1, 1], [4]),
        ("F32", [2, 2], [1, 2, 3, 4]),
        ("F64", [3, 3], [2, 1, 0, 1, 3, 1, 0, 1, 2]),
        ("F32", [4, 4], [4, 1, 2, 0, 0, 3, -1, 1, 2, 0, 5, 2, 1, 1, 0, 2]),
        ("I32", [2, 2], [2, 1, 1, 3]),
        ("F32", [2, 2, 2], [1, 2, 3, 4, 2, 0, 0, 5]),
        ("F32", [2, 2], [1, 2, 2, 4]),
        ("F32", [0, 0], []),
    ]:
        math_add("det", "linalg.det", [source(dtype, shape_, values)], policy="reduction_float")

    for shape_, values in [
        ([2, 2], [2, 0, 0, 3]),
        ([2, 2], [-2, 0, 0, 3]),
        ([2, 2], [1, 2, 2, 4]),
        ([3, 2, 2], [2, 0, 0, 3, -2, 0, 0, 4, 1, 2, 2, 4]),
        ([4, 4], [4, 1, 2, 0, 0, 3, -1, 1, 2, 0, 5, 2, 1, 1, 0, 2]),
        ([0, 0], []),
        ([2, 2], [1e30, 0, 0, 1e30]),
    ]:
        math_add("slogdet", "linalg.slogdet", [source("F32", shape_, values)], policy="reduction_float")

    linspace_cases = [
        (0.0, 1.0, 50, True, "F32", "ops.linspace.f32"),
        (0.0, 1.0, 5, True, "F32", "ops.linspace.f32"),
        (0.0, 1.0, 5, False, "F32", "ops.linspace.f32"),
        (1.0, -1.0, 4, True, "F32", "ops.linspace.f32"),
        (2.0, 2.0, 3, False, "F32", "ops.linspace.f32"),
        (1.0, 4.0, 0, True, "F32", "ops.linspace.f32"),
        (1.0, 4.0, 1, False, "F32", "ops.linspace.f32"),
        (16777217.0, 16777219.0, 3, True, "F64", "ops.linspace.f64"),
        (-3.0, 4.0, 5, True, "I32", "ops.linspace.i32"),
    ]
    for start, stop, count, endpoint, dtype, rust_call in linspace_cases:
        math_add("linspace", rust_call, [], [f64_arg("start", start), f64_arg("stop", stop), arg_scalar("count", "i32", value=count), arg_scalar("endpoint", "bool", value=endpoint), dtype_arg(dtype)], "exact_bits")

    scan_input = [1000.0, 1001.0, -float("inf"), float("inf")]
    for reverse in (False, True):
        for inclusive in (False, True):
            math_add("logcumsumexp", "array.logcumsumexp", [source("F32", [2, 2], scan_input)], [axis(None), arg_scalar("reverse", "bool", value=reverse), arg_scalar("inclusive", "bool", value=inclusive)], "elementwise_float")
    math_add("logcumsumexp", "array.logcumsumexp", [source("F16", [2, 3], [-2, -1, 0, 1, 2, 3])], [axis(-1), arg_scalar("reverse", "bool", value=False), arg_scalar("inclusive", "bool", value=True)], "low_precision_float")
    math_add("logcumsumexp", "array.logcumsumexp", [source("BF16", [2, 3], [-2, -1, 0, 1, 2, 3])], [axis(0), arg_scalar("reverse", "bool", value=True), arg_scalar("inclusive", "bool", value=False)], "low_precision_float")
    math_add("logcumsumexp", "array.logcumsumexp", [source("C64", [3], [[1, 2], [3, -1], [0, 4]])], [axis(-1), arg_scalar("reverse", "bool", value=False), arg_scalar("inclusive", "bool", value=True)], "elementwise_float")
    math_add("logcumsumexp", "array.logcumsumexp", [source("F32", [0], [])], [axis(None), arg_scalar("reverse", "bool", value=False), arg_scalar("inclusive", "bool", value=True)], "exact_bits")

    math_add("logical_xor", "ops.logical_xor", [source("BOOL", [4], [False, False, True, True]), source("BOOL", [4], [False, True, False, True])])
    math_add("logical_xor", "ops.logical_xor", [source("I32", [2, 1], [0, 2]), source("F32", [1, 3], [0, -1, float("nan")])])
    math_add("logical_xor", "ops.logical_xor", [source("BOOL", [0, 3], []), source("BOOL", [1, 3], [True, False, True])])

    search_sequence = source("I32", [4], [1, 2, 2, 3])
    for side_name in ("left", "right"):
        math_add("search_sorted", f"array.search_sorted.{side_name}", [search_sequence, source("I32", [3], [0, 2, 4])], [arg_scalar("right", "bool", value=side_name == "right")])
    math_add("search_sorted", "array.search_sorted.left", [source("F32", [4], [1, 2, 3, 4]), source("F32", [], [2.5])], [arg_scalar("right", "bool", value=False)])
    math_add("search_sorted", "array.search_sorted.right", [source("F32", [4], [1, 2, 3, 4]), source("I16", [2, 2], [0, 2, 3, 5])], [arg_scalar("right", "bool", value=True)])
    math_add("search_sorted", "array.search_sorted.left", [source("F32", [0], []), source("F32", [2], [1, float("nan")])], [arg_scalar("right", "bool", value=False)])
    math_add("search_sorted", "array.search_sorted.left", [source("F32", [4], [1, 2, float("nan"), float("nan")]), source("F32", [2], [float("nan"), 2])], [arg_scalar("right", "bool", value=False)])
    math_add("search_sorted", "array.search_sorted.left", [source("I32", [4], [2, 1, 4, 3]), source("I32", [2], [2, 3])], [arg_scalar("right", "bool", value=False)])

    math_add("trace", "array.trace.default", [source("F32", [3, 3], list(range(9)))], policy="reduction_float")
    for offset in (-2, 1):
        math_add("trace", "array.trace.options", [source("F32", [2, 3, 4], list(range(24)))], [arg_scalar("offset", "i32", value=offset), {"name": "axis1", "kind": "axis", "value": -2}, {"name": "axis2", "kind": "axis", "value": -1}], "reduction_float")
    math_add("trace", "array.trace.options", [source("I32", [2, 3, 4], list(range(24)))], [arg_scalar("offset", "i32", value=0), {"name": "axis1", "kind": "axis", "value": 0}, {"name": "axis2", "kind": "axis", "value": 2}])
    math_add("trace", "array.trace.options", [source("F32", [2, 3], list(range(6)))], [arg_scalar("offset", "i32", value=5), {"name": "axis1", "kind": "axis", "value": 0}, {"name": "axis2", "kind": "axis", "value": 1}], "reduction_float")
    math_add("trace", "array.trace.dtype", [source("F32", [3, 3], list(range(9)))], [arg_scalar("offset", "i32", value=0), {"name": "axis1", "kind": "axis", "value": 0}, {"name": "axis2", "kind": "axis", "value": 1}, dtype_arg("I32")])

    math_add("trunc", "array.trunc", [source("F32", [6], [-2.75, -0.0, 0.0, 1.25, 2.99, -3.01])], policy="exact_bits")
    math_add("trunc", "array.trunc", [source("F16", [3], [-1.5, 0.0, 2.5])], policy="exact_bits")
    math_add("trunc", "array.trunc", [source("F64", [3], [-1.5, 0.0, 2.5])], policy="exact_bits")
    math_add("trunc", "array.trunc", [source("I32", [3], [-1, 0, 2])])

    math_add("unstack", "array.unstack", [source("I32", [2, 3], [1, 2, 3, 4, 5, 6])], [{"name": "axis", "kind": "axis", "value": 0}])
    math_add("unstack", "array.unstack", [source("F32", [2, 3, 2], list(range(12)))], [{"name": "axis", "kind": "axis", "value": 1}], "exact_bits")
    math_add("unstack", "array.unstack", [source("F32", [2, 3, 2], list(range(12)))], [{"name": "axis", "kind": "axis", "value": -1}], "exact_bits")
    math_add("unstack", "array.unstack", [source("I16", [2, 0, 3], [])], [{"name": "axis", "kind": "axis", "value": 1}])

    math_add("vecdot", "ops.vecdot", [source("I32", [2, 3], [1, 2, 3, 4, 5, 6]), source("I32", [2, 3], [6, 5, 4, 3, 2, 1])], [{"name": "axis", "kind": "axis", "value": -1}])
    math_add("vecdot", "ops.vecdot", [source("F32", [2, 3, 4], list(range(24))), source("F32", [2, 1, 4], list(range(8)))], [{"name": "axis", "kind": "axis", "value": 0}], "reduction_float")
    math_add("vecdot", "ops.vecdot", [source("C64", [2], [[1, 1], [2, -1]]), source("C64", [2], [[3, -1], [1, 2]])], [{"name": "axis", "kind": "axis", "value": -1}], "elementwise_float")
    for index, (op, call, inputs, extra, policy, explicit) in enumerate(math_specs, 1):
        specs.append(case(f"math.{index:03d}", "math", op, call, inputs, extra_args=extra, policy=policy, explicit=explicit))
    specs.append(case(
        "math.078", "math", "search_sorted", "array.search_sorted.explicit_gpu",
        [search_sequence, source("I32", [3], [0, 2, 4])],
        extra_args=[arg_scalar("right", "bool", value=False)],
        execution_target="explicit_gpu",
    ))

    indexing_specs = []
    index_add = lambda call, src, update, index, mode, indices=None: indexing_specs.append(
        (call, [src, update] + ([] if indices is None else [indices]), [index_recipe(index), update_mode(mode)])
    )
    base = source("I32", [6], [1, 2, 3, 4, 5, 6])
    update = source("I32", [3], [10, 2, -1])
    for mode in ("replace", "add", "min", "max", "product"):
        index_add("array.try_index_update", base, update, "positive_slice", mode)
    reverse_base = source("I32", [6], [0, 1, 2, 3, 4, 5])
    reverse_update = source("I32", [3], [10, 20, 30])
    for mode in ("replace", "add", "min", "max", "product"):
        index_add("array.try_index_update", reverse_base, reverse_update, "negative_stride", mode)
    advanced_indices = source("I32", [3], [0, 2, 4])
    for mode in ("replace", "add", "min", "max", "product"):
        index_add(
            "array.try_index_update",
            source("I32", [5], [2, 4, 6, 8, 10]),
            source("I32", [3], [3, 5, 7]),
            "advanced",
            mode,
            advanced_indices,
        )
    duplicate_indices = source("I32", [3], [1, 1, 1])
    for mode in ("add", "min", "max", "product"):
        index_add(
            "array.try_index_update",
            source("I32", [3], [10, 10, 10]),
            source("I32", [3], [2, 3, 4]),
            "duplicate_advanced",
            mode,
            duplicate_indices,
        )
    index_add("array.try_index_update", source("I32", [3, 3], list(range(9))), source("I32", [], [50]), "tuple_2d", "replace")
    index_add("array.try_index_update", source("I32", [2, 3], list(range(6))), source("I32", [3], [10, 20, 30]), "negative_index", "add")
    index_add("array.try_index_update", source("I32", [2, 3], list(range(6))), source("I32", [], [9]), "ellipsis_new_axis", "replace")
    index_add("array.try_index_update", base, source("I32", [], [2]), "positive_slice", "add")
    index_add("array.try_index_update", source("I32", [2, 3], list(range(6))), source("I32", [2, 1], [40, 50]), "tuple_columns", "replace")
    index_add("array.try_index_update", base, source("F32", [3], [1.75, -2.25, 3.5]), "positive_slice", "replace")
    index_add("array.try_index_update", base, source("I32", [], [3]), "full", "product")
    index_add("array.try_index_update", base, source("I32", [0], []), "empty", "min")
    index_add("array.try_index_update", base, source("I32", [], [1]), "clipped", "add")
    index_add("array.try_index_update", base, source("I32", [], [7]), "noop", "product")
    index_add("array.try_index_update", base, source("I32", [3], [8, 9, 10]), "negative_bounds", "replace")
    index_add(
        "array.try_index_update",
        source("I32", [3, 3], list(range(9))),
        source("I32", [2], [20, 30]),
        "advanced_tuple",
        "max",
        source("I32", [2], [0, 2]),
    )
    index_add("array.try_index_update.source_unchanged", base, update, "positive_slice", "add")
    indexing_specs[-1][2].append(arg_scalar("return_source", "bool", value=True))
    index_add("array.try_index_mut.compatibility", base, update, "positive_slice", "replace")
    index_add("array.try_index_update", base, update, "zero_stride", "replace")
    index_add("array.try_index_update", base, source("I32", [2], [1, 2]), "positive_slice", "add")
    for index, (call, inputs, extra) in enumerate(indexing_specs, 1):
        spec = case(f"indexing.{index:03d}", "indexing", "index_update", call, inputs, extra_args=extra)
        if index == len(indexing_specs) - 1:
            spec["error"] = ("invoke_or_eval", "zero stride is rejected before FFI", "ZeroStride", "indexing.001")
        elif index == len(indexing_specs):
            spec["error"] = ("invoke_or_eval", "update cannot broadcast to selected slice", "ValueError", "indexing.002")
        specs.append(spec)

    errors = [
        case("errors.001", "errors", "add", "ops.add.array_array", [source("F32", [2]), source("F32", [3])]),
        case("errors.002", "errors", "reshape", "ops.reshape", [source("F32", [6])], extra_args=[shape([4, 2])]),
        case("errors.003", "errors", "take", "ops.take_axis.array_indices_axis", [source("I32", [3], [10, 20, 30]), source("I32", [2], [0, 5])], extra_args=[{"name": "axis", "kind": "axis", "value": 2}]),
        case("errors.004", "errors", "negative", "array.negative", [source("BOOL", [2], [True, False])]),
        case("errors.005", "errors", "diff", "array.diff", [source("F32", [3], [1, 2, 3])], extra_args=[arg_scalar("n", "i32", value=-1), {"name": "axis", "kind": "axis", "value": 0}]),
        case("errors.006", "errors", "diff", "array.diff", [source("F32", [3], [1, 2, 3])], extra_args=[arg_scalar("n", "i32", value=1), {"name": "axis", "kind": "axis", "value": 2}]),
        case("errors.007", "errors", "det", "linalg.det", [source("F32", [2, 3], [1, 2, 3, 4, 5, 6])]),
        case("errors.008", "errors", "det", "linalg.det", [source("C64", [2, 2], [[1, 0], [0, 0], [0, 0], [1, 0]])]),
        case("errors.009", "errors", "linspace", "ops.linspace.f32", [], extra_args=[f64_arg("start", 0.0), f64_arg("stop", 1.0), arg_scalar("count", "i32", value=-1), arg_scalar("endpoint", "bool", value=True), dtype_arg("F32")]),
        case("errors.010", "errors", "search_sorted", "array.search_sorted.left", [source("I32", [2, 2], [1, 2, 3, 4]), source("I32", [1], [2])], extra_args=[arg_scalar("right", "bool", value=False)]),
        case("errors.011", "errors", "trace", "array.trace.options", [source("F32", [2, 2], [1, 2, 3, 4])], extra_args=[arg_scalar("offset", "i32", value=0), {"name": "axis1", "kind": "axis", "value": 0}, {"name": "axis2", "kind": "axis", "value": 3}]),
        case("errors.012", "errors", "trunc", "array.trunc", [source("C64", [1], [[1, 2]])]),
        case("errors.013", "errors", "unstack", "array.unstack", [source("F32", [2, 2], [1, 2, 3, 4])], extra_args=[{"name": "axis", "kind": "axis", "value": 3}]),
        case("errors.014", "errors", "unstack", "array.unstack", [source("F32", [], [1])], extra_args=[{"name": "axis", "kind": "axis", "value": 0}]),
        case("errors.015", "errors", "vecdot", "ops.vecdot", [source("F32", [], [1]), source("F32", [], [2])], extra_args=[{"name": "axis", "kind": "axis", "value": -1}]),
        case("errors.016", "errors", "vecdot", "ops.vecdot", [source("F32", [2], [1, 2]), source("F32", [3], [1, 2, 3])], extra_args=[{"name": "axis", "kind": "axis", "value": -1}]),
        case("errors.017", "errors", "vecdot", "ops.vecdot", [source("F32", [2, 2], [1, 2, 3, 4]), source("F32", [2], [1, 2])], extra_args=[{"name": "axis", "kind": "axis", "value": -1}]),
        case("errors.018", "errors", "det", "linalg.det", [source("F32", [2], [1, 2])]),
        case("errors.019", "errors", "det_gpu", "linalg.det.gpu", [source("F32", [2, 2], [1, 0, 0, 1])]),
        case("errors.020", "errors", "trace", "array.trace.default", [source("F32", [], [1])]),
        case("errors.021", "errors", "vecdot", "ops.vecdot", [source("F32", [2, 2], [1, 2, 3, 4]), source("F32", [2, 2], [4, 3, 2, 1])], extra_args=[{"name": "axis", "kind": "axis", "value": 2}]),
    ]
    errors[0]["error"] = ("invoke_or_eval", "incompatible broadcast shapes", "ValueError", "arithmetic.001")
    errors[1]["error"] = ("invoke_or_eval", "reshape changes element count", "ValueError", "shapes.001")
    errors[2]["error"] = ("invoke_or_eval", "take axis exceeds input rank", "ValueError", "arithmetic.001")
    errors[3]["error"] = ("invoke_or_eval", "boolean negation is unsupported", "ValueError", "arithmetic.007")
    error_reasons = [
        ("negative diff order", "ValueError", "math.008"),
        ("diff axis exceeds input rank", "ValueError", "math.009"),
        ("det requires square matrices", "ValueError", "math.019"),
        ("complex determinant is unsupported", "ValueError", "math.020"),
        ("negative linspace count", "ValueError", "math.034"),
        ("search sequence must be one-dimensional", "ValueError", "math.054"),
        ("trace axis exceeds input rank", "ValueError", "math.061"),
        ("complex truncation is unsupported", "ValueError", "math.067"),
        ("unstack axis exceeds input rank", "ValueError", "math.071"),
        ("unstack requires an array axis", "ValueError", "math.071"),
        ("vecdot requires a non-scalar axis", "ValueError", "math.075"),
        ("vecdot reduction extents must match", "ValueError", "math.075"),
        ("vecdot unequal-rank axis behavior is deferred", "ValueError", "math.075"),
        ("det requires rank at least two", "ValueError", "math.019"),
        ("det is unsupported on GPU", "ValueError", "math.019"),
        ("trace requires rank at least two", "ValueError", "math.061"),
        ("vecdot axis exceeds input rank", "ValueError", "math.075"),
    ]
    for error_case, (reason, exception_type, control) in zip(errors[4:], error_reasons):
        error_case["error"] = ("invoke_or_eval", reason, exception_type, control)
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
    if "bits" in spec:
        values = np.array([int(value, 16) for value in spec["bits"]], dtype=np.uint32).view(np.float32)
        return mx.array(values, dtype=mx.float32).reshape(spec["shape"])
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
    if scalar_type == "f64":
        value = struct.unpack("<d", struct.pack("<Q", int(arg["bits"], 16)))[0]
        return mx.array(value, dtype=mx.float64)
    real = struct.unpack("<f", struct.pack("<I", int(arg["real_bits"], 16)))[0]
    imag = struct.unpack("<f", struct.pack("<I", int(arg["imag_bits"], 16)))[0]
    return mx.array(complex(real, imag), dtype=mx.complex64)


def call_recipe(mx, op, arrays, extra, execution_target):
    stream = {"explicit_cpu": mx.cpu, "explicit_gpu": mx.gpu}.get(execution_target)
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
    if op == "contiguous":
        view = by_name["view"]["value"]
        if view == 0:
            value = arrays[0].transpose()
        elif view == 1:
            value = arrays[0][1:7:2]
        elif view == 2:
            value = mx.broadcast_to(arrays[0], by_name["shape"]["values"])
        else:
            raise ValueError(f"unknown contiguous view {view}")
        return [mx.contiguous(value, allow_col_major=by_name["allow_col_major"]["value"], **kwargs)]
    if op == "rms_norm":
        weight = arrays[1] if by_name["has_weight"]["value"] else None
        return [mx.fast.rms_norm(arrays[0], weight, scalar_array_value(by_name["eps"]), **kwargs)]
    if op in ("sum", "sum_axes"):
        selected_axis = by_name.get("axis", by_name.get("axes", {"value": None})).get("value", by_name.get("axes", {}).get("values"))
        return [mx.sum(arrays[0], axis=selected_axis, keepdims=by_name["keepdims"]["value"] or False, **kwargs)]
    if op == "take":
        axis_arg = by_name.get("axis")
        axis_value = axis_arg["value"] if axis_arg else None
        return [mx.take(arrays[0], arrays[1], axis=axis_value, **kwargs)]
    if op == "astype":
        return [arrays[0].astype(dtype_objects(mx)[by_name["dtype"]["value"]], **kwargs)]
    if op in ("bartlett", "blackman", "hamming", "hann"):
        python_name = "hanning" if op == "hann" else op
        return [getattr(mx, python_name)(by_name["size"]["value"], **kwargs)]
    if op in ("fftfreq", "rfftfreq"):
        return [getattr(mx.fft, op)(
            by_name["n"]["value"], scalar_array_value(by_name["d"]), **kwargs
        )]
    if op == "count_nonzero":
        selected_axis = by_name.get("axis", by_name.get("axes"))
        selected_axis = None if selected_axis is None else selected_axis.get("value", selected_axis.get("values"))
        return [mx.count_nonzero(arrays[0], axis=selected_axis, keepdims=by_name["keepdims"]["value"] or False, **kwargs)]
    if op == "diff":
        return [mx.diff(arrays[0], n=by_name["n"]["value"], axis=by_name["axis"]["value"], **kwargs)]
    if op == "flip":
        selected_axis = by_name.get("axis", by_name.get("axes"))
        selected_axis = None if selected_axis is None else selected_axis.get("value", selected_axis.get("values"))
        return [mx.flip(arrays[0], axis=selected_axis, **kwargs)]
    if op == "det":
        return [mx.linalg.det(arrays[0], **kwargs)]
    if op == "det_gpu":
        return [mx.linalg.det(arrays[0], stream=mx.gpu)]
    if op == "slogdet":
        return list(mx.linalg.slogdet(arrays[0], **kwargs))
    if op == "linspace":
        return [mx.linspace(
            scalar_array_value(by_name["start"]),
            scalar_array_value(by_name["stop"]),
            num=by_name["count"]["value"],
            endpoint=by_name["endpoint"]["value"],
            dtype=dtype_objects(mx)[by_name["dtype"]["value"]],
            **kwargs,
        )]
    if op == "logcumsumexp":
        return [mx.logcumsumexp(
            arrays[0], axis=by_name["axis"]["value"], reverse=by_name["reverse"]["value"],
            inclusive=by_name["inclusive"]["value"], **kwargs,
        )]
    if op == "logical_xor":
        return [mx.logical_xor(arrays[0], arrays[1], **kwargs)]
    if op == "search_sorted":
        return [mx.searchsorted(
            arrays[0], arrays[1], side="right" if by_name["right"]["value"] else "left", **kwargs
        )]
    if op == "trace":
        trace_kwargs = {}
        if "offset" in by_name:
            trace_kwargs.update(offset=by_name["offset"]["value"], axis1=by_name["axis1"]["value"], axis2=by_name["axis2"]["value"])
        if "dtype" in by_name:
            trace_kwargs["dtype"] = dtype_objects(mx)[by_name["dtype"]["value"]]
        return [mx.trace(arrays[0], **trace_kwargs, **kwargs)]
    if op == "trunc":
        return [mx.trunc(arrays[0], **kwargs)]
    if op == "unstack":
        return list(mx.unstack(arrays[0], axis=by_name["axis"]["value"], **kwargs))
    if op == "vecdot":
        return [mx.vecdot(arrays[0], arrays[1], axis=by_name["axis"]["value"], **kwargs)]
    if op == "index_update":
        kind = by_name["index"]["value"]
        if kind in ("advanced", "duplicate_advanced"):
            index = arrays[2]
        elif kind == "advanced_tuple":
            index = (arrays[2], -1)
        else:
            index = {
                "positive_slice": slice(1, 4),
                "negative_stride": slice(4, 1, -1),
                "tuple_2d": (slice(1, 3), slice(0, 2)),
                "negative_index": -1,
                "ellipsis_new_axis": (..., None, slice(1, 3)),
                "tuple_columns": (slice(None), slice(1, 3)),
                "full": slice(None),
                "empty": slice(3, 3),
                "clipped": slice(-100, 100),
                "noop": slice(100, 200),
                "negative_bounds": slice(-4, -1),
            }.get(kind)
        if by_name["index"]["value"] == "zero_stride":
            raise ValueError("zero stride is rejected before FFI")
        base = mx.add(arrays[0], mx.zeros_like(arrays[0]), **kwargs)
        mode = by_name["mode"]["value"]
        if mode == "replace":
            base.__setitem__(index, arrays[1])
            result = base
        else:
            method = {"add": "add", "min": "minimum", "max": "maximum", "product": "multiply"}[mode]
            result = getattr(base.at[index], method)(arrays[1])
        if by_name.get("return_source", {}).get("value"):
            return [result, arrays[0]]
        return [result]
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
        elif op == "contiguous":
            view = by_name["view"]["value"]
            if view == 0: result = [np.ascontiguousarray(values[0].transpose())]
            elif view == 1: result = [np.ascontiguousarray(values[0][1:7:2])]
            else: result = [np.ascontiguousarray(np.broadcast_to(values[0], by_name["shape"]["values"]))]
        elif op == "rms_norm":
            eps = scalar_array_value(by_name["eps"])
            normalized = values[0] / np.sqrt(np.mean(np.square(values[0]), axis=-1, keepdims=True) + eps)
            if by_name["has_weight"]["value"]:
                normalized = normalized * values[1]
            result = [normalized]
        elif op in ("sum", "sum_axes"):
            selected_axis = by_name.get("axis", by_name.get("axes", {"value": None})).get("value", by_name.get("axes", {}).get("values"))
            result = [np.sum(values[0], axis=selected_axis, keepdims=by_name["keepdims"]["value"] or False)]
        elif op == "take": result = [np.take(values[0], values[1])]
        elif op == "astype": result = [values[0].astype({"F32": np.float32}[by_name["dtype"]["value"]])]
        elif op in ("bartlett", "blackman", "hamming", "hann"):
            numpy_name = "hanning" if op == "hann" else op
            result = [getattr(np, numpy_name)(by_name["size"]["value"]).astype(np.asarray(outputs[0]).dtype)]
        elif op in ("fftfreq", "rfftfreq"):
            result = [getattr(np.fft, op)(
                by_name["n"]["value"], scalar_array_value(by_name["d"])
            ).astype(np.asarray(outputs[0]).dtype)]
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
        if op in ("bartlett", "blackman", "hamming", "hann", "fftfreq", "rfftfreq"):
            if not np.allclose(got, expected, rtol=0.0, atol=np.finfo(expected.dtype).eps, equal_nan=True):
                return False
        elif not np.array_equal(got, expected, equal_nan=True):
            return False
    return True


def scalar_array_value(arg):
    if arg["type"] in ("bool", "i32"):
        return arg["value"]
    if arg["type"] == "f32":
        return struct.unpack("<f", struct.pack("<I", int(arg["bits"], 16)))[0]
    if arg["type"] == "f64":
        return struct.unpack("<d", struct.pack("<Q", int(arg["bits"], 16)))[0]
    real = struct.unpack("<f", struct.pack("<I", int(arg["real_bits"], 16)))[0]
    imag = struct.unpack("<f", struct.pack("<I", int(arg["imag_bits"], 16)))[0]
    return complex(real, imag)


def gguf_string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def gguf_metadata_value(dtype, value):
    if dtype == "STRING":
        return struct.pack("<I", GGUF_VALUE_TYPES[dtype]) + gguf_string(value)
    if dtype == "ARRAY":
        element_dtype, values = value
        encoded = b"".join(struct.pack(GGUF_PACK_FORMATS[element_dtype], item) for item in values)
        return (
            struct.pack("<IIQ", GGUF_VALUE_TYPES[dtype], GGUF_VALUE_TYPES[element_dtype], len(values))
            + encoded
        )
    return struct.pack("<I", GGUF_VALUE_TYPES[dtype]) + struct.pack(GGUF_PACK_FORMATS[dtype], value)


def write_manual_gguf(path, tensors, metadata=None):
    metadata = [("general.alignment", "U32", 32)] + list(metadata or [])
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, dtype, value in metadata:
        header.extend(gguf_string(key))
        header.extend(gguf_metadata_value(dtype, value))
    offset = 0
    tensor_data = bytearray()
    for name, shape, tensor_type, data in tensors:
        offset += -offset % 32
        tensor_data.extend(b"\0" * (offset - len(tensor_data)))
        header.extend(gguf_string(name))
        header.extend(struct.pack("<I", len(shape)))
        header.extend(struct.pack(f"<{len(shape)}Q", *reversed(shape)))
        header.extend(struct.pack("<IQ", GGUF_TENSOR_TYPES[tensor_type], offset))
        tensor_data.extend(data)
        offset += len(data)
    header.extend(b"\0" * (-len(header) % 32))
    path.write_bytes(header + tensor_data)


def quantized_block(tensor_type, block_index):
    if tensor_type == "Q4_0":
        scale = 0.25 + block_index * 0.03125
        quants = [(index * 3 + block_index) % 16 for index in range(32)]
        packed = bytes(quants[index] | (quants[index + 16] << 4) for index in range(16))
        return struct.pack("<e", scale) + packed
    if tensor_type == "Q4_1":
        scale = 0.125 + block_index * 0.015625
        minimum = -1.0 + block_index * 0.25
        quants = [(index * 5 + block_index) % 16 for index in range(32)]
        packed = bytes(quants[index] | (quants[index + 16] << 4) for index in range(16))
        return struct.pack("<ee", scale, minimum) + packed
    if tensor_type == "Q5_0":
        scale = 0.0625 + block_index * 0.0078125
        quants = [(index * 7 + block_index) % 32 for index in range(32)]
        high = sum(((value >> 4) & 1) << index for index, value in enumerate(quants))
        packed = bytes(
            (quants[index] & 0xF) | ((quants[index + 16] & 0xF) << 4)
            for index in range(16)
        )
        return struct.pack("<eI", scale, high) + packed
    scale = 0.03125 + block_index * 0.00390625
    quants = [((index * 11 + block_index) % 255) - 127 for index in range(32)]
    return struct.pack("<e32b", scale, *quants)


def write_quantized_gguf(path, tensor_type):
    data = b"".join(quantized_block(tensor_type, block) for block in range(2))
    metadata = [("quantization.fixture", "STRING", tensor_type)]
    # MLX 0.32.2's upstream GGUF loader advances F64 metadata offsets by 4 instead of 8 bytes.
    write_manual_gguf(path, [("quantized.weight", [2, 32], tensor_type, data)], metadata)


def gguf_expected_array(fixtures, case_id, key, array, policy=None):
    if policy is None:
        dtype = dtype_name(array)
        policy = (
            "low_precision_float" if dtype in ("F16", "BF16")
            else "elementwise_float" if dtype in ("F32", "F64", "C64")
            else "exact_numeric"
        )
    ref = f"{case_id}.{key}"
    fixtures[ref] = array
    return {
        "key": key,
        "ref": ref,
        "dtype": dtype_name(array),
        "shape": list(array.shape),
        "policy": policy,
    }


def gguf_load_case(case_id, file, arrays, metadata, fixtures, execution="default_cpu"):
    expected_arrays = [
        gguf_expected_array(fixtures, case_id, key, array)
        for key, array in sorted(arrays.items())
    ]
    expected_metadata = []
    for key, value in sorted(metadata.items()):
        if isinstance(value, str):
            expected_metadata.append({"key": key, "kind": "string", "value": value})
        elif isinstance(value, list):
            expected_metadata.append({"key": key, "kind": "strings", "value": value})
        else:
            item = gguf_expected_array(fixtures, case_id, f"metadata.{key}", value)
            item["key"] = key
            expected_metadata.append({"kind": "array", **item})
    return {
        "id": case_id,
        "rust_call": "gguf.load",
        "recipe": {"kind": "load", "path": f"fixtures/gguf/{file}", "execution": execution},
        "expected": {
            "status": "success",
            "array_keys": sorted(arrays),
            "arrays": expected_arrays,
            "metadata": expected_metadata,
        },
    }


def generate_gguf(target, mx, np):
    directory = target / "fixtures" / "gguf"
    directory.mkdir()
    fixtures = {}
    cases = []

    tensor_dtypes = {
        "f32": mx.float32,
        "f16": mx.float16,
        "i8": mx.int8,
        "i16": mx.int16,
        "i32": mx.int32,
    }
    basic_arrays = {
        f"tensor.{name}": mx.array([-3, -1, 0, 2, 7, 11], dtype=dtype).reshape(2, 3)
        for name, dtype in tensor_dtypes.items()
    }
    mx.save_gguf(str(directory / "basic.gguf"), basic_arrays, {})
    arrays, metadata = mx.load(str(directory / "basic.gguf"), return_metadata=True, stream=mx.cpu)
    cases.append(gguf_load_case("gguf.001", "basic.gguf", arrays, metadata, fixtures))

    metadata_dtypes = {
        "bool": mx.bool_,
        "i8": mx.int8,
        "i16": mx.int16,
        "i32": mx.int32,
        "i64": mx.int64,
        "u8": mx.uint8,
        "u16": mx.uint16,
        "u32": mx.uint32,
        "u64": mx.uint64,
        "f32": mx.float32,
    }
    metadata_values = {"text": "metadata", "texts": ["one", "two", "three"]}
    for index, (name, dtype) in enumerate(metadata_dtypes.items(), 1):
        metadata_values[f"scalar.{name}"] = mx.array(index, dtype=dtype)
        metadata_values[f"vector.{name}"] = mx.array([index, index + 1], dtype=dtype)
    mx.save_gguf(str(directory / "metadata.gguf"), {"shared": mx.array([1], dtype=mx.int32)}, metadata_values)
    arrays, metadata = mx.load(str(directory / "metadata.gguf"), return_metadata=True, stream=mx.cpu)
    cases.append(gguf_load_case("gguf.002", "metadata.gguf", arrays, metadata, fixtures))

    unicode_arrays = {"张量.🦀": mx.array([1.5, -2.25], dtype=mx.float32)}
    unicode_metadata = {"clé.日本語": "naïve 🦀", "列表": ["α", "雪", "emoji 😀"]}
    mx.save_gguf(str(directory / "unicode.gguf"), unicode_arrays, unicode_metadata)
    arrays, metadata = mx.load(str(directory / "unicode.gguf"), return_metadata=True, stream=mx.cpu)
    cases.append(gguf_load_case("gguf.003", "unicode.gguf", arrays, metadata, fixtures))

    large = mx.arange(129 * 257, dtype=mx.float32).reshape(129, 257)
    mx.save_gguf(
        str(directory / "large-asymmetric.gguf"),
        {"asymmetric": large, "non_contiguous": large.transpose()},
        {},
    )
    arrays, metadata = mx.load(str(directory / "large-asymmetric.gguf"), return_metadata=True, stream=mx.cpu)
    cases.append(gguf_load_case("gguf.004", "large-asymmetric.gguf", arrays, metadata, fixtures))

    mx.save_gguf(str(directory / "empty.gguf"), {}, {})
    arrays, metadata = mx.load(str(directory / "empty.gguf"), return_metadata=True, stream=mx.cpu)
    cases.append(gguf_load_case("gguf.005", "empty.gguf", arrays, metadata, fixtures))
    (directory / "not-gguf.gguf").write_bytes(b"deterministic non-GGUF fixture\n")

    cases.extend([
        {"id": "gguf.006", "rust_call": "gguf.load_error", "recipe": {"kind": "load", "path": "fixtures/gguf/missing.gguf", "execution": "default_cpu"}, "expected": {"status": "error", "variant": "not_file"}},
        {"id": "gguf.007", "rust_call": "gguf.load_error", "recipe": {"kind": "load", "path": "fixtures/gguf/not-gguf.gguf", "execution": "default_cpu"}, "expected": {"status": "error", "variant": "exception"}},
        {"id": "gguf.008", "rust_call": "gguf.absence", "recipe": {"kind": "absence", "path": "fixtures/gguf/metadata.gguf", "array_key": "absent", "metadata_key": "absent"}, "expected": {"status": "success", "array_absent": True, "metadata_absent": True}},
        {"id": "gguf.009", "rust_call": "gguf.wrong_kind", "recipe": {"kind": "wrong_kind", "path": "fixtures/gguf/metadata.gguf", "key": "text", "requested": "array"}, "expected": {"status": "error", "variant": "wrong_metadata_kind", "expected_kind": "array", "actual_kind": "string"}},
        {"id": "gguf.010", "rust_call": "gguf.wrong_kind", "recipe": {"kind": "wrong_kind", "path": "fixtures/gguf/metadata.gguf", "key": "texts", "requested": "string"}, "expected": {"status": "error", "variant": "wrong_metadata_kind", "expected_kind": "string", "actual_kind": "strings"}},
        {"id": "gguf.011", "rust_call": "gguf.wrong_kind", "recipe": {"kind": "wrong_kind", "path": "fixtures/gguf/metadata.gguf", "key": "scalar.i32", "requested": "strings"}, "expected": {"status": "error", "variant": "wrong_metadata_kind", "expected_kind": "strings", "actual_kind": "array"}},
    ])

    for file, tensor_type, case_id in [
        ("q4-0.gguf", "Q4_0", "gguf.012"),
        ("q4-1.gguf", "Q4_1", "gguf.013"),
        ("q8-0.gguf", "Q8_0", "gguf.014"),
    ]:
        write_quantized_gguf(directory / file, tensor_type)
        arrays, metadata = mx.load(str(directory / file), return_metadata=True, stream=mx.cpu)
        case = gguf_load_case(case_id, file, arrays, metadata, fixtures)
        bits = 8 if tensor_type == "Q8_0" else 4
        case["recipe"]["dequantize"] = {"group_size": 32, "bits": bits}
        dequantized = mx.dequantize(
            arrays["quantized.weight"],
            arrays["quantized.scales"],
            arrays["quantized.biases"],
            group_size=32,
            bits=bits,
        )
        mx.eval(dequantized)
        case["expected"]["dequantized"] = gguf_expected_array(
            fixtures, case_id, "dequantized", dequantized, "low_precision_float"
        )
        cases.append(case)

    q5_file = "q5-0-unsupported.gguf"
    write_quantized_gguf(directory / q5_file, "Q5_0")
    try:
        mx.load(str(directory / q5_file), return_metadata=True, stream=mx.cpu)
    except RuntimeError:
        cases.append({
            "id": "gguf.015",
            "rust_call": "gguf.load_error",
            "recipe": {
                "kind": "load",
                "path": f"fixtures/gguf/{q5_file}",
                "execution": "default_cpu",
            },
            "expected": {"status": "error", "variant": "exception"},
        })
    else:
        raise AssertionError("Q5_0 unexpectedly loaded")

    cases.extend([
        {"id": "gguf.016", "rust_call": "gguf.prevalidation", "recipe": {"kind": "tensor_rejects", "accepted": ["F32", "F16", "I8", "I16", "I32"], "dtypes": ["BOOL", "U8", "U16", "U32", "U64", "I64", "BF16", "C64"]}, "expected": {"status": "error", "variant": "unsupported_tensor_dtype"}},
        {"id": "gguf.017", "rust_call": "gguf.prevalidation", "recipe": {"kind": "metadata_rejects", "accepted": ["BOOL", "I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64", "F32"], "dtypes": ["F16", "BF16", "C64"], "ranks": [2], "empty": True}, "expected": {"status": "error", "variants": ["unsupported_metadata_array_dtype", "invalid_metadata_array_rank", "empty_metadata_array"]}},
        {"id": "gguf.018", "rust_call": "gguf.load", "recipe": {"kind": "load", "path": "fixtures/gguf/basic.gguf", "execution": "default_cpu"}, "expected": cases[0]["expected"]},
        {"id": "gguf.019", "rust_call": "gguf.load", "recipe": {"kind": "load", "path": "fixtures/gguf/basic.gguf", "execution": "explicit_cpu"}, "expected": cases[0]["expected"]},
        {
            "id": "gguf.020",
            "rust_call": "gguf.construct",
            "recipe": {
                "kind": "construct_save",
                "path": "rust-save-qualified.gguf",
                "same_spelling": "shared",
                "metadata_value": "metadata",
                "non_contiguous_shape": [129, 257],
            },
            "expected": {
                "status": "success",
                "array_keys": ["shared"],
                "arrays": [
                    gguf_expected_array(fixtures, "gguf.020", "shared", large.transpose())
                ],
                "metadata": [{"key": "shared", "kind": "string", "value": "metadata"}],
                "duplicate_array_variant": "array_key_already_exists",
                "duplicate_metadata_variant": "metadata_key_already_exists",
            },
        },
    ])

    write_safetensors(target / "fixtures" / "gguf.safetensors", fixtures, mx, np)
    suite = {"schema_version": 1, "name": "gguf", "fixture": "fixtures/gguf.safetensors", "cases": cases}
    (target / "suites" / "gguf.json").write_text(json.dumps(suite, indent=2, allow_nan=False) + "\n")
    qualification = {
        "schema_version": 1,
        "artifact": "qualification/rust-save-qualified.gguf",
        "artifact_sha256": None,
        "producer_revision": None,
        "python": "3.12.14",
        "mlx": "0.32.2",
        "compared_fields": ["array_keys", "metadata_kinds", "dtypes", "shapes", "values"],
        "verdict": "pending",
    }
    qualification_dir = target / "qualification"
    qualification_dir.mkdir()
    (qualification_dir / "gguf-save.json").write_text(json.dumps(qualification, indent=2) + "\n")


def generate_tree(target, mx, np):
    (target / "suites").mkdir(parents=True)
    (target / "fixtures").mkdir()
    specs = build_specs()
    old_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        with mx.stream(mx.cpu):
            for suite in OP_SUITES:
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
                            outputs = call_recipe(mx, spec["recipe"], arrays, spec["extra_args"], spec["execution"]["target"])
                            invoked = True
                            for output in outputs:
                                mx.eval(output)
                        except Exception as error:
                            if stage == "eval_only" and not invoked:
                                raise RuntimeError(f"{spec['id']} did not reach evaluation") from error
                            if stage == "invoke_only" and invoked:
                                raise RuntimeError(f"{spec['id']} reached evaluation") from error
                            record["expected"] = {"status": "error", "allowed_stage": stage, "reason": reason, "python_exception": {"module": type(error).__module__, "type": type(error).__name__}, "control_case_id": control, "diagnostic": str(error)}
                            if "rust_diagnostic" in spec:
                                record["expected"]["rust_diagnostic"] = spec["rust_diagnostic"]
                            if spec["suite"] == "indexing":
                                record["expected"]["rust_error_variant"] = "zero_stride" if exception_type == "ZeroStride" else "exception"
                        else:
                            raise RuntimeError(f"{spec['id']} did not raise")
                    else:
                        outputs = call_recipe(mx, spec["recipe"], arrays, spec["extra_args"], spec["execution"]["target"])
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
            generate_gguf(target, mx, np)
    finally:
        mx.set_default_device(old_device)

    generator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fixture_shards = {
        f"fixtures/{name}.safetensors": f"sha256:{hashlib.sha256((target / 'fixtures' / f'{name}.safetensors').read_bytes()).hexdigest()}"
        for name in SUITES
    }
    gguf_fixtures = {
        f"fixtures/gguf/{path.name}": {
            "sha256": f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
            "recipe": (
                "manual_gguf_quantized"
                if path.name.startswith(("q4-", "q5-", "q8-"))
                else "raw_non_gguf"
                if path.name == "not-gguf.gguf"
                else "mlx.core.save_gguf"
            ),
        }
        for path in sorted((target / "fixtures" / "gguf").glob("*.gguf"))
    }
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
        "fixture_shards": fixture_shards,
        "gguf_fixtures": gguf_fixtures,
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
            {"id": "count_nonzero_drop_keepdims", "base_case_id": "math.004", "kind": "count_nonzero_drop_keepdims", "expected_class": "shape"},
            {"id": "count_nonzero_axis_selection", "base_case_id": "math.004", "kind": "count_nonzero_axis_selection", "expected_class": "shape"},
            {"id": "linspace_endpoint", "base_case_id": "math.036", "kind": "linspace_endpoint", "expected_class": "value"},
            {"id": "search_sorted_side", "base_case_id": "math.054", "kind": "search_sorted_side", "expected_class": "value"},
            {"id": "logcumsumexp_reverse", "base_case_id": "math.047", "kind": "logcumsumexp_reverse", "expected_class": "value"},
            {"id": "logcumsumexp_inclusive", "base_case_id": "math.044", "kind": "logcumsumexp_inclusive", "expected_class": "infinity_sign"},
            {"id": "logcumsumexp_naive", "base_case_id": "math.044", "kind": "logcumsumexp_naive", "expected_class": "infinity_sign"},
            {"id": "vecdot_without_conjugation", "base_case_id": "math.077", "kind": "vecdot_without_conjugation", "expected_class": "value"},
            {"id": "trace_ignore_dtype", "base_case_id": "math.066", "kind": "trace_ignore_dtype", "expected_class": "dtype"},
            {"id": "slogdet_reverse_outputs", "base_case_id": "math.027", "kind": "slogdet_reverse_outputs", "expected_class": "value"},
            {"id": "unstack_drop_output", "base_case_id": "math.071", "kind": "unstack_drop_output", "expected_class": "output_count"},
            {"id": "unstack_reorder_outputs", "base_case_id": "math.071", "kind": "unstack_reorder_outputs", "expected_class": "value"},
            {"id": "index_update_mode_substitution", "base_case_id": "indexing.002", "kind": "index_update_mode_substitution", "expected_class": "value"},
            {"id": "index_update_stride_reversal", "base_case_id": "indexing.007", "kind": "index_update_stride_reversal", "expected_class": "value"},
            {"id": "index_update_force_scatter", "base_case_id": "indexing.002", "kind": "index_update_force_scatter", "expected_class": "value"},
            {"id": "index_update_force_slice", "base_case_id": "indexing.012", "kind": "index_update_force_slice", "expected_class": "value"},
            {"id": "contiguous_value_mangle", "base_case_id": "shapes.015", "kind": "contiguous_value_mangle", "expected_class": "signed_zero"},
            {"id": "rms_norm_weight_ignored", "base_case_id": "fast.002", "kind": "rms_norm_weight_ignored", "expected_class": "value"},
            {"id": "rms_norm_eps_ignored", "base_case_id": "fast.002", "kind": "rms_norm_eps_ignored", "expected_class": "value"},
            {"id": "gguf_array_dtype_changed_values_equal", "base_case_id": "gguf.001", "kind": "gguf_array_dtype_changed_values_equal", "expected_class": "dtype"},
            {"id": "gguf_array_beyond_tolerance", "base_case_id": "gguf.001", "kind": "gguf_array_beyond_tolerance", "expected_class": "value_relative"},
            {"id": "gguf_array_key_removed", "base_case_id": "gguf.001", "kind": "gguf_array_key_removed", "expected_class": "array_keys"},
            {"id": "gguf_metadata_kind_swapped", "base_case_id": "gguf.002", "kind": "gguf_metadata_kind_swapped", "expected_class": "metadata_kind"},
            {"id": "gguf_metadata_entry_missing", "base_case_id": "gguf.002", "kind": "gguf_metadata_entry_missing", "expected_class": "metadata_missing"},
            {"id": "gguf_error_variant_mismatch", "base_case_id": "gguf.006", "kind": "gguf_error_variant_mismatch", "expected_class": "error_variant"},
            {"id": "gguf_wrong_kind_fields_009", "base_case_id": "gguf.009", "kind": "gguf_wrong_kind_fields", "expected_class": "wrong_kind_fields"},
            {"id": "gguf_wrong_kind_fields_010", "base_case_id": "gguf.010", "kind": "gguf_wrong_kind_fields", "expected_class": "wrong_kind_fields"},
            {"id": "gguf_wrong_kind_fields_011", "base_case_id": "gguf.011", "kind": "gguf_wrong_kind_fields", "expected_class": "wrong_kind_fields"},
            {"id": "gguf_dequantized_observation_dropped", "base_case_id": "gguf.012", "kind": "gguf_dequantized_observation_dropped", "expected_class": "dequantized_missing"},
            {"id": "gguf_dequantized_beyond_tolerance", "base_case_id": "gguf.012", "kind": "gguf_dequantized_beyond_tolerance", "expected_class": "value_relative"},
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
        for name in ("corpus.json", "qualification.json", "qualification", "suites", "fixtures"):
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
