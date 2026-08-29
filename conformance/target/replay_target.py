#!/usr/bin/env python3
import argparse
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import signal
import shutil
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

EXPECTED_PYTHON = (3, 12, 14)
EXPECTED_ARCH = "arm64"
EXPECTED_MLX = "0.32.2"
EXPECTED_NUMPY = "2.2.6"
TARGET = Path(__file__).resolve().parent
CONFORMANCE = TARGET.parent
REPO_ROOT = CONFORMANCE.parent
VENV = CONFORMANCE / ".venv-target"
LOCK = TARGET / "requirements-target.lock"
REPORT = TARGET / "replay-report.json"
EXPECTATIONS = TARGET / "target-expectations"
PROBE_SUITES = ("probe_oob_take", "probe_singular_inv")


@dataclasses.dataclass(frozen=True)
class TensorValue:
    dtype: str
    shape: list
    values: object

    def bytes(self):
        return self.values.tobytes(order="C")

    def to_json(self):
        return {
            "dtype": self.dtype,
            "shape": self.shape,
            "data_hex": self.bytes().hex(),
        }


@dataclasses.dataclass(frozen=True)
class RawTensor:
    dtype: str
    shape: list
    data: bytes


NUMPY_DTYPES = {
    "BOOL": "u1",
    "U8": "u1",
    "U16": "<u2",
    "U32": "<u4",
    "U64": "<u8",
    "I8": "i1",
    "I16": "<i2",
    "I32": "<i4",
    "I64": "<i8",
    "F16": "<f2",
    "F32": "<f4",
    "F64": "<f8",
    "BF16": "<u2",
}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def payload_sha256(payload):
    return f"sha256:{hashlib.sha256(canonical_json(payload)).hexdigest()}"


def document_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def document_sha256(value):
    return f"sha256:{hashlib.sha256(document_bytes(value)).hexdigest()}"


def file_sha256(path):
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def compare_outputs(expected, actual, policy):
    if len(expected) != len(actual):
        return {
            "verdict": "dtype_or_shape_changed",
            "expected_output_count": len(expected),
            "target_output_count": len(actual),
        }
    for index, (old, new) in enumerate(zip(expected, actual)):
        if old.dtype != new.dtype or old.shape != new.shape:
            return {
                "verdict": "dtype_or_shape_changed",
                "output": index,
                "expected_dtype": old.dtype,
                "target_dtype": new.dtype,
                "expected_shape": old.shape,
                "target_shape": new.shape,
            }
    max_error = 0.0
    changed = False
    non_finite_change = False
    for old, new in zip(expected, actual):
        old_values = comparison_values(old)
        new_values = comparison_values(new)
        if policy["kind"] == "exact_numeric":
            equal = _array_equal(old_values, new_values, True)
            error = _max_error(old_values, new_values)
        elif policy["kind"] == "float":
            equal, error = _float_equal(old_values, new_values, policy)
            if policy.get("signed_zero") and equal:
                equal = old.bytes() == new.bytes()
        else:
            raise ValueError(f"unknown comparison policy {policy['kind']}")
        if error is None:
            non_finite_change = True
        else:
            max_error = max(max_error, error)
        if not equal:
            changed = True
    if changed:
        return {
            "verdict": "value_changed",
            "max_error": None if non_finite_change else max_error,
        }
    return {"verdict": "identical"}


def comparison_values(tensor):
    if tensor.dtype != "BF16":
        return tensor.values
    import numpy as np

    words = tensor.values.astype(np.uint32) << 16
    return words.view(np.float32)


def _array_equal(old, new, nan_equal):
    import numpy as np

    try:
        return bool(np.array_equal(old, new, equal_nan=nan_equal))
    except TypeError:
        return bool(np.array_equal(old, new))


def _float_equal(old, new, policy):
    import numpy as np

    if np.iscomplexobj(old):
        components = ((old.real, new.real), (old.imag, new.imag))
    else:
        components = ((old, new),)
    max_error = 0.0
    for left, right in components:
        left_nan = np.isnan(left)
        right_nan = np.isnan(right)
        if not policy.get("nan_equal", False) and (left_nan.any() or right_nan.any()):
            return False, None
        if not np.array_equal(left_nan, right_nan):
            return False, None
        left_inf = np.isinf(left)
        right_inf = np.isinf(right)
        if not np.array_equal(left_inf, right_inf):
            return False, None
        if policy.get("infinity_sign", True) and not np.array_equal(
            np.signbit(left[left_inf]), np.signbit(right[right_inf])
        ):
            return False, None
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            errors = np.abs(left[finite] - right[finite])
            max_error = max(max_error, float(errors.max(initial=0.0)))
            allowed = policy.get("atol", 0.0) + policy.get("rtol", 0.0) * np.abs(
                left[finite]
            )
            if not bool(np.all(errors <= allowed)):
                return False, max_error
    return True, max_error


def _max_error(old, new):
    import numpy as np

    if old.size == 0:
        return 0.0
    try:
        if np.issubdtype(old.dtype, np.integer) and np.issubdtype(new.dtype, np.integer):
            return float(max(abs(int(left) - int(right)) for left, right in zip(old.flat, new.flat)))
        difference = np.abs(old.astype(np.complex128) - new.astype(np.complex128))
        finite = np.isfinite(difference)
        return float(difference[finite].max(initial=0.0)) if finite.any() else 0.0
    except (TypeError, ValueError):
        return 0.0


def compare_behavior(old, target, policy):
    if old["status"] != target["status"]:
        return {"verdict": "error_behavior_changed"}
    if old["status"] == "error":
        if old.get("exception") != target.get("exception"):
            return {"verdict": "error_behavior_changed"}
        if old.get("allowed_stage") == "eval_only" and target.get("stage") != "eval":
            return {"verdict": "error_behavior_changed"}
        return {"verdict": "identical"}
    return compare_outputs(old["outputs"], target["outputs"], policy)


def process_record(returncode):
    if returncode < 0:
        return {"exit_code": None, "signal": -returncode}
    return {"exit_code": returncode, "signal": None}


def complete_aborted_suite(case_ids, completed, returncode):
    by_id = {case["id"]: case for case in completed}
    process = process_record(returncode)
    return [
        by_id.get(
            case_id,
            {"id": case_id, "verdict": "recipe_failed", "process": process},
        )
        for case_id in case_ids
    ]


def classify_expected_abort(returncode):
    tensors = read_safetensors(CONFORMANCE / "fixtures/dtypes.safetensors")
    input_refs, input_digest = case_input_identity(
        [{"kind": "tensor", "ref": "dtypes.012.input.1"}], tensors
    )
    if returncode == -signal.SIGABRT:
        return {
            "id": "probe.singular_inv",
            "recipe": "inv_broadcast_committed_zero",
            "seed_hex": None,
            "input_refs": input_refs,
            "input_sha256": input_digest,
            "verdict": "identical",
            "old_behavior": "process_abort",
            "target_behavior": "process_abort",
            "process": process_record(returncode),
        }
    verdict = "recipe_failed" if returncode < 0 else "error_behavior_changed"
    return {
        "id": "probe.singular_inv",
        "recipe": "inv_broadcast_committed_zero",
        "seed_hex": None,
        "input_refs": input_refs,
        "input_sha256": input_digest,
        "verdict": verdict,
        "old_behavior": "process_abort",
        "target_behavior": "worker_exit",
        "process": process_record(returncode),
    }


def read_safetensors(path):
    encoded = path.read_bytes()
    if len(encoded) < 8:
        raise ValueError(f"{path} is shorter than a safetensors header")
    header_len = struct.unpack("<Q", encoded[:8])[0]
    header_end = 8 + header_len
    header = json.loads(encoded[8:header_end])
    body = encoded[header_end:]
    tensors = {}
    for name, item in header.items():
        start, end = item["data_offsets"]
        tensors[name] = RawTensor(item["dtype"], item["shape"], body[start:end])
    return tensors


def raw_numpy(raw):
    import numpy as np

    dtype = NUMPY_DTYPES.get(raw.dtype)
    if dtype is None:
        raise ValueError(f"unsupported safetensors dtype {raw.dtype}")
    values = np.frombuffer(raw.data, dtype=dtype).copy().reshape(raw.shape)
    if raw.dtype == "BOOL":
        values = values.astype(np.bool_)
    return values


def raw_value(raw):
    return TensorValue(raw.dtype, raw.shape, raw_numpy(raw))


def raw_to_mx(raw, mx):
    values = raw_numpy(raw)
    if raw.dtype == "BF16":
        return mx.view(mx.array(values, dtype=mx.uint16), mx.bfloat16)
    dtype = {
        "BOOL": mx.bool_,
        "U8": mx.uint8,
        "U16": mx.uint16,
        "U32": mx.uint32,
        "U64": mx.uint64,
        "I8": mx.int8,
        "I16": mx.int16,
        "I32": mx.int32,
        "I64": mx.int64,
        "F16": mx.float16,
        "F32": mx.float32,
        "F64": mx.float64,
    }[raw.dtype]
    return mx.array(values, dtype=dtype)


def dtype_name(array):
    text = str(array.dtype).rsplit(".", 1)[-1]
    return {
        "bool": "BOOL",
        "uint8": "U8",
        "uint16": "U16",
        "uint32": "U32",
        "uint64": "U64",
        "int8": "I8",
        "int16": "I16",
        "int32": "I32",
        "int64": "I64",
        "float16": "F16",
        "bfloat16": "BF16",
        "float32": "F32",
        "float64": "F64",
        "complex64": "C64",
    }[text]


def value_from_mx(array, mx):
    import numpy as np

    dtype = dtype_name(array)
    if dtype == "BF16":
        values = np.array(mx.view(array, mx.uint16), dtype=np.uint16)
    elif dtype == "BOOL":
        values = np.array(array, dtype=np.bool_)
    elif dtype == "C64":
        values = np.array(array, dtype=np.complex64)
    else:
        values = np.array(array)
    return TensorValue(dtype, list(array.shape), values)


def value_from_fixture(ref, record, tensors):
    if record.get("encoding") == "complex64_split":
        import numpy as np

        real = raw_numpy(tensors[ref]).astype(np.float32)
        imag = raw_numpy(tensors[record["imag_ref"]]).astype(np.float32)
        return TensorValue("C64", list(real.shape), (real + 1j * imag).astype(np.complex64))
    return raw_value(tensors[ref])


def tensor_arg(arg, tensors, mx):
    if arg.get("encoding") == "complex64_split":
        import numpy as np

        real = raw_numpy(tensors[arg["ref"]]).astype(np.float32)
        imag = raw_numpy(tensors[arg["imag_ref"]]).astype(np.float32)
        values = (real + 1j * imag).astype(np.complex64)
        return mx.array(values, dtype=mx.complex64)
    return raw_to_mx(tensors[arg["ref"]], mx)


def case_input_identity(args, tensors):
    digest = hashlib.sha256()
    refs = []
    for arg in args:
        if arg["kind"] != "tensor":
            continue
        for field in ("ref", "imag_ref"):
            if field not in arg:
                continue
            ref = arg[field]
            refs.append(ref)
            digest.update(ref.encode())
            digest.update(b"\0")
            digest.update(tensors[ref].data)
    return refs, f"sha256:{digest.hexdigest()}"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_environment():
    if sys.version_info[:3] != EXPECTED_PYTHON:
        raise SystemExit(f"requires Python 3.12.14, got {platform.python_version()}")
    if platform.machine() != EXPECTED_ARCH:
        raise SystemExit(f"requires arm64, got {platform.machine()}")
    if Path(sys.prefix).resolve() != VENV.resolve():
        raise SystemExit(f"requires pinned venv {VENV}, got {sys.prefix}")
    versions = {
        "mlx": importlib.metadata.version("mlx"),
        "mlx_metal": importlib.metadata.version("mlx-metal"),
        "numpy": importlib.metadata.version("numpy"),
    }
    if versions["mlx"] != EXPECTED_MLX:
        raise SystemExit(f"requires mlx {EXPECTED_MLX}, got {versions['mlx']}")
    if versions["mlx_metal"] != EXPECTED_MLX:
        raise SystemExit(f"requires mlx-metal {EXPECTED_MLX}, got {versions['mlx_metal']}")
    if versions["numpy"] != EXPECTED_NUMPY:
        raise SystemExit(f"requires numpy {EXPECTED_NUMPY}, got {versions['numpy']}")
    import mlx.core as mx

    runtime = getattr(mx, "__version__", None)
    if runtime != EXPECTED_MLX:
        raise SystemExit(f"requires MLX runtime {EXPECTED_MLX}, got {runtime!r}")
    return mx, {
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "venv": "conformance/.venv-target",
        **versions,
        "mlx_runtime": runtime,
        "device": "cpu",
    }


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_bytes(document_bytes(value))
    os.replace(temporary, path)


def target_behavior(call, mx):
    try:
        outputs = call()
    except Exception as error:
        return {
            "status": "error",
            "stage": "invoke",
            "exception": {"module": type(error).__module__, "type": type(error).__name__},
        }
    try:
        mx.eval(*outputs)
    except Exception as error:
        return {
            "status": "error",
            "stage": "eval",
            "exception": {"module": type(error).__module__, "type": type(error).__name__},
        }
    return {"status": "success", "outputs": [value_from_mx(output, mx) for output in outputs]}


def old_behavior(expected, tensors):
    if expected["status"] == "error":
        return {
            "status": "error",
            "exception": expected["python_exception"],
            "allowed_stage": expected["allowed_stage"],
        }
    return {
        "status": "success",
        "outputs": [
            value_from_fixture(output["ref"], output, tensors)
            for output in expected["outputs"]
        ],
    }


def expectation_json(behavior):
    if behavior["status"] == "error":
        return behavior
    return {"status": "success", "outputs": [value.to_json() for value in behavior["outputs"]]}


def run_ops_worker(suite_name, output):
    mx, _ = check_environment()
    corpus = json.loads((CONFORMANCE / "corpus.json").read_text())
    suite = json.loads((CONFORMANCE / "suites" / f"{suite_name}.json").read_text())
    tensors = read_safetensors(CONFORMANCE / suite["fixture"])
    recipes = load_module("conformance_generate", CONFORMANCE / "generate.py")
    cases = []
    old_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        with mx.stream(mx.cpu):
            for case in suite["cases"]:
                arrays = [
                    tensor_arg(arg, tensors, mx)
                    for arg in case["args"]
                    if arg["kind"] == "tensor"
                ]
                extra = [
                    arg
                    for arg in case["args"]
                    if arg["kind"] not in ("tensor", "execution")
                ]
                execution = next(arg for arg in case["args"] if arg["kind"] == "execution")
                target = target_behavior(
                    lambda: recipes.call_recipe(
                        mx,
                        case["recipe"],
                        arrays,
                        extra,
                        execution["target"] == "explicit_cpu",
                    ),
                    mx,
                )
                old = old_behavior(case["expected"], tensors)
                policy_name = (
                    case["expected"]["outputs"][0]["policy"]
                    if case["expected"]["status"] == "success"
                    else None
                )
                comparison = compare_behavior(
                    old,
                    target,
                    corpus["tolerance_policies"].get(policy_name) if policy_name else None,
                )
                refs, input_digest = case_input_identity(case["args"], tensors)
                seed = hashlib.sha256(
                    (corpus["corpus_seed"] + "\0" + case["id"]).encode()
                ).digest()[:16]
                record = {
                    "id": case["id"],
                    "recipe": case["recipe"],
                    "seed_hex": seed.hex(),
                    "input_refs": refs,
                    "input_sha256": input_digest,
                    "target": expectation_json(target),
                    **comparison,
                }
                cases.append(record)
                atomic_json(output, {"schema_version": 1, "suite": suite_name, "cases": cases})
    finally:
        mx.set_default_device(old_device)


def refs_in(value):
    refs = []
    if isinstance(value, dict):
        if isinstance(value.get("ref"), str):
            refs.append(value["ref"])
        for child in value.values():
            refs.extend(refs_in(child))
    elif isinstance(value, list):
        for child in value:
            refs.extend(refs_in(child))
    return refs


def state_inputs(module, manifest, tensors):
    module.INITIAL_PARAMETERS = {
        name: raw_numpy(tensors[record["ref"]]).tolist()
        for name, record in manifest["model"]["parameters"].items()
    }
    module.GRADIENTS = [
        {
            name: raw_numpy(tensors[record["ref"]]).tolist()
            for name, record in step.items()
        }
        for step in manifest["model"]["gradients"]
    ]
    module.TRANSFORM_INPUTS = {
        name: raw_numpy(tensors[record["ref"]]).tolist()
        for name, record in manifest["transforms"]["inputs"].items()
    }


def compare_refs(refs, generated, old_tensors, policy, mx):
    old = [raw_value(old_tensors[ref]) for ref in refs]
    target = [
        value_from_mx(value, mx) if hasattr(value, "dtype") and not hasattr(value, "tobytes") else _numpy_value(value)
        for value in (generated[ref] for ref in refs)
    ]
    return compare_outputs(old, target, policy), target


def _numpy_value(value):
    import numpy as np

    values = np.asarray(value)
    dtype = {
        "bool": "BOOL",
        "uint8": "U8",
        "uint16": "U16",
        "uint32": "U32",
        "uint64": "U64",
        "int8": "I8",
        "int16": "I16",
        "int32": "I32",
        "int64": "I64",
        "float16": "F16",
        "float32": "F32",
        "float64": "F64",
        "complex64": "C64",
    }[values.dtype.name]
    return TensorValue(dtype, list(values.shape), values)


def state_policy(manifest):
    policy = manifest["tolerance_policy"]
    return {
        "kind": "float",
        "rtol": policy["rtol"],
        "atol": policy["atol"],
        "nan_equal": True,
        "infinity_sign": True,
    }


def state_input_identity(manifest, tensors):
    refs = refs_in(manifest["model"])
    digest = hashlib.sha256()
    for ref in refs:
        digest.update(ref.encode())
        digest.update(b"\0")
        digest.update(tensors[ref].data)
    return refs, f"sha256:{digest.hexdigest()}"


def run_state_worker(output):
    mx, _ = check_environment()
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np

    manifest = json.loads((CONFORMANCE / "state/manifest.json").read_text())
    tensors = read_safetensors(CONFORMANCE / "state/state.safetensors")
    recipes = load_module("conformance_state_generate", CONFORMANCE / "state/generate_state.py")
    state_inputs(recipes, manifest, tensors)
    input_refs, input_digest = state_input_identity(manifest, tensors)
    cases = []
    old_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        with mx.stream(mx.cpu):
            for old_case in manifest["trajectories"]:
                generated = {}
                target_case = recipes.run_trajectory(
                    old_case["id"],
                    old_case["optimizer"],
                    bool(old_case["frozen_parameters"]),
                    generated,
                    mx,
                    nn,
                    optim,
                    np,
                )
                refs = refs_in(old_case["steps"])
                comparison, outputs = compare_refs(
                    refs, generated, tensors, state_policy(manifest), mx
                )
                record = {
                    "id": f"state.{old_case['id']}",
                    "recipe": old_case["optimizer"],
                    "seed_hex": None,
                    "input_refs": input_refs,
                    "input_sha256": input_digest,
                    "target": {
                        "status": "success",
                        "outputs": [
                            {"ref": ref, **value.to_json()}
                            for ref, value in zip(refs, outputs)
                        ],
                        "state_keys": [step["expected_state_keys"] for step in target_case["steps"]],
                    },
                    **comparison,
                }
                cases.append(record)
                atomic_json(output, {"schema_version": 1, "suite": "state", "cases": cases})
    finally:
        mx.set_default_device(old_device)


def run_transforms_worker(output):
    mx, _ = check_environment()
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np

    manifest = json.loads((CONFORMANCE / "state/manifest.json").read_text())
    tensors = read_safetensors(CONFORMANCE / "state/state.safetensors")
    recipes = load_module("conformance_state_generate", CONFORMANCE / "state/generate_state.py")
    state_inputs(recipes, manifest, tensors)
    generated = {}
    old_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        with mx.stream(mx.cpu):
            target_records = recipes.add_transforms(generated, mx, nn, np)
    finally:
        mx.set_default_device(old_device)
    groups = [
        ("nonlinear_value_and_grad", manifest["transforms"]["nonlinear_value_and_grad"]),
        ("argnums_selection", manifest["transforms"]["argnums_selection"]),
        ("jvp", manifest["transforms"]["directional_products"]["jvp"]),
        ("vjp", manifest["transforms"]["directional_products"]["vjp"]),
        ("module_value_and_grad", manifest["transforms"]["module_value_and_grad"]),
    ]
    input_refs = refs_in(manifest["transforms"]["inputs"])
    digest = hashlib.sha256()
    for ref in input_refs:
        digest.update(ref.encode())
        digest.update(b"\0")
        digest.update(tensors[ref].data)
    input_digest = f"sha256:{digest.hexdigest()}"
    cases = []
    for name, old_group in groups:
        refs = refs_in(old_group)
        comparison, outputs = compare_refs(refs, generated, tensors, state_policy(manifest), mx)
        cases.append(
            {
                "id": f"transforms.{name}",
                "recipe": name,
                "seed_hex": None,
                "input_refs": input_refs,
                "input_sha256": input_digest,
                "target": {
                    "status": "success",
                    "outputs": [
                        {"ref": ref, **value.to_json()}
                        for ref, value in zip(refs, outputs)
                    ],
                },
                **comparison,
            }
        )
    if not target_records:
        raise RuntimeError("transform recipes returned no records")
    atomic_json(output, {"schema_version": 1, "suite": "transforms", "cases": cases})


def run_oob_probe(output):
    mx, _ = check_environment()
    tensors = read_safetensors(CONFORMANCE / "fixtures/errors.safetensors")
    data = raw_to_mx(tensors["errors.003.input.0"], mx)
    indices = raw_to_mx(tensors["errors.003.input.1"], mx)
    target = target_behavior(lambda: [mx.take(data, indices, axis=0)], mx)
    input_args = [
        {"kind": "tensor", "ref": "errors.003.input.0"},
        {"kind": "tensor", "ref": "errors.003.input.1"},
    ]
    input_refs, input_digest = case_input_identity(input_args, tensors)
    verdict = "identical" if target["status"] == "success" else "error_behavior_changed"
    record = {
        "id": "probe.oob_take",
        "recipe": "take_valid_axis_oob_index",
        "seed_hex": None,
        "input_refs": input_refs,
        "input_sha256": input_digest,
        "target": expectation_json(target),
        "old_behavior": "success_unchecked_values",
        "target_behavior": target["status"],
        "verdict": verdict,
    }
    atomic_json(output, {"schema_version": 1, "suite": "probe_oob_take", "cases": [record]})


def run_singular_probe(output):
    mx, _ = check_environment()
    tensors = read_safetensors(CONFORMANCE / "fixtures/dtypes.safetensors")
    zero = raw_to_mx(tensors["dtypes.012.input.1"], mx)
    matrix = mx.broadcast_to(zero, (2, 2))
    target = target_behavior(lambda: [mx.linalg.inv(matrix)], mx)
    input_args = [{"kind": "tensor", "ref": "dtypes.012.input.1"}]
    input_refs, input_digest = case_input_identity(input_args, tensors)
    record = {
        "id": "probe.singular_inv",
        "recipe": "inv_broadcast_committed_zero",
        "seed_hex": None,
        "input_refs": input_refs,
        "input_sha256": input_digest,
        "target": expectation_json(target),
        "old_behavior": "process_abort",
        "target_behavior": target["status"],
        "verdict": "error_behavior_changed",
    }
    atomic_json(output, {"schema_version": 1, "suite": "probe_singular_inv", "cases": [record]})


def worker_case_ids(suite):
    if suite in PROBE_SUITES:
        return ["probe.oob_take" if suite == "probe_oob_take" else "probe.singular_inv"]
    if suite == "state":
        manifest = json.loads((CONFORMANCE / "state/manifest.json").read_text())
        return [f"state.{case['id']}" for case in manifest["trajectories"]]
    if suite == "transforms":
        return [
            "transforms.nonlinear_value_and_grad",
            "transforms.argnums_selection",
            "transforms.jvp",
            "transforms.vjp",
            "transforms.module_value_and_grad",
        ]
    document = json.loads((CONFORMANCE / "suites" / f"{suite}.json").read_text())
    return [case["id"] for case in document["cases"]]


def run_worker(kind, suite, output):
    if kind == "handshake":
        _, handshake = check_environment()
        atomic_json(output, handshake)
    elif kind == "ops":
        run_ops_worker(suite, output)
    elif kind == "state":
        run_state_worker(output)
    elif kind == "transforms":
        run_transforms_worker(output)
    elif kind == "probe_oob_take":
        run_oob_probe(output)
    elif kind == "probe_singular_inv":
        run_singular_probe(output)
    else:
        raise SystemExit(f"unknown worker kind {kind}")


def run_subprocess(python, kind, output, suite=None):
    command = [str(python), str(Path(__file__).resolve()), "--worker", kind, "--output", str(output)]
    if suite:
        command.extend(["--suite", suite])
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def source_artifacts(corpus):
    paths = [
        Path("conformance/generate.py"),
        Path("conformance/corpus.json"),
        Path("conformance/state/generate_state.py"),
        Path("conformance/state/manifest.json"),
        Path("conformance/state/state.safetensors"),
        Path("conformance/target/replay_target.py"),
        Path("conformance/target/requirements-target.lock"),
    ]
    for suite_path in corpus["suites"]:
        paths.append(Path("conformance") / suite_path)
    for fixture_path in corpus["fixture_shards"]:
        paths.append(Path("conformance") / fixture_path)
    return {path.as_posix(): file_sha256(REPO_ROOT / path) for path in sorted(paths)}


def run_once(python, directory):
    handshake_path = directory / "handshake.json"
    result = run_subprocess(python, "handshake", handshake_path)
    if result.returncode != 0 or not handshake_path.is_file():
        message = result.stderr.decode(errors="replace").strip()
        raise SystemExit(f"target handshake failed: {message or process_record(result.returncode)}")
    handshake = json.loads(handshake_path.read_text())
    corpus = json.loads((CONFORMANCE / "corpus.json").read_text())
    ops = [Path(path).stem for path in corpus["suites"]]
    suite_specs = [(suite, "ops", suite) for suite in ops]
    suite_specs.extend(
        [
            ("state", "state", None),
            ("transforms", "transforms", None),
            ("probe_oob_take", "probe_oob_take", None),
            ("probe_singular_inv", "probe_singular_inv", None),
        ]
    )
    shards = {}
    summaries = []
    for suite, kind, argument in suite_specs:
        output = directory / f"{suite}.json"
        result = run_subprocess(python, kind, output, argument)
        partial = (
            json.loads(output.read_text())
            if output.is_file()
            else {"schema_version": 1, "suite": suite, "cases": []}
        )
        if result.returncode != 0:
            if suite == "probe_singular_inv" and result.returncode < 0:
                partial["cases"] = [classify_expected_abort(result.returncode)]
            else:
                partial["cases"] = complete_aborted_suite(
                    worker_case_ids(suite), partial["cases"], result.returncode
                )
        shard = {
            "schema_version": 1,
            "suite": suite,
            "old_environment": json.loads((CONFORMANCE / "corpus.json").read_text())["environment"]
            if suite in ops or suite in PROBE_SUITES
            else json.loads((CONFORMANCE / "state/manifest.json").read_text())["provenance"]["environment"],
            "target_environment": handshake,
            "cases": partial["cases"],
        }
        shard_digest = document_sha256(shard)
        counts = Counter(case["verdict"] for case in shard["cases"])
        verdict = "fail" if counts["recipe_failed"] else "pass"
        shards[suite] = shard
        summaries.append(
            {
                "id": suite,
                "expectation_shard": f"target-expectations/{suite}.json",
                "sha256": shard_digest,
                "case_count": len(shard["cases"]),
                "verdict_counts": dict(sorted(counts.items())),
                "verdict": verdict,
            }
        )
    payload = {
        "handshake": handshake,
        "isolation": {
            "process_scope": "fresh_subprocess_per_suite",
            "state_reset": "new_model_and_optimizer_per_trajectory",
        },
        "source_artifacts": source_artifacts(corpus),
        "suites": summaries,
    }
    return payload, shards


def ensure_venv():
    if sys.version_info[:3] != EXPECTED_PYTHON:
        raise SystemExit(f"requires Python 3.12.14, got {platform.python_version()}")
    if platform.machine() != EXPECTED_ARCH:
        raise SystemExit(f"requires arm64, got {platform.machine()}")
    python = VENV / "bin/python"
    if not python.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--require-hashes",
                "-r",
                str(LOCK),
            ],
            check=True,
        )
    return python


def publish_report(payload, shards, first_hash, second_hash, update):
    staged = Path(tempfile.mkdtemp(prefix="mlx-target-publish-", dir=TARGET))
    try:
        staged_expectations = staged / "target-expectations"
        staged_expectations.mkdir()
        for name, shard in shards.items():
            atomic_json(staged_expectations / f"{name}.json", shard)
        verdict = "pass" if all(suite["verdict"] == "pass" for suite in payload["suites"]) else "fail"
        report = {
            "schema_version": 1,
            "command": "target-replay",
            "verdict": verdict,
            "payload": payload,
            "payload_sha256": payload_sha256(payload),
            "double_run": {
                "first_run_sha256": first_hash,
                "second_run_sha256": second_hash,
                "identical": first_hash == second_hash,
            },
        }
        atomic_json(staged / "replay-report.json", report)
        expected_files = {
            REPORT: (staged / "replay-report.json").read_bytes(),
            **{
                EXPECTATIONS / f"{name}.json": (staged_expectations / f"{name}.json").read_bytes()
                for name in shards
            },
        }
        existing_files = set(EXPECTATIONS.glob("*.json")) if EXPECTATIONS.is_dir() else set()
        baseline_exists = REPORT.exists() or EXPECTATIONS.exists()
        baseline_matches = (
            REPORT.is_file()
            and existing_files == set(expected_files) - {REPORT}
            and all(path.is_file() and path.read_bytes() == content for path, content in expected_files.items())
        )
        if baseline_exists and not update:
            if not baseline_matches:
                raise SystemExit(
                    "target replay differs from the committed baseline; inspect the staged semantics and rerun with --update"
                )
            return "verified"
        EXPECTATIONS.mkdir(parents=True, exist_ok=True)
        if EXPECTATIONS.exists():
            shutil.rmtree(EXPECTATIONS)
        os.replace(staged_expectations, EXPECTATIONS)
        os.replace(staged / "replay-report.json", REPORT)
        return "updated" if baseline_exists else "created"
    finally:
        shutil.rmtree(staged, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--suite")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.output is None:
            parser.error("--worker requires --output")
        run_worker(args.worker, args.suite, args.output)
        return
    python = ensure_venv()
    first_dir = Path(tempfile.mkdtemp(prefix="mlx-target-first-"))
    second_dir = Path(tempfile.mkdtemp(prefix="mlx-target-second-"))
    try:
        first_payload, first_shards = run_once(python, first_dir)
        second_payload, second_shards = run_once(python, second_dir)
        first_hash = payload_sha256({"payload": first_payload, "shards": first_shards})
        second_hash = payload_sha256({"payload": second_payload, "shards": second_shards})
        if first_hash != second_hash:
            raise SystemExit(f"target replay is not deterministic: {first_hash} != {second_hash}")
        status = publish_report(
            second_payload, second_shards, first_hash, second_hash, args.update
        )
        print(
            json.dumps(
                {
                    "report": str(REPORT),
                    "payload_sha256": payload_sha256(second_payload),
                    "baseline": status,
                }
            )
        )
    finally:
        shutil.rmtree(first_dir, ignore_errors=True)
        shutil.rmtree(second_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
