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
ROOT = Path(__file__).resolve().parent
VENV = ROOT.parent / ".venv"
FIXTURE_NAME = "state.safetensors"
MANIFEST_NAME = "manifest.json"
ATOL = 1.0e-7
RTOL = 1.0e-6

INITIAL_PARAMETERS = {
    "weight": [[0.75, -1.25, 2.0], [-0.5, 1.5, -2.25]],
    "bias": [0.2, -0.4, 0.8],
}

GRADIENTS = [
    {
        "weight": [[0.3, -0.2, 0.5], [-0.7, 0.4, -0.1]],
        "bias": [0.25, -0.5, 0.75],
    },
    {
        "weight": [[-0.6, 0.15, 0.35], [0.2, -0.45, 0.55]],
        "bias": [-0.4, 0.1, 0.3],
    },
    {
        "weight": [[0.05, 0.8, -0.25], [-0.35, 0.6, -0.9]],
        "bias": [0.6, -0.2, -0.15],
    },
]

TRANSFORM_INPUTS = {
    "x": [0.4, -0.7, 1.1],
    "weight": [[0.3, -0.8, 0.5], [1.2, 0.4, -0.6]],
    "bias": [0.15, -0.35],
    "a": [0.25, -0.9],
    "c": [1.4, -0.3],
    "a_tangent": [-0.2, 0.45],
    "c_tangent": [0.6, -0.1],
    "output0_cotangent": [0.7, -1.1],
    "output1_cotangent": 0.35,
}

OPTIMIZERS = {
    "sgd": {
        "python_class": "mlx.optimizers.SGD",
        "rust_type": "mlx_rs::optimizers::Sgd",
        "hyperparameters": {
            "learning_rate": 0.035,
            "momentum": 0.8,
            "weight_decay": 0.03,
            "dampening": 0.0,
            "nesterov": True,
        },
        "state_key_mapping": [
            {"python": "weight.v", "rust": "weight"},
            {"python": "bias.v", "rust": "bias"},
        ],
    },
    "adam": {
        "python_class": "mlx.optimizers.Adam",
        "rust_type": "mlx_rs::optimizers::Adam",
        "hyperparameters": {
            "learning_rate": 0.025,
            "betas": [0.8, 0.95],
            "eps": 1.0e-6,
            "bias_correction": False,
        },
        "state_key_mapping": [
            {"python": "weight.m", "rust": "weight.0"},
            {"python": "weight.v", "rust": "weight.1"},
            {"python": "bias.m", "rust": "bias.0"},
            {"python": "bias.v", "rust": "bias.1"},
        ],
    },
    "adamw": {
        "python_class": "mlx.optimizers.AdamW",
        "rust_type": "mlx_rs::optimizers::AdamW",
        "hyperparameters": {
            "learning_rate": 0.025,
            "betas": [0.8, 0.95],
            "eps": 1.0e-6,
            "weight_decay": 0.04,
            "bias_correction": False,
        },
        "state_key_mapping": [
            {"python": "weight.m", "rust": "weight.0"},
            {"python": "weight.v", "rust": "weight.1"},
            {"python": "bias.m", "rust": "bias.0"},
            {"python": "bias.v", "rust": "bias.1"},
        ],
    },
    "adamax": {
        "python_class": "mlx.optimizers.Adamax",
        "rust_type": "mlx_rs::optimizers::Adamax",
        "hyperparameters": {
            "learning_rate": 0.025,
            "betas": [0.8, 0.95],
            "eps": 1.0e-6,
        },
        "state_key_mapping": [
            {"python": "weight.m", "rust": "weight.0"},
            {"python": "weight.v", "rust": "weight.1"},
            {"python": "bias.m", "rust": "bias.0"},
            {"python": "bias.v", "rust": "bias.1"},
        ],
    },
    "adagrad": {
        "python_class": "mlx.optimizers.Adagrad",
        "rust_type": "mlx_rs::optimizers::AdaGrad",
        "hyperparameters": {"learning_rate": 0.04, "eps": 1.0e-6},
        "state_key_mapping": [
            {"python": "weight.v", "rust": "weight"},
            {"python": "bias.v", "rust": "bias"},
        ],
    },
    "adadelta": {
        "python_class": "mlx.optimizers.AdaDelta",
        "rust_type": "mlx_rs::optimizers::AdaDelta",
        "hyperparameters": {
            "learning_rate": 0.7,
            "rho": 0.9,
            "eps": 1.0e-6,
        },
        "state_key_mapping": [
            {"python": "weight.v", "rust": "weight.0"},
            {"python": "weight.u", "rust": "weight.1"},
            {"python": "bias.v", "rust": "bias.0"},
            {"python": "bias.u", "rust": "bias.1"},
        ],
    },
    "rmsprop": {
        "python_class": "mlx.optimizers.RMSprop",
        "rust_type": "mlx_rs::optimizers::RmsProp",
        "hyperparameters": {
            "learning_rate": 0.03,
            "alpha": 0.91,
            "eps": 1.0e-6,
        },
        "state_key_mapping": [
            {"python": "weight.v", "rust": "weight"},
            {"python": "bias.v", "rust": "bias"},
        ],
    },
    "lion": {
        "python_class": "mlx.optimizers.Lion",
        "rust_type": "mlx_rs::optimizers::Lion",
        "hyperparameters": {
            "learning_rate": 0.012,
            "betas": [0.82, 0.96],
            "weight_decay": 0.07,
        },
        "state_key_mapping": [
            {"python": "weight.m", "rust": "weight"},
            {"python": "bias.m", "rust": "bias"},
        ],
    },
    "adafactor": {
        "python_class": "mlx.optimizers.Adafactor",
        "rust_type": "mlx_rs::optimizers::Adafactor",
        "hyperparameters": {
            "learning_rate": 0.03,
            "eps": [1.0e-30, 1.0e-3],
            "clip_threshold": 1.0,
            "decay_rate": -0.8,
            "beta_1": 0.9,
            "weight_decay": 0.02,
            "scale_parameter": False,
            "relative_step": False,
            "warmup_init": False,
        },
        "state_key_mapping": [
            {"python": "step", "rust": "weight.step", "conversion": "uint64_to_int32"},
            {"python": "weight.exp_avg_sq_row", "rust": "weight.exp_avg_sq_row"},
            {"python": "weight.exp_avg_sq_col", "rust": "weight.exp_avg_sq_col"},
            {"python": "weight.exp_avg", "rust": "weight.exp_avg"},
            {"python": "step", "rust": "bias.step", "conversion": "uint64_to_int32"},
            {"python": "bias.exp_avg_sq", "rust": "bias.exp_avg_sq"},
            {"python": "bias.exp_avg", "rust": "bias.exp_avg"},
        ],
    },
}


def check_environment():
    if sys.version_info[:3] != EXPECTED_PYTHON:
        raise SystemExit(f"requires Python 3.12.14, got {platform.python_version()}")
    if platform.machine() != EXPECTED_ARCH:
        raise SystemExit(f"requires arm64, got {platform.machine()}")
    if Path(sys.prefix).resolve() != VENV.resolve():
        raise SystemExit(f"requires pinned venv {VENV}, got {sys.prefix}")
    if importlib.metadata.version("mlx") != EXPECTED_MLX:
        raise SystemExit(f"requires mlx {EXPECTED_MLX}")
    if importlib.metadata.version("mlx-metal") != EXPECTED_MLX:
        raise SystemExit(f"requires mlx-metal {EXPECTED_MLX}")

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np

    if np.__version__ != EXPECTED_NUMPY:
        raise SystemExit(f"requires numpy {EXPECTED_NUMPY}, got {np.__version__}")
    return mx, nn, optim, np


def make_optimizer(name, optim):
    h = OPTIMIZERS[name]["hyperparameters"]
    if name == "sgd":
        return optim.SGD(
            learning_rate=h["learning_rate"],
            momentum=h["momentum"],
            weight_decay=h["weight_decay"],
            dampening=h["dampening"],
            nesterov=h["nesterov"],
        )
    if name == "adam":
        return optim.Adam(
            learning_rate=h["learning_rate"],
            betas=h["betas"],
            eps=h["eps"],
            bias_correction=h["bias_correction"],
        )
    if name == "adamw":
        return optim.AdamW(
            learning_rate=h["learning_rate"],
            betas=h["betas"],
            eps=h["eps"],
            weight_decay=h["weight_decay"],
            bias_correction=h["bias_correction"],
        )
    if name == "adamax":
        return optim.Adamax(
            learning_rate=h["learning_rate"], betas=h["betas"], eps=h["eps"]
        )
    if name == "adagrad":
        return optim.Adagrad(learning_rate=h["learning_rate"], eps=h["eps"])
    if name == "adadelta":
        return optim.AdaDelta(
            learning_rate=h["learning_rate"], rho=h["rho"], eps=h["eps"]
        )
    if name == "rmsprop":
        return optim.RMSprop(
            learning_rate=h["learning_rate"], alpha=h["alpha"], eps=h["eps"]
        )
    if name == "lion":
        return optim.Lion(
            learning_rate=h["learning_rate"],
            betas=h["betas"],
            weight_decay=h["weight_decay"],
        )
    if name == "adafactor":
        return optim.Adafactor(
            learning_rate=h["learning_rate"],
            eps=tuple(h["eps"]),
            clip_threshold=h["clip_threshold"],
            decay_rate=h["decay_rate"],
            beta_1=h["beta_1"],
            weight_decay=h["weight_decay"],
            scale_parameter=h["scale_parameter"],
            relative_step=h["relative_step"],
            warmup_init=h["warmup_init"],
        )
    raise ValueError(f"unknown optimizer {name}")


def make_model(nn, mx):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.array(INITIAL_PARAMETERS["weight"], dtype=mx.float32)
            self.bias = mx.array(INITIAL_PARAMETERS["bias"], dtype=mx.float32)

    return TinyModel()


def get_path(tree, path):
    value = tree
    for part in path.split("."):
        value = value[part]
    return value


def tensor_record(ref, value):
    return {
        "ref": ref,
        "dtype": str(value.dtype).rsplit(".", 1)[-1],
        "shape": list(value.shape),
    }


def state_value(mapping, state, mx):
    value = get_path(state, mapping["python"])
    if mapping.get("conversion") == "uint64_to_int32":
        value = value.astype(mx.int32)
    return value


def tensor_bytes(mx, np, array):
    dtype_name = str(array.dtype).rsplit(".", 1)[-1]
    dtype = {"float32": "F32", "int32": "I32"}.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported dtype {array.dtype}")
    return dtype, np.array(array).tobytes(order="C")


def write_safetensors(path, tensors, mx, np):
    header = {}
    payloads = []
    offset = 0
    for name, array in sorted(tensors.items()):
        dtype, payload = tensor_bytes(mx, np, array)
        end = offset + len(payload)
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, end],
        }
        payloads.append(payload)
        offset = end
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads))


def add_inputs(fixtures, mx):
    inputs = {"parameters": {}, "gradients": []}
    for name, values in INITIAL_PARAMETERS.items():
        ref = f"input.param.{name}"
        value = mx.array(values, dtype=mx.float32)
        fixtures[ref] = value
        inputs["parameters"][name] = tensor_record(ref, value)
    for step, gradients in enumerate(GRADIENTS, 1):
        record = {}
        for name, values in gradients.items():
            ref = f"input.gradient.step{step}.{name}"
            value = mx.array(values, dtype=mx.float32)
            fixtures[ref] = value
            record[name] = tensor_record(ref, value)
        inputs["gradients"].append(record)
    return inputs


def snapshot_tensor(fixtures, ref, value, np):
    snapshot = np.array(value, copy=True)
    fixtures[ref] = snapshot
    return tensor_record(ref, snapshot)


def add_transforms(fixtures, mx, nn, np):
    arrays = {
        name: mx.array(values, dtype=mx.float32)
        for name, values in TRANSFORM_INPUTS.items()
    }
    inputs = {
        name: snapshot_tensor(fixtures, f"transform.input.{name}", value, np)
        for name, value in arrays.items()
    }

    def nonlinear(x, weight, bias):
        return mx.sum(mx.square(mx.tanh(weight @ x + bias)))

    value_and_grads = mx.value_and_grad(nonlinear, argnums=(0, 1, 2))
    value, gradients = value_and_grads(
        arrays["x"], arrays["weight"], arrays["bias"]
    )
    mx.eval(value, *gradients)
    nonlinear_records = {
        "value": snapshot_tensor(
            fixtures, "transform.nonlinear.value", value, np
        ),
        "gradients": {
            name: snapshot_tensor(
                fixtures, f"transform.nonlinear.gradient.{name}", gradient, np
            )
            for name, gradient in zip(("x", "weight", "bias"), gradients)
        },
    }

    selected_value_and_grads = mx.value_and_grad(nonlinear, argnums=(0, 2))
    selected_value, selected_gradients = selected_value_and_grads(
        arrays["x"], arrays["weight"], arrays["bias"]
    )
    mx.eval(selected_value, *selected_gradients)
    argnums_records = {
        "argnums": [0, 2],
        "value": snapshot_tensor(
            fixtures, "transform.argnums.value", selected_value, np
        ),
        "gradients": {
            name: snapshot_tensor(
                fixtures, f"transform.argnums.gradient.{name}", gradient, np
            )
            for name, gradient in zip(("x", "bias"), selected_gradients)
        },
    }

    def multi_output(a, c):
        return mx.tanh(a * c + mx.square(a)), mx.sum(a * mx.square(c))

    primals = (arrays["a"], arrays["c"])
    tangents = (arrays["a_tangent"], arrays["c_tangent"])
    jvp_values, jvp_tangents = mx.jvp(multi_output, primals, tangents)
    cotangents = (
        arrays["output0_cotangent"],
        arrays["output1_cotangent"],
    )
    vjp_values, vjp_cotangents = mx.vjp(multi_output, primals, cotangents)
    mx.eval(*jvp_values, *jvp_tangents, *vjp_values, *vjp_cotangents)

    def records(prefix, values):
        return [
            snapshot_tensor(fixtures, f"{prefix}.{index}", value, np)
            for index, value in enumerate(values)
        ]

    directional_records = {
        "jvp": {
            "values": records("transform.jvp.value", jvp_values),
            "tangents": records("transform.jvp.tangent", jvp_tangents),
        },
        "vjp": {
            "values": records("transform.vjp.value", vjp_values),
            "cotangents": records("transform.vjp.cotangent", vjp_cotangents),
        },
    }

    class TransformModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = arrays["weight"]
            self.bias = arrays["bias"]

        def __call__(self, x):
            return mx.sum(mx.square(mx.tanh(self.weight @ x + self.bias)))

    model = TransformModel()
    module_value_and_grad = nn.value_and_grad(model, model.__call__)
    module_value, module_gradients = module_value_and_grad(arrays["x"])
    mx.eval(module_value, *module_gradients.values())
    module_records = {
        "value": snapshot_tensor(
            fixtures, "transform.module.value", module_value, np
        ),
        "gradients": {
            name: snapshot_tensor(
                fixtures, f"transform.module.gradient.{name}", gradient, np
            )
            for name, gradient in module_gradients.items()
        },
    }

    return {
        "function": "sum(tanh(weight @ x + bias) ** 2)",
        "inputs": inputs,
        "nonlinear_value_and_grad": nonlinear_records,
        "argnums_selection": argnums_records,
        "directional_function": [
            "tanh(a * c + a ** 2)",
            "sum(a * c ** 2)",
        ],
        "directional_products": directional_records,
        "module_value_and_grad": module_records,
    }


def run_trajectory(case_id, optimizer_name, frozen_bias, fixtures, mx, nn, optim, np):
    model = make_model(nn, mx)
    if frozen_bias:
        model.freeze(keys="bias", strict=True)
    optimizer = make_optimizer(optimizer_name, optim)
    mapping = [
        item
        for item in OPTIMIZERS[optimizer_name]["state_key_mapping"]
        if not frozen_bias or not item["rust"].startswith("bias")
    ]
    steps = []
    for step, gradient_values in enumerate(GRADIENTS, 1):
        gradients = {
            name: mx.array(values, dtype=mx.float32)
            for name, values in gradient_values.items()
            if not frozen_bias or name != "bias"
        }
        optimizer.update(model, gradients)
        parameter_values = model.parameters()
        state_values = [state_value(item, optimizer.state, mx) for item in mapping]
        mx.eval(*parameter_values.values(), *state_values)

        params = {}
        for name, value in parameter_values.items():
            ref = f"{case_id}.step{step}.param.{name}"
            snapshot = np.array(value, copy=True)
            fixtures[ref] = snapshot
            params[name] = tensor_record(ref, snapshot)
        states = {}
        for item, value in zip(mapping, state_values):
            rust_name = item["rust"]
            ref = f"{case_id}.step{step}.state.{rust_name}"
            snapshot = np.array(value, copy=True)
            fixtures[ref] = snapshot
            states[rust_name] = tensor_record(ref, snapshot)
        steps.append(
            {
                "step": step,
                "parameters": params,
                "state": states,
                "expected_state_keys": sorted(states),
            }
        )
    return {
        "id": case_id,
        "optimizer": optimizer_name,
        "frozen_parameters": ["bias"] if frozen_bias else [],
        "gradient_keys": ["weight"] if frozen_bias else ["bias", "weight"],
        "steps": steps,
    }


def write_tree(target, mx, nn, optim, np):
    fixtures = {}
    inputs = add_inputs(fixtures, mx)
    transforms = add_transforms(fixtures, mx, nn, np)
    trajectories = []
    for name in OPTIMIZERS:
        trajectories.append(
            run_trajectory(name, name, False, fixtures, mx, nn, optim, np)
        )
    trajectories.append(
        run_trajectory("adam_frozen_bias", "adam", True, fixtures, mx, nn, optim, np)
    )
    write_safetensors(target / FIXTURE_NAME, fixtures, mx, np)

    generator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fixture_digest = hashlib.sha256((target / FIXTURE_NAME).read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "provenance": {
            "computed_by": "Python MLX optimizer implementations",
            "oracle_independence": "No mlx-rs output is read or used to derive expectations.",
            "canonical_device": "cpu",
            "generator_sha256": generator_digest,
            "fixture_sha256": fixture_digest,
            "environment": {
                "python": "3.12.14",
                "architecture": EXPECTED_ARCH,
                "pinned_venv": "conformance/.venv",
                "mlx": EXPECTED_MLX,
                "mlx_metal": EXPECTED_MLX,
                "numpy": EXPECTED_NUMPY,
            },
        },
        "tolerance_policy": {
            "name": "optimizer_f32_chain",
            "rtol": RTOL,
            "atol": ATOL,
            "non_finite": "same_kind_and_infinity_sign",
            "rationale": "Tight f32 allowance for backend evaluation or fusion ordering; integer step slots remain exact.",
        },
        "python_state_not_mapped": {
            "step": "mlx-rs has no global step except Adafactor, where this value maps to each per-parameter step slot.",
            "learning_rate": "mlx-rs stores learning rate as optimizer configuration, not flattened optimizer state.",
        },
        "model": {
            "type": "two_parameter_tensor_model",
            "frozen_variant": "adam_frozen_bias",
            **inputs,
        },
        "optimizers": [
            {"id": name, **spec}
            for name, spec in OPTIMIZERS.items()
        ],
        "transforms": transforms,
        "trajectories": trajectories,
        "fault_matrix": [
            {"id": "no_op_learning_rate", "expected_class": "parameter.weight"},
            {"id": "stuck_step_counter", "expected_class": "state.weight.step"},
            {"id": "reordered_state_tensors", "expected_class": "state.weight.0"},
            {"id": "frozen_parameter_mutation", "expected_class": "parameter.bias"},
            {"id": "wrong_step_expectation", "expected_class": "parameter.weight"},
            {"id": "perturbed_input_gradient", "expected_class": "transform.gradient.x"},
            {"id": "swapped_directional_product", "expected_class": "transform.jvp.tangent.0"},
            {"id": "output_split_shift", "expected_class": "compile.output_count"},
            {"id": "duplicate_retry", "expected_class": "compile.retry.counter"},
        ],
    }
    (target / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")


def tree_hash(path):
    digest = hashlib.sha256()
    for item in sorted(entry for entry in path.rglob("*") if entry.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


def main():
    mx, nn, optim, np = check_environment()
    first = Path(tempfile.mkdtemp(prefix="mlx-rs-state-a-"))
    second = Path(tempfile.mkdtemp(prefix="mlx-rs-state-b-"))
    old_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        with mx.stream(mx.cpu):
            write_tree(first, mx, nn, optim, np)
            write_tree(second, mx, nn, optim, np)
        first_hash = tree_hash(first)
        second_hash = tree_hash(second)
        if first_hash != second_hash:
            raise SystemExit(f"generation is not reproducible: {first_hash} != {second_hash}")
        for name in (FIXTURE_NAME, MANIFEST_NAME):
            os.replace(first / name, ROOT / name)
        sizes = {
            name: (ROOT / name).stat().st_size for name in (FIXTURE_NAME, MANIFEST_NAME)
        }
        print(json.dumps({"tree_sha256": first_hash, "sizes": sizes}, sort_keys=True))
    finally:
        mx.set_default_device(old_device)
        shutil.rmtree(first, ignore_errors=True)
        shutil.rmtree(second, ignore_errors=True)


if __name__ == "__main__":
    main()
