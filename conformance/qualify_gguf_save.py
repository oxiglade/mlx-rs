#!/usr/bin/env python3
"""Semantic qualification of mlx-tests' ignored GGUF save artifact writer."""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def expectations():
    values = [-3, -1, 0, 2, 7, 11]
    arrays = {"tensor." + dtype: np.array(values, dtype=dtype).reshape(2, 3)
              for dtype in ("f4", "f2", "i1", "i2", "i4")}
    arrays = {dict(zip(("tensor.f4", "tensor.f2", "tensor.i1", "tensor.i2", "tensor.i4"),
                       ("tensor.f32", "tensor.f16", "tensor.i8", "tensor.i16", "tensor.i32")))[key]: value
              for key, value in arrays.items()}
    arrays["shared"] = np.arange(129 * 257, dtype=np.float32).reshape(129, 257).T
    metadata = {"shared": "metadata", "metadata.array": np.array([17, 19], dtype=np.int32),
                "metadata.string": "qualification", "metadata.strings": ["one", "two", "three"]}
    return arrays, metadata


def compare(expected, actual):
    for family, wanted, observed in zip(("arrays", "metadata"), expected, actual):
        if wanted.keys() != observed.keys():
            raise AssertionError(f"{family} keys")
        for key, value in wanted.items():
            other = observed[key]
            if type(value) is not type(other):
                raise AssertionError(f"{key} metadata kind")
            if isinstance(value, np.ndarray):
                if value.dtype != other.dtype:
                    raise AssertionError(f"{key} dtype")
                if value.shape != other.shape:
                    raise AssertionError(f"{key} shape")
                if value.tobytes() != other.tobytes():
                    raise AssertionError(f"{key} array bits")
            elif value != other:
                raise AssertionError(f"{key} value")


def qualify_mutations(actual):
    mutations = {}
    for name in ("value", "dtype", "metadata_kind"):
        expected = copy.deepcopy(expectations())
        if name == "value":
            expected[0]["tensor.f32"][0, 0] += 1
        elif name == "dtype":
            expected[0]["tensor.f32"] = expected[0]["tensor.f32"].astype(np.float16)
        else:
            expected[1]["metadata.string"] = ["qualification"]
        try:
            compare(expected, actual)
        except AssertionError:
            mutations[name] = "killed"
        else:
            raise AssertionError(f"inert save checker mutation: {name}")
    return mutations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--producer-revision", required=True, help="full revision that built the ignored Rust writer")
    args = parser.parse_args()
    if re.fullmatch("[0-9a-f]{40}", args.producer_revision) is None:
        parser.error("producer revision must be a full commit SHA")
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "mlx-lm"))
    from manifest import check_environment
    pins = check_environment()
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    arrays, metadata = mx.load(str(args.input), return_metadata=True)
    actual = ({key: np.array(value) for key, value in arrays.items()},
              {key: np.array(value) if isinstance(value, mx.array) else value for key, value in metadata.items()})
    compare(expectations(), actual)
    mutations = qualify_mutations(actual)
    result = {"schema_version": 1, "artifact": args.input.name,
              "artifact_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "producer_revision": args.producer_revision, "python": pins["python"], "mlx": pins["mlx"],
              "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "compared_fields": ["array_keys", "metadata_kinds", "dtypes", "shapes", "values"],
              "mutations": mutations, "verdict": "pass"}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
