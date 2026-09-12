import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from numpy_reference.common import read_json, read_safetensors
from numpy_reference.run import DECODE_STEPS, ROOT, fixture_directories


def expected_shapes(config, prompt_length):
    layers = config["num_hidden_layers"]
    heads = config.get("num_key_value_heads") or config["num_attention_heads"]
    dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    vocab = config["vocab_size"]
    kinds = config.get("layer_types") or ["full_attention"] * layers
    if len(kinds) != layers:
        raise ValueError("layer_types length mismatch")
    shapes = {"prefill.full.logits": (1, prompt_length, vocab)}
    for step in range(DECODE_STEPS):
        shapes[f"decode.step{step}.logits"] = (1, vocab)
    for i, kind in enumerate(kinds):
        retained = prompt_length + DECODE_STEPS
        if kind == "sliding_attention":
            retained = min(config["sliding_window"], retained)
        for stage, length in (("after_prefill", prompt_length), ("after_decode", retained)):
            for component in ("keys", "values"):
                shapes[f"cache.{stage}.layer{i}.{component}"] = (1, heads, length, dim)
    return shapes


def compare_tensors(oracle, reference, shapes, tolerances):
    for name in ("cache", "logits"):
        if name not in tolerances:
            raise ValueError(f"missing tolerance policy: {name}")
    for name, policy in sorted(tolerances.items()):
        for field in ("atol", "rtol"):
            value = policy[field]
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}.{field} tolerance")
    reports = {}
    for key in sorted(shapes.keys() | reference.keys()):
        report = {"passed": False, "max_abs": None, "max_rel": None}
        reports[key] = report
        if key not in shapes:
            report["error"] = "unexpected NumPy key"
            continue
        if key not in oracle or key not in reference:
            report["error"] = "missing oracle key" if key not in oracle else "missing NumPy key"
            continue
        expected, actual = oracle[key], reference[key]
        if expected.shape != shapes[key] or actual.shape != shapes[key]:
            report["error"] = "shape mismatch"
            report["oracle_shape"] = list(expected.shape)
            report["numpy_shape"] = list(actual.shape)
            report["required_shape"] = list(shapes[key])
            continue
        if expected.dtype != np.float32 or actual.dtype != np.float32:
            report["error"] = "dtype mismatch: both outputs must be f32"
            continue
        if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
            report["error"] = "non-finite tensor"
            continue
        expected = expected.astype(np.float64)
        actual = actual.astype(np.float64)
        delta = np.abs(expected - actual)
        policy = "cache" if key.startswith("cache.") else "logits"
        atol, rtol = tolerances[policy]["atol"], tolerances[policy]["rtol"]
        denominator = np.maximum(np.abs(expected), np.finfo(np.float32).tiny)
        report.update(
            policy=policy,
            max_abs=float(np.max(delta, initial=0)),
            max_rel=float(np.max(delta / denominator, initial=0)),
            passed=bool(np.all(delta <= atol + rtol * np.abs(expected))),
        )
        if not report["passed"]:
            report["error"] = "value exceeds tolerance"
    return reports


def corpus_manifest(fixtures_root, numpy_dir, policies):
    files = {}
    for directory, prefix in ((fixtures_root, "fixtures"), (numpy_dir, "numpy_reference/out")):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                name = f"{prefix}/{path.relative_to(directory).as_posix()}"
                files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"schema_version": 1, "files": files, "tolerance_policies": policies}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def coordinate(fixtures, numpy_dir, fixtures_root, oracle_dir=None):
    fixtures = fixture_directories(fixtures)
    policies = {}
    reports = {}
    for fixture in fixtures:
        try:
            config = read_json(fixture / "config.json")
            expectations = read_json(fixture / "expectations.json")
            shapes = expected_shapes(config, expectations["prefill"]["T"])
            oracle_path = (
                oracle_dir / f"{fixture.name}.safetensors"
                if oracle_dir is not None
                else fixture / "expectations.safetensors"
            )
            oracle = read_safetensors(oracle_path)
            reference = read_safetensors(numpy_dir / f"{fixture.name}.safetensors")
            comparisons = compare_tensors(oracle, reference, shapes, expectations["tolerances"])
            reports[fixture.name] = {
                "passed": all(report["passed"] for report in comparisons.values()),
                "keys": comparisons,
            }
            policies[fixture.name] = expectations["tolerances"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            reports[fixture.name] = {"passed": False, "error": str(error)}
    passed = all(report["passed"] for report in reports.values())
    report = {
        "passed": passed,
        "relative_error_denominator_floor": float(np.finfo(np.float32).tiny),
        "fixtures": reports,
    }
    manifest = None
    if passed:
        discovered = {path.resolve() for path in fixtures_root.iterdir() if path.is_dir()}
        outputs = {path.stem for path in numpy_dir.glob("*.safetensors")}
        if discovered != set(fixtures) or outputs != reports.keys():
            report["passed"] = False
            report["error"] = "corpus requires agreement for every fixture and NumPy output"
        else:
            manifest = corpus_manifest(fixtures_root, numpy_dir, policies)
    return report, manifest


def main():
    parser = argparse.ArgumentParser(
        description="Compare both decoder oracles and hash the agreed corpus"
    )
    parser.add_argument("fixtures", nargs="*", type=Path)
    parser.add_argument("--fixtures-root", type=Path, default=ROOT / "fixtures")
    parser.add_argument("--numpy-dir", type=Path, default=ROOT / "numpy_reference" / "out")
    parser.add_argument("--oracle-dir", type=Path)
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus.json")
    parser.add_argument("-o", "--output", type=Path, help="write the comparison result JSON here")
    args = parser.parse_args()
    try:
        discovered = (path for path in args.fixtures_root.glob("*") if path.is_dir())
        report, manifest = coordinate(
            args.fixtures or discovered,
            args.numpy_dir,
            args.fixtures_root,
            args.oracle_dir,
        )
        if manifest is not None:
            write_json(args.corpus, manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {"passed": False, "error": str(error)}
    if args.output is not None:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
