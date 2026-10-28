#!/usr/bin/env python3
"""Local, independent real-checkpoint oracle and manifest preparation."""

import argparse
import dataclasses
import hashlib
import importlib
import json
import os
import re
import struct
import sys
import tempfile
import unittest
from pathlib import Path


CASES = ("llama-1b-bf16", "llama-1b-4bit", "qwen3-06b-bf16", "qwen3-06b-4bit",
         "gguf-tinyllama-q4_0", "gguf-qwen3-q8_0")
TOKENIZERS = {"gguf-tinyllama-q4_0": "gguf-tinyllama-tokenizer",
              "gguf-qwen3-q8_0": "gguf-qwen3-tokenizer"}
REPOSITORIES = dict(zip((*CASES, *TOKENIZERS.values()), (
    "mlx-community/Llama-3.2-1B-Instruct-bf16", "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Qwen3-0.6B-bf16", "mlx-community/Qwen3-0.6B-4bit",
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "ggml-org/Qwen3-0.6B-GGUF",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen3-0.6B")))
GGUF_FILES = {"gguf-tinyllama-q4_0": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
              "gguf-qwen3-q8_0": "Qwen3-0.6B-Q8_0.gguf"}
CI_KEYS = ("CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE",
           "JENKINS_URL", "TEAMCITY_VERSION", "TF_BUILD", "CIRCLECI", "TRAVIS", "BLAZE_JOB_ID", "BUILD_BUILDID")
ROOT = Path(__file__).resolve().parent


def refuse_ci():
    if any(key in os.environ for key in CI_KEYS):
        raise ValueError("local harness refuses CI environments")


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def full_revision(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def relative_path(value):
    if not value or "\\" in value or value.startswith("/") or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError(f"invalid relative path: {value}")
    return Path(value)


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
        file.write("\n")


def verify_entry(entry):
    if not full_revision(entry["revision"]) or not entry["repository"]:
        raise ValueError("repository and full immutable revision required")
    root = Path(entry["local_directory"])
    if not root.is_absolute() or not root.is_dir() or not entry["files"]:
        raise ValueError(f"missing absolute local directory/files: {root}")
    total = 0
    for name, digest in entry["files"].items():
        path = root / relative_path(name)
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or sha256(path) != digest:
            raise ValueError(f"hash mismatch: {path}")
        total += path.stat().st_size
    if total != entry["bytes"]:
        raise ValueError(f"byte total mismatch: {root}")
    # Loaders discover sidecars and shards themselves; an unlisted discoverable file is a failure.
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".safetensors", ".gguf", ".model", ".txt", ".jinja", ".tiktoken", ".py", ".jsonl"):
            if path.relative_to(root).as_posix() not in entry["files"]:
                raise ValueError(f"unlisted consumed/discoverable file: {path}")
    return root


def verify_identity(name, entry):
    if entry["repository"] != REPOSITORIES[name]:
        raise ValueError(f"wrong repository for {name}")
    if name in GGUF_FILES and GGUF_FILES[name] not in entry["files"]:
        raise ValueError(f"missing required GGUF file for {name}")


def read_manifest(path, prepared=True):
    manifest = json.loads(Path(path).read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("requires schema_version 1")
    for name in (*CASES, *TOKENIZERS.values()):
        verify_identity(name, manifest["entries"][name])
        verify_entry(manifest["entries"][name])
    if prepared:
        if manifest["devices"] != ["cpu", "metal"]:
            raise ValueError("both devices must be explicit: cpu metal")
        for name in CASES:
            case = manifest["cases"][name]
            if case["max_tokens"] != 32 or case["stop_token_ids"] != [] or not case["prompt_ids"]:
                raise ValueError(f"invalid fixed-count case: {name}")
            if case["tokenizer_entry"] != TOKENIZERS.get(name, name) or case["prefill_chunk_size"] <= 0:
                raise ValueError(f"invalid tokenizer/chunk: {name}")
    return manifest


def contract(manifest):
    return {key: manifest[key] for key in ("schema_version", "entries", "cases", "devices", "python_environment")}


def tokenizer_for(manifest, name):
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    # Passing a Path to the tokenizer-only loader cannot resolve a remote repository.
    path = Path(manifest["entries"][TOKENIZERS.get(name, name)]["local_directory"])
    return load_tokenizer(path, {"local_files_only": True, "trust_remote_code": False})


def tokenization(tokenizer, text):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(encoded, skip_special_tokens=False)
    special = tokenizer.bos_token is None or not text.startswith(tokenizer.bos_token)
    return {"encoded_ids": encoded, "decoded_text": decoded,
            "prompt_ids": tokenizer.encode(text, add_special_tokens=special)}


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def resolved_config(model, raw, settings=None):
    args = dataclasses.asdict(model.args)
    d = args.get("head_dim") or args["hidden_size"] // args["num_attention_heads"]
    rope = args.get("rope_scaling")
    if rope is None:
        scaling = {"type": "none"}
    else:
        scaling = {"type": rope.get("rope_type", rope.get("type")),
                   **{k: f32(v) if isinstance(v, float) else v for k, v in rope.items() if k not in ("type", "rope_type")}}
    quant = raw.get("quantization")
    if settings is not None:
        affine = next((v for _, v in sorted(settings.items()) if v is not False), None)
        quant = None if affine is None else {**affine, **settings}
    if quant:
        default = {"group_size": quant["group_size"], "bits": quant["bits"]}
        layers = {k: False if v is False else {"group_size": v["group_size"], "bits": v["bits"]}
                  for k, v in quant.items() if k not in ("group_size", "bits", "mode")}
        quant = {"default": default, "layers": layers}
    return {"model_type": args["model_type"], "hidden_size": args["hidden_size"],
            "layer_count": args["num_hidden_layers"], "intermediate_size": args["intermediate_size"],
            "attention_heads": args["num_attention_heads"], "kv_heads": args["num_key_value_heads"],
            "head_dim": d, "vocabulary_size": args["vocab_size"],
            "max_positions": args.get("max_position_embeddings"), "rms_norm_epsilon": f32(args["rms_norm_eps"]),
            "rope": {"dimensions": d, "theta": f32(args["rope_theta"]),
                     "traditional": args.get("rope_traditional", False), "scaling": scaling},
            "attention": [{"type": "full"} if t == "full_attention" else {"type": "sliding", "window": args["sliding_window"]}
                          for t in args.get("layer_types", ["full_attention"] * args["num_hidden_layers"])],
            "tie_word_embeddings": args["tie_word_embeddings"], "attention_bias": args.get("attention_bias", False),
            "mlp_bias": args.get("mlp_bias", False), "quantization": quant}


def gguf_header(path):
    with path.open("rb") as file:
        def take(n):
            data = file.read(n)
            if len(data) != n:
                raise ValueError("truncated GGUF header")
            return data

        def number(fmt):
            return struct.unpack("<" + fmt, take(struct.calcsize("<" + fmt)))[0]

        def string():
            return take(number("Q")).decode("utf-8")

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                element, count = number("I"), number("Q")
                return [value(element) for _ in range(count)]
            return number({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[kind])

        if take(4) != b"GGUF" or number("I") != 3:
            raise ValueError("requires GGUF v3")
        count, nmeta = number("Q"), number("Q")
        metadata = {}
        for _ in range(nmeta):
            name = string()
            if name in metadata:
                raise ValueError("duplicate GGUF metadata")
            metadata[name] = value(number("I"))
        tensors = {}
        for _ in range(count):
            name, rank = string(), number("I")
            shape = list(reversed([number("Q") for _ in range(rank)]))
            kind, offset = number("I"), number("Q")
            if name in tensors or not 1 <= rank <= 4 or not all(shape):
                raise ValueError(f"duplicate/invalid GGUF tensor: {name}")
            tensors[name] = {"shape": shape, "type": kind, "offset": offset}
    return metadata, tensors


def load_case(manifest, name):
    import mlx.core as mx
    from mlx_lm.utils import load_model

    path = Path(manifest["entries"][name]["local_directory"])
    tokenizer = tokenizer_for(manifest, name)
    if name not in TOKENIZERS:
        model, raw = load_model(path, lazy=False)
        model.eval()
        return model, tokenizer, resolved_config(model, raw), {"kind": "native_safetensors", "config": raw}
    from gguf_bridge import build, mapped_config

    names = [n for n in manifest["entries"][name]["files"] if n.endswith(".gguf")]
    if len(names) != 1:
        raise ValueError("requires exactly one actual GGUF file")
    file = path / names[0]
    metadata, tensors = gguf_header(file)
    arch = metadata["general.architecture"]
    expected_arch, expected_type = ("llama", 2) if name == "gguf-tinyllama-q4_0" else ("qwen3", 8)
    if arch != expected_arch or expected_type not in {t["type"] for t in tensors.values()}:
        raise ValueError("GGUF architecture/storage role differs from the required case")
    dimensions = []
    for stem, key in (("attn_q", "head_count"), ("attn_k", "head_count_kv"), ("attn_v", "head_count_kv")):
        rows = tensors[f"blk.0.{stem}.weight"]["shape"][0]
        heads = metadata[f"{arch}.attention.{key}"]
        if rows % heads:
            raise ValueError("GGUF head division is inexact")
        dimensions.append(rows // heads)
    if len(set(dimensions)) != 1:
        raise ValueError("GGUF Q/K/V head dimensions disagree")
    d = dimensions[0]
    for suffix in ("attention.key_length", "attention.value_length", "rope.dimension_count"):
        if metadata.get(f"{arch}.{suffix}", d) != d:
            raise ValueError(f"unsupported GGUF {suffix}")
    if metadata.get(f"{arch}.rope.scaling.type", "none") not in ("none", "linear"):
        raise ValueError("unsupported GGUF RoPE scaling")
    for suffix, expected in (("attention.causal", True), ("use_parallel_residual", False), ("attention.sliding_window", 0)):
        if metadata.get(f"{arch}.{suffix}", expected) != expected:
            raise ValueError(f"unsupported GGUF {suffix}")
    if any(n.endswith(".bias") or "rope_freq" in n for n in tensors):
        raise ValueError("GGUF biases/frequency tensors outside profile")
    bridge_metadata = dict(metadata)
    bridge_metadata[f"{arch}.attention.key_length"] = d
    bridge_metadata.setdefault(f"{arch}.rope.freq_base", 10000.0)
    with mx.stream(mx.cpu):
        converted, _ = mx.load(str(file), return_metadata=True)
    raw = mapped_config(bridge_metadata, converted)
    model, settings = build(raw, converted)
    model.eval()
    provenance = {"kind": "actual_gguf_reviewed_bridge", "gguf_file": names[0],
                  "gguf_sha256": manifest["entries"][name]["files"][names[0]],
                  "bridge_sha256": sha256(ROOT / "gguf_bridge.py"), "head_dim_from_qkv_rows": d,
                  "unqualified_original_type_fallbacks": sorted({t["type"] for t in tensors.values()} - {0, 1, 2, 3, 8}),
                  "metadata": {k: v for k, v in metadata.items() if not k.startswith("tokenizer.")},
                  "tensors": tensors, "per_group_quantization": settings, "config": raw,
                  "tokenizer_entry": TOKENIZERS[name]}
    return model, tokenizer, resolved_config(model, raw, settings), provenance


def select_device(name):
    import mlx.core as mx

    module = importlib.import_module("mlx_lm.generate")
    device = {"cpu": mx.cpu, "metal": mx.gpu}[name]
    if name == "metal" and not mx.metal.is_available():
        raise ValueError("Metal requested but unavailable")
    mx.set_default_device(device)
    module.generation_stream = mx.new_stream(device)
    return module


def prepare(manifest, report):
    from manifest import check_environment

    manifest["python_environment"] = check_environment()
    manifest["devices"] = ["cpu", "metal"]
    manifest["cases"] = {}
    for name in CASES:
        text = "The capital of France is"
        tokens = tokenization(tokenizer_for(manifest, name), text)
        manifest["cases"][name] = {"canonical_text": text, "prompt_ids": tokens["prompt_ids"],
                                   "max_tokens": 32, "prefill_chunk_size": 128, "stop_token_ids": [],
                                   "tokenizer_entry": TOKENIZERS.get(name, name)}
    tokenizer = tokenizer_for(manifest, "qwen3-06b-4bit")
    source = "The capital of France is Paris. The capital of Italy is Rome. " * 32
    manifest["benchmark"] = {"case": "qwen3-06b-4bit", "source_text": source,
                             "prompt_rule": "first_128_without_special_tokens",
                             "prompt_ids": tokenizer.encode(source, add_special_tokens=False)[:128],
                             "tokenizer_sha256": manifest["entries"]["qwen3-06b-4bit"]["files"]["tokenizer.json"],
                             "max_tokens": 256, "prefill_chunk_size": 128, "stop_token_ids": []}
    manifest["expected_report"] = {"path": str(report.resolve()), "sha256": None}
    return manifest


def collect_results(devices, run_case):
    results = {}
    for name in CASES:
        results[name] = {}
        for device in devices:
            try:
                result = run_case(name, device)
                print(f"PASS: {name}/{device}", flush=True)
            except Exception as error:
                result = {"completed": False, "error": f"{type(error).__name__}: {error}"}
                print(f"FAIL: {name}/{device}: {result['error']}", flush=True)
            results[name][device] = result
    return results


def generate_report(manifest):
    import mlx.core as mx
    from manifest import check_environment

    environment = check_environment()
    if environment != manifest["python_environment"]:
        raise ValueError("pinned Python environment differs from manifest")
    report = {"schema_version": 1, "contract": contract(manifest), "environment": environment, "results": {}}

    def run_case(name, device):
        case = manifest["cases"][name]
        module = select_device(device)
        model = None
        try:
            model, tokenizer, config, provenance = load_case(manifest, name)
            tokens = tokenization(tokenizer, case["canonical_text"])
            if tokens["decoded_text"] != case["canonical_text"] or tokens["prompt_ids"] != case["prompt_ids"]:
                raise ValueError(f"canonical tokenizer contract failed: {name}/{device}")
            steps = []

            def record_step(_, logits):
                if len(steps) < 32:
                    dtype = {mx.bfloat16: "bfloat16", mx.float16: "float16", mx.float32: "float32"}[logits.dtype]
                    row = logits[0].astype(mx.float32)
                    top1 = int(mx.argmax(row).item())
                    top2 = int(mx.argmax(mx.where(mx.arange(row.size) == top1, -float("inf"), row)).item())
                    first, second = float(row[top1].item()), float(row[top2].item())
                    steps.append({"top1_id": top1, "top2_id": top2,
                                  "top1_logit": first, "gap": first - second, "dtype": dtype})
                return logits

            ids = [int(token) for token, _ in module.generate_step(mx.array(case["prompt_ids"]), model,
                   max_tokens=32, prefill_step_size=case["prefill_chunk_size"], max_kv_size=None,
                   logits_processors=[record_step])]
            if len(ids) != 32 or len(steps) != 32:
                raise ValueError(f"incomplete generation: {name}/{device}")
            if ids != [step["top1_id"] for step in steps]:
                raise ValueError(f"raw-logit top-1 differs from generated IDs: {name}/{device}")
            return {"tokenization": tokens, "resolved_config": config, "greedy_ids": ids,
                    "greedy_steps": steps, "completed": True, "provenance": provenance}
        finally:
            mx.synchronize(module.generation_stream)
            del model
            mx.clear_cache()

    report["results"] = collect_results(manifest["devices"], run_case)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--devices", nargs="+", default=["cpu", "metal"])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--expected-report", type=Path)
    parser.add_argument("--bind-report", type=Path)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[sys.argv[0]])
        return
    refuse_ci()
    if not args.manifest or not args.output or args.devices != ["cpu", "metal"]:
        parser.error("--manifest, --output and both devices cpu metal are required")
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    manifest = read_manifest(args.manifest, prepared=not args.prepare)
    if args.prepare:
        if not args.expected_report:
            parser.error("--prepare requires --expected-report")
        output = prepare(manifest, args.expected_report)
    elif args.bind_report:
        report = json.loads(args.bind_report.read_text())
        if report["contract"] != contract(manifest):
            raise ValueError("report contract differs from manifest")
        manifest["expected_report"] = {"path": str(args.bind_report.resolve()), "sha256": sha256(args.bind_report)}
        output = manifest
    else:
        if args.output.resolve() != Path(manifest["expected_report"]["path"]):
            raise ValueError("output differs from manifest expected-report path")
        output = generate_report(manifest)
    write_new(args.output, output)
    if not args.prepare and not args.bind_report:
        if any(not result["completed"] for case in output["results"].values() for result in case.values()):
            raise SystemExit(1)


class ContractTests(unittest.TestCase):
    def test_matrix_retains_failures_and_runs_later_cases(self):
        def run_case(name, device):
            if (name, device) in ((CASES[0], "cpu"), (CASES[2], "metal")):
                raise ValueError("injected failure")
            return {"completed": True}

        results = collect_results(["cpu", "metal"], run_case)
        self.assertEqual(sum(len(case) for case in results.values()), 12)
        self.assertEqual(sum(not result["completed"] for case in results.values()
                             for result in case.values()), 2)
        self.assertIn("injected failure", results[CASES[0]]["cpu"]["error"])
        self.assertTrue(results[CASES[-1]]["metal"]["completed"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_new(path, {"results": results})
            self.assertEqual(json.loads(path.read_text())["results"], results)

    def test_fixed_case_identity(self):
        with self.assertRaisesRegex(ValueError, "repository"):
            verify_identity("qwen3-06b-4bit", {"repository": "owner/replacement"})
        with self.assertRaisesRegex(ValueError, "GGUF file"):
            verify_identity("gguf-qwen3-q8_0", {"repository": REPOSITORIES["gguf-qwen3-q8_0"], "files": {}})

    def test_hash_and_unlisted_sidecar_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            entry = {"repository": "owner/model", "revision": "a" * 40,
                     "local_directory": directory, "files": {"config.json": "0" * 64}, "bytes": 2}
            with self.assertRaisesRegex(ValueError, "hash"):
                verify_entry(entry)
            entry["files"]["config.json"] = sha256(root / "config.json")
            verify_entry(entry)
            (root / "tokenizer.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "unlisted"):
                verify_entry(entry)

    def test_ref_and_path_admission(self):
        self.assertFalse(full_revision("main"))
        self.assertTrue(full_revision("a" * 40))
        for path in ("../config.json", "/config.json", "a/../config.json", "a\\b"):
            with self.assertRaises(ValueError):
                relative_path(path)

    def test_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_new(path, {"original": True})
            with self.assertRaises(FileExistsError):
                write_new(path, {})
            self.assertEqual(json.loads(path.read_text()), {"original": True})


if __name__ == "__main__":
    main()
