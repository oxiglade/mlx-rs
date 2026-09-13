#!/usr/bin/env python3
"""Local ruling-C coordinator and pinned Python worker; no pipeline admission."""

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import real_checkpoints as real

ORDER_SEED = 41005
BOOTSTRAP_SEED = 41006
BOOTSTRAP_REPLICATES = 10000


def metrics(sample):
    times = sample["token_ns"]
    n = len(sample["ids"])
    if (not sample["completed"] or n < 2 or len(times) != n or
            not sample["start_ns"] < times[0] or not times[-1] <= sample["completion_ns"] or
            any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError("incomplete/non-monotonic sample")
    return {"steady_tps": (n - 1) * 1e9 / (times[-1] - times[0]),
            "prefill_seconds": (times[0] - sample["start_ns"]) / 1e9,
            "whole_request_tps": n * 1e9 / (times[-1] - sample["start_ns"]),
            "drain_seconds": (sample["completion_ns"] - times[-1]) / 1e9}


def interval(values):
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP_REPLICATES))
    return [estimates[int(0.025 * BOOTSTRAP_REPLICATES)], estimates[int(0.975 * BOOTSTRAP_REPLICATES) - 1]]


def summarize(ratios):
    ci = interval(ratios)
    median = statistics.median(ratios)
    return {"median_paired_ratio": median, "ratio_ci95": ci,
            "median_gap": 1 - median, "gap_ci95": [1 - ci[1], 1 - ci[0]],
            "ratio_mad": statistics.median(abs(x - median) for x in ratios),
            "ratio_min": min(ratios), "ratio_max": max(ratios), "paired_ratios": ratios}


def decision(gap_ci, pairs):
    if gap_ci[1] <= 0.05:
        return "C closed: no qualifying gap"
    if gap_ci[0] > 0.05:
        return "C open: attribution required"
    return "extend to 30 pairs" if pairs < 30 else "inconclusive"


def order(pair):
    first = random.Random(ORDER_SEED).randrange(2)
    return ["rust", "python"] if (pair + first) % 2 == 0 else ["python", "rust"]


def ablate_source(source, yield_line="        yield y.item(), logprobs\n"):
    block = ("        if n != max_tokens:\n"
             "            next_y, next_logprobs = _step(y)\n"
             "            mx.async_eval(next_y, next_logprobs)\n")
    if source.count(block) != 1 or source.count(yield_line) != 1:
        raise ValueError("pinned generate_step scheduling/yield block changed")
    return source.replace(block, "", 1).replace(yield_line, yield_line + block, 1)


class CountedModel(dict):
    def __init__(self, model):
        # wired_limit traverses the model as a parameter tree.
        super().__init__(model)
        self.model = model
        self.forwards = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        self.forwards += 1
        return self.model(*args, **kwargs)


def python_request(model, tokenizer, module, step, benchmark, wrapper=False):
    import mlx.core as mx

    model.forwards = 0
    prompt = benchmark["prompt_ids"]
    chunk = benchmark["prefill_chunk_size"]
    # Construct the input inside the timed request on both sides.
    ids, times, sink = [], [], []
    start = time.perf_counter_ns()
    if wrapper:
        generator = module.stream_generate(model, tokenizer, prompt, max_tokens=256,
                                            prefill_step_size=chunk, max_kv_size=None)
        for response in generator:
            sink.append(response.text)
            ids.append(int(response.token))
            times.append(time.perf_counter_ns())
    else:
        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        generator = step(mx.array(prompt), model, max_tokens=256,
                         prefill_step_size=chunk, max_kv_size=None)
        for token, _ in generator:
            detokenizer.add_token(token)
            # Rust finalizes before publishing its terminal Token event.
            if len(ids) == 255:
                detokenizer.finalize()
            sink.append(detokenizer.last_segment)
            ids.append(int(token))
            times.append(time.perf_counter_ns())
    generator.close()
    mx.synchronize(module.generation_stream)
    completion = time.perf_counter_ns()
    expected_forwards = math.ceil(127 / chunk) + 257
    if len(ids) != 256 or model.forwards != expected_forwards:
        raise ValueError(f"Python fixed-count/forward mismatch: ids={len(ids)}, forwards={model.forwards}, expected={expected_forwards}")
    return {"prompt_ids": prompt, "ids": ids, "text": "".join(sink), "start_ns": start,
            "token_ns": times, "completion_ns": completion, "forward_count": model.forwards,
            "forward_count_method": "counted model __call__, including final unused lookahead",
            "drain": "generator exhausted/closed and generation stream synchronized", "completed": True}


def python_worker(args):
    from manifest import check_environment

    environment = check_environment()
    manifest = json.loads(args.manifest.read_text())
    real.verify_identity(args.case, manifest["entries"][args.case])
    real.verify_entry(manifest["entries"][args.case])
    if environment != manifest["python_environment"]:
        raise ValueError("Python environment mismatch")
    benchmark = validate_benchmark(manifest)
    module = real.select_device(args.device)
    model, tokenizer, config, provenance = real.load_case(manifest, args.case)
    if tokenizer.encode(benchmark["source_text"], add_special_tokens=False)[:128] != benchmark["prompt_ids"]:
        raise ValueError("Python source-text prompt prefix differs")
    tokenizer.eos_token_ids = set()
    counted = CountedModel(model)
    step = module.generate_step
    upstream_source = inspect.getsource(step)
    ablation_digest = None
    if args.python_no_lookahead:
        changed = ablate_source(upstream_source)
        namespace = dict(module.__dict__)
        exec(compile(changed, "<benchmark-no-lookahead>", "exec"), namespace)
        step = namespace["generate_step"]
        ablation_digest = hashlib.sha256(changed.encode()).hexdigest()
    wrapper = args.worker == "python-user-path"
    if wrapper and (args.device != "metal" or args.python_no_lookahead):
        raise ValueError("user path must be unmodified stream_generate on Metal")
    warmups = [python_request(counted, tokenizer, module, step, benchmark, wrapper) for _ in range(3)]
    sample = python_request(counted, tokenizer, module, step, benchmark, wrapper)
    real.write_new(args.output, {"schema_version": 1, "language": "python", "case": args.case,
                   "device": args.device, "stream": f"explicit {args.device} generation_stream",
                   "protocol": benchmark, "resolved_config": config, "provenance": provenance,
                   "environment": environment, "python_no_lookahead": args.python_no_lookahead,
                   "generate_step_sha256": hashlib.sha256(upstream_source.encode()).hexdigest(),
                   "ablation_sha256": ablation_digest,
                   "allocator_policy": "upstream stream_generate wired_limit" if wrapper else "unchanged defaults; no wired-limit adjustment",
                   "warmups": warmups, "sample": sample})


def validate_benchmark(manifest):
    b = manifest["benchmark"]
    if (b["case"] != "qwen3-06b-4bit" or len(b["prompt_ids"]) != 128 or b["max_tokens"] != 256 or
            b["stop_token_ids"] != [] or b["prefill_chunk_size"] <= 0 or
            b["prompt_rule"] != "first_128_without_special_tokens" or
            b["tokenizer_sha256"] != manifest["entries"][b["case"]]["files"]["tokenizer.json"]):
        raise ValueError("invalid fixed ruling-C benchmark manifest")
    return b


def command_output(command):
    return subprocess.check_output(command, text=True).strip()


def run_context(path, runner):
    context = json.loads(path.read_text())
    native = context["native_mlx"]
    if (native["version"] != "0.32.2" or not real.full_revision(native["rust_source_commit"]) or
            native["rust_source_commit"] != native["python_source_commit"]):
        raise ValueError("same pinned native MLX version/source commit required")
    for key in ("rust_build", "python_build"):
        if not native[key]:
            raise ValueError(f"missing native MLX {key} identity")
    if not context["rust_release_flags"] or context["idle_machine_confirmed"] is not True:
        raise ValueError("release flags and idle-machine confirmation required")
    repo = real.ROOT.parent.parent
    context.update({"source_commit": command_output(["git", "-C", str(repo), "rev-parse", "HEAD"]),
                    "source_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--"])).hexdigest(),
                    "source_files": {str(p.relative_to(repo)): real.sha256(p) for p in [
                        real.ROOT / "benchmark.py", real.ROOT / "real_checkpoints.py",
                        repo / "mlx-lm/examples/benchmark.rs", repo / "mlx-lm/tests/real_checkpoints.rs", repo / "Cargo.lock"]},
                    "rust_runner": str(runner.resolve()), "rust_runner_sha256": real.sha256(runner),
                    "rustc": command_output(["rustc", "-Vv"]), "os": platform.platform(),
                    "hardware": command_output(["sysctl", "-n", "hw.model", "hw.memsize", "machdep.cpu.brand_string"]),
                    "power_mode": command_output(["pmset", "-g", "custom"]),
                    "python_executable": sys.executable, "allocator_policy": "controlled workers keep defaults on both sides"})
    return context


def validate_worker(report, benchmark, device, expected_ids=None, config=None):
    if report["protocol"] != benchmark or report["device"] != device or len(report["warmups"]) != 3:
        raise ValueError("worker protocol/device/warmup mismatch")
    if config is not None and report["resolved_config"] != config:
        raise ValueError("resolved configs differ")
    for sample in [*report["warmups"], report["sample"]]:
        if sample["prompt_ids"] != benchmark["prompt_ids"] or len(sample["ids"]) != 256 or not sample["completed"]:
            raise ValueError("incomplete worker or prompt prefix differs")
        if expected_ids is not None and sample["ids"] != expected_ids:
            raise ValueError("greedy prefixes differ; throughput must not be interpreted")
        expected_ids = sample["ids"]
    return expected_ids


def coordinator(args):
    from manifest import check_environment

    manifest = real.read_manifest(args.manifest)
    benchmark = validate_benchmark(manifest)
    environment = check_environment()
    if environment != manifest["python_environment"]:
        raise ValueError("Python environment mismatch")
    if args.python_no_lookahead and not args.baseline:
        raise ValueError("--python-no-lookahead requires --baseline for independent prefix and attribution checks")
    context = run_context(args.run_context, args.rust_runner)
    context["devices"] = args.devices
    context["devices_justification"] = args.devices_justification
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    if baseline and (baseline["benchmark"] != benchmark or baseline["python_no_lookahead"] or
                     baseline["entries"] != manifest["entries"] or baseline["environment"] != environment or
                     baseline["context"]["rust_runner_sha256"] != context["rust_runner_sha256"] or
                     baseline["context"]["native_mlx"] != context["native_mlx"]):
        raise ValueError("baseline provenance/protocol differs")
    samples_dir = Path(str(args.output) + ".samples")
    samples_dir.mkdir()
    report = {"schema_version": 1, "benchmark": benchmark, "entries": manifest["entries"],
              "manifest_sha256": real.sha256(args.manifest), "environment": environment, "context": context,
              "python_no_lookahead": args.python_no_lookahead,
              "baseline": None if not args.baseline else {"path": str(args.baseline.resolve()), "sha256": real.sha256(args.baseline)},
              "order_seed": ORDER_SEED, "bootstrap_seed": BOOTSTRAP_SEED,
              "bootstrap_replicates": BOOTSTRAP_REPLICATES, "devices": {}, "schedule": [],
              "status": "C open: measurement incomplete"}

    def worker(language, device, pair, user_path=False):
        label = "python-user-path" if user_path else language
        output = samples_dir / f"{device}-{pair:02d}-{label}.json"
        if language == "rust":
            command = [str(args.rust_runner.resolve()), "--manifest", str(args.manifest.resolve()),
                       "--case", args.case, "--device", device, "--output", str(output.resolve())]
        else:
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", label,
                       "--manifest", str(args.manifest.resolve()), "--case", args.case, "--device", device,
                       "--output", str(output.resolve())]
            if args.python_no_lookahead and not user_path:
                command.append("--python-no-lookahead")
        record = {"device": device, "pair": pair, "runner": label, "command": command,
                  "start_monotonic_ns": time.monotonic_ns()}
        report["schedule"].append(record)
        with output.with_suffix(".stdout").open("x") as stdout, output.with_suffix(".stderr").open("x") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr)
        record.update(completion_monotonic_ns=time.monotonic_ns(), returncode=result.returncode)
        if result.returncode:
            raise ValueError(f"{label}/{device} failed; inspect {output.with_suffix('.stderr')}")
        record["report_sha256"] = real.sha256(output)
        return json.loads(output.read_text())

    try:
        for device in args.devices:
            pairs, expected_ids, config = [], None, None
            if baseline:
                expected_ids = baseline["devices"][device]["pairs"][0]["python"]["sample"]["ids"]
            target = args.pairs
            while len(pairs) < target:
                pair = {}
                index = len(pairs)
                for language in order(index):
                    result = worker(language, device, index)
                    expected_ids = validate_worker(result, benchmark, device, expected_ids, config)
                    config = result["resolved_config"]
                    pair[language] = result
                pairs.append(pair)
                # No rate is calculated until all warm-up and measured prefixes in the pair agree.
                for result in pair.values():
                    result["metrics"] = metrics(result["sample"])
                ratios = [p["rust"]["metrics"]["steady_tps"] / p["python"]["metrics"]["steady_tps"] for p in pairs]
                if len(pairs) == target:
                    summary = summarize(ratios)
                    state = decision(summary["gap_ci95"], len(pairs))
                    if state == "extend to 30 pairs":
                        target = 30
            report["devices"][device] = {"pairs": pairs, "summary": summary, "decision": state,
                "dispersion": {language: {"steady_tps_median": statistics.median(p[language]["metrics"]["steady_tps"] for p in pairs),
                    "steady_tps_min": min(p[language]["metrics"]["steady_tps"] for p in pairs),
                    "steady_tps_max": max(p[language]["metrics"]["steady_tps"] for p in pairs)} for language in ("rust", "python")}}
            if args.python_no_lookahead:
                original = baseline["devices"][device]["pairs"]
                if len(original) != len(pairs):
                    raise ValueError("ablation and baseline require the same paired run count; rerun with --pairs 30")
                explained, fractions = [], []
                for old, new in zip(original, pairs):
                    duration = lambda result: (result["sample"]["token_ns"][-1] - result["sample"]["token_ns"][0]) / 1e9
                    excess = duration(old["rust"]) - duration(old["python"])
                    effect = duration(new["python"]) - duration(old["python"])
                    explained.append(effect)
                    fractions.append(effect / excess if excess > 0 else None)
                ci = interval(explained)
                qualified = all(f is not None for f in fractions) and statistics.median(fractions) >= 0.5 and ci[0] > 0
                report["devices"][device]["ablation"] = {"paired_explained_seconds": explained,
                    "explained_seconds_ci95": ci, "paired_fraction_of_baseline_excess": fractions,
                    "counterfactual_bar_met": qualified,
                    "decision": "C open: scheduling trace and behavior-qualified candidate still required"}
            if device == "metal":
                user = worker("python", device, 0, user_path=True)
                validate_worker(user, benchmark, device, expected_ids, config)
                user["metrics"] = metrics(user["sample"])
                report["devices"][device]["unmodified_stream_generate"] = user
        states = [d["decision"] for d in report["devices"].values()]
        report["status"] = ("C closed: no qualifying gap" if not args.python_no_lookahead and
                            all(s == "C closed: no qualifying gap" for s in states) else
                            "C open: inconclusive" if "inconclusive" in states else "C open: attribution/candidate evidence required")
    except BaseException as error:
        report["failure"] = f"{type(error).__name__}: {error}"
        real.write_new(args.output, report)
        raise
    real.write_new(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--case", default="qwen3-06b-4bit")
    parser.add_argument("--rust-runner", type=Path)
    parser.add_argument("--run-context", type=Path)
    parser.add_argument("--devices", nargs="+", default=["cpu", "metal"])
    parser.add_argument("--devices-justification")
    parser.add_argument("--device", choices=["cpu", "metal"])
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python-no-lookahead", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--worker", choices=["python", "python-user-path"])
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[sys.argv[0]])
        return
    real.refuse_ci()
    if not args.output or not args.manifest:
        parser.error("--output and --manifest are required")
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.case != "qwen3-06b-4bit" or args.prompt_tokens != 128 or args.max_tokens != 256 or
            args.warmups != 3 or args.pairs < 10):
        parser.error("fixed protocol: qwen3-06b-4bit, 128 IDs, 256 tokens, 3 warmups, >=10 pairs")
    if args.devices != ["cpu", "metal"] and not args.devices_justification:
        parser.error("a device subset requires --devices-justification, recorded in the report")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.worker:
        if not args.device:
            parser.error("worker needs explicit --device")
        python_worker(args)
    else:
        if not args.rust_runner or not args.run_context:
            parser.error("coordinator needs --rust-runner and --run-context")
        coordinator(args)


class ProtocolTests(unittest.TestCase):
    def test_one_stateful_detokenizer_per_request(self):
        created = []

        class Detokenizer:
            def reset(self):
                self.tokens = []
                self.finalized = False

            def add_token(self, token):
                self.tokens.append(token)

            def finalize(self):
                self.finalized = True

            @property
            def last_segment(self):
                return str(self.tokens[-1]) + ("!" if self.finalized else ",")

        class Tokenizer:
            @property
            def detokenizer(self):
                instance = Detokenizer()
                created.append(instance)
                return instance

        def step(prompt, model, **kwargs):
            model.forwards = 258
            yield from ((token, None) for token in range(256))

        mx = types.ModuleType("mlx.core")
        mx.array = list
        mx.synchronize = lambda stream: None
        mlx = types.ModuleType("mlx")
        mlx.core = mx
        with patch.dict(sys.modules, {"mlx": mlx, "mlx.core": mx}):
            sample = python_request(CountedModel({}), Tokenizer(),
                                    types.SimpleNamespace(generation_stream=None), step,
                                    {"prompt_ids": list(range(128)), "prefill_chunk_size": 128})
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].tokens, list(range(256)))
        self.assertTrue(created[0].finalized)
        self.assertEqual(sample["text"], "".join(f"{n}," for n in range(255)) + "255!")

    def test_prefix_mismatch_rejects_warmup_and_measured_reports(self):
        import copy

        benchmark = {"prompt_ids": list(range(128))}
        sample = {"prompt_ids": benchmark["prompt_ids"], "ids": [7] * 256, "completed": True}
        report = {"protocol": benchmark, "device": "cpu", "resolved_config": {},
                  "warmups": [copy.deepcopy(sample) for _ in range(3)], "sample": copy.deepcopy(sample)}
        validate_worker(report, benchmark, "cpu", [7] * 256, {})
        report["warmups"][0]["ids"][0] = 8
        with self.assertRaisesRegex(ValueError, "prefixes differ"):
            validate_worker(report, benchmark, "cpu", [7] * 256, {})

    def test_timing_numerator_and_drain(self):
        sample = {"ids": [1, 2, 3], "start_ns": 1000000000,
                  "token_ns": [2000000000, 3000000000, 4000000000],
                  "completion_ns": 5000000000, "completed": True}
        self.assertEqual(metrics(sample), {"steady_tps": 1.0, "prefill_seconds": 1.0,
                                          "whole_request_tps": 1.0, "drain_seconds": 1.0})
        sample["token_ns"][1] = sample["token_ns"][0]
        with self.assertRaises(ValueError):
            metrics(sample)

    def test_paired_bootstrap_and_order(self):
        result = summarize([0.8] * 10)
        self.assertEqual(result["ratio_ci95"], [0.8, 0.8])
        self.assertAlmostEqual(result["median_gap"], 0.2)
        self.assertEqual(result, summarize([0.8] * 10))
        self.assertNotEqual(order(0), order(1))
        self.assertEqual(order(0), order(2))
        self.assertEqual(decision([0.01, 0.04], 10), "C closed: no qualifying gap")
        self.assertEqual(decision([0.04, 0.06], 10), "extend to 30 pairs")
        self.assertEqual(decision([0.04, 0.06], 30), "inconclusive")
        self.assertEqual(decision([0.06, 0.08], 10), "C open: attribution required")

    def test_ablation_moves_construction_after_yield(self):
        source = '''def generate_step():
    y, logprobs = 0, None
    n, max_tokens = 0, 3
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y, next_logprobs)
        if n == max_tokens:
            break
        yield y, logprobs
        y, logprobs = next_y, next_logprobs
        n += 1
'''
        transformed = ablate_source(source, yield_line="        yield y, logprobs\n")
        calls = []
        class MX:
            @staticmethod
            def async_eval(*args):
                pass
        namespace = {"mx": MX, "_step": lambda y: (calls.append(y) or y + 1, None)}
        exec(transformed, namespace)
        generator = namespace["generate_step"]()
        self.assertEqual(next(generator), (0, None))
        self.assertEqual(calls, [])
        self.assertEqual(list(generator), [(1, None), (2, None)])
        self.assertEqual(calls, [0, 1, 2])
        with self.assertRaises(ValueError):
            ablate_source("changed upstream source")


if __name__ == "__main__":
    main()
