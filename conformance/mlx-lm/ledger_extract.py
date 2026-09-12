#!/usr/bin/env python3
"""Inventory the pinned Python source without importing its Metal runtime."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

MODULES = ("__init__", "generate", "sample_utils", "tokenizer_utils", "utils", "models.cache")
PINNED_PACKAGE = Path("/Users/ci/hub/scratch/mlx-lm/oracle/.venv-lm/lib/python3.12/site-packages/mlx_lm")


def default_package():
    local = Path(__file__).resolve().parents[1] / ".venv-mlx-lm/lib/python3.12/site-packages/mlx_lm"
    return local if local.is_dir() else PINNED_PACKAGE


class Source:
    def __init__(self, package):
        self.package = package
        self.files = {}
        self.paths = {}

    def module(self, name):
        if name not in self.files:
            path = self.package / (name.replace(".", "/") + ".py")
            if not path.is_file():
                path = path.with_suffix("") / "__init__.py"
            data = path.read_bytes()
            self.paths[name] = "mlx_lm/" + path.relative_to(self.package).as_posix()
            self.files[name] = (ast.parse(data, filename=str(path)), hashlib.sha256(data).hexdigest())
        return self.files[name][0]

    def resolve(self, module, name, seen=()):
        key = (module, name)
        if key in seen:
            return None
        for node in self.module(module).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                return module, node
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if (alias.asname or alias.name) != name:
                        continue
                    if node.level:
                        parent = module.split(".")[:-1]
                        target = parent[:len(parent) - node.level + 1] + (node.module or "").split(".")
                        target = ".".join(filter(None, target))
                    elif (node.module or "").startswith("mlx_lm."):
                        target = node.module[len("mlx_lm."):]
                    else:
                        return None
                    return self.resolve(target, alias.name, seen + (key,))
        return None

    def signature(self, node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result = f"({ast.unparse(node.args)})"
            return result + (f" -> {ast.unparse(node.returns)}" if node.returns else "")
        constructor = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if constructor:
            return self.signature(constructor)
        if any(ast.unparse(d).split("(")[0] == "dataclass" for d in node.decorator_list):
            fields = []
            for field in node.body:
                if isinstance(field, ast.AnnAssign):
                    fields.append(ast.unparse(field))
            return "(" + ", ".join(fields) + ")"
        return "(inherited constructor)" if node.bases else "()"

    def entry(self, name, module, node):
        return {
            "name": name,
            "kind": "class" if isinstance(node, ast.ClassDef) else "function",
            "signature": self.signature(node),
            "source_path": self.paths[module],
            "source_sha256": self.files[module][1],
        }

    def inventory(self):
        version = self.module("_version")
        versions = [ast.literal_eval(n.value) for n in version.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)]
        if versions != ["0.31.3"]:
            raise ValueError(f"expected mlx_lm 0.31.3, found {versions}")
        entries = {}
        for module in MODULES:
            tree = self.module(module)
            names = set()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, ast.ImportFrom):
                    names.update(a.asname or a.name for a in node.names)
            exports = next((ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)), None)
            for name in sorted(exports if exports is not None else names):
                if name.startswith("_"):
                    continue
                resolved = self.resolve(module, name)
                if resolved is None:
                    if exports is not None:
                        raise ValueError(f"cannot resolve public export {module}.{name}")
                    continue
                source_module, node = resolved
                public_name = "mlx_lm." + ("" if module == "__init__" else module + ".") + name
                entries[public_name] = self.entry(public_name, source_module, node)
                if isinstance(node, ast.ClassDef):
                    for member in node.body:
                        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and not member.name.startswith("_"):
                            member_name = public_name + "." + member.name
                            entries[member_name] = self.entry(member_name, source_module, member)
        return [entries[name] for name in sorted(entries)]


def disposition(entry):
    name = entry["name"]
    parts = name.split(".")
    leaf = parts[-1]
    owner = parts[-2]
    path = None
    if leaf == "convert":
        status, trigger = "skipped", "Admit a separate checkpoint-conversion product."
    elif "speculative" in name:
        status, trigger = "skipped", "Approve a speculative-decoding design with draft-model parity."
    elif any(word in name for word in ("Batch", "batch_generate", "SequenceStateMachine")):
        status, trigger = "deferred", "Approve batching, per-request cache ownership, and batch parity fixtures."
    elif leaf == "generate":
        status, trigger = "intentionally_unexposed", "Revisit only if iterator collection cannot express a supported text consumer."
    elif leaf == "stream_generate":
        status, path, trigger = "planned_wrapped", "mlx_lm::Model::generate", "Tranche 3 generation and event parity."
    elif leaf == "generate_step":
        status, trigger = "internal", "Tranche 3 internal transactional generation engine."
    elif leaf == "wired_limit":
        status, trigger = "intentionally_unexposed", "Require an explicit process-wide resource-control design."
    elif any(word in name for word in ("QuantizedKV", "quantize_kv", "to_quantized", "LRUPrompt", "PromptTrie", "TokenBuffer", "save_prompt_cache", "load_prompt_cache")):
        status, trigger = "deferred", "Approve quantized or persisted prompt-cache ownership and compatibility fixtures."
    elif leaf in ("KVCache", "RotatingKVCache") or owner in ("KVCache", "RotatingKVCache"):
        status, path, trigger = "planned_wrapped", "mlx_lm::CachePolicy", "Tranche 2 full and rotating cache parity."
    elif "cache" in parts:
        status, trigger = "internal", "Revisit cache helpers when an admitted architecture requires a new cache policy."
    elif leaf in ("load_adapters",):
        status, trigger = "deferred", "Approve adapter loading, composition, and weight-validation fixtures."
    elif leaf in ("load", "load_model", "load_config", "hf_repo_to_path") and "tokenizer_utils" not in parts:
        status, path, trigger = "planned_wrapped", "mlx_lm::Model::from_dir", "Tranche 2 strict local loading; tranche 4 opt-in Hub loading."
    elif leaf in ("make_sampler", "apply_top_k", "apply_top_p", "apply_min_p", "categorical_sampling"):
        status, path, trigger = "planned_wrapped", "mlx_lm::SamplerOptions", "Tranche 3 CPU sampler sequences and support parity."
    elif leaf in ("make_repetition_penalty", "make_logits_processors"):
        status, path, trigger = "planned_wrapped", "mlx_lm::RepetitionPenaltyOptions", "Tranche 3 repetition processor parity."
    elif leaf in ("apply_xtc", "make_presence_penalty", "make_frequency_penalty"):
        status, trigger = "deferred", "Admit these processors with named use cases and isolated sampler fixtures."
    elif "tokenizer_utils" in parts or leaf in ("load_tokenizer", "TokenizerWrapper") or owner == "TokenizerWrapper":
        if any(word in name for word in ("tool", "think")):
            status, trigger = "deferred", "Approve structured tools or reasoning-channel tokenizer semantics."
        else:
            status, path, trigger = "planned_wrapped", "mlx_lm::Tokenizer", "Tranche 2 textual tokenizer/chat and tranche 3 streaming-delta parity."
    elif leaf == "GenerationResponse":
        status, path, trigger = "planned_wrapped", "mlx_lm::GenerationEvent", "Tranche 3 token, text, and finish events; omit timing, logprobs, and draft metadata."
    elif leaf in ("str2bool", "setup_arg_parser", "main", "sharded_load", "pipeline_load", "make_shards", "create_model_card", "upload_to_hub", "save_model", "quantize_model", "dequantize_model", "save_config", "save"):
        status, trigger = "skipped", "Approve a separate CLI, distributed execution, or checkpoint-conversion product."
    elif leaf in ("get_total_parameters", "compute_bits_per_weight"):
        status, trigger = "deferred", "Admit model diagnostics with a defined public statistics contract."
    else:
        status, trigger = "internal", "Revisit only when a supported model or generation path requires this helper publicly."
    differences = []
    if leaf in ("load", "load_model") and "tokenizer_utils" not in parts:
        differences = ["Local safetensors via Model::from_dir; Hub via opt-in Model::from_hub.", "Adapters deferred until adapter design; lazy/custom Python model and remote code skipped."]
    if leaf in ("make_sampler", "make_logits_processors"):
        differences = ["Only greedy, temperature, top-p, top-k, min-p and repetition penalty admitted.", "XTC, bias, presence/frequency penalties and custom callables deferred until named use cases and fixtures."]
    if "Tokenizer" in name or "tokenizer_utils" in parts:
        differences.append("Textual chat only; tools/documents deferred until structured-message design; remote code skipped.")
    if leaf == "GenerationResponse":
        differences.append("Iterator event carries token, delta and final reason; logprobs/draft metadata deferred; timing and process-global peak memory intentionally unexposed.")
    return dict(entry, disposition=status, rust_path=None, planned_rust_path=path, trigger=trigger,
                semantic_differences=differences, evidence=["conformance/mlx-lm/SCHEMA.md"])


def extract(package, names_only=False):
    source = Source(package)
    entries = source.inventory()
    return {
        "schema_version": 1,
        "upstream_version": "0.31.3",
        "upstream_commit": "ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd",
        "source_kind": "installed_package_ast",
        "scope": list(MODULES),
        "inventory_rule": "Public module definitions and package-owned reexports; root __all__; public class methods/properties. External dependency reexports excluded.",
        "source_files": {source.paths[name]: digest for name, (_, digest) in sorted(source.files.items())},
        "entries": entries if names_only else [disposition(entry) for entry in entries],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=default_package())
    parser.add_argument("--names", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = json.dumps(extract(args.package, args.names), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
