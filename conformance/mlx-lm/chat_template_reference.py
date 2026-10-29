#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from manifest import ROOT, check_environment, write_json


def generate(output):
    environment = check_environment()
    from transformers import AutoTokenizer

    for name, base, both in (
        ("qwen3-2507-jinja", "qwen3-base", False),
        ("qwen3-2507-both", "qwen3-base", True),
        ("llama-jinja", "llama-base", False),
    ):
        source = ROOT / "fixtures" / base
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        config = json.loads((source / "tokenizer_config.json").read_text())
        template = config.pop("chat_template")
        if base == "qwen3-base":
            template = template.split("{%- if enable_thinking", 1)[0] + "{%- endif %}"
        if both:
            config["chat_template"] = "config template must lose"
        write_json(directory / "tokenizer_config.json", config)
        (directory / "tokenizer.json").write_bytes((source / "tokenizer.json").read_bytes())
        (directory / "chat_template.jinja").write_text(template, encoding="utf-8")
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        if tokenizer.chat_template != template:
            raise RuntimeError(f"{name}: file precedence changed")
        cases = json.loads((source / "expectations.json").read_text())["chat"]
        for case in cases.values():
            options = {
                "add_generation_prompt": case["continuation"] == "start_assistant",
                "continue_final_message": case["continuation"] == "continue_last",
                **case["template_options"],
            }
            rendered = tokenizer.apply_chat_template(case["messages"], tokenize=False, **options)
            case["rendered_utf8_hex"] = rendered.encode("utf-8").hex()
            case["token_ids"] = tokenizer.apply_chat_template(case["messages"], tokenize=True, return_dict=False, **options)
        write_json(directory / "expectations.json", {
            "environment": environment,
            "source": "transformers.AutoTokenizer.from_pretrained + apply_chat_template",
            "assets": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                       for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")},
            "chat": cases,
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "chat-template-fixtures")
    generate(parser.parse_args().output_dir)
