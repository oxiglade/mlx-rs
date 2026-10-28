#!/usr/bin/env python3
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models

from manifest import ROOT, write_json


def filter_case(stops, deltas, final_delta=""):
    if any(not stop for stop in stops):
        raise ValueError("EmptyStopString")
    held = ""
    stopped = False

    def push(delta):
        nonlocal held, stopped
        if stopped:
            return ""
        pending = held + delta
        matches = [pending.index(stop) for stop in stops if stop in pending]
        if matches:
            stopped = True
            held = ""
            return pending[:min(matches)]
        prefixes = [pending[-length:] for length in range(1, len(pending) + 1)
                    if any(stop.startswith(pending[-length:]) and length < len(stop) for stop in stops)]
        held = max(prefixes, key=len, default="")
        return pending[:len(pending) - len(held)]

    events = []
    for delta in deltas:
        if stopped:
            break
        output = push(delta)
        events.append({"text": output, "stopped": stopped, "held": held})
    flushed = None
    if not stopped:
        flushed = push(final_delta)
        if not stopped:
            flushed += held
            held = ""
    return {"stops": stops, "deltas": deltas, "final_delta": final_delta,
            "events": events, "finish_flush": flushed, "stopped": stopped,
            "finish_reason": "stop" if stopped else "length"}


def decoder_cases():
    results = {}
    for name, vocab, decoder, sequences in (
        ("byte_fallback", {"<unk>": 0, "<0xC3>": 1, "<0xA9>": 2, "<0x20>": 3}, decoders.ByteFallback(),
         {"complete": [1, 2, 3], "incomplete": [1], "empty": [], "rewrite": [1, 2, 1]}),
        ("byte_level", {"<unk>": 0, "Ã": 1, "©": 2, "Ġ": 3}, decoders.ByteLevel(),
         {"incomplete_tail": [1, 2, 1]}),
    ):
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tokenizer.decoder = decoder
        for case, ids in sequences.items():
            # Only these named stable fragments are admitted; Python streaming boundaries are not a contract.
            emitted = "é " if case == "complete" else "é" if len(ids) == 3 else ""
            final = tokenizer.decode(ids, skip_special_tokens=False)
            stable = final.startswith(emitted)
            results[f"{name}_{case}"] = {
                "tokenizer": json.loads(tokenizer.to_str()), "token_ids": ids,
                "decoded_utf8_hex": final.encode("utf-8").hex(), "emitted_prefix": emitted,
                "preserves_prefix": stable, "finish_flush": final[len(emitted):] if stable else None,
                "expected_error": None if stable else "DecodeStreamError::InvalidPrefix",
            }
    if results["byte_fallback_rewrite"]["preserves_prefix"]:
        raise RuntimeError("pinned ByteFallback rewrite no longer rewrites the emitted prefix")
    return results


def reference():
    version = importlib.metadata.version("tokenizers")
    if version != "0.23.2":
        raise RuntimeError(f"requires tokenizers 0.23.2, got {version}")
    recipes = {
        "exact": (["END"], ["END"]),
        "split": (["END"], ["hello E", "N", "D"]),
        "mismatch_release": (["END"], ["hello E", "x!"]),
        "unmatched_terminal_prefix": (["END"], ["hello EN"]),
        "multiple": (["STOP", "END"], ["hi END STOP"]),
        "overlapping": (["aba", "ab"], ["zaba"]),
        "overlap_abc_one_delta": (["abc", "b"], ["abc"]),
        "overlap_abc_split": (["abc", "b"], ["ab", "c"]),
        "same_delta_trailing": (["END"], ["hello END hidden"]),
        "non_ascii": (["雪é"], ["café 雪", "é tail"]),
        "composed_match": (["é"], ["caf", "é"]),
        "decomposed_mismatch": (["é"], ["cafe", "\u0301"]),
        "decomposed_match": (["e\u0301"], ["cafe", "\u0301"]),
        "duplicate_stops": (["END", "END"], ["END"]),
        "no_stops": ([], ["café", " 雪"]),
    }
    cases = {name: filter_case(*recipe) for name, recipe in recipes.items()}
    cases["final_flush_match"] = filter_case(["END"], ["hello E"], "ND tail")
    cases["final_flush_mismatch"] = filter_case(["END"], ["hello E"], "x")
    return {"schema_version": 1,
            "provenance": {"tokenizers": version, "generator": "text_reference.py@" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "stop_strings": cases, "decoders": decoder_cases(),
            "errors": {"empty_stop": {"stops": [""], "expected_error": "EmptyStopString"}}}


def main():
    parser = argparse.ArgumentParser(description="Independent exact UTF-8 stop-string and tokenizer final-decode reference; no MLX.")
    parser.add_argument("--check", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "fixtures/llama-base/text_cases.json")
    args = parser.parse_args()
    expected = reference()
    if args.check:
        actual = json.loads(args.check.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit(f"text reference differs: {args.check}")
        print(f"text reference matches: {args.check}")
    else:
        write_json(args.output, expected)


if __name__ == "__main__":
    main()
