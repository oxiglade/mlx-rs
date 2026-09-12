import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from numpy_reference.common import load_weights, read_json, write_safetensors
from numpy_reference.llama import Llama
from numpy_reference.qwen3 import Qwen3


ROOT = Path(__file__).resolve().parents[1]
NUMPY_VERSION = "2.2.6"
DECODE_STEPS = 8


def fixture_directories(paths):
    fixtures = sorted((Path(path).resolve() for path in paths), key=lambda path: path.name)
    if not fixtures:
        raise ValueError("no fixture directories supplied or discovered")
    if any(not path.is_dir() for path in fixtures):
        raise ValueError("fixture paths must be directories")
    if len({path.name for path in fixtures}) != len(fixtures):
        raise ValueError("fixture directory names must be unique")
    return fixtures


def reference_tensors(fixture):
    fixture = Path(fixture)
    config = read_json(fixture / "config.json")
    expectations = read_json(fixture / "expectations.json")
    architectures = {"llama": Llama, "qwen3": Qwen3}
    model = architectures[config["model_type"]](config, load_weights(fixture))
    prompt = expectations["prefill"]["token_ids"]
    if len(prompt) != expectations["prefill"]["T"]:
        raise ValueError("prefill T disagrees with token_ids")
    logits = model.forward(np.array([prompt], dtype=np.int32))
    tensors = {"prefill.full.logits": logits}
    tensors.update(model.cache_tensors("after_prefill"))
    for step in range(DECODE_STEPS):
        token = np.argmax(logits[:, -1, :], axis=-1)[:, None]
        logits = model.forward(token)
        tensors[f"decode.step{step}.logits"] = logits[:, -1, :]
    tensors.update(model.cache_tensors("after_decode"))
    return tensors


def main():
    parser = argparse.ArgumentParser(
        description="Generate independent f32 NumPy decoder expectations"
    )
    parser.add_argument("fixtures", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "numpy_reference" / "out")
    args = parser.parse_args()
    if np.__version__ != NUMPY_VERSION:
        parser.error(f"requires numpy {NUMPY_VERSION}, got {np.__version__}")
    discovered = (path for path in (ROOT / "fixtures").glob("*") if path.is_dir())
    fixtures = fixture_directories(args.fixtures or discovered)
    for fixture in fixtures:
        path = args.output_dir / f"{fixture.name}.safetensors"
        write_safetensors(path, reference_tensors(fixture))
        print(path)


if __name__ == "__main__":
    main()
