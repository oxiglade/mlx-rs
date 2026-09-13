"""Materialize malformed GGUF cases from the qualified fixtures, without MLX."""

import hashlib
import json
import math
import tempfile
from pathlib import Path

import gguf_writer
from numpy_reference.gguf import read


def materialize(source, destination, recipe):
    metadata, kinds, descriptors, _, _ = read(source)
    raw = Path(source).read_bytes()
    # The parser's last tensor ends at EOF for the deterministic writer.
    sizes = {name: (shape[0] * shape[1] // 32 * {2: 18, 3: 20, 8: 34}[kind]
                    if kind in (2, 3, 8) else math.prod(shape) * (4 if kind == 0 else 2))
             for name, (shape, kind, _) in descriptors.items()}
    end = max(offset + sizes[name] for name, (_, _, offset) in descriptors.items())
    base = len(raw) - end
    names = {value: key for key, value in gguf_writer.TYPES.items()}
    tensors = {name: (shape, names[kind], raw[base + offset:base + offset + sizes[name]])
               for name, (shape, kind, offset) in descriptors.items()}
    metadata = {key: (kinds[key], value) for key, value in metadata.items()}
    operation, key = recipe["operation"], recipe.get("key")
    if operation == "remove_metadata":
        del metadata[key]
    elif operation == "set_metadata":
        metadata[key] = (recipe["type"], recipe["value"])
    elif operation == "remove_tensor":
        del tensors[key]
    elif operation in ("copy_tensor", "rename_tensor"):
        tensors[recipe["target"]] = tensors[key]
        if operation == "rename_tensor":
            del tensors[key]
    elif operation == "truncate_rows":
        shape, kind, data = tensors[key]
        rows = recipe["rows"]
        tensors[key] = ((rows, shape[1]), kind, data[:len(data) // shape[0] * rows])
    elif operation == "truncate_columns":
        shape, kind, data = tensors[key]
        stride = len(data) // shape[0]
        columns = recipe["columns"]
        width = stride * columns // shape[1]
        tensors[key] = ((shape[0], columns), kind, b"".join(data[i:i + width] for i in range(0, len(data), stride)))
    else:
        raise ValueError(operation)
    gguf_writer.write(destination, metadata, tensors)


def freeze_hashes(fixtures, cases_path):
    cases_path = Path(cases_path)
    document = json.loads(cases_path.read_text())
    with tempfile.TemporaryDirectory(prefix="gguf-recipes-") as temporary:
        output = Path(temporary) / "recipe.gguf"
        for recipe in document["cases"]:
            if recipe["operation"] == "load_core":
                data = (cases_path.parent / "fixtures" / recipe["base"]).read_bytes()
            else:
                materialize(Path(fixtures) / recipe["base"] / "model.gguf", output, recipe)
                data = output.read_bytes()
            recipe["materialized_sha256"] = hashlib.sha256(data).hexdigest()
    # Keep one recipe per line so changes to the protected contract remain reviewable.
    lines = ["{", '  "schema_version": ' + str(document["schema_version"]) + ",",
             '  "scope": ' + json.dumps(document["scope"]) + ","]
    for field in ("cases", "tokenizer_mutations"):
        lines.append(f'  "{field}": [')
        lines.extend("    " + json.dumps(item, allow_nan=False) + ("," if i + 1 < len(document[field]) else "")
                     for i, item in enumerate(document[field]))
        lines.append("  ]" + ("," if field == "cases" else ""))
    cases_path.write_text("\n".join([*lines, "}", ""]))
