# Python mlx-lm oracle

This oracle pins `mlx-lm==0.31.3` on MLX 0.32.2, Python 3.12.14, and macOS arm64.
The interface is [SCHEMA.md](SCHEMA.md). Generation uses Python `mlx_lm` on CPU and
does not read or execute Rust code or Cargo artifacts. MLX still requires an
available Metal device during initialization on this macOS wheel.

## Environment and regeneration

From the repository root, with Python 3.12.14 available:

```sh
python3.12 -m venv conformance/.venv-mlx-lm
conformance/.venv-mlx-lm/bin/python -m pip install --require-hashes -r conformance/mlx-lm/requirements.lock
conformance/.venv-mlx-lm/bin/python conformance/mlx-lm/manifest.py
conformance/.venv-mlx-lm/bin/python conformance/mlx-lm/generate_mlx_lm.py
```

For the initial lock, or an approved dependency change, run this before the
installation step:

```sh
PYTHON=conformance/.venv-mlx-lm/bin/python sh conformance/mlx-lm/lock.sh
```

`requirements.in` contains every version from the approved
`oracle/requirements-lm.freeze`, with its NumPy entry replaced by the schema's
2.2.6 pin. `lock.sh` checks Python and architecture, downloads one compatible wheel
per pin into a temporary directory, checks dependency closure offline, and uses
`pip hash` to write `requirements.lock`. It needs package-index access. It does
not resolve newer transitive versions. Install the result with `--require-hashes`.
The separate core oracle environment at `conformance/.venv` is never modified.

`manifest.py` checks every installed dependency against the lock and input pins.
It records the environment, Transformers major version, lock digest, mlx-lm wheel
digest, and upstream commit `ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd`.
The wheel digest comes from the lock; installation with pip's hash checking is
the link between that artifact and the installed environment. Use `--lock PATH`
to select a lock and `-o PATH` to select the manifest destination.

The generator checks the same handshake before importing MLX. It creates two
complete temporary fixture trees, compares their path-and-content hashes, and
copies the six variant directories into `fixtures/` only if both trees match.
`--output-dir PATH` selects another fixture root. It does not write a manifest
or corpus catalog; run the manifest command and the separate NumPy/coordinator
workflow for those artifacts. Outputs contain no timestamps or temporary paths.

## Fixture conventions

The six variants are `llama-base`, `llama-sliding`, `llama-sharded`,
`llama-quant4`, `qwen3-base`, and `qwen3-quant4`. All use hidden size 64,
four query heads, two KV heads, head dimension 16, and vocabulary size 64.
Llama base has an untied output head; sliding Llama and Qwen3 have tied embeddings.
Sliding Llama has three layers, a five-position window, and Llama3 RoPE whose
frequencies cover low, middle, and high bands. The other variants have two layers.
The sharded fixture has exactly the base weights split across two numbered
safetensors files, with `model.safetensors.index.json` mapping their keys.

Weights use NumPy PCG64 seed 1729 and are loaded into the upstream model before
serialization. Affine variants call `mlx.nn.quantize` with group size 32 and four
bits. Every expectation is then produced from `mlx_lm.load` of the saved fixture.
Packed `quant.*.weight` expectations retain uint32; scales, biases, and all other
array expectations are float32. Both model and expectation files use the
`safetensors` library. JSON keys are sorted and non-finite JSON numbers are rejected;
filtered logprob tensors retain negative infinity in safetensors.

The canonical prompt has eight tokens. Full prefill is one model call. Chunked
prefill records all output positions from upstream `generate_step` with
`prefill_step_size` 1, 3, and 8. That function always processes the final prompt
token separately, including when the requested chunk size is 8. Generation with
zero output tokens collects prefill without its extra decode lookahead.

`decode.greedy_ids[j]` is selected from the preceding logits (the last prefill
position for j=0). `decode.step<j>.logits` is the forward result after feeding that
ID back to the model. Thus the final cache offset is T+8. Sliding caches preserve
all T positions after one-shot prefill and emit the last five positions after
decode, reordered with the upstream cache's temporal-order method. The additional
`cache` JSON object records processed offsets and half-open retained ranges.

Sampling resets `mx.random.seed(1729)` before every case and uses the CPU generation
stream. Each case records eight IDs from upstream `generate_step`. Filtered
logprobs come from its first returned logprobs, passed through upstream top-p,
min-p, and top-k functions before temperature scaling. A combined case pins filter
ordering. Upstream 0.31.3's repetition processor sees only the final prompt token
on its first sampling call, because prefill bypasses the processor. The fixture
records `processor_token_histories` to make that behavior explicit.

The local WordLevel tokenizer includes Unicode, special tokens, and textual chat
templates using Llama-3 or Qwen3 delimiters. Qwen3 renders with thinking disabled.
All three chat continuations record rendered UTF-8 bytes and tokenizer IDs.
Generation EOS values differ from config and tokenizer defaults to exercise
precedence. `decode.token_ids` and `text_deltas` record every upstream streaming
response, including the final detokenizer flush. Additional `length_case` and
`stop_case` records use an empty EOS set and the first greedy ID as EOS,
respectively, to guarantee both finish paths are covered.

## Loading errors and protection

Each error recipe is applied to a temporary fixture copy and attempted with the
Python loader. The recorded `python_result` is an exception class or `accepted`;
it is separate from the expected Rust class. Python accepts duplicate shard keys
by overwriting them, while the approved Rust boundary is strict.

| Mutation | Expected Rust class |
| --- | --- |
| Missing indexed shard | `WeightError::MissingShard` |
| Duplicate tensor across shards | `WeightError::DuplicateTensor` |
| Wrong query-projection shape | `WeightError::ShapeMismatch` |
| Unsupported RoPE type | `ConfigError::UnsupportedRope` |
| Unknown model type | `ConfigError::UnsupportedArchitecture` |
| Unsupported affine bit count | `ConfigError::UnsupportedQuantization` |

The design explicitly names `UnsupportedRope`; the other variant names above
spell out its required stable failure categories for the later Rust loader.

The [core oracle boundary](../README.md#oracle-separation) protects conformance
inputs, generators, dependency pins, manifests, fixtures, tolerance policies, and
comparators. Updates belong in a separate oracle change, reviewed independently
from inference implementation changes. Rust conformance tests consume committed
fixtures offline and do not regenerate or bless them.
