# mlx-lm

`mlx-lm` provides synchronous, single-request text generation for Llama and
Qwen3 using MLX. It includes strict checkpoint loading, tokenization and textual
chat templates, greedy and filtered sampling, streaming text, and model-bound KV
caches with snapshots and exact-prefix reuse. The library and its separate
[`mlx-lm` CLI](../mlx-lm-cli/README.md) are version 0.32.0 and remain
`publish = false`. Publication requires David's explicit go.

The supported runtime is macOS on Apple silicon with a working native MLX
installation. Rust 1.88 is the workspace minimum. Select a CPU or Metal device
through `mlx-rs` on the owning thread; Cargo features do not select a device.
A CPU calculation can still require Metal during native MLX initialization.
Python MLX is needed to regenerate oracle evidence, not to use the Rust library
or consume the committed fixtures.

## Capability table

| Source or operation | Supported profile |
| --- | --- |
| Local safetensors, `Model::from_dir` | Llama and Qwen3; single files and indexed shards; floating weights (including BF16) and affine quantization with strict key, shape and dtype validation. Local configuration supports full/sliding attention and none, linear or Llama3 RoPE scaling within each architecture's validated profile. |
| GGUF, `Model::from_gguf` | Llama and Qwen3; F32, F16, and Q4_0/Q4_1/Q8_0 converted affine forms. Dense, biasless models with full causal attention, full-head RoPE, and absent/none or linear scaling. No frequency-factor tensors. |
| Hub, `Model::from_hub` | Behind `hf-hub`; resolves a repository/revision to a commit and downloads selected safetensors and JSON/tokenizer assets, then uses the local loader. No remote Python code or Hub GGUF loading. |
| Generation | One synchronous iterator for text or token-ID prompts, chunked prefill, temperature/top-k/top-p/min-p sampling, repetition/presence/frequency penalties, token and text stops. |
| Cache | Full and rotating KV storage, model-bound snapshots, exact-prefix reuse. Snapshots exclude RNG, decoder and iterator state. |

The GGUF affine layouts have group size 32, 4 or 8 bits, and F16 scales and
biases. Admission examines the arrays after core conversion. Other original
encodings can succeed through core fallback conversion, but are unqualified;
there is no original-format allowlist guarantee. BF16 GGUF is not a qualified
source form.

Tiny GGUF fixtures prove self-consistency with our writer and check tensor
mapping, Q/K orientation and quantization arithmetic. Llama naming also matches
the pinned upstream exporter. Third-party compatibility is evidenced separately
by the local real-checkpoint report: TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF Q4_0
and ggml-org/Qwen3-0.6B-GGUF Q8_0. Those two files are representative coverage,
not every architecture/storage combination.

mlx-lm 0.31.3 supplies the architecture and generation oracle and a Llama-family GGUF exporter; it does NOT supply a GGUF model loader. GGUF ingestion is qualified by Python MLX container conversion plus a reviewed reference bridge and independent NumPy decoding.

The [conformance schema](../conformance/mlx-lm/SCHEMA.md),
[GGUF bridge](../conformance/mlx-lm/GGUF_BRIDGE.md) and
[compatibility ledger](../ledger/mlx-lm-parity.json) distinguish these evidence
sources. The ledger's original `NOT RUN` real-compatibility fields are historical;
the [release identity](../ledger/mlx-lm-release-identity.json) records the later
host reports without changing the protected ledger.

## GGUF tokenizer pairing

Supply a `Tokenizer` loaded with `Tokenizer::from_dir` from the exact original
model revision; embedded GGUF token strings do not reconstruct a tokenizer.
Pairing checks vocabulary and added-token ID bounds, configured BOS/EOS bounds,
and embedded special-token IDs and vocabulary spellings when available. BOS
IDs are compared only when the supplied tokenizer declares its own BOS; a
checkpoint BOS with no tokenizer BOS is not itself a mismatch.

These checks cannot establish tokenizer identity: tokenizers can share sizes
and special IDs while differing on ordinary tokens. Retain the GGUF producer
and source revision, hashes of `tokenizer.json`, `tokenizer_config.json`,
`config.json` and `generation_config.json` when present, and a canonical prompt's
original token IDs. `Tokenizer::encode` adds no special tokens. Text generation
uses the tokenizer post-processor unless the original prompt starts with its
configured BOS string.

## Hub receipts and offline use

Default features are empty. `hf-hub` enables the Hub API and optional `hf-hub`
and `sha2` dependencies; `oracle-hooks` separately enables hidden conformance
observations. Neither is activated in the base library build.

`HubOptions` selects revision, cache directory and explicit offline mode.
`Model::hub_provenance` exposes the repository, requested revision and resolved
commit. Online resolution uses anonymous requests and pins selected downloads
to that commit; an online error is not silently retried as an offline load.
Offline mode constructs no network client and accepts only a completed snapshot
with this resolver's receipt. An unreceipted third-party cache is an offline miss.

The receipt proves completeness and recorded commit identity, not tamper
resistance. Content hashes are recorded once at download completion as
provenance. Offline and reuse validation check presence, paths, sizes and the
recorded commit/receipt identity without rereading file contents. A same-size
content change is therefore undetected. A writer who can modify the cache can
also modify the receipt. Keep cache contents unchanged while loading.

## Ownership and generation

`Model`, `Generation`, `Cache` and `CacheSnapshot` are `!Send` and `!Sync`.
Create, use and drop them on one OS thread, inside the chosen device/stream
scope. `GenerationOptions` and `GenerationEvent` are owned, sendable values.
For integration into an async application, send requests to an owning worker
and receive events over bounded channels.

The complete [worker example](examples/worker.rs) creates its model and caches
inside a Metal worker thread, converts errors there, and cancels by dropping
the iterator when the receiver disconnects. Run it with a local safetensors
checkpoint:

```sh
cargo run -p mlx-lm --release --example worker -- /path/to/checkpoint
```

Generation emits prefill progress and token events containing the raw token ID,
text delta and optional finish reason. Drop performs no further inference.
An error is yielded once and fuses the iterator. Successful steps publish cache,
RNG and accepted history only after evaluation and text preparation succeed.
After generated token `j` (counted from 1), the cache represents `P + j - 1`
tokens for prompt length `P`; it has not forwarded that final sampled token.
Cache reuse requires the same model, compatible policy and an exact cached
prompt prefix with a nonempty suffix.

Rotation can retain capacity plus a prefill chunk until later updates; snapshots
retain old storage. Streaming decoding retains generated IDs and emitted text.
There is no constant-total-host-memory or zero-allocation guarantee. Chat
templates have an unresolved resource-exhaustion limitation: arbitrary model
templates are not safely sandboxed. Arbitrary cache mutation is not exposed.

## Errors

Errors retain typed boundaries and source chains; public error enums are
non-exhaustive, so callers should include a fallback match arm.

| Error | Boundary |
| --- | --- |
| `ConfigError` | Invalid or unsupported architecture, dimensions, RoPE, quantization or GGUF metadata. |
| `WeightError` | Missing, duplicate or unexpected weights/shards, shape/dtype mismatch, container parsing and strict assignment. |
| `LoadError` | Local/GGUF loading, including tokenizer pairing; wraps configuration, weights, tokenizer, template and native errors. |
| `TokenizerError`, `ChatTemplateError` | Tokenizer assets, encoding/decoding, template selection and rendering. |
| `CacheError` | Cache layout, policy, model identity, transaction or restoration. |
| `SamplingError` | Invalid sampling/penalty options, logits or empty candidate support. |
| `InferenceError` | Architecture execution and cache/native failures. |
| `GenerationError` | Prompt/stop validation, reuse and sequence limits; wraps tokenizer, sampling, cache and inference failures. Runtime evaluation exceptions remain `GenerationError::Exception`. |
| `HubError` | Repository/revision, missing assets, unsafe paths, receipt integrity, offline miss, transport or wrapped local loading. The type exists in base builds; its API-error variant requires `hf-hub`. |

## Throughput

Measured on the same machine and weights with Metal, Qwen3-0.6B-4bit, a
128-token prompt and 256 greedy tokens, Rust produced **38.2 tokens/s** against
upstream Python's **125.5 tokens/s**: a **3.3× gap**. The gap is under investigation
in its own throughput tranche. Reports are
`/Users/ci/hub/scratch/mlx-lm/t4/bench/c-baseline-metal.json` and
`/Users/ci/hub/scratch/mlx-lm/t4/bench/c-ablation-metal.json`; their identities are
recorded in the [release record](../ledger/mlx-lm-release-identity.json).

Ruling C is closed as “gap attributed elsewhere; no pipeline”. The no-lookahead
ablation measured 103.0 tokens/s and explained 9.6% of the excess time per token.
No pipeline change was admitted.

## Demand-gated work

These capabilities require a separate approved design and qualification:

- Vision, images, audio and other multimodal inputs.
- Training, adapters, LoRA and distributed execution.
- Batching, continuous generation and speculative decoding.
- A server, async generation or arbitrary custom architecture registration.
- Checkpoint conversion/upload, persisted prompt caches and quantized KV caches.
- MXFP4, BitNet, AWQ/GPTQ conversion and activation quantization.
- Structured tool/document chat messages, XTC, logit bias and custom sampling callables.

See the [completed-wave status](SKELETON.md) for tranche deliveries, report
locations and the distinction between host qualification and this freeze's
sandbox checks.

## Credits and license

MIT OR Apache-2.0, with existing source attribution retained. The wave draws on
community contributions #276 (original crates, Qwen3 and tokenizers), #352 (Hub
cache and revisions), #335 (later Qwen3.5 semantics), #287 (Gemma integration
constraints), and #356 (later model, multimodal, cache and speculative work).
They are design and compatibility references; deferred models and modalities
are not supported by this release.
