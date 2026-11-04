# mlx-lm

`mlx-lm` provides synchronous, single-request text generation for Llama and
Qwen3 using MLX. It includes strict checkpoint loading, tokenization and textual
chat templates, greedy and filtered sampling, streaming text, and model-bound KV
caches with snapshots and exact-prefix reuse. The library and its separate
[`mlx-lm` CLI](../mlx-lm-cli/README.md) are version 0.33.0 and remain
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
[support decisions](SUPPORT.md) distinguish these evidence sources.
[SKELETON.md](SKELETON.md#evidence-and-report-locations) records historical host
results; current bump admission requires fresh executed configuration reports.

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
text delta and optional finish reason. Drop queues no further inference; an already
queued GPU step still finishes, and a later cache read waits for it.
Pipelined generation runs on the stream current at `generate()`.
An error is yielded once and fuses the iterator. Successful steps publish cache,
RNG and accepted history only after evaluation and text preparation succeed.
After generated token `j` (counted from 1), the cache represents `P + j - 1`
tokens for prompt length `P`; it has not forwarded that final sampled token.
Cache reuse requires the same model, compatible policy and an exact cached
prompt prefix with a nonempty suffix.

An abandoned step preserves committed tokens and K/V contents up to the committed
length. Before evaluation or submission, it restores the original handles. After
successful evaluation or submission, full caches keep the new buffers at the
committed length. Unused tail contents and capacity may change. Rotating caches
copy handles because they overwrite committed positions. Full caches append in
place on GPU streams. Reusing a rolled-back suffix or changing streams waits for
pending readers. Full-cache snapshots evaluate the source arrays, allocate full-capacity K/V
storage for each full layer with spare capacity, and copy only the committed prefixes.
Full layers without spare capacity and rotating layers may share storage until an
update replaces it. Copies for all layers are evaluated together. For copied layers,
allocation and initialization scale with capacity; prefix copying scales with committed
length. Restore consumes snapshot storage
without copying or evaluating it. CPU and rotating caches use `slice_update`.

On an M5 Max, measured decode throughput is level at 128 tokens, about 7% higher
at ~4k and 0.4% lower at ~16k.

Failed evaluation or submission invalidates only the cache involved. This includes
failed speculative submission even if generation stops at the preceding token.
Reads, updates, snapshots, restores and reuse return `CacheError::Invalidated`.
Call `Cache::reset()` and prefill again. Reset discards all cached tokens and buffers.
Existing snapshots and other caches restored from them remain valid.
A sampling-only failure leaves an untouched cache valid. A snapshot copy failure
also leaves the source valid; a failure evaluating the source graph invalidates it.

Rotation can retain capacity plus a prefill chunk until later updates; snapshots
retain independent full-cache storage or old rotating storage. Streaming decoding retains generated IDs and emitted text.
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

Sampling checks raw, processed, filtered and scaled values in that order after a
successful joint evaluation. Invalid values precede empty support within each
stage. If evaluation fails, its original `GenerationError::Exception` wins over
value errors. The failed graph is never inspected or evaluated again.

## Throughput

C closed: qualified pipeline admitted (2026-11-02, M5 Max Metal):
606.95 Rust versus 617.93 Python tok/s, ratio 0.9821
[0.9810, 0.9837] (95% paired CI). Recovery is
93.44% [93.18%, 94.14%], with observed Rust–Python excess as denominator.
Using the full Python lookahead ablation effect as denominator, recovery is
67.30% [65.99%, 68.56%] (95% paired CIs).
The CPU path is unchanged by construction, satisfying the other-device ≤5% regression criterion.
Qualification used paired alternation on a shared host, deviating from the protocol's idle-host requirement.

The measurement uses Qwen3-0.6B-4bit, a 128-token prompt, 256 greedy tokens,
three warm-ups and ten alternating Rust/Python pairs per group. The fresh #444
synchronous baseline measured 481.82 Rust versus 615.99 Python tok/s;
Python without lookahead measured 444.49 tok/s. The baseline and ablation
are retained from the prior qualification; the candidate rerun uses contemporaneous
Python. The intervals describe repeatability within these runs.

GPU generation with full caches queues one token ahead and submits only sampling
roots. Each decoder's updated K/V feeds attention and logits. Cache objects stay
alive; committed lengths, RNG and text advance only when a token is accepted.
CPU and rotating-cache decode remain synchronous. This measurement does not
establish CPU throughput parity or the full real-checkpoint matrix.

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
locations and the limits of historical host qualification.

## Credits and license

MIT OR Apache-2.0, with existing source attribution retained. The wave draws on
community contributions #276 (original crates, Qwen3 and tokenizers), #352 (Hub
cache and revisions), #335 (later Qwen3.5 semantics), #287 (Gemma integration
constraints), and #356 (later model, multimodal, cache and speculative work).
They are design and compatibility references; deferred models and modalities
are not supported by this release.
