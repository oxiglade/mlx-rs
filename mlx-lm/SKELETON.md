# Completed wave: mlx-lm 0.32.0

Tranches 1–4b delivered the synchronous Llama/Qwen3 library, alternate sources,
consumer CLI and qualification harnesses. Both packages keep `publish = false`;
publication is David's separate decision. The
[capability table](README.md#capability-table) defines supported sources and
operations. The [release identity](../ledger/mlx-lm-release-identity.json)
records the source, dependency and evidence hashes and this freeze's checks.

| Tranche | Delivered |
| --- | --- |
| 1: oracle and gates | Pinned Python environment, independent NumPy reference, tiny fixtures, frozen expectations, comparator/mutation qualification, parity ledger and API/oracle-boundary gates. |
| 2: typed model foundation | Typed configuration/errors, Llama/Qwen3 architectures, strict safetensors/shard/affine loading, tokenizer/chat ownership, full/rotating cache storage and keyed assignment. The utilities were folded into the library. |
| 3: generation and sampling | One transactional generation iterator, streaming text, stop policy, sampling and penalties, exact-prefix cache reuse, snapshots, rollback tests, FFI workloads and the owning-thread worker example. |
| 4a: GGUF ingestion | `Model::from_gguf`, GGUF metadata normalization, architecture tensor maps and shared weight backing; ten fixtures covering two architectures and five storage forms, reference-bridge/NumPy qualification, malformed recipes and effective mutation checks. |
| 4b: Hub, CLI and freeze | Optional Hub resolver with completion receipts and commit provenance; public-API CLI with versioned output; offline Hub/CLI contract goldens; compiled feature consumers; local real-checkpoint and benchmark harnesses; release docs and identity record. |

The loading implementations are `Model::from_dir`, `Model::from_gguf`,
`Model::from_hub` (under `hf-hub`), `WeightManifest::from_gguf` and the
Llama/Qwen3 `map_gguf_key` implementations. There are no remaining loader
placeholder rows. The private `NotYetImplemented` formatter is absent from
`mlx-lm/src`; this freeze adds only crate-level documentation there.

## Evidence and report locations

- [Conformance README](../conformance/mlx-lm/README.md) and
  [schema](../conformance/mlx-lm/SCHEMA.md): fixture contracts and reproduction
  commands, including local real checkpoints and the ruling-C protocol.
- [GGUF bridge](../conformance/mlx-lm/GGUF_BRIDGE.md),
  [GGUF cases](../conformance/mlx-lm/gguf_cases.json) and
  [save qualification](../conformance/qualification/gguf-save.json): separate
  container conversion, model arithmetic and writer evidence. Tiny fixtures do
  not establish third-party compatibility.
- [Parity ledger](../ledger/mlx-lm-parity.json) and
  [API baseline](../ledger/mlx-lm-api-baseline.json): compatibility dispositions
  and source API inventory. The protected ledger retains historical `NOT RUN`
  real-compatibility fields; the release record identifies the later host runs.
- [Hub cases](../conformance/mlx-lm/hub_cases.json) and
  [CLI cases](../conformance/mlx-lm/cli_cases.json): offline contracts.
  `cargo run -p xtask -- verify-lm-features` compiles positive and negative
  external consumers; separate dependency-tree checks verify base-build isolation.

The host's real-checkpoint inputs are
`/Users/ci/hub/assets/mlx-lm-checkpoints/real-checkpoints.json`; `bound.json` in
that directory binds `python-real-report.json`. The Rust result is
`/Users/ci/hub/scratch/mlx-lm/t4/real-rust3.log`. It reports all six CPU cases
with exact equality for 32 greedy IDs. Metal passed all six under ruling Q:
Llama BF16 is tie-limited at step 13 (gap 0.125), Qwen3 BF16 at step 1
(gap 0.125), and Qwen3 four-bit at step 1 (gap 0); Llama four-bit and both
GGUF cases match all 32 IDs. These are prior host runs, not sandbox reruns.

Ruling C is closed by `t4/DECISIONS.md` ruling S as
“gap attributed elsewhere; no pipeline”. The Metal reports are
`/Users/ci/hub/scratch/mlx-lm/t4/bench/c-baseline-metal.json` and
`/Users/ci/hub/scratch/mlx-lm/t4/bench/c-ablation-metal.json`. Their generated
open-status strings predate that ruling; the release record preserves both
report hashes and the ruling that closes the decision. CPU benchmark context
was not completed under ruling R because a 256-token generation took about
1182 seconds. No pipeline was admitted.

## Open work and verification limits

Measured on the same machine and weights with Metal, Qwen3-0.6B-4bit and
256 greedy tokens, Rust produced 38.2 tokens/s against upstream's 125.5, a
3.3× gap under investigation in its own throughput tranche (ruling T).
The no-lookahead ablation produced 103.0 tokens/s and explained 9.6% of the
excess time per token. The report paths above contain the measurements.

The [demand-gated list](README.md#demand-gated-work) remains open: vision/audio
and other multimodal input, LoRA/training/adapters, distributed execution,
batching, speculative decoding, a server, async generation, additional
architectures and the other listed cache/conversion/sampling capabilities.
Arbitrary chat templates retain the unresolved resource-exhaustion limitation.

This freeze's environment has no Metal or Python MLX. Compilation, docs,
external feature consumers and pure tests are recorded separately from
unexecuted inference, real-checkpoint, benchmark and allocator tests in the
release identity. The requested feature/docs matrix passes. The wider API-baseline check finds
pre-existing Hub additions missing from `ledger/mlx-lm-api-baseline.json`;
refreshing that baseline is outside this item's ownership. Offline package
verification cannot download the `mlx-rs 0.32.0` dependency archive from the
offline cache. These
remain publication prerequisites, together with Rust 1.88 qualification and
archive consumer checks; `publish = false` stays set.
