# Completed wave: mlx-lm 0.32.0

Tranches 1–4b delivered the synchronous Llama/Qwen3 library, alternate sources,
consumer CLI and qualification harnesses. Both packages keep `publish = false`;
publication is David's separate decision. The
[capability table](README.md#capability-table) defines supported sources and
operations. The [support decisions](SUPPORT.md) retain the compatibility limits
and release conditions.

| Tranche | Delivered |
| --- | --- |
| 1: oracle and gates | Pinned Python environment, independent NumPy reference, tiny fixtures, frozen expectations, comparator/mutation qualification and oracle-boundary gates. |
| 2: typed model foundation | Typed configuration/errors, Llama/Qwen3 architectures, strict safetensors/shard/affine loading, tokenizer/chat ownership, full/rotating cache storage and keyed assignment. The utilities were folded into the library. |
| 3: generation and sampling | One transactional generation iterator, streaming text, stop policy, sampling and penalties, exact-prefix cache reuse, snapshots, rollback tests, FFI workloads and the owning-thread worker example. |
| 4a: GGUF ingestion | `Model::from_gguf`, GGUF metadata normalization, architecture tensor maps and shared weight backing; ten fixtures covering two architectures and five storage forms, reference-bridge/NumPy qualification, malformed recipes and effective mutation checks. |
| 4b: Hub, CLI and freeze | Optional Hub resolver with completion receipts and commit provenance; public-API CLI with versioned output; offline Hub/CLI contract goldens; compiled feature consumers; local real-checkpoint and benchmark harnesses; release documentation. |

The loading implementations are `Model::from_dir`, `Model::from_gguf`,
`Model::from_hub` (under `hf-hub`), `WeightManifest::from_gguf` and the
Llama/Qwen3 `map_gguf_key` implementations. There are no remaining loader
placeholder rows.

## Evidence and report locations

- [Conformance README](../conformance/mlx-lm/README.md) and
  [schema](../conformance/mlx-lm/SCHEMA.md): fixture contracts and reproduction
  commands, including local real checkpoints and the ruling-C protocol.
- [GGUF bridge](../conformance/mlx-lm/GGUF_BRIDGE.md),
  [GGUF cases](../conformance/mlx-lm/gguf_cases.json) and
  [save qualification](../conformance/qualification/gguf-save.json): separate
  container conversion, model arithmetic and writer evidence. Tiny fixtures do
  not establish third-party compatibility.
- [Support decisions](SUPPORT.md): supported subset, generation/cache contracts,
  GGUF evidence limits and demand-gated work.
- [Hub cases](../conformance/mlx-lm/hub_cases.json) and
  [CLI cases](../conformance/mlx-lm/cli_cases.json): offline contracts.
  `cargo xtask verify-lm-features` compiles positive and negative
  external consumers; separate dependency-tree checks verify base-build isolation.

The host's real-checkpoint inputs are
`/Users/ci/hub/assets/mlx-lm-checkpoints/real-checkpoints.json`; `bound.json` in
that directory binds `python-real-report.json`. The Rust result is
`/Users/ci/hub/scratch/mlx-lm/t4/real-rust3.log`. It reports all six CPU cases
with exact equality for 32 greedy IDs. Metal passed all six under ruling Q:
Llama BF16 is tie-limited at step 13 (gap 0.125), Qwen3 BF16 at step 1
(gap 0.125), and Qwen3 four-bit at step 1 (gap 0); Llama four-bit and both
GGUF cases match all 32 IDs. These are prior host runs, not sandbox reruns.

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

## Open work and verification limits

GPU generation with full caches queues one token ahead and submits only sampling
roots. Each decoder's updated K/V feeds attention and logits. Cache objects stay
alive; committed lengths, RNG and text advance only when a token is accepted.
CPU and rotating-cache decode remain synchronous. This measurement does not
establish CPU throughput parity or the full real-checkpoint matrix.

The [demand-gated list](README.md#demand-gated-work) remains open: vision/audio
and other multimodal input, LoRA/training/adapters, distributed execution,
batching, speculative decoding, a server, async generation, additional
architectures and the other listed cache/conversion/sampling capabilities.
Arbitrary chat templates retain the unresolved resource-exhaustion limitation.

These historical host reports do not establish qualification at the current
source commit. Publication still requires fresh MSRV and package/archive
consumer checks against the intended dependency archives, plus David's explicit
authorization; `publish = false` stays set. Current bump checks and their required
execution evidence are documented in [VERIFICATION.md](../VERIFICATION.md).
