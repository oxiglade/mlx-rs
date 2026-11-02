# Supported capabilities and decisions

The [capability table](README.md#capability-table) defines the supported Llama
and Qwen3 profiles: strict local safetensors/shard/affine loading, the qualified
GGUF subset, optional Hub loading, synchronous generation, and full/rotating KV
caches. This is a supported subset of Python mlx-lm, not Python API equivalence.
Public helpers and new cache policies require an admitted model or consumer use.

## Generation and cache contracts

Generation exposes fallible prefill progress and token/text/finish events.
Timing, process-global peak memory, logprobs and draft metadata are not public
iterator output. GPU full-cache generation queues one token ahead. CPU and
rotating-cache generation remain synchronous. Stop strings use exact UTF-8,
hold possible matching suffixes and exclude the match; a final-flush match wins
over the length limit. Empty stop strings are errors. Text-delta parity covers
named WordLevel fixtures, and an unstable final decode is `InvalidPrefix`.

Sampling admits greedy, temperature, top-k/top-p/min-p and repetition/presence/
frequency penalties. Each generation owns its optional `u64` seed and private
RNG; greedy creates none. Exact stochastic token IDs are a CPU, pinned-oracle
claim. Penalties run in repetition, presence, frequency order with independent
trailing contexts. Presence/frequency coefficients are finite and signed; zero
is identity. Repetition penalties must be finite and positive. History
starts at the final prompt token and includes accepted generated tokens, excluding
earlier prompt tokens and any reused prefix. Presence deduplicates; frequency
counts occurrences.

Sampling checks raw, processed, filtered and scaled values in that order after a
successful joint evaluation. Invalid values precede empty support within each
stage. If evaluation fails, its original `GenerationError::Exception` wins over
value errors. The failed graph is never inspected or evaluated again.

An abandoned step preserves committed tokens and K/V contents up to the committed
length. Before evaluation or submission, it restores the original handles. After
successful evaluation or submission, full caches keep the new buffers at the
committed length. Unused tail contents and capacity may change. Rotating caches
copy handles because they overwrite committed positions. Live snapshots can
prevent buffer donation.

Failed evaluation or submission invalidates only the cache involved. This includes
failed speculative submission even if generation stops at the preceding token.
Reads, updates, snapshots, restores and reuse return `CacheError::Invalidated`.
Call `Cache::reset()` and prefill again. Reset discards all cached tokens and buffers.
Existing snapshots and other caches restored from them remain valid.
A sampling-only failure leaves an untouched cache valid.

Cache reuse requires the full intended prompt, an exact represented prefix,
the same model instance and at least one uncached token. No public trim API is
exposed; wrapped rotating caches reject trim without changing storage or position.
Text prompts check the original string for configured BOS content before special
token encoding; whitespace before BOS can insert a second BOS. Chat is textual,
with no remote Python code. These contracts are exercised by the
[parity suites](tests/parity), [generation tests](src/model/tests.rs) and
[cache tests](src/cache).

## GGUF evidence

Python mlx-lm 0.31.3 provides architecture/generation expectations and a Llama
exporter, not a GGUF model loader. Ingestion uses Python MLX container conversion,
a reviewed [bridge](../conformance/mlx-lm/GGUF_BRIDGE.md) and independent NumPy
block decoding. Tiny fixtures qualify mapping, Q/K orientation and quantization
arithmetic, not third-party compatibility. F32, F16, Q4_0, Q4_1 and Q8_0 are the
qualified source forms; fallback successes do not qualify other encodings.
The [checkpoint contract](../conformance/mlx-lm/bump-oracle.json) includes
representative TinyLlama Q4_0 and Qwen3 Q8_0 files. CPU requires exact greedy IDs;
Metal comparisons allow only the recorded dtype-ULP tie rule. Historical host
results in [SKELETON.md](SKELETON.md) do not establish execution at the current SHA.

## Deferred work and release decisions

The [demand-gated list](README.md#demand-gated-work) requires separate designs
and fixtures, including batching, speculative decoding, persisted/quantized
caches, structured tools/reasoning channels, adapters and checkpoint conversion.
XTC, logit bias and custom sampling callables need named uses and sampler fixtures.
Model diagnostics need a public statistics contract; process-wide resource
controls need an explicit design. Arbitrary chat-template resource exhaustion
remains unresolved.

C closed: qualified pipeline admitted (2026-11-02, M5 Max Metal):
606.95 Rust versus 617.93 Python tok/s, ratio 0.9821
[0.9810, 0.9837] (95% paired CI). Recovery is
93.44% [93.18%, 94.14%], with observed Rust–Python excess as denominator.
Using the full Python lookahead ablation effect as denominator, recovery is
67.30% [65.99%, 68.56%] (95% paired CIs).
The CPU path is unchanged by construction, satisfying the other-device ≤5% regression criterion.
Qualification used paired alternation on a shared host, deviating from the protocol's idle-host requirement.

GPU generation with full caches queues one token ahead and submits only sampling
roots. Each decoder's updated K/V feeds attention and logits. Cache objects stay
alive; committed lengths, RNG and text advance only when a token is accepted.
CPU and rotating-cache decode remain synchronous. This measurement does not
establish CPU throughput parity or the full real-checkpoint matrix.

See the [measurement protocol](README.md#throughput). Both packages retain
`publish = false`. Publication requires David's authorization, MSRV qualification and verification
of packaged external consumers against the intended dependency archives.
