# Supported capabilities and decisions

The [capability table](README.md#capability-table) defines the supported Llama
and Qwen3 profiles: strict local safetensors/shard/affine loading, the qualified
GGUF subset, optional Hub loading, synchronous generation, and full/rotating KV
caches. This is a supported subset of Python mlx-lm, not Python API equivalence.
Public helpers and new cache policies require an admitted model or consumer use.

## Generation and cache contracts

Generation exposes fallible prefill progress and token/text/finish events.
Timing, process-global peak memory, logprobs and draft metadata are not public
iterator output. There is no decode lookahead. Stop strings use exact UTF-8,
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

Ruling S closed ruling C as “gap attributed elsewhere; no pipeline”. The measured
Metal throughput gap remains a separate investigation under ruling T; the
[benchmark measurements](README.md#throughput) do not authorize a pipeline change
or claim CPU throughput parity. Both packages retain `publish = false`.
Publication requires David's authorization, MSRV qualification and verification
of packaged external consumers against the intended dependency archives.
