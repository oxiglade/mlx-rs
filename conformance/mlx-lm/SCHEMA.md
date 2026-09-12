# mlx-lm oracle fixtures: layout and expectation schema (tranche 1 contract)

This file is the interface between the three tranche-1 work items (Python mlx_lm generator,
NumPy reference generator, Rust comparator). Changing it is an `oracle-change:` commit.

## Environment
- Oracle venv: `conformance/.venv-mlx-lm` built from `conformance/mlx-lm/requirements.lock`
  (pip hashes; python 3.12.14; mlx-lm==0.31.3; mlx==0.32.2; mlx-metal==0.32.2; numpy==2.2.6;
  transformers 5.x). `manifest.json` records the handshake (arch=arm64, versions, upstream
  commit ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd, sha256 of the mlx-lm wheel).
- The core oracle venv `conformance/.venv` is never touched.

## Fixture directories
`conformance/mlx-lm/fixtures/<model>-<variant>/` where `<model>` in {llama, qwen3} and
`<variant>` in {base, quant4} (quant4 = affine, group_size 32, bits 4; Llama base also carries
a sliding-window layer pattern and llama3 rope in the `llama-sliding` variant). Each directory:
- `config.json` — HF-style config as mlx_lm expects (tiny: hidden<=64, layers 2..3, vocab<=64).
- `model.safetensors` (+ `model.safetensors.index.json` for the one `llama-sharded` variant).
- `tokenizer.json`, `tokenizer_config.json` (chat_template included), `generation_config.json`
  (eos_token_id present to exercise precedence).
- `expectations.json` — scalars, ids, strings, error cases, provenance (see below).
- `expectations.safetensors` — every array expectation, f32 unless stated, keys below.
- `inputs.json` — the prompt strings, token id lists, seeds, sampler cases, chunk sizes used.

## Shared math pins
- RoPE llama3 smoothing follows mlx_lm 0.31.3 `rope_utils.py` exactly (angle = pos / f_i with
  f_i = theta^(2i/d); low band f_i *= factor; mid band f_i /= ((1-s)/factor + s)); both
  generators derive from that file's math, the NumPy one without importing it.
- Affine dequant per /Users/ci/hub/scratch/mlx-lm/t1/affine-packing.md (recorded in-repo as
  conformance/mlx-lm/numpy_reference/PACKING.md by the NumPy item).

## Array keys in expectations.safetensors
- `prefill.full.logits` — `[1, T, V]` for the canonical prompt, one-shot prefill.
- `prefill.chunk<N>.logits` — same tensor produced with prefill chunk size N, for N in
  {1, 3, T}; N=3 must be a non-divisor of T.
- `cache.after_prefill.layer<i>.keys` / `.values` — logical K/V `[1, kv_heads, T, head_dim]`
  after a ONE-SHOT prefill, in temporal (position) order. Sliding-window layers emit all T
  positions here too: mlx's RotatingKVCache keeps the whole first concat and only trims on
  later updates, and both oracles pin that exact state rather than an idealized window.
- `decode.step<j>.logits` — `[1, V]` for greedy decode steps j in 0..8.
- `cache.after_decode.layer<i>.keys` / `.values` — logical K/V after the 8 decode steps, in
  temporal order: full layers all T+8 positions; sliding layers (window W) the last
  min(W, T+8) positions. mlx stores the rotating buffer in rotated memory order; the Python
  generator must emit temporal order (reorder by the cache's rotation index), and the NumPy
  reference keeps a logical temporal cache and emits its tail.
- `sampling.<case>.filtered_logprobs` — `[V]` logprobs after the case's filters (top_p, min_p,
  top_k) at the first decode position, before temperature division; -inf for filtered entries.
- `quant.<param>` — for quant4 variants: the packed weight/scales/biases arrays of two named
  parameters, to pin layout.

## expectations.json (JSON, sorted keys, no NaN)
{
  "schema_version": 1,
  "provenance": { "python": "3.12.14", "mlx_lm": "0.31.3", "mlx": "0.32.2", "numpy": "2.2.6",
                  "seed": <int>, "upstream_commit": "ed1fca4c…", "generator": "<file>@<sha256>" },
  "config": { "resolved": { hidden_size, layer_count, intermediate_size, attention_heads,
               kv_heads, head_dim, vocabulary_size, rms_norm_epsilon, rope: {dimensions, theta,
               traditional, scaling}, attention_kinds: ["full"|{"sliding": W}, …],
               tie_word_embeddings, quantization: null|{group_size, bits} } },
  "tokenizer": { "encodings": { "<prompt name>": [ids…] }, "decodings": { "<name>": "text" },
                 "eos_tokens": [ids…], "eos_source": "generation_config"|"config"|"tokenizer" },
  "chat": { "<case>": { "messages": [...], "continuation": "closed"|"start_assistant"|"continue_last",
             "rendered_utf8_hex": "…", "token_ids": [ … ] } },
  "prefill": { "prompt": "<name>", "token_ids": [ … ], "T": <int> },
  "decode": { "greedy_ids": [8 ids], "text_deltas": ["…", …], "finish_reason": "length"|"stop",
              "stop_token": <id|null> },
  "sampling": { "<case>": { "options": { temperature, top_p, top_k, min_p, min_tokens_to_keep,
                 repetition_penalty, repetition_context_size }, "seed": <int>,
                 "cpu_ids": [ … 8 ids … ] } },
  "errors": { "<case>": { "mutation": "<what was changed in a copy of the fixture>",
              "expected_rust_error": "ConfigError::…|WeightError::…|LoadError::…" } },
  "tolerances": { "logits": {"atol": 2e-4, "rtol": 2e-4}, "cache": {"atol": 2e-4, "rtol": 2e-4},
                  "logprobs": {"atol": 1e-4, "rtol": 1e-4} }
}

## NumPy reference output
`conformance/mlx-lm/numpy_reference/out/<model>-<variant>.safetensors` with the SAME array keys
for prefill.full.logits, cache.after_prefill.*, decode.step<j>.logits, produced from the committed
fixture inputs by an independent f32 NumPy implementation (no mlx import). The coordinator
`conformance/mlx-lm/coordinate.py` asserts agreement within `tolerances` and writes
`conformance/mlx-lm/corpus.json` (sha256 of every committed file + policies). Two full runs must
produce identical trees.

## Comparator failure classes (Rust side, one class per failure)
config, tokenizer, chat, shape, dtype, value, cache_offset, cache_range, cache_value,
sampling_support, sampled_id, text_delta, finish_reason, error_class, output_count,
processor, text_stop, progress, cache_trim.

The protected parity adapter (`mlx-lm/tests/parity/**`) produces observations for every class
except `processor`, `text_stop` and `cache_trim`; those three are consumed from the same
goldens by the crate's own tests (`sampling/tests.rs`, `tokenizer/text_tests.rs`,
`cache/tests.rs`), while their mutation qualification stays in `tests/parity/mutations.rs`.

## Tranche 3 cohorts

The normative contracts are position-astra.md, “Processor and sampler contract”,
“EOS, length, stop strings and stable text”, and “Oracle-change list / Minimal new
golden cohorts”, with DECISIONS A/I/J overriding the prompt rule. No tolerance
changes accompany these additions.

Only llama-base adds the following expectation keys:

- `sampling.presence_penalty`, `sampling.frequency_penalty`, and
  `sampling.penalties_combined`, with matching `inputs.sampling` recipes and
  `sampling.<case>.filtered_logprobs` tensors. They use temperature 0.7, disabled
  support filters, seed 1729, eight CPU IDs, and context 3. Enabled coefficients
  are repetition 1.3, presence 0.6, and frequency 0.4. Options add
  `presence_penalty`, `presence_context_size`, `frequency_penalty`, and
  `frequency_context_size` as applicable. `processor_token_histories` records
  each of the eight calls before per-processor cropping; history starts with the
  final prompt token and appends previously accepted generated IDs.
- `processing.<case>` contains `options`, `input_logits`, and `histories`.
  Cases are `presence_negative`, `presence_zero`, `presence_positive`,
  `frequency_negative`, `frequency_zero`, `frequency_positive`, `repetition`,
  and `combined`. The additive coefficients are -0.5, 0, 0.5; repetition is
  1.3; combined uses 1.3/0.6/0.4. All windows are 3. Each corresponding
  `processing.<case>.logits` f32 tensor has shape `[7, 5]`, with one row per
  nonempty prefix of `[1,2,1,2,4,1,3]`, processed from `[0,2,-3,1,-0.5]` by
  pinned `make_logits_processors`. The existing logits tolerance applies.
- `prefill.progress.<case>` contains `token_ids`, `ceiling`, and ordered
  `[processed,total]` `pairs`. Cases are `eight_chunk1`, `eight_chunk3`,
  `eight_chunk8`, and `one_chunk1`. Callback capture uses `max_tokens=0` and
  is cross-checked with an independent integer schedule. Eight-token counts
  are `[0,1,2,3,4,5,6,7,8]`, `[0,3,6,7,8]`, and `[0,7,8]`; one token gives
  `[0,1]`. Initial and final pairs occur exactly once.
- `tokenizer.prompt_encoding` contains `canonical`, `canonical_with_bos`,
  and `whitespace_before_bos` ID lists captured at the model call from
  `stream_generate`. The base tokenizer has a TemplateProcessing post-processor
  prepending configured BOS ID 4. The first two cases forward `[4,12,...]`;
  the third forwards `[4,4,12,...]`. The decision uses the original string's
  prefix, without trimming, encoding first, or decoding the first ID.
  `tokenizer.encodings.special_with_defaults` and the corresponding input
  encoding gain ID 4. All token-ID-driven arrays, sampling, decode, and chat
  expectations remain unchanged.

Only llama-sliding adds `cache.trim_after_wrap`. Its `capacity`, `keep`,
`trim_requested`, and `trim_return` are 5, 2, 2, and 0. Stages `before_trim`,
`after_trim`, and `after_append` record `offset`, `rotation_index`, `can_trim`,
`temporal_positions`, `retained_prefix`, and `retained_tail`. Prefix and tail
are position lists, not a contiguous retained range. Temporal positions are
`[0,1,7,8,9]` before/after trim and `[0,1,8,9,10]` after appending position 10.
For each stage, `cache.trim_after_wrap.<stage>.<kind>` stores f32 `[1,1,5,1]`
for kinds `raw_keys`, `raw_values`, `temporal_keys`, and `temporal_values`.
Keys encode position p; values encode `100 + 3*p`. These synthetic tensors
compare with exact bits. The guarded helper `trim_prompt_cache` is called;
calling the unguarded cache object's `trim` would change wrapped state.

### text_cases.json

This independent llama-base artifact has `schema_version: 1` and `provenance`
with the pinned `tokenizers` version and `text_reference.py@sha256`.
`stop_strings.<case>` contains input `stops`, `deltas`, `final_delta`, the
consumed `events` (`text`, `held`, `stopped`), `finish_flush`, `stopped`, and
`finish_reason`. A null flush means a match already terminated the stream;
an empty string means finishing ran and emitted nothing. Events stop at the
first complete match; later input deltas remain recorded but are not consumed.
Matching is exact Unicode content/UTF-8, without normalization. An unmatched
held suffix is released only after processing the final delta. With stops
`abc` and `b`, `abc` emits nothing, while `ab`, `c` emits `a` and stops in
its first event.

`decoders.<case>` contains a standalone `tokenizer` definition, `token_ids`,
`decoded_utf8_hex` from pinned Python tokenizers, `emitted_prefix`,
`preserves_prefix`, `finish_flush`, and `expected_error`. Named cases reproduce
ByteFallback complete/incomplete/empty/rewrite and ByteLevel incomplete-tail
inputs from tokenizer/tests.rs. Emitted prefixes are the explicitly admitted
Rust traces; final decode bytes are independently computed. The ByteFallback
rewrite has `DecodeStreamError::InvalidPrefix` and a null flush.
`errors.empty_stop` records `stops: [""]` and `expected_error: EmptyStopString`.

### Comparison and qualification

New classes are `processor` (raw table and processor histories), `text_stop`
(the text artifact excluding provenance), `progress` (including counts), and
`cache_trim` (synthetic wrapped-cache metadata and tensors). BOS encoding stays
in `tokenizer`. Existing classes and tolerance policies are unchanged.
Mutations qualify presence-as-frequency, frequency deduplication, expired
context retention, reversed processor order, partial-stop leakage, match
emission, missing held-suffix flush, omitted/duplicated final progress, changed
wrapped trim state, and first-ID/decoded-first-ID BOS shortcuts.
