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

## Array keys in expectations.safetensors
- `prefill.full.logits` — `[1, T, V]` for the canonical prompt, one-shot prefill.
- `prefill.chunk<N>.logits` — same tensor produced with prefill chunk size N, for N in
  {1, 3, T}; N=3 must be a non-divisor of T.
- `cache.after_prefill.layer<i>.keys` / `.values` — logical K/V `[1, kv_heads, T, head_dim]`
  after prefill (for rotating layers: the retained window only).
- `decode.step<j>.logits` — `[1, V]` for greedy decode steps j in 0..8.
- `cache.after_decode.layer<i>.keys` / `.values` — logical K/V after the 8 decode steps.
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
sampling_support, sampled_id, text_delta, finish_reason, error_class, output_count.
