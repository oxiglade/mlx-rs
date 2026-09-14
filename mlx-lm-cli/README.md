# mlx-lm CLI

An unpublished consumer of the public `mlx-lm` API. The package is `mlx-lm-cli`;
its only binary is `mlx-lm`. It has two commands:

```text
mlx-lm generate (--model DIR | --repo REPO) --prompt TEXT
    [--revision REF] [--offline] [--cache-dir DIR]
    [--max-tokens N] [--temperature T]
    [--top-p P] [--top-k K] [--min-p P] [--min-tokens-to-keep N]
    [--seed U64] [--stop STRING]... [--format text|jsonl]

mlx-lm info (--model DIR | --repo REPO)
    [--revision REF] [--offline] [--cache-dir DIR]
    [--format text|json]
```

`--model` explicitly selects a local safetensors directory, including indexed
shards. `--repo`, `--revision`, `--offline`, and `--cache-dir` exist only in builds
with `--features hf-hub`. The last three require `--repo` and reject `--model`.
There is no path/repository guessing or direct GGUF input.

Generation passes the prompt unchanged to `Prompt::Text`. Defaults come from
`GenerationOptions`: 256 tokens, temperature zero (greedy), filters disabled,
and no seed. Integer limits must be positive. Temperature must be finite and
nonnegative, top-p in `(0, 1]`, and min-p in `[0, 1]`. Min-p retains at least one
candidate unless `--min-tokens-to-keep` specifies another positive count; that
flag requires `--min-p`. Each `--stop` adds an exact, nonempty string. These checks
run before loading. Vocabulary bounds use public sampling validation after load.

Text output is the exact concatenation of public token deltas, flushed after
each delta, with no prompt echo or appended newline. Prefill diagnostics go to
stderr. JSONL emits every public event, including empty token text and the final
finish reason, and flushes every record:

```jsonl
{"version":1,"event":"prefill","processed":0,"total":9}
{"version":1,"event":"token","token_id":12,"text":"hello","finish_reason":null}
{"version":1,"event":"token","token_id":13,"text":"","finish_reason":"length"}
```

The field order above is fixed. Finish reasons are `null`, `"stop"`, or `"length"`.
Unknown future public event/finish variants produce an error.

`info` strictly loads and evaluates model parameters before writing any output.
Its JSON root contains exactly `version`, `config`, `tokenizer`, and
`hub_provenance`, in that order. `version` is 1. Configuration mirrors the public
`Config` and nested dimension fields. Model types and parameter paths are strings;
map keys sort lexicographically. Attention is `{"kind":"full"}` or
`{"kind":"sliding","window":N}`. RoPE scaling is tagged with `kind` equal to
`none`, `linear`, or `llama3`, followed by the public variant fields. Quantization
is null or `{"default":{"bits":B,"group_size":N},"layers":{...}}`; layer
values have `kind` equal to `unquantized` or `affine`, with `bits` and `group_size`
for affine overrides.

Tokenizer facts are `{"bos_token":string-or-null,"eos_tokens":[IDs]}`. Provenance
is null for local loads and non-Hub builds, or contains `repo`,
`requested_revision`, and `resolved_revision` from the public Hub accessor.
Text info labels the same configuration, tokenizer, and provenance facts.
No filesystem paths or inferred weight dtypes appear in info output.

Syntax errors exit 2. Load, generation, serialization, and output failures exit 1,
with their source chain on stderr. BrokenPipe exits 0 and drops generation without
requesting another event. Help and version also use stderr; stdout carries only
requested model output.

## Build and verification

The package is a workspace member, keeps `publish = false`, and defaults to no
features. Publication requires David's explicit go. Its `hf-hub` feature forwards
only to `mlx-lm/hf-hub`; it has no direct core or oracle-hooks dependency.

```sh
cargo build -p mlx-lm-cli
cargo build -p mlx-lm-cli --features hf-hub
cargo test -p mlx-lm-cli --no-default-features -- --test-threads=1
cargo test -p mlx-lm-cli --features hf-hub -- --test-threads=1
```

Hub provenance reports the resolved commit. Completion receipts establish
completeness and recorded commit identity, not tamper resistance. Hashes are
recorded at download completion; reuse/offline validation checks presence,
paths, sizes and receipt identity without rereading contents. Same-size edits
are undetected, and an unreceipted third-party cache is an offline miss.

Tests consume committed tiny fixtures offline and isolate subprocess caches.
Writer tests cover short writes, Interrupted, WriteZero, ordinary errors and
BrokenPipe. Process tests check the version-1 wire schema against
[`cli_cases.json`](../conformance/mlx-lm/cli_cases.json); Hub contracts use
[`hub_cases.json`](../conformance/mlx-lm/hub_cases.json). Inference tests need
native MLX initialization, even for CPU execution.

The [library README](../mlx-lm/README.md) records source profiles, ownership,
errors, demand-gated capabilities and measured throughput: 38.2 tokens/s against
upstream's 125.5 on the same machine and weights (Metal, Qwen3-0.6B-4bit,
256 greedy tokens), a 3.3× gap under investigation in its own tranche. Report
paths and hashes are in the [release identity](../ledger/mlx-lm-release-identity.json),
which also distinguishes this freeze's checks from earlier host runs.
