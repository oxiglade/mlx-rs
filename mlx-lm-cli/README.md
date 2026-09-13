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

## Workspace integration and verification

Add exactly `"mlx-lm-cli",` to the root `[workspace].members` array. The package
has no library target, keeps `publish = false`, and defaults to no features.
Its `hf-hub` feature forwards only to `mlx-lm/hf-hub`. It has no direct core or
oracle-hooks dependency. Root Cargo files are owned by the integration item.

The Hub build requires these approved public declarations in `mlx-lm`:

```rust
#[cfg(feature = "hf-hub")]
impl Model {
    pub fn hub_provenance(&self) -> Option<&HubProvenance>;
}

#[cfg(feature = "hf-hub")]
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct HubProvenance {
    pub repo: String,
    pub requested_revision: String,
    pub resolved_revision: String,
}
```

`HubProvenance` must be re-exported at the library root. `Model::from_hub(repo:
&str, options: HubOptions) -> Result<Model, HubError>` also needs its approved
resolver implementation; the current worktree still has the stub.

After integration, run:

```sh
cargo test -p mlx-lm-cli --no-default-features -- --test-threads=1
cargo test -p mlx-lm-cli --features hf-hub -- --test-threads=1
cargo tree -p mlx-lm-cli --no-default-features -e normal
```

Tests consume committed tiny fixtures offline and isolate subprocess caches.
Writer tests inject short writes, Interrupted, WriteZero, ordinary errors, and
BrokenPipe. Process tests check output against the version-1 schema and public
Text-generation events. They deliberately do not reuse the token-ID generation
goldens, whose prompt omits BOS. `conformance/mlx-lm/cli_cases.json` is absent in
this worktree; independent CLI golden comparison remains an integration task.
Enabled offline Hub success additionally needs item 3's completed-snapshot receipt
recipe and item 1's reviewed `hub_cases.json`; no private receipt format is guessed
here. See [VERIFICATION.md](VERIFICATION.md) for executed and unexecuted checks.
