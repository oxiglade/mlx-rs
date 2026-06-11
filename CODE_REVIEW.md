# Code Review Guidelines — mlx-rs

What to look for when reviewing a PR or commit in this workspace. Each item is a check, not an assertion.

Items already enforced by clippy / rustc lints are not repeated here. Trust CI for those; review covers what tools cannot. The rules below are the ones that bit us in practice — most are MLX/FFI hazards that compile cleanly and fail (or silently corrupt output) only at runtime on the Metal stream.

## MLX state, threading, and Metal

- **No `thread_local!` for compiled-graph / kernel-cache state.** mlx-c v0.31 SIGSEGVs when a `thread_local` destructor races the GPU stream during thread exit. Caches that hold `Array`s or `Compiled<…>` must live as fields on the owning struct and drop deterministically with it — not in a `thread_local!`. (In-tree latent example: `nn::positional_encoding::ALIBI_CACHE` is a `thread_local! RefCell<HashMap<_, Array>>`; it predates this rule and only fires on `Alibi`, but it is exactly the shape the rule forbids.)
- **Module-level statics: only FFI callback slots and parse-once constants.** Comment them as such. Persistent runtime state holding GPU handles belongs in struct ownership or `Arc<Mutex<T>>`, never a free `static`/`thread_local`.
- **Cache kernel names; never rebuild per call.** mlx caches compiled Metal kernels by name. Bump a version suffix (`_v10` → `_v11`) on every kernel-body change or the stale binary persists across runs.
- **Metal grid semantics: grid = total threads.** `.grid(N, …)` is `threadgroup_size × num_groups`, not `num_groups`. Wrong = only thread 0 runs.
- **Reject f64 in kernel-adjacent code.** `Array::from_f64` lands on the Metal stream and is rejected. Build sentinel values as f32, then `.as_dtype(target)`.
- **Convert bool masks to additive form before adding to scores.** A causal mask built as bool, added directly to scores, silently broadcasts and gives garbage output (KL ≈ 7) while still running. Use `where(mask, scores, neg_inf_sentinel)` or convert bool → (0 / −inf) explicitly.
- **Watch for intrinsics shadowed by locals in kernel source** — `float simd_sum = simd_sum(x)` shadows the intrinsic. Name locals `lane_sum`.
- **Run kernel/Metal tests single-threaded** (`--test-threads=1`); mlx-c shared global state crashes under parallel execution.

## FFI and bindgen

- **`mlx-sys` is the only crate that talks to mlx-c directly.** Anything else importing `mlx_sys::*` is a layering violation; wrap in `mlx-rs` first.
- **New bindings come from the bindgen step, not hand-written `extern "C"`.** Verify the submodule SHA and the generated bindings match.
- **FFI calls live in `unsafe` blocks and check status before using output pointers.** Raw mlx-c calls return status codes; treating an output pointer as valid without checking is a latent crash.
- **Check `Drop` impls on FFI-owned handles** — `mlx_*_free` must be called exactly once. Double-free is silent until address-sanitizer or a stress test.
- **Verify the mlx-c version pin is bumped in one place** — `[workspace.metadata.mlx]` in the root `Cargo.toml` — and that the `mlx-sys/src/mlx-c` submodule SHA matches it. Note these track *two* upstream projects: the metadata pins `ml-explore/mlx` (the C++ kernel lib), the submodule pins `ml-explore/mlx-c` (the C ABI shim). Two different SHAs by design.

## Array clones, ownership, and FFI roundtrips

Every `Array::clone()` is a refcount-bump FFI call across the mlx-c boundary. One saved clone is one fewer `mlx_clone` round-trip per token per layer; with dozens of layers the cost compounds fast.

- **Flag `.clone()` on `Array` inside a per-token or per-layer loop.** First fix is ownership: if the source isn't reused after the call, move it.
- **Flag `concatenate_axis(&[x.clone(), x], …)` / `stack_axis(&[x.clone(), …])`.** These take `&[&Array]` / `impl AsRef<Array>`; pass `&x` borrows.
- **Flag clones inserted to "satisfy the borrow checker".** Restructure the call site first — consume by move, take a borrow earlier, or rebind. A per-layer `residual = h.clone()` where `h` is not rebound before the consumer is pure waste.
- **Flag fn signatures taking `Array` by value when they only read it.** `scaled_dot_product_attention` takes `queries: &Array` precisely so cache `attention()` impls pass a borrow straight through. New attention helpers follow.
- **Flag silent eager evaluation in the decode loop.** `item()`, `as_slice()`, `save_*` force a sync barrier and kill GPU pipelining. `Array::try_item` must `eval()` exactly once. Every new sync-forcing call inside a hot loop needs justification.
- **Verify async_eval scheduling in any decode loop.** Submit step N+1's forward + sample via `async_eval` *before* the host blocks on N (e.g. an EOS-check `item()`), so the GPU runs N+1 while the host pays N's coherence barrier. The pattern is non-obvious and easy to lose in a rewrite.
- **Flag growable buffers that push inside the decode loop without pre-allocation** — a `Vec<u32>` of produced tokens, logit buffers, token-id windows. Pre-size to `max_tokens`.
- **Flag stateless per-forward compute held as a per-layer field.** A rope-table / mask / dtype-bound-scalar struct instantiated once per layer but called with identical inputs every forward should be hoisted to the decoder level, computed once, and threaded through the layer loop as borrows. Any `*::new(cfg)` field that takes only config (no learnable params) is a candidate.

## Dtype management and silent precision loss

mlx is dtype-strict: ops between mismatched dtypes either error or silently promote. Promotion is the dangerous case — promoting bf16/fp16 to f32 mid-graph poisons every downstream op for the rest of the forward and quietly halves throughput.

- **Flag `Array::from_f32(scalar) * inputs` without an explicit `as_dtype(target)`.** `queries * f32_scale` promotes bf16/fp16 inputs to f32. Stage the scalar into the input dtype first: `Array::from_f32(scale).as_dtype(q_dtype)?`.
- **Flag `Array::from_f64(_)` anywhere kernel-adjacent.** Metal rejects f64; build sentinels as f32 then `.as_dtype(target)`.
- **Flag dtype-cast chains inside loops.** Each `.as_dtype()` allocates a graph node. Cache the dtype-promoted constant outside the loop (e.g. an `inv_temp` / `neg_inf` scalar bound to the logits dtype on first use, reused per token).
- **Flag bool masks added directly to scores** (see Metal section — silently broken).
- **Flag `f32` accumulators added to bf16/fp16 graphs without a cast back.** End-of-op `as_dtype(input_dtype)` keeps the rest of the graph in the original precision.
- **Flag a deliberate f32 round-trip in a hot path "for precision".** A cast pair bracketing a fused multiply-add chain per layer per token is expensive (cast launches + 2× bandwidth). Prove the f32 width is needed against the input-dtype baseline before accepting the cost. Compute frequency tables (cos/sin) in f32 once, but cast the *output* to input dtype before threading into per-layer compute.

## Allocation patterns (fixed-N, hot paths)

When N is known at compile time, draining FFI vectors or iterators through a `Vec<T>` is a wasted heap alloc + dealloc per call. On per-token paths this is measurable.

- **Flag `Vec<T>` → `[T; N]` via `try_into` / `collect`** when N is a compile-time constant. Use a `MaybeUninit<[T; N]>` slot filled by index, with drop-on-error cleanup. `VectorArray::try_into_array<N>` is the in-tree reference.
- **Flag helpers taking `Vec<T>` when callers always pass fixed-size data.** `&[T; N]` / `[T; N]` on the signature pushes the alloc off the hot path.
- **Flag per-step `Array::from_slice(&[scalar], &[1])` / `Array::from_iter` in the decode loop.** Each is a host→device upload + a fresh graph node. Stash the scalar on a struct field and reuse.
- **Don't apply fixed-N rewrites when N varies at runtime** (variable sequence/layer length). `Vec` is correct there.
- **Flag iterator chains materialising intermediate `Vec`s** when the consumer takes `&[T]` / `impl Iterator`.

## Performance and benchmarks

- **Flag perf-sensitive changes that lack a bench run.** Attention, KV-cache, generation loop, kernel work — run `cargo bench -p mlx-lm --bench lm_decode` before commit and compare against `BENCHMARK.md`. An accepted change updates the table on the same commit.
- **Verify the decode-only methodology** — criterion `iter_custom`, prefill outside the timing band, single-threaded, cooldown between cells. The bench must drive the production decode path (the `Generate` iterator), not a hand-rolled loop that pipelines differently from real callers.
- **Flag `O(N²)` allocations/copies in the decode hot path even when the microbench bypasses them.** "Not measured here" ≠ "free" — a per-step cost invisible to the bench is paid by every real consumer.
- **Single-cell process isolation is the gold standard for regression checks.** Run one model per `cargo bench` via `MLX_LM_BENCH_ONLY=<cell>`: each invocation is a fresh process, fresh kernel cache, fresh mlx-c global state. Reserve the full sweep for publishing baseline numbers.
- **mlx-c version bumps require `cargo clean` + a full bench rerun.** Cross-version comparisons against cached old artefacts lie.
- **Don't run anything else on the machine while a bench cell is timing.** Concurrent work — a parallel `cargo build`, an editor save triggering rust-analyzer — steals CPU/GPU cycles from criterion's `iter_custom` window and produces phantom regressions. Kill background compiles first.
- **`BENCHMARK.md` numbers are a captured snapshot, not ground truth.** A 2–5% delta vs the table is more reliably checked by rerunning baseline + candidate in the *same session* than by trusting the recorded number, which may have been measured on a quieter machine.

## Error handling

- **Fallible conversions use `TryFrom`, not `From`.** A value parsed from a config string (a `Dtype`, an enum variant) must be `TryFrom`; never silent-default on unknown input.
- **Flag stringly-typed state for finite value sets.** A `&str`/`String` field accepting a closed set of values should be a `#[derive(Deserialize)]` enum that fails at the deserialize boundary, not at the kernel call site 30 layers deep.
- **Flag silent `.ok()`** that discards a `Result` from a meaningful op — channel sends, I/O writes, weight-loader probes should `log::warn!` on error. `.ok()` is fine only for genuinely inconsequential ops.
- **Verify error variants carry context.** A bare `Error::Other(String)` loses call-site detail; prefer typed variants where the recovery path differs.
- **Flag `?` in `main` that swallows context** — prefer explicit top-level handling.

## Constants and magic values

- **Flag magic numbers** — buffer sizes, head dims, group sizes, KV bits, sliding-window length. Name them `const` at module top.
- **Flag `const` inside fn bodies or closures.** `const` belongs at module top.
- **Check tensor-shape literals are named or asserted** — `[B, n_kv_heads, n_repeats, L, D]` chains are easy to typo. Name the axis count or assert shape at the entry point.

## Type system, lifetimes, ownership

- **Flag `.clone()` on plain Rust types that exist only to satisfy the borrow checker.** Restructure ownership: drop the field, consume by move (`into_inner()`), or rework call sites. (For `Array` see the FFI section — those have their own cost.)
- **Flag wrapper struct + parallel data** — change the base type or define a richer local type.
- **Flag `'static` bounds added without justification** — usually a workaround for an ownership problem.
- **Check for `Arc<Mutex<T>>` where `&mut T` would suffice** — shared ownership for single-threaded code is overkill, and on mlx-c the shared-state cost is real.
- **Verify `Drop` impls don't panic** — panic-in-drop is UB during unwinding; on FFI handles it is catastrophic.

## Tests

- **Tests live in the same file** as the code under test — `#[cfg(test)] mod tests`.
- **`mlx-lm` lib tests run with `--test-threads=1`.** mlx-c shared state crashes under parallel execution; a test that passes single-threaded but fails in the default run is mlx-c noise, not a real failure.
- **Flag tests depending on global state** — env vars, cwd, fixed temp paths. Use a process-id-suffixed temp dir.
- **`#[ignore]` tests that need real checkpoints**, and document the download line in the test header.
- **Flag tests asserting on log output** — test the side effect, not the log line.

## Comments and docs

- **Flag verbose doc comments** — one short line max for non-obvious *why*.
- **Flag noise comments** — narrating the diff (`// removed unused field`), describing the obvious (`// increment counter`), or referencing call sites that rot.
- **Flag decorative section dividers** (`// ──── Section ────`).
- **Comments must be self-contained** — readable in 6 months with no PR context. "TODO: ask user", "see PR #123" belongs in chat, not the file.
- **Flag stale Python references** — inline Python comments are tolerated only as the spec being implemented; drop them once the Rust path is canonical.

## Imports and paths

- **Flag inline multi-segment path qualifiers** in signatures, type annotations, struct fields, generic args, call sites. `fn foo(b: std::collections::HashMap<K, V>)` → `use std::collections::HashMap;` + `HashMap<K, V>`. Single-segment `crate::Foo` / `super::Foo` is fine.
- **Flag `use` inside fn bodies.** Exceptions: `#[cfg(test)] mod tests { use super::*; }` and `#[cfg(...)]`-gated fns whose imports are also `#[cfg(...)]`-only.
- **Flag `use std::fmt::Result`** — shadows the prelude `Result`. Use `use std::fmt;` then `fmt::Result`.

## Concurrency

- **Verify channel send/recv errors are handled** — a silent `.ok()` on `Sender::send` is a dropped token in the streaming path.
- **The generation loop is sync by design.** Flag `tokio` / async runtimes inside the LM core; mlx-c isn't async-safe.

## Duplication

- **Search for similar fns before approving a new one** — re-implementing an existing parser, cache helper, or attention wrapper is a frequent failure mode.
- **Three near-identical lines is fine; a premature helper is not** — a fourth duplicate is the time to extract.

## Git hygiene

- **Every commit leaves the repo green** — `cargo check` + `cargo clippy -p <crate> --all-targets -- -D warnings` + `cargo test -p <crate>`.
- **No fixup commits stacked on top.** Fold fixes into the commit that introduced the issue (`git commit --amend --no-edit`).
- **`cargo fmt` + clippy fixes fold into the commit that introduced them**, not a separate "fmt fix" commit.
- **Conventional prefix mandatory** — `feat(scope):`, `fix(scope):`, `refactor(scope):`, etc. No forward refs or session context in messages.
