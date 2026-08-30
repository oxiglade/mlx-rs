# mlx-rs API design charter

This charter governs the mlx-c `c74db5307cc8ce122f48d97ef951b30578674e7f` / MLX `0.32.2`
target tuple. The target adds 81 ABI entries: six support existing cumulative and median APIs,
while 75 remain deferred. The 81-entry disposition table below covers those 75 additions plus six
changed, currently unwrapped entries.

## Charter rules

1. **The C ABI is an implementation input, not the Rust product backlog.**
   *Migration: bump-time.* `mlx-sys` mirrors every target ABI entity, while `mlx-rs` exposes only
   coherent Rust capabilities. Every entity must be classified as compatibility, internal,
   public, deferred, intentionally unexposed, or blocked, with evidence and a revisit trigger;
   deliberate omission is a completed decision, not parity debt.

2. **Each capability has one canonical public home.**
   *Migration: idiom-wave.* An operation with an obvious array receiver is a method; constructors,
   symmetric multi-input operations, multi-output operations, and namespaced algorithms are module
   functions. New APIs must not generate the present method/free-function/macro/device matrix, and
   legacy duplicates become forwarding compatibility shims before deprecation.

3. **Stream selection is execution-local context, named according to its actual scope.**
   *Migration: bump-time for context correctness; idiom-wave for legacy twins.* Canonical
   operations are unsuffixed and use a scoped override, falling back to the target runtime's
   current-thread default. Explicit selection uses nestable, panic-safe synchronous
   `with_stream(&Stream, f)` and `with_device(Device, f)` scopes; selecting one operation means
   opening a one-operation scope. A `thread_local!` implementation must be called thread-local and
   must not promise propagation across `.await`; true async task propagation requires a real
   task-local facility. Per-operation optional streams, `_device` twins, and operation-mirroring
   context objects remain rejected because they reproduce the duplicate surface rather than
   defining one ambient execution policy.

4. **FFI artifacts never dictate high-level Rust types.**
   *Migration: bump-time.* Every `new/free` family becomes one RAII abstraction, while C handles,
   backing structs, typedefs, and destructors remain private. Integer conversions are checked,
   cleanup statuses are consumed, and each symbol's status contract is handled explicitly rather
   than assuming every nonzero status has populated the global error slot. An out-parameter is
   never inspected after a failing status, and `Drop` never panics.

5. **Errors are typed only when the binding knows a stable fact.**
   *Migration: idiom-wave; new status-specific errors obey this at bump-time.* `Exception` remains
   an opaque upstream diagnostic carrying its message and location; no `ErrorKind` may be inferred
   from strings. Rust validation, I/O, conversion, missing-key, and documented numeric status
   outcomes get dedicated enums, while a semantic upstream taxonomy waits for stable mlx-c error
   codes. The generic guard must provide a diagnostic fallback rather than panic when a nonzero
   status has not populated the global error slot.

6. **Every operation has a canonical named `Result` path; operators are a closed Rust-only
   façade.**
   *Migration: bump-time.* Retain only conventional array-producing `Add`, `Sub`, `Mul`, `Div`,
   `Rem`, `Neg`, and already-supported integer bitwise/shift traits. Each delegates to the named
   operation and panics only after that Rust call returns a synchronous construction error. Lazy
   failures remain `Result` from checked evaluation. `PartialEq`, `PartialOrd`, tensor comparison,
   indexing, and semantically fallible update operations are not operator façades. A C callback may
   record the first diagnostic and return, but no panic or unwind may cross C. This keeps operator
   convenience narrow without creating an independent FFI or stream-selection path.

7. **Rust names describe behavior; upstream names remain searchable metadata.**
   *Migration: idiom-wave.* The canonical names should be `concatenate(arrays, axis)`,
   `concatenate_flat`, `stack`, `select`, `matmul`, `split_equal`, and `split_at_indices`; legacy
   names remain deprecated aliases and never silently change behavior under an unchanged
   signature. Deviate when upstream names are keywords, encode overload mechanics, mislead about
   semantics, or conflict with established Rust vocabulary, while recording the C name, Python
   name, Rust path, and `semantic_op` in the ledger.

8. **Required inputs are explicit; independent defaults use concrete options, while mutually
   exclusive forms use semantic sum types.**
   *Migration: bump-time for new APIs; idiom-wave for legacy signatures.* Represent axis selection
   as `Axes::{All, One, Many}` and uniform/per-dimension spatial values with an equivalent concrete
   enum. Use `FooOptions: Default` for independent knobs, and validate correlated fields such as
   FFT lengths and axes together. `Option<T>` means genuine absence, not an overload family.
   Builders remain limited to durable modules, optimizers, and reusable validated configurations;
   their `build` remains fallible whenever validation or allocation can fail. This avoids both
   optional-argument permutation APIs and builders for one-shot tensor operations.

9. **Public macros are reserved for syntax, not simulated keyword arguments.**
   *Migration: bump-time freeze; idiom-wave retirement.* Add no new `generate_macro` operation
   macros or generated optional-argument permutations; the 245 existing op macros are frozen and
   deprecated with their duplicate functions. `array!` remains valid syntax with contractual
   defaults of `i32`, `f32`, and empty `f32`, but gains an explicit-dtype form, and verification,
   serialization, and API-boundary code must use it.

10. **`Array` is a lazy shared handle; observer APIs disclose every effect.**
    *Migration: idiom-wave.* `Clone` remains an O(1) handle duplication. Borrowed access is
    exact-dtype, row-major contiguous, lifetime-bound, and never casts, copies, evaluates, or
    repairs layout implicitly. Owned observers distinguish `to_vec_exact` from `to_vec_cast`;
    scalar observers similarly distinguish exact extraction from conversion. Any evaluation
    performed by an observer must be explicit in its name or required beforehand. Equality remains
    explicit and fallible as exact, cross-dtype value, elementwise, or approximate comparison;
    `PartialEq` stays removed. These contracts prevent a buffer-like API from hiding evaluation,
    allocation, layout repair, or dtype conversion.

11. **Thread traits are structural claims proven for the exact runtime tuple.**
    *Migration: bump-time.* Retain `Array: Send` only if target CPU and Metal tests cover both
    moving one handle and sending independently cloned handles to separate threads while using
    per-thread defaults; otherwise remove the unsafe impl. Never infer `Sync`, and do not treat an
    external-serialization note as evidence of Rust memory safety. `Stream` and `Compiled` remain
    structurally `!Send + !Sync` until their ownership, cache, and concurrent-state models are
    proven. Thread-local scoped defaults must not be described as task-local. In particular,
    `mlx_stream_new_thread_unsafe` remains unexposed because it does not establish safe thread-local
    ownership.

12. **Mutable state has one keyed, presence-preserving schema.**
    *Migration: bump-time for the fail-closed compiled-layout subset; idiom-wave for the unified
    public abstraction.* One `StateProjection` derives length, stable key order, immutable and
    mutable traversal, serialization, and compiled layout. Optional slots remain named and
    presence-tagged; they may not disappear through `compactMap`. Compile captures a key/layout
    fingerprint and applies outputs by key, requiring exact cardinality unless the compiler returns
    an explicit keyed pruning map. Positional `zip` is forbidden. Presence is part of layout, so
    silent partial application or positional reassignment is rejected during the bump even though
    the complete public state redesign follows in the idiom wave.

13. **The dependency bump and public feature admission are separate changes.**
    *Migration: bump-time.* The immutable 0.32.2 bump rebinds compatibility and internal machinery
    but adds no unrelated high-level capability; new public cohorts follow separately with semantic
    mapping, independent oracle cases, ownership/leak evidence, and thread/state evidence where
    applicable. Before the idiom wave, the API gate must use the compiled, feature-qualified
    surface so generated builders and cfg availability are no longer invisible, consistent with
    the roadmap's vertical-admission rule.

## Concrete 0.32.2 rebind decisions

The 23 changed entries already backing Rust APIs, plus six added axis symbols needed by them, are
compatibility work:

| Family | Required bump behavior |
|---|---|
| `cummax/min/prod/sum` — 4 changed, 4 added | Route flattened calls to the new base symbols and explicit axes to `*_axis`; preserve reverse/inclusive behavior and pass no dtype for sum/product. |
| `median` — 1 changed, 2 added | Route all-element, one-axis, and many-axis calls to the matching symbols without introducing new Rust names. |
| Twelve FFT transforms | Pass `MLX_FFT_NORM_BACKWARD`, matching the existing behavior; keep normalization private until a coherent FFT options redesign, and add asymmetric cases. |
| `quantize`, `dequantize`, `qqmm` | Pass absent global-scale handles to preserve affine/current defaults; expose scale controls only with mode-specific conformance. |
| Fast scaled-dot-product attention | Pass `force_fused = false`, the target Python default, and add a case that distinguishes heuristic and forced behavior. |
| Compile clear/erase | Retain the originating cache handle with `Compiled` and pass that exact handle to erase; resolve the caller's current cache on each `clear_cache()` call; consume all statuses and define clone, drop, and thread affinity. |

The six changed-but-unwrapped entries—distributed group handle, init, availability, and split; base
`logcumsumexp`; and base `trace`—are rebound in `mlx-sys` but remain high-level deferred.

## Disposition of the 81 deferred ledger entries

The counts below total 75 added-deferred and six changed-deferred:

| Family | Added | Changed | Charter disposition |
|---|---:|---:|---|
| `compile_cache` handle/type/new/free/detail | 6 | 0 | **Bump-time internal.** One private RAII cache associated with `Compiled`; no six-item public mirror. |
| `mlx_fft_norm` enum/type artifacts | 3 | 0 | **Bump-time internal.** Use `Backward` for compatibility; a public `FftNorm` waits for the FFT options cohort. |
| Bartlett, Blackman, Hamming, Hanning windows | 4 | 0 | **First post-bump cohort.** `ops::windows::{bartlett, blackman, hamming, hann}(usize)`, with checked conversion, no stream argument, twin, or macro. |
| `fftfreq`, `rfftfreq` | 2 | 0 | **First post-bump cohort.** `fft::{fftfreq, rfftfreq}(usize, f64)` with odd/even and negative-frequency oracle cases. |
| GGUF I/O and handle artifacts | 19 | 0 | **Separate feature cohort.** Collapse to an RAII `GgufFile` and typed `GgufMetadata`; status 2 maps to absence and status 3 to wrong metadata type, not `Exception` or panic. |
| Slice-update add/max/min/prod | 4 | 0 | **Idiom-wave indexing cohort.** One index-update operation with `UpdateMode::{Replace, Add, Min, Max, Product}`, shared by slice and advanced indexing. |
| General array/math additions | 16 | 0 | **Deferred semantic cohorts.** Includes `count_nonzero`, `diff`, `flip`, det/slogdet, linspace endpoint, xor, searchsorted, trunc, unstack, and vecdot; base/axis/axes C overloads collapse into one Rust concept. |
| Fast cross-entropy and gather-QQMM | 2 | 0 | **Demand-gated.** Require a model-level use case plus dtype, gradient, and quantization evidence. |
| `logcumsumexp_axis` and `trace_axes` | 2 | 2 | **Deferred as complete families.** Design each base/axes API once rather than exposing only the newly added half. |
| `ones_like_dtype`, `zeros_like_dtype` | 2 | 0 | **Intentionally unexposed.** Existing shape-plus-dtype factories express the same Rust capability. |
| `positive` | 1 | 0 | **Intentionally unexposed.** It is semantic identity and Rust has no unary-plus trait worth emulating as a function. |
| Node namer, DOT export, graph printing | 9 | 0 | **Demand-gated diagnostic cohort.** If admitted, expose Rust writers/path APIs and one ownership abstraction, never raw `FILE*` or seven handle operations. |
| Metallib path get/set | 2 | 0 | **Blocked.** The process-global lifecycle and initialization constraints need a stable contract before a safe API. |
| Distributed group new/free plus changed family | 2 | 4 | **Deferred as a subsystem.** Requires RAII ownership, backend availability, multi-process evidence, and consistent error semantics. |
| `mlx_stream_new_thread_unsafe` | 1 | 0 | **Intentionally unexposed.** This is a globally registered unsynchronized stream, not a thread-local-stream constructor. |
| **Total** | **75** | **6** | |

## Explicit non-goals

- Full Python MLX, NumPy, or C-symbol parity.
- Python keyword-argument emulation through public macros.
- Implicit evaluation, synchronization, materialization, copying, dtype conversion, or contiguity
  repair.
- Safe wrappers around upstream facilities whose synchronization or global-lifecycle invariant
  cannot be encoded.
- Exporter, graph-debugging, distributed, or model-convenience features solely because Python
  received them.
- Stable categorization of upstream error strings.
- Treating `deferred` as an obligation to reach zero outside a separately designated feature
  cohort.

## Signature

- **Owner:** David Chavez
- **Accepted:** 2026-08-30
- **Target tuple:** mlx-c `c74db5307cc8ce122f48d97ef951b30578674e7f` / MLX `0.32.2`
