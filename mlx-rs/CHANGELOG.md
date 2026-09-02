# Changelog

## 0.32.0

This release moves to MLX 0.32.2 and reworks the public API around the design
rules in `CHARTER.md`. Most callers will need small edits; the deprecated
forwarding shims point at the replacements.

Breaking

- Update `mlx-sys` to 0.6.0: mlx-c `c74db530` (v0.6.0 plus seven commits), MLX 0.32.2.
- `Array` no longer implements `PartialEq`. Compare with `eq_exact` or `array_eq`.
- `item` and `to_vec` are split into `item_exact`/`to_vec_exact` (dtype must match)
  and `item_cast`/`to_vec_cast`.
- `try_as_slice` returns `AsSliceError::NotContiguous` for strided views instead of
  reading them; materialize with `Array::contiguous()` first.
- Stream selection is scoped: `with_stream` and `with_device` replace the `_device`
  function variants, which remain as deprecated shims.
- Ops with axis variants collapse into one function taking `Axes` or an options
  struct: `linspace` takes `LinspaceOptions`, `trace` takes `TraceOptions`,
  `logcumsumexp` takes `LogCumsumExpOptions`, and so on.
- Behavioral renames: `concatenate_axis`, `split_equal`, `split_at_indices`, `select`.
- `fast::rms_norm` takes `weight: Option<&Array>`.
- `Compiled` is `!Send`; compiled functions carry per-instance state instead of a
  shared cache, and optimizer state goes through a keyed `StateProjection`.
- The op-generation macros are deprecated.

Added

- `io::GgufFile`: load and save GGUF with typed metadata (arrays, strings, string
  lists). Q4_0, Q4_1 and Q8_0 tensors load as quantized triplets.
- `memory`: the full allocator family (active, cache and peak memory, limits,
  `clear_cache`).
- `metal::metallib_path` and `metal::set_metallib_path`.
- `TryIndexUpdateOp` with `UpdateMode::{Replace, Add, Min, Max, Product}` for
  functional slice and scatter updates.
- Ops: `count_nonzero`, `diff`, `flip`, `trace`, `trunc`, `unstack`, `vecdot`,
  `logical_xor`, `logcumsumexp`, `search_sorted`, `linalg::det`, `linalg::slogdet`,
  window functions (`bartlett`, `blackman`, `hamming`, `hann`), `fft::fftfreq`,
  `fft::rfftfreq`, backward FFT normalization, `Array::contiguous`.

Fixed

- The MLX error handler leaked every overwritten message.
- `SafeTensors` leaked its iterator on the error path; the ambient stream is restored
  on unwind; panics inside closures no longer cross the FFI boundary.
- Implicit RNG state was process-global and carried a stream across threads; it is
  thread-local now.
- Compile cache handles leaked on drop.
- `linspace` narrowed `f64` endpoints through `f32`.
- Adafactor state omitted `step`, so a restored optimizer resumed on the wrong
  schedule.
- The metallib path baked at build time did not survive `cargo install` (#327,
  thanks @rgbkrk); it now lives under `~/.mlx/lib/<mlx-c key>/`, or wherever
  `MLX_RS_METAL_PATH` points.
- RoPE runs on the input shape instead of a reshaped 3D tensor (#357, thanks
  @sergey-scherbina).

Verification

Every op above is checked against pinned Python MLX fixtures, an FFI leak gate runs
in CI, and `ROADMAP.md` tracks what is still deferred. See `CHARTER.md` for the
design rules and `conformance/README.md` for the oracle setup.

## 0.25.3

- @dshan4585 Prevent premature destructuring of closures & Add atan2 (#286)
- @Vlad-Shcherbina Fix not one but two leaks related to gradients (#296)
- @scttfrdmn Fix: Add missing Float64 pattern in safetensors conversion (#295)
- @Vlad-Shcherbina Add missing #[param] attributes to InstanceNorm (#300)

## 0.25.2

- Introduce initial support for mlx-lm
  - impl `Parameter` trait for `Option<T>` where `T: ModuleParameters`
  - Add `finfo_max` and `finfo_min`
  - impl `Quantizable` for `Option<T>` where `T: Quantizable`

## 0.25.1

- Fix bug with `index_mut`

## 0.25.0

- Update `mlx-c` to version "0.2.0" and changes function signatures to
  match the new API
- Update `thiserror` to version "2"
- Fix wrong states number in `compile_with_state`
- Remove unnecessary evaluation in fft ops

## 0.23.0

- Update `mlx-c` to "0.1.2"
- Added `dilation` and `groups` parameters to the convolution layer

## 0.21.1

- Fix `mlx-sys` dependency to patch version in workspace

## 0.21.0

- Initial feature-complete release
