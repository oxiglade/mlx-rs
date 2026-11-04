# CHANGELOG

## Unreleased

- Build against an installed MLX with `MLX_SYS_PREBUILT`, set to its prefix: mlx-c is built
  against that MLX and linked to its `libmlx.dylib`, so a machine without Xcode's Metal compiler
  can build. The prefix's `lib/` is published as `DEP_MLX_LIB_DIR`, for dependents' rpath

- Publish the native compilation database and Metal feature state as
  `DEP_MLX_COMPILE_COMMANDS` and `DEP_MLX_METAL`, so dependent build scripts can
  match MLX's compiler, SDK and target flags

- Generate the bindings with bindgen 0.73; the system opaque types `_malloc_zone_t`
  and `__sFILEX` no longer derive `Copy` and `Clone`

- `MLX_SYS_SYSTEM_SDK=1` builds MLX with the system Xcode's SDK, compilers and Metal
  compiler in a separate CMake build directory, for environments such as Nix whose
  SDK would otherwise turn off the M5 neural-accelerator kernels (#450)

- Keep metallibs in `~/.mlx/lib/` per platform, deployment target, SDK and build
  profile, so builds that differ in any of these no longer overwrite each other's
  kernels; binaries built earlier keep loading the old directory (#450)

## 0.7.0

- Update the pinned native tuple to mlx-c `a341b49` (v0.7.0); MLX stays at `0.32.2`.
  The C API adds 19 stream functions, and the `mlx_io_vtable_` callbacks change
  return types: `seek` returns `int`, `read` and `write` return `size_t`
- Declare `links = "mlx"` and publish the MLX include path (MLX, metal-cpp and
  mlx-c), the MLX defines and the pinned MLX version as `DEP_MLX_*`, so
  dependent build scripts can compile C++ against the exact MLX this crate built
  (#390, thanks @SneakyBerry)
- Add the `metal-jit` feature, which builds MLX with `MLX_METAL_JIT` and keeps its
  metallib in `~/.mlx/lib/<mlx-c key>-jit/` (#391, thanks @anuejn)
- Link the `libjaccl` archive MLX builds, so `mlx_distributed_*` links on macOS SDK
  26.2 and later
- The generated bindings no longer declare `malloc` and `realloc`, which Rust 1.99
  rejects

## 0.6.0

- Update the pinned native tuple to mlx-c `c74db530` and MLX `0.32.2`
- Link the vendored `gguflib` archive; the GGUF symbols were unusable before
- Write the Metal library to `~/.mlx/lib/<mlx-c key>/` so binaries from
  `cargo install` can find it; `MLX_RS_METAL_PATH` overrides the location

## 0.1.2-release

- Update generated bindings to mlx-c 0.1.2

## ~~0.1.2~~

- ~~Update generated bindings to mlx-c 0.1.2~~
- Mistakenly published 0.1.0 as 0.1.2

## 0.1.0

- Update generated bindings to mlx-c 0.1.0
