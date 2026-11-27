# mlx-sys

Rust bindings to the mlx-c API. Generated using bindgen.

The crate version follows mlx-c. Version 0.8.0 targets mlx-c
`a26f2d584a14ec619ae8db9a09be03becd47598a`, whose CMake configuration pins MLX `v0.32.4`.

## macOS SDK and deployment target

Set `MLX_SYS_SYSTEM_SDK=1` to build MLX with the system Xcode selected by
`xcode-select`. The project's devenv shell exports this opt-in on Darwin. Its
compiler, sysroot, SDK queries and Metal compiler apply only to MLX's CMake build
and its dependent native shim; other crates keep their C/C++ build environment.
Devenv also uses Xcode for Rust linking, with system libc++ and without Nix linker
flags. Without the opt-in, the native build keeps its normal CMake configuration,
including in Nix derivations and nix-darwin environments.

Opt-in builds use a separate CMake directory, so entering or leaving devenv does
not reuse a compiler cache from the other mode. Changes to the selected SDK's
`SDKSettings.json` rerun configuration. After switching Xcode with `xcode-select`,
run `cargo clean -p mlx-sys` to select its SDK in opt-in mode.
Changing the deployment target discards the opt-in CMake build tree, including
Metal objects, and rebuilds it.

With `MACOSX_DEPLOYMENT_TARGET` unset, MLX targets the host macOS version. M5
neural-accelerator (NAX) kernels need both a macOS 26.2+ SDK and a 26.2+ deployment
target, with Metal 4 support. A default build on a recent host and Xcode gets these kernels.
On the default path, CMake reads an explicit deployment target at first configure.
Changing it requires `cargo clean -p mlx-sys` or a separate Cargo target directory.
With no explicit target, MLX follows the host version on every configure; after a
macOS upgrade, `cargo clean -p mlx-sys` rebuilds the kernels for the new host.
Set a lower target to support older macOS, at the cost of NAX kernels:

```sh
MACOSX_DEPLOYMENT_TARGET=14.0 cargo build --release
```

The project's devenv shell sets the deployment target to the host macOS version.
Set `MACOSX_DEPLOYMENT_TARGET` after entering the shell if you need an older target.
The SDK selection stays local to MLX; do not export the system `SDKROOT` across devenv.

## Metal library location

Metal builds place `mlx.metallib` in
`~/.mlx/lib/<mlx-c-key>-<platform>-<deployment-target>-<profile>-<hash>/`.
The hash covers CMake's platform, resolved deployment target, SDK path and build
profile. CMake resolves the directory once after loading its cache and toolchain
file, then uses it for compilation and installation. Different settings have
separate directories.

`<mlx-c-key>` is the first 12 characters of the pinned mlx-c commit. Packaged
source without Git metadata uses a deterministic hash of the mlx-c headers and
CMake configuration instead. The stable location allows binaries produced by
`cargo install` to keep loading the library after Cargo removes its temporary
build directory.

Set `MLX_RS_METAL_PATH` to use a different directory verbatim as CMake's `MLX_METAL_PATH`.
When this override is set, the build does not read or write `HOME`, which supports sandboxed and
Nix builds.

The `metal-jit` feature builds MLX with `MLX_METAL_JIT`: the metallib shrinks from about 136 MB
to 1.5 MB at this pin, and most kernels compile from source the first time they are used.
MLX caches compiled libraries only within a process. The system Metal shader cache can speed
up later runs, but previously unused kernels still compile when first needed.
Results can differ from the precompiled kernels in the last bits.

JIT builds append `-jit` to `<mlx-c-key>` to keep their libraries separate.
`MLX_RS_METAL_PATH` is used as given, without any suffix, so use separate override
directories for different platforms, deployment targets, SDKs, profiles and JIT modes.

## Native build directory

MLX's CMake tree lives in Cargo's `OUT_DIR`, so every Cargo target directory builds MLX again.
Set `MLX_SYS_BUILD_DIR` to an absolute path to keep the tree there instead, in a subdirectory
named after the target, the profile and the `accelerate`, `metal` and `metal-jit` features, such
as `aarch64-apple-darwin/debug-accelerate-metal`. Builds with the same target, profile and
features then share one tree across target directories and toolchains, and the directory can be
saved and restored by a CI cache.

Builds that share a tree must not run at the same time, and must agree on `MLX_RS_METAL_PATH`:
MLX compiles the metallib location in, so a different one rebuilds it. `cargo clean` leaves this
directory alone; delete it by hand where this page says `cargo clean -p mlx-sys`.

## A prebuilt MLX

Building MLX with Metal from source requires Xcode's Metal compiler, including with `metal-jit`.
Command Line Tools alone can build against a prebuilt MLX. Set `MLX_SYS_PREBUILT` to
an absolute path to its prefix. An unset or empty value builds MLX from source.
The prefix contains `include/`, `share/cmake/MLX/`, `lib/libmlx.dylib`,
`lib/libjaccl.dylib` and `lib/mlx.metallib`.

Use the MLX version mlx-c pins, named above. A different or unreadable version
stops the build. The `mlx-metal` wheel is a zip archive with this prefix under
`mlx/`. CI uses `mlx_metal-0.32.2-py3-none-macosx_26_0_arm64.whl`.
Its `libmlx.dylib` requires macOS 26.2 or newer, despite the `26_0` tag.
For older macOS, choose the matching `macosx_14_0` or `macosx_15_0` wheel of the
same MLX version and check its library's minimum OS version.
The library must have Metal enabled and JIT disabled. Builds with `metal-jit`, or
without the `metal` or `cuda` feature, are rejected. For CUDA, see
[A prebuilt CUDA MLX](#a-prebuilt-cuda-mlx).

MLX loads `mlx.metallib` from beside `libmlx.dylib`. The wheel determines kernel
availability, including NAX. `MLX_RS_METAL_PATH` and the source build's deployment
target, SDK and profile settings cannot change those kernels.
With `MLX_SYS_SYSTEM_SDK=1`, the selected system SDK and compilers apply only to
mlx-c. Prebuilt and source builds use separate CMake directories.

Binaries must be able to find `@rpath/libmlx.dylib` and its dependency
`@rpath/libjaccl.dylib`. Applications that depend on `mlx-rs` or `mlx-lm` can set
an rpath in `.cargo/config.toml`:

```toml
[target.aarch64-apple-darwin]
rustflags = ["-C", "link-arg=-Wl,-rpath,/absolute/path/to/mlx/lib"]
```

For workspace tests and examples, the same setting can come from `RUSTFLAGS`.
Doctests also need `RUSTDOCFLAGS`, which Cargo does not fill from `RUSTFLAGS`:

```sh
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$MLX_SYS_PREBUILT/lib"
export RUSTDOCFLAGS="$RUSTFLAGS"
```

For local runs without an rpath, set `DYLD_FALLBACK_LIBRARY_PATH` to the prefix's
`lib/` directory. Keep the prefix at the same path and version for as long as its
binaries are used. For distribution, install these runtime files at a stable
location or bundle both dylibs and the metallib together with the application
and use an application-relative rpath, such as `@executable_path/../Frameworks`.

A binary package that directly depends on `mlx-sys` can instead use its metadata
in `build.rs`. `DEP_MLX_*` reaches only immediate dependents. A library's build
script cannot add an rpath to its consumers' binaries.

```rust
if let Ok(dir) = std::env::var("DEP_MLX_LIB_DIR") {
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
}
```

Cargo watches the prefix's library, version and CMake files and the wheel's
installation metadata. If a non-wheel prefix is replaced with files whose
timestamps are preserved, run `cargo clean -p mlx-sys` before rebuilding.

## CUDA (Linux)

The `cuda` feature builds MLX's CUDA backend (`MLX_BUILD_CUDA`) for NVIDIA GPUs on Linux,
with `libmlx` and `libmlxc` as shared libraries:

```toml
mlx-sys = { version = "0.7", default-features = false, features = ["cuda"] }
mlx-rs = { version = "0.34", default-features = false }
```

It cannot be combined with `metal` or `accelerate`. The build needs the CUDA toolkit with
cuDNN, and BLAS and LAPACK(E) for MLX's CPU side, where MLX's CMake finds them.

- `MLX_CUDA_ARCHITECTURES` selects the GPU architectures to compile kernels for, such as `89`
  for an L4. When unset, MLX uses the GPU of the build machine.
- `CUDA_HOME` or `CUDA_PATH` selects the toolkit's `nvcc`. Otherwise `nvcc` comes from `PATH`,
  then `/usr/local/cuda/bin`.

The libraries are installed under `OUT_DIR/build/lib` with an `$ORIGIN` rpath, and published as
`DEP_MLX_LIB_DIR`. Binaries need that directory in their rpath or `LD_LIBRARY_PATH`. At run
time `libmlx` compiles its kernels with NVRTC. It finds MLX's kernel headers in the `include/`
directory beside its `lib/` directory, and the CUDA headers under `CUDA_HOME`, by default
`/usr/local/cuda`.

On x86_64, mlx-c declares its `float16`/`bfloat16` functions only on aarch64. The build forces
`src/cuda_half.h` into mlx-c and the bindings to declare them, as on aarch64.

### A prebuilt CUDA MLX

With `cuda`, `MLX_SYS_PREBUILT` takes the prefix of a CUDA build of MLX, with `include/`,
`share/cmake/MLX/` and `lib/libmlx.so`, and builds mlx-c against it as with Metal. MLX's
CMake build of the CUDA kernels is skipped, but CMake, a C++ compiler and the CUDA toolkit
with cuDNN are still needed. `CUDA_HOME` or `CUDA_PATH`, when set, locates the toolkit.
mlx-c is linked statically and `libmlx.so` dynamically. Its MLX version must match the one
mlx-c pins, and its `MLXConfig.cmake` must have CUDA enabled.

`OUT_DIR/build` of an earlier `cuda` source build is such a prefix. Copy its `lib/`,
`include/` and `share/` to keep it, for example in a separate container build stage.
`DEP_MLX_LIB_DIR` is the prefix's `lib/`.

## Build metadata for dependents

`DEP_MLX_INCLUDE`, `DEP_MLX_DEFINES` and `DEP_MLX_VERSION` are emitted since 0.7.0.
`DEP_MLX_COMPILE_COMMANDS`, `DEP_MLX_METAL` and `DEP_MLX_LIB_DIR` are unreleased
additions after 0.7.0.

`mlx-sys` declares `links = "mlx"`. Cargo passes this metadata to build scripts of
packages that directly depend on `mlx-sys`:

| Variable | Value |
|---|---|
| `DEP_MLX_INCLUDE` | every directory the include path needs, as a path list: the MLX source tree (`mlx/...`), metal-cpp with the `metal` feature, and mlx-c (`mlx/c/...`) |
| `DEP_MLX_DEFINES` | common header definitions, currently `MLX_STATIC`; empty with `cuda`, whose MLX is a shared library |
| `DEP_MLX_COMPILE_COMMANDS` | CMake compilation database with the native compiler, include paths, definitions, SDK and target flags |
| `DEP_MLX_METAL` | `true` when the native build enables Metal, otherwise `false` |
| `DEP_MLX_VERSION` | the MLX version mlx-c pins, when it can be read |
| `DEP_MLX_LIB_DIR` | with `MLX_SYS_PREBUILT`: the prefix's `lib/`, which holds `libmlx.dylib`, `libjaccl.dylib` and `mlx.metallib` in the wheel; with `cuda`: the directory of `libmlx.so` |

With `MLX_SYS_PREBUILT`, `DEP_MLX_INCLUDE` holds the prefix's `include/` (and `include/metal_cpp`
with the `metal` feature) and mlx-c, `DEP_MLX_DEFINES` is empty, since MLX is then a shared library
(no `MLX_STATIC`), and `DEP_MLX_VERSION` is the installed MLX's version. `DEP_MLX_METAL`
is `true` with `metal`. `DEP_MLX_COMPILE_COMMANDS` is absent: an installed MLX has no compilation database.
Dependents that compile against MLX internals must handle a missing
`DEP_MLX_COMPILE_COMMANDS`. Cargo always sets it in source mode. A shared MLX
exports only its public symbols; backend internals such as `copy_gpu_inplace`
may have headers but no exported symbol. Such extensions need a fallback or
must require a source build.

C++ that extends MLX, such as a custom `mlx::core::Primitive`, must compile against the same
headers `libmlx` was built from. A dependent can use these with the `cc` crate. Passing the
directories with `-isystem` keeps warnings from MLX's own headers out of the dependent's build:

```rust
let mut build = cc::Build::new();
build.cpp(true).std("c++20");
for dir in std::env::split_paths(&std::env::var_os("DEP_MLX_INCLUDE").unwrap()) {
    build.flag("-isystem").flag(dir);
}
for define in std::env::var("DEP_MLX_DEFINES").unwrap().split_whitespace() {
    match define.split_once('=') {
        Some((name, value)) => build.define(name, value),
        None => build.define(define, None),
    };
}
```

In source mode, use the `mlx/array.cpp` entry for the native compiler settings in
`DEP_MLX_COMPILE_COMMANDS`. Keep its compiler and flags, replacing only the source and output
arguments. Add the mlx-c include directory from `DEP_MLX_INCLUDE` when using its private bridge.
Run the compiler in the entry's `directory` to preserve relative arguments. Compilation databases
require a Makefiles or Ninja CMake generator.
