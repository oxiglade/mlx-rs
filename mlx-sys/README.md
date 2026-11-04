# mlx-sys

Rust bindings to the mlx-c API. Generated using bindgen.

The crate version is independent of its native source tuple. This revision targets mlx-c
`a341b4925024b88b2c593468f16e12f5e17315da`, whose CMake configuration pins MLX `v0.32.2`.

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
The library must have Metal enabled and JIT disabled. Builds without the `metal`
feature or with `metal-jit` are rejected.

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

## Build metadata for dependents

`DEP_MLX_INCLUDE`, `DEP_MLX_DEFINES` and `DEP_MLX_VERSION` are emitted since 0.7.0.
`DEP_MLX_COMPILE_COMMANDS`, `DEP_MLX_METAL` and `DEP_MLX_LIB_DIR` are unreleased
additions after 0.7.0.

`mlx-sys` declares `links = "mlx"`. Cargo passes this metadata to build scripts of
packages that directly depend on `mlx-sys`:

| Variable | Value |
|---|---|
| `DEP_MLX_INCLUDE` | every directory the include path needs, as a path list: the MLX source tree (`mlx/...`), metal-cpp with the `metal` feature, and mlx-c (`mlx/c/...`) |
| `DEP_MLX_DEFINES` | common header definitions, currently `MLX_STATIC` |
| `DEP_MLX_COMPILE_COMMANDS` | CMake compilation database with the native compiler, include paths, definitions, SDK and target flags |
| `DEP_MLX_METAL` | `true` when the native build enables Metal, otherwise `false` |
| `DEP_MLX_VERSION` | the MLX version mlx-c pins, when it can be read |
| `DEP_MLX_LIB_DIR` | with `MLX_SYS_PREBUILT` only: the prefix's `lib/`, which holds `libmlx.dylib`, `libjaccl.dylib` and `mlx.metallib` in the wheel |

With `MLX_SYS_PREBUILT`, `DEP_MLX_INCLUDE` holds the prefix's `include/` (and `include/metal_cpp`
with the `metal` feature) and mlx-c, `DEP_MLX_DEFINES` is empty, since MLX is then a shared library
(no `MLX_STATIC`), and `DEP_MLX_VERSION` is the installed MLX's version. `DEP_MLX_METAL`
is `true`. `DEP_MLX_COMPILE_COMMANDS` is absent: an installed MLX has no compilation database.
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
