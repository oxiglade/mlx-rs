# mlx-sys

Rust bindings to the mlx-c API. Generated using bindgen.

The crate version is independent of its native source tuple. This revision targets mlx-c
`a341b4925024b88b2c593468f16e12f5e17315da`, whose CMake configuration pins MLX `v0.32.2`.

## Metal library location

Metal builds place `mlx.metallib` in `~/.mlx/lib/<mlx-c-key>/`, where `<mlx-c-key>` is the
first 12 characters of the pinned mlx-c commit. Packaged source without Git metadata uses a
deterministic hash of the mlx-c headers and CMake configuration instead. The stable location
allows binaries produced by `cargo install` to keep loading the library after Cargo removes its
temporary build directory.

Set `MLX_RS_METAL_PATH` to use a different directory verbatim as CMake's `MLX_METAL_PATH`.
When this override is set, the build does not read or write `HOME`, which supports sandboxed and
Nix builds.

The `metal-jit` feature builds MLX with `MLX_METAL_JIT`: the metallib shrinks from about 136 MB
to 1.5 MB at this pin, and most kernels compile from source the first time they are used.
MLX caches compiled libraries only within a process. The system Metal shader cache can speed
up later runs, but previously unused kernels still compile when first needed.
Results can differ from the precompiled kernels in the last bits.

JIT builds keep their metallib in `~/.mlx/lib/<mlx-c-key>-jit/`, so they don't overwrite the full
library that normal builds use. `MLX_RS_METAL_PATH` is used as given, without the `-jit` suffix,
so if you set it, point JIT and non-JIT builds at different directories.

## Build metadata for dependents

`DEP_MLX_INCLUDE`, `DEP_MLX_DEFINES` and `DEP_MLX_VERSION` are emitted since 0.7.0.
`DEP_MLX_COMPILE_COMMANDS` and `DEP_MLX_METAL` are unreleased additions after 0.7.0.

`mlx-sys` declares `links = "mlx"`, so Cargo runs a dependent's build script after this one and
passes it the locations of the native sources this build compiled:

| Variable | Value |
|---|---|
| `DEP_MLX_INCLUDE` | every directory the include path needs, as a path list: the MLX source tree (`mlx/...`), metal-cpp with the `metal` feature, and mlx-c (`mlx/c/...`) |
| `DEP_MLX_DEFINES` | common header definitions, currently `MLX_STATIC` |
| `DEP_MLX_COMPILE_COMMANDS` | CMake compilation database with the native compiler, include paths, definitions, SDK and target flags |
| `DEP_MLX_METAL` | `true` when the native build enables Metal, otherwise `false` |
| `DEP_MLX_VERSION` | the MLX version mlx-c pins, when it can be read |

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

For an exact match to the native compiler settings, use the `mlx/array.cpp` entry in
`DEP_MLX_COMPILE_COMMANDS`. Keep its compiler and flags, replacing only the source and output
arguments. Add the mlx-c include directory from `DEP_MLX_INCLUDE` when using its private bridge.
Run the compiler in the entry's `directory` to preserve relative arguments. Compilation databases
require a Makefiles or Ninja CMake generator.
