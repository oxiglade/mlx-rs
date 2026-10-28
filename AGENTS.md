## Building and testing

Use the repo's devenv environment, activated through direnv with `direnv allow`.
Run tools directly once inside it. Generate the product lockfile once with
`cargo generate-lockfile` before running locked build commands. Build with `cargo build --workspace`.
Run tests with `cargo test --workspace -- --test-threads=1`: MLX's global state
is not safe for parallel tests.

`xtask/` is a separate Cargo workspace with its own committed lockfile.
Run tools from the repo root with `cargo xtask <command>`; the alias selects
`xtask/Cargo.toml`, and subprocesses operate on the product workspace.
Root workspace commands exclude tooling, so check it explicitly:

```sh
cargo fmt --manifest-path xtask/Cargo.toml -- --check
cargo clippy --all-targets --manifest-path xtask/Cargo.toml -- -D warnings
cargo test --manifest-path xtask/Cargo.toml
```

The devenv script `validate` runs fmt, clippy, and tests for both workspaces.
`validate-fast` runs the same checks without doctests.
CI uses the latest stable Rust, so new lints can appear there before they appear locally.

## Before a PR

Run these checks as well as the tests for your change:

```sh
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo xtask verify-oracle-boundary --base main
rustup toolchain install "$(cat xtask/api-nightly.txt)" --profile minimal
cargo xtask verify-ledger
cargo xtask verify-bump
cargo xtask api-baseline --check ledger/rust-api-baseline.json
```

Only API extraction uses the pinned nightly. Set `MLX_RS_API_TARGET_DIR` and
`MLX_RS_METAL_PATH` to scratch directories for its native builds and metallibs.

When touching FFI ownership, also run `cargo xtask verify-ffi`.

## Rules for every change

- An implementation change must not share a commit with oracle changes to goldens,
  tolerances, or comparator logic.
- Add no new `_device` twins or `generate_macro` operation macros.
- Use `with_stream` and `with_device` for stream selection.
- Commit subjects follow the existing `type(scope): subject` history.

## Read before changing

- Read [CHARTER.md](CHARTER.md) before adding or changing public API.
- Read [VERIFICATION.md](VERIFICATION.md) before touching `conformance/`, `ledger/`,
  the FFI gate, or an mlx-c bump.
