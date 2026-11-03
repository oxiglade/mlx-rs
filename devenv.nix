{ pkgs, lib, ... }:

{
  # rust-version in Cargo.toml is 1.85.0; CI builds both that and stable.
  languages.rust = {
    enable = true;
    channel = "stable";
    components = [ "rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" ];
  };

  packages = with pkgs; [
    # mlx-sys builds mlx-c (and, through it, MLX itself) via the cmake crate.
    cmake
    ninja
    # xtask regenerates mlx-sys bindings with bindgen, which needs libclang.
    libclang.lib
    git
    # Oracle side of the conformance corpus: generate.py runs under a venv
    # pinned by conformance/requirements.lock (mlx==0.30.6 — the same MLX the
    # bindings link — plus numpy).
    python312
  ];

  env = {
    # bindgen finds libclang through this, not through PATH.
    LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
  };

  # Mirrors .github/workflows/validate.yml so a green local run means a green CI run.
  scripts.validate.exec = ''
    set -e
    echo "=== fmt ==="
    cargo fmt --manifest-path xtask/Cargo.toml -- --check
    cargo clippy --all-targets --manifest-path xtask/Cargo.toml -- -D warnings
    cargo test --manifest-path xtask/Cargo.toml
    cargo fmt -- --check
    echo "=== clippy ==="
    cargo clippy --all-targets -- -D warnings
    echo "=== tests ==="
    cargo test --workspace
    echo ""
    echo "✅ Validation complete!"
  '';

  scripts.validate-fast.exec = ''
    set -e
    cargo fmt --manifest-path xtask/Cargo.toml -- --check
    cargo clippy --all-targets --manifest-path xtask/Cargo.toml -- -D warnings
    cargo test --manifest-path xtask/Cargo.toml
    cargo fmt -- --check
    cargo clippy --all-targets -- -D warnings
    # Skip doctests: they dominate wall clock (196s of a ~220s run).
    cargo test --workspace --lib --tests
  '';

  enterShell = lib.optionalString pkgs.stdenv.isDarwin ''
    # nixpkgs supplies 14.0; target the host for Rust and MLX.
    # Outside devenv, only MLX defaults to the host.
    export MLX_SYS_SYSTEM_SDK=1
    export MACOSX_DEPLOYMENT_TARGET="$(/usr/bin/sw_vers -productVersion)"

    # `xcrun metal` does not work from a devenv shell by default: nixpkgs points
    # DEVELOPER_DIR at its own apple-sdk and puts its own xcrun ahead of the system
    # one, and that combination resolves metal to a stub which exits with "missing
    # Metal Toolchain". The real compiler ships as a separate Xcode component in a
    # cryptex mount only the system Xcode can find.
    #
    # mlx-sys bypasses this shim for its CMake build and uses the system SDK for
    # both C++ and Metal. Other crates keep Nix's SDK.
    # Exporting SDKROOT shell-wide mixes Nix libcxx headers with the system SDK,
    # which breaks esaxx-rs via tokenizers.
    xcrun_shim_dir="$DEVENV_STATE/system-xcrun"
    mkdir -p "$xcrun_shim_dir"
    cat > "$xcrun_shim_dir/xcrun" <<'SHIM'
#!/bin/sh
# Defer to the system Xcode so the Metal toolchain is visible. Scoped to this
# process: the surrounding shell keeps devenv's SDK for everything else.
#
# SDK-version queries are the exception: they must describe the headers the
# nix compiler actually uses, or SDK-gated code enables against missing headers.
case " $* " in
  *" --show-sdk-version "*|*" --show-sdk-platform-version "*)
    exec /usr/bin/xcrun "$@" ;;
esac
DEVELOPER_DIR="$(env -u DEVELOPER_DIR /usr/bin/xcode-select -p)"
export DEVELOPER_DIR
unset SDKROOT
exec /usr/bin/xcrun "$@"
SHIM
    chmod +x "$xcrun_shim_dir/xcrun"

    # Nix's linker and SDK cannot resolve the newer symbols in system-built MLX.
    # Rust links with system libc++ and ignores NIX_LDFLAGS; native builds keep Nix.
    cat > "$xcrun_shim_dir/rust-linker" <<'SHIM'
#!/bin/sh
export DEVELOPER_DIR="$(env -u DEVELOPER_DIR /usr/bin/xcode-select -p)"
unset SDKROOT
exec /usr/bin/clang "$@"
SHIM
    chmod +x "$xcrun_shim_dir/rust-linker"
    export CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER="$xcrun_shim_dir/rust-linker"
    export CARGO_TARGET_X86_64_APPLE_DARWIN_LINKER="$xcrun_shim_dir/rust-linker"

    export PATH="$xcrun_shim_dir:$PATH"

    # Probe by *running* metal, not by locating it: the stub above resolves
    # fine under `xcrun -f` and only fails on execution, so a which-style check
    # reports success and the build then dies inside cmake with a grep error
    # that names no cause.
    if ! echo "__METAL_VERSION__" | xcrun -sdk macosx metal -E -x metal -P - >/dev/null 2>&1; then
      echo "warning: the Metal compiler is unavailable, so MLX will not build." >&2
      echo "  Xcode 16+ ships it as a separately downloaded component:" >&2
      echo "    xcodebuild -downloadComponent MetalToolchain" >&2
    fi

  '' + ''
    # To stderr, not stdout: `xtask verify-ffi` writes JSON to stdout and a
    # banner in front of it makes the output unparseable.
    echo "mlx-rs — Rust bindings for MLX" >&2
    echo "  validate        fmt + clippy + full test suite (matches CI)" >&2
    echo "  validate-fast   the same without doctests" >&2
    echo "  cargo build --workspace       builds mlx-c + MLX (slow the first time)" >&2
    echo "  cargo xtask <command>         runs repository verification tools" >&2
  '';
}
