{ pkgs, ... }:

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
  ];

  env = {
    # bindgen finds libclang through this, not through PATH.
    LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
  };

  # Mirrors .github/workflows/validate.yml so a green local run means a green CI
  # run. MLX is not thread safe, hence --test-threads=1.
  scripts.validate.exec = ''
    set -e
    echo "=== fmt ==="
    cargo fmt -- --check
    echo "=== clippy ==="
    cargo clippy -- -D warnings
    echo "=== tests ==="
    cargo test --workspace -- --test-threads=1
    echo ""
    echo "✅ Validation complete!"
  '';

  scripts.validate-fast.exec = ''
    set -e
    cargo fmt -- --check
    cargo clippy -- -D warnings
    # Skip doctests: they dominate wall clock (196s of a ~220s run).
    cargo test --workspace --lib --tests -- --test-threads=1
  '';

  enterShell = ''
    # `xcrun metal` does not work from a devenv shell by default: nixpkgs points
    # DEVELOPER_DIR at its own apple-sdk and puts its own xcrun ahead of the system
    # one, and that combination resolves metal to a stub which exits with "missing
    # Metal Toolchain". The real compiler ships as a separate Xcode component in a
    # cryptex mount only the system Xcode can find.
    #
    # Keep the override inside the shim below. Exporting SDKROOT shell-wide makes
    # nix's clang++ mix nix libcxx headers with the system SDK, which breaks
    # esaxx-rs via tokenizers.
    system_developer_dir="$(readlink /var/db/xcode_select_link 2>/dev/null || true)"
    if [ -z "$system_developer_dir" ]; then
      system_developer_dir="$(DEVELOPER_DIR= /usr/bin/xcode-select -p 2>/dev/null || true)"
    fi

    xcrun_shim_dir="$DEVENV_STATE/system-xcrun"
    mkdir -p "$xcrun_shim_dir"
    cat > "$xcrun_shim_dir/xcrun" <<SHIM
#!/bin/sh
# Defer to the system Xcode so the Metal toolchain is visible. Scoped to this
# process: the surrounding shell keeps devenv's SDK for everything else.
DEVELOPER_DIR="$system_developer_dir"
export DEVELOPER_DIR
unset SDKROOT
exec /usr/bin/xcrun "\$@"
SHIM
    chmod +x "$xcrun_shim_dir/xcrun"
    # MLX's CMakeLists invokes `xcrun` through /bin/sh, so the shim has to win
    # on PATH rather than merely be called by absolute path here.
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

    # To stderr, not stdout: `xtask verify-ffi` writes JSON to stdout and a
    # banner in front of it makes the output unparseable.
    echo "mlx-rs — Rust bindings for MLX" >&2
    echo "  validate        fmt + clippy + full test suite (matches CI)" >&2
    echo "  validate-fast   the same without doctests" >&2
    echo "  cargo build --workspace       builds mlx-c + MLX (slow the first time)" >&2
    echo "  cargo run -p xtask [tag]      diff mlx-c bindings against a newer tag" >&2
  '';
}
