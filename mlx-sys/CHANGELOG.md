# CHANGELOG

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
