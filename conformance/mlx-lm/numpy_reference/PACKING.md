# Affine packing

MLX 0.32.2 stores affine weights as `<name>.weight` (uint32), `<name>.scales`,
and `<name>.biases`. The optional linear `<name>.bias` is separate.

Four-bit weights pack eight elements per word: element k occupies bits 4k through
4k+3, least significant nibble first. Eight-bit weights pack four elements per
word, with element k at bits 8k through 8k+7. Safetensors words are little-endian.

Unpack along the last axis, then compute `w = scale * q + bias`. Scales and biases
have one entry per group along that axis. The quant4 fixtures use group size 32.

Source: `/Users/ci/hub/scratch/mlx-lm/t1/affine-packing.md`, probed against
`mx.dequantize` on MLX 0.32.2. The reference reads the packed fixture parameters
directly and performs these operations in NumPy f32.
