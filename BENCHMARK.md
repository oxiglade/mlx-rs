# mlx-lm decode bench

## Running

```
cargo bench -p mlx-lm --bench lm_decode
```

Single cell:

```
MLX_LM_BENCH_ONLY=qwen3_decode_large_bf16 cargo bench -p mlx-lm --bench lm_decode
```

## Environment knobs

- `MLX_LM_BENCH_CACHE` — checkpoint cache root (default `~/.cache/mlx-rs-bench`).
- `MLX_LM_BENCH_NO_DOWNLOAD=1` — skip cells whose checkpoint isn't cached.
- `MLX_LM_BENCH_SET={trimmed,full}` — `trimmed` (default) runs llama 1B + qwen3 1.7B; `full` adds llama 3B + qwen3 0.6B.
- `MLX_LM_BENCH_ONLY=<substr>` — substring filter on per-cell group prefix.

Checkpoints download via `hf` CLI on first use; cells skip silently if `hf` is unavailable or download fails.

## Cells

- `llama_decode_small_{bf16,q8,q4}` — `mlx-community/Llama-3.2-1B-Instruct-{bf16,8bit,4bit}`
- `qwen3_decode_large_{bf16,q8,q4}` — `mlx-community/Qwen3-1.7B-{bf16,8bit,4bit}`

Each cell runs `prefill_short` (13-token prompt), `prefill_long` (1024), `decode_short` (99 tokens after short prompt), `decode_long` (99 after long prompt).

Methodology: criterion 10-sample × 20 s window. `WARMUP_TOKENS = 4` decode steps outside timing; `DECODE_TOKENS = 100` timed. Decode drives the production `Generate` iterator (which submits step N+1 before yielding N) and fences each token with `eval`, not `.item()` — `.item()` reads back to host, adding a per-token coherence barrier that hides the GPU decode cost.

## Results

Median times in milliseconds. Hardware: Apple Silicon laptop. Each cell run as
an isolated process (`MLX_LM_BENCH_ONLY=<cell>`) so model load, kernel cache,
and mlx-c state are fresh per measurement.

### Branch HEAD (mlx-c v0.31.2)

| cell | prefill_short (13) | prefill_long (1024) | decode_short (99) | decode_long (99) |
|---|---|---|---|---|
| llama_decode_small_bf16 | 12.80 | 215.13 | 563.90 | 582.50 |
| llama_decode_small_q8 | 10.32 | 250.85 | 337.37 | 391.30 |
| llama_decode_small_q4 | 9.78 | 258.94 | 231.89 | 277.95 |
| qwen3_decode_large_bf16 | 19.82 | 304.10 | 842.22 | 877.93 |
| qwen3_decode_large_q8 | 18.29 | 400.63 | 585.87 | 575.31 |
| qwen3_decode_large_q4 | 16.09 | 401.92 | 363.42 | 393.53 |

Quantized (q4/q8) cells have no pre-bump baseline: loading a v0.31
`QuantizedLinear` checkpoint requires the `.weight → .inner.weight` redirect
and the `rms_norm` weight handling introduced with the mlx-c bump. Before the
bump those cells fail at load, so a baseline-vs-HEAD delta exists for bf16 only.

### mlx-c version cost (bf16, baseline → bump → HEAD)

Baseline = `ce7f7a51` (bench harness, mlx-c v0.5.0). Bump = `faa3ace` (mlx-c
v0.31.2, no later perf work). HEAD = full branch.

llama_decode_small_bf16:

| phase | v0.5.0 | v0.31.2 (bump) | HEAD | net Δ |
|---|---|---|---|---|
| prefill_short | 12.81 | — | 12.80 | −0.0% |
| prefill_long | 212.59 | — | 215.13 | +1.2% |
| decode_short | 560.90 | — | 563.90 | +0.5% |
| decode_long | 596.33 | — | 582.50 | −2.3% |

qwen3_decode_large_bf16:

| phase | v0.5.0 | v0.31.2 (bump) | HEAD | net Δ |
|---|---|---|---|---|
| prefill_short | 18.84 | 18.87 | 19.82 | +5.2% |
| prefill_long | 301.10 | 299.27 | 304.10 | +1.0% |
| decode_short | 809.83 | 824.74 | 842.22 | +4.0% |
| decode_long | 811.32 | 903.50 | 877.93 | +8.2% |

The qwen3 `decode_long` cost enters at the **mlx-c bump** (811 → 904, +11%),
not in the Rust layer; the later cache-cap + pre-allocated KVCache work recovers
~3% of it (904 → 878). llama (no q/k RmsNorm in attention) is flat across the
bump. The remaining qwen3 regression is an upstream mlx-c v0.5 → v0.31 kernel
characteristic for this attention shape, accepted as the cost of the version
bump (which is what unlocks quantized-checkpoint loading above).
