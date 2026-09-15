# CLI implementation plan

The approved design is item 4 and §2.8 of
`/Users/ci/hub/scratch/mlx-lm/t4/position-astra.md`, overridden by DECISIONS A–M.
Only `mlx-lm-cli/**` may change; no commits, downloads, Metal execution, or Python MLX.

1. Add the unpublished binary manifest and `args.rs`. Test defaults, exact prompt/stop
   preservation, numeric ranges, source exclusivity, and feature-gated Hub grammar.
2. Add `output.rs` and `error.rs`. Test exact JSONL order, empty deltas, all public
   configuration encodings, short writes, Interrupted, write/flush failures and BrokenPipe.
3. Wire strict public loading and generation in `main.rs`, with stderr diagnostics,
   syntax exit 2, runtime exit 1, and BrokenPipe exit 0.
4. Add offline process tests using committed fixtures. Compile all tests; execute
   pure tests only. Record each unexecuted test and absent integration seam.
5. Document grammar/schema, add the exact workspace integration entry to the handoff,
   and inspect formatting, dependency boundary, and ownership diff.

Standalone verification uses an ignored manifest in `.verify/` with source/test paths
pointing at this package and a path dependency on the unchanged library. The delivered
manifest remains ready for the root workspace member entry.
