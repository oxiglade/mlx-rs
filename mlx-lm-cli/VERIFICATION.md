# Tranche 4b CLI verification

Worktree base: `2ef76383`. Only `mlx-lm-cli/**` changed. No commit was made.
The root workspace member entry remains `"mlx-lm-cli",` for the integration owner.

## Executed

Rust 1.98.0; `CARGO_TARGET_DIR=/private/tmp/codex-target` retained. The root manifest
was not edited. An ignored standalone manifest in `.verify/` points at the actual
CLI source/tests and unchanged public library, with an empty `[workspace]` of its
own. No library source or public declarations were patched for verification.

The first Cargo `--offline` build encountered CMake FetchContent, which attempted
to clone MLX and failed on DNS. No download succeeded. Subsequent builds used
CMake's `MLX_C_USE_SYSTEM_MLX=ON` and `FETCHCONTENT_FULLY_DISCONNECTED=ON`, with the
existing local MLX v0.32.2 archive under
`/Users/ci/hub/scratch/verify-target/debug/build/mlx-sys-a2784c607c853d3e/out/build`.
The MLX source tag was checked as `v0.32.2`. This worktree's C wrapper was compiled.
CMake staged the native archives in the preset target directory. Its missing
metallib warning was retained; no model initialization or Metal test was attempted.
No Python MLX import, checkpoint download, or real checkpoint was used.

Passed checks:

- All no-default-feature binary and process-test targets compiled.
- All 16 no-default-feature unit tests passed.
- Five pure process tests passed: `disabled_hub_grammar`,
  `help_and_syntax_keep_stdout_clean`, `invalid_options_rejected_before_load`,
  `missing_model_files`, `missing_weights_after_valid_sidecars`.
- Eight Hub-feature argument tests passed in an ignored argument-only harness
  containing the unchanged argument module/test source. This used the real
  `mlx-lm` dependency with `hf-hub`, not a replacement API. It does not establish
  that the full Hub CLI binary compiles.
- No-default-feature Clippy, all targets, with `-D warnings` passed.
- `rustfmt --edition 2021 --check` passed for all source and tests.
- Normal direct dependencies were exactly clap, mlx-lm, serde, serde_json,
  and thiserror. No direct mlx-rs dependency or oracle-hooks feature.
- A separate read-only code review found no further correctness defects.

The final pure-test command, after supplying the local native CMake configuration:

```sh
cargo test --manifest-path mlx-lm-cli/.verify/Cargo.toml --offline \
  --no-default-features -- --test-threads=1 \
  --skip local_llama_text_generation --skip local_llama_jsonl_generation \
  --skip local_qwen3_quantized_info --skip sharded_load --skip early_broken_pipe
```

Hub compilation was attempted with `--features hf-hub --no-run`. It failed with
E0599 at `model.hub_provenance()` in `src/output.rs`. The exact required public
signature/type are in README.md. `from_hub` is also still a stub in this worktree.

## Written tests not executed

These process tests compiled without Hub but were NOT RUN because model loading
or generation requires MLX initialization:

- `local_llama_text_generation`
- `local_llama_jsonl_generation`
- `local_qwen3_quantized_info`
- `sharded_load`
- `early_broken_pipe`

The full Hub binary/test targets could not compile without the provenance seam.
All Hub-mode process tests were therefore NOT RUN:

- `local_llama_text_generation`
- `local_llama_jsonl_generation`
- `local_qwen3_quantized_info`
- `sharded_load`
- `invalid_options_rejected_before_load`
- `missing_model_files`
- `missing_weights_after_valid_sidecars`
- `help_and_syntax_keep_stdout_clean`
- `enabled_offline_hub_miss`
- `early_broken_pipe`

The following output unit tests passed without Hub but were NOT RUN in a Hub
binary. Their feature-independent writer code is covered by the default run;
the Hub provenance accessor itself remains unverified:

- `output::tests::config_encoding_covers_public_variants_and_sorts_paths`
- `output::tests::info_root_schema_and_writer_errors`
- `output::tests::info_writers_retry_and_preserve_io_failures`
- `output::tests::jsonl_exact_schema_order_and_empty_final_delta`
- `output::tests::text_is_exact_deltas_with_one_flush_each`
- `output::tests::writers_preserve_write_and_flush_error_kinds`
- `output::tests::writers_retry_interrupted_and_short_writes`
- `output::tests::zero_write_is_output_failure`

These Hub argument cases ran only in the separate harness, not the full binary:
`defaults_match_library`, `exactly_two_commands_and_explicit_source`,
`generate_accepts_explicit_local_source`, `hub_flags_require_repo_and_translate`,
`min_p_defaults_to_one_and_accepts_endpoints`, `rejects_invalid_options_before_loading`,
`translates_every_sampling_flag_and_literal_stops`, and
`vocabulary_limits_are_left_to_public_validation`.

## Integration gaps

`conformance/mlx-lm/cli_cases.json` is absent. The written process tests use the
specified version-1 schema and public Text-generation events, without claiming
independent oracle parity. Golden comparison is NOT RUN for the five fixture
process tests named above.

Enabled offline Hub success coverage is NOT IMPLEMENTED, blocked on item 3's
completed-snapshot receipt contract and item 1's reviewed `hub_cases.json`, both
absent here. Constructing an accepted offline cache requires that private data
contract; the public API has no offline cache-construction operation. Integration
must supply a tiny completed-cache recipe, then add the corresponding successful
process assertion. The CLI's Hub loading/options/provenance code is implemented
against the approved public signatures; no extra library API is requested.
