# Hub integration seams

The item-3 source consumes the following declaration patch. These files are
outside this item's ownership; the patch is supplied only in the temporary
verification copy.

In `src/model.rs`, declare the module and provenance type:

```rust
#[cfg(feature = "hf-hub")]
mod hub;

/// Resolved Hub source identity.
#[cfg(feature = "hf-hub")]
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct HubProvenance {
    /// Repository identifier.
    pub repo: String,
    /// Requested revision, with an absent request spelled `main`.
    pub requested_revision: String,
    /// Resolved full commit SHA.
    pub resolved_revision: String,
}
```

Add this private field to `Model`:

```rust
#[cfg(feature = "hf-hub")]
hub_provenance: Option<HubProvenance>,
```

Initialize it to `None` in `Model::from_dir`, `model::gguf::load`, and the scripted
model constructor in `src/model/tests.rs`. `hub::load` sets it after local loading
and a second receipt validation. The permitted `from_hub` body and accessor are
already updated in the item-3 patch.

In `src/lib.rs`, change the gated export to:

```rust
#[cfg(feature = "hf-hub")]
pub use model::{HubOptions, HubProvenance};
```

Remove the unused `NotYetImplemented` struct and its `Display` implementation.

Add the paper's variants to `HubError` in `src/error.rs`, with public field
documentation and `thiserror` display attributes:

```rust
InvalidRepository(String),
MissingFile { repo: String, revision: String, filename: String },
UnsafePath { path: std::path::PathBuf },
RevisionMismatch { expected: String, actual: String },
SnapshotIntegrity { path: std::path::PathBuf, reason: String },
CacheDirectoryUnavailable,
```

The existing `OfflineCacheMiss`, `InvalidRevision`, `Api`, `Io`, and `Load`
variants are preserved and used. No shared shard-parser changes are needed:
`weights::ShardIndex::from_bytes` and `shard_paths` already exist.

In `mlx-lm/Cargo.toml`, add SHA-256 behind the same feature, then have the manifest
owner update the lockfile:

```toml
[features]
hf-hub = ["dep:hf-hub", "dep:sha2"]

[dependencies]
sha2 = { version = "0.10", optional = true }
```

## Receipt and test contracts

Private receipts live at
`models--<repo>/.mlx-lm/snapshots/<commit>.json`. Version 1 has `schema_version`,
`repo`, `commit`, `weight_mode` (`single` or `indexed`), `selected_files`
(`path`, byte `size`, lowercase `sha256`), and `absent_files`. Both file lists
are strictly sorted and unique. Single-file mode records index absence as well
as absent tokenizer sidecars. Indexed mode ignores any unselected single weight
file, as the local loader does. Completion is invalidated before downloading;
receipts and named refs are replaced atomically only after validation.

`conformance/mlx-lm/hub_cases.json` was absent in this worktree. The logical-plan
reader implements the paper's documented fields with the following spellings:
`schema_version: 1`, `plans`, and `mutations`; each plan has `name`,
`requested_ref`, `returned_commit`, `siblings`, `selected_files`,
`recorded_absences`, `expected_transport_calls`, `expected_provenance`,
`expected_error`, and an optional `weight_map`. Calls are strings
`info:org/model@<ref>` and `download:org/model@<commit>:<filename>`. Error names
match the HubError variants. Mutation records use `name`; unknown names fail.
The fixture owner must reconcile these spellings with the frozen file and remove
the two explicit missing-fixture ignores when supplying it. No frozen-fixture
conformance is claimed by the inline schema test.

NOT RUN because that file is absent:

- `model::hub::tests::hub_cases_snapshot_plans`
- `model::hub::tests::hub_cases_mutations`

Model-loading tests to run on the host, compiled but not executed in the sandbox:

- `public_offline_single_snapshot_provenance`
- `public_offline_indexed_snapshot_provenance`
- `public_offline_standard_blob_snapshot_provenance`
- `public_gguf_provenance_is_none`

The private resolver tests run with an in-process fake, including exact call logs.
Offline resolution receives a factory that panics on construction and a transport
whose methods panic. The public constructor uses that same resolver, with its
production factory invoked only after the offline branch returns. No public test
injection API is added. The anonymous-client test inspects a prepared request's
Authorization header without sending it.

## Verification result

Verified against base `2ef76383` in
`/private/tmp/mlx-lm-t4b-hub-verify-l2tf8rpk`, with the declaration patch above:

- 20 private Hub tests passed; the two missing-fixture tests were ignored by name.
- Four public pure offline tests and three shared shard-parser tests passed.
- `cargo test --offline -p mlx-lm --all-features --all-targets --no-run` passed.
- Both requested `cargo check` feature combinations passed.
- `cargo clippy --offline -p mlx-lm --all-targets --all-features -- -D warnings` passed.
- Formatting checks for all changed Rust files and `git diff --check` passed.

The SDK/compiler override used `/usr/bin`,
`SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`, and
`DEVELOPER_DIR=/Library/Developer/CommandLineTools`. Native dependency fetching
was unavailable, so the temporary copy's build script reused MLX archives and
bindings from
`/Users/ci/hub/scratch/verify-target-final/debug/build/mlx-sys-a2784c607c853d3e/out`.
No native build-script changes were made in the worktree. Native source rebuild,
Metal execution, and Python MLX execution are not claimed.

`verification.json` in the temporary copy records the commands' results and all
test names. `unexecuted-tests.txt` lists all 180 compiled tests not executed in
this item, including the six named exceptions above and unrelated existing
suites. The four public model-loading tests remain enabled for normal host runs.

Independent review found a symlink-parent traversal defect; new regressions
demonstrated it before the fix. Validation now checks every visited path
component, rejects excursions through refs or another snapshot, and verifies
that the original path is an openable regular file resolving to the same target.
The final review found no remaining important issues. No commit was created.
