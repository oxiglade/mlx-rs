//! Family-agnostic speculative-decode primitives (draft/verify sampling +
//! cache rollback), shared by MTP heads and draft-model drafters.

pub mod sampling;
pub mod snapshot;

pub use sampling::{
    accept_mask, draft_confidence, draft_gate_for, resample_on_reject, sample_draft,
    top_p_keep_mask,
};
pub use snapshot::CacheSnapshot;
