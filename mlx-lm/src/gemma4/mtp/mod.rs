//! Gemma 4 MTP (multi-token-prediction) speculative decode: a separate
//! `gemma4_assistant` drafter checkpoint that shares the target's KV cache +
//! last hidden, drafts γ tokens, and the target verifies them in parallel.

pub mod centroid;
pub mod config;
pub mod decode;
pub mod drafter;
pub mod weights;

pub use config::DrafterConfig;
pub use decode::{mtp_step, MtpContext};
pub use drafter::Drafter;
pub use weights::load_drafter;
