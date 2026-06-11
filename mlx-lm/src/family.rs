//! Family-agnostic config helpers.

use serde::Deserialize;

use crate::language_model::{LanguageModel, UserInputProcessor};

/// What every family's `load_context` returns: the boxed model +
/// processor + EOS-id list. `crate::model_context::load` wraps it into
/// a `ModelContext`.
pub type LoadedContext = (
    Box<dyn LanguageModel>,
    Box<dyn UserInputProcessor>,
    Vec<u32>,
);

/// `config.json::eos_token_id` — either a single id or a list. Each
/// family envelope carries `Option<EosSpec>` so the value is parsed
/// once at load and read off the typed struct afterwards.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosSpec {
    Single(u32),
    Many(Vec<u32>),
}

impl EosSpec {
    /// Flatten to a `Vec<u32>`; empty when `None`.
    pub fn to_vec(spec: Option<&Self>) -> Vec<u32> {
        match spec {
            Some(Self::Single(id)) => vec![*id],
            Some(Self::Many(ids)) => ids.clone(),
            None => Vec::new(),
        }
    }
}
