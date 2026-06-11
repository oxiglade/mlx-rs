//! Llama text family.

pub mod text;

use std::path::Path;

use crate::config::ModelConfig;
use crate::error::Error;
use crate::family::LoadedContext;

pub(crate) fn load_context(cfg: &ModelConfig, dir: &Path) -> Result<LoadedContext, Error> {
    let env = cfg
        .family
        .as_llama()
        .ok_or_else(|| Error::config("llama::load_context: wrong family"))?;
    text::adapter::load_context(cfg, env, dir)
}
