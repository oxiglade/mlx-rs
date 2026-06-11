//! Qwen3.5 text model: config, RoPE, attention/MLP, the gated-delta-net
//! linear-attention path, caches, decoder stack, weights, dense adapter.

pub mod adapter_dense;
pub mod adapter_moe;
pub mod cache;
pub mod config;
pub mod gated_delta;
pub mod gated_delta_block;
pub mod layer;
pub mod moe;
pub mod rope;
#[allow(
    clippy::module_inception,
    reason = "text-family core types live in text.rs"
)]
pub mod text;
pub mod weights;

pub(crate) use adapter_dense::load_context_dense;

use std::path::Path;

use crate::chat_template::ChatTemplate;
use crate::error::Error;
use crate::family::EosSpec;
use crate::loader::load_tokenizer;
use config::ModelConfig;

/// EOS ids: config `eos_token_id`, plus the chat-template `<|im_end|>`.
fn eos_ids(cfg: &ModelConfig) -> Vec<u32> {
    let mut ids = EosSpec::to_vec(cfg.eos_token_id.as_ref());
    if !ids.contains(&config::QWEN_CHAT_EOS_TOKEN_ID) {
        ids.push(config::QWEN_CHAT_EOS_TOKEN_ID);
    }
    ids
}

/// Load tokenizer + chat template + EOS ids shared by qwen3.5 adapters.
pub(crate) fn load_common(
    cfg: &ModelConfig,
    dir: &Path,
) -> Result<(tokenizers::Tokenizer, ChatTemplate, Vec<u32>), Error> {
    let tokenizer = load_tokenizer(dir)?;
    let chat_template = ChatTemplate::from_dir(dir)?;
    Ok((tokenizer, chat_template, eos_ids(cfg)))
}

/// Build a leftover-keys error after weight binding (`family` = dense/moe/vlm).
pub(crate) fn leftover_keys_error(family: &str, leftover: &[String]) -> Error {
    let mut names: Vec<&str> = leftover.iter().map(String::as_str).collect();
    names.sort_unstable();
    Error::config(format!(
        "qwen3.5 {family}: {} weights not bound to params: {:?}",
        names.len(),
        names
    ))
}
