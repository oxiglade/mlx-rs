//! Qwen3 [`LanguageModel`] adapter: wraps the qwen3 model graph + its
//! KV cache behind the family-agnostic runtime trait.

use std::path::Path;

use mlx_rs::{module::Module, ops::indexing::IndexOp, Array};

use crate::cache::KVCache;
use crate::chat_template::ChatTemplate;
use crate::config::ModelConfig;
use crate::error::Error;
use crate::family::{EosSpec, LoadedContext};
use crate::language_model::{LanguageModel, TextOnlyProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult};
use crate::loader::resolve_bos_id;
use crate::nn::ModelInput;
use crate::qwen3::text::config::ModelArgs;
use crate::qwen3::text::model::{load_qwen3_model, load_qwen3_tokenizer, Model};

struct Qwen3Adapter {
    model: Model,
    cache: Vec<Option<KVCache>>,
    vocab_size: i32,
}

impl LanguageModel for Qwen3Adapter {
    fn reset(&mut self) {
        self.cache.clear();
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        let tokens = input.text.tokens;
        let logits = self.model.forward(ModelInput {
            inputs: &tokens,
            mask: None,
            cache: &mut self.cache,
        })?;
        Ok(PrepareResult::Logits(logits.index((.., -1, ..))))
    }

    fn step(&mut self, last_token: &Array) -> Result<LMOutput, Error> {
        let inputs = last_token.reshape(&[1, 1])?;
        let logits = self.model.forward(ModelInput {
            inputs: &inputs,
            mask: None,
            cache: &mut self.cache,
        })?;
        Ok(LMOutput {
            logits: logits.index((.., -1, ..)),
        })
    }

    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }
}

pub(crate) fn load_context(
    _cfg: &ModelConfig,
    args: &ModelArgs,
    dir: &Path,
) -> Result<LoadedContext, Error> {
    let vocab_size = args.vocab_size;
    let eos_ids = EosSpec::to_vec(args.eos_token_id.as_ref());

    let model = load_qwen3_model(dir)?;
    let tokenizer = load_qwen3_tokenizer(dir)?;
    let bos_id = resolve_bos_id(dir, &tokenizer);
    let chat_template = ChatTemplate::from_dir(dir)?;
    let processor = TextOnlyProcessor::new("qwen3", tokenizer, chat_template, bos_id);

    let adapter = Qwen3Adapter {
        model,
        cache: Vec::new(),
        vocab_size,
    };
    Ok((Box::new(adapter), Box::new(processor), eos_ids))
}
