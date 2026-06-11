//! Llama [`LanguageModel`] adapter: wraps the llama model graph + its
//! KV cache behind the family-agnostic runtime trait.

use std::path::Path;

use mlx_rs::{module::Module, ops::indexing::IndexOp, Array};

use crate::cache::KVCache;
use crate::chat_template::ChatTemplate;
use crate::config::ModelConfig;
use crate::error::Error;
use crate::family::{EosSpec, LoadedContext};
use crate::language_model::{LanguageModel, TextOnlyProcessor};
use crate::llama::text::config::ModelArgs;
use crate::llama::text::model::{load_llama_model, load_llama_tokenizer, Model};
use crate::lm_input::{LMInput, LMOutput, PrepareResult};
use crate::loader::resolve_bos_id;
use crate::nn::ModelInput;

struct LlamaAdapter {
    model: Model,
    cache: Vec<Option<KVCache>>,
    vocab_size: i32,
}

impl LanguageModel for LlamaAdapter {
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

    let model = load_llama_model(dir)?;
    let tokenizer = load_llama_tokenizer(dir)?;
    let bos_id = resolve_bos_id(dir, &tokenizer);
    let chat_template = ChatTemplate::from_dir(dir)?;
    let processor = TextOnlyProcessor::new("llama", tokenizer, chat_template, bos_id);

    let adapter = LlamaAdapter {
        model,
        cache: Vec::new(),
        vocab_size,
    };
    Ok((Box::new(adapter), Box::new(processor), eos_ids))
}
