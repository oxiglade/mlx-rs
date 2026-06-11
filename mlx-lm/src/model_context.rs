//! The runtime: [`load`] + [`ModelContext`] + [`generate`].
//!
//! [`load`] parses `config.json`, dispatches on the typed family, and
//! returns a [`ModelContext`]. [`generate`] runs the full
//! prepare → sample → step loop with optional per-token streaming.

use std::ops::ControlFlow;
use std::path::Path;

use mlx_rs::{memory::clear_cache, ops::indexing::IndexOp, transforms::async_eval, Array};

use crate::config::{Family, ModelConfig};
use crate::error::Error;
use crate::family::LoadedContext;
use crate::language_model::{LanguageModel, UserInputProcessor};
use crate::lm_input::{LMInput, PrepareResult, Text};
use crate::sampler::{Sampler, SamplerState};
use crate::user_input::UserInput;
use crate::{gemma4, gemma4_unified, llama, qwen3, qwen3_5};

/// Sampling + stopping knobs handed to [`generate`].
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// Max new tokens (excluding prompt). The loop exits early on EOS.
    pub max_new_tokens: i32,
    /// Sampling strategy: greedy / categorical / nucleus.
    pub sampling: Sampler,
    /// Stop tokens beyond the model-default EOS list.
    pub extra_stop_ids: Vec<u32>,
    /// Force the non-MTP path even on MTP models (parity A/B).
    pub disable_mtp: bool,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sampling: Sampler::default(),
            extra_stop_ids: Vec::new(),
            disable_mtp: false,
        }
    }
}

/// Reason a [`generate`] call returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// An EOS token (or a caller `extra_stop_id`) was sampled.
    Stop,
    /// `max_new_tokens` reached, or the streaming callback broke.
    Length,
}

/// Output of one [`generate`] call.
#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub text: String,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub finish_reason: FinishReason,
}

/// Per-token streaming callback; receives each token id + its UTF-8
/// delta. Return `ControlFlow::Break` to stop early.
pub type TokenCallback<'cb> = dyn FnMut(u32, &str) -> ControlFlow<()> + 'cb;

/// The loaded model + its preprocessor + the stop-token list.
pub struct ModelContext {
    /// The boxed model; owns its KV cache. [`generate`] resets it per turn.
    pub model: Box<dyn LanguageModel>,
    /// The boxed input processor (tokenise, chat template, modality).
    pub processor: Box<dyn UserInputProcessor>,
    /// Terminal token ids from `config.json::eos_token_id`.
    pub eos_ids: Vec<u32>,
}

impl ModelContext {
    /// Drop the model + processor and unmap mlx-core's buffer cache.
    /// Dropping alone returns buffers to the reuse pool but keeps it
    /// alive (right for REPL/server reuse); this also clears the pool,
    /// for consumers that load + drop many distinct models.
    pub fn unload(self) {
        drop(self);
        clear_cache();
    }
}

/// Parse `<dir>/config.json` once and dispatch to the family loader.
pub fn load(dir: impl AsRef<Path>) -> Result<ModelContext, Error> {
    load_with_drafter(dir, None)
}

/// Like [`load`] but with an optional MTP drafter (assistant) checkpoint dir.
/// Only the gemma4 / gemma4_unified families consume it; passing a drafter for
/// any other family is an error.
pub fn load_with_drafter(
    dir: impl AsRef<Path>,
    draft_dir: Option<&Path>,
) -> Result<ModelContext, Error> {
    let dir = dir.as_ref();
    let cfg = ModelConfig::from_dir(dir)?;
    let (model, processor, eos_ids) = dispatch_load(&cfg, dir, draft_dir)?;
    Ok(ModelContext {
        model,
        processor,
        eos_ids,
    })
}

/// Route the typed [`Family`] to its family `load_context`. No string
/// compare reaches runtime; serde picked the variant at parse.
fn dispatch_load(
    cfg: &ModelConfig,
    dir: &Path,
    draft_dir: Option<&Path>,
) -> Result<LoadedContext, Error> {
    if draft_dir.is_some() && !matches!(cfg.family, Family::Gemma4(_) | Family::Gemma4Unified(_)) {
        return Err(Error::config(
            "an MTP drafter is only supported for the gemma4 families",
        ));
    }
    match &cfg.family {
        Family::Llama(_) => llama::load_context(cfg, dir),
        Family::Qwen3(_) => qwen3::load_context(cfg, dir),
        Family::Qwen35(_) | Family::Qwen35Moe(_) | Family::Qwen35Vl(_) => {
            qwen3_5::load_context(cfg, dir)
        }
        Family::Gemma4(_) => gemma4::load_context(cfg, dir, draft_dir),
        Family::Gemma4Unified(_) => gemma4_unified::load_context(cfg, dir, draft_dir),
    }
}

/// Per-token streaming decoder. Sliding window over the last `WINDOW`
/// tokens with a BPE-merge fallback — bounded work per token instead of
/// the naive O(N²) full re-decode.
struct IncrementalDecoder {
    ids: Vec<u32>,
    committed_tokens: usize,
    committed: String,
    window: String,
}

impl IncrementalDecoder {
    /// ≥ the longest BPE merge reaching into earlier tokens (4–5 in
    /// practice for Qwen / Llama).
    const WINDOW: usize = 8;

    fn with_capacity(cap: usize) -> Self {
        Self {
            ids: Vec::with_capacity(cap),
            committed_tokens: 0,
            committed: String::new(),
            window: String::new(),
        }
    }

    /// Push a token, return the new UTF-8 delta to stream.
    fn push(&mut self, token: u32, processor: &dyn UserInputProcessor) -> Result<String, Error> {
        self.ids.push(token);

        let new_window = processor.decode(&self.ids[self.committed_tokens..])?;
        let delta: String = if new_window.starts_with(self.window.as_str()) {
            new_window[self.window.len()..].to_owned()
        } else {
            new_window.clone()
        };
        self.window = new_window;

        if self.ids.len() - self.committed_tokens > Self::WINDOW {
            let lead_idx = self.committed_tokens;
            let after_lead = processor.decode(&self.ids[lead_idx + 1..])?;
            let mut lead_byte_len = self.window.len().saturating_sub(after_lead.len());
            while lead_byte_len > 0 && !self.window.is_char_boundary(lead_byte_len) {
                lead_byte_len -= 1;
            }
            if lead_byte_len > 0 {
                let moved = self.window.drain(..lead_byte_len).collect::<String>();
                self.committed.push_str(&moved);
                self.committed_tokens += 1;
            }
        }

        Ok(delta)
    }

    fn into_text(mut self) -> String {
        self.committed.push_str(&self.window);
        self.committed
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}

/// One pipelined decode step: forward on `pending`, sample, `async_eval`
/// the result so N+1 GPU compute overlaps the caller's sync on N. Sole
/// source of the N+1-before-N pattern, shared by `generate` and the bench.
pub fn decode_step(
    model: &mut dyn LanguageModel,
    sampler: &mut SamplerState,
    pending: &Array,
) -> Result<Array, Error> {
    let next_logits = model.step(pending)?.logits;
    let next = sampler.sample(&next_logits)?;
    async_eval([&next])?;
    Ok(next)
}

/// Run one prompt → tokens loop on `ctx`. Streaming is per-token via
/// `on_token`; pass `&mut |_, _| ControlFlow::Continue(())` to disable.
pub fn generate(
    ctx: &mut ModelContext,
    input: UserInput,
    params: GenerateParams,
    on_token: &mut TokenCallback<'_>,
) -> Result<GenerateResult, Error> {
    ctx.model.reset();

    let lm_input = ctx.processor.prepare(input)?;
    let prompt_tokens = lm_input.text.tokens.shape()[1];
    let initial_logits = run_prefill(ctx.model.as_mut(), lm_input)?;

    let vocab = ctx.model.vocab_size();
    let cap = params.max_new_tokens.max(0) as usize;
    let mut decoder = IncrementalDecoder::with_capacity(cap);
    let mut finish_reason = FinishReason::Length;

    if params.max_new_tokens == 0 {
        return Ok(GenerateResult {
            text: decoder.into_text(),
            prompt_tokens,
            completion_tokens: 0,
            finish_reason,
        });
    }

    let mut sampler = SamplerState::new(params.sampling);
    let mut pending_id = sampler.sample(&initial_logits)?;
    async_eval([&pending_id])?;

    // MTP runs on MTP models unless disabled. Greedy + MTP is
    // byte-identical to greedy; sampled MTP uses rejection sampling.
    let use_mtp = ctx.model.has_mtp() && !params.disable_mtp;

    if use_mtp {
        run_mtp_loop(
            ctx,
            pending_id,
            &params,
            &mut sampler,
            &mut decoder,
            &mut finish_reason,
            on_token,
            vocab,
        )?;
    } else {
        for _ in 0..params.max_new_tokens {
            // Submit N+1 before syncing on N — overlap the host
            // coherence sync with N+1 GPU compute.
            let next_pending = decode_step(ctx.model.as_mut(), &mut sampler, &pending_id)?;

            let id_i32 = pending_id.item::<i32>();
            if id_i32 < 0 || id_i32 >= vocab {
                return Err(Error::shape(format!(
                    "sampler returned out-of-vocab id {id_i32} (vocab = {vocab})"
                )));
            }
            let token = id_i32 as u32;
            pending_id = next_pending;

            if ctx.eos_ids.contains(&token) || params.extra_stop_ids.contains(&token) {
                finish_reason = FinishReason::Stop;
                break;
            }

            let delta = decoder.push(token, ctx.processor.as_ref())?;
            if matches!(on_token(token, &delta), ControlFlow::Break(())) {
                break;
            }
        }
    }

    let completion_tokens = decoder.len() as i32;
    Ok(GenerateResult {
        text: decoder.into_text(),
        prompt_tokens,
        completion_tokens,
        finish_reason,
    })
}

/// MTP self-speculative loop. `try_mtp_decode` commits 1–2 tokens per
/// call and returns the next pending token.
#[allow(clippy::too_many_arguments)]
fn run_mtp_loop(
    ctx: &mut ModelContext,
    initial_pending: Array,
    params: &GenerateParams,
    sampler: &mut SamplerState,
    decoder: &mut IncrementalDecoder,
    finish_reason: &mut FinishReason,
    on_token: &mut TokenCallback<'_>,
    vocab: i32,
) -> Result<(), Error> {
    let mut pending = initial_pending;
    let mut budget = params.max_new_tokens;
    while budget > 0 {
        let (tokens, next_pending) = ctx
            .model
            .try_mtp_decode(&pending, sampler)?
            .ok_or_else(|| Error::config("MTP loop on a model that no longer reports has_mtp"))?;
        if tokens.is_empty() {
            return Err(Error::config("MTP returned zero tokens"));
        }
        pending = next_pending;
        for token in tokens.iter().copied() {
            if token >= vocab as u32 {
                return Err(Error::shape(format!(
                    "MTP returned out-of-vocab id {token} (vocab = {vocab})"
                )));
            }
            if ctx.eos_ids.contains(&token) || params.extra_stop_ids.contains(&token) {
                *finish_reason = FinishReason::Stop;
                return Ok(());
            }
            let delta = decoder.push(token, ctx.processor.as_ref())?;
            if matches!(on_token(token, &delta), ControlFlow::Break(())) {
                return Ok(());
            }
            budget -= 1;
            if budget == 0 {
                return Ok(());
            }
        }
    }
    Ok(())
}

/// Run prefill: ingest the prompt, return the next-token logits (or
/// prime then `step` if the model defers logits). Prompts longer than
/// `prefill_chunk_size` feed all-but-last chunk through `prefill_chunk`.
fn run_prefill(model: &mut dyn LanguageModel, mut input: LMInput) -> Result<Array, Error> {
    let prompt_len = input.text.tokens.shape()[1];
    if let Some(window) = model.prefill_chunk_size() {
        if prompt_len > window {
            let tokens = input.text.tokens;
            let mut start = 0_i32;
            while prompt_len - start > window {
                let end = start + window;
                model.prefill_chunk(&tokens.index((.., start..end)))?;
                start = end;
            }
            let tail = tokens.index((.., start..prompt_len));
            let mask = input
                .text
                .mask
                .as_ref()
                .map(|m| m.index((.., start..prompt_len)));
            input = LMInput {
                text: Text { tokens: tail, mask },
                #[cfg(feature = "image")]
                image: None,
                #[cfg(feature = "audio")]
                audio: None,
            };
        }
    }

    match model.prepare(input)? {
        PrepareResult::Logits(arr) => Ok(arr),
        PrepareResult::Primed => {
            let seed = Array::from_slice::<i32>(&[0], &[1]);
            Ok(model.step(&seed)?.logits)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm_input::Text;

    /// Decodes tokens via a fixed lookup table (per-token byte slices,
    /// concatenated) so the test exercises O(N) streaming with no model.
    struct FakeProcessor {
        pieces: Vec<&'static str>,
    }

    impl UserInputProcessor for FakeProcessor {
        fn family(&self) -> &'static str {
            "fake"
        }
        fn prepare(&mut self, _input: UserInput) -> Result<LMInput, Error> {
            Ok(LMInput {
                text: Text {
                    tokens: Array::from_slice::<i32>(&[], &[1, 0]),
                    mask: None,
                },
                #[cfg(feature = "image")]
                image: None,
                #[cfg(feature = "audio")]
                audio: None,
            })
        }
        fn decode(&self, ids: &[u32]) -> Result<String, Error> {
            let mut out = String::new();
            for &id in ids {
                if let Some(p) = self.pieces.get(id as usize) {
                    out.push_str(p);
                }
            }
            Ok(out)
        }
    }

    fn assert_incremental_matches_naive(pieces: &[&'static str], ids: &[u32]) {
        let processor = FakeProcessor {
            pieces: pieces.to_vec(),
        };
        let naive_full = processor.decode(ids).unwrap();
        let mut dec = IncrementalDecoder::with_capacity(ids.len());
        let mut streamed = String::new();
        for &id in ids {
            streamed.push_str(&dec.push(id, &processor).unwrap());
        }
        let final_text = dec.into_text();
        assert_eq!(streamed, final_text);
        assert_eq!(naive_full, final_text);
    }

    #[test]
    fn incremental_matches_naive_ascii() {
        let pieces = vec!["hello", " ", "world", ".", " ", "foo", " ", "bar"];
        let ids: Vec<u32> = (0..pieces.len() as u32).collect();
        assert_incremental_matches_naive(&pieces, &ids);
    }

    #[test]
    fn incremental_matches_naive_multibyte() {
        let pieces = vec!["你", "好", "世", "界", "🎉", " ", "🍕", "!"];
        let ids: Vec<u32> = (0..pieces.len() as u32).collect();
        assert_incremental_matches_naive(&pieces, &ids);
    }

    #[test]
    fn incremental_handles_empty_run() {
        let dec = IncrementalDecoder::with_capacity(0);
        assert_eq!(dec.into_text(), "");
    }

    #[test]
    fn incremental_window_advances_across_long_response() {
        let pieces: Vec<&'static str> = vec!["a"; 100];
        let processor = FakeProcessor { pieces };
        let mut dec = IncrementalDecoder::with_capacity(100);
        for id in 0..100_u32 {
            dec.push(id, &processor).unwrap();
        }
        assert!(dec.committed_tokens >= 100 - IncrementalDecoder::WINDOW - 1);
    }
}
