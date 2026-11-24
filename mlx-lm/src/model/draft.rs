use super::*;
use crate::arch::DraftContext;
use mlx_rs::ops::{self, indexing::IndexOp};
use std::collections::VecDeque;

/// Drafted decoding of one request: the checkpoint's drafter proposes the next tokens and one
/// forward pass over them tells which the model would have chosen itself.
pub(super) struct Drafting {
    /// Rows of a verification pass, the anchor included.
    block: usize,
    context: DraftContext,
    /// Verified tokens not yet published. The row of each joins the cache when the token
    /// after it is published, so the cache never holds a token ahead of its event.
    queue: VecDeque<(TokenId, PendingSample)>,
}

impl Drafting {
    pub(super) fn new(block: usize) -> Self {
        Self {
            block,
            context: DraftContext::default(),
            queue: VecDeque::new(),
        }
    }

    /// Adds prefilled rows to the drafter's context.
    pub(super) fn observe(
        &mut self,
        decoder: &dyn DecoderModel,
        taps: &Array,
    ) -> Result<(), GenerationError> {
        decoder.observe(&mut self.context, taps)?;
        // Or the context keeps every prefill chunk's taps alive until the first round.
        mlx_rs::transforms::eval(self.context.layers.iter().flat_map(|(k, v)| [k, v]))
            .map_err(crate::InferenceError::from)?;
        Ok(())
    }
}

impl Generation<'_> {
    pub(super) fn next_drafted(
        &mut self,
        previous: TokenId,
    ) -> Result<GenerationEvent, GenerationError> {
        let stream = self.pipeline_stream.as_ref().unwrap().clone();
        mlx_rs::with_stream(&stream, || {
            if self.drafting.as_ref().unwrap().queue.is_empty() {
                self.round(previous)?;
            }
            let queue = &mut self.drafting.as_mut().unwrap().queue;
            let (token, sample) = queue.pop_front().expect("a round verifies a token");
            let stop = self.stop_tokens.contains(&token);
            let text = self
                .text
                .take()
                .expect("active generation has a text decoder");
            let (next_text, text, reason) =
                text.prepare(token, stop, self.accepted + 1 == self.max_tokens.get())?;
            self.cache.get_mut().commit_input(previous)?;
            self.publish(sample, token, next_text, text, reason)
        })
    }

    /// Verifies the drafter's proposals after `anchor` in one pass and queues those the model
    /// agrees with, then the model's own token at the first disagreement.
    fn round(&mut self, anchor: TokenId) -> Result<(), GenerationError> {
        let drafting = self.drafting.as_mut().unwrap();
        let remaining = self.max_tokens.get() - self.accepted;
        let count = (drafting.block - 1).min(remaining);
        let proposed = self.decoder.propose(&drafting.context, anchor, count)?;
        let tokens = ops::concatenate(&[&token_array(&[anchor])?, &proposed], 1)
            .map_err(crate::InferenceError::from)?;
        let cache = self.cache.get_mut();
        let offset = cache.tokens()?.len();
        let mut step = cache.speculative_step(offset)?;
        let (logits, taps) = self.decoder.forward_taps(&tokens, None, &mut step)?;
        // Without penalties a plan is the row's greedy token and its check for invalid logits.
        let plans = (0..=count as i32)
            .map(|row| {
                let row = logits.index((.., row, ..));
                self.sampling.prepare(&self.history, None, row, None, false)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut outputs: Vec<_> = plans.iter().flat_map(|plan| plan.outputs(&[])).collect();
        outputs.push(&taps);
        step.submit(&outputs)?;
        let mut failed = |error: &GenerationError| {
            if matches!(error, GenerationError::Exception(_)) {
                cache.invalidate();
            }
        };
        proposed
            .eval()
            .map_err(GenerationError::Exception)
            .inspect_err(&mut failed)?;
        let proposed = proposed
            .to_vec_exact::<u32>()
            .map_err(crate::SamplingError::Conversion)?;
        // Rows after the first disagreement are never read, so only the rows kept are checked.
        for (row, plan) in plans.into_iter().enumerate() {
            let sample = plan.finish().inspect_err(&mut failed)?;
            let token = sample_token(&sample)?;
            drafting.queue.push_back((token, sample));
            if proposed.get(row).copied().map(TokenId::from) != Some(token) {
                break;
            }
        }
        drafting.queue.truncate(remaining);
        // The rows this round commits if it is published to its end.
        let kept = drafting.queue.len() as i32;
        self.decoder
            .observe(&mut drafting.context, &taps.index((.., ..kept, ..)))?;
        Ok(())
    }
}
