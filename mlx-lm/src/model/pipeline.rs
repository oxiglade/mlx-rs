use super::*;
use crate::sampling::SamplePlan;

impl Generation<'_> {
    pub(super) fn first_pipelined(
        &mut self,
        logits: Array,
    ) -> Result<GenerationEvent, GenerationError> {
        let stream = self.pipeline_stream.as_ref().unwrap().clone();
        mlx_rs::with_stream(&stream, || {
            let plan = self.sampling.prepare(
                &self.history,
                None,
                logits,
                self.rng.as_ref(),
                self.capture_logprobs,
            )?;
            plan.submit()?;
            self.finish_pipelined(plan, None)
        })
    }

    pub(super) fn next_pipelined(
        &mut self,
        previous: TokenId,
    ) -> Result<GenerationEvent, GenerationError> {
        let stream = self.pipeline_stream.as_ref().unwrap().clone();
        mlx_rs::with_stream(&stream, || {
            let plan = self
                .pending
                .take()
                .expect("active pipeline has a submitted step")?;
            self.finish_pipelined(plan, Some(previous))
        })
    }

    fn build_ahead(
        &mut self,
        sample: &PendingSample,
        offset: usize,
    ) -> Result<SamplePlan, GenerationError> {
        let tokens = sample.token.reshape(&[1, 1])?;
        let mut step = self.cache.get_mut().speculative_step(offset)?;
        let logits = self.decoder.forward(&tokens, &mut step)?;
        let row = last_row(&logits)?;
        let plan = self.sampling.prepare(
            &self.history,
            Some(&sample.token),
            row,
            sample.rng.as_ref(),
            false,
        )?;
        step.submit(&plan.outputs(&[]))?;
        Ok(plan)
    }

    fn finish_pipelined(
        &mut self,
        plan: SamplePlan,
        input: Option<TokenId>,
    ) -> Result<GenerationEvent, GenerationError> {
        if self.accepted + 1 < self.max_tokens.get() {
            if let Some(sample) = plan.candidate() {
                let offset = self.cache.get().tokens()?.len() + usize::from(input.is_some());
                self.pending = Some(self.build_ahead(sample, offset));
            }
        }
        let sample = plan.finish().inspect_err(|error| {
            if matches!(error, GenerationError::Exception(_))
                && (input.is_some() || matches!(self.pending, Some(Ok(_))))
            {
                self.cache.get_mut().invalidate();
            }
        })?;
        let token = sample_token(&sample)?;
        let stop = self.stop_tokens.contains(&token);
        let text = self
            .text
            .take()
            .expect("active generation has a text decoder");
        let (next_text, text, reason) =
            text.prepare(token, stop, self.accepted + 1 == self.max_tokens.get())?;
        if let Some(input) = input {
            if self.cache.get().ensure_valid().is_ok() {
                self.cache.get_mut().commit_input(input)?;
            }
        }
        if reason.is_some() {
            self.discard_pending();
        }
        self.publish(sample, token, next_text, text, reason)
    }

    pub(super) fn discard_pending(&mut self) {
        self.pending = None;
    }
}
