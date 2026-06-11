//! Gemma 4 Unified encoder-free audio feature extraction.
//!
//! Pad the 16 kHz mono waveform to a multiple of `audio_samples_per_token`
//! (640), then reshape into `[num_tokens, 640]` raw frames — no mel, no FFT.
//! 640 samples = 40 ms ⇒ 25 tokens/sec.

/// Pad + reshape raw samples into `(frames, samples_per_token)`. Returns the
/// flattened row-major buffer and the token count.
pub fn frame_waveform(samples: &[f32], samples_per_token: i32) -> (Vec<f32>, i32) {
    let spt = samples_per_token as usize;
    let pad = (spt - (samples.len() % spt)) % spt;
    let num_tokens = (samples.len() + pad) / spt;
    let mut frames = Vec::with_capacity(num_tokens * spt);
    frames.extend_from_slice(samples);
    frames.resize(num_tokens * spt, 0.0);
    (frames, num_tokens as i32)
}

/// Soft tokens a clip of `len` samples expands to.
pub fn num_audio_tokens(len: usize, samples_per_token: i32) -> i32 {
    let spt = samples_per_token as usize;
    len.div_ceil(spt) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_pad_to_token_multiple() {
        // 700 samples, 640/token → 2 tokens (700 padded to 1280).
        let samples = vec![1.0f32; 700];
        let (frames, n) = frame_waveform(&samples, 640);
        assert_eq!(n, 2);
        assert_eq!(frames.len(), 2 * 640);
        // Real samples preserved, tail zero-padded.
        assert_eq!(frames[699], 1.0);
        assert_eq!(frames[700], 0.0);
        assert_eq!(num_audio_tokens(700, 640), 2);
    }

    #[test]
    fn exact_multiple_no_pad() {
        let samples = vec![0.5f32; 1280];
        let (frames, n) = frame_waveform(&samples, 640);
        assert_eq!(n, 2);
        assert_eq!(frames.len(), 1280);
        assert_eq!(num_audio_tokens(1280, 640), 2);
    }
}
