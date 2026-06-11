//! Log-mel front-end for the gemma4 audio tower.
//!
//! 16 kHz mono → `[1, T, 128]` log-mel. Frame 320 / hop 160, periodic Hann,
//! semicausal left-pad (frame/2), FFT 512, magnitude spectrum, HTK mel
//! filterbank (128 bins, 0–8000 Hz, no norm), `log(mel + 1e-3)`. The encoder
//! subsamples T by 4, so the soft-token count is derived from the frame count
//! so it equals the encoder output rows.

use std::sync::OnceLock;

use mlx_rs::{
    fft::rfft,
    ops::{abs, expand_dims, matmul, reshape},
    Array,
};

use crate::error::Error;
use crate::gemma4::audio::config::AUDIO_SAMPLE_RATE;

const FRAME_LENGTH: usize = 320; // 20 ms @ 16 kHz
const HOP_LENGTH: usize = 160; // 10 ms
const FFT_LENGTH: usize = 512; // next pow2 of frame
const N_MELS: usize = 128;
const MEL_FLOOR: f32 = 1e-3;
const MIN_FREQ: f32 = 0.0;
const MAX_FREQ: f32 = 8000.0;
/// Clips longer than this are truncated before framing.
const MAX_SAMPLES: usize = 480_000; // 30 s @ 16 kHz

/// Number of audio soft tokens the encoder emits for `num_samples` 16 kHz
/// samples: `mel_frames` passed through the two stride-2 conv subsamples. Sizes
/// the `<audio>` placeholder so the scatter count equals the encoder rows.
pub fn num_audio_tokens(num_samples: usize) -> i32 {
    let mel_frames = num_mel_frames(num_samples);
    // Two stride-2 convs with pad 1: t → (t + 2 - 3)/2 + 1 = (t-1)/2 + 1, twice.
    let mut t = mel_frames as i32;
    for _ in 0..2 {
        t = (t + 2 - 3) / 2 + 1;
    }
    t
}

/// Mel frame count: cap at `MAX_SAMPLES`, semicausal left-pad (frame/2), then
/// unfold by hop with a `frame_length + 1` window (the trailing sample is
/// dropped, leaving `frame_length` per frame). All emitted frames are valid —
/// no trailing padding frames, so the encoder needs no padding mask (B=1).
fn num_mel_frames(num_samples: usize) -> usize {
    let padded = num_samples.min(MAX_SAMPLES) + FRAME_LENGTH / 2;
    let unfold = FRAME_LENGTH + 1;
    if padded < unfold {
        return 0;
    }
    (padded - unfold) / HOP_LENGTH + 1
}

/// Periodic Hann window `[FRAME_LENGTH]`, built once.
fn hann() -> &'static Vec<f32> {
    static W: OnceLock<Vec<f32>> = OnceLock::new();
    W.get_or_init(|| {
        (0..FRAME_LENGTH)
            .map(|n| {
                let t = std::f32::consts::TAU * n as f32 / FRAME_LENGTH as f32;
                0.5 - 0.5 * t.cos()
            })
            .collect()
    })
}

/// HTK mel filterbank host data `[FFT_LENGTH/2 + 1, N_MELS]` row-major (no
/// normalization), cached. Stored as a `Vec<f32>` (not `Array`) because `Array`
/// is not `Sync`; the device tensor is rebuilt per call (cheap, ~37 k f32).
fn mel_filters() -> &'static Vec<f32> {
    static M: OnceLock<Vec<f32>> = OnceLock::new();
    M.get_or_init(build_mel_filters)
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

fn build_mel_filters() -> Vec<f32> {
    let n_freqs = FFT_LENGTH / 2 + 1;
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * AUDIO_SAMPLE_RATE as f32 / FFT_LENGTH as f32)
        .collect();
    let mel_min = hz_to_mel(MIN_FREQ);
    let mel_max = hz_to_mel(MAX_FREQ);
    // N_MELS + 2 mel points → N_MELS triangular filters.
    let mel_points: Vec<f32> = (0..N_MELS + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let mut fb = vec![0f32; n_freqs * N_MELS];
    for m in 0..N_MELS {
        let (left, center, right) = (hz_points[m], hz_points[m + 1], hz_points[m + 2]);
        for (f, &freq) in fft_freqs.iter().enumerate() {
            let w = if freq >= left && freq <= center {
                (freq - left) / (center - left)
            } else if freq > center && freq <= right {
                (right - freq) / (right - center)
            } else {
                0.0
            };
            fb[f * N_MELS + m] = w.max(0.0);
        }
    }
    fb
}

/// `waveform` (16 kHz mono `f32`) → `[1, T, 128]` log-mel.
pub fn log_mel(waveform: &[f32]) -> Result<Array, Error> {
    let frames = num_mel_frames(waveform.len());
    if frames == 0 {
        return Err(Error::shape(
            "gemma4 audio: clip too short for one mel frame",
        ));
    }
    // Effective signal: cap + right-pad to multiple (zeros), then semicausal
    // left-pad (frame/2). Out-of-range reads return 0, covering both pads.
    let cap = waveform.len().min(MAX_SAMPLES);
    let pad = FRAME_LENGTH / 2;
    let win = hann();
    let mut framed = vec![0f32; frames * FFT_LENGTH];
    for fi in 0..frames {
        let start = fi * HOP_LENGTH; // index into the left-padded signal
        for j in 0..FRAME_LENGTH {
            let idx = start + j;
            // padded[idx] = 0 for idx < pad or past the capped signal.
            let sample = if idx < pad || idx - pad >= cap {
                0.0
            } else {
                waveform[idx - pad]
            };
            framed[fi * FFT_LENGTH + j] = sample * win[j];
        }
        // remaining [FRAME_LENGTH..FFT_LENGTH] stay zero-padded for the FFT.
    }
    let framed = Array::from_slice(&framed, &[frames as i32, FFT_LENGTH as i32]);
    let spec = rfft(&framed, FFT_LENGTH as i32, -1)?; // [frames, n_freqs] complex
    let mag = abs(&spec)?; // magnitude
    let n_freqs = (FFT_LENGTH / 2 + 1) as i32;
    let filters = Array::from_slice(mel_filters(), &[n_freqs, N_MELS as i32]);
    let mel = matmul(&mag, &filters)?; // [frames, N_MELS]
    let logmel = mel.add(Array::from_f32(MEL_FLOOR))?.log()?;
    Ok(expand_dims(
        &reshape(&logmel, &[frames as i32, N_MELS as i32])?,
        0,
    )?)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;

    #[test]
    fn log_mel_shape() {
        // 1 second of silence → [1, T, 128].
        let wav = vec![0f32; AUDIO_SAMPLE_RATE as usize];
        let mel = log_mel(&wav).unwrap();
        let s = mel.shape();
        assert_eq!(s[0], 1);
        assert_eq!(s[2], N_MELS as i32);
        assert_eq!(s[1], num_mel_frames(wav.len()) as i32);
    }

    #[test]
    fn token_count_matches_subsample() {
        // 1 s → ~100 mel frames → /4 ≈ 25 tokens.
        let n = num_audio_tokens(AUDIO_SAMPLE_RATE as usize);
        assert!((20..=30).contains(&n), "got {n}");
    }
}
