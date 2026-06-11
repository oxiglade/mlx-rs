//! Audio encoder math regression test (CI-safe, no checkpoint).
//!
//! Seeds the RNG, builds a small `AudioEncoder` with random weights, runs the
//! committed sine clip's log-mel through it, and asserts the output matches a
//! committed golden `.f32`. Locks the encoder MATH (subsample conv, chunked
//! attention + relshift, depthwise conv, macaron FFN, norms) against silent
//! drift. It does NOT exercise weight loading — the real-weight load path
//! (clip buffers, key binding) is covered by the model-gated unit test in
//! `gemma4/vlm/weights.rs`.
//!
//! Regenerate after an intentional encoder change:
//! `GEN_AUDIO_GOLDEN=1 cargo test -p mlx-tests --test gemma4_audio_encoder \
//!   --features audio -- --ignored regenerate_encoder_golden`

#![cfg(feature = "audio")]

use std::path::PathBuf;

use mlx_lm::gemma4::audio::config::AudioConfig;
use mlx_lm::gemma4::audio::encoder::AudioEncoder;
use mlx_lm::gemma4::audio::feature::log_mel;
use mlx_rs::transforms::eval;
use mlx_rs::{ops::reshape, random, Array};

/// Fixed RNG seed so the random-init weights are reproducible across runs.
const SEED: u64 = 0x6175_6469_6f5f_3031; // "audio_01"

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/audio")
        .join(name)
}

/// Small but structurally faithful config: 2 layers, chunk 12 / past 12 so the
/// chunked-attention + relshift paths run, mel input dim fixed at 128.
fn synthetic_config() -> AudioConfig {
    AudioConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        conv_kernel_size: 5,
        output_proj_dims: Some(48),
        rms_norm_eps: 1e-6,
        residual_weight: 0.5,
        gradient_clipping: 1e10,
        attention_chunk_size: 12,
        attention_context_left: 13,
        attention_context_right: 0,
        attention_logit_cap: 50.0,
        attention_invalid_logits_value: -1e9,
    }
}

fn load_wav(path: &PathBuf) -> Vec<f32> {
    let reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
    let mut reader = reader;
    reader
        .samples::<i32>()
        .map(|s| s.expect("wav sample") as f32 / max)
        .collect()
}

fn to_vec(a: &Array) -> Vec<f32> {
    let total: i32 = a.shape().iter().product();
    let flat = reshape(a, &[total]).unwrap();
    eval([&flat]).unwrap();
    flat.as_slice::<f32>().to_vec()
}

fn read_golden(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read golden");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Build a deterministic encoder + its output on the sine clip's log-mel.
fn encoder_output() -> Vec<f32> {
    random::seed(SEED).unwrap();
    let mut enc = AudioEncoder::new(&synthetic_config()).unwrap();
    let mel = log_mel(&load_wav(&fixture("sine_440hz_16k.wav"))).unwrap();
    to_vec(&enc.forward(&mel).unwrap())
}

#[test]
fn encoder_matches_golden() {
    let out = encoder_output();
    let golden = read_golden(&fixture("encoder_synth.f32"));
    assert_eq!(out.len(), golden.len(), "encoder output shape drifted");
    let max_abs = out
        .iter()
        .zip(&golden)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_abs < 1e-4,
        "encoder output drifted from golden: max abs {max_abs}"
    );
}

#[test]
#[ignore = "regenerates the committed encoder golden; run with GEN_AUDIO_GOLDEN=1 after an intentional change"]
fn regenerate_encoder_golden() {
    assert!(
        std::env::var("GEN_AUDIO_GOLDEN").is_ok(),
        "set GEN_AUDIO_GOLDEN=1 to overwrite the committed golden"
    );
    let out = encoder_output();
    let bytes: Vec<u8> = out.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(fixture("encoder_synth.f32"), bytes).expect("write golden");
    eprintln!("wrote encoder golden: {} f32 values", out.len());
}
