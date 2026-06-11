//! Log-mel front-end regression test (no model, CI-safe).
//!
//! Decodes a committed 16 kHz mono WAV through `log_mel` and asserts the result
//! matches a committed golden `.f32` (this crate's own output, validated once
//! against the reference at <0.003 max-abs). Locks the front-end against silent
//! numerical drift — a real regression here flips the golden.
//!
//! Regenerate the golden after an intentional front-end change:
//! `GEN_AUDIO_GOLDEN=1 cargo test -p mlx-tests --test gemma4_audio_logmel \
//!   --features audio -- --ignored regenerate_golden`

#![cfg(feature = "audio")]

use std::path::PathBuf;

use mlx_lm::gemma4::audio::feature::log_mel;
use mlx_rs::transforms::eval;
use mlx_rs::{ops::reshape, Array};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/audio")
        .join(name)
}

/// Decode the committed 16 kHz mono WAV to `f32` samples in `[-1, 1]`.
fn load_wav(path: &PathBuf) -> Vec<f32> {
    let reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000);
    assert_eq!(spec.channels, 1);
    let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
    let mut reader = reader;
    reader
        .samples::<i32>()
        .map(|s| s.expect("wav sample") as f32 / max)
        .collect()
}

/// Flatten an `Array` to a host `Vec<f32>`.
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

#[test]
fn log_mel_matches_golden() {
    let wav = load_wav(&fixture("sine_440hz_16k.wav"));
    let mel = to_vec(&log_mel(&wav).unwrap());
    let golden = read_golden(&fixture("sine_440hz_16k.logmel.f32"));
    assert_eq!(mel.len(), golden.len(), "frame count drifted from golden");
    let max_abs = mel
        .iter()
        .zip(&golden)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_abs < 1e-4,
        "log-mel drifted from golden: max abs {max_abs}"
    );
}

#[test]
#[ignore = "regenerates the committed golden; run with GEN_AUDIO_GOLDEN=1 after an intentional change"]
fn regenerate_golden() {
    assert!(
        std::env::var("GEN_AUDIO_GOLDEN").is_ok(),
        "set GEN_AUDIO_GOLDEN=1 to overwrite the committed golden"
    );
    let wav = load_wav(&fixture("sine_440hz_16k.wav"));
    let mel = to_vec(&log_mel(&wav).unwrap());
    let bytes: Vec<u8> = mel.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(fixture("sine_440hz_16k.logmel.f32"), bytes).expect("write golden");
    eprintln!("wrote golden: {} f32 values", mel.len());
}
