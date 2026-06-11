//! Gemma 4 audio end-to-end: load a real `gemma-4-e2b/e4b-it` checkpoint,
//! attach a 16 kHz mono WAV, transcribe/answer, confirm it is coherent.
//! Ignored by default — needs a checkpoint via `MODEL`, a WAV via `AUDIO`, the
//! `audio` feature, and a Metal device. Run:
//! `MODEL=<e4b dir> AUDIO=<clip.wav> cargo test -p mlx-tests \
//!   --test gemma4_audio_e2e --features audio -- --ignored --nocapture \
//!   --test-threads=1`.

#![cfg(feature = "audio")]

use std::ops::ControlFlow;

use mlx_lm::{generate, load, Audio, GenerateParams, Sampler, UserInput};

fn load_wav_16k_mono(path: &str) -> Vec<f32> {
    let reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "wav must be 16 kHz");
    assert_eq!(spec.channels, 1, "wav must be mono");
    let mut reader = reader;
    match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / max)
                .collect()
        }
    }
}

#[test]
#[ignore = "requires a local gemma-4 e2b/e4b checkpoint via MODEL + a 16kHz mono WAV via AUDIO"]
fn audio_generates_coherent_text() {
    let dir = std::env::var("MODEL").expect("set MODEL=<gemma4 e2b/e4b checkpoint dir>");
    let wav = std::env::var("AUDIO").expect("set AUDIO=<16kHz mono wav path>");
    let samples = load_wav_16k_mono(&wav);

    let mut ctx = load(&dir).expect("load");
    let params = GenerateParams {
        max_new_tokens: 64,
        sampling: Sampler::Greedy,
        ..Default::default()
    };
    let input = UserInput::text("Transcribe this audio.").with_audio(vec![Audio { samples }]);

    let mut text = String::new();
    generate(&mut ctx, input, params, &mut |_id, delta| {
        text.push_str(delta);
        ControlFlow::Continue(())
    })
    .expect("generate");

    println!("TRANSCRIPT: {text}");
    assert!(!text.trim().is_empty(), "generated text is empty");
    assert!(
        text.chars().any(|c| c.is_alphabetic()),
        "no alphabetic characters: {text:?}"
    );
}
