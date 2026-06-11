//! Qwen3.5-VL end-to-end: load a real VL checkpoint, attach an image,
//! generate a short caption, confirm it's coherent. Ignored by default —
//! needs a VL checkpoint via `MODEL`, an image via `IMAGE`, and a Metal
//! device. Run:
//! `MODEL=<vl_dir> IMAGE=<path> cargo test -p mlx-tests --test qwen3_5_vl_e2e \
//!   --features image -- --ignored --nocapture --test-threads=1`.
//!
//! The existing A3B fixture is text MoE, NOT VL — a dense qwen3.5-VL
//! checkpoint (e.g. an `mlx-community` qwen3.5-VL quant, or chandra-ocr-2)
//! must be downloaded to run this.

#![cfg(feature = "image")]

use std::ops::ControlFlow;

use mlx_lm::{generate, load, GenerateParams, Image, Sampler, UserInput};

#[test]
#[ignore = "requires a local Qwen3.5-VL checkpoint via MODEL + an image via IMAGE"]
fn vl_generates_coherent_caption() {
    let dir = std::env::var("MODEL").expect("set MODEL=<vl checkpoint dir>");
    let img_path = std::env::var("IMAGE").expect("set IMAGE=<image path>");
    let img = image::open(&img_path).expect("open image");

    let mut ctx = load(&dir).expect("load");
    let params = GenerateParams {
        max_new_tokens: 40,
        sampling: Sampler::Greedy,
        ..Default::default()
    };
    let input = UserInput::text("Describe this image.").with_images(vec![Image::Decoded(img)]);

    let mut text = String::new();
    generate(&mut ctx, input, params, &mut |_id, delta| {
        text.push_str(delta);
        ControlFlow::Continue(())
    })
    .expect("generate");

    println!("CAPTION: {text}");
    assert!(!text.trim().is_empty(), "generated caption is empty");
    // No NaN garbling: a coherent caption is printable ASCII-ish text, not
    // a run of replacement characters.
    assert!(
        text.chars().any(|c| c.is_alphabetic()),
        "caption has no alphabetic characters: {text:?}"
    );
}
