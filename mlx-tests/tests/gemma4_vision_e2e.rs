//! Gemma 4 vision end-to-end: load a real `gemma-4-*-it` VLM checkpoint,
//! attach an image, generate a short caption, confirm it is coherent.
//! Ignored by default — needs a checkpoint via `MODEL`, an image via `IMAGE`,
//! and a Metal device. Run:
//! `MODEL=<gemma4 it dir> IMAGE=<path> cargo test -p mlx-tests \
//!   --test gemma4_vision_e2e --features image -- --ignored --nocapture \
//!   --test-threads=1`.

#![cfg(feature = "image")]

use std::ops::ControlFlow;

use mlx_lm::{generate, load, GenerateParams, Image, Sampler, UserInput};

#[test]
#[ignore = "requires a local gemma-4 it VLM checkpoint via MODEL + an image via IMAGE"]
fn vision_generates_coherent_caption() {
    let dir = std::env::var("MODEL").expect("set MODEL=<gemma4 it checkpoint dir>");
    let img_path = std::env::var("IMAGE").expect("set IMAGE=<image path>");
    let img = image::open(&img_path).expect("open image");

    let mut ctx = load(&dir).expect("load");
    let params = GenerateParams {
        max_new_tokens: 40,
        sampling: Sampler::Greedy,
        ..Default::default()
    };
    let input = UserInput::text("Describe this image in one sentence.")
        .with_images(vec![Image::Decoded(img)]);

    let mut text = String::new();
    generate(&mut ctx, input, params, &mut |_id, delta| {
        text.push_str(delta);
        ControlFlow::Continue(())
    })
    .expect("generate");

    println!("CAPTION: {text}");
    assert!(!text.trim().is_empty(), "generated caption is empty");
    assert!(
        text.chars().any(|c| c.is_alphabetic()),
        "caption has no alphabetic characters: {text:?}"
    );
}
