//! Dense Gemma 4 (sliding/global hybrid) end-to-end load + greedy decode.
//! Ignored by default — needs a local checkpoint. Run with:
//! `MODEL=<dir> cargo test -p mlx-tests --test gemma4_dense_e2e -- --ignored --nocapture`
//!
//! Target: `gemma-4-31b-it-4bit` (the pure dense path). Smoke test only —
//! asserts the full load → prefill → decode pipeline runs and produces
//! coherent, non-empty text, not a correctness benchmark.

use std::ops::ControlFlow;

use mlx_lm::{generate, load, GenerateParams, Sampler, UserInput};

#[test]
#[ignore = "requires a local dense Gemma 4 checkpoint via MODEL env var"]
fn dense_generates_coherent_text() {
    let dir = std::env::var("MODEL").expect("set MODEL=<checkpoint dir>");
    let mut ctx = load(&dir).expect("load");
    let params = GenerateParams {
        max_new_tokens: 40,
        sampling: Sampler::Greedy,
        ..Default::default()
    };
    let mut text = String::new();
    let res = generate(
        &mut ctx,
        UserInput::text("The capital of France is"),
        params,
        &mut |_id, delta| {
            text.push_str(delta);
            ControlFlow::Continue(())
        },
    )
    .expect("generate");
    println!("OUT: {}", res.text);
    assert!(
        res.text.contains("Paris"),
        "expected 'Paris' in dense gemma4 output, got: {}",
        res.text
    );
}
