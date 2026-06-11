//! MoE Gemma 4 (26b-a4b: dual-branch dense MLP + 128-expert top-8)
//! end-to-end load + greedy decode. Ignored by default — needs a local
//! checkpoint. Run with:
//! `MODEL=<dir> cargo test -p mlx-tests --test gemma4_moe_e2e -- --ignored --nocapture`
//!
//! Target: `gemma-4-26b-a4b-it-8bit`. Smoke test only — asserts the full
//! load → prefill → decode pipeline runs and produces coherent text.
//!
//! Uses a chat-formatted prompt: this IT checkpoint is instruction-tuned
//! and emits incoherent text on a bare completion prompt, so the test
//! exercises the real usage path (chat template → `<start_of_turn>` wrap).

use std::ops::ControlFlow;

use mlx_lm::chat_template::ChatMessage;
use mlx_lm::{generate, load, GenerateParams, Sampler, UserInput};

#[test]
#[ignore = "requires a local MoE Gemma 4 checkpoint via MODEL env var"]
fn moe_generates_coherent_text() {
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
        UserInput::chat(vec![ChatMessage::user("What is the capital of France?")]),
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
        "expected 'Paris' in MoE gemma4 output, got: {}",
        res.text
    );
}
