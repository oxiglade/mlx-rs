//! E2B / E4B Gemma 4 (per-layer-input embeddings + KV-sharing) end-to-end
//! load + greedy decode. Ignored by default — needs a local checkpoint.
//! Run for either variant via the `MODEL` env var:
//!   `MODEL=<gemma-4-e4b-it-8bit dir> cargo test -p mlx-tests \
//!     --test gemma4_e2b_e4b_e2e -- --ignored --nocapture`
//!
//! Chat-formatted prompt (the IT checkpoints emit incoherent text on a bare
//! completion prompt). Smoke test: full load → prefill → decode → "Paris".

use std::ops::ControlFlow;

use mlx_lm::chat_template::ChatMessage;
use mlx_lm::{generate, load, GenerateParams, Sampler, UserInput};

#[test]
#[ignore = "requires a local E2B/E4B Gemma 4 checkpoint via MODEL env var"]
fn per_layer_input_generates_coherent_text() {
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
        "expected 'Paris' in E2B/E4B gemma4 output, got: {}",
        res.text
    );
}
