//! Qwen3.5-MoE end-to-end: load a real MoE checkpoint, generate a short
//! completion, and verify MTP self-speculative decode is greedy-exact
//! vs the plain decode path. Ignored by default — needs a checkpoint via
//! `MODEL` + a Metal device. Run:
//! `MODEL=<dir> cargo test -p mlx-tests --test qwen3_5_moe_e2e -- --ignored --nocapture`.

use std::ops::ControlFlow;

use mlx_lm::{generate, load, GenerateParams, Sampler, UserInput};

const PROMPT: &str = "The capital of France is";

fn greedy_token_ids(disable_mtp: bool, max_new_tokens: i32) -> (Vec<u32>, String) {
    let dir = std::env::var("MODEL").expect("set MODEL=<checkpoint dir>");
    let mut ctx = load(&dir).expect("load");
    let params = GenerateParams {
        max_new_tokens,
        sampling: Sampler::Greedy,
        disable_mtp,
        ..Default::default()
    };
    let mut ids = Vec::new();
    let mut text = String::new();
    generate(
        &mut ctx,
        UserInput::text(PROMPT),
        params,
        &mut |id, delta| {
            ids.push(id);
            text.push_str(delta);
            ControlFlow::Continue(())
        },
    )
    .expect("generate");
    (ids, text)
}

#[test]
#[ignore = "requires a local Qwen3.5-MoE checkpoint via MODEL env var"]
fn moe_generates_coherent_text() {
    let (_ids, text) = greedy_token_ids(false, 40);
    println!("OUT: {text}");
    assert!(!text.trim().is_empty(), "generated text is empty");
}

#[test]
#[ignore = "requires a local Qwen3.5-MoE checkpoint via MODEL env var"]
fn mtp_greedy_matches_plain_decode() {
    let (with_mtp, with_text) = greedy_token_ids(false, 32);
    let (no_mtp, no_text) = greedy_token_ids(true, 32);
    println!("WITH_MTP: {with_text}");
    println!("NO_MTP:   {no_text}");
    assert_eq!(
        with_mtp, no_mtp,
        "MTP greedy decode must match plain greedy decode token-for-token"
    );
}
