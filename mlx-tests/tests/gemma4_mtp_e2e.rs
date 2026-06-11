//! Gemma 4 MTP end-to-end: load a real target + assistant (drafter)
//! checkpoint, generate a short completion, and verify speculative decode is
//! greedy-exact vs the plain decode path (the spec-decode guarantee). Ignored
//! by default — needs both checkpoints + a Metal device. Run:
//! `MODEL=<target> DRAFT_MODEL=<assistant> \
//!   cargo test -p mlx-tests --test gemma4_mtp_e2e -- --ignored --nocapture`.

use std::ops::ControlFlow;
use std::path::PathBuf;

use mlx_lm::{generate, load_with_drafter, GenerateParams, Sampler, UserInput};

const PROMPT: &str = "The capital of France is";

fn target_dir() -> String {
    std::env::var("MODEL").expect("set MODEL=<target checkpoint dir>")
}

fn draft_dir() -> PathBuf {
    PathBuf::from(std::env::var("DRAFT_MODEL").expect("set DRAFT_MODEL=<assistant checkpoint dir>"))
}

/// Both arms load WITH the drafter (same text adapter, same prompt handling)
/// and differ only in `disable_mtp`, isolating the speculative path. Loading
/// plain `load()` instead would route an e2b/e4b VLM checkpoint to a different
/// processor and confound the comparison.
fn greedy_ids(disable_mtp: bool, max_new_tokens: i32) -> (Vec<u32>, String) {
    let mut ctx = load_with_drafter(target_dir(), Some(&draft_dir())).expect("load with drafter");
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
#[ignore = "requires MODEL + DRAFT_MODEL gemma4 checkpoints"]
fn drafter_generates_coherent_text() {
    let (_ids, text) = greedy_ids(false, 40);
    println!("OUT: {text}");
    assert!(!text.trim().is_empty(), "generated text is empty");
}

#[test]
#[ignore = "requires MODEL + DRAFT_MODEL gemma4 checkpoints"]
fn mtp_greedy_matches_plain_decode() {
    let (with_mtp, with_text) = greedy_ids(false, 32);
    let (no_mtp, no_text) = greedy_ids(true, 32);
    println!("WITH_DRAFTER: {with_text}");
    println!("NO_DRAFTER:   {no_text}");
    assert_eq!(
        with_mtp, no_mtp,
        "drafter greedy decode must match plain greedy decode token-for-token"
    );
}
