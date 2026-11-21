//! Replay tokenized prompts through the public greedy generation path.
use anyhow::{ensure, Result};
use mlx_lm::{GenerationEvent, GenerationOptions, Model, Prompt, TokenId};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() >= 4,
        "usage: qwen_hybrid CHECKPOINT PROMPT_IDS_JSON TOKEN_COUNT [MORE_PROMPT_IDS_JSON ...]"
    );
    let mut model = Model::from_dir(&args[1])?;
    let mut options = GenerationOptions::default();
    options.max_tokens = args[3].parse()?;
    for prompt in [&args[2]].into_iter().chain(&args[4..]) {
        let ids: Vec<u32> = serde_json::from_str(prompt)?;
        let ids: Vec<TokenId> = ids.into_iter().map(Into::into).collect();
        let start = Instant::now();
        let mut first = None;
        let mut emitted = 0;
        for event in model.generate(Prompt::Tokens(&ids), options.clone())? {
            if let GenerationEvent::Token {
                token_id,
                finish_reason,
                ..
            } = event?
            {
                let now = Instant::now();
                let first = *first.get_or_insert_with(|| {
                    eprintln!("first token {:.3}s", start.elapsed().as_secs_f64());
                    now
                });
                emitted += 1;
                println!("{}", u32::from(token_id));
                if let Some(reason) = finish_reason {
                    eprintln!(
                        "emitted {emitted}, finish {reason:?}, decode {:.2} tokens/s",
                        (emitted - 1) as f64 / now.duration_since(first).as_secs_f64()
                    );
                }
            }
        }
    }
    Ok(())
}
