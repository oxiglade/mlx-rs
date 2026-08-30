use std::path::Path;

use mlx_lm::{cache::ConcatKeyValueCache, models::qwen3::load_qwen3_model};
use mlx_lm_utils::tokenizer::{
    load_model_chat_template_from_file, ApplyChatTemplateArgs, Conversation, Role, Tokenizer,
};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array,
};

const CACHED_TEST_MODEL_DIR: &str = "./cache/Qwen3-4B-bf16";

fn qwen3() -> anyhow::Result<()> {
    let model_dir = Path::new(CACHED_TEST_MODEL_DIR);

    let model_id = "mlx-community/Qwen3-4B-bf16".to_string();
    let tokenizer_file = model_dir.join("tokenizer.json");
    let tokenizer_config_file = model_dir.join("tokenizer_config.json");
    let mut tokenizer =
        Tokenizer::from_file(tokenizer_file).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let model_chat_template = load_model_chat_template_from_file(tokenizer_config_file)?
        .expect("Model chat template not found");

    let conversations = vec![Conversation {
        role: Role::User,
        content: "what's your name?",
    }];
    let args = ApplyChatTemplateArgs {
        conversations: vec![conversations.into()],
        documents: None,
        model_id: &model_id,
        chat_template_id: None,
        add_generation_prompt: None,
        continue_final_message: None,
    };
    let encodings = tokenizer.apply_chat_template_and_encode(model_chat_template, args)?;
    let prompt: Vec<u32> = encodings
        .iter()
        .flat_map(|encoding| encoding.get_ids())
        .copied()
        .collect();
    let prompt_tokens = Array::from(&prompt[..]).index(NewAxis);

    let mut cache = Vec::new();
    let mut model = load_qwen3_model(model_dir)?;
    let generate = mlx_lm::models::qwen3::Generate::<ConcatKeyValueCache>::new(
        &mut model,
        &mut cache,
        0.2,
        &prompt_tokens,
    );

    let mut tokens = Vec::new();
    for (token, ntoks) in generate.zip(0..256) {
        let token = token.unwrap();
        tokens.push(token.clone());

        if ntoks == 0 {
            eval(&tokens).unwrap();
        }

        if tokens.len() % 20 == 0 {
            eval(&tokens).unwrap();
            let slice: Vec<u32> = tokens.drain(..).map(|t| t.item_exact::<u32>()).collect();
            let s = tokenizer.decode(&slice, true).unwrap();
            print!("{s}");
        }
    }

    eval(&tokens).unwrap();
    let slice: Vec<u32> = tokens.drain(..).map(|t| t.item_exact::<u32>()).collect();
    let s = tokenizer.decode(&slice, true).unwrap();
    println!("{s}");

    println!("------");

    Ok(())
}

fn main() -> anyhow::Result<()> {
    qwen3()
}
