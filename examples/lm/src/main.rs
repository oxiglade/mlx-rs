use std::io::{self, Write};

use mlx_lm::{
    ChatContinuation, ChatTemplateOptions, GenerationEvent, GenerationOptions, Message, Model,
    Prompt, Role,
};

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: lm <model-directory>"))?;
    let mut model = Model::from_dir(path)?;
    let prompt = model.tokenizer().render_chat(
        &[Message {
            role: Role::User,
            content: "what's your name?".into(),
        }],
        ChatTemplateOptions {
            continuation: ChatContinuation::StartAssistant,
            ..Default::default()
        },
    )?;
    let mut stdout = io::stdout().lock();
    for event in model.generate(Prompt::Text(&prompt), GenerationOptions::default())? {
        if let GenerationEvent::Token { text, .. } = event? {
            stdout.write_all(text.as_bytes())?;
            stdout.flush()?;
        }
    }
    writeln!(stdout)?;
    Ok(())
}
