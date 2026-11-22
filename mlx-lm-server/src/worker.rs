use std::{error::Error, path::Path, sync::mpsc::SyncSender};

use mlx_lm::{
    ChatContinuation, ChatTemplateOptions, FinishReason, GenerationError, GenerationEvent, Model,
    Prompt, ResponsePart,
};
use tokio::sync::mpsc::{Receiver, Sender};

use crate::api::ChatRequest;

pub(crate) struct Job {
    pub request: ChatRequest,
    pub reply: Sender<Result<Update, Failure>>,
}
pub(crate) enum Update {
    Prefill {
        processed: usize,
        total: usize,
    },
    Parts(Vec<ResponsePart>),
    Done {
        finish: FinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
}
pub(crate) enum Failure {
    Request(String),
    Generation(String),
}

fn chain(error: &dyn Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(error) = source {
        message += &format!(": {error}");
        source = error.source();
    }
    message
}

/// Loads the model and serves jobs one at a time; `Model` cannot leave its thread.
pub(crate) fn run(
    directory: &Path,
    mut jobs: Receiver<Job>,
    ready: SyncSender<Result<(), String>>,
) {
    let mut model = match Model::from_dir(directory) {
        Ok(model) => model,
        Err(error) => {
            let _ = ready.send(Err(chain(&error)));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        return;
    }
    while let Some(Job { request, reply }) = jobs.blocking_recv() {
        if let Err(failure) = serve(&mut model, &request, &reply) {
            let _ = reply.blocking_send(Err(failure));
        }
    }
}

/// Stops early once the reply's receiver is gone, which is a client that stopped waiting.
fn serve(
    model: &mut Model,
    request: &ChatRequest,
    reply: &Sender<Result<Update, Failure>>,
) -> Result<(), Failure> {
    let invalid = |error: &dyn Error| Failure::Request(chain(error));
    let tools = request.tools.clone().unwrap_or_default();
    let tokenizer = model.tokenizer();
    let template = ChatTemplateOptions {
        continuation: ChatContinuation::StartAssistant,
        tools: tools.clone(),
        ..Default::default()
    };
    let prompt = tokenizer
        .render_chat(&request.messages, template)
        .map_err(|error| invalid(&error))?;
    let tokens = tokenizer.encode(&prompt).map_err(|error| invalid(&error))?;
    let mut parser = tokenizer.response_parser(&prompt, &tools);
    let options = request.options();
    options
        .sampling
        .validate(model.config().dimensions.vocabulary_size)
        .map_err(|error| invalid(&GenerationError::from(error)))?;
    let generation = model
        .generate(Prompt::Tokens(&tokens), options)
        .map_err(|error| invalid(&error))?;

    let send = |update| {
        let _ = reply.blocking_send(Ok(update));
    };
    let mut completion_tokens = 0;
    for event in generation {
        if reply.is_closed() {
            break;
        }
        match event.map_err(|error| Failure::Generation(chain(&error)))? {
            GenerationEvent::Prefill {
                processed, total, ..
            } => send(Update::Prefill { processed, total }),
            GenerationEvent::Token {
                text,
                finish_reason,
                ..
            } => {
                completion_tokens += 1;
                let mut parts = parser.push(&text);
                let Some(finish) = finish_reason else {
                    if !parts.is_empty() {
                        send(Update::Parts(parts));
                    }
                    continue;
                };
                parts.extend(parser.finish(finish));
                send(Update::Parts(parts));
                send(Update::Done {
                    finish,
                    prompt_tokens: tokens.len(),
                    completion_tokens,
                });
                break;
            }
            _ => {}
        }
    }
    Ok(())
}
