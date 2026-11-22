use std::{error::Error, path::Path, sync::mpsc::SyncSender};

use mlx_lm::{
    Cache, CacheCheckpoint, CacheError, ChatContinuation, ChatTemplateOptions, FinishReason,
    GenerationError, GenerationEvent, Model, Prompt, ResponsePart, TemplateValue, TokenId,
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
        cached_tokens: usize,
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

/// How many checkpoints of each kind a conversation keeps.
const KEPT: usize = 4;

/// The cache of the latest request and the checkpoints it can rewind to, each a prefix
/// of the cache's tokens, shortest first.
struct Conversation {
    cache: Cache,
    checkpoints: Vec<Checkpoint>,
}
struct Checkpoint {
    state: CacheCheckpoint,
    /// Ends the system messages or a user message, where a later request most often
    /// starts to differ. The others end a whole request.
    turn: bool,
}
impl Checkpoint {
    fn len(&self) -> usize {
        self.state.tokens().len()
    }
}
impl Conversation {
    /// Brings the cache to the longest state it holds that `tokens` continues and returns
    /// that state's token count.
    fn reuse(&mut self, tokens: &[TokenId]) -> Result<usize, CacheError> {
        // Generation needs at least one token to prefill.
        let reusable = &tokens[..tokens.len().saturating_sub(1)];
        let common = match self.cache.tokens() {
            Ok(cached) => {
                let same = cached.iter().zip(reusable).take_while(|(a, b)| a == b);
                let common = same.count();
                if common == cached.len() {
                    return Ok(common);
                }
                common
            }
            Err(_) => 0,
        };
        let within = |checkpoint: &Checkpoint| checkpoint.len() <= common;
        let Some(index) = self.checkpoints.iter().rposition(within) else {
            self.checkpoints.clear();
            self.cache.reset()?;
            return Ok(0);
        };
        self.cache.rewind(&self.checkpoints[index].state)?;
        self.checkpoints.truncate(index + 1);
        Ok(self.checkpoints[index].len())
    }
}
fn keep(checkpoints: &mut Vec<Checkpoint>, checkpoint: Checkpoint) {
    let turn = checkpoint.turn;
    checkpoints.push(checkpoint);
    let alike = (0..checkpoints.len()).filter(|&index| checkpoints[index].turn == turn);
    let alike: Vec<_> = alike.collect();
    if alike.len() > KEPT {
        // The first turn checkpoint is the one a new conversation can share.
        checkpoints.remove(alike[usize::from(turn)]);
    }
}

/// Loads the model and serves jobs one at a time; `Model` cannot leave its thread.
pub(crate) fn run(
    directory: &Path,
    mut jobs: Receiver<Job>,
    ready: SyncSender<Result<(), String>>,
) {
    let load = || {
        let model = Model::from_dir(directory).map_err(|error| chain(&error))?;
        let cache = model.new_cache(Default::default());
        Ok((model, cache.map_err(|error| chain(&error))?))
    };
    let (mut model, cache) = match load() {
        Ok(loaded) => loaded,
        Err(message) => {
            let _ = ready.send(Err(message));
            return;
        }
    };
    let mut conversation = Conversation {
        cache,
        checkpoints: Vec::new(),
    };
    if ready.send(Ok(())).is_err() {
        return;
    }
    while let Some(Job { request, reply }) = jobs.blocking_recv() {
        if let Err(failure) = serve(&mut model, &mut conversation, &request, &reply) {
            let _ = reply.blocking_send(Err(failure));
        }
    }
}

fn has_role(message: &TemplateValue, role: &str) -> bool {
    let sent = message.get_attr("role");
    sent.is_ok_and(|sent| sent.as_str() == Some(role))
}

/// Stops early once the reply's receiver is gone, which is a client that stopped waiting.
fn serve(
    model: &mut Model,
    conversation: &mut Conversation,
    request: &ChatRequest,
    reply: &Sender<Result<Update, Failure>>,
) -> Result<(), Failure> {
    let invalid = |error: &dyn Error| Failure::Request(chain(error));
    let failed = |error: CacheError| Failure::Generation(chain(&error));
    let tools = request.tools.clone().unwrap_or_default();
    let tokenizer = model.tokenizer();
    let template = |continuation| ChatTemplateOptions {
        continuation,
        tools: tools.clone(),
        ..Default::default()
    };
    let messages = request.messages().map_err(Failure::Request)?;
    let prompt = tokenizer
        .render_chat(&messages, template(ChatContinuation::StartAssistant))
        .map_err(|error| invalid(&error))?;
    let tokens = tokenizer.encode(&prompt).map_err(|error| invalid(&error))?;
    let mut parser = tokenizer.response_parser(&prompt, &tools);

    let shared = |messages: &[TemplateValue]| {
        let closed = template(ChatContinuation::Closed);
        let rendered = tokenizer.render_chat(messages, closed).ok()?;
        let alone = tokenizer.encode(&rendered).ok()?;
        let same = alone.iter().zip(&tokens).take_while(|(a, b)| a == b);
        Some(same.count())
    };
    let system = messages
        .iter()
        .take_while(|message| has_role(message, "system"));
    let system = system.count();
    let user = messages
        .iter()
        .rposition(|message| has_role(message, "user"));
    let user = user.map_or(0, |index| index + 1);
    // Qwen's template refuses messages without a user's, so the system messages are
    // followed by an empty one, as in mlx_lm.server.
    let empty = TemplateValue::from_pairs([("role", "user"), ("content", "")]);
    let boundaries: Vec<(usize, bool)> = [
        (system > 0).then(|| (shared(&[&messages[..system], &[empty]].concat()), true)),
        (user > 0).then(|| (shared(&messages[..user]), true)),
        (user < messages.len()).then(|| (shared(&messages), false)),
    ]
    .into_iter()
    .flatten()
    .filter_map(|(count, turn)| Some((count?, turn)))
    .collect();

    let mut options = request.options();
    options.prefill_boundaries = boundaries.iter().map(|(count, _)| *count).collect();
    options
        .sampling
        .validate(model.config().dimensions.vocabulary_size)
        .map_err(|error| invalid(&GenerationError::from(error)))?;
    let cached_tokens = conversation.reuse(&tokens).map_err(failed)?;
    let Conversation { cache, checkpoints } = conversation;
    let mut generation = model
        .generate_with_cache(Prompt::Tokens(&tokens), options, cache)
        .map_err(|error| invalid(&error))?;

    let send = |update| {
        let _ = reply.blocking_send(Ok(update));
    };
    let mut completion_tokens = 0;
    while let Some(event) = generation.next() {
        if reply.is_closed() {
            break;
        }
        match event.map_err(|error| Failure::Generation(chain(&error)))? {
            GenerationEvent::Prefill {
                processed, total, ..
            } => {
                let count = cached_tokens + processed;
                let boundary = boundaries.iter().find(|(boundary, _)| *boundary == count);
                let newest = checkpoints.last().map_or(0, Checkpoint::len);
                if let Some(&(_, turn)) = boundary.filter(|_| newest < count) {
                    let state = generation.checkpoint().map_err(failed)?;
                    keep(checkpoints, Checkpoint { state, turn });
                }
                send(Update::Prefill { processed, total });
            }
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
                    cached_tokens,
                    completion_tokens,
                });
                break;
            }
            _ => {}
        }
    }
    Ok(())
}
