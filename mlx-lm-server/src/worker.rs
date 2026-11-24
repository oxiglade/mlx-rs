use std::{
    collections::BTreeMap,
    error::Error,
    num::NonZeroUsize,
    path::Path,
    sync::mpsc::SyncSender,
    thread,
    time::{Duration, SystemTime},
};

use mlx_lm::{
    Cache, CacheCheckpoint, CacheError, ChatContinuation, ChatTemplateOptions, FinishReason,
    GenerationError, GenerationEvent, Model, Prompt, ResponsePart, TemplateValue, TokenId,
};
use tokio::sync::mpsc::{error::TryRecvError, Receiver, Sender};

use crate::{api::ChatRequest, disk::Disk};

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
/// How long the server is idle before it writes to disk, so that a client between two
/// tool calls does not wait for a write.
const SETTLE: Duration = Duration::from_secs(1);

/// A cache and the checkpoints it can rewind to, each a prefix of the cache's tokens,
/// shortest first.
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
    /// The longest state this cache can be brought to that `tokens` continues: its token
    /// count and, unless the cache is in that state, the index of its checkpoint. None
    /// when that would drop more than half of the cache, llama.cpp's rule: a new chat that
    /// shares a system prompt must not cost this one its history.
    fn longest(&self, tokens: &[TokenId]) -> Option<(usize, Option<usize>)> {
        // Generation needs at least one token to prefill.
        let reusable = &tokens[..tokens.len().saturating_sub(1)];
        let cached = self.cache.tokens().ok()?;
        let same = cached.iter().zip(reusable).take_while(|(a, b)| a == b);
        let common = same.count();
        if common == cached.len() {
            return Some((common, None));
        }
        let within = |checkpoint: &Checkpoint| checkpoint.len() <= common;
        let index = self.checkpoints.iter().rposition(within)?;
        let count = self.checkpoints[index].len();
        (2 * count >= cached.len()).then_some((count, Some(index)))
    }
    /// Writes one file the disk lacks of the two states a later process resumes from:
    /// the first, which a new conversation can share, and the newest.
    fn save(&self, disk: &mut Disk, since: SystemTime) -> bool {
        let ends = [self.checkpoints.first(), self.checkpoints.last()];
        let ends = ends.into_iter().flatten();
        let ends: Vec<_> = ends.map(|checkpoint| &checkpoint.state).collect();
        disk.save(&self.cache, &ends, since)
    }
    fn nbytes(&self) -> usize {
        let checkpoints = self.checkpoints.iter();
        let sizes = checkpoints.map(|checkpoint| checkpoint.state.nbytes());
        let sizes = sizes.chain([self.cache.nbytes()]);
        sizes.sum::<Result<usize, _>>().unwrap_or(usize::MAX)
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

#[derive(Clone, Copy)]
pub(crate) struct Limits {
    pub max_tokens: Option<NonZeroUsize>,
    pub max_prompt_tokens: Option<usize>,
    pub draft: bool,
}

/// Loads the model and serves jobs one at a time; `Model` cannot leave its thread.
pub(crate) fn run(
    directory: &Path,
    budget: usize,
    mut disk: Option<Disk>,
    limits: Limits,
    variables: &BTreeMap<String, TemplateValue>,
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
    // Least recently used first.
    let mut conversations: Vec<Conversation> = Vec::new();
    // When the last request ended, until the disk has its conversation.
    let mut unsaved: Option<SystemTime> = None;
    loop {
        let job = match (jobs.try_recv(), &mut disk, unsaved) {
            (Ok(job), ..) => Some(job),
            // A closed channel is a server that is stopping, which writes without the wait.
            (Err(error), Some(disk), Some(since)) => {
                let settling = since.elapsed().is_ok_and(|idle| idle < SETTLE);
                if error == TryRecvError::Empty && settling {
                    thread::sleep(Duration::from_millis(5));
                } else if !conversations
                    .last()
                    .is_some_and(|last| last.save(disk, since))
                {
                    unsaved = None;
                }
                continue;
            }
            _ => jobs.blocking_recv(),
        };
        let Some(Job { request, reply }) = job else {
            return;
        };
        let served = serve(
            &mut model,
            &mut conversations,
            disk.as_mut(),
            limits,
            variables,
            &request,
            &reply,
        );
        if let Err(failure) = served {
            let _ = reply.blocking_send(Err(failure));
        }
        // A failed request can leave an empty or an invalidated cache.
        conversations.retain(|conversation| {
            let tokens = conversation.cache.tokens();
            tokens.is_ok_and(|tokens| !tokens.is_empty())
        });
        let mut total = 0usize;
        let fits = conversations.iter().rev().take_while(|conversation| {
            total = total.saturating_add(conversation.nbytes());
            total <= budget
        });
        let kept = fits.count().max(1);
        conversations.drain(..conversations.len().saturating_sub(kept));
        unsaved = Some(SystemTime::now());
    }
}

fn has_role(message: &TemplateValue, role: &str) -> bool {
    let sent = message.get_attr("role");
    sent.is_ok_and(|sent| sent.as_str() == Some(role))
}

/// Stops early once the reply's receiver is gone, which is a client that stopped waiting.
fn serve(
    model: &mut Model,
    conversations: &mut Vec<Conversation>,
    disk: Option<&mut Disk>,
    limits: Limits,
    variables: &BTreeMap<String, TemplateValue>,
    request: &ChatRequest,
    reply: &Sender<Result<Update, Failure>>,
) -> Result<(), Failure> {
    let invalid = |error: &dyn Error| Failure::Request(chain(error));
    let failed = |error: CacheError| Failure::Generation(chain(&error));
    let tools = request.tools.clone().unwrap_or_default();
    let tokenizer = model.tokenizer();
    // The order of precedence is vLLM's.
    let sent = request.chat_template_kwargs.clone().unwrap_or_default();
    let mut variables = variables.clone();
    if let Some(effort) = &request.reasoning_effort {
        let thinks = effort.as_str() != Some("none");
        variables.insert("enable_thinking".into(), thinks.into());
    }
    variables.extend(sent);
    if let Some(effort) = &request.reasoning_effort {
        variables.insert("reasoning_effort".into(), effort.clone());
    }
    let template = |continuation| ChatTemplateOptions {
        continuation,
        tools: tools.clone(),
        variables: variables.clone(),
        ..Default::default()
    };
    let messages = request.messages().map_err(Failure::Request)?;
    let prompt = tokenizer
        .render_chat(&messages, template(ChatContinuation::StartAssistant))
        .map_err(|error| invalid(&error))?;
    let tokens = tokenizer.encode(&prompt).map_err(|error| invalid(&error))?;
    let count = tokens.len();
    let exceeded = limits.max_prompt_tokens.filter(|limit| count > *limit);
    if let Some(limit) = exceeded {
        let message = format!("the prompt has {count} tokens and the server takes at most {limit}");
        return Err(Failure::Request(message));
    }
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

    let context = model.config().dimensions.max_positions;
    // The library takes no sequence longer than this.
    let context = context.unwrap_or(i32::MAX as usize);
    let room = NonZeroUsize::new(context.saturating_sub(tokens.len()))
        .ok_or_else(|| Failure::Request("the prompt fills the model's context".into()))?;
    let mut options = request.options(limits.max_tokens.unwrap_or(room));
    options.draft = limits.draft;
    options.prefill_boundaries = boundaries.iter().map(|(count, _)| *count).collect();
    // A prefill step's attention scores take its rows times the context, so past 65,536 tokens
    // the rows shrink to keep that product.
    let rows = options.prefill_chunk_size.get() * 65_536 / tokens.len().max(65_536);
    options.prefill_chunk_size = NonZeroUsize::new(rows).unwrap_or(NonZeroUsize::MIN);
    options
        .sampling
        .validate(model.config().dimensions.vocabulary_size)
        .map_err(|error| invalid(&GenerationError::from(error)))?;
    let states = conversations.iter().enumerate();
    let states =
        states.filter_map(|(index, conversation)| Some((conversation.longest(&tokens)?, index)));
    // Of two that hold as much, the one that needs no rewind loses nothing.
    let best =
        states.max_by_key(|((count, checkpoint), index)| (*count, checkpoint.is_none(), *index));
    let held = best.map_or(0, |((count, _), _)| count);
    let restored = disk.and_then(|disk| disk.restore(model, &tokens, held));
    let (cached_tokens, conversation) = match (restored, best) {
        (Some((cache, state)), _) => {
            let checkpoints = vec![Checkpoint { state, turn: true }];
            (checkpoints[0].len(), Conversation { cache, checkpoints })
        }
        (None, Some(((count, checkpoint), index))) => {
            let mut conversation = conversations.remove(index);
            if let Some(index) = checkpoint {
                let state = &conversation.checkpoints[index].state;
                conversation.cache.rewind(state).map_err(failed)?;
                conversation.checkpoints.truncate(index + 1);
            }
            (count, conversation)
        }
        (None, None) => {
            let cache = model.new_cache(Default::default()).map_err(failed)?;
            let checkpoints = Vec::new();
            (0, Conversation { cache, checkpoints })
        }
    };
    let last = conversations.len();
    conversations.push(conversation);
    let Conversation { cache, checkpoints } = &mut conversations[last];
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
