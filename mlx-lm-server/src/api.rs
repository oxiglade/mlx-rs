use std::{
    collections::BTreeMap,
    convert::Infallible,
    future::ready,
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::{rejection::JsonRejection, State},
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::{stream, StreamExt};
use mlx_lm::{FinishReason, GenerationOptions, MinPOptions, ResponsePart, TemplateValue};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc::{self, Sender};

use crate::worker::{Failure, Job, Update};

#[derive(Deserialize)]
pub(crate) struct ChatRequest {
    messages: Vec<TemplateValue>,
    pub tools: Option<Vec<TemplateValue>>,
    model: Option<String>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    max_completion_tokens: Option<NonZeroUsize>,
    max_tokens: Option<NonZeroUsize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    seed: Option<u64>,
    stop: Option<Stop>,
    pub chat_template_kwargs: Option<BTreeMap<String, TemplateValue>>,
    pub reasoning_effort: Option<TemplateValue>,
}
#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}
#[derive(Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

/// `map` with `key` set to `value`, in the key's place when it has one.
fn with(map: &TemplateValue, key: &str, value: TemplateValue) -> TemplateValue {
    let mut value = Some(value);
    let mut pairs = Vec::new();
    for name in map.try_iter().into_iter().flatten() {
        let replaced = (name.as_str() == Some(key)).then(|| value.take()).flatten();
        let item = replaced.or_else(|| map.get_item(&name).ok());
        pairs.push((name, item.unwrap_or_default()));
    }
    pairs.extend(value.map(|value| (key.into(), value)));
    TemplateValue::from_pairs(pairs)
}

fn prepared(message: &TemplateValue) -> Result<TemplateValue, String> {
    let Ok(Value::Object(fields)) = serde_json::to_value(message) else {
        return Ok(message.clone());
    };
    let mut message = message.clone();
    match fields.get("content") {
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                match (&part["type"], &part["text"]) {
                    (Value::String(kind), Value::String(part)) if kind == "text" => text += part,
                    _ => return Err("only text content parts are supported".into()),
                }
            }
            message = with(&message, "content", text.into());
        }
        None | Some(Value::Null) => message = with(&message, "content", "".into()),
        _ => {}
    }
    if fields.get("tool_calls").is_some_and(Value::is_array) {
        let mut calls = Vec::new();
        let sent = message.get_attr("tool_calls").unwrap_or_default();
        for call in sent.try_iter().into_iter().flatten() {
            let function = call.get_attr("function").unwrap_or_default();
            let arguments = function.get_attr("arguments").unwrap_or_default();
            calls.push(match arguments.as_str() {
                Some(text) if !text.is_empty() => {
                    let arguments = serde_json::from_str(text)
                        .map_err(|error| format!("tool call arguments are not JSON: {error}"))?;
                    with(&call, "function", with(&function, "arguments", arguments))
                }
                _ => call,
            });
        }
        message = with(&message, "tool_calls", calls.into_iter().collect());
    }
    Ok(message)
}

impl ChatRequest {
    /// The messages as `mlx_lm.server` hands them to the template: text parts joined into
    /// one string, absent or null content empty, and tool-call arguments sent as a JSON
    /// string parsed into an object.
    pub fn messages(&self) -> Result<Vec<TemplateValue>, String> {
        self.messages.iter().map(prepared).collect()
    }
    /// A request without a limit gets `max_tokens`. Other absent fields take
    /// `mlx_lm.server`'s defaults: greedy, no filters. As there, a zero `top_p`, `top_k` or
    /// `min_p` and an empty `stop` string switch that setting off.
    pub fn options(&self, max_tokens: NonZeroUsize) -> GenerationOptions {
        let mut options = GenerationOptions::default();
        let limit = self.max_completion_tokens.or(self.max_tokens);
        options.max_tokens = limit.unwrap_or(max_tokens);
        if let Some(temperature) = self.temperature {
            options.sampling.temperature = temperature;
        }
        options.sampling.top_p = self.top_p.filter(|&p| p != 0.0);
        options.sampling.top_k = self.top_k.and_then(NonZeroUsize::new);
        options.sampling.min_p =
            self.min_p
                .filter(|&probability| probability != 0.0)
                .map(|probability| MinPOptions {
                    probability,
                    min_tokens_to_keep: NonZeroUsize::MIN,
                });
        options.sampling.seed = self.seed;
        options.stop.stop_strings = match &self.stop {
            Some(Stop::One(stop)) if !stop.is_empty() => vec![stop.clone()],
            Some(Stop::Many(stops)) => stops.clone(),
            _ => Vec::new(),
        };
        options
    }
}

#[derive(Clone)]
struct App {
    model: Arc<str>,
    reasoning: Arc<str>,
    jobs: Sender<Job>,
}

pub(crate) fn router(model: String, reasoning: String, jobs: Sender<Job>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .with_state(App {
            model: model.into(),
            reasoning: reasoning.into(),
            jobs,
        })
}

fn now() -> std::time::Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
}
/// An identifier no other response or tool call of this server shares.
fn unique(prefix: &str) -> String {
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let count = COUNT.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{:x}-{count:x}", now().as_nanos())
}
fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({ "error": message.into() }))).into_response()
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn models(State(app): State<App>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{ "id": &*app.model, "object": "model", "created": now().as_secs() }],
    }))
}

#[derive(Default)]
struct ToolCalls {
    /// The model wrote a call, readable or not.
    made: bool,
    read: usize,
}

/// The content, reasoning and tool calls in `parts`, as the fields of a message or, with
/// `stream`, of a delta.
fn fields(
    parts: Vec<ResponsePart>,
    calls: &mut ToolCalls,
    stream: bool,
    reasoning: &str,
) -> Map<String, Value> {
    let (mut content, mut thoughts) = (String::new(), String::new());
    let mut tool_calls = Vec::new();
    for part in parts {
        match part {
            ResponsePart::Reasoning(text) => thoughts += &text,
            ResponsePart::Content(text) => content += &text,
            ResponsePart::ToolCall(call) => {
                let mut tool_call = json!({
                    "id": unique("call"),
                    "type": "function",
                    "function": { "name": call.name, "arguments": call.arguments },
                });
                if stream {
                    tool_call["index"] = calls.read.into();
                }
                calls.read += 1;
                calls.made = true;
                tool_calls.push(tool_call);
            }
            // mlx_lm.server drops a call it cannot read and still reports tool_calls.
            ResponsePart::InvalidToolCall(text) => {
                calls.made = true;
                eprintln!("mlx-lm-server: unreadable tool call: {text}");
            }
            _ => {}
        }
    }
    let mut fields = Map::new();
    for (name, text) in [("content", content), (reasoning, thoughts)] {
        if !text.is_empty() {
            fields.insert(name.into(), text.into());
        }
    }
    if !tool_calls.is_empty() {
        fields.insert("tool_calls".into(), tool_calls.into());
    }
    fields
}
fn finish_reason(finish: FinishReason, calls: &ToolCalls) -> &'static str {
    match finish {
        FinishReason::Length => "length",
        _ if calls.made => "tool_calls",
        _ => "stop",
    }
}
fn usage(prompt_tokens: usize, cached_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": { "cached_tokens": cached_tokens },
    })
}

async fn chat(State(app): State<App>, body: Result<Json<ChatRequest>, JsonRejection>) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => return error(StatusCode::BAD_REQUEST, rejection.body_text()),
    };
    let stopped = || {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "the model worker stopped",
        )
    };
    let streaming = request.stream;
    let include_usage = matches!(&request.stream_options, Some(options) if options.include_usage);
    let (id, created) = (unique("chatcmpl"), now().as_secs());
    let model = request
        .model
        .clone()
        .unwrap_or_else(|| app.model.to_string());
    let envelope = move |object: &str, choices: Value| {
        json!({
            "id": &id,
            "object": object,
            "created": created,
            "model": &model,
            "system_fingerprint": concat!("mlx-lm-server-", env!("CARGO_PKG_VERSION")),
            "choices": choices,
        })
    };
    let (reply, mut updates) = mpsc::channel(32);
    if app.jobs.send(Job { request, reply }).await.is_err() {
        return stopped();
    }
    // A request the worker refuses fails before its first update, while the status can
    // still say so.
    let first = match updates.recv().await {
        Some(Ok(update)) => update,
        Some(Err(Failure::Request(message))) => return error(StatusCode::BAD_REQUEST, message),
        Some(Err(Failure::Generation(message))) => {
            return error(StatusCode::INTERNAL_SERVER_ERROR, message)
        }
        None => return stopped(),
    };
    let mut updates =
        stream::once(ready(Ok(first))).chain(stream::poll_fn(move |cx| updates.poll_recv(cx)));
    let mut calls = ToolCalls::default();

    if streaming {
        let events = updates.flat_map(move |update| {
            let chunk = |delta: Map<String, Value>, finish: Option<&str>| {
                let choice = json!({ "index": 0, "finish_reason": finish, "delta": delta });
                Event::default()
                    .data(envelope("chat.completion.chunk", json!([choice])).to_string())
            };
            let role = || Map::from_iter([("role".to_owned(), "assistant".into())]);
            let events = match update {
                Ok(Update::Prefill { processed, total }) => {
                    vec![Event::default().comment(format!("keepalive {processed}/{total}"))]
                }
                Ok(Update::Parts(parts)) => {
                    let mut delta = fields(parts, &mut calls, true, &app.reasoning);
                    if delta.is_empty() {
                        Vec::new()
                    } else {
                        delta.extend(role());
                        vec![chunk(delta, None)]
                    }
                }
                Ok(Update::Done {
                    finish,
                    prompt_tokens,
                    cached_tokens,
                    completion_tokens,
                }) => {
                    let mut events = vec![chunk(role(), Some(finish_reason(finish, &calls)))];
                    if include_usage {
                        let mut last = envelope("chat.completion.chunk", json!([]));
                        last["usage"] = usage(prompt_tokens, cached_tokens, completion_tokens);
                        events.push(Event::default().data(last.to_string()));
                    }
                    events.push(Event::default().data("[DONE]"));
                    events
                }
                Err(Failure::Request(message) | Failure::Generation(message)) => {
                    vec![Event::default().data(json!({ "error": message }).to_string())]
                }
            };
            stream::iter(events.into_iter().map(Ok::<_, Infallible>))
        });
        return Sse::new(events).into_response();
    }

    let mut parts = Vec::new();
    while let Some(update) = updates.next().await {
        match update {
            Ok(Update::Prefill { .. }) => {}
            Ok(Update::Parts(more)) => parts.extend(more),
            Ok(Update::Done {
                finish,
                prompt_tokens,
                cached_tokens,
                completion_tokens,
            }) => {
                let mut message = fields(parts, &mut calls, false, &app.reasoning);
                message.insert("role".into(), "assistant".into());
                message.entry("content").or_insert(Value::Null);
                let choice = json!({
                    "index": 0,
                    "finish_reason": finish_reason(finish, &calls),
                    "message": message,
                });
                let mut completion = envelope("chat.completion", json!([choice]));
                completion["usage"] = usage(prompt_tokens, cached_tokens, completion_tokens);
                return Json(completion).into_response();
            }
            Err(Failure::Request(message) | Failure::Generation(message)) => {
                return error(StatusCode::INTERNAL_SERVER_ERROR, message)
            }
        }
    }
    stopped()
}
