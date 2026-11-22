use super::{holds, python_json, TemplateValue};
use crate::FinishReason;
use minijinja::value::ValueKind;
use std::fmt::Write;

/// One piece of a response, in the order the model generated it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResponsePart {
    /// Reasoning text.
    Reasoning(String),
    /// Answer text.
    Content(String),
    /// A complete tool call.
    ToolCall(ToolCall),
    /// The text of a tool call that could not be read, such as one cut off by the token limit.
    InvalidToolCall(String),
}
/// A tool call made by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ToolCall {
    /// The tool's name.
    pub name: String,
    /// The arguments as JSON, written as Python's `json.dumps` writes them.
    pub arguments: String,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Skipped,
    Reasoning,
    Content,
    Call,
}
/// Inkling opens each part of a message with one of these tokens.
const CHANNELS: [(&str, State); 6] = [
    ("<|content_thinking|>", State::Reasoning),
    ("<|content_text|>", State::Content),
    ("<|content_invoke_tool_json|>", State::Call),
    ("<|end_message|>", State::Skipped),
    ("<|message_model|>", State::Skipped),
    ("<|content_model_end_sampling|>", State::Skipped),
];
/// How a checkpoint marks reasoning and tool calls in the text it generates.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Format {
    /// `<think>` blocks, and `<tool_call>` blocks holding `<function=name>` with
    /// `<parameter=name>` values, each read only when the checkpoint writes it.
    Tags { think: bool, calls: bool },
    /// Inkling's channel tokens, with a tool call as a JSON object of `name` and `args`.
    Channels,
}
impl Format {
    pub(crate) fn infer(template: Option<&str>, has_token: impl Fn(&str) -> bool) -> Self {
        if CHANNELS[..3].iter().all(|(marker, _)| has_token(marker)) {
            return Self::Channels;
        }
        Self::Tags {
            think: has_token("<think>") && has_token("</think>"),
            // mlx-lm reads calls this way when the template writes them this way.
            calls: template.is_some_and(|template| {
                template.contains("<tool_call>\n<function=")
                    || template.contains("<tool_call>\\n<function=")
            }),
        }
    }
}
/// Splits generated text into reasoning, content and tool calls as it arrives.
///
/// Qwen-style `<think>` and `<tool_call>` output follows mlx-lm's server. Inkling 2 Mini
/// output is split wherever the text of one of its channel tokens appears. Push the text
/// of every generation event, the last one included.
#[derive(Debug)]
pub struct ResponseParser {
    format: Format,
    /// The markers that end a state, the state each begins, and `None` for any state.
    edges: Vec<(Option<State>, &'static str, State)>,
    state: State,
    held: String,
    call: String,
    tools: Vec<TemplateValue>,
}
impl ResponseParser {
    pub(crate) fn new(format: Format, prompt: &str, tools: &[TemplateValue]) -> Self {
        use State::{Call, Content, Reasoning, Skipped};
        let (edges, state) = match format {
            Format::Channels => (
                CHANNELS.map(|(marker, to)| (None, marker, to)).to_vec(),
                Skipped,
            ),
            Format::Tags { think, calls } => {
                let mut edges = Vec::new();
                if think {
                    edges.extend([
                        (Some(Content), "<think>", Reasoning),
                        (Some(Reasoning), "</think>", Content),
                    ]);
                }
                if calls {
                    edges.extend([
                        (Some(Content), "<tool_call>", Call),
                        (Some(Reasoning), "<tool_call>", Call),
                        (Some(Call), "</tool_call>", Content),
                    ]);
                }
                // A prompt may end inside a reasoning block that the model continues.
                let open = think && prompt.rfind("<think>") > prompt.rfind("</think>");
                (edges, if open { Reasoning } else { Content })
            }
        };
        Self {
            format,
            edges,
            state,
            held: String::new(),
            call: String::new(),
            tools: tools.to_vec(),
        }
    }
    /// Reads more generated text and returns the parts it completes. Text that may be
    /// the start of a marker is held until later text settles it.
    pub fn push(&mut self, text: &str) -> Vec<ResponsePart> {
        let mut parts = Vec::new();
        for ch in text.chars() {
            self.held.push(ch);
            let state = self.state;
            let mut markers = self
                .edges
                .iter()
                .filter(|(from, ..)| from.is_none_or(|from| from == state))
                .map(|&(_, marker, to)| (marker, to));
            if let Some((marker, to)) = markers
                .clone()
                .find(|(marker, _)| self.held.ends_with(marker))
            {
                self.held.truncate(self.held.len() - marker.len());
                self.release(&mut parts);
                if state == State::Call {
                    self.end_call(&mut parts);
                }
                self.state = to;
            } else if !markers
                .any(|(marker, _)| (1..marker.len()).any(|len| self.held.ends_with(&marker[..len])))
            {
                self.release(&mut parts);
            }
        }
        parts
    }
    /// Ends the response and returns its remaining parts.
    pub fn finish(mut self, reason: FinishReason) -> Vec<ResponsePart> {
        let mut parts = Vec::new();
        // mlx-lm drops text held as a possible marker when the model stops.
        if matches!(self.format, Format::Tags { .. }) && reason == FinishReason::Stop {
            self.held.clear();
        }
        self.release(&mut parts);
        if self.state == State::Call && !self.call.is_empty() {
            // Inkling's server drops a call the token limit ends, even one that parses.
            if matches!(self.format, Format::Channels) && reason == FinishReason::Length {
                parts.push(ResponsePart::InvalidToolCall(self.call));
            } else {
                self.end_call(&mut parts);
            }
        }
        parts
    }
    fn release(&mut self, parts: &mut Vec<ResponsePart>) {
        match (self.state, parts.last_mut()) {
            (State::Skipped, _) => {}
            (State::Call, _) => self.call.push_str(&self.held),
            (State::Reasoning, Some(ResponsePart::Reasoning(text)))
            | (State::Content, Some(ResponsePart::Content(text))) => text.push_str(&self.held),
            (State::Reasoning, _) if !self.held.is_empty() => {
                parts.push(ResponsePart::Reasoning(self.held.clone()));
            }
            (State::Content, _) if !self.held.is_empty() => {
                parts.push(ResponsePart::Content(self.held.clone()));
            }
            _ => {}
        }
        self.held.clear();
    }
    fn end_call(&mut self, parts: &mut Vec<ResponsePart>) {
        let text = std::mem::take(&mut self.call);
        let call = match self.format {
            Format::Tags { .. } => tagged_call(&text, &self.tools),
            Format::Channels if text.is_empty() => return,
            Format::Channels => json_call(&text),
        };
        parts.push(match call {
            Some(call) => ResponsePart::ToolCall(call),
            None => ResponsePart::InvalidToolCall(text),
        });
    }
}
fn json_call(text: &str) -> Option<ToolCall> {
    let call: TemplateValue = serde_json::from_str(text).ok()?;
    let arguments = call
        .get_attr("args")
        .ok()
        .filter(|args| !args.is_undefined())?;
    Some(ToolCall {
        name: call.get_attr("name").ok()?.as_str()?.to_owned(),
        arguments: python_json(&arguments, (", ", ": "), false, true).ok()?,
    })
}
/// mlx-lm's `qwen3_coder` tool parser.
fn tagged_call(text: &str, tools: &[TemplateValue]) -> Option<ToolCall> {
    // The function has to end the text, or come before one last newline.
    let text = text.strip_suffix('\n').unwrap_or(text);
    let text = text.strip_suffix("</function>")?;
    let (name, mut rest) = leading_name(text.split_once("<function=")?.1)?;
    let properties = tools
        .iter()
        .filter_map(|tool| tool.get_attr("function").ok())
        .find(|function| {
            function
                .get_attr("name")
                .is_ok_and(|n| n.as_str() == Some(name))
        })
        .and_then(|function| function.get_attr("parameters").ok())
        .and_then(|parameters| parameters.get_attr("properties").ok())
        .unwrap_or_default();
    let mut arguments = Vec::new();
    while let Some((_, after)) = rest.split_once("<parameter=") {
        let Some((parameter, after)) = after.split_once("</parameter>") else {
            break;
        };
        rest = after;
        let Some((key, value)) = leading_name(parameter) else {
            continue;
        };
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        let schema = properties.get_attr(key).unwrap_or_default();
        arguments.push((TemplateValue::from(key), argument(value, &schema)?));
    }
    Some(ToolCall {
        name: name.to_owned(),
        arguments: python_json(
            &TemplateValue::from_pairs(arguments),
            (", ", ": "),
            false,
            false,
        )
        .ok()?,
    })
}
/// qwen3_coder's name pattern, `\s*([^\s<>]+)>?`.
fn leading_name(text: &str) -> Option<(&str, &str)> {
    let text = text.trim_start();
    let end = text
        .find(|c: char| c.is_whitespace() || c == '<' || c == '>')
        .unwrap_or(text.len());
    let (name, rest) = text.split_at(end);
    (!name.is_empty()).then_some((name, rest.strip_prefix('>').unwrap_or(rest)))
}
/// Reads a parameter's text as the type its schema names. `None` is a value mlx-lm
/// rejects, which drops the whole call.
fn argument(value: &str, schema: &TemplateValue) -> Option<TemplateValue> {
    if value.to_lowercase() == "null" {
        return Some(().into());
    }
    if !schema.is_true() {
        return Some(value.into());
    }
    let kind = schema.get_attr("type").unwrap_or_default();
    if kind.is_undefined() {
        return Some(
            serde_json::from_str::<TemplateValue>(value)
                .ok()
                .filter(|json| matches!(json.kind(), ValueKind::Map | ValueKind::Seq))
                .unwrap_or_else(|| value.into()),
        );
    }
    // Only a string names a type; mlx-lm reads anything else as an unknown type.
    let kind = kind.as_str().unwrap_or_default().trim().to_lowercase();
    let is = |names: &[&str]| names.contains(&kind.as_str());
    let starts = |prefixes: &[&str]| prefixes.iter().any(|prefix| kind.starts_with(prefix));
    let number = || value.trim().parse().ok().filter(|n: &f64| n.is_finite());
    let whole =
        |n: f64| (n.fract() == 0.0 && n.abs() < 1e38).then(|| TemplateValue::from(n as i128));
    Some(
        if is(&["string", "str", "text", "varchar", "char", "enum"]) {
            value.into()
        } else if starts(&["int", "uint", "long", "short", "unsigned"]) {
            whole(number()?)?
        } else if starts(&["num", "float"]) {
            let number = number()?;
            whole(number).unwrap_or_else(|| number.into())
        } else if is(&["boolean", "bool", "binary"]) {
            (value.to_lowercase() == "true").into()
        } else {
            (is(&["object", "array", "arr"]) || starts(&["dict", "list"]))
                .then(|| lenient_json(value))
                .flatten()
                .or_else(|| python_literal(value))
                .unwrap_or_else(|| value.into())
        },
    )
}
/// Python's `json.loads(strict=False)`, which lets control characters stand unescaped
/// inside strings.
fn lenient_json(text: &str) -> Option<TemplateValue> {
    let mut json = String::with_capacity(text.len());
    let (mut quoted, mut escaped) = (false, false);
    for ch in text.chars() {
        if quoted && !escaped && ch < ' ' {
            write!(json, "\\u{:04x}", u32::from(ch)).expect("writing to a string");
        } else {
            json.push(ch);
        }
        if escaped {
            escaped = false;
        } else if ch == '\\' {
            escaped = quoted;
        } else if ch == '"' {
            quoted = !quoted;
        }
    }
    serde_json::from_str(&json).ok()
}
/// The part of Python's `ast.literal_eval` that JSON also spells: numbers, double-quoted
/// strings, lists and dicts, besides a bare `True`, `False` or `None`. JSON's `true`,
/// `false` and `null` are not Python literals, so text holding one is not a literal.
fn python_literal(text: &str) -> Option<TemplateValue> {
    match text.trim() {
        "True" => Some(true.into()),
        "False" => Some(false.into()),
        "None" => Some(().into()),
        _ => serde_json::from_str::<TemplateValue>(text)
            .ok()
            .filter(|json| !holds(json, &[ValueKind::Bool, ValueKind::None])),
    }
}
