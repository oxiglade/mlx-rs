mod gguf;
mod response;
pub use crate::error::{ChatTemplateError, TokenizerError};
use minijinja::{
    context,
    syntax::SyntaxConfig,
    value::{Kwargs, Serde, ValueKind},
    Environment,
};
use serde::Serialize;
use serde_json::Value;
use std::{collections::BTreeMap, io, path::Path};

/// A value passed to a chat template. Objects keep the key order they were built with.
pub use minijinja::Value as TemplateValue;
pub use response::{ResponseParser, ResponsePart, ToolCall};

/// A token identifier in the model's vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenId(u32);
impl From<u32> for TokenId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}
impl From<TokenId> for u32 {
    fn from(value: TokenId) -> Self {
        value.0
    }
}
/// Token encoding, decoding, EOS metadata, and compiled chat templates.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    env: Environment<'static>,
    eos: Vec<TokenId>,
    special_tokens: serde_json::Map<String, Value>,
    has_template: bool,
    response: response::Format,
}
impl Tokenizer {
    /// Loads tokenizer assets with generation-config, model-config, then tokenizer EOS precedence.
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let path = path.as_ref();
        let mut tokenizer = Self::load_assets(path)?;
        tokenizer.eos = tokenizer.resolve_eos(path)?.0;
        Ok(tokenizer)
    }

    fn load_assets(path: &Path) -> Result<Self, TokenizerError> {
        let mut tokenizer = Self::from_file(path.join("tokenizer.json"))?;
        let config = read_optional_json(&path.join("tokenizer_config.json"))?;
        let template = match config.get("chat_template") {
            Some(Value::String(template)) => Some(template.as_str()),
            Some(Value::Object(templates)) => templates.get("default").and_then(Value::as_str),
            Some(Value::Array(templates)) => templates.iter().rev().find_map(|template| {
                (template.get("name").and_then(Value::as_str) == Some("default"))
                    .then(|| template.get("template").and_then(Value::as_str))
                    .flatten()
            }),
            _ => None,
        };
        let file_template = match std::fs::read_to_string(path.join("chat_template.jinja")) {
            Ok(template) => Some(template),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        let template = file_template.as_deref().or(template);
        tokenizer.response = response::Format::infer(template, |token| {
            tokenizer.inner.token_to_id(token).is_some()
        });
        if let Some(template) = template {
            tokenizer
                .env
                .add_template_owned("chat", without_generation_blocks(template))
                .map_err(ChatTemplateError::from)?;
            tokenizer.has_template = true;
        }
        if let Some(config) = config.as_object() {
            for (key, value) in config {
                if key.ends_with("_token") {
                    if let Some(content) = token_content(value) {
                        tokenizer.special_tokens.insert(key.clone(), content.into());
                    }
                } else if key == "additional_special_tokens" {
                    if let Some(values) = value.as_array() {
                        let tokens: Vec<Value> = values
                            .iter()
                            .filter_map(token_content)
                            .map(Value::from)
                            .collect();
                        tokenizer.special_tokens.insert(key.clone(), tokens.into());
                    }
                }
            }
        }
        Ok(tokenizer)
    }

    fn resolve_eos(&self, path: &Path) -> Result<(Vec<TokenId>, &'static str), TokenizerError> {
        let generation = read_optional_json(&path.join("generation_config.json"))?;
        if let Some(value) = generation
            .get("eos_token_id")
            .filter(|value| !value.is_null())
        {
            let ids = parse_eos(value)?;
            // mlx_lm 0.31.3 applies generation overrides only when Python considers them truthy.
            if !ids.is_empty() && value.as_u64() != Some(0) {
                return Ok((ids, "generation_config"));
            }
        }
        let config = read_optional_json(&path.join("config.json"))?;
        if let Some(value) = config.get("eos_token_id").filter(|value| !value.is_null()) {
            return Ok((parse_eos(value)?, "config"));
        }
        let eos = self.special_tokens.get("eos_token").and_then(Value::as_str);
        let ids = match eos {
            Some(token) => {
                vec![TokenId(self.inner.token_to_id(token).ok_or_else(|| {
                    TokenizerError::InvalidEos(token.to_owned())
                })?)]
            }
            None => Vec::new(),
        };
        Ok((ids, "tokenizer"))
    }
    /// Loads a local tokenizer JSON file without network access.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                TokenizerError::MissingFile(path.to_owned())
            } else {
                TokenizerError::Io(error)
            }
        })?;
        Self::from_bytes(&bytes)
    }
    /// Parses a tokenizer JSON document without a chat template or EOS overrides.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes)?;
        Ok(Self {
            response: response::Format::infer(None, |token| inner.token_to_id(token).is_some()),
            inner,
            env: environment(),
            eos: Vec::new(),
            special_tokens: Default::default(),
            has_template: false,
        })
    }
    /// Encodes text without adding special tokens beyond the supplied text.
    pub fn encode(&self, text: &str) -> Result<Vec<TokenId>, TokenizerError> {
        self.encode_with_special_tokens(text, false)
    }

    /// Encodes text with optional special-token insertion by the post-processor.
    pub fn encode_with_special_tokens(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<TokenId>, TokenizerError> {
        Ok(self
            .inner
            .encode(text, add_special_tokens)?
            .get_ids()
            .iter()
            .copied()
            .map(TokenId)
            .collect())
    }
    /// Configured BOS token content from `tokenizer_config.json`, or `None` if absent.
    pub fn bos_token(&self) -> Option<&str> {
        self.special_tokens.get("bos_token").and_then(Value::as_str)
    }
    /// Decodes IDs, preserving special tokens as in Transformers’ default decode.
    pub fn decode(&self, ids: &[TokenId]) -> Result<String, TokenizerError> {
        self.decode_with_special_tokens(ids, false)
    }

    fn decode_with_special_tokens(
        &self,
        ids: &[TokenId],
        skip_special_tokens: bool,
    ) -> Result<String, TokenizerError> {
        let ids: Vec<_> = ids.iter().map(|id| id.0).collect();
        Ok(self.inner.decode(&ids, skip_special_tokens)?)
    }

    /// Generation feeds only generated non-stop IDs, then consumes `finish` on EOS or length.
    pub(crate) fn decode_stream(&self) -> StreamingDecoder<'_> {
        StreamingDecoder {
            tokenizer: self,
            inner: self.inner.decode_stream(false),
            ids: Vec::new(),
            emitted: String::new(),
        }
    }
    /// Renders the compiled model template with a typed continuation policy.
    ///
    /// Each message serializes to the object the template reads. [`Message`] covers plain
    /// text. A conversation with tool calls, tool results or reasoning passes objects in the
    /// OpenAI chat format, with each tool call's `arguments` as an object. Parse such JSON
    /// straight into [`TemplateValue`]: `serde_json::Value` sorts object keys, which
    /// reorders what a template prints.
    pub fn render_chat<M: Serialize>(
        &self,
        messages: &[M],
        options: ChatTemplateOptions,
    ) -> Result<String, ChatTemplateError> {
        if !self.has_template {
            return Err(ChatTemplateError::MissingTemplate);
        }
        render_chat(&self.env, &self.special_tokens, messages, options)
    }
    /// Starts splitting the text generated after `prompt` into reasoning, content and tool
    /// calls. `tools` are the definitions the prompt was rendered with; a tool's parameter
    /// types decide how its arguments are read.
    pub fn response_parser(&self, prompt: &str, tools: &[TemplateValue]) -> ResponseParser {
        ResponseParser::new(self.response, prompt, tools)
    }
    /// Borrows the resolved model EOS set.
    pub fn eos_tokens(&self) -> &[TokenId] {
        &self.eos
    }
}
/// A textual message's role in a conversation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Role {
    /// Instructions for the conversation.
    System,
    /// Input from the user.
    User,
    /// Output from the assistant.
    Assistant,
    /// Text returned by a tool.
    Tool,
    /// A model-specific textual role.
    Other(String),
}
impl Serialize for Role {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Other(role) => role,
        })
    }
}
/// One textual conversation message.
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    /// Semantic role passed to the template.
    pub role: Role,
    /// UTF-8 message content.
    pub content: String,
}
/// Selects how the rendered conversation ends.
#[derive(Debug, Clone, Copy, Default)]
pub enum ChatContinuation {
    /// Closes every message normally.
    #[default]
    Closed,
    /// Appends the template's assistant-start prompt.
    StartAssistant,
    /// Leaves the final message open for continuation.
    ContinueLast,
}
/// Options for rendering one conversation.
#[derive(Debug, Clone, Default)]
pub struct ChatTemplateOptions {
    /// How the final message should be continued.
    pub continuation: ChatContinuation,
    /// `None` leaves the template variable undefined; a value overrides its thinking default.
    pub enable_thinking: Option<bool>,
    /// Tool definitions the template reads as `tools`, usually OpenAI function schemas.
    /// The template sees none when this is empty.
    pub tools: Vec<TemplateValue>,
    /// Further template variables, such as `reasoning_effort`.
    pub variables: BTreeMap<String, TemplateValue>,
}
fn environment() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_syntax(
        SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .expect("default delimiters are valid"),
    );
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |message: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    env.add_filter("tojson", tojson);
    env
}
struct PythonJson<'a> {
    item: &'a str,
    key: &'a str,
    ensure_ascii: bool,
}
impl serde_json::ser::Formatter for PythonJson<'_> {
    fn write_f64<W: ?Sized + io::Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        let text = format!("{value:?}");
        // Python signs every exponent and pads it to two digits.
        match text.split_once('e') {
            Some((mantissa, exponent)) => {
                let (sign, digits) = match exponent.strip_prefix('-') {
                    Some(digits) => ('-', digits),
                    None => ('+', exponent),
                };
                write!(writer, "{mantissa}e{sign}{digits:0>2}")
            }
            None => writer.write_all(text.as_bytes()),
        }
    }
    fn write_string_fragment<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        if !self.ensure_ascii {
            return writer.write_all(fragment.as_bytes());
        }
        for unit in fragment.encode_utf16() {
            match u8::try_from(unit) {
                Ok(byte @ b' '..=b'~') => writer.write_all(&[byte])?,
                _ => write!(writer, "\\u{unit:04x}")?,
            }
        }
        Ok(())
    }
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        writer.write_all(if first { b"" } else { self.item.as_bytes() })
    }
    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        writer.write_all(if first { b"" } else { self.item.as_bytes() })
    }
    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(self.key.as_bytes())
    }
}
/// Whether `value`, or a value at any depth inside it, is one of `kinds`.
pub(crate) fn holds(value: &TemplateValue, kinds: &[ValueKind]) -> bool {
    match value.kind() {
        kind if kinds.contains(&kind) => true,
        ValueKind::Seq => value
            .try_iter()
            .is_ok_and(|mut items| items.any(|item| holds(&item, kinds))),
        ValueKind::Map => value.try_iter().is_ok_and(|mut keys| {
            keys.any(|key| value.get_item(&key).is_ok_and(|item| holds(&item, kinds)))
        }),
        _ => false,
    }
}
/// Python's `json.dumps` with the given separators.
pub(crate) fn python_json(
    value: &TemplateValue,
    (item, key): (&str, &str),
    sort_keys: bool,
    ensure_ascii: bool,
) -> Result<String, serde_json::Error> {
    let mut json = Vec::new();
    let formatter = PythonJson {
        item,
        key,
        ensure_ascii,
    };
    let mut serializer = serde_json::Serializer::with_formatter(&mut json, formatter);
    if sort_keys {
        let mut value = serde_json::to_value(value)?;
        value.sort_all_objects();
        value.serialize(&mut serializer)?;
    } else {
        value.serialize(&mut serializer)?;
    }
    Ok(String::from_utf8(json).expect("serde_json writes UTF-8"))
}
/// Transformers' `tojson`.
fn tojson(value: &TemplateValue, options: Kwargs) -> Result<String, minijinja::Error> {
    let separators: Option<Vec<String>> = options.get("separators")?;
    let sort_keys: Option<bool> = options.get("sort_keys")?;
    let ensure_ascii: Option<bool> = options.get("ensure_ascii")?;
    let indent: Option<TemplateValue> = options.get("indent")?;
    options.assert_all_used()?;
    // Python refuses an undefined value at any depth; serializing one would print null.
    if holds(value, &[ValueKind::Undefined]) {
        return Err(minijinja::Error::new(
            minijinja::ErrorKind::UndefinedError,
            "tojson cannot serialize an undefined value",
        ));
    }
    if indent.is_some() {
        return Err(minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "tojson does not support indent",
        ));
    }
    let separators = match separators.as_deref() {
        None => (", ", ": "),
        Some([item, key]) => (item.as_str(), key.as_str()),
        Some(_) => {
            return Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                "separators takes an item and a key separator",
            ))
        }
    };
    python_json(
        value,
        separators,
        sort_keys == Some(true),
        ensure_ascii == Some(true),
    )
    .map_err(|error| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "cannot serialize to JSON",
        )
        .with_source(error)
    })
}
/// Transformers' `generation` block marks assistant text for training masks and renders its
/// body unchanged. minijinja has no such tag, so the block becomes an always-true `if`.
fn without_generation_blocks(template: &str) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{%") {
        let Some(len) = rest[start + 2..].find("%}") else {
            break;
        };
        let tag = &rest[start + 2..start + 2 + len];
        let word = tag.trim_matches(|c: char| c == '-' || c == '+' || c.is_whitespace());
        out.push_str(&rest[..start + 2]);
        match word {
            "generation" => out.push_str(&tag.replace(word, "if true")),
            "endgeneration" => out.push_str(&tag.replace(word, "endif")),
            _ => out.push_str(tag),
        }
        rest = &rest[start + 2 + len..];
    }
    out + rest
}
fn read_optional_json(path: &Path) -> Result<Value, TokenizerError> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Value::Null),
        Err(error) => Err(error.into()),
    }
}
fn token_content(value: &Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("content").and_then(Value::as_str))
}
fn parse_eos(value: &Value) -> Result<Vec<TokenId>, TokenizerError> {
    let values = match value {
        Value::Array(values) => values.as_slice(),
        Value::Null => return Ok(Vec::new()),
        _ => std::slice::from_ref(value),
    };
    let mut ids = values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .map(TokenId)
                .ok_or_else(|| TokenizerError::InvalidEos(value.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    ids.sort_unstable_by_key(|id| id.0);
    ids.dedup();
    Ok(ids)
}
/// Transformers drops a message's null fields and exposes its reasoning as both
/// `reasoning_content` and `thinking`, whichever name it arrived under.
fn transformers_message(message: TemplateValue) -> Result<TemplateValue, minijinja::Error> {
    if message.kind() != ValueKind::Map {
        return Ok(message);
    }
    let mut fields = Vec::new();
    for key in message.try_iter()? {
        let value = message.get_item(&key)?;
        if !value.is_none() {
            fields.push((key, value));
        }
    }
    let reasoning = ["reasoning_content", "thinking", "reasoning"]
        .into_iter()
        .find_map(|name| {
            let (_, value) = fields.iter().find(|(key, _)| key.as_str() == Some(name))?;
            Some(value.clone())
        });
    let Some(reasoning) = reasoning else {
        return Ok(TemplateValue::from_pairs(fields));
    };
    Ok(TemplateValue::from_pairs(
        [("thinking".into(), reasoning.clone())]
            .into_iter()
            .chain(fields)
            .chain([("reasoning_content".into(), reasoning)]),
    ))
}
fn render_chat<M: Serialize>(
    env: &Environment<'_>,
    special_tokens: &serde_json::Map<String, Value>,
    messages: &[M],
    options: ChatTemplateOptions,
) -> Result<String, ChatTemplateError> {
    const CONTINUE_TAG: &str = "CONTINUE_FINAL_MESSAGE_TAG ";
    let mut messages = messages
        .iter()
        .map(|message| transformers_message(Serde(message).into()))
        .collect::<Result<Vec<_>, _>>()?;
    let final_message = if matches!(options.continuation, ChatContinuation::ContinueLast) {
        let last = messages
            .last_mut()
            .ok_or(ChatTemplateError::IncompatibleContinuation)?;
        let content = last.get_attr("content")?;
        let content = content
            .as_str()
            .ok_or(ChatTemplateError::IncompatibleContinuation)?;
        let tagged = TemplateValue::from_pairs(last.try_iter()?.map(|key| {
            let value = if key.as_str() == Some("content") {
                format!("{content}{CONTINUE_TAG}").into()
            } else {
                last.get_item(&key).unwrap_or_default()
            };
            (key, value)
        }));
        *last = tagged;
        Some(content.to_owned())
    } else {
        None
    };
    let mut variables: BTreeMap<_, _> = special_tokens
        .iter()
        .map(|(name, value)| (name.clone(), TemplateValue::from(Serde(value))))
        .collect();
    variables.extend(options.variables);
    if let Some(enable_thinking) = options.enable_thinking {
        variables.insert("enable_thinking".into(), enable_thinking.into());
    }
    let mut rendered = env.get_template("chat")?.render(context! {
        messages,
        tools => (!options.tools.is_empty()).then_some(options.tools),
        documents => Option::<bool>::None,
        add_generation_prompt => matches!(options.continuation, ChatContinuation::StartAssistant),
        ..TemplateValue::from_pairs(variables)
    })?;
    if let Some(message) = &final_message {
        if !rendered.contains(message.trim()) {
            return Err(ChatTemplateError::FinalMessageNotFound);
        }
        // Transformers 5 uses a marker so repeated message text cannot select an earlier turn.
        let start = rendered
            .rfind(CONTINUE_TAG.trim_end())
            .ok_or(ChatTemplateError::FinalMessageNotFound)?;
        let preserves_spacing = rendered
            .get(start..)
            .is_some_and(|tail| tail.starts_with(CONTINUE_TAG));
        rendered.truncate(start);
        if !preserves_spacing {
            rendered.truncate(rendered.trim_end().len());
        }
    }
    Ok(rendered)
}

/// DecodeStream delays incomplete UTF-8. Retaining IDs permits one final decode to flush it.
pub(crate) struct StreamingDecoder<'a> {
    tokenizer: &'a Tokenizer,
    inner: tokenizers::DecodeStream<
        'a,
        tokenizers::ModelWrapper,
        tokenizers::NormalizerWrapper,
        tokenizers::PreTokenizerWrapper,
        tokenizers::PostProcessorWrapper,
        tokenizers::DecoderWrapper,
    >,
    ids: Vec<TokenId>,
    emitted: String,
}

impl StreamingDecoder<'_> {
    pub(crate) fn step(&mut self, id: TokenId) -> Result<Option<String>, TokenizerError> {
        let delta = self.inner.step(id.0)?;
        self.ids.push(id);
        if let Some(text) = &delta {
            self.emitted.push_str(text);
        }
        Ok(delta)
    }

    /// Consuming the stream prevents duplicate final flushes. Special tokens are preserved,
    /// matching `decode` and Python’s streaming detokenizer.
    pub(crate) fn finish(self) -> Result<String, TokenizerError> {
        let decoded = self.tokenizer.decode(&self.ids)?;
        decoded
            .strip_prefix(&self.emitted)
            .map(str::to_owned)
            .ok_or_else(|| {
                TokenizerError::Tokenizer(Box::new(
                    tokenizers::tokenizer::DecodeStreamError::InvalidPrefix {
                        token_id: self.ids.last().map_or(0, |id| id.0),
                        expected_prefix: self.emitted,
                        actual_string: decoded,
                    },
                ))
            })
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod text_tests;
