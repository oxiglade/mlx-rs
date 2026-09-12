pub use crate::error::{ChatTemplateError, TokenizerError};
use minijinja::{context, Environment};
use serde::Serialize;
use serde_json::Value;
use std::path::Path;

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
}
impl Tokenizer {
    /// Loads tokenizer assets with generation-config, model-config, then tokenizer EOS precedence.
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let path = path.as_ref();
        let mut tokenizer = Self::load_assets(path)?;
        tokenizer.eos = tokenizer.resolve_eos(path)?.0;
        Ok(tokenizer)
    }

    /// The model loader supplies its resolved EOS set, including an explicitly empty set.
    #[allow(dead_code)] // The foundation loader is implemented in a separate tranche item.
    pub(crate) fn from_dir_with_eos(
        path: impl AsRef<Path>,
        eos: Vec<TokenId>,
    ) -> Result<Self, TokenizerError> {
        let mut tokenizer = Self::load_assets(path.as_ref())?;
        tokenizer.eos = eos;
        tokenizer.eos.sort_unstable_by_key(|id| id.0);
        tokenizer.eos.dedup();
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
        if let Some(template) = template {
            tokenizer
                .env
                .add_template_owned("chat", template.to_owned())
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
        Ok(Self {
            inner: tokenizers::Tokenizer::from_bytes(bytes)?,
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

    fn encode_with_special_tokens(
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
    #[allow(dead_code)] // The generation item consumes this seam in tranche 3.
    pub(crate) fn decode_stream(&self) -> StreamingDecoder<'_> {
        StreamingDecoder {
            tokenizer: self,
            inner: self.inner.decode_stream(false),
            ids: Vec::new(),
            emitted: String::new(),
        }
    }
    /// Renders the compiled model template with a typed continuation policy.
    pub fn render_chat(
        &self,
        messages: &[Message],
        options: ChatTemplateOptions,
    ) -> Result<String, ChatTemplateError> {
        if !self.has_template {
            return Err(ChatTemplateError::MissingTemplate);
        }
        render_chat(&self.env, &self.special_tokens, messages, options)
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
/// Options for rendering one textual conversation.
#[derive(Debug, Clone, Default)]
pub struct ChatTemplateOptions {
    /// How the final message should be continued.
    pub continuation: ChatContinuation,
    /// `None` leaves the template variable undefined; a value overrides its thinking default.
    pub enable_thinking: Option<bool>,
}
fn environment() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
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
    env
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
fn render_chat(
    env: &Environment<'_>,
    special_tokens: &serde_json::Map<String, Value>,
    messages: &[Message],
    options: ChatTemplateOptions,
) -> Result<String, ChatTemplateError> {
    const CONTINUE_TAG: &str = "CONTINUE_FINAL_MESSAGE_TAG ";
    let mut continued;
    let (messages, final_message) =
        if matches!(options.continuation, ChatContinuation::ContinueLast) {
            let final_message = &messages
                .last()
                .ok_or(ChatTemplateError::IncompatibleContinuation)?
                .content;
            continued = messages.to_vec();
            continued
                .last_mut()
                .ok_or(ChatTemplateError::IncompatibleContinuation)?
                .content
                .push_str(CONTINUE_TAG);
            (continued.as_slice(), Some(final_message.as_str()))
        } else {
            (messages, None)
        };
    let mut variables = special_tokens.clone();
    if let Some(enable_thinking) = options.enable_thinking {
        variables.insert("enable_thinking".into(), enable_thinking.into());
    }
    let mut rendered = env.get_template("chat")?.render(context! {
        messages => messages,
        tools => Option::<bool>::None,
        documents => Option::<bool>::None,
        add_generation_prompt => matches!(options.continuation, ChatContinuation::StartAssistant),
        ..minijinja::Value::from_serialize(variables)
    })?;
    if let Some(message) = final_message {
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
#[allow(dead_code)] // The generation item consumes this seam in tranche 3.
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

#[allow(dead_code)] // The generation item consumes this seam in tranche 3.
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
