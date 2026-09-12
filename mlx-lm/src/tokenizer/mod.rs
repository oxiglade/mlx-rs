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
    pub(crate) inner: tokenizers::Tokenizer,
    env: Environment<'static>,
    eos: Vec<TokenId>,
    special_tokens: serde_json::Map<String, Value>,
    has_template: bool,
}
impl Tokenizer {
    /// Loads tokenizer assets and resolves generation-config EOS precedence.
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let path = path.as_ref();
        let mut tokenizer = Self::from_file(path.join("tokenizer.json"))?;
        let tokenizer_config = read_optional_json(&path.join("tokenizer_config.json"))?;
        if let Some(template) = tokenizer_config
            .get("chat_template")
            .and_then(Value::as_str)
        {
            tokenizer
                .env
                .add_template_owned("chat", template.to_owned())
                .map_err(ChatTemplateError::from)?;
            tokenizer.has_template = true;
        }
        if let Some(config) = tokenizer_config.as_object() {
            for (key, value) in config {
                if key.ends_with("_token") {
                    let content = value
                        .as_str()
                        .or_else(|| value.get("content").and_then(Value::as_str));
                    if let Some(content) = content {
                        tokenizer.special_tokens.insert(key.clone(), content.into());
                    }
                }
            }
        }
        let config = read_optional_json(&path.join("config.json"))?;
        let generation_config = read_optional_json(&path.join("generation_config.json"))?;
        let eos = generation_config
            .get("eos_token_id")
            .or_else(|| config.get("eos_token_id"));
        if let Some(eos) = eos {
            tokenizer.eos = parse_eos(eos)?;
        } else if let Some(eos) = tokenizer
            .special_tokens
            .get("eos_token")
            .and_then(Value::as_str)
        {
            if let Some(id) = tokenizer.inner.token_to_id(eos) {
                tokenizer.eos.push(TokenId(id));
            }
        }
        Ok(tokenizer)
    }
    /// Loads a local tokenizer JSON file without network access.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        Self::from_bytes(&std::fs::read(path)?)
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
        Ok(self
            .inner
            .encode(text, false)?
            .get_ids()
            .iter()
            .copied()
            .map(TokenId)
            .collect())
    }
    /// Decodes IDs while omitting special tokens.
    pub fn decode(&self, ids: &[TokenId]) -> Result<String, TokenizerError> {
        let ids: Vec<_> = ids.iter().map(|id| id.0).collect();
        Ok(self.inner.decode(&ids, true)?)
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
}
fn environment() -> Environment<'static> {
    let mut env = Environment::new();
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
fn parse_eos(value: &Value) -> Result<Vec<TokenId>, TokenizerError> {
    let values = match value {
        Value::Array(values) => values.as_slice(),
        Value::Null => return Ok(Vec::new()),
        _ => std::slice::from_ref(value),
    };
    values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .map(TokenId)
                .ok_or_else(|| TokenizerError::InvalidEos(value.to_string()))
        })
        .collect()
}
fn render_chat(
    env: &Environment<'_>,
    special_tokens: &serde_json::Map<String, Value>,
    messages: &[Message],
    options: ChatTemplateOptions,
) -> Result<String, ChatTemplateError> {
    let final_message = if matches!(options.continuation, ChatContinuation::ContinueLast) {
        let message = messages
            .last()
            .ok_or(ChatTemplateError::IncompatibleContinuation)?;
        if message.content.trim().is_empty() {
            return Err(ChatTemplateError::IncompatibleContinuation);
        }
        Some(message.content.as_str())
    } else {
        None
    };
    let mut rendered = env.get_template("chat")?.render(context! {
        messages => messages,
        add_generation_prompt => matches!(options.continuation, ChatContinuation::StartAssistant),
        ..minijinja::Value::from_serialize(special_tokens)
    })?;
    if let Some(message) = final_message {
        let start = rendered
            .rfind(message.trim())
            .ok_or(ChatTemplateError::FinalMessageNotFound)?;
        let tail = rendered
            .get(start..)
            .ok_or(ChatTemplateError::FinalMessageNotFound)?;
        let length = if tail.starts_with(message.trim_start()) {
            message.trim_start().len()
        } else {
            message.trim().len()
        };
        rendered.truncate(start + length);
    }
    Ok(rendered)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qwen3_template_fixture() -> Result<(), Box<dyn std::error::Error>> {
        let config: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/qwen3/tokenizer_config.json"
        ))?;
        let template = config
            .get("chat_template")
            .and_then(Value::as_str)
            .ok_or(ChatTemplateError::MissingTemplate)?;
        let mut env = environment();
        env.add_template_owned("chat", template.to_owned())?;
        let messages = [Message {
            role: Role::User,
            content: "hello".into(),
        }];
        let rendered = render_chat(
            &env,
            &Default::default(),
            &messages,
            ChatTemplateOptions::default(),
        )?;
        assert_eq!(rendered, "<|im_start|>user\nhello<|im_end|>\n");
        Ok(())
    }
    #[test]
    fn continuation_handles_trimmed_unicode_and_typed_failures() -> Result<(), ChatTemplateError> {
        let mut env = environment();
        env.add_template("chat", "{{ messages[-1].content | trim }}!")?;
        let options = ChatTemplateOptions {
            continuation: ChatContinuation::ContinueLast,
        };
        let message = [Message {
            role: Role::Assistant,
            content: " café  ".into(),
        }];
        assert_eq!(
            render_chat(&env, &Default::default(), &message, options.clone())?,
            "café"
        );
        assert!(matches!(
            render_chat(&env, &Default::default(), &[], options.clone()),
            Err(ChatTemplateError::IncompatibleContinuation)
        ));
        env.add_template("chat", "removed")?;
        assert!(matches!(
            render_chat(&env, &Default::default(), &message, options),
            Err(ChatTemplateError::FinalMessageNotFound)
        ));
        Ok(())
    }
}
