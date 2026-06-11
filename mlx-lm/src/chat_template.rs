//! Jinja chat-template rendering.
//!
//! Mirrors HF `tokenizer.apply_chat_template`: load
//! `chat_template.jinja` (preferred) or
//! `tokenizer_config.json::chat_template` (fallback), then render
//! `ChatMessage`s through it. `MessageContent` covers both the
//! plain-string form (llama-family) and the parts-list form
//! (multimodal).

use std::collections::HashMap;
use std::path::Path;

use minijinja::{context, Environment, Value};
use serde::{Deserialize, Serialize};

use crate::error::Error;

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `"system"`, `"user"`, `"assistant"`, or `"tool"`.
    pub role: String,
    /// String or list-of-parts content.
    pub content: MessageContent,
}

impl ChatMessage {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: MessageContent::Text(text.into()),
        }
    }

    /// User message: `[image, text]`. Vision models splice the image
    /// into the image-pad slot at runtime.
    pub fn user_with_image(text: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart::Image,
                ContentPart::Text { text: text.into() },
            ]),
        }
    }

    /// User message: `[audio, text]`. Audio models splice the clip into
    /// the audio-pad slot at runtime.
    pub fn user_with_audio(text: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart::Audio,
                ContentPart::Text { text: text.into() },
            ]),
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: MessageContent::Text(text.into()),
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: MessageContent::Text(text.into()),
        }
    }
}

/// Plain string (llama-family) or typed-parts list (multimodal).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// One element of a parts-list message content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentPart {
    Text {
        text: String,
    },
    /// Image placeholder; template emits image-pad token(s).
    Image,
    /// Audio placeholder; template emits audio-pad token(s).
    Audio,
}

/// A parsed chat template, ready to render messages.
pub struct ChatTemplate {
    source: String,
}

impl ChatTemplate {
    /// Load from a checkpoint dir: `chat_template.jinja` first, then
    /// `tokenizer_config.json::chat_template`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let jinja = dir.join("chat_template.jinja");
        if jinja.exists() {
            let source = std::fs::read_to_string(&jinja)?;
            return Ok(Self { source });
        }
        let tokcfg_path = dir.join("tokenizer_config.json");
        let raw = std::fs::read_to_string(&tokcfg_path)?;
        let parsed: serde_json::Value = serde_json::from_str(&raw)?;
        let source = parsed
            .get("chat_template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                Error::Other(
                    format!("no chat_template at {} or {tokcfg_path:?}", jinja.display()).into(),
                )
            })?
            .to_owned();
        Ok(Self { source })
    }

    /// Build from a raw template string (tests / non-standard sources).
    pub fn from_source(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
        }
    }

    /// Render `messages`. `add_generation_prompt` appends the assistant
    /// turn-start for inference; `kwargs` are extra named template
    /// values (e.g. `enable_thinking`).
    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        kwargs: &HashMap<String, serde_json::Value>,
    ) -> Result<String, Error> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template("chat", &self.source)
            .map_err(|e| Error::Other(format!("compiling chat template: {e}").into()))?;
        let tmpl = env
            .get_template("chat")
            .map_err(|e| Error::Other(format!("loading chat template: {e}").into()))?;
        let messages_value = Value::from_serialize(messages);
        let kwargs_value = Value::from_serialize(kwargs);
        let ctx = context! {
            messages => messages_value,
            add_generation_prompt => add_generation_prompt,
            ..kwargs_value
        };
        tmpl.render(ctx)
            .map_err(|e| Error::Other(format!("rendering chat template: {e}").into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY_TMPL: &str = "\
        {% for m in messages %}\
        {{ m.role }}={% if m.content is string %}{{ m.content }}\
        {% else %}\
        {% for p in m.content %}<{{ p.type }}>{% if p.type == 'text' %}{{ p.text }}{% endif %}{% endfor %}\
        {% endif %}|\
        {% endfor %}\
        {% if add_generation_prompt %}assistant={% endif %}";

    fn empty_kwargs() -> HashMap<String, serde_json::Value> {
        HashMap::new()
    }

    #[test]
    fn renders_plain_user_message() {
        let tmpl = ChatTemplate::from_source(TINY_TMPL);
        let out = tmpl
            .render(&[ChatMessage::user("Hello")], true, &empty_kwargs())
            .unwrap();
        assert!(out.contains("user=Hello"), "got: {out}");
        assert!(out.ends_with("assistant="), "got: {out}");
    }

    #[test]
    fn renders_parts_list_user_message() {
        let tmpl = ChatTemplate::from_source(TINY_TMPL);
        let out = tmpl
            .render(
                &[ChatMessage::user_with_image("What is this?")],
                true,
                &empty_kwargs(),
            )
            .unwrap();
        assert!(
            out.contains("user=<image><text>What is this?"),
            "got: {out}"
        );
    }

    #[test]
    fn template_kwargs_are_exposed_to_jinja() {
        let src = "\
            {% if enable_thinking is defined and enable_thinking %}thinking-on\
            {% elif enable_thinking is defined %}thinking-off\
            {% else %}thinking-unset{% endif %}";
        let tmpl = ChatTemplate::from_source(src);

        let on = tmpl
            .render(
                &[],
                false,
                &HashMap::from([("enable_thinking".to_owned(), serde_json::Value::Bool(true))]),
            )
            .unwrap();
        assert_eq!(on, "thinking-on");

        let unset = tmpl.render(&[], false, &HashMap::new()).unwrap();
        assert_eq!(unset, "thinking-unset");
    }
}
