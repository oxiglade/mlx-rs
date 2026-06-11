//! Caller-facing input to [`crate::generate`].
//!
//! One struct carries the prompt (text or chat) plus optional images
//! (gated on the `image` feature) and template kwargs.

use std::collections::HashMap;

#[cfg(feature = "image")]
use image::DynamicImage;
#[cfg(feature = "image")]
use mlx_rs::Array;

use crate::chat_template::ChatMessage;

/// Top-level user-facing input handed to [`crate::generate`].
pub struct UserInput {
    /// What the user said: plain text or structured chat.
    pub prompt: Prompt,

    /// Images attached to the conversation, in order. The chat-template
    /// image slots consume them in sequence. Gated on `image`.
    #[cfg(feature = "image")]
    pub images: Vec<Image>,

    /// Audio clips attached to the conversation, in order. Gated on `audio`.
    #[cfg(feature = "audio")]
    pub audio: Vec<Audio>,

    /// Named values forwarded to the chat-template render (e.g.
    /// `enable_thinking`). Empty by default.
    pub template_kwargs: HashMap<String, serde_json::Value>,
}

/// Conversation shape. `Text` is the one-shot fast path; `Chat` carries
/// structured history the model's Jinja template renders.
pub enum Prompt {
    Text(String),
    Chat(Vec<ChatMessage>),
}

impl UserInput {
    /// Plain-text prompt.
    pub fn text(prompt: impl Into<String>) -> Self {
        Self {
            prompt: Prompt::Text(prompt.into()),
            #[cfg(feature = "image")]
            images: Vec::new(),
            #[cfg(feature = "audio")]
            audio: Vec::new(),
            template_kwargs: HashMap::new(),
        }
    }

    /// Structured chat conversation.
    pub fn chat(messages: Vec<ChatMessage>) -> Self {
        Self {
            prompt: Prompt::Chat(messages),
            #[cfg(feature = "image")]
            images: Vec::new(),
            #[cfg(feature = "audio")]
            audio: Vec::new(),
            template_kwargs: HashMap::new(),
        }
    }

    /// Attach images, builder-style.
    #[cfg(feature = "image")]
    #[must_use]
    pub fn with_images(mut self, images: Vec<Image>) -> Self {
        self.images = images;
        self
    }

    /// Attach audio clips, builder-style.
    #[cfg(feature = "audio")]
    #[must_use]
    pub fn with_audio(mut self, audio: Vec<Audio>) -> Self {
        self.audio = audio;
        self
    }

    /// Set one template kwarg, builder-style.
    #[must_use]
    pub fn with_template_kwarg(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.template_kwargs.insert(key.into(), value);
        self
    }
}

/// One image attached to a [`UserInput`].
///
/// - [`Image::Decoded`]: a CPU-decoded image; the processor resizes,
///   normalises, and packs it into the tower's patch layout.
/// - [`Image::Pixels`]: an already-preprocessed pixel array + its
///   `[t, h, w]` grid; the processor validates geometry and feeds it
///   straight to the tower (skips CPU preprocessing).
#[cfg(feature = "image")]
pub enum Image {
    /// Raw decoded image. Processor resizes + normalises + packs.
    Decoded(DynamicImage),

    /// Already in the tower's pixel-array layout.
    Pixels {
        /// `[num_patches, feature_dim]` `f32` array.
        array: Array,
        /// `[t, h, w]` patch counts; product must equal `array.shape[0]`.
        grid: [i32; 3],
    },
}

/// One audio clip attached to a [`UserInput`]: 16 kHz mono `f32` samples.
/// The processor computes the log-mel features; the caller pre-resamples to
/// 16 kHz mono.
#[cfg(feature = "audio")]
pub struct Audio {
    /// Mono PCM samples at 16 kHz.
    pub samples: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_input_constructs() {
        let input = UserInput::text("hi");
        assert!(matches!(input.prompt, Prompt::Text(ref s) if s == "hi"));
    }

    #[test]
    fn chat_input_constructs() {
        let input = UserInput::chat(vec![
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi"),
        ]);
        let Prompt::Chat(ref msgs) = input.prompt else {
            panic!("expected Chat prompt");
        };
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[1].role, "assistant");
    }
}
