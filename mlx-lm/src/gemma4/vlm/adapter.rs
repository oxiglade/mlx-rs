//! Gemma 4 multimodal [`LanguageModel`] + [`UserInputProcessor`].
//!
//! `prepare()` embeds the (image/audio-token-masked) ids, runs each attached
//! image through the vision tower and each audio clip through the audio tower,
//! stitches their projected features into the matching token slots, and decodes
//! via `Model::forward_embeds`. `step()` is the plain 1-D-rope text path; a turn
//! with no modality defers to `Model::forward`.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    module::Module,
    ops::{concatenate_axis, indexing::IndexOp, r#where},
    Array,
};

use crate::cache::{effective_prefill_chunk_opt, CacheOptions};
use crate::chat_template::{ChatMessage, ChatTemplate, ContentPart, MessageContent};
use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::family::{EosSpec, LoadedContext};
#[cfg(feature = "audio")]
use crate::gemma4::audio::encoder::{AudioEncoder, EmbedAudio};
#[cfg(feature = "audio")]
use crate::gemma4::audio::feature::{log_mel, num_audio_tokens};
#[cfg(feature = "audio")]
use crate::gemma4::audio::multimodal::stitch_audio_features;
use crate::gemma4::image::multimodal::stitch_image_features;
use crate::gemma4::image::processor::Gemma4ImageProcessor;
use crate::gemma4::image::vision::{EmbedVision, PatchGrid, VisionModel};
use crate::gemma4::text::cache::{make_caches, LayerCache};
use crate::gemma4::text::config::{ModelConfig, TextConfig};
use crate::gemma4::text::text::Model;
use crate::gemma4::vlm::weights::{load_full_model, LoadedTowers};
use crate::language_model::{LanguageModel, UserInputProcessor};
#[cfg(feature = "audio")]
use crate::lm_input::ProcessedAudio;
use crate::lm_input::{LMInput, LMOutput, PrepareResult, ProcessedImage, Text};
use crate::loader::{load_tokenizer, resolve_bos_id};
use crate::nn::ModelInput;
#[cfg(feature = "audio")]
use crate::user_input::Audio;
use crate::user_input::{Image, Prompt, UserInput};

/// Image placeholder the gemma chat template emits per `ContentPart::Image`.
const IMAGE_MARKER: &str = "<|image|>";

pub(crate) struct Gemma4VlmAdapter {
    model: Model,
    vision: VisionModel,
    embed_vision: EmbedVision,
    #[cfg(feature = "audio")]
    audio: Option<(AudioEncoder, EmbedAudio)>,
    cache: Vec<Option<LayerCache>>,
    args: TextConfig,
    image_token_id: u32,
    #[cfg(feature = "audio")]
    audio_token_id: u32,
    cache_options: CacheOptions,
    vocab_size: i32,
}

impl Gemma4VlmAdapter {
    fn new(towers: LoadedTowers, env: &ModelConfig) -> Self {
        let LoadedTowers {
            text: model,
            vision,
            embed_vision,
            #[cfg(feature = "audio")]
            audio,
        } = towers;
        let args = model.args.clone();
        let vocab_size = args.vocab_size;
        let cache_options = CacheOptions::default();
        let cache = make_caches(&args, cache_options);
        Self {
            model,
            vision,
            embed_vision,
            #[cfg(feature = "audio")]
            audio,
            cache,
            args,
            image_token_id: env.image_token_id,
            #[cfg(feature = "audio")]
            audio_token_id: env.audio_token_id,
            cache_options,
            vocab_size,
        }
    }

    /// Run the tower + projector over every image and concatenate the soft
    /// tokens along axis 0 (`[total_soft_tokens, text_hidden]`).
    fn encode_images(&mut self, pixels: &Array, grids: &[[i32; 3]]) -> Result<Array, Error> {
        let mut feats: Vec<Array> = Vec::with_capacity(grids.len());
        for (i, &[_, ph, pw]) in grids.iter().enumerate() {
            let i = i as i32;
            let img = pixels.index((i..i + 1, .., .., ..));
            let out = self.vision.forward(&img, PatchGrid::new(ph, pw))?;
            let projected = self.embed_vision.forward(&out)?;
            let shape = projected.shape();
            feats.push(projected.reshape(&[shape[1], shape[2]])?);
        }
        if feats.len() == 1 {
            return Ok(feats.into_iter().next().expect("len == 1"));
        }
        Ok(concatenate_axis(&feats, 0)?)
    }

    /// Run the audio tower + projector over `[1, T, 128]` log-mel → soft tokens
    /// `[T', text_hidden]`.
    #[cfg(feature = "audio")]
    fn encode_audio(&mut self, mel: &Array) -> Result<Array, Error> {
        let (enc, embed) = self
            .audio
            .as_mut()
            .ok_or_else(|| Error::config("gemma4 audio: checkpoint has no audio tower"))?;
        let features = enc.forward(mel)?;
        let projected = embed.forward(&features)?;
        let shape = projected.shape();
        Ok(projected.reshape(&[shape[1], shape[2]])?)
    }
}

impl LanguageModel for Gemma4VlmAdapter {
    fn reset(&mut self) {
        self.cache = make_caches(&self.args, self.cache_options);
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        let image = input.image;
        #[cfg(feature = "audio")]
        let audio = input.audio;
        #[cfg(not(feature = "audio"))]
        let audio: Option<()> = None;

        // No modality attached → plain text decode.
        if image.is_none() && audio.is_none() {
            let logits = self.model.forward(ModelInput {
                inputs: &input.text.tokens,
                mask: None,
                cache: &mut self.cache,
            })?;
            return Ok(PrepareResult::Logits(logits.index((.., -1, ..))));
        }

        let input_ids = input.text.tokens;

        // Per-layer inputs index `embed_tokens_per_layer` with the ids, so the
        // image/audio token slots must map to a real id (0) — mirrors the
        // reference masking before `get_per_layer_inputs`.
        let zeros = Array::from_int(0).as_dtype(input_ids.dtype())?;
        let is_image = input_ids.eq(Array::from_int(self.image_token_id as i32))?;
        #[cfg(feature = "audio")]
        let is_special =
            is_image.logical_or(&input_ids.eq(Array::from_int(self.audio_token_id as i32))?)?;
        #[cfg(not(feature = "audio"))]
        let is_special = is_image;
        let masked_ids = r#where(&is_special, &zeros, &input_ids)?;

        let mut embeds = self.model.model.embed_scaled(&masked_ids)?;
        if let Some(image) = image {
            let features = self.encode_images(&image.pixels, image.grids.as_slice())?;
            embeds = stitch_image_features(&features, &embeds, &input_ids, self.image_token_id)?;
        }
        #[cfg(feature = "audio")]
        if let Some(audio) = audio {
            let features = self.encode_audio(&audio.features)?;
            embeds = stitch_audio_features(&features, &embeds, &input_ids, self.audio_token_id)?;
        }
        let logits = self
            .model
            .forward_embeds(embeds, &masked_ids, &mut self.cache)?;
        Ok(PrepareResult::Logits(logits.index((.., -1, ..))))
    }

    fn step(&mut self, last_token: &Array) -> Result<LMOutput, Error> {
        let inp = last_token.reshape(&[1, 1])?;
        let logits = self.model.forward(ModelInput {
            inputs: &inp,
            mask: None,
            cache: &mut self.cache,
        })?;
        Ok(LMOutput {
            logits: logits.index((.., -1, ..)),
        })
    }

    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }

    fn prefill_chunk_size(&self) -> Option<i32> {
        effective_prefill_chunk_opt(&self.cache, self.cache_options.max_prefill_chunk)
    }

    fn prefill_chunk(&mut self, tokens: &Array) -> Result<(), Error> {
        let _ = self.model.forward(ModelInput {
            inputs: tokens,
            mask: None,
            cache: &mut self.cache,
        })?;
        Ok(())
    }

    fn set_cache_options(&mut self, options: CacheOptions) -> Result<(), Error> {
        self.cache = make_caches(&self.args, options);
        self.cache_options = options;
        Ok(())
    }
}

/// Gemma 4 `UserInputProcessor`: render chat, preprocess images, expand each
/// `<|image|>` marker to `boi + image_token×N + eoi`, tokenize, assert the
/// image-token count matches the soft-token total.
pub(crate) struct Gemma4Processor {
    tokenizer: tokenizers::Tokenizer,
    chat_template: ChatTemplate,
    image_processor: Gemma4ImageProcessor,
    bos_id: Option<u32>,
    image_token_id: u32,
    boi_token_id: u32,
    eoi_token_id: u32,
    pooling_kernel_size: i32,
    patch_size: i32,
    #[cfg(feature = "audio")]
    audio_token_id: u32,
    #[cfg(feature = "audio")]
    boa_token_id: u32,
    #[cfg(feature = "audio")]
    eoa_token_id: u32,
}

/// Audio placeholder the gemma chat template emits per `ContentPart::Audio`.
#[cfg(feature = "audio")]
const AUDIO_MARKER: &str = "<|audio|>";

impl UserInputProcessor for Gemma4Processor {
    fn family(&self) -> &'static str {
        "gemma4"
    }

    fn prepare(&mut self, input: UserInput) -> Result<LMInput, Error> {
        // Preprocess every image; collect channel-first pixel planes + grids.
        let mut planes: Vec<Vec<f32>> = Vec::with_capacity(input.images.len());
        let mut grids: Vec<[i32; 3]> = Vec::with_capacity(input.images.len());
        let mut soft_tokens: Vec<i32> = Vec::with_capacity(input.images.len());
        let mut dims: Option<(i32, i32)> = None;
        for image in input.images {
            let processed = match image {
                Image::Decoded(img) => self.image_processor.preprocess_image(img)?,
                Image::Pixels { .. } => {
                    return Err(Error::config(
                        "gemma4 vlm: Image::Pixels bypass not supported; pass Image::Decoded",
                    ));
                }
            };
            if let Some((h, w)) = dims {
                if (h, w) != (processed.height, processed.width) {
                    return Err(Error::shape(
                        "gemma4 vlm: multiple images must resize to identical dims",
                    ));
                }
            } else {
                dims = Some((processed.height, processed.width));
            }
            soft_tokens.push(processed.num_soft_tokens(self.pooling_kernel_size));
            grids.push([
                1,
                processed.height / self.patch_size,
                processed.width / self.patch_size,
            ]);
            planes.push(processed.pixel_values);
        }

        // Preprocess audio clips → log-mel + soft-token counts. One clip per
        // turn is the supported path; multiple are concatenated only if equal
        // length (single-utterance OCR/transcribe).
        #[cfg(feature = "audio")]
        let (audio_mel, audio_soft_tokens) = self.preprocess_audio(&input.audio)?;
        #[cfg(not(feature = "audio"))]
        let audio_soft_tokens: Vec<i32> = Vec::new();

        let mut expanded = render_prompt(
            &self.chat_template,
            input.prompt,
            grids.len(),
            audio_soft_tokens.len(),
        )?;
        expanded = self.expand_image_markers(&expanded, &soft_tokens)?;
        #[cfg(feature = "audio")]
        {
            expanded = self.expand_audio_markers(&expanded, &audio_soft_tokens)?;
        }

        let enc = self
            .tokenizer
            .encode(expanded.as_str(), false)
            .map_err(|e| Error::Other(format!("tokenizer encode: {e}").into()))?;
        let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        if let Some(bos) = self.bos_id {
            if ids.first() != Some(&(bos as i32)) {
                ids.insert(0, bos as i32);
            }
        }

        count_match(&ids, self.image_token_id, &soft_tokens, "image")?;
        #[cfg(feature = "audio")]
        count_match(&ids, self.audio_token_id, &audio_soft_tokens, "audio")?;

        let s = ids.len() as i32;
        let tokens = Array::from_slice(&ids, &[1, s]);

        let image = if grids.is_empty() {
            None
        } else {
            let (h, w) = dims.expect("non-empty images set dims");
            let n = grids.len() as i32;
            let mut all = Vec::with_capacity(planes.iter().map(Vec::len).sum());
            for p in planes {
                all.extend(p);
            }
            let pixels = Array::from_slice(&all, &[n, 3, h, w]);
            Some(ProcessedImage {
                pixels,
                grids,
                position_ids: None,
            })
        };

        Ok(LMInput {
            text: Text { tokens, mask: None },
            image,
            #[cfg(feature = "audio")]
            audio: audio_mel.map(|features| ProcessedAudio { features }),
        })
    }

    fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| Error::Other(format!("tokenizer decode: {e}").into()))
    }
}

impl Gemma4Processor {
    /// Replace each `<|image|>` marker (left to right) with
    /// `boi + image_token×N_i + eoi`.
    fn expand_image_markers(&self, text: &str, soft_tokens: &[i32]) -> Result<String, Error> {
        let boi = self.token_str(self.boi_token_id)?;
        let img = self.token_str(self.image_token_id)?;
        let eoi = self.token_str(self.eoi_token_id)?;
        expand_markers(text, IMAGE_MARKER, &boi, &img, &eoi, soft_tokens, "image")
    }

    /// Replace each `<|audio|>` marker (left to right) with
    /// `boa + audio_token×N_i + eoa`.
    #[cfg(feature = "audio")]
    fn expand_audio_markers(&self, text: &str, soft_tokens: &[i32]) -> Result<String, Error> {
        let boa = self.token_str(self.boa_token_id)?;
        let aud = self.token_str(self.audio_token_id)?;
        let eoa = self.token_str(self.eoa_token_id)?;
        expand_markers(text, AUDIO_MARKER, &boa, &aud, &eoa, soft_tokens, "audio")
    }

    fn token_str(&self, id: u32) -> Result<String, Error> {
        self.tokenizer
            .id_to_token(id)
            .ok_or_else(|| Error::config(format!("gemma4 vlm: token id {id} has no string")))
    }

    /// Log-mel for the (single) audio clip + its soft-token count. One clip per
    /// turn; the encoder takes one `[1, T, 128]` tensor.
    #[cfg(feature = "audio")]
    fn preprocess_audio(&self, clips: &[Audio]) -> Result<(Option<Array>, Vec<i32>), Error> {
        match clips {
            [] => Ok((None, Vec::new())),
            [clip] => {
                let mel = log_mel(&clip.samples)?;
                let n = num_audio_tokens(clip.samples.len());
                Ok((Some(mel), vec![n]))
            }
            _ => Err(Error::config(
                "gemma4 vlm: multiple audio clips per turn not supported; pass one",
            )),
        }
    }
}

/// Replace each `marker` (left to right) with `begin + tok×N_i + end`. Errors
/// if the marker count doesn't match `counts.len()`.
fn expand_markers(
    text: &str,
    marker: &str,
    begin: &str,
    tok: &str,
    end: &str,
    counts: &[i32],
    kind: &str,
) -> Result<String, Error> {
    let parts: Vec<&str> = text.split(marker).collect();
    let markers = parts.len() - 1;
    if markers != counts.len() {
        return Err(Error::shape(format!(
            "gemma4 vlm: template emitted {markers} {kind} markers but {} {kind}(s) supplied",
            counts.len()
        )));
    }
    let mut out = String::with_capacity(text.len());
    for (i, seg) in parts.iter().enumerate() {
        out.push_str(seg);
        if i < markers {
            out.push_str(begin);
            for _ in 0..counts[i] {
                out.push_str(tok);
            }
            out.push_str(end);
        }
    }
    Ok(out)
}

/// Render the chat template with one `ContentPart::Image` per image then one
/// `ContentPart::Audio` per clip, followed by the text.
fn render_prompt(
    template: &ChatTemplate,
    prompt: Prompt,
    num_images: usize,
    num_audio: usize,
) -> Result<String, Error> {
    let kwargs: HashMap<String, serde_json::Value> = HashMap::new();
    match prompt {
        Prompt::Text(text) => {
            if num_images == 0 && num_audio == 0 {
                template.render(&[ChatMessage::user(text)], true, &kwargs)
            } else {
                let mut parts: Vec<ContentPart> = Vec::with_capacity(num_images + num_audio + 1);
                parts.extend((0..num_images).map(|_| ContentPart::Image));
                parts.extend((0..num_audio).map(|_| ContentPart::Audio));
                parts.push(ContentPart::Text { text });
                let msg = ChatMessage {
                    role: "user".into(),
                    content: MessageContent::Parts(parts),
                };
                template.render(&[msg], true, &kwargs)
            }
        }
        Prompt::Chat(messages) => template.render(&messages, true, &kwargs),
    }
}

/// Assert the rendered prompt holds exactly `expected.sum()` `token_id` slots.
fn count_match(ids: &[i32], token_id: u32, expected: &[i32], kind: &str) -> Result<(), Error> {
    let observed = ids.iter().filter(|&&t| (t as u32) == token_id).count() as i32;
    let want: i32 = expected.iter().sum();
    if observed != want {
        return Err(Error::shape(format!(
            "gemma4 vlm: prompt has {observed} {kind} tokens but {} {kind}(s) expand to {want}",
            expected.len()
        )));
    }
    Ok(())
}

pub(crate) fn load_context_vlm(
    cfg: &Config,
    env: &ModelConfig,
    dir: &Path,
) -> Result<LoadedContext, Error> {
    let vision_cfg = env
        .vision_config
        .as_ref()
        .ok_or_else(|| Error::config("gemma4 vlm: config has no vision_config"))?;
    let towers = load_full_model(cfg, env, vision_cfg, dir)?;

    let tokenizer = load_tokenizer(dir)?;
    let bos_id = resolve_bos_id(dir, &tokenizer);
    let chat_template = ChatTemplate::from_dir(dir)?;
    let image_processor = Gemma4ImageProcessor::from_dir(dir)?;
    let eos_ids = EosSpec::to_vec(env.eos_token_id.as_ref());

    let pooling_kernel_size = image_processor.config.pooling_kernel_size;
    let patch_size = image_processor.config.patch_size;
    let adapter = Gemma4VlmAdapter::new(towers, env);
    let processor = Gemma4Processor {
        tokenizer,
        chat_template,
        image_processor,
        bos_id,
        image_token_id: env.image_token_id,
        boi_token_id: env.boi_token_id,
        eoi_token_id: env.eoi_token_id,
        pooling_kernel_size,
        patch_size,
        #[cfg(feature = "audio")]
        audio_token_id: env.audio_token_id,
        #[cfg(feature = "audio")]
        boa_token_id: env.boa_token_id,
        #[cfg(feature = "audio")]
        eoa_token_id: env.eoa_token_id,
    };
    Ok((Box::new(adapter), Box::new(processor), eos_ids))
}
