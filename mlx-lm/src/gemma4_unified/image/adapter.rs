//! Gemma 4 Unified multimodal [`LanguageModel`] + [`UserInputProcessor`].
//!
//! `prepare()` embeds the (image-token-masked) ids, runs each image's merged
//! patches through the encoder-free [`VisionEmbedder`], strips padding patches,
//! stitches the projected soft tokens into the `image_token_id` slots, and
//! decodes via `Model::forward_embeds`. An optional MTP drafter composes:
//! image affects prefill only, so MTP decodes off the populated KV cache.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    module::Module,
    ops::{concatenate_axis, indexing::IndexOp},
    Array,
};

use crate::cache::{effective_prefill_chunk_opt, CacheOptions};
use crate::chat_template::{ChatMessage, ChatTemplate, ContentPart, MessageContent};
use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::family::{EosSpec, LoadedContext};
#[cfg(feature = "audio")]
use crate::gemma4::audio::multimodal::stitch_audio_features;
use crate::gemma4::image::multimodal::stitch_image_features;
use crate::gemma4::mtp::config::DrafterConfig;
use crate::gemma4::mtp::decode::{mtp_step, MtpContext};
use crate::gemma4::mtp::drafter::Drafter;
use crate::gemma4::mtp::weights::load_drafter;
use crate::gemma4::text::cache::{make_caches, LayerCache};
use crate::gemma4::text::config::TextConfig;
use crate::gemma4::text::text::Model;
#[cfg(feature = "audio")]
use crate::gemma4_unified::audio::embedder::AudioEmbedder;
#[cfg(feature = "audio")]
use crate::gemma4_unified::audio::feature::{frame_waveform, num_audio_tokens};
#[cfg(feature = "audio")]
use crate::gemma4_unified::audio::weights::load_audio_embedder;
use crate::gemma4_unified::config::ModelConfig;
use crate::gemma4_unified::image::embedder::VisionEmbedder;
use crate::gemma4_unified::image::processor::Gemma4UnifiedImageProcessor;
use crate::gemma4_unified::image::weights::{load_full_model, LoadedVlm};
use crate::language_model::{LanguageModel, UserInputProcessor};
#[cfg(feature = "audio")]
use crate::lm_input::ProcessedAudio;
use crate::lm_input::{LMInput, LMOutput, PrepareResult, ProcessedImage, Text};
use crate::loader::{load_tokenizer, resolve_bos_id};
use crate::nn::ModelInput;
use crate::sampler::SamplerState;
use crate::user_input::{Image, Prompt, UserInput};

/// Image placeholder the gemma chat template emits per `ContentPart::Image`.
const IMAGE_MARKER: &str = "<|image|>";

/// Upper bound on drafter depth γ.
const MAX_DRAFT_DEPTH: u32 = 8;

pub(crate) struct Gemma4UnifiedVlmAdapter {
    model: Model,
    embedder: VisionEmbedder,
    #[cfg(feature = "audio")]
    audio_embedder: Option<AudioEmbedder>,
    #[cfg(feature = "audio")]
    audio_token_id: u32,
    cache: Vec<Option<LayerCache>>,
    args: TextConfig,
    image_token_id: u32,
    cache_options: CacheOptions,
    vocab_size: i32,
    /// MTP drafter + state. Image affects prefill only; MTP then decodes off
    /// the populated cache, so the two compose. `None` without a drafter.
    drafter: Option<Drafter>,
    prev_hidden: Option<Array>,
    draft_depth: u32,
}

impl Gemma4UnifiedVlmAdapter {
    fn new(
        loaded: LoadedVlm,
        env: &ModelConfig,
        #[cfg(feature = "audio")] audio_embedder: Option<AudioEmbedder>,
        drafter: Option<Drafter>,
        draft_depth: u32,
    ) -> Self {
        let LoadedVlm {
            text: model,
            embedder,
        } = loaded;
        let args = model.args.clone();
        let vocab_size = args.vocab_size;
        let cache_options = CacheOptions::default();
        let cache = make_caches(&args, cache_options);
        Self {
            model,
            embedder,
            #[cfg(feature = "audio")]
            audio_embedder,
            #[cfg(feature = "audio")]
            audio_token_id: env.audio_token_id,
            cache,
            args,
            image_token_id: env.image_token_id,
            cache_options,
            vocab_size,
            drafter,
            prev_hidden: None,
            draft_depth,
        }
    }

    /// Run the audio embedder over raw frames `[1, T, samples_per_token]` →
    /// soft tokens `[T, text_hidden]`.
    #[cfg(feature = "audio")]
    fn encode_audio(&mut self, frames: &Array) -> Result<Array, Error> {
        let embedder = self
            .audio_embedder
            .as_mut()
            .ok_or_else(|| Error::config("gemma4_unified: checkpoint has no audio embedder"))?;
        let out = embedder.forward(frames)?;
        let shape = out.shape();
        Ok(out.reshape(&[shape[1], shape[2]])?)
    }

    /// Embed each image's merged patches, strip padding (position id `-1`),
    /// and concatenate the valid soft tokens along axis 0
    /// (`[total_valid, text_hidden]`).
    fn encode_images(&mut self, image: &ProcessedImage) -> Result<Array, Error> {
        let positions = image
            .position_ids
            .as_ref()
            .ok_or_else(|| Error::config("gemma4_unified vlm: image lacks position_ids"))?;
        let n = image.grids.len();
        let mut feats: Vec<Array> = Vec::with_capacity(n);
        for (i, &[_, valid, _]) in image.grids.iter().enumerate() {
            let i = i as i32;
            let pix = image.pixels.index((i..i + 1, .., ..)); // [1, num_soft, patch_dim]
            let pos = positions.index((i..i + 1, .., ..)); // [1, num_soft, 2]
            let out = self.embedder.forward(&pix, &pos)?; // [1, num_soft, hidden]
            let shape = out.shape();
            let flat = out.reshape(&[shape[1], shape[2]])?;
            // Keep only the leading `valid` patches (padding is trailing).
            feats.push(flat.index((0..valid, ..)));
        }
        if feats.len() == 1 {
            return Ok(feats.into_iter().next().expect("len == 1"));
        }
        Ok(concatenate_axis(&feats, 0)?)
    }
}

impl Gemma4UnifiedVlmAdapter {
    /// Plain text forward, advancing the cache; captures the last-position
    /// hidden as the next MTP anchor when a drafter is loaded.
    fn forward_text(&mut self, inputs: &Array) -> Result<Array, Error> {
        if self.drafter.is_some() {
            let (hidden, logits) = self
                .model
                .forward_hidden_and_logits(inputs, &mut self.cache)?;
            self.prev_hidden = Some(hidden.index((.., -1.., ..)));
            Ok(logits)
        } else {
            self.model.forward(ModelInput {
                inputs,
                mask: None,
                cache: &mut self.cache,
            })
        }
    }
}

impl LanguageModel for Gemma4UnifiedVlmAdapter {
    fn reset(&mut self) {
        self.cache = make_caches(&self.args, self.cache_options);
        self.prev_hidden = None;
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        let image = input.image;
        #[cfg(feature = "audio")]
        let audio = input.audio;
        #[cfg(not(feature = "audio"))]
        let audio: Option<()> = None;

        // No modality → plain text decode (MTP-aware).
        if image.is_none() && audio.is_none() {
            let logits = self.forward_text(&input.text.tokens)?;
            return Ok(PrepareResult::Logits(logits.index((.., -1, ..))));
        }

        let input_ids = input.text.tokens;
        let mut embeds = self.model.model.embed_scaled(&input_ids)?;
        if let Some(image) = image {
            let features = self.encode_images(&image)?;
            embeds = stitch_image_features(&features, &embeds, &input_ids, self.image_token_id)?;
        }
        #[cfg(feature = "audio")]
        if let Some(audio) = audio {
            let features = self.encode_audio(&audio.features)?;
            embeds = stitch_audio_features(&features, &embeds, &input_ids, self.audio_token_id)?;
        }
        // Modalities affect prefill only; capture the hidden so MTP can decode
        // off the populated cache.
        let logits = if self.drafter.is_some() {
            let (hidden, logits) =
                self.model
                    .forward_embeds_hidden_and_logits(embeds, &input_ids, &mut self.cache)?;
            self.prev_hidden = Some(hidden.index((.., -1.., ..)));
            logits
        } else {
            self.model
                .forward_embeds(embeds, &input_ids, &mut self.cache)?
        };
        Ok(PrepareResult::Logits(logits.index((.., -1, ..))))
    }

    fn step(&mut self, last_token: &Array) -> Result<LMOutput, Error> {
        let inp = last_token.reshape(&[1, 1])?;
        let logits = self.forward_text(&inp)?;
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
        let _ = self.forward_text(tokens)?;
        Ok(())
    }

    fn has_mtp(&self) -> bool {
        self.drafter.is_some()
    }

    fn set_mtp_depth(&mut self, n: u32) {
        self.draft_depth = n.clamp(1, MAX_DRAFT_DEPTH);
    }

    fn try_mtp_decode(
        &mut self,
        last_token: &Array,
        sampler: &mut SamplerState,
    ) -> Result<Option<(Vec<u32>, Array)>, Error> {
        let Some(drafter) = self.drafter.as_mut() else {
            return Ok(None);
        };
        let ctx = MtpContext {
            model: &mut self.model,
            cache: &mut self.cache,
            drafter,
            prev_hidden: &mut self.prev_hidden,
            depth: self.draft_depth,
            vocab_size: self.vocab_size,
        };
        mtp_step(ctx, last_token, sampler).map(Some)
    }

    fn set_cache_options(&mut self, options: CacheOptions) -> Result<(), Error> {
        self.cache = make_caches(&self.args, options);
        self.cache_options = options;
        Ok(())
    }
}

/// Gemma 4 Unified `UserInputProcessor`: render chat, preprocess images into
/// merged patches + position ids, expand each `<|image|>` marker to
/// `boi + image_token×valid + eoi`, tokenise, assert the image-token count.
pub(crate) struct Gemma4UnifiedProcessor {
    tokenizer: tokenizers::Tokenizer,
    chat_template: ChatTemplate,
    image_processor: Gemma4UnifiedImageProcessor,
    bos_id: Option<u32>,
    image_token_id: u32,
    boi_token_id: u32,
    eoi_token_id: u32,
    /// `Some` iff the checkpoint declares an `audio_config`.
    #[cfg(feature = "audio")]
    audio_samples_per_token: Option<i32>,
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

impl UserInputProcessor for Gemma4UnifiedProcessor {
    fn family(&self) -> &'static str {
        "gemma4_unified"
    }

    fn prepare(&mut self, input: UserInput) -> Result<LMInput, Error> {
        let mut patch_planes: Vec<Vec<f32>> = Vec::with_capacity(input.images.len());
        let mut pos_planes: Vec<Vec<i32>> = Vec::with_capacity(input.images.len());
        let mut grids: Vec<[i32; 3]> = Vec::with_capacity(input.images.len());
        let mut valid_counts: Vec<i32> = Vec::with_capacity(input.images.len());
        let mut shape: Option<(i32, i32)> = None;
        for image in input.images {
            let processed = match image {
                Image::Decoded(img) => self.image_processor.preprocess_image(img)?,
                Image::Pixels { .. } => {
                    return Err(Error::config(
                        "gemma4_unified vlm: Image::Pixels bypass unsupported; pass Image::Decoded",
                    ));
                }
            };
            let dims = (processed.num_patches, processed.patch_dim);
            match shape {
                Some(s) if s != dims => {
                    return Err(Error::shape(
                        "gemma4_unified vlm: images must share patch count + dim",
                    ));
                }
                _ => shape = Some(dims),
            }
            valid_counts.push(processed.num_valid);
            grids.push([1, processed.num_valid, 0]); // [_, valid, _] for the adapter
            patch_planes.push(processed.patches);
            pos_planes.push(processed.positions);
        }

        // Preprocess audio clips → raw frames + per-clip token counts.
        #[cfg(feature = "audio")]
        let (audio_frames, audio_counts) = self.preprocess_audio(&input.audio)?;
        #[cfg(not(feature = "audio"))]
        let audio_counts: Vec<i32> = Vec::new();

        let mut expanded = render_prompt(
            &self.chat_template,
            input.prompt,
            grids.len(),
            audio_counts.len(),
        )?;
        expanded = self.expand_image_markers(&expanded, &valid_counts)?;
        #[cfg(feature = "audio")]
        {
            expanded = self.expand_audio_markers(&expanded, &audio_counts)?;
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
        count_match(&ids, self.image_token_id, &valid_counts, "image")?;
        #[cfg(feature = "audio")]
        count_match(&ids, self.audio_token_id, &audio_counts, "audio")?;

        let s = ids.len() as i32;
        let tokens = Array::from_slice(&ids, &[1, s]);

        let image = if grids.is_empty() {
            None
        } else {
            let (num_patches, patch_dim) = shape.expect("non-empty images set shape");
            let n = grids.len() as i32;
            let mut patches = Vec::with_capacity(patch_planes.iter().map(Vec::len).sum());
            for p in patch_planes {
                patches.extend(p);
            }
            let mut positions = Vec::with_capacity(pos_planes.iter().map(Vec::len).sum());
            for p in pos_planes {
                positions.extend(p);
            }
            let pixels = Array::from_slice(&patches, &[n, num_patches, patch_dim]);
            let position_ids = Array::from_slice(&positions, &[n, num_patches, 2]);
            Some(ProcessedImage {
                pixels,
                grids,
                position_ids: Some(position_ids),
            })
        };

        #[cfg(feature = "audio")]
        let audio = audio_frames.map(|features| ProcessedAudio { features });

        Ok(LMInput {
            text: Text { tokens, mask: None },
            image,
            #[cfg(feature = "audio")]
            audio,
        })
    }

    fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| Error::Other(format!("tokenizer decode: {e}").into()))
    }
}

impl Gemma4UnifiedProcessor {
    /// Replace each `<|image|>` marker (left to right) with
    /// `boi + image_token×valid_i + eoi`.
    fn expand_image_markers(&self, text: &str, counts: &[i32]) -> Result<String, Error> {
        let boi = self.token_str(self.boi_token_id)?;
        let img = self.token_str(self.image_token_id)?;
        let eoi = self.token_str(self.eoi_token_id)?;
        let parts: Vec<&str> = text.split(IMAGE_MARKER).collect();
        let markers = parts.len() - 1;
        if markers != counts.len() {
            return Err(Error::shape(format!(
                "gemma4_unified vlm: template emitted {markers} image markers but {} image(s) supplied",
                counts.len()
            )));
        }
        let mut out = String::with_capacity(text.len());
        for (i, seg) in parts.iter().enumerate() {
            out.push_str(seg);
            if i < markers {
                out.push_str(&boi);
                for _ in 0..counts[i] {
                    out.push_str(&img);
                }
                out.push_str(&eoi);
            }
        }
        Ok(out)
    }

    fn token_str(&self, id: u32) -> Result<String, Error> {
        self.tokenizer.id_to_token(id).ok_or_else(|| {
            Error::config(format!("gemma4_unified vlm: token id {id} has no string"))
        })
    }

    /// Frame each clip into raw `[T, samples_per_token]` rows, stack along the
    /// clip axis (`[num_clips, T, spt]`), and return per-clip token counts.
    /// One clip per turn is the supported path; multiple require equal length.
    #[cfg(feature = "audio")]
    fn preprocess_audio(
        &self,
        clips: &[crate::user_input::Audio],
    ) -> Result<(Option<Array>, Vec<i32>), Error> {
        if clips.is_empty() {
            return Ok((None, Vec::new()));
        }
        let spt = self.audio_samples_per_token.ok_or_else(|| {
            Error::config("gemma4_unified: audio clip supplied but checkpoint has no audio_config")
        })?;
        let mut frames: Vec<f32> = Vec::new();
        let mut counts: Vec<i32> = Vec::with_capacity(clips.len());
        let mut tokens: Option<i32> = None;
        for clip in clips {
            let (clip_frames, n) = frame_waveform(&clip.samples, spt);
            match tokens {
                Some(t) if t != n => {
                    return Err(Error::shape(
                        "gemma4_unified audio: multiple clips must share length",
                    ));
                }
                _ => tokens = Some(n),
            }
            counts.push(num_audio_tokens(clip.samples.len(), spt));
            frames.extend(clip_frames);
        }
        let n_clips = clips.len() as i32;
        let t = tokens.expect("non-empty clips set token count");
        let arr = Array::from_slice(&frames, &[n_clips, t, spt]);
        Ok((Some(arr), counts))
    }

    /// Replace each `<|audio|>` marker (left to right) with
    /// `boa + audio_token×count_i + eoa`.
    #[cfg(feature = "audio")]
    fn expand_audio_markers(&self, text: &str, counts: &[i32]) -> Result<String, Error> {
        let boa = self.token_str(self.boa_token_id)?;
        let aud = self.token_str(self.audio_token_id)?;
        let eoa = self.token_str(self.eoa_token_id)?;
        let parts: Vec<&str> = text.split(AUDIO_MARKER).collect();
        let markers = parts.len() - 1;
        if markers != counts.len() {
            return Err(Error::shape(format!(
                "gemma4_unified audio: template emitted {markers} audio markers but {} clip(s) supplied",
                counts.len()
            )));
        }
        let mut out = String::with_capacity(text.len());
        for (i, seg) in parts.iter().enumerate() {
            out.push_str(seg);
            if i < markers {
                out.push_str(&boa);
                for _ in 0..counts[i] {
                    out.push_str(&aud);
                }
                out.push_str(&eoa);
            }
        }
        Ok(out)
    }
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
            "gemma4_unified vlm: prompt has {observed} {kind} tokens but {kind}(s) expand to {want}"
        )));
    }
    Ok(())
}

pub(crate) fn load_context_vlm(
    cfg: &Config,
    env: &ModelConfig,
    dir: &Path,
    draft_dir: Option<&Path>,
) -> Result<LoadedContext, Error> {
    let loaded = load_full_model(cfg, env, dir)?;
    let tokenizer = load_tokenizer(dir)?;
    let bos_id = resolve_bos_id(dir, &tokenizer);
    let chat_template = ChatTemplate::from_dir(dir)?;
    let image_processor = Gemma4UnifiedImageProcessor::from_dir(dir)?;
    let eos_ids = EosSpec::to_vec(env.eos_token_id.as_ref());

    // Optional MTP drafter: image affects prefill only, so MTP decode composes
    // with vision over the same KV cache.
    let (drafter, draft_depth) = match draft_dir {
        Some(d) => {
            let dcfg = DrafterConfig::from_dir(d)?;
            let depth = dcfg.default_depth();
            (Some(load_drafter(&dcfg, d)?), depth)
        }
        None => (None, 0),
    };
    // Optional encoder-free audio embedder.
    #[cfg(feature = "audio")]
    let audio_embedder = match env.audio_config.as_ref() {
        Some(acfg) => Some(load_audio_embedder(cfg, acfg, dir)?),
        None => None,
    };

    let adapter = Gemma4UnifiedVlmAdapter::new(
        loaded,
        env,
        #[cfg(feature = "audio")]
        audio_embedder,
        drafter,
        draft_depth,
    );
    let processor = Gemma4UnifiedProcessor {
        tokenizer,
        chat_template,
        image_processor,
        bos_id,
        image_token_id: env.image_token_id,
        boi_token_id: env.boi_token_id,
        eoi_token_id: env.eoi_token_id,
        #[cfg(feature = "audio")]
        audio_samples_per_token: env.audio_config.as_ref().map(|a| a.audio_samples_per_token),
        #[cfg(feature = "audio")]
        audio_token_id: env.audio_token_id,
        #[cfg(feature = "audio")]
        boa_token_id: env.boa_token_id,
        #[cfg(feature = "audio")]
        eoa_token_id: env.eoa_token_id,
    };
    Ok((Box::new(adapter), Box::new(processor), eos_ids))
}
