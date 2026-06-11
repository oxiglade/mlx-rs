//! Qwen3.5 vision-language [`crate::LanguageModel`] + processor.
//!
//! The VLM path wraps the [`Qwen35DenseAdapter`] dense decoder, adds
//! the vision tower, and routes `prepare()` through a multimodal
//! prefill: pre-process images → vision tower → stitch features into
//! the prompt embedding sequence → call
//! [`Qwen35DenseAdapter::prefill_embeds`].
//!
//! [`Qwen35Processor`] is the matching `UserInputProcessor`: renders
//! the chat template (with image placeholders), expands
//! `<|image_pad|>` to one-per-merged-patch, tokenises, and
//! preprocesses every image (or accepts the [`crate::Image::Pixels`]
//! bypass).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{ops::concatenate_axis, Array};

use crate::cache::CacheOptions;
use crate::chat_template::{ChatMessage, ChatTemplate, ContentPart, MessageContent};
use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::family::LoadedContext;
use crate::language_model::{LanguageModel, UserInputProcessor};
use crate::lm_input::{LMInput, LMOutput, PrepareResult, ProcessedImage, Text};
use crate::qwen3_5::image::multimodal::{
    get_rope_index_single_batch, merge_input_ids_with_image_features, pack_position_ids,
};
use crate::qwen3_5::image::processor::{ProcessedImage as VlmRawImage, Qwen35ImageProcessor};
use crate::qwen3_5::image::vision::VisionModel;
use crate::qwen3_5::image::weights::load_full_model;
use crate::qwen3_5::text::adapter_dense::Qwen35DenseAdapter;
use crate::qwen3_5::text::config::ModelConfig;
use crate::qwen3_5::text::{leftover_keys_error, load_common};
use crate::user_input::{Image, Prompt, UserInput};

const IMAGE_PAD_STR: &str = "<|image_pad|>";

/// Adapter for qwen3_5 VLM checkpoints (Qwen3.5-VL, chandra). Owns the
/// text decoder + the vision tower.
pub(crate) struct Qwen35VlmAdapter {
    dense: Qwen35DenseAdapter,
    vision: VisionModel,
}

impl Qwen35VlmAdapter {
    pub(crate) fn new(dense: Qwen35DenseAdapter, vision: VisionModel) -> Self {
        Self { dense, vision }
    }
}

impl LanguageModel for Qwen35VlmAdapter {
    fn reset(&mut self) {
        self.dense.reset();
    }

    fn prepare(&mut self, input: LMInput) -> Result<PrepareResult, Error> {
        let Some(image) = input.image else {
            // Text-only request against a VLM checkpoint: defer to the
            // dense path (no `<|image_pad|>` in the prompt).
            return self.dense.prepare(LMInput {
                text: input.text,
                image: None,
                #[cfg(feature = "audio")]
                audio: None,
            });
        };

        // Vision tower runs over the per-image patches.
        let image_features = self.vision.forward(&image.pixels, image.grids.as_slice())?;

        // Embed the input ids, then splice image features into the
        // placeholder positions.
        let input_ids = input.text.tokens;
        let embeds = self.dense.model.embed_tokens(&input_ids)?;

        let cfg = &self.dense.cfg;
        let stitched = merge_input_ids_with_image_features(
            &image_features,
            &embeds,
            &input_ids,
            cfg.image_token_id,
            cfg.video_token_id,
        )?;

        // Build `[3, 1, S]` mrope position ids + the per-image rope_delta.
        // `get_rope_index_single_batch` works on host-side `&[i32]`.
        let s = input_ids.shape()[1];
        let host_ids: Vec<i32> = input_ids.reshape(&[s])?.as_slice::<i32>().to_vec();
        let merge = cfg
            .vision_config
            .as_ref()
            .ok_or_else(|| Error::config("vlm prepare: config has no vision_config"))?
            .spatial_merge_size;
        let (t_pos, h_pos, w_pos, rope_delta) = get_rope_index_single_batch(
            &host_ids,
            image.grids.as_slice(),
            merge,
            cfg.image_token_id,
            cfg.video_token_id,
            cfg.vision_start_token_id,
        )?;
        let position_ids = pack_position_ids(&t_pos, &h_pos, &w_pos)?;

        let logits = self
            .dense
            .prefill_embeds(stitched, position_ids, rope_delta)?;
        Ok(PrepareResult::Logits(logits))
    }

    fn step(&mut self, last_token: &Array) -> Result<LMOutput, Error> {
        self.dense.step(last_token)
    }

    fn vocab_size(&self) -> i32 {
        self.dense.vocab_size()
    }

    fn prefill_chunk_size(&self) -> Option<i32> {
        self.dense.prefill_chunk_size()
    }

    fn prefill_chunk(&mut self, tokens: &Array) -> Result<(), Error> {
        self.dense.prefill_chunk(tokens)
    }

    fn set_cache_options(&mut self, options: CacheOptions) -> Result<(), Error> {
        self.dense.set_cache_options(options)
    }
}

/// The qwen3_5 `UserInputProcessor`. Renders the chat template, expands
/// image-pad placeholders to the per-patch count, tokenises, and runs
/// image preprocessing (or validates the [`Image::Pixels`] bypass shape).
pub(crate) struct Qwen35Processor {
    tokenizer: tokenizers::Tokenizer,
    chat_template: ChatTemplate,
    image_processor: Qwen35ImageProcessor,
    cfg: ModelConfig,
}

impl Qwen35Processor {
    pub(crate) fn new(
        tokenizer: tokenizers::Tokenizer,
        chat_template: ChatTemplate,
        image_processor: Qwen35ImageProcessor,
        cfg: ModelConfig,
    ) -> Self {
        Self {
            tokenizer,
            chat_template,
            image_processor,
            cfg,
        }
    }
}

impl UserInputProcessor for Qwen35Processor {
    fn family(&self) -> &'static str {
        "qwen3_5"
    }

    fn prepare(&mut self, input: UserInput) -> Result<LMInput, Error> {
        let merge = self
            .cfg
            .vision_config
            .as_ref()
            .ok_or_else(|| {
                Error::config("vlm processor: config has no vision_config; not a VLM checkpoint")
            })?
            .spatial_merge_size;
        let mut grids: Vec<[i32; 3]> = Vec::with_capacity(input.images.len());
        let mut pixel_arrays: Vec<Array> = Vec::with_capacity(input.images.len());
        let mut expected_pad_total = 0_usize;

        for image in input.images {
            let (array, grid) = match image {
                Image::Decoded(img) => {
                    let processed = self.image_processor.preprocess_image(img)?;
                    pixels_array_from(processed)
                }
                Image::Pixels { array, grid } => {
                    validate_bypass_geometry(&array, grid, &self.image_processor)?;
                    (array, grid)
                }
            };
            let expected =
                (grid[0] as usize) * ((grid[1] / merge) as usize) * ((grid[2] / merge) as usize);
            expected_pad_total += expected;
            grids.push(grid);
            pixel_arrays.push(array);
        }

        // Render chat template. Images route to the parts-list form (one
        // ContentPart::Image per attached image then the text part);
        // plain text goes through verbatim.
        let prompt_text = render_prompt(
            &self.chat_template,
            input.prompt,
            grids.len(),
            &input.template_kwargs,
        )?;

        // Expand each single `<|image_pad|>` placeholder the template
        // emits into one-per-merged-patch (matches mlx-vlm's processor).
        let mut expanded = prompt_text;
        for grid in &grids {
            let expected =
                (grid[0] as usize) * ((grid[1] / merge) as usize) * ((grid[2] / merge) as usize);
            let replacement = IMAGE_PAD_STR.repeat(expected);
            expanded = expanded.replacen(IMAGE_PAD_STR, &replacement, 1);
        }

        let enc = self
            .tokenizer
            .encode(expanded.as_str(), false)
            .map_err(|e| Error::Other(format!("tokenizer encode: {e}").into()))?;
        let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();

        // Sanity-check the template + tokenise round-trip produced the
        // expected number of image-pad tokens.
        let observed = ids
            .iter()
            .filter(|&&t| (t as u32) == self.cfg.image_token_id)
            .count();
        if observed != expected_pad_total {
            return Err(Error::shape(format!(
                "qwen3_5 vlm: rendered prompt has {observed} image-pad tokens \
                 but {} image(s) expand to {expected_pad_total} merged patches",
                grids.len()
            )));
        }

        let s = ids.len() as i32;
        let tokens = Array::from_slice(&ids, &[1, s]);

        let image = if grids.is_empty() {
            None
        } else {
            // Concatenate per-image pixel arrays along the patch axis so
            // the vision tower sees one `[total_patches, feature_dim]`
            // input; `vision.forward` slices them back apart using `grids`.
            let pixels = concat_patches(pixel_arrays)?;
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
            audio: None,
        })
    }

    fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| Error::Other(format!("tokenizer decode: {e}").into()))
    }
}

/// Convert a CPU-side [`VlmRawImage`] into the `(array, grid)` pair the
/// LMInput pipeline consumes.
fn pixels_array_from(processed: VlmRawImage) -> (Array, [i32; 3]) {
    let VlmRawImage {
        pixel_values,
        grid_thw,
        feature_dim,
    } = processed;
    let num_patches = (pixel_values.len() / feature_dim as usize) as i32;
    let array = Array::from_slice(&pixel_values, &[num_patches, feature_dim]);
    (array, grid_thw)
}

/// Validate that a caller-supplied `Image::Pixels` array matches the
/// shape the processor would have produced for the same grid.
fn validate_bypass_geometry(
    array: &Array,
    grid: [i32; 3],
    processor: &Qwen35ImageProcessor,
) -> Result<(), Error> {
    let shape = array.shape();
    if shape.len() != 2 {
        return Err(Error::shape(format!(
            "Image::Pixels: array must be 2-D [num_patches, feature_dim], got {shape:?}"
        )));
    }
    let expected_patches = grid[0] * grid[1] * grid[2];
    if shape[0] != expected_patches {
        return Err(Error::shape(format!(
            "Image::Pixels: array.shape[0] = {} but grid t*h*w = {}*{}*{} = {}",
            shape[0], grid[0], grid[1], grid[2], expected_patches
        )));
    }
    let cfg = &processor.config;
    let expected_dim = cfg.patch_size * cfg.patch_size * cfg.temporal_patch_size * 3;
    if shape[1] != expected_dim {
        return Err(Error::shape(format!(
            "Image::Pixels: array.shape[1] = {} but processor expects feature_dim = {}",
            shape[1], expected_dim
        )));
    }
    Ok(())
}

/// Concatenate per-image patch tensors along axis 0. Consumes the vec so
/// the single-image fast path moves its one array out instead of cloning.
fn concat_patches(arrays: Vec<Array>) -> Result<Array, Error> {
    if arrays.len() == 1 {
        return Ok(arrays.into_iter().next().expect("len == 1 guard above"));
    }
    Ok(concatenate_axis(&arrays, 0)?)
}

/// Render the chat template against `prompt`. When `num_images > 0` the
/// prompt is wrapped in the parts-list form with one image placeholder
/// per attached image, matching the template's expectations.
fn render_prompt(
    template: &ChatTemplate,
    prompt: Prompt,
    num_images: usize,
    kwargs: &HashMap<String, serde_json::Value>,
) -> Result<String, Error> {
    match prompt {
        Prompt::Text(text) => {
            if num_images == 0 {
                let msg = ChatMessage::user(text);
                template.render(&[msg], true, kwargs)
            } else {
                let mut parts: Vec<ContentPart> =
                    (0..num_images).map(|_| ContentPart::Image).collect();
                parts.push(ContentPart::Text { text });
                let msg = ChatMessage {
                    role: "user".into(),
                    content: MessageContent::Parts(parts),
                };
                template.render(&[msg], true, kwargs)
            }
        }
        Prompt::Chat(mut messages) => {
            // Caller-built chat: trust the messages as-is unless images
            // were supplied with no image parts — then splice into the
            // last user message.
            if num_images > 0
                && messages.iter().all(|m| {
                    !matches!(&m.content, MessageContent::Parts(parts)
                    if parts.iter().any(|p| matches!(p, ContentPart::Image)))
                })
            {
                if let Some(last_user) = messages.iter_mut().rev().find(|m| m.role == "user") {
                    let existing_text = match std::mem::replace(
                        &mut last_user.content,
                        MessageContent::Text(String::new()),
                    ) {
                        MessageContent::Text(t) => t,
                        MessageContent::Parts(parts) => parts
                            .into_iter()
                            .filter_map(|p| match p {
                                ContentPart::Text { text } => Some(text),
                                ContentPart::Image | ContentPart::Audio => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    };
                    let mut new_parts: Vec<ContentPart> =
                        (0..num_images).map(|_| ContentPart::Image).collect();
                    new_parts.push(ContentPart::Text {
                        text: existing_text,
                    });
                    last_user.content = MessageContent::Parts(new_parts);
                }
            }
            template.render(&messages, true, kwargs)
        }
    }
}

/// Load a qwen3_5 VLM checkpoint at `dir`. Caller (the family-level
/// `qwen3_5::load_context`) guarantees the directory carries a vision
/// tower (`preprocessor_config.json` present).
pub(crate) fn load_context_vlm(
    cfg: &Config,
    env: &ModelConfig,
    dir: &Path,
) -> Result<LoadedContext, Error> {
    let (tokenizer, chat_template, eos_ids) = load_common(env, dir)?;
    let (model, vision, leftover) = load_full_model(cfg, env, dir)?;
    if !leftover.is_empty() {
        return Err(leftover_keys_error("vlm", &leftover));
    }
    let image_processor = Qwen35ImageProcessor::from_dir(dir)?;
    let dense = Qwen35DenseAdapter::new(model, env.clone())?;
    let vlm = Qwen35VlmAdapter::new(dense, vision);
    let processor = Qwen35Processor::new(tokenizer, chat_template, image_processor, env.clone());
    Ok((Box::new(vlm), Box::new(processor), eos_ids))
}
