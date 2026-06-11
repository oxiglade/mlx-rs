//! Gemma 4 image preprocessing.
//!
//! Aspect-preserving resize so the patch grid fits the soft-token budget and
//! both dims are `pooling_kernel_size · patch_size`-divisible, then rescale to
//! `[0, 1]`, channels-first. The vision tower handles `[-1, 1]` normalization
//! and patchification, so the processor only resizes + rescales.

use std::path::Path;

use image::{imageops::FilterType, DynamicImage};
use serde::Deserialize;

use crate::error::Error;

/// `processor_config.json` envelope: the image knobs nest under
/// `image_processor`.
#[derive(Debug, Clone, Deserialize)]
struct ProcessorConfigFile {
    image_processor: ImageProcessorConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageProcessorConfig {
    #[serde(default = "default_patch_size")]
    pub patch_size: i32,
    #[serde(default = "default_pooling_kernel_size")]
    pub pooling_kernel_size: i32,
    #[serde(default = "default_max_soft_tokens")]
    pub max_soft_tokens: i32,
    #[serde(default = "default_rescale_factor")]
    pub rescale_factor: f32,
}

fn default_patch_size() -> i32 {
    16
}
fn default_pooling_kernel_size() -> i32 {
    3
}
fn default_max_soft_tokens() -> i32 {
    280
}
fn default_rescale_factor() -> f32 {
    1.0 / 255.0
}

/// A preprocessed image: channels-first `[3, H, W]` row-major `f32` pixels in
/// `[0, 1]`, plus the patch grid `(ph, pw)`.
#[derive(Debug)]
pub struct ProcessedGemmaImage {
    pub pixel_values: Vec<f32>,
    pub height: i32,
    pub width: i32,
    /// Patch grid: `(ph, pw) = (H/patch, W/patch)`.
    pub ph: i32,
    pub pw: i32,
}

impl ProcessedGemmaImage {
    /// Soft tokens this image expands to: `(ph/k)*(pw/k)`.
    pub fn num_soft_tokens(&self, pooling_kernel_size: i32) -> i32 {
        (self.ph / pooling_kernel_size) * (self.pw / pooling_kernel_size)
    }
}

#[derive(Debug, Clone)]
pub struct Gemma4ImageProcessor {
    pub config: ImageProcessorConfig,
}

impl Gemma4ImageProcessor {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let path = dir.as_ref().join("processor_config.json");
        let f = std::fs::File::open(&path)?;
        let parsed: ProcessorConfigFile = serde_json::from_reader(f)?;
        Ok(Self {
            config: parsed.image_processor,
        })
    }

    /// Largest `(h, w)` preserving aspect ratio that patchifies to at most
    /// `max_patches` patches with both dims divisible by `pooling·patch`.
    fn target_size(&self, h: i32, w: i32) -> (i32, i32) {
        let patch = self.config.patch_size;
        let k = self.config.pooling_kernel_size;
        let max_patches = self.config.max_soft_tokens * k * k;
        let side_mult = k * patch;
        let target_px = (max_patches * patch * patch) as f64;
        let factor = (target_px / (h as f64 * w as f64)).sqrt();
        let mut th = ((factor * h as f64 / side_mult as f64).floor() as i32) * side_mult;
        let mut tw = ((factor * w as f64 / side_mult as f64).floor() as i32) * side_mult;
        let max_side = (max_patches / (k * k)) * side_mult;
        if th == 0 {
            th = side_mult;
            tw = (((w / h) * side_mult).min(max_side)).max(side_mult);
        } else if tw == 0 {
            tw = side_mult;
            th = (((h / w) * side_mult).min(max_side)).max(side_mult);
        }
        (th, tw)
    }

    pub fn preprocess_image(&self, image: DynamicImage) -> Result<ProcessedGemmaImage, Error> {
        let rgb = image.to_rgb8();
        let (w, h) = rgb.dimensions();
        let (th, tw) = self.target_size(h as i32, w as i32);
        let resized = image::imageops::resize(&rgb, tw as u32, th as u32, FilterType::CatmullRom);

        // [H, W, 3] u8 → [3, H, W] f32 in [0, 1].
        let (rh, rw) = (th as usize, tw as usize);
        let mut pixels = vec![0f32; 3 * rh * rw];
        let scale = self.config.rescale_factor;
        for y in 0..rh {
            for x in 0..rw {
                let px = resized.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    pixels[c * rh * rw + y * rw + x] = px[c] as f32 * scale;
                }
            }
        }
        Ok(ProcessedGemmaImage {
            pixel_values: pixels,
            height: th,
            width: tw,
            ph: th / self.config.patch_size,
            pw: tw / self.config.patch_size,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;

    fn proc() -> Gemma4ImageProcessor {
        Gemma4ImageProcessor {
            config: ImageProcessorConfig {
                patch_size: 16,
                pooling_kernel_size: 3,
                max_soft_tokens: 280,
                rescale_factor: 1.0 / 255.0,
            },
        }
    }

    #[test]
    fn target_size_divisible_by_side_mult_and_within_budget() {
        let p = proc();
        for (h, w) in [(640, 480), (1024, 768), (200, 1000), (50, 50)] {
            let (th, tw) = p.target_size(h, w);
            assert_eq!(th % 48, 0, "h {th} not divisible by 48");
            assert_eq!(tw % 48, 0, "w {tw} not divisible by 48");
            let patches = (th / 16) * (tw / 16);
            assert!(patches <= 280 * 9, "{patches} patches exceeds budget");
            assert!(th > 0 && tw > 0);
        }
    }

    #[test]
    fn preprocess_produces_channel_first_rescaled() {
        let p = proc();
        let img =
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(96, 96, image::Rgb([255, 0, 0])));
        let out = p.preprocess_image(img).unwrap();
        assert_eq!(out.pixel_values.len() as i32, 3 * out.height * out.width);
        // Red channel ~1.0, green/blue ~0.0.
        let plane = (out.height * out.width) as usize;
        assert!((out.pixel_values[0] - 1.0).abs() < 1e-6);
        assert!(out.pixel_values[plane].abs() < 1e-6);
    }
}
