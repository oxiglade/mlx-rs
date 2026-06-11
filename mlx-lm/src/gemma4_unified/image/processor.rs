//! Gemma 4 Unified encoder-free image preprocessing.
//!
//! Aspect-preserving resize (dims divisible by `pooling·patch`, ≤ the
//! soft-token budget), rescale to `[0, 1]`, then extract `model_patch_size`
//! (= `patch·pooling`) pixel blocks in row-major H,W,C order — each block is
//! one merged "model patch" of dim `model_patch_size²·3`. Per block a 2D
//! `(x, y)` model-grid position id; the block list is padded to
//! `num_soft_tokens` with zero pixels and `-1` positions.
//!
//! The HWC-block extraction is exactly the HF `convert_image_to_patches` +
//! `patches_merge` result: both flatten to `(h, w, c)` row-major, so a merged
//! k×k group is the contiguous `(k·patch)²·3` pixel block (see the
//! `merge_matches_reference` test).

use std::path::Path;

use image::{imageops::FilterType, DynamicImage};
use serde::Deserialize;

use crate::error::Error;

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

/// Merged model patches `[num_soft_tokens, model_patch_size²·3]` (row-major
/// H,W,C per patch) + per-patch `(x, y)` position ids `[num_soft_tokens, 2]`
/// (`-1` for padding). `num_valid` is the count of real (non-padding) patches.
#[derive(Debug)]
pub struct ProcessedUnifiedImage {
    pub patches: Vec<f32>,
    pub positions: Vec<i32>,
    pub num_patches: i32,
    pub patch_dim: i32,
    pub num_valid: i32,
}

pub struct Gemma4UnifiedImageProcessor {
    config: ImageProcessorConfig,
}

impl Gemma4UnifiedImageProcessor {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let path = dir.as_ref().join("processor_config.json");
        let raw = std::fs::read_to_string(&path)?;
        let file: ProcessorConfigFile = serde_json::from_str(&raw)?;
        Ok(Self {
            config: file.image_processor,
        })
    }

    /// Largest `(h, w)` preserving aspect ratio that yields at most
    /// `max_soft_tokens` model patches, with both dims divisible by
    /// `pooling·patch` (= `model_patch_size`).
    fn target_size(&self, h: i32, w: i32) -> (i32, i32) {
        let mp = self.config.pooling_kernel_size * self.config.patch_size;
        let max_model_patches = self.config.max_soft_tokens;
        let target_px = (max_model_patches * mp * mp) as f64;
        let factor = (target_px / (h as f64 * w as f64)).sqrt();
        let mut th = ((factor * h as f64 / mp as f64).floor() as i32) * mp;
        let mut tw = ((factor * w as f64 / mp as f64).floor() as i32) * mp;
        let max_side = max_model_patches * mp;
        if th == 0 {
            th = mp;
            tw = (((w / h).max(1) * mp).min(max_side)).max(mp);
        } else if tw == 0 {
            tw = mp;
            th = (((h / w).max(1) * mp).min(max_side)).max(mp);
        }
        // Cap total model patches to the budget (aspect math can round over).
        while (th / mp) * (tw / mp) > max_model_patches {
            if th >= tw {
                th -= mp;
            } else {
                tw -= mp;
            }
        }
        (th.max(mp), tw.max(mp))
    }

    pub fn preprocess_image(&self, image: DynamicImage) -> Result<ProcessedUnifiedImage, Error> {
        let mp = (self.config.pooling_kernel_size * self.config.patch_size) as usize;
        let rgb = image.to_rgb8();
        let (w, h) = rgb.dimensions();
        let (th, tw) = self.target_size(h as i32, w as i32);
        let resized = image::imageops::resize(&rgb, tw as u32, th as u32, FilterType::CatmullRom);

        let (gh, gw) = (th as usize / mp, tw as usize / mp); // model-patch grid
        let num_valid = gh * gw;
        let num_soft = self.config.max_soft_tokens as usize;
        let patch_dim = mp * mp * 3;
        let scale = self.config.rescale_factor;

        let mut patches = vec![0f32; num_soft * patch_dim];
        let mut positions = vec![-1i32; num_soft * 2];
        for my in 0..gh {
            for mx in 0..gw {
                let pi = my * gw + mx; // row-major model-patch index
                let base = pi * patch_dim;
                // Flatten the mp×mp pixel block in H,W,C order.
                for py in 0..mp {
                    for px in 0..mp {
                        let pixel = resized.get_pixel((mx * mp + px) as u32, (my * mp + py) as u32);
                        let off = base + (py * mp + px) * 3;
                        patches[off] = pixel[0] as f32 * scale;
                        patches[off + 1] = pixel[1] as f32 * scale;
                        patches[off + 2] = pixel[2] as f32 * scale;
                    }
                }
                positions[pi * 2] = mx as i32; // x
                positions[pi * 2 + 1] = my as i32; // y
            }
        }

        Ok(ProcessedUnifiedImage {
            patches,
            positions,
            num_patches: num_soft as i32,
            patch_dim: patch_dim as i32,
            num_valid: num_valid as i32,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;

    fn proc() -> Gemma4UnifiedImageProcessor {
        Gemma4UnifiedImageProcessor {
            config: ImageProcessorConfig {
                patch_size: 16,
                pooling_kernel_size: 3,
                max_soft_tokens: 280,
                rescale_factor: 1.0 / 255.0,
            },
        }
    }

    #[test]
    fn target_size_divisible_and_within_budget() {
        let p = proc();
        for (h, w) in [(640, 480), (1024, 768), (200, 1000), (50, 50)] {
            let (th, tw) = p.target_size(h, w);
            assert_eq!(th % 48, 0, "h {th} not divisible by 48");
            assert_eq!(tw % 48, 0, "w {tw} not divisible by 48");
            assert!((th / 48) * (tw / 48) <= 280);
            assert!(th > 0 && tw > 0);
        }
    }

    #[test]
    fn pads_to_soft_token_budget_with_neg1_positions() {
        let p = proc();
        let img =
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(96, 96, image::Rgb([255, 0, 0])));
        let out = p.preprocess_image(img).unwrap();
        assert_eq!(out.num_patches, 280);
        assert_eq!(out.patch_dim, 48 * 48 * 3);
        assert_eq!(out.patches.len(), 280 * 48 * 48 * 3);
        // Valid patches carry red ~1.0; padding patches are zero with -1 pos.
        assert!((out.patches[0] - 1.0).abs() < 1e-6);
        assert!(out.patches[1].abs() < 1e-6);
        let last = (out.num_patches as usize - 1) * 2;
        assert_eq!(out.positions[last], -1);
        assert_eq!(out.positions[last + 1], -1);
        // First valid patch sits at model-grid (0, 0).
        assert_eq!(out.positions[0], 0);
        assert_eq!(out.positions[1], 0);
    }

    /// Cross-check the HWC-block simplification against the literal HF
    /// `convert_image_to_patches` + `patches_merge` index math on a tiny
    /// `(k·patch) = 4` image (`patch=2`, `k=2`): a merged patch must equal the
    /// contiguous `4×4×3` pixel block in H,W,C order, and its position must be
    /// `(mx, my)`.
    #[test]
    fn merge_matches_reference() {
        let (patch, k) = (2usize, 2usize);
        let mp = patch * k; // 4
        let (gh, gw) = (1usize, 1usize); // one model patch
        let (h, w) = (gh * mp, gw * mp); // 4×4 image
                                         // Distinct pixel values: pixel(y,x,c) = (y*w + x) * 3 + c.
        let mut img = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    img[(y * w + x) * 3 + c] = ((y * w + x) * 3 + c) as u8;
                }
            }
        }
        // Reference: HWC block flatten of the single 4×4 model patch.
        let mut expected = vec![0u8; mp * mp * 3];
        for py in 0..mp {
            for px in 0..mp {
                for c in 0..3 {
                    expected[(py * mp + px) * 3 + c] = img[(py * w + px) * 3 + c];
                }
            }
        }
        // Our extraction (raw, no rescale) for model patch (0,0).
        let mut got = vec![0u8; mp * mp * 3];
        let (mx, my) = (0usize, 0usize);
        for py in 0..mp {
            for px in 0..mp {
                for c in 0..3 {
                    got[(py * mp + px) * 3 + c] =
                        img[((my * mp + py) * w + (mx * mp + px)) * 3 + c];
                }
            }
        }
        assert_eq!(got, expected);
        let _ = (patch, k, gh, gw);
    }
}
