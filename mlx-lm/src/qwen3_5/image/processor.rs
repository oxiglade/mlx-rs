//! Pure-Rust port of `Qwen3VLImageProcessor` / `Qwen2VLImageProcessorFast`.
//!
//! Produces:
//!
//! - `pixel_values`: an `f32` tensor of shape
//!   `(grid_t * grid_h * grid_w, in_channels * temporal_patch_size *
//!   patch_size * patch_size)`. For single-image inputs `grid_t = 1` and the
//!   image is duplicated along the temporal axis so the model's
//!   `temporal_patch_size` is satisfied.
//! - `grid_thw`: a `[grid_t, grid_h, grid_w]` tuple.
//!
//! Resizing uses `image::imageops::FilterType::CatmullRom` (cubic — the same
//! filter `PIL.Image.BICUBIC` selects when downscaling). Normalisation uses
//! the model's `image_mean` / `image_std` after rescaling pixels by
//! `rescale_factor` (defaults `0.5, 0.5, 0.5` and `1/255`).

use std::path::Path;

use image::{imageops::FilterType, DynamicImage, ImageBuffer, Rgb};
use serde::Deserialize;

use crate::error::Error;

/// Number of colour channels we operate on.
pub const CHANNELS: usize = 3;

/// Reject images whose longest/shortest side ratio exceeds this — matches
/// HF `smart_resize`'s guard against degenerate aspect ratios.
const MAX_ASPECT_RATIO: f64 = 200.0;

/// Hyperparameters parsed from a checkpoint's `preprocessor_config.json`.
///
/// All defaults match `Qwen2VLImageProcessorFast` / `Qwen3VLImageProcessor`.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageProcessorConfig {
    #[serde(default = "default_patch_size")]
    pub patch_size: i32,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: i32,
    #[serde(default = "default_merge_size")]
    pub merge_size: i32,
    #[serde(default = "default_min_pixels")]
    pub min_pixels: i32,
    #[serde(default = "default_max_pixels")]
    pub max_pixels: i32,
    #[serde(default = "default_do_rescale")]
    pub do_rescale: bool,
    #[serde(default = "default_rescale_factor")]
    pub rescale_factor: f32,
    #[serde(default = "default_do_normalize")]
    pub do_normalize: bool,
    #[serde(default = "default_image_mean")]
    pub image_mean: [f32; 3],
    #[serde(default = "default_image_std")]
    pub image_std: [f32; 3],
}

fn default_patch_size() -> i32 {
    16
}
fn default_temporal_patch_size() -> i32 {
    2
}
fn default_merge_size() -> i32 {
    2
}
fn default_min_pixels() -> i32 {
    56 * 56
}
fn default_max_pixels() -> i32 {
    14 * 14 * 4 * 1280
}
fn default_do_rescale() -> bool {
    true
}
fn default_rescale_factor() -> f32 {
    1.0 / 255.0
}
fn default_do_normalize() -> bool {
    true
}
fn default_image_mean() -> [f32; 3] {
    [0.5, 0.5, 0.5]
}
fn default_image_std() -> [f32; 3] {
    [0.5, 0.5, 0.5]
}

impl Default for ImageProcessorConfig {
    fn default() -> Self {
        Self {
            patch_size: default_patch_size(),
            temporal_patch_size: default_temporal_patch_size(),
            merge_size: default_merge_size(),
            min_pixels: default_min_pixels(),
            max_pixels: default_max_pixels(),
            do_rescale: default_do_rescale(),
            rescale_factor: default_rescale_factor(),
            do_normalize: default_do_normalize(),
            image_mean: default_image_mean(),
            image_std: default_image_std(),
        }
    }
}

impl ImageProcessorConfig {
    /// Load `preprocessor_config.json` from a checkpoint directory.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let f = std::fs::File::open(path.as_ref())?;
        Ok(serde_json::from_reader(f)?)
    }
}

/// A single preprocessed image: patches + the `[grid_t, grid_h, grid_w]` tuple.
/// Host-side (`Vec<f32>`) form; the adapter uploads it to a device `Array`.
#[derive(Debug)]
pub struct ProcessedImage {
    /// Flat patch tensor of shape
    /// `(grid_t * grid_h * grid_w, in_channels * temporal_patch_size *
    /// patch_size * patch_size)`. Row-major.
    pub pixel_values: Vec<f32>,
    /// Number of patches along the time / vertical / horizontal axes.
    pub grid_thw: [i32; 3],
    /// Per-patch feature dim. Stored for convenience.
    pub feature_dim: i32,
}

impl ProcessedImage {
    /// Total number of patches `grid_t * grid_h * grid_w`.
    pub fn num_patches(&self) -> i32 {
        self.grid_thw[0] * self.grid_thw[1] * self.grid_thw[2]
    }
}

/// Compute the target `(h_bar, w_bar)` so both dims are multiples of `factor`
/// (`patch_size * merge_size`) and total area sits in `[min_pixels, max_pixels]`.
///
/// Mirrors HF's `smart_resize` from `qwen2_vl`.
pub fn smart_resize(
    height: i32,
    width: i32,
    factor: i32,
    min_pixels: i32,
    max_pixels: i32,
) -> Result<(i32, i32), Error> {
    let (h, w) = (height as f64, width as f64);
    let aspect = h.max(w) / h.min(w);
    if aspect > MAX_ASPECT_RATIO {
        return Err(Error::config(format!(
            "smart_resize: aspect ratio {aspect} exceeds {MAX_ASPECT_RATIO}"
        )));
    }

    let f = factor as f64;
    let mut h_bar = ((h / f).round() * f) as i32;
    let mut w_bar = ((w / f).round() * f) as i32;
    if h_bar < 1 {
        h_bar = factor;
    }
    if w_bar < 1 {
        w_bar = factor;
    }

    if (h_bar as i64) * (w_bar as i64) > max_pixels as i64 {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = (((h / beta) / f).floor() * f) as i32;
        w_bar = (((w / beta) / f).floor() * f) as i32;
        h_bar = h_bar.max(factor);
        w_bar = w_bar.max(factor);
    } else if (h_bar as i64) * (w_bar as i64) < min_pixels as i64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = (((h * beta) / f).ceil() * f) as i32;
        w_bar = (((w * beta) / f).ceil() * f) as i32;
    }

    Ok((h_bar, w_bar))
}

/// The image preprocessor.
#[derive(Debug, Clone)]
pub struct Qwen35ImageProcessor {
    pub config: ImageProcessorConfig,
}

impl Qwen35ImageProcessor {
    /// Create a processor with the given config.
    pub fn new(config: ImageProcessorConfig) -> Self {
        Self { config }
    }

    /// Load `preprocessor_config.json` from a checkpoint directory.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let path = dir.as_ref().join("preprocessor_config.json");
        Ok(Self::new(ImageProcessorConfig::from_file(path)?))
    }

    /// Preprocess a file on disk.
    pub fn preprocess_path(&self, path: impl AsRef<Path>) -> Result<ProcessedImage, Error> {
        let img = image::open(path.as_ref())
            .map_err(|e| Error::Other(format!("open image: {e}").into()))?;
        self.preprocess_image(img)
    }

    /// Preprocess a decoded image.
    pub fn preprocess_image(&self, image: DynamicImage) -> Result<ProcessedImage, Error> {
        let rgb = image.to_rgb8();
        let (w, h) = rgb.dimensions();
        let factor = self.config.patch_size * self.config.merge_size;
        let (target_h, target_w) = smart_resize(
            h as i32,
            w as i32,
            factor,
            self.config.min_pixels,
            self.config.max_pixels,
        )?;

        let resized = resize_rgb(rgb, target_w as u32, target_h as u32);
        let chw = hwc_u8_to_chw_f32(&resized);
        let img_f32 = self.rescale_and_normalize(chw);

        self.patchify(img_f32, target_h, target_w)
    }

    fn rescale_and_normalize(&self, mut img: Vec<f32>) -> Vec<f32> {
        if self.config.do_rescale {
            let s = self.config.rescale_factor;
            for x in &mut img {
                *x *= s;
            }
        }
        if self.config.do_normalize {
            // CHW layout
            let chan_len = img.len() / CHANNELS;
            for c in 0..CHANNELS {
                let mean = self.config.image_mean[c];
                let std = self.config.image_std[c];
                let slice = &mut img[c * chan_len..(c + 1) * chan_len];
                for x in slice.iter_mut() {
                    *x = (*x - mean) / std;
                }
            }
        }
        img
    }

    fn patchify(
        &self,
        img_chw: Vec<f32>,
        target_h: i32,
        target_w: i32,
    ) -> Result<ProcessedImage, Error> {
        let ps = self.config.patch_size as usize;
        let tps = self.config.temporal_patch_size as usize;
        let ms = self.config.merge_size as usize;
        let c = CHANNELS;
        let grid_t = 1_usize;
        let grid_h = (target_h as usize) / ps;
        let grid_w = (target_w as usize) / ps;
        if grid_h == 0 || grid_w == 0 {
            return Err(Error::shape(format!(
                "patchify: target {target_h}x{target_w} smaller than patch_size {ps}"
            )));
        }
        if !grid_h.is_multiple_of(ms) || !grid_w.is_multiple_of(ms) {
            return Err(Error::shape(format!(
                "patchify: grid {grid_h}x{grid_w} not divisible by merge_size {ms}"
            )));
        }

        // The temporal axis duplicates the single frame `tps` times so the
        // reshape works out — every patch ends up with the same `tps` copies
        // side-by-side along the inner dim. We bake that duplication directly
        // into the inner loop instead of materialising the
        // repeat-then-reshape-then-transpose chain.
        //
        // Output axes after the reference transpose are
        //   (B, gT, GH, GW, ms, ms, C, T, ps, ps)
        // with [GH, GW, ms, ms] flattening into the patch axis and
        // [C, T, ps, ps] into the feature axis.
        let feature_dim = c * tps * ps * ps;
        let num_patches = grid_t * grid_h * grid_w;
        let mut out = vec![0_f32; num_patches * feature_dim];

        let chan_stride = (target_h as usize) * (target_w as usize);

        for gh in 0..(grid_h / ms) {
            for gw in 0..(grid_w / ms) {
                for mh in 0..ms {
                    for mw in 0..ms {
                        let patch_idx = ((gh * (grid_w / ms) + gw) * ms + mh) * ms + mw;
                        let base_y = gh * ms * ps + mh * ps;
                        let base_x = gw * ms * ps + mw * ps;
                        let patch_offset = patch_idx * feature_dim;
                        for ch in 0..c {
                            let ch_base = ch * chan_stride;
                            for t in 0..tps {
                                let inner_base = patch_offset + (ch * tps + t) * ps * ps;
                                for py in 0..ps {
                                    let src_row =
                                        ch_base + (base_y + py) * (target_w as usize) + base_x;
                                    let dst_row = inner_base + py * ps;
                                    out[dst_row..dst_row + ps]
                                        .copy_from_slice(&img_chw[src_row..src_row + ps]);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(ProcessedImage {
            pixel_values: out,
            grid_thw: [grid_t as i32, grid_h as i32, grid_w as i32],
            feature_dim: feature_dim as i32,
        })
    }
}

fn resize_rgb(
    rgb: ImageBuffer<Rgb<u8>, Vec<u8>>,
    target_w: u32,
    target_h: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (w, h) = rgb.dimensions();
    if target_w == w && target_h == h {
        return rgb;
    }
    image::imageops::resize(&rgb, target_w, target_h, FilterType::CatmullRom)
}

fn hwc_u8_to_chw_f32(rgb: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Vec<f32> {
    let (w, h) = rgb.dimensions();
    let w = w as usize;
    let h = h as usize;
    let mut out = vec![0_f32; CHANNELS * h * w];
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x as u32, y as u32);
            for c in 0..CHANNELS {
                out[c * h * w + y * w + x] = p.0[c] as f32;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn smart_resize_rounds_to_factor_multiple() {
        let (h, w) = smart_resize(800, 1200, 32, 56 * 56, 14 * 14 * 4 * 1280).unwrap();
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
    }

    #[test]
    fn smart_resize_clamps_against_max_pixels() {
        // Way too many pixels -> must shrink so h*w <= max_pixels and both dims
        // remain multiples of `factor`.
        let max_pixels = 1_003_520;
        let factor = 32;
        let (h, w) = smart_resize(4096, 4096, factor, 3136, max_pixels).unwrap();
        assert_eq!(h % factor, 0);
        assert_eq!(w % factor, 0);
        assert!((h * w) <= max_pixels);
    }

    #[test]
    fn smart_resize_clamps_against_min_pixels() {
        // Way too small -> must grow.
        let min_pixels = 3136;
        let factor = 32;
        let (h, w) = smart_resize(16, 16, factor, min_pixels, 1_000_000).unwrap();
        assert_eq!(h % factor, 0);
        assert_eq!(w % factor, 0);
        assert!((h * w) >= min_pixels);
    }

    #[test]
    fn smart_resize_rejects_extreme_aspect_ratio() {
        assert!(smart_resize(1, 500, 32, 3136, 1_000_000).is_err());
    }

    #[test]
    fn preprocess_produces_expected_patch_grid_and_normalisation() {
        // Build a deterministic 64x32 RGB image: red linearly across x, green
        // across y, blue constant 128. Then verify the output shape and the
        // mean-zero normalisation invariant.
        let w = 64_u32;
        let h = 32_u32;
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(w, h, |x, y| Rgb([(x * 4) as u8, (y * 8) as u8, 128]));
        let proc = Qwen35ImageProcessor::new(ImageProcessorConfig {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            min_pixels: 64 * 64, // force the small image to grow
            max_pixels: 1_003_520,
            ..ImageProcessorConfig::default()
        });
        let processed = proc.preprocess_image(DynamicImage::ImageRgb8(img)).unwrap();

        let [t, h, w] = processed.grid_thw;
        assert_eq!(t, 1);
        // Both grid dims must be multiples of `merge_size = 2`.
        assert_eq!(h % 2, 0);
        assert_eq!(w % 2, 0);
        // Total area resized to >= min_pixels and a multiple of factor = 32.
        let resized_h = h * 16;
        let resized_w = w * 16;
        assert!(resized_h * resized_w >= 64 * 64);
        assert_eq!(resized_h % 32, 0);
        assert_eq!(resized_w % 32, 0);

        // Feature dim = C * tps * ps * ps = 3 * 2 * 16 * 16 = 1536.
        assert_eq!(processed.feature_dim, 1536);
        assert_eq!(
            processed.pixel_values.len(),
            (processed.num_patches() as usize) * (processed.feature_dim as usize)
        );

        // Default mean/std = 0.5/0.5 -> normalised value = (v/255 - 0.5)/0.5.
        // A constant blue channel of 128 normalises to (128/255 - 0.5)/0.5.
        // pixel_values layout per-patch is [C, T, ps, ps]; the blue block
        // starts after the R and G blocks.
        let blue_start = 2 * 2 * 16 * 16;
        let blue_slice = &processed.pixel_values[blue_start..blue_start + 16 * 16];
        let avg: f32 = blue_slice.iter().sum::<f32>() / (blue_slice.len() as f32);
        let expected = (128.0 / 255.0 - 0.5) / 0.5;
        assert!(
            (avg - expected).abs() < 1e-3,
            "blue channel avg {avg} should be ~{expected}"
        );
    }

    #[test]
    fn preprocess_temporal_axis_is_duplicated() {
        // temporal_patch_size = 2 means the same frame is repeated twice along
        // the inner feature dim. Verify the two halves agree exactly.
        let w = 32_u32;
        let h = 32_u32;
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(w, h, |x, y| Rgb([(x as u8).wrapping_mul(3), y as u8, 200]));
        let proc = Qwen35ImageProcessor::new(ImageProcessorConfig {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            min_pixels: 32 * 32,
            max_pixels: 1_003_520,
            ..ImageProcessorConfig::default()
        });
        let processed = proc.preprocess_image(DynamicImage::ImageRgb8(img)).unwrap();
        let ps = 16_usize;
        let tps = 2_usize;
        let c = CHANNELS;
        let fdim = processed.feature_dim as usize;
        assert_eq!(fdim, c * tps * ps * ps);
        let patch0 = &processed.pixel_values[0..fdim];
        // layout per-patch: [C][tps][ps][ps] flat row-major. The two t-slices
        // for each channel must match byte-for-byte.
        for ch in 0..c {
            let chan_off = ch * tps * ps * ps;
            let t0 = &patch0[chan_off..chan_off + ps * ps];
            let t1 = &patch0[chan_off + ps * ps..chan_off + 2 * ps * ps];
            assert_eq!(t0, t1, "channel {ch} temporal slices differ");
        }
    }
}
