use super::forward::Weights;
use crate::{Image, InferenceError};
use mlx_rs::{
    error::Exception,
    fast::{self, MetalKernel, MetalKernelOutput},
    nn,
    ops::{self, indexing::IndexOp},
    Array, Dtype,
};

type Result<T> = std::result::Result<T, InferenceError>;

/// Pixels per image token along each side; the encoder sees four 20-pixel patches in each.
const PATCH: usize = 40;

/// Normalized `[tokens, 40, 40, 3]` pixels with the token grid's rows and columns.
fn patches(image: &Image<'_>) -> Result<(Array, i32, i32)> {
    let (width, height) = (image.width, image.height);
    if width == 0
        || height == 0
        || width.checked_mul(height).and_then(|n| n.checked_mul(3)) != Some(image.rgb.len())
    {
        return Err(Exception::custom("image is not width x height RGB bytes").into());
    }
    let (rows, cols) = (height.div_ceil(PATCH), width.div_ceil(PATCH));
    let normalize = |value: f32| (value * (1. / 255.) - 0.5) / 0.5;
    // The reference pads the right and bottom edges with -1 before normalizing.
    let mut data = vec![normalize(-1.); rows * cols * PATCH * PATCH * 3];
    for (y, line) in image.rgb.chunks_exact(width * 3).enumerate() {
        for (x, pixel) in line.as_chunks::<3>().0.iter().enumerate() {
            let patch = y / PATCH * cols + x / PATCH;
            let at = ((patch * PATCH + y % PATCH) * PATCH + x % PATCH) * 3;
            for (out, &value) in data[at..at + 3].iter_mut().zip(pixel) {
                *out = normalize(f32::from(value));
            }
        }
    }
    let side = PATCH as i32;
    let (rows, cols) = (rows as i32, cols as i32);
    Ok((
        Array::from_slice(&data, &[rows * cols, side, side, 3]),
        rows,
        cols,
    ))
}

/// Source indices and weights of a two-tap bilinear resize along one axis.
fn taps(input: i32, output: i32) -> Result<(Array, Array)> {
    let scale = (f64::from(input) / f64::from(output)) as f32;
    let source = ops::maximum(
        Array::arange::<_, f32>(None, output as f32, None)?
            .add(Array::from_f32(0.5))?
            .multiply(Array::from_f32(scale))?
            .subtract(Array::from_f32(0.5))?,
        Array::from_f32(0.),
    )?;
    let base = source.floor()?;
    let fraction = source.subtract(&base)?;
    let position = base
        .expand_dims(1)?
        .add(Array::arange::<_, f32>(None, 2., None)?.expand_dims(0)?)?;
    Ok((
        ops::clip(position.as_dtype(Dtype::Int32)?, (0, input - 1))?,
        ops::stack(&[Array::from_f32(1.).subtract(&fraction)?, fraction], 1)?,
    ))
}

/// The rows that replace one image's tokens, `[tokens, 2304]`.
pub(super) fn encode(w: &Weights, resize: &MetalKernel, image: &Image<'_>) -> Result<Array> {
    let (pixels, rows, cols) = patches(image)?;
    let count = rows * cols;
    let t = |name: &str| w.tensor(&format!("model.visual.{name}"));
    let linear = |x: &Array, name: &str| -> Result<Array> {
        Ok(ops::addmm(
            t(&format!("{name}.bias"))?,
            x,
            t(&format!("{name}.weight"))?.t(),
            None,
            None,
        )?)
    };
    let norm = |x: &Array, name: &str| -> Result<Array> {
        Ok(fast::layer_norm(
            x,
            t(&format!("{name}.weight"))?,
            t(&format!("{name}.bias"))?,
            1e-5,
        )?)
    };
    let dtype = t("backbone.ln_pre.weight")?.dtype();
    let x = pixels
        .as_dtype(dtype)?
        .reshape(&[count, 2, 20, 2, 20, 3])?
        .transpose_axes(&[0, 1, 3, 2, 4, 5])?
        .reshape(&[count * 4, -1])?
        .matmul(t("backbone.embeddings.patch_embedding.weight")?.t())?;

    let learned = t("backbone.embeddings.position_embedding")?
        .reshape(&[1, 32, 32, 1024])?
        .as_dtype(Dtype::Float32)?;
    let (grid_h, grid_w) = (rows * 2, cols * 2);
    let position = if (grid_h, grid_w) == (32, 32) {
        learned
    } else {
        let (iy, wy) = taps(32, grid_h)?;
        let (ix, wx) = taps(32, grid_w)?;
        // SAFETY: the tap indices are clipped to the 32-entry axes they index, and each thread
        // writes its own element of the [1, grid_h, grid_w, 1024] output.
        unsafe {
            resize.apply(
                &[&learned, &iy, &wy, &ix, &wx],
                &[MetalKernelOutput {
                    shape: vec![1, grid_h, grid_w, 1024],
                    dtype: Dtype::Float32,
                }],
                (grid_w * 1024, grid_h, 1),
                (256, 1, 1),
                Default::default(),
            )?
        }
        .remove(0)
    };
    let position = position
        .reshape(&[rows, 2, cols, 2, 1024])?
        .transpose_axes(&[0, 2, 1, 3, 4])?
        .reshape(x.shape())?;
    let class = t("backbone.embeddings.class_embedding")?
        .add(t("backbone.embeddings.cls_position_embedding")?)?
        .as_dtype(dtype)?;
    let mut x = norm(
        &ops::concatenate(
            &[class.expand_dims(0)?, x.add(position.as_dtype(dtype)?)?],
            0,
        )?,
        "backbone.ln_pre",
    )?;

    // The class token sits at the origin; patch coordinates start at one.
    let mut coordinates = vec![0f32; 2];
    for row in 0..rows {
        for col in 0..cols {
            for y in 0..2 {
                for x in 0..2 {
                    coordinates.extend([(col * 2 + x + 1) as f32, (row * 2 + y + 1) as f32]);
                }
            }
        }
    }
    let length = count * 4 + 1;
    let frequencies = ops::power(
        Array::from_f32(10000.),
        Array::arange::<_, i32>(None, 32, 2)?
            .negative()?
            .divide(Array::from_int(32))?,
    )?;
    let angles = Array::from_slice(&coordinates, &[length, 2, 1])
        .multiply(frequencies)?
        .reshape(&[length, -1])?;
    let angles = ops::repeat_axis::<f32>(angles, 2, -1)?.reshape(&[1, length, 1, 64])?;
    let (cos, sin) = (angles.cos()?, angles.sin()?);
    let rotate = |a: Array| -> Result<Array> {
        let a32 = a.as_dtype(Dtype::Float32)?;
        let pairs = a32.reshape(&[1, length, 16, 32, 2])?;
        let turned = ops::stack(
            &[
                pairs.index((.., .., .., .., 1)).negative()?,
                pairs.index((.., .., .., .., 0)),
            ],
            -1,
        )?
        .reshape(a32.shape())?;
        Ok(a32
            .multiply(&cos)?
            .add(turned.multiply(&sin)?)?
            .as_dtype(a.dtype())?)
    };

    for layer in 0..23 {
        let p = format!("backbone.layers.{layer}");
        let h = norm(&x, &format!("{p}.ln_1"))?;
        let heads = |name: &str| -> Result<Array> {
            Ok(linear(&h, &format!("{p}.self_attn.{name}_proj"))?.reshape(&[1, length, 16, 64])?)
        };
        let split = |a: Array| a.transpose_axes(&[0, 2, 1, 3]);
        let attention = fast::scaled_dot_product_attention(
            split(rotate(heads("q")?)?)?,
            split(rotate(heads("k")?)?)?,
            split(heads("v")?)?,
            0.125,
            None,
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(x.shape())?;
        let attention = linear(&attention, &format!("{p}.self_attn.out_proj"))?;
        x = x.add(t(&format!("{p}.ls_1.gamma"))?.multiply(attention)?)?;
        let mlp = linear(
            &nn::gelu(linear(
                &norm(&x, &format!("{p}.ln_2"))?,
                &format!("{p}.mlp.c_fc"),
            )?)?,
            &format!("{p}.mlp.c_proj"),
        )?;
        x = x.add(t(&format!("{p}.ls_2.gamma"))?.multiply(mlp)?)?;
    }
    let merged = x
        .index(1..)
        .reshape(&[count, 4096])?
        .matmul(t("proj_in.weight")?.t())?;
    Ok(nn::gelu(merged)?.matmul(t("proj_out.weight")?.t())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patches_pad_the_right_and_bottom_edges() -> Result<()> {
        let rgb: Vec<u8> = (0..41 * 2 * 3).map(|i| (i % 251) as u8).collect();
        let (pixels, rows, cols) = patches(&Image {
            width: 41,
            height: 2,
            rgb: &rgb,
        })?;
        assert_eq!((rows, cols), (1, 2));
        let pixels = pixels.as_slice::<f32>();
        let at = |patch: usize, y: usize, x: usize, c: usize| {
            pixels[((patch * 40 + y) * 40 + x) * 3 + c]
        };
        let value =
            |y: usize, x: usize, c: usize| f32::from(rgb[(y * 41 + x) * 3 + c]) / 127.5 - 1.;
        assert!((at(0, 1, 39, 2) - value(1, 39, 2)).abs() < 1e-6);
        assert!((at(1, 1, 0, 1) - value(1, 40, 1)).abs() < 1e-6);
        let padding = -1. / 127.5 - 1.;
        assert!((at(1, 0, 1, 0) - padding).abs() < 1e-6);
        assert!((at(0, 2, 0, 0) - padding).abs() < 1e-6);
        Ok(())
    }
}
