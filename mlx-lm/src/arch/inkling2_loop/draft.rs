use super::forward::{f, norm, proj, Weights};
use crate::{
    arch::DraftContext,
    weights::{WeightDisposition, WeightManifest},
    InferenceError, LoadError, ParameterPath, TokenId,
};
use mlx_rs::{
    fast,
    ops::{self, indexing::IndexOp},
    utils::StateProjection,
    Array, Dtype,
};
use std::{collections::HashMap, path::Path};

type Result<T> = std::result::Result<T, InferenceError>;

/// The `mtp_config` values the drafter hard-codes.
pub(super) const PROFILE: &str = r#"{"block_size": 8, "head_dim": 128, "hidden_size": 2304,
"intermediate_size": 9216, "local_layer_ids": [0, 2], "markov_head_type": "rnn_hidden",
"markov_rank": 256, "num_attention_heads": 24, "num_hidden_layers": 3, "num_key_value_heads": 8,
"o_bias": false, "q_bias": false, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0,
"sliding_window_size": 1024, "target_layer_ids": [5, 23, 41, 48]}"#;

/// Rows of a verification step: the anchor and seven proposals.
pub(super) const BLOCK: usize = 8;
/// Context rows the first and last layers attend to; the middle layer attends to all.
const WINDOW: i32 = 1024;

/// The drafter the checkpoint bundles as `mtp.safetensors`: three bidirectional layers over
/// the anchor and seven mask rows, attending to the target's hidden rows, and a recurrent head.
pub(super) struct Drafter(HashMap<String, Array>);

fn slots() -> std::result::Result<HashMap<String, Array>, LoadError> {
    let mut m = HashMap::new();
    let mut slot = |name: String, shape: &[i32]| -> std::result::Result<(), LoadError> {
        m.insert(name, ops::zeros_dtype(shape, Dtype::Bfloat16)?);
        Ok(())
    };
    slot("fuse_weight".into(), &[4, 4])?;
    slot("mask_embed".into(), &[2304])?;
    slot("norm.weight".into(), &[2304])?;
    for i in 0..3 {
        let p = format!("layers.{i}");
        for name in ["dflare_norm", "input_layernorm", "post_attention_layernorm"] {
            slot(format!("{p}.{name}.weight"), &[2304])?;
        }
        for (name, rows, columns) in [
            ("self_attn.q_proj", 3072, 2304),
            ("self_attn.k_proj", 1024, 2304),
            ("self_attn.v_proj", 1024, 2304),
            ("self_attn.k_proj_ctx", 1024, 2304),
            ("self_attn.v_proj_ctx", 1024, 2304),
            ("self_attn.o_proj", 2304, 3072),
            ("mlp.gate_proj", 9216, 2304),
            ("mlp.up_proj", 9216, 2304),
            ("mlp.down_proj", 2304, 9216),
        ] {
            slot(format!("{p}.{name}.weight"), &[rows, columns])?;
        }
        for name in ["q_norm", "k_norm", "k_norm_ctx"] {
            slot(format!("{p}.self_attn.{name}.weight"), &[128])?;
        }
    }
    for (name, shape) in [
        ("dflare_norm.weight", &[2304][..]),
        ("hidden_out.weight", &[2304, 256]),
        ("hidden_out.bias", &[2304]),
        ("markov_w1.weight", &[201024, 256]),
        ("rnn_proj.weight", &[768, 5120]),
        ("rnn_proj.bias", &[768]),
    ] {
        slot(format!("markov_head.{name}"), shape)?;
    }
    Ok(m)
}

fn heads(x: &Array, count: i32) -> Result<Array> {
    Ok(x.reshape(&[1, x.shape()[1], count, 128])?
        .transpose_axes(&[0, 2, 1, 3])?)
}
fn rope(x: &Array, position: i32) -> Result<Array> {
    Ok(fast::rope(x, 128, false, 1e6, 1., position, None)?)
}

impl Drafter {
    pub(super) fn load(path: &Path, target: &Weights) -> std::result::Result<Self, LoadError> {
        let mut w = slots()?;
        let mut projection = StateProjection::new();
        for (key, value) in &mut w {
            projection
                .required(key.as_str(), value)
                .map_err(crate::WeightError::from)?;
        }
        WeightManifest::from_safetensors(path)?.load_with(&mut projection, |key| {
            WeightDisposition::Parameter(ParameterPath::new(key))
        })?;
        drop(projection);
        let wide = |name: &str| w[name].as_dtype(Dtype::Float32);
        // The reference rounds the mixing weights to 16 bits before it uses them.
        let alpha = ops::softmax_axis(wide("fuse_weight")?, -1, None)?
            .as_dtype(Dtype::Bfloat16)?
            .as_dtype(Dtype::Float32)?;
        // The head's output projection goes through the unembedding; folded once here.
        let unembed = &target.0["model.llm.unembed.weight"];
        let fold = |name: &str| {
            unembed
                .matmul(wide(&format!("markov_head.hidden_out.{name}"))?)?
                .as_dtype(Dtype::Bfloat16)
        };
        // Columns: the state and the previous token's embedding, then the block row and
        // the newest context row.
        let rnn = wide("markov_head.rnn_proj.weight")?;
        let derived = [
            ("alpha", alpha),
            ("fold.weight", fold("weight")?),
            ("fold.bias", fold("bias")?),
            ("rnn.state", rnn.index((.., ..512)).t()),
            ("rnn.context", rnn.index((.., 512..)).t()),
            ("rnn.bias", wide("markov_head.rnn_proj.bias")?),
        ];
        w.retain(|name, _| {
            !["fuse_weight", "hidden_out", "rnn_proj"]
                .iter()
                .any(|folded| name.contains(folded))
        });
        w.extend(derived.map(|(name, array)| (name.to_owned(), array)));
        mlx_rs::transforms::eval(w.values())?;
        mlx_rs::memory::clear_cache()?;
        Ok(Self(w))
    }

    fn tensor(&self, name: &str) -> Result<&Array> {
        self.0.get(name).ok_or_else(|| {
            mlx_rs::error::Exception::custom(format!("missing drafter {name}")).into()
        })
    }

    /// The four tapped states of each row, weighted for one layer (row 3 is the head's).
    fn mix(&self, taps: &Array, row: i32, weight: &str) -> Result<Array> {
        let parts = f(&taps.reshape(&[1, taps.shape()[1], 4, 2304])?)?;
        let alpha = self.tensor("alpha")?.index(row);
        // Summed in the reference's order.
        let mut mixed = parts.index((.., .., 0, ..)).multiply(alpha.index(0))?;
        for j in 1..4 {
            mixed = mixed.add(parts.index((.., .., j, ..)).multiply(alpha.index(j))?)?;
        }
        Ok(norm(&mixed, Some(self.tensor(weight)?))?.as_dtype(Dtype::Bfloat16)?)
    }

    pub(super) fn observe(&self, context: &mut DraftContext, taps: &Array) -> Result<()> {
        let rows = taps.shape()[1];
        if context.layers.is_empty() {
            let empty = ops::zeros_dtype(&[1, 8, 0, 128], Dtype::Bfloat16)?;
            context.layers = vec![(empty.clone(), empty); 3];
        }
        for (i, (keys, values)) in context.layers.iter_mut().enumerate() {
            let p = format!("layers.{i}");
            let windowed = i != 1;
            // Rows a windowed layer would drop at once are not projected.
            let skip = if windowed { (rows - WINDOW).max(0) } else { 0 };
            let mixed = self.mix(
                &taps.index((.., skip.., ..)),
                i as i32,
                &format!("{p}.dflare_norm.weight"),
            )?;
            let side = |name: &str| {
                let w = self.tensor(&format!("{p}.self_attn.{name}_proj_ctx.weight"))?;
                heads(&proj(&mixed, w)?, 8)
            };
            let k = norm(
                &side("k")?,
                Some(self.tensor(&format!("{p}.self_attn.k_norm_ctx.weight"))?),
            )?;
            *keys = ops::concatenate(&[&*keys, &rope(&k, context.seen + skip)?], 2)?;
            *values = ops::concatenate(&[&*values, &side("v")?], 2)?;
            if windowed {
                let start = (keys.shape()[2] - WINDOW).max(0);
                *keys = keys.index((.., .., start.., ..));
                *values = values.index((.., .., start.., ..));
            }
        }
        context.seen += rows;
        context.last = Some(taps.index((.., -1.., ..)));
        Ok(())
    }

    pub(super) fn propose(
        &self,
        target: &Weights,
        context: &DraftContext,
        anchor: TokenId,
        count: usize,
    ) -> Result<Array> {
        let last = context.last.as_ref().ok_or_else(|| {
            InferenceError::from(mlx_rs::error::Exception::custom("drafter has no context"))
        })?;
        let bf16 = Dtype::Bfloat16;
        let mut previous = Array::from_slice(&[u32::from(anchor)], &[1]);
        let embed = norm(
            &target
                .tensor("model.llm.embed.weight")?
                .take_axis(&previous, 0)?,
            Some(target.tensor("model.llm.embed_norm.weight")?),
        )?
        .reshape(&[1, 1, 2304])?;
        // The layers are bidirectional and were trained on full blocks, so every mask row
        // runs even when fewer proposals are wanted.
        let masks = ops::broadcast_to(self.tensor("mask_embed")?, &[1, BLOCK as i32 - 1, 2304])?;
        let mut x = ops::concatenate(&[embed, masks], 1)?;
        let mut residual: Option<Array> = None;
        for (i, (keys, values)) in context.layers.iter().enumerate() {
            let w = |name: &str| self.tensor(&format!("layers.{i}.{name}.weight"));
            // Sums are normalized unrounded; the residual carried on is their 16-bit rounding.
            let summed = match &residual {
                Some(residual) => f(&x)?.add(f(residual)?)?,
                None => x.clone(),
            };
            let carried = summed.as_dtype(bf16)?;
            let h = norm(&summed, Some(w("input_layernorm")?))?.as_dtype(bf16)?;
            let side = |name: &str, count: i32| -> Result<Array> {
                let projected = heads(&proj(&h, w(&format!("self_attn.{name}_proj"))?)?, count)?;
                rope(
                    &norm(&projected, Some(w(&format!("self_attn.{name}_norm"))?))?,
                    context.seen,
                )
            };
            let k = ops::concatenate(&[keys, &side("k", 8)?], 2)?;
            let v = ops::concatenate(&[values, &heads(&proj(&h, w("self_attn.v_proj")?)?, 8)?], 2)?;
            let attended = fast::scaled_dot_product_attention(
                side("q", 24)?,
                k,
                v,
                128f64.powf(-0.5) as f32,
                None,
                None,
            )?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[1, BLOCK as i32, 3072])?;
            let summed = f(&proj(&attended, w("self_attn.o_proj")?)?)?.add(f(&carried)?)?;
            let m = norm(&summed, Some(w("post_attention_layernorm")?))?.as_dtype(bf16)?;
            let gate = f(&proj(&m, w("mlp.gate_proj")?)?)?;
            let up = f(&proj(&m, w("mlp.up_proj")?)?)?;
            // The MLP's output joins the residual in the next layer's sum, or the final norm's.
            x = proj(
                &gate
                    .multiply(ops::sigmoid(&gate)?)?
                    .multiply(up)?
                    .as_dtype(bf16)?,
                w("mlp.down_proj")?,
            )?;
            residual = Some(summed.as_dtype(bf16)?);
        }
        let residual = residual.expect("the drafter has layers");
        let x = norm(
            &f(&x)?.add(f(&residual)?)?,
            Some(self.tensor("norm.weight")?),
        )?
        .as_dtype(bf16)?
        .index((.., 1..count as i32 + 1, ..));

        let base = proj(&f(&x)?, target.tensor("model.llm.unembed.weight")?)?.as_dtype(bf16)?;
        let newest = self.mix(last, 3, "markov_head.dflare_norm.weight")?;
        let hidden = f(&ops::concatenate(
            &[&x, &ops::broadcast_to(&newest, x.shape())?],
            -1,
        )?)?
        .matmul(self.tensor("rnn.context")?)?
        .add(self.tensor("rnn.bias")?)?;
        let one = Array::from_f32(1.).as_dtype(bf16)?;
        let mut state = ops::zeros_dtype(&[1, 256], bf16)?;
        let mut tokens = Vec::with_capacity(count);
        for i in 0..count as i32 {
            let embedded = self
                .tensor("markov_head.markov_w1.weight")?
                .take_axis(&previous, 0)?;
            let raw = f(&ops::concatenate(&[&state, &embedded], -1)?)?
                .matmul(self.tensor("rnn.state")?)?
                .add(hidden.index((.., i, ..)))?
                .as_dtype(bf16)?;
            let [gate, candidate, output]: [Array; 3] = raw
                .split_equal(3, -1)?
                .try_into()
                .expect("split into three parts");
            let gate = ops::sigmoid(f(&gate)?)?.as_dtype(bf16)?;
            let candidate = ops::tanh(f(&candidate)?)?.as_dtype(bf16)?;
            state = gate
                .multiply(&state)?
                .add(one.subtract(&gate)?.multiply(candidate)?)?;
            let latent = ops::tanh(f(&output)?)?.as_dtype(bf16)?;
            let logits = base.index((.., i, ..)).add(ops::addmm(
                self.tensor("fold.bias")?,
                latent,
                self.tensor("fold.weight")?.t(),
                None,
                None,
            )?)?;
            let logits = ops::tanh(f(&logits)?.divide(Array::from_f32(30.))?)?
                .multiply(Array::from_f32(30.))?
                .as_dtype(bf16)?;
            previous = ops::indexing::argmax_axis(logits.index((.., ..200058)), -1, false)?;
            tokens.push(previous.clone());
        }
        Ok(ops::stack(&tokens, 1)?)
    }
}
