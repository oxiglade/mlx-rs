use crate::{cache::CacheStep, InferenceError};
type Result<T> = std::result::Result<T, InferenceError>;
use mlx_rs::{
    fast, nn,
    ops::{self, indexing::IndexOp},
    Array, Dtype,
};
use std::{cell::RefCell, collections::HashMap};

type Compiled = RefCell<Box<dyn FnMut(&[Array]) -> mlx_rs::error::Result<Vec<Array>>>>;

fn compiled(f: fn(&[Array]) -> mlx_rs::error::Result<Vec<Array>>) -> Compiled {
    RefCell::new(Box::new(mlx_rs::transforms::compile::compile(f, true)))
}

thread_local! {
    static SWIGLU: Compiled = compiled(|a| Ok(vec![a[0].multiply(ops::sigmoid(&a[0])?)?.multiply(&a[1])?]));
    static SILU: Compiled = compiled(|a| Ok(vec![a[0].multiply(ops::sigmoid(&a[0])?)?]));
    static DECAY: Compiled = compiled(|a| Ok(vec![a[0].as_dtype(Dtype::Float32)?.exp()?.negative()?.multiply(nn::softplus(a[1].add(&a[2])?)?)?.exp()?]));
}

fn swiglu(g: &Array, u: &Array) -> mlx_rs::error::Result<Array> {
    SWIGLU.with_borrow_mut(|f| Ok(f(&[g.clone(), u.clone()])?.remove(0)))
}

fn silu(x: Array) -> mlx_rs::error::Result<Array> {
    SILU.with_borrow_mut(|f| Ok(f(&[x])?.remove(0)))
}

pub(super) struct Weights(
    pub(super) HashMap<String, Array>,
    pub(super) fast::MetalKernel,
);
impl Weights {
    fn tensor(&self, name: &str) -> Result<&Array> {
        self.0
            .get(name)
            .ok_or_else(|| mlx_rs::error::Exception::custom(format!("missing {name}")).into())
    }
    fn norm(&self, x: &Array, name: &str) -> Result<Array> {
        Ok(fast::rms_norm(
            x,
            Some(self.tensor(&format!("{name}.weight"))?),
            1e-6,
        )?)
    }
    fn linear(&self, x: &Array, name: &str, idx: Option<&Array>) -> Result<Array> {
        let w = self.tensor(&format!("{name}.weight"))?;
        let s = self.tensor(&format!("{name}.scales"))?;
        let b = self.tensor(&format!("{name}.biases"))?;
        let bits = Some(32 * w.shape()[w.ndim() - 1] / (64 * s.shape()[s.ndim() - 1]));
        Ok(if let Some(ids) = idx {
            ops::gather_qmm(
                x,
                w,
                s,
                b,
                ops::GatherQmmOptions {
                    rhs_indices: Some(ids),
                    sorted_indices: ids.ndim() == 1,
                    bits,
                    group_size: Some(64),
                    ..Default::default()
                },
            )
            .map_err(mlx_rs::error::Exception::from)?
        } else {
            ops::quantized_matmul(
                x,
                w,
                s,
                b,
                ops::QuantizedMatmulOptions {
                    transpose: true,
                    group_size: Some(64),
                    bits,
                    ..Default::default()
                },
            )
            .map_err(mlx_rs::error::Exception::from)?
        })
    }
    fn embed(&self, ids: &Array) -> Result<Array> {
        let rows = |field| -> Result<Array> {
            Ok(self
                .tensor(&format!("language_model.model.embed_tokens.{field}"))?
                .take_axis(ids, 0)?)
        };
        Ok(ops::dequantize(
            rows("weight")?,
            rows("scales")?,
            &rows("biases")?,
            ops::DequantizeOptions {
                bits: Some(4),
                group_size: Some(64),
                ..Default::default()
            },
        )
        .map_err(mlx_rs::error::Exception::from)?
        .reshape(&[1, -1, 2048])?)
    }
    fn moe(&self, x: &Array, p: &str) -> Result<Array> {
        let linear = |x, n| self.linear(x, &format!("{p}.{n}"), None);
        let probs = ops::softmax_axis(linear(x, "gate")?, -1, true)?;
        let indices = ops::argpartition_axis(&probs, -8, -1)?.index((.., .., -8..));
        let scores = probs.take_along_axis(&indices, -1)?;
        let scores = scores.divide(scores.sum_axis(-1, true)?)?;
        let (expanded, routes, restore) = if indices.size() >= 64 {
            let flat = indices.reshape(&[-1])?;
            let order = ops::argsort_axis(&flat, 0)?;
            (
                x.reshape(&[-1, 1, 2048])?
                    .take_axis(order.floor_divide(Array::from_slice(&[8u32], &[]))?, 0)?,
                flat.take_axis(&order, 0)?,
                Some(ops::argsort_axis(order, 0)?),
            )
        } else {
            (x.reshape(&[1, -1, 1, 1, 2048])?, indices.clone(), None)
        };
        let expert = |x: &Array, n| self.linear(x, &format!("{p}.switch_mlp.{n}"), Some(&routes));
        let selected = swiglu(
            &expert(&expanded, "gate_proj")?,
            &expert(&expanded, "up_proj")?,
        )?;
        let projected = expert(&selected, "down_proj")?;
        let out = if let Some(order) = restore {
            let output = fast::MetalKernelOutput {
                shape: x.shape().to_vec(),
                dtype: x.dtype(),
            };
            // SAFETY: order permutes the eight routed rows per token; each lane writes one channel.
            unsafe {
                self.1.apply(
                    &[projected, order, scores],
                    &[output],
                    (2048, x.shape()[1], 1),
                    (128, 1, 1),
                    Default::default(),
                )
            }?
            .remove(0)
        } else {
            projected
                .reshape(&[1, -1, 8, 2048])?
                .multiply(scores.reshape(&[1, -1, 8, 1])?)?
                .sum_axis(-2, false)?
        };
        let shared = swiglu(
            &linear(x, "shared_expert.gate_proj")?,
            &linear(x, "shared_expert.up_proj")?,
        )?;
        Ok(out.add(
            self.linear(&shared, &format!("{p}.shared_expert.down_proj"), None)?
                .multiply(ops::sigmoid(linear(x, "shared_expert_gate")?)?)?,
        )?)
    }
    fn attention(
        &self,
        x: &Array,
        p: &str,
        cache: &mut CacheStep<'_>,
        layer: usize,
        offset: i32,
    ) -> Result<Array> {
        let linear = |n| self.linear(x, &format!("{p}.{n}_proj"), None);
        let qg = linear("q")?
            .reshape(&[1, -1, 16, 512])?
            .split_equal(2, -1)?;
        let rotate = |x: &Array, n| -> Result<Array> {
            Ok(fast::rope(
                self.norm(x, &format!("{p}.{n}_norm"))?
                    .transpose_axes(&[0, 2, 1, 3])?,
                64,
                false,
                10_000_000.,
                1.,
                offset,
                None,
            )?)
        };
        let q = rotate(&qg[0], "q")?;
        let k = rotate(&linear("k")?.reshape(&[1, -1, 2, 256])?, "k")?;
        let v = linear("v")?
            .reshape(&[1, -1, 2, 256])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let (k, v) = cache.update_and_fetch(layer, k, v)?;
        let out = fast::scaled_dot_product_attention(
            q,
            k,
            v,
            1. / 16.,
            Some(fast::ScaledDotProductAttentionMask::Causal),
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[1, -1, 4096])?;
        self.linear(
            &out.multiply(ops::sigmoid(qg[1].reshape(&[1, -1, 4096])?)?)?,
            &format!("{p}.o_proj"),
            None,
        )
    }
    fn delta(
        &self,
        x: &Array,
        p: &str,
        (history, state): (Array, Array),
        kernel: &fast::MetalKernel,
    ) -> Result<((Array, Array), Array)> {
        let linear = |n| self.linear(x, &format!("{p}.{n}"), None);
        let t = x.shape()[1];
        let qkv = linear("in_proj_qkv")?;
        let conv = ops::concatenate(&[history, qkv], 1)?;
        let parts = silu(ops::conv1d(
            &conv,
            self.tensor(&format!("{p}.conv1d.weight"))?,
            1,
            0,
            1,
            8192,
        )?)?
        .split_at_indices(&[2048, 4096], -1)?;
        let norm = |x: &Array, s: f32| -> Result<Array> {
            Ok(
                fast::rms_norm(x.reshape(&[t, 16, 128])?, None, 1e-6 / 128.)?
                    .multiply(Array::from_f32(s).as_dtype(x.dtype())?)?,
            )
        };
        let q = norm(&parts[0], 1. / 128.)?;
        let k = norm(&parts[1], 1. / 128f32.sqrt())?;
        let v = parts[2].reshape(&[t, 32, 128])?;
        let args = [
            self.tensor(&format!("{p}.A_log"))?.clone(),
            linear("in_proj_a")?,
            self.tensor(&format!("{p}.dt_bias"))?.clone(),
        ];
        let g = DECAY
            .with_borrow_mut(|f| f(&args))?
            .remove(0)
            .reshape(&[t, 32])?;
        let b = ops::sigmoid(linear("in_proj_b")?)?.reshape(&[t, 32])?;
        let outputs = [
            fast::MetalKernelOutput {
                shape: v.shape().to_vec(),
                dtype: x.dtype(),
            },
            fast::MetalKernelOutput {
                shape: state.shape().to_vec(),
                dtype: Dtype::Float32,
            },
        ];
        // SAFETY: each value row owns 128 state elements and one output writer.
        let result = unsafe {
            kernel.apply(
                &[q, k, v, g, b, state],
                &outputs,
                (32, if t > 1 { 16 } else { 128 }, 32),
                (32, if t > 1 { 1 } else { 4 }, 1),
                Default::default(),
            )
        }?;
        let state = (conv.index((.., -3.., ..)), result[1].clone());
        let z = linear("in_proj_z")?
            .reshape(&[t, 32, 128])?
            .as_dtype(Dtype::Float32)?;
        let u = self
            .norm(&result[0], &format!("{p}.norm"))?
            .as_dtype(Dtype::Float32)?
            .multiply(silu(z)?)?
            .as_dtype(x.dtype())?
            .reshape(&[1, t, 4096])?;
        Ok((state, self.linear(&u, &format!("{p}.out_proj"), None)?))
    }
    pub(super) fn forward(
        &self,
        ids: &Array,
        cache: &mut CacheStep<'_>,
        kernel: &fast::MetalKernel,
    ) -> Result<Array> {
        let &[1, t] = ids.shape() else {
            return Err(crate::CacheError::InvalidState("expected one sequence".into()).into());
        };
        if t <= 0 || t > i32::MAX / 16384 {
            return Err(crate::CacheError::InvalidState(
                "chunk length outside kernel index range".into(),
            )
            .into());
        }
        let offset = i32::try_from(cache.info(0)?.processed_tokens)
            .map_err(|_| crate::CacheError::InvalidState("position exceeds MLX range".into()))?;
        let mut x = self.embed(ids)?;
        for i in 0..40 {
            let p = format!("language_model.model.layers.{i}");
            let h = self.norm(&x, &format!("{p}.input_layernorm"))?;
            x = x.add(if i % 4 == 3 {
                self.attention(&h, &format!("{p}.self_attn"), cache, i, offset)?
            } else {
                cache.recurrent(i, t as usize, |state| {
                    self.delta(&h, &format!("{p}.linear_attn"), state, kernel)
                })?
            })?;
            x = x.add(self.moe(
                &self.norm(&x, &format!("{p}.post_attention_layernorm"))?,
                &format!("{p}.mlp"),
            )?)?;
        }
        self.linear(
            &self.norm(&x, "language_model.model.norm")?,
            "language_model.lm_head",
            None,
        )
    }
}
