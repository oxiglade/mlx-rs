use crate::{cache::CacheStep, CacheError, InferenceError, LoadError};
type Result<T> = std::result::Result<T, InferenceError>;
use mlx_rs::{
    error::Exception,
    fast::{self, MetalKernelApplyOptions, MetalKernelOutput, TemplateArg},
    ops::{
        self,
        indexing::{Ellipsis, IndexOp},
    },
    Array, Dtype,
};
use std::{cell::RefCell, collections::HashMap, thread::LocalKey};

/// Stored layer of each visit: the middle twelve layers run three times per token.
pub(super) fn visits() -> impl Iterator<Item = usize> {
    (0..18).chain(6..18).chain(6..24)
}
/// The visits whose input the drafter reads, next to the final state.
const TAPS: [usize; 3] = [5, 23, 41];

const QK_CLIP: f32 = 22.627417;

type Compiled = RefCell<Box<dyn FnMut(&[Array]) -> mlx_rs::error::Result<Vec<Array>>>>;

fn compiled(f: fn(&[Array]) -> mlx_rs::error::Result<Vec<Array>>) -> Compiled {
    RefCell::new(Box::new(mlx_rs::transforms::compile::compile(f, false)))
}

// The chains the reference compiles, so each run of elementwise steps is one GPU operation.
thread_local! {
    static CLIP: Compiled = compiled(|a| {
        let mut x = a[0].as_dtype(Dtype::Float32)?;
        if let Some(scale) = a.get(2) {
            x *= scale.as_dtype(Dtype::Float32)?;
        }
        let length = x.square()?.sum_axis(-1, true)?.sqrt()?;
        Ok(vec![(&x * ops::minimum(&a[1] / (length + 1e-6), Array::from_f32(1.))?)
            .as_dtype(a[0].dtype())?])
    });
    static ROUTE: Compiled = compiled(|a| {
        let scores = ops::sigmoid(a[0].as_dtype(Dtype::Float32)?.matmul(a[1].t())?)?;
        let ids =
            ops::argpartition_axis((&scores + &a[2]).negative()?, 7, -1)?.index((.., .., ..8));
        let s = scores.take_along_axis(&ids, -1)?;
        Ok(vec![ids, &s / (s.sum_axis(-1, true)? + 1e-20) * 8.])
    });
    static SWIGLU: Compiled = compiled(|a| {
        let g = a[0].as_dtype(Dtype::Float32)?;
        let u = a[1].as_dtype(Dtype::Float32)?;
        Ok(vec![
            (ops::tanh(&g / 4.)? * 4. * ops::sigmoid(g)? * ops::tanh(u / 25.)? * 25.)
                .as_dtype(a[0].dtype())?,
        ])
    });
    static CONV_TAIL: Compiled = compiled(|a| {
        Ok(vec![((&a[0] + a[1].as_dtype(Dtype::Float32)?) / 5f32.sqrt())
            .as_dtype(a[1].dtype())?])
    });
    static DYNAMIC_CONV: Compiled = compiled(|a| {
        let (ctx, x) = (&a[0], &a[1]);
        let n = x.shape()[1];
        let w = &a[3] + a[2].as_dtype(Dtype::Float32)?.reshape(&[1, n, 8, 1, 4])?;
        let mut sum = ops::zeros_dtype(&[1, n, 8, x.shape()[2] / 8], Dtype::Float32)?;
        for j in 0..4 {
            sum += ctx
                .index((.., 3 - j..3 - j + n, ..))
                .reshape(&[1, n, 8, -1])?
                * w.index((Ellipsis, j));
        }
        let xf = x.as_dtype(Dtype::Float32)?.reshape(&[1, n, 8, -1])?;
        Ok(vec![((sum + xf) / 3.)
            .as_dtype(x.dtype())?
            .reshape(x.shape())?])
    });
    static BLEND: Compiled = compiled(|a| {
        Ok(vec![
            (Array::from_f32(1.) - ops::tanh(&a[0])?) * &a[2] + ops::tanh(&a[1])? * &a[3],
        ])
    });
    static GATE: Compiled = compiled(|a| {
        Ok(vec![(a[0].as_dtype(Dtype::Float32)?
            * ops::sigmoid(a[1].as_dtype(Dtype::Float32)?)?)
        .as_dtype(a[0].dtype())?])
    });
    static RESIDUAL: Compiled = compiled(|a| {
        Ok(vec![
            a[0].as_dtype(Dtype::Float32)? + a[1].as_dtype(Dtype::Float32)?,
        ])
    });
    static COMBINE: Compiled = compiled(|a| {
        Ok(vec![(a[0].as_dtype(Dtype::Float32)? * a[1].expand_dims(-1)?)
            .sum_axis(-2, false)?
            .as_dtype(a[0].dtype())?])
    });
}

fn run(chain: &'static LocalKey<Compiled>, args: &[Array]) -> Result<Array> {
    Ok(chain.with_borrow_mut(|f| f(args))?.remove(0))
}
fn f(x: &Array) -> Result<Array> {
    Ok(x.as_dtype(Dtype::Float32)?)
}
fn norm(x: &Array, w: Option<&Array>) -> Result<Array> {
    Ok(fast::rms_norm(x, w, 1e-6)?)
}
fn proj(x: &Array, w: &Array) -> Result<Array> {
    Ok(x.matmul(w.t())?)
}
fn clip(x: Array, scale: Option<&Array>, cap: f32) -> Result<Array> {
    let args: Vec<_> = [x, Array::from_f32(cap)]
        .into_iter()
        .chain(scale.cloned())
        .collect();
    run(&CLIP, &args)
}
// The reference multiplies by hidden_size^-0.5; dividing by 48 rounds differently.
fn scale(x: Array) -> Result<Array> {
    Ok(x.multiply(Array::from_f32(1. / 48.))?)
}
/// MXFP8 through mlx-rs; NVFP4 through MLX directly, because mlx-rs does not expose the
/// global scale those layers need.
fn gather_qmm(
    x: &Array,
    w: &Array,
    scales: &Array,
    routes: &Array,
    sorted: bool,
    global: Option<&Array>,
) -> Result<Array> {
    let Some(global) = global else {
        return Ok(ops::gather_qmm(
            x,
            w,
            scales,
            None,
            ops::GatherQmmOptions {
                mode: ops::QuantizationMode::Mxfp8,
                bits: Some(8),
                group_size: Some(32),
                rhs_indices: Some(routes),
                sorted_indices: sorted,
                ..Default::default()
            },
        )
        .map_err(Exception::from)?);
    };
    let int = |value| mlx_sys::mlx_optional_int {
        value,
        has_value: true,
    };
    let stream = mlx_rs::Stream::thread_local_or_default();
    let mut out = mlx_sys::mlx_array {
        ctx: std::ptr::null_mut(),
    };
    // SAFETY: every handle outlives the call, and an empty array marks an absent argument.
    let status = unsafe {
        let absent = Array::from_ptr(mlx_sys::mlx_array_new());
        mlx_sys::mlx_gather_qmm(
            &mut out,
            x.as_ptr(),
            w.as_ptr(),
            scales.as_ptr(),
            absent.as_ptr(),
            absent.as_ptr(),
            routes.as_ptr(),
            true,
            int(16),
            int(4),
            c"nvfp4".as_ptr(),
            global.as_ptr(),
            sorted,
            stream.as_ptr(),
        )
    };
    if status != 0 {
        return Err(Exception::custom("NVFP4 gather_qmm failed").into());
    }
    // SAFETY: MLX initialized `out` when it reported success.
    Ok(unsafe { Array::from_ptr(out) })
}
/// The recurrent windows a step can leave behind: after each of its `n` rows from `first` on.
/// `source` holds the old window and then the step's rows in its first `len` positions.
fn windows(source: &Array, len: i32, n: i32, cap: i32, first: i32) -> Vec<Array> {
    (first..=n)
        .map(|row| {
            let end = len - n + row;
            source.index((Ellipsis, end - end.min(cap)..end, ..))
        })
        .collect()
}
fn mix(blocks: &[Array], x: &Array, q: &Array) -> Result<Array> {
    let rows: Vec<_> = blocks.iter().chain([x]).collect();
    let keys = norm(&f(&ops::stack(&rows, -2)?)?, None)?;
    let scores = scale(proj(&keys, &q.reshape(&[-1, 2304])?)?)?;
    Ok(ops::softmax_axis(scores, -2, false)?
        .swap_axes(-1, -2)?
        .matmul(keys)?)
}

pub(super) struct Weights(
    pub(super) HashMap<String, Array>,
    pub(super) fast::MetalKernel,
);
impl Weights {
    pub(super) fn tensor(&self, name: &str) -> Result<&Array> {
        self.0
            .get(name)
            .ok_or_else(|| Exception::custom(format!("missing {name}")).into())
    }
    pub(super) fn prepare(&mut self) -> std::result::Result<(), LoadError> {
        let w = &mut self.0;
        for i in 0..24 {
            let p = format!("model.llm.layers.{i}.mlp.experts");
            for name in ["w13_weight", "w2_weight"] {
                let packed = w[&format!("{p}.{name}")].view::<u32>()?;
                w.insert(format!("{p}.{name}"), packed);
                // Only the NVFP4 layers carry a per-expert maximum.
                let Some(amax) = w.get(&format!("{p}.{name}_amax")) else {
                    continue;
                };
                let global = amax
                    .divide(&w[&format!("{p}.{name}_outer_scale_denominator")])?
                    .multiply(Array::from_f32(2688.))?;
                w.insert(format!("{p}.{name}_global"), global);
            }
        }
        // 16-bit weights that only ever meet 32-bit inputs, widened once instead of on every
        // token. The embedding norm and layer 0's attention norm take 16-bit inputs.
        for (name, a) in w.iter_mut() {
            let narrow_norm =
                name.ends_with("embed_norm.weight") || name.ends_with("layers.0.attn_norm.weight");
            if name.ends_with("sconv.weight") {
                *a = a
                    .as_dtype(Dtype::Float32)?
                    .transpose_axes(&[0, 2, 1])?
                    .contiguous()?;
            } else if name.ends_with("static_w") {
                *a = a.t().reshape(&[8, -1, 4])?.as_dtype(Dtype::Float32)?;
            } else if [
                "unembed.weight",
                "gate.weight",
                "forget_weight",
                "gain_weight",
            ]
            .iter()
            .any(|s| name.ends_with(s))
                || name.ends_with("norm.weight") && !narrow_norm
            {
                *a = a.as_dtype(Dtype::Float32)?;
            }
        }
        mlx_rs::transforms::eval(w.values())?;
        // The replaced tensors would otherwise sit in MLX's buffer cache.
        mlx_rs::memory::clear_cache()?;
        Ok(())
    }
    fn conv(
        &self,
        x: &Array,
        p: &str,
        window: Array,
        dynamic: Option<&Array>,
        first: i32,
    ) -> Result<(Vec<Array>, Array)> {
        let n = x.shape()[1];
        let ctx = ops::concatenate(&[window, f(x)?], 1)?;
        let next = windows(&ctx, 3 + n, n, 3, first);
        let out = if let Some(d) = dynamic {
            let args = [
                ctx,
                x.clone(),
                proj(d, self.tensor(&format!("{p}.weight_proj"))?)?,
                self.tensor(&format!("{p}.static_w"))?.clone(),
            ];
            run(&DYNAMIC_CONV, &args)?
        } else {
            let w = self.tensor(&format!("{p}.weight"))?;
            let args = [
                ops::conv1d(ctx, w, None, None, None, x.shape()[2])?,
                x.clone(),
            ];
            run(&CONV_TAIL, &args)?
        };
        Ok((next, out))
    }
    fn attention(
        &self,
        parts: &[Array],
        p: &str,
        local: bool,
        cache: &mut CacheStep<'_>,
        slot: usize,
    ) -> Result<Array> {
        let x = &parts[0];
        let n = x.shape()[1];
        let dtype = x.dtype();
        let w = |s: &str| self.tensor(&format!("{p}.{s}"));
        let linear = |v: &Array, s: &str| proj(v, w(&format!("{s}.weight"))?);
        let mut q = clip(
            linear(x, "wq_du")?.reshape(&[1, n, 24, 128])?,
            Some(w("q_l2_clip.scale")?),
            QK_CLIP,
        )?;
        let mut rel = clip(
            linear(x, "wr_du")?.reshape(&[1, n, 24, 16])?,
            Some(w("rel_l2_clip.scale")?),
            8.,
        )?;
        let first = if cache.per_row() { 1 } else { n };
        let (key, value) = cache.recurrent_rows(slot + 1, n as usize, |(k_window, v_window)| {
            let side = |j: usize, name: &str, window: Array| {
                self.conv(
                    &linear(&parts[j], &format!("w{name}_dv"))?,
                    &format!("{p}.{name}_sconv"),
                    window,
                    (!local).then_some(&parts[j]),
                    first,
                )
            };
            let (k_windows, k) = side(1, "k", k_window)?;
            let (v_windows, v) = side(2, "v", v_window)?;
            Ok((k_windows.into_iter().zip(v_windows).collect(), (k, v)))
        })?;
        let k =
            clip(key.reshape(&[1, n, 8, 128])?, None, QK_CLIP)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = norm(&f(&value.reshape(&[1, n, 8, 128])?)?, None)?
            .as_dtype(dtype)?
            .transpose_axes(&[0, 2, 1, 3])?;
        if !local {
            let offset = i32::try_from(cache.info(slot)?.processed_tokens)
                .map_err(|_| CacheError::InvalidState("position exceeds MLX range".into()))?;
            let one = Array::from_f32(1.);
            let tau = ops::maximum(
                f(&Array::arange::<_, i32>(offset, offset + n, None)?)?
                    .add(&one)?
                    .divide(Array::from_f32(8192.))?,
                &one,
            )?
            .log()?
            .multiply(Array::from_f32(0.1))?
            .add(&one)?
            .reshape(&[1, n, 1, 1])?;
            q = f(&q)?.multiply(&tau)?.as_dtype(dtype)?;
            rel = f(&rel)?.multiply(&tau)?.as_dtype(dtype)?;
        }
        // A zero key and value stand for the attention sink.
        let zero = ops::zeros_dtype(&[1, 8, 1, 128], dtype)?;
        let with_sink = |rows: &[&Array]| Ok::<_, InferenceError>(ops::concatenate(rows, 2)?);
        let (k, v) = if local {
            cache.recurrent_rows(slot, n as usize, |(k_window, v_window)| {
                let k = with_sink(&[&k_window, &k, &zero])?;
                let v = with_sink(&[&v_window, &v, &zero])?;
                let len = k.shape()[2] - 1;
                let next = windows(&k, len, n, 1023, first)
                    .into_iter()
                    .zip(windows(&v, len, n, 1023, first))
                    .collect();
                Ok((next, (k, v)))
            })?
        } else {
            let (k, v) = cache.update_and_fetch(slot, k, v)?;
            (with_sink(&[&k, &zero])?, with_sink(&[&v, &zero])?)
        };
        let size = k.shape()[2] - 1;
        let up = |v: i32| (v + 7) / 8 * 8;
        // The kernel writes the sink column itself: appending it with MLX 0.32.4's concatenate
        // writes out of bounds once the mask passes 2^31 elements (ml-explore/mlx#4758).
        // SAFETY: each thread reads one row of `rel` and one column of the projection, or one
        // sink element, inside their shapes and writes its own element of the output.
        let mask = unsafe {
            self.1.apply(
                &[
                    &rel,
                    w("rel_logits_proj.proj")?,
                    &k,
                    &linear(x, "attn_sink_proj")?,
                ],
                &[MetalKernelOutput {
                    shape: vec![1, 24, n, size + 1],
                    dtype,
                }],
                (up(size + 1), up(n), 24),
                (8, 8, 1),
                MetalKernelApplyOptions {
                    template_args: vec![
                        ("T".into(), TemplateArg::Dtype(dtype)),
                        ("D_REL".into(), TemplateArg::Int(16)),
                        ("REL_EXTENT".into(), TemplateArg::Int(1024)),
                        (
                            "SLIDING".into(),
                            TemplateArg::Int(if local { 1024 } else { 0 }),
                        ),
                    ],
                    ..Default::default()
                },
            )?
        }
        .remove(0);
        let o = fast::scaled_dot_product_attention(
            q.transpose_axes(&[0, 2, 1, 3])?,
            k,
            v,
            1. / 128f32.sqrt(),
            Some(fast::ScaledDotProductAttentionMask::Array(&mask)),
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[1, n, -1])?;
        linear(&run(&GATE, &[o, linear(x, "gate_attn")?])?, "wo_ud")
    }
    fn moe(&self, x: &Array, p: &str) -> Result<Array> {
        let w = |s: &str| self.tensor(&format!("{p}.{s}"));
        let n = x.shape()[1];
        let route = [
            x.clone(),
            w("gate.weight")?.clone(),
            w("gate.bias")?.clone(),
        ];
        let mut route = ROUTE.with_borrow_mut(|f| f(&route))?;
        let (ids, s) = (route.remove(0), route.remove(0));
        // The reference sorts routes by expert once a call carries 64 or more of them.
        let (expanded, routes, restore) = if n * 8 >= 64 {
            let flat = ids.reshape(&[-1])?;
            let order = ops::argsort_axis(&flat, 0)?;
            (
                x.reshape(&[-1, 1, 2304])?
                    .take_axis(order.floor_divide(Array::from_slice(&[8u32], &[]))?, 0)?,
                flat.take_axis(&order, 0)?,
                Some(ops::argsort_axis(&order, 0)?),
            )
        } else {
            (x.reshape(&[1, n, 1, 1, 2304])?, ids, None)
        };
        let expert = |x: &Array, name: &str| {
            gather_qmm(
                x,
                w(&format!("experts.{name}"))?,
                w(&format!("experts.{name}_scale"))?,
                &routes,
                restore.is_some(),
                self.0.get(&format!("{p}.experts.{name}_global")),
            )
        };
        // Gate and up rows are stored interleaved.
        let gate_up = ops::unflatten(expert(&expanded, "w13_weight")?, -1, &[1152, 2])?;
        let gate_up = [gate_up.index((Ellipsis, 0)), gate_up.index((Ellipsis, 1))];
        let mut out = expert(&run(&SWIGLU, &gate_up)?, "w2_weight")?;
        if let Some(order) = &restore {
            out = out.take_axis(order, 0)?;
        }
        run(&COMBINE, &[out.reshape(&[1, n, 8, 2304])?, s])
    }
    pub(super) fn forward(
        &self,
        ids: &Array,
        images: Option<&Array>,
        cache: &mut CacheStep<'_>,
    ) -> Result<(Array, Array)> {
        let &[1, n] = ids.shape() else {
            return Err(CacheError::InvalidState("expected one sequence".into()).into());
        };
        let mut x = norm(
            &self.tensor("model.llm.embed.weight")?.take_axis(ids, 0)?,
            Some(self.tensor("model.llm.embed_norm.weight")?),
        )?;
        let dtype = x.dtype();
        if let Some(rows) = images {
            let slots = ids.eq(Array::from_int(super::IMAGE_TOKEN as i32))?;
            let row = ops::maximum(
                slots
                    .as_dtype(Dtype::Int32)?
                    .cumsum(-1, None, None)?
                    .subtract(Array::from_int(1))?,
                Array::from_int(0),
            )?;
            x = ops::select(
                slots.expand_dims(-1)?,
                rows.take_axis(row.reshape(&[-1])?, 0)?
                    .reshape(x.shape())?
                    .as_dtype(dtype)?,
                &x,
            )?;
        }
        let mut blocks = Vec::new();
        let mut prelude: Option<(Array, Vec<Array>)> = None;
        let mut bank = None;
        let mut taps = Vec::with_capacity(4);
        for (number, i) in visits().enumerate() {
            if i == 6 {
                if let Some((h, b)) = &prelude {
                    let mut rows = blocks[b.len() + 1..].to_vec();
                    rows.push(x);
                    bank = Some(norm(&f(&ops::stack(&rows, -2)?)?, None)?);
                    x = h.clone();
                    blocks = b.clone();
                } else {
                    prelude = Some((x.clone(), blocks.clone()));
                }
            }
            if TAPS.contains(&number) {
                taps.push(x.clone());
            }
            let p = format!("model.llm.layers.{i}");
            let local = i % 6 != 5;
            let messages = if let Some(b) = bank.as_ref().filter(|_| i < 18) {
                let idx = (i as i32 - 6) * 2;
                let q = self.tensor("model.llm.middle_cycle_cross_pass_queries")?;
                let kv = self.tensor("model.llm.middle_cycle_cross_pass_kv_queries")?;
                let q = ops::concatenate(
                    &[
                        q.index(idx..idx + 1),
                        kv.index(idx..idx + 2),
                        q.index(idx + 1..idx + 2),
                    ],
                    0,
                )?;
                Some(
                    ops::softmax_axis(scale(q.matmul(b.swap_axes(-1, -2)?)?)?, -1, false)?
                        .matmul(b)?,
                )
            } else {
                None
            };
            let control = |v: Array, j: i32, name: &str| -> Result<Array> {
                let Some(m) = &messages else { return Ok(v) };
                let m = m.index((.., .., j, ..));
                let w = |k| self.tensor(&format!("{p}.controllers.{name}.{k}_weight"));
                let down = w("down")?;
                let pair = ops::concatenate(&[norm(&v, None)?, norm(&m, None)?], -1)?;
                let h = f(&proj(&pair.as_dtype(down.dtype())?, down)?)?;
                run(
                    &BLEND,
                    &[proj(&h, w("forget")?)?, proj(&h, w("gain")?)?, v, m],
                )
            };
            let mixed = if i == 0 {
                x.expand_dims(-2)?
            } else {
                mix(
                    &blocks,
                    &x,
                    self.tensor(&format!("model.llm.attn_res.attn_queries.{}", i - 1))?,
                )?
            };
            let mut parts: Vec<Array> = Vec::new();
            for (j, (name, input_norm)) in [
                ("attn", "q_input_norm"),
                ("k", "k_input_norm"),
                ("v", "v_input_norm"),
            ]
            .into_iter()
            .enumerate()
            {
                if j > 0 && local && messages.is_none() {
                    parts.push(parts[0].clone());
                    continue;
                }
                let row = mixed.index((.., .., if local { 0 } else { j as i32 }, ..));
                let weight = format!(
                    "{p}.{}.weight",
                    if local { "attn_norm" } else { input_norm }
                );
                parts.push(
                    norm(&control(row, j as i32, name)?, Some(self.tensor(&weight)?))?
                        .as_dtype(dtype)?,
                );
            }
            // Three cache layers per visit, in the order the factory lays them out.
            let slot = number * 3;
            let attn = self.attention(&parts, &format!("{p}.attn"), local, cache, slot)?;
            let first = if cache.per_row() { 1 } else { n };
            x = cache.recurrent_rows(slot + 2, n as usize, |(attn_window, mlp_window)| {
                let (attn_windows, r) =
                    self.conv(&attn, &format!("{p}.attn_sconv"), attn_window, None, first)?;
                let r = if i % 6 == 0 {
                    blocks.push(x.clone());
                    f(&r)?
                } else {
                    run(&RESIDUAL, &[r, x.clone()])?
                };
                let m = mix(
                    &blocks,
                    &r,
                    self.tensor(&format!("model.llm.attn_res.mlp_queries.{i}"))?,
                )?
                .index((.., .., 0, ..));
                let m = norm(
                    &control(m, 3, "mlp")?,
                    Some(self.tensor(&format!("{p}.mlp_norm.weight"))?),
                )?
                .as_dtype(dtype)?;
                let m = self.moe(&m, &format!("{p}.mlp"))?;
                let (mlp_windows, m) =
                    self.conv(&m, &format!("{p}.mlp_sconv"), mlp_window, None, first)?;
                Ok((
                    attn_windows.into_iter().zip(mlp_windows).collect(),
                    run(&RESIDUAL, &[r, m])?.as_dtype(dtype)?,
                ))
            })?;
        }
        let h = mix(&blocks, &x, self.tensor("model.llm.attn_res.output_query")?)?.index((
            ..,
            ..,
            0,
            ..,
        ));
        let h = norm(&h, Some(self.tensor("model.llm.norm.weight")?))?.as_dtype(dtype)?;
        let logits = proj(&f(&h)?, self.tensor("model.llm.unembed.weight")?)?;
        taps.push(h);
        Ok((
            ops::tanh(logits.divide(Array::from_f32(30.))?)?
                .multiply(Array::from_f32(30.))?
                .index((.., .., ..200058)),
            ops::concatenate(&taps, -1)?,
        ))
    }
}
