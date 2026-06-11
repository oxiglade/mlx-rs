//! Gated DeltaNet (Mamba2-style) recurrent operator used by every
//! `linear_attention` layer of Qwen3.5. Provides both a pure-ops scan
//! ([`gated_delta_update_ops`]) and a Metal kernel fast path
//! ([`gated_delta_update_metal`]).
//!
//! Shapes:
//! - `q`, `k`: `[B, T, Hk, Dk]`
//! - `v`: `[B, T, Hv, Dv]`
//! - `a`, `b`, `dt_bias`, `A_log`: `[B, T, Hv]` (the projector outputs) and
//!   `[Hv]` (the learned per-head params).
//! - returned `y`: `[B, T, Hv, Dv]`
//! - returned `state`: `[B, Hv, Dv, Dk]`

// See `mlx_lm::activations` for the rationale on the fn-item-to-fn-pointer
// coercion required at the `Compile::compile_with_id` call site.
#![allow(
    trivial_casts,
    reason = "fn-item ZST → fn-pointer coercion for shared compile cache"
)]

use mlx_rs::{
    error::Exception,
    fast::{metal_kernel, MetalKernel, MetalKernelConfig},
    nn,
    ops::{
        exp as exp_op, expand_dims, indexing::take_axis, r#where, repeat_axis, reshape, sigmoid,
        stack_axis, zeros, zeros_dtype,
    },
    transforms::compile::{shape::ThreeArgs, CallMut, Compile, Compiled},
    Array, Dtype, Stream,
};

use crate::error::Error;

/// Apple-Silicon SIMD lane width (`simd_sum` quad-pair). The GDN kernel
/// reduces across the key dimension via `simd_sum`, so `Dk` must be a
/// multiple of this constant for the kernel path to be usable.
const SIMD_WIDTH: i32 = 32;

pub(crate) type ComputeGCompiled = Compiled<
    fn((&Array, &Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
    ThreeArgs,
>;

/// Cached compiled-graph slot for [`compute_g`]. Owned by the calling
/// [`super::gated_delta_block::GatedDeltaNet`].
#[derive(Default)]
pub(crate) struct ComputeGCache(pub(crate) Option<ComputeGCompiled>);

impl std::fmt::Debug for ComputeGCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputeGCache")
            .field("filled", &self.0.is_some())
            .finish()
    }
}

/// Compute the per-step decay `g = exp(-exp(A_log) * softplus(a + dt_bias))`.
///
/// Caller passes a `&mut ComputeGCache` owned by the surrounding
/// block; the closure is shape-compiled and reused across decode steps.
pub(crate) fn compute_g(
    cache: &mut ComputeGCache,
    a_log: &Array,
    a: &Array,
    dt_bias: &Array,
) -> Result<Array, Exception> {
    let compiled = cache.0.get_or_insert_with(|| {
        Compile::<(&Array, &Array, &Array), Array, Exception>::compile(
            compute_g_inner as fn((&Array, &Array, &Array)) -> Result<Array, Exception>,
            true,
        )
    });
    CallMut::call_mut(compiled, (a_log, a, dt_bias))
}

fn compute_g_inner((a_log, a, dt_bias): (&Array, &Array, &Array)) -> Result<Array, Exception> {
    let a_log_f32 = a_log.as_dtype(Dtype::Float32)?;
    let inner = a.add(dt_bias)?;
    let s = nn::softplus(&inner)?;
    exp_op(a_log_f32)?.negative()?.multiply(&s)?.exp()
}

/// Run one recurrent step of the gated delta SSM.
///
/// Inputs:
/// - `q_t`, `k_t`: `[B, Hv, Dk]` (with key heads already broadcast to Hv).
/// - `v_t`: `[B, Hv, Dv]`.
/// - `g_t`: `[B, Hv]` (scalar decay) or `[B, Hv, Dk]` (vectorised decay).
/// - `beta_t`: `[B, Hv]`.
/// - `state`: `[B, Hv, Dv, Dk]`.
/// - `mask_t`: optional `[B]` bool — when false at element `bi`, the new
///   state for that batch element is replaced with the previous state and `y`
///   is zeroed.
///
/// Returns `(y, new_state)` with `y` shaped `[B, Hv, Dv]`.
pub(crate) fn step_ops(
    q_t: &Array,
    k_t: &Array,
    v_t: &Array,
    g_t: &Array,
    beta_t: &Array,
    state: &Array,
    mask_t: Option<&Array>,
) -> Result<(Array, Array), Error> {
    let old_state = state.clone();

    let decay = match g_t.ndim() {
        2 => {
            // [B, Hv] -> [B, Hv, 1, 1]
            let e = expand_dims(g_t, 2)?;
            expand_dims(&e, 3)?
        }
        3 => {
            // [B, Hv, Dk] -> [B, Hv, 1, Dk]
            expand_dims(g_t, 2)?
        }
        n => return Err(Error::shape(format!("step_ops: unsupported g ndim {n}"))),
    };
    let state = state.multiply(&decay)?;
    // k[..., None, :] -> [B, Hv, 1, Dk]
    let k_b = expand_dims(k_t, 2)?;
    let kv_mem = state.multiply(&k_b)?.sum_axes(&[-1], false)?; // [B, Hv, Dv]
                                                                // beta[..., None] -> [B, Hv, 1]
    let beta_b = expand_dims(beta_t, 2)?;
    let delta = v_t.subtract(&kv_mem)?.multiply(&beta_b)?; // [B, Hv, Dv]
                                                           // delta[..., None] -> [B, Hv, Dv, 1]
    let delta_b = expand_dims(&delta, 3)?;
    let state = state.add(&k_b.multiply(&delta_b)?)?;

    // y = (state * q[..., None, :]).sum(-1) -> [B, Hv, Dv]
    let q_b = expand_dims(q_t, 2)?;
    let y = state.multiply(&q_b)?.sum_axes(&[-1], false)?;

    let (state, y) = if let Some(mask_t) = mask_t {
        // mask_t: [B] -> for state [B, 1, 1, 1], for y [B, 1, 1]
        let m_state = expand_dims(mask_t, 1)?;
        let m_state = expand_dims(&m_state, 2)?;
        let m_state = expand_dims(&m_state, 3)?;
        let m_y = expand_dims(mask_t, 1)?;
        let m_y = expand_dims(&m_y, 2)?;
        let state_dtype = state.dtype();
        let new_state = r#where(&m_state, &state, &old_state)?;
        let zero = zeros_dtype(y.shape(), y.dtype())?;
        let _ = state_dtype;
        let new_y = r#where(&m_y, &y, &zero)?;
        (new_state, new_y)
    } else {
        (state, y)
    };
    let y = y.as_dtype(q_t.dtype())?;
    Ok((y, state))
}

/// Run the full sequential scan over `T` steps using the ops-only kernel.
///
/// `state` is optional — `None` initialises a zero state of dtype
/// float32. Returns `(y, final_state)`.
#[allow(
    clippy::too_many_arguments,
    reason = "GDN operator: every input is load-bearing"
)]
pub(crate) fn gated_delta_update_ops(
    compute_g_cache: &mut ComputeGCache,
    q: &Array,
    k: &Array,
    v: &Array,
    a: &Array,
    b: &Array,
    a_log: &Array,
    dt_bias: &Array,
    state: Option<&Array>,
    mask: Option<&Array>,
) -> Result<(Array, Array), Error> {
    let q_shape = q.shape();
    let v_shape = v.shape();
    if q_shape.len() != 4 || v_shape.len() != 4 {
        return Err(Error::shape(
            "gated_delta_update_ops: q/v must be 4-D [B, T, H, D]",
        ));
    }
    let batch = q_shape[0];
    let time = q_shape[1];
    let hk = q_shape[2];
    let dk = q_shape[3];
    let hv = v_shape[2];
    let dv = v_shape[3];

    let beta = sigmoid(b)?;
    let g = compute_g(compute_g_cache, a_log, a, dt_bias)?;

    let owned_state;
    let state = if let Some(s) = state {
        s.clone()
    } else {
        owned_state = zeros::<f32>(&[batch, hv, dv, dk])?;
        owned_state
    };

    if hv % hk != 0 {
        return Err(Error::shape(format!(
            "gated_delta_update_ops: Hv ({hv}) must be divisible by Hk ({hk})"
        )));
    }
    let (q_eff, k_eff) = if hv == hk {
        (q.clone(), k.clone())
    } else {
        let rep = hv / hk;
        let q_r = repeat_axis::<f32>(q.clone(), rep, -2)?;
        let k_r = repeat_axis::<f32>(k.clone(), rep, -2)?;
        (q_r, k_r)
    };

    let mut state = state;
    let mut ys = Vec::with_capacity(time as usize);
    // Build the T index arrays once. `slice_t` previously rebuilt one per
    // call (6× per step), turning the ops scan into 6T host→device
    // allocations per forward.
    let idxs: Vec<Array> = (0..time).map(|t| Array::from_slice(&[t], &[1])).collect();
    for idx in &idxs {
        let q_t = slice_t(&q_eff, idx)?;
        let k_t = slice_t(&k_eff, idx)?;
        let v_t = slice_t(v, idx)?;
        let g_t = slice_t(&g, idx)?;
        let beta_t = slice_t(&beta, idx)?;
        let mask_t = match mask {
            Some(m) => Some(slice_t(m, idx)?),
            None => None,
        };
        let (y_t, new_state) = step_ops(&q_t, &k_t, &v_t, &g_t, &beta_t, &state, mask_t.as_ref())?;
        state = new_state;
        ys.push(y_t);
    }
    let y = stack_axis(&ys, 1)?;
    Ok((y, state))
}

/// Compile a fresh GDN scan kernel.
///
/// The caller is expected to cache the returned handle for the lifetime of
/// the block — see [`gated_delta_update_metal`] for why per-call recreation
/// breaks chained launches.
pub(crate) fn make_gated_delta_kernel() -> Result<MetalKernel, Error> {
    Ok(metal_kernel(
        // Bump the `_vN` suffix on every change to GATED_DELTA_STEP_SOURCE
        // so the mlx kernel-name cache doesn't serve a stale binary.
        "qwen3_5_gated_delta_step_v2",
        &["q", "k", "v", "g", "beta", "state_in"],
        &["y", "state_out"],
        GATED_DELTA_STEP_SOURCE,
        "",
        true,
        false,
    )?)
}

/// Metal-kernel fast path for the gated-delta T-step scan, plumbed
/// through `mlx_rs::fast::metal_kernel`.
#[allow(
    clippy::too_many_arguments,
    reason = "GDN operator: every input is load-bearing"
)]
pub(crate) fn gated_delta_update_metal(
    kernel: &MetalKernel,
    compute_g_cache: &mut ComputeGCache,
    q: &Array,
    k: &Array,
    v: &Array,
    a: &Array,
    b: &Array,
    a_log: &Array,
    dt_bias: &Array,
    state: Option<&Array>,
) -> Result<(Array, Array), Error> {
    let q_shape = q.shape();
    let v_shape = v.shape();
    if q_shape.len() != 4 || v_shape.len() != 4 {
        return Err(Error::shape("gated_delta_update_metal: q/v must be 4-D"));
    }
    let batch = q_shape[0];
    let time = q_shape[1];
    let hk = q_shape[2];
    let dk = q_shape[3];
    let hv = v_shape[2];
    let dv = v_shape[3];

    if dk % SIMD_WIDTH != 0 {
        return Err(Error::shape(format!(
            "gated_delta_update_metal: Dk={dk} must be a multiple of {SIMD_WIDTH}"
        )));
    }

    let beta = sigmoid(b)?;
    let g = compute_g(compute_g_cache, a_log, a, dt_bias)?;
    let owned_state;
    let state_in: &Array = if let Some(s) = state {
        s
    } else {
        owned_state = zeros::<f32>(&[batch, hv, dv, dk])?;
        &owned_state
    };

    let input_dtype = q.dtype();
    let state_dtype = state_in.dtype();
    let config = MetalKernelConfig::new()
        .add_output([batch, time, hv, dv], input_dtype)
        .add_output([batch, hv, dv, dk], state_dtype)
        .grid(SIMD_WIDTH, dv, batch * hv)
        .thread_group(SIMD_WIDTH, 4, 1)
        .add_template("InT", input_dtype)?
        .add_template("StT", state_dtype)?
        .add_template("Dk", dk)?
        .add_template("Dv", dv)?
        .add_template("Hk", hk)?
        .add_template("Hv", hv)?
        .add_template("T", time)?;

    let [y, state_out] =
        kernel.apply_array::<2>(&[q, k, v, &g, &beta, state_in], config, Stream::default())?;
    Ok((y, state_out))
}

const GATED_DELTA_STEP_SOURCE: &str = r#"
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    // dispatch_threads rounds y up to the threadgroup multiple (4);
    // skip overshoot when Dv % 4 != 0.
    if (dv_idx >= Dv) return;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }

    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g_[hv_idx];
        kv_mem += state[i] * static_cast<float>(k_[s_idx]);
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * beta_[hv_idx];

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
        out += state[i] * static_cast<float>(q_[s_idx]);
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }

      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }

    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
"#;

/// Index the time axis with a pre-built length-1 index Array and squeeze
/// the axis out. `x[:, t]` in numpy / mlx. The caller hoists the index
/// Array out of any per-step loop — see [`gated_delta_update_ops`] for
/// why per-call rebuild costs add up.
fn slice_t(x: &Array, idx: &Array) -> Result<Array, Error> {
    // Length-1 index preserves the axis at size 1 so `take_axis` plus
    // explicit reshape squeezes it deterministically across mlx builds
    // (a 0-D scalar index drops the axis in some builds but not others).
    let y = take_axis(x, idx, 1)?;
    let shape = y.shape();
    if shape[1] != 1 {
        return Err(Error::shape(format!(
            "slice_t: expected axis 1 to be size 1 after take_axis, got shape {shape:?}"
        )));
    }
    let new_shape: Vec<i32> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, s)| if i == 1 { None } else { Some(*s) })
        .collect();
    Ok(reshape(&y, &new_shape)?)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    #![allow(clippy::print_stdout, reason = "test code")]
    #![allow(clippy::print_stderr, reason = "test code")]
    use super::*;
    use mlx_rs::{random::uniform, transforms::eval};

    fn rand(shape: &[i32]) -> Array {
        uniform::<_, f32>(0.0, 1.0, shape, None).unwrap()
    }

    fn flatten_f32(arr: &Array) -> Vec<f32> {
        let total: i32 = arr.shape().iter().product();
        let flat = reshape(arr, &[total]).unwrap();
        let zero = Array::from_f32(0.0);
        let evald = flat.add(&zero).unwrap();
        eval([&evald]).unwrap();
        evald.as_slice::<f32>().to_vec()
    }

    #[test]
    fn compute_g_shape_and_range() {
        let hv = 4;
        let a_log = rand(&[hv]);
        let dt_bias = rand(&[hv]);
        let a = rand(&[2, 3, hv]);
        let mut cache = ComputeGCache::default();
        let g = compute_g(&mut cache, &a_log, &a, &dt_bias).unwrap();
        assert_eq!(g.shape(), &[2, 3, hv]);
        // g = exp(-exp(A_log_pos) * softplus(...)) <= 1.0 strictly.
        let max = g.max(None).unwrap().item::<f32>();
        assert!(max <= 1.0 + 1e-5, "g max {max} should be <= 1");
        let min = g.min(None).unwrap().item::<f32>();
        assert!(min >= 0.0, "g min {min} should be >= 0");
    }

    #[test]
    fn zero_state_input_and_random_inputs_have_expected_shapes() {
        let (b, t, hk, dk) = (1, 3, 2, 4);
        let (hv, dv) = (4, 4);
        let q = rand(&[b, t, hk, dk]);
        let k = rand(&[b, t, hk, dk]);
        let v = rand(&[b, t, hv, dv]);
        let a = rand(&[b, t, hv]);
        let bb = rand(&[b, t, hv]);
        let a_log = rand(&[hv]);
        let dt_bias = rand(&[hv]);

        let mut cache = ComputeGCache::default();
        let (y, state) = gated_delta_update_ops(
            &mut cache, &q, &k, &v, &a, &bb, &a_log, &dt_bias, None, None,
        )
        .unwrap();
        assert_eq!(y.shape(), &[b, t, hv, dv]);
        assert_eq!(state.shape(), &[b, hv, dv, dk]);
    }

    #[test]
    fn single_step_with_zero_state_and_zero_decay() {
        // When state starts at zero and `decay = g` is irrelevant for the
        // first step, the produced y at t=0 reduces to:
        //   y = (state' * q[None, :]).sum(-1)
        // with state' = k[..., None, :] * delta[..., None] and
        // delta = v * beta (since kv_mem = 0).
        let b = 1;
        let t = 1;
        let hk = 1;
        let dk = 2;
        let hv = 1;
        let dv = 2;
        let q = Array::from_slice(&[1.0_f32, 1.0], &[b, t, hk, dk]);
        let k = Array::from_slice(&[1.0_f32, 0.0], &[b, t, hk, dk]);
        let v = Array::from_slice(&[0.5_f32, 0.25], &[b, t, hv, dv]);
        let a = Array::from_slice(&[0.0_f32], &[b, t, hv]); // softplus(0) = ln2
        let bb = Array::from_slice(&[0.0_f32], &[b, t, hv]); // sigmoid(0) = 0.5
        let a_log = Array::from_slice(&[-30.0_f32], &[hv]); // exp(-30) ≈ 0 -> g ≈ 1
        let dt_bias = Array::from_slice(&[0.0_f32], &[hv]);

        let mut cache = ComputeGCache::default();
        let (y, _state) = gated_delta_update_ops(
            &mut cache, &q, &k, &v, &a, &bb, &a_log, &dt_bias, None, None,
        )
        .unwrap();
        let y_flat = flatten_f32(&y);
        // Manual derivation:
        //   g ≈ 1, beta = 0.5, state starts zero -> kv_mem = 0
        //   delta = (v - 0) * 0.5 = [0.25, 0.125]
        //   state += k[..., None, :] * delta[..., None]
        //     k = [1, 0], delta = [0.25, 0.125]
        //     state[Dv=0, Dk=0] = 1 * 0.25 = 0.25
        //     state[Dv=0, Dk=1] = 0 * 0.25 = 0
        //     state[Dv=1, Dk=0] = 1 * 0.125 = 0.125
        //     state[Dv=1, Dk=1] = 0 * 0.125 = 0
        //   y[Dv=0] = sum_k state[Dv=0, :] * q = 0.25*1 + 0*1 = 0.25
        //   y[Dv=1] = 0.125*1 + 0*1 = 0.125
        assert!((y_flat[0] - 0.25).abs() < 1e-5, "y[0] = {}", y_flat[0]);
        assert!((y_flat[1] - 0.125).abs() < 1e-5, "y[1] = {}", y_flat[1]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_kernel_matches_ops_chandra_shape() {
        // Chandra-q8 shapes: Hk=16, Hv=32, Dk=Dv=128, bf16. Smaller batch +
        // time so the test stays fast.
        let (b, t, hk, dk, hv, dv) = (1, 4, 16, 128, 32, 128);
        let q = rand(&[b, t, hk, dk]).as_dtype(Dtype::Bfloat16).unwrap();
        let k = rand(&[b, t, hk, dk]).as_dtype(Dtype::Bfloat16).unwrap();
        let v = rand(&[b, t, hv, dv]).as_dtype(Dtype::Bfloat16).unwrap();
        let a = rand(&[b, t, hv]).as_dtype(Dtype::Bfloat16).unwrap();
        let bb = rand(&[b, t, hv]).as_dtype(Dtype::Bfloat16).unwrap();
        let a_log = rand(&[hv]).as_dtype(Dtype::Bfloat16).unwrap();
        let dt_bias = rand(&[hv]).as_dtype(Dtype::Bfloat16).unwrap();

        let kernel = make_gated_delta_kernel().unwrap();
        let mut cache_ops = ComputeGCache::default();
        let mut cache_kern = ComputeGCache::default();
        let (y_ops, _state_ops) = gated_delta_update_ops(
            &mut cache_ops,
            &q,
            &k,
            &v,
            &a,
            &bb,
            &a_log,
            &dt_bias,
            None,
            None,
        )
        .unwrap();
        let (y_kern, _state_kern) = gated_delta_update_metal(
            &kernel,
            &mut cache_kern,
            &q,
            &k,
            &v,
            &a,
            &bb,
            &a_log,
            &dt_bias,
            None,
        )
        .unwrap();

        let y_ops_flat = flatten_f32(&y_ops);
        let y_kern_flat = flatten_f32(&y_kern);
        let mut max_y = 0.0_f32;
        for (a, b) in y_ops_flat.iter().zip(&y_kern_flat) {
            let d = (a - b).abs();
            if d > max_y {
                max_y = d;
            }
        }
        eprintln!("chandra shape max_abs(y) = {max_y}");
        // bf16 tolerance, scaled by GQA broadcast + repeated state updates.
        assert!(
            max_y < 0.5,
            "kernel diverged on chandra shape: max_abs={max_y}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_kernel_matches_ops_path() {
        // Tiny shapes that satisfy the kernel's `Dk % 32 == 0` requirement
        // (Dk=32, Dv=32, Hk=Hv=1, B=1, T=4).
        let (b, t, hk, dk, hv, dv) = (1, 4, 1, 32, 1, 32);
        let q = rand(&[b, t, hk, dk]);
        let k = rand(&[b, t, hk, dk]);
        let v = rand(&[b, t, hv, dv]);
        let a = rand(&[b, t, hv]);
        let bb = rand(&[b, t, hv]);
        let a_log = rand(&[hv]);
        let dt_bias = rand(&[hv]);

        let kernel = make_gated_delta_kernel().unwrap();
        let mut cache_ops = ComputeGCache::default();
        let mut cache_kern = ComputeGCache::default();
        let (y_ops, state_ops) = gated_delta_update_ops(
            &mut cache_ops,
            &q,
            &k,
            &v,
            &a,
            &bb,
            &a_log,
            &dt_bias,
            None,
            None,
        )
        .unwrap();
        let (y_kern, state_kern) = gated_delta_update_metal(
            &kernel,
            &mut cache_kern,
            &q,
            &k,
            &v,
            &a,
            &bb,
            &a_log,
            &dt_bias,
            None,
        )
        .unwrap();
        let y_ops_flat = flatten_f32(&y_ops);
        let y_kern_flat = flatten_f32(&y_kern);
        let mut max_y = 0.0_f32;
        for (a, b) in y_ops_flat.iter().zip(&y_kern_flat) {
            let d = (a - b).abs();
            if d > max_y {
                max_y = d;
            }
        }
        assert!(max_y < 1e-3, "kernel/y vs ops/y max_abs={max_y}");
        let s_ops_flat = flatten_f32(&state_ops);
        let s_kern_flat = flatten_f32(&state_kern);
        let mut max_s = 0.0_f32;
        for (a, b) in s_ops_flat.iter().zip(&s_kern_flat) {
            let d = (a - b).abs();
            if d > max_s {
                max_s = d;
            }
        }
        assert!(max_s < 1e-3, "kernel/state vs ops/state max_abs={max_s}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn chained_kernel_matches_ops_24_recurrent_steps() {
        let (b, t, hk, dk, hv, dv) = (1, 4, 1, 32, 1, 32);
        let layers = 24;
        let q = rand(&[b, t, hk, dk]);
        let k = rand(&[b, t, hk, dk]);
        let v = rand(&[b, t, hv, dv]);
        let a = rand(&[b, t, hv]);
        let bb = rand(&[b, t, hv]);
        let a_log = rand(&[hv]);
        let dt_bias = rand(&[hv]);
        let kernel = make_gated_delta_kernel().unwrap();

        let mut state_kern: Option<Array> = None;
        let mut state_ops: Option<Array> = None;
        let mut cache_kern = ComputeGCache::default();
        let mut cache_ops = ComputeGCache::default();
        for _ in 0..layers {
            let (_, sk) = gated_delta_update_metal(
                &kernel,
                &mut cache_kern,
                &q,
                &k,
                &v,
                &a,
                &bb,
                &a_log,
                &dt_bias,
                state_kern.as_ref(),
            )
            .unwrap();
            let (_, so) = gated_delta_update_ops(
                &mut cache_ops,
                &q,
                &k,
                &v,
                &a,
                &bb,
                &a_log,
                &dt_bias,
                state_ops.as_ref(),
                None,
            )
            .unwrap();
            state_kern = Some(sk);
            state_ops = Some(so);
        }
        let sk = state_kern.unwrap();
        let so = state_ops.unwrap();
        let dim: i32 = sk.shape().iter().product();
        let fk = reshape(&sk, &[dim]).unwrap();
        let fo = reshape(&so, &[dim]).unwrap();
        eval([&fk, &fo]).unwrap();
        let vk: Vec<f32> = fk.as_slice::<f32>().to_vec();
        let vo: Vec<f32> = fo.as_slice::<f32>().to_vec();
        let max = vk
            .iter()
            .zip(&vo)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max < 1e-3,
            "kernel/state vs ops/state after {layers} chained calls max_abs={max}"
        );
    }

    #[test]
    fn mask_zeros_y_and_freezes_state() {
        let b = 1;
        let t = 2;
        let hk = 1;
        let dk = 2;
        let hv = 1;
        let dv = 2;
        let q = rand(&[b, t, hk, dk]);
        let k = rand(&[b, t, hk, dk]);
        let v = rand(&[b, t, hv, dv]);
        let a = rand(&[b, t, hv]);
        let bb = rand(&[b, t, hv]);
        let a_log = rand(&[hv]);
        let dt_bias = rand(&[hv]);

        let mask = Array::from_slice(&[true, false], &[b, t]);
        let mut cache = ComputeGCache::default();
        let (y, _) = gated_delta_update_ops(
            &mut cache,
            &q,
            &k,
            &v,
            &a,
            &bb,
            &a_log,
            &dt_bias,
            None,
            Some(&mask),
        )
        .unwrap();
        assert_eq!(y.shape(), &[b, t, hv, dv]);
        let y_flat = flatten_f32(&y);
        // y[:, t=1] must be zero.
        let off = (hv * dv) as usize;
        for i in 0..(hv * dv) as usize {
            let v = y_flat[off + i];
            assert!(
                v.abs() < 1e-6,
                "y[t=1][{i}] = {v} should be zero under mask"
            );
        }
    }
}
