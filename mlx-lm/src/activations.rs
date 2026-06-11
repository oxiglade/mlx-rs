//! Shared compiled activation helpers.
//!
//! Each function wraps a `transforms::compile`-fused inner kernel in a
//! caller-owned cache so every decoder layer per token reuses one fused
//! graph instead of rebuilding it per call. The cache lives on the owning
//! module struct (e.g. `Mlp::swiglu_cache`) and is borrowed `&mut` per
//! call; mlx core's compile/encoder state is thread-local since mlx 0.31,
//! so the cache must be dropped on the thread that calls it.

// Each `as fn(...) -> ...` below coerces a zero-sized fn-item to a shared
// fn-pointer type. Without the cast every fn-item would yield a distinct
// `Compiled<F, _>` and the cache slot could not be reused. Clippy's
// trivial_casts diagnostic prints identical source/dest types, but the
// source is the fn-item ZST, not a fn-pointer.
#![allow(
    trivial_casts,
    reason = "fn-item ZST → fn-pointer coercion for shared compile cache"
)]

use std::sync::OnceLock;

use mlx_rs::{
    error::Exception,
    nn,
    ops::{sigmoid, tanh},
    transforms::compile::{
        allocate_compile_id,
        shape::{ThreeArgs, TwoArgs},
        CallMut, Compile, Compiled,
    },
    Array,
};

/// Process-wide cache ids — one slot per logical activation, shared across
/// every cache instance. Lets MLX's `compiler_cache` reuse a single
/// compiled Metal kernel across all decoder layers instead of JIT-compiling
/// one redundant copy per layer.
fn swiglu_id() -> usize {
    static ID: OnceLock<usize> = OnceLock::new();
    *ID.get_or_init(allocate_compile_id)
}
fn attention_gate_id() -> usize {
    static ID: OnceLock<usize> = OnceLock::new();
    *ID.get_or_init(allocate_compile_id)
}
fn geglu_id() -> usize {
    static ID: OnceLock<usize> = OnceLock::new();
    *ID.get_or_init(allocate_compile_id)
}
fn logit_softcap_id() -> usize {
    static ID: OnceLock<usize> = OnceLock::new();
    *ID.get_or_init(allocate_compile_id)
}
fn residual_add_scale_id() -> usize {
    static ID: OnceLock<usize> = OnceLock::new();
    *ID.get_or_init(allocate_compile_id)
}

pub type SwigluCompiled = Compiled<
    fn((&Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
    TwoArgs,
>;

pub type AttentionGateCompiled = Compiled<
    fn((&Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
    TwoArgs,
>;

pub type GegluCompiled = Compiled<
    fn((&Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
    TwoArgs,
>;

pub type LogitSoftcapCompiled = Compiled<
    fn((&Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
    TwoArgs,
>;

pub type ResidualAddScaleCompiled = Compiled<
    fn((&Array, &Array, &Array)) -> Result<Array, Exception>,
    Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
    ThreeArgs,
>;

/// Cached compiled-graph slot for [`swiglu`]. Owned by the calling module
/// (typically a per-layer `Mlp::swiglu_cache`). Initialised lazily on first
/// call. Custom `Debug` is opaque — the inner `Compiled` wraps a
/// `Box<dyn FnMut>` that has no `Debug` impl.
#[derive(Default)]
pub struct SwigluCache(pub Option<SwigluCompiled>);

#[derive(Default)]
pub struct AttentionGateCache(pub Option<AttentionGateCompiled>);

/// Cached compiled-graph slot for [`geglu`].
#[derive(Default)]
pub struct GegluCache(pub Option<GegluCompiled>);

/// Cached compiled-graph slot for [`logit_softcap`].
#[derive(Default)]
pub struct LogitSoftcapCache(pub Option<LogitSoftcapCompiled>);

/// Cached compiled-graph slot for [`residual_add_scale`].
#[derive(Default)]
pub struct ResidualAddScaleCache(pub Option<ResidualAddScaleCompiled>);

impl std::fmt::Debug for SwigluCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SwigluCache")
            .field("filled", &self.0.is_some())
            .finish()
    }
}

impl std::fmt::Debug for AttentionGateCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttentionGateCache")
            .field("filled", &self.0.is_some())
            .finish()
    }
}

impl std::fmt::Debug for GegluCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GegluCache")
            .field("filled", &self.0.is_some())
            .finish()
    }
}

impl std::fmt::Debug for LogitSoftcapCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogitSoftcapCache")
            .field("filled", &self.0.is_some())
            .finish()
    }
}

impl std::fmt::Debug for ResidualAddScaleCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidualAddScaleCache")
            .field("filled", &self.0.is_some())
            .finish()
    }
}

/// `silu(gate) * x` as a compile-fused kernel. Caller passes a
/// `&mut SwigluCache` owned by the surrounding module; the compiled graph
/// is built on first call and reused thereafter.
pub fn swiglu(cache: &mut SwigluCache, gate: &Array, x: &Array) -> Result<Array, Exception> {
    let compiled = cache.0.get_or_insert_with(|| {
        Compile::<(&Array, &Array), Array, Exception>::compile_with_id(
            swiglu_inner as fn((&Array, &Array)) -> Result<Array, Exception>,
            swiglu_id(),
            true,
        )
    });
    CallMut::call_mut(compiled, (gate, x))
}

fn swiglu_inner((gate, x): (&Array, &Array)) -> Result<Array, Exception> {
    nn::silu(gate)?.multiply(x)
}

/// `output * sigmoid(gate)` — trailing fused op of Qwen3.5 full-attention.
/// Caller-owned cache, same shape as [`swiglu`].
pub fn attention_gate(
    cache: &mut AttentionGateCache,
    output: &Array,
    gate: &Array,
) -> Result<Array, Exception> {
    let compiled = cache.0.get_or_insert_with(|| {
        Compile::<(&Array, &Array), Array, Exception>::compile_with_id(
            attention_gate_inner as fn((&Array, &Array)) -> Result<Array, Exception>,
            attention_gate_id(),
            true,
        )
    });
    CallMut::call_mut(compiled, (output, gate))
}

fn attention_gate_inner((output, gate): (&Array, &Array)) -> Result<Array, Exception> {
    sigmoid(gate)?.multiply(output)
}

/// `gelu_approximate(gate) * up` as a compile-fused kernel — Gemma's GeGLU
/// MLP activation. Caller-owned cache, same shape as [`swiglu`].
pub fn geglu(cache: &mut GegluCache, gate: &Array, up: &Array) -> Result<Array, Exception> {
    let compiled = cache.0.get_or_insert_with(|| {
        Compile::<(&Array, &Array), Array, Exception>::compile_with_id(
            geglu_inner as fn((&Array, &Array)) -> Result<Array, Exception>,
            geglu_id(),
            true,
        )
    });
    CallMut::call_mut(compiled, (gate, up))
}

fn geglu_inner((gate, up): (&Array, &Array)) -> Result<Array, Exception> {
    gelu_approximate_in_dtype(gate)?.multiply(up)
}

/// Dtype-preserving gelu approximation. `mlx_rs::nn::gelu_approximate`
/// builds its constants as `array!(0.5_f32)` etc., promoting bf16/f16
/// inputs to f32 — and that f32 then cascades through the whole MoE
/// forward (residual → next layer → expert `gather_qmm` in f32), which is
/// ~4× slower at decode. Staging the scalars into the input dtype keeps
/// the graph in-place.
pub(crate) fn gelu_approximate_in_dtype(x: &Array) -> Result<Array, Exception> {
    let dt = x.dtype();
    let cast = |c: f32| -> Result<Array, Exception> { Array::from_f32(c).as_dtype(dt) };
    let half = cast(0.5)?;
    let one = cast(1.0)?;
    let sqrt_2_over_pi = cast((2.0_f32 / std::f32::consts::PI).sqrt())?;
    let k = cast(0.044715)?;
    let x3 = x.multiply(x)?.multiply(x)?;
    let inner = x.add(&k.multiply(&x3)?)?;
    let scaled = sqrt_2_over_pi.multiply(&inner)?;
    let t = tanh(&scaled)?;
    half.multiply(x)?.multiply(&one.add(&t)?)
}

/// `tanh(x / cap) * cap` — Gemma final-logit soft-capping. `cap` is passed
/// as a 0-d `Array` so the compiled graph stays stable across calls.
pub fn logit_softcap(
    cache: &mut LogitSoftcapCache,
    x: &Array,
    cap: &Array,
) -> Result<Array, Exception> {
    let compiled = cache.0.get_or_insert_with(|| {
        Compile::<(&Array, &Array), Array, Exception>::compile_with_id(
            logit_softcap_inner as fn((&Array, &Array)) -> Result<Array, Exception>,
            logit_softcap_id(),
            true,
        )
    });
    CallMut::call_mut(compiled, (x, cap))
}

fn logit_softcap_inner((x, cap): (&Array, &Array)) -> Result<Array, Exception> {
    tanh(&x.divide(cap)?)?.multiply(cap)
}

/// `(residual + ff_out) * layer_scalar` as one compile-fused kernel —
/// Gemma's per-layer epilogue on bf16/fp32. `layer_scalar` is a 0-d/`[1]`
/// `Array`. (fp16 callers take the unfused `clip_residual` path instead.)
pub fn residual_add_scale(
    cache: &mut ResidualAddScaleCache,
    residual: &Array,
    ff_out: &Array,
    layer_scalar: &Array,
) -> Result<Array, Exception> {
    let compiled = cache.0.get_or_insert_with(|| {
        Compile::<(&Array, &Array, &Array), Array, Exception>::compile_with_id(
            residual_add_scale_inner as fn((&Array, &Array, &Array)) -> Result<Array, Exception>,
            residual_add_scale_id(),
            true,
        )
    });
    CallMut::call_mut(compiled, (residual, ff_out, layer_scalar))
}

fn residual_add_scale_inner(
    (residual, ff_out, layer_scalar): (&Array, &Array, &Array),
) -> Result<Array, Exception> {
    residual.add(ff_out)?.multiply(layer_scalar)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]
    #![allow(clippy::missing_assert_message, reason = "test code")]
    use super::*;

    #[test]
    fn swiglu_matches_manual_silu_multiply() {
        let gate = Array::from_slice(&[1.0_f32, -1.0, 0.5, 2.0], &[2, 2]);
        let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let mut cache = SwigluCache::default();
        let fused = swiglu(&mut cache, &gate, &x).unwrap();
        let manual = nn::silu(&gate).unwrap().multiply(&x).unwrap();
        let max = fused
            .subtract(&manual)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(max < 1e-5, "fused vs manual swiglu diverge: {max}");
    }

    /// Both activations compile the same `(&Array, &Array)` signature.
    /// Distinct compile ids must keep their graphs separate even when
    /// invoked in sequence with the same shapes — a TypeId-keyed cache
    /// would make `attention_gate` return `sigmoid(output) * gate`
    /// after `swiglu` warmed the slot.
    #[test]
    fn attention_gate_after_swiglu_does_not_collide() {
        let output = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let gate = Array::from_slice(&[0.0_f32, 1.0, -1.0, 2.0], &[2, 2]);

        let mut swiglu_cache = SwigluCache::default();
        let _ = swiglu(&mut swiglu_cache, &gate, &output).unwrap();

        let mut ag_cache = AttentionGateCache::default();
        let fused = attention_gate(&mut ag_cache, &output, &gate).unwrap();
        let manual = sigmoid(&gate).unwrap().multiply(&output).unwrap();
        let max = fused
            .subtract(&manual)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(max < 1e-5, "attention_gate diverged after swiglu: {max}");
    }

    fn max_abs_diff(a: &Array, b: &Array) -> f32 {
        a.subtract(b)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>()
    }

    #[test]
    fn geglu_matches_manual_gelu_multiply() {
        let gate = Array::from_slice(&[1.0_f32, -1.0, 0.5, 2.0], &[2, 2]);
        let up = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let mut cache = GegluCache::default();
        let fused = geglu(&mut cache, &gate, &up).unwrap();
        let manual = nn::gelu_approximate(&gate).unwrap().multiply(&up).unwrap();
        assert!(
            max_abs_diff(&fused, &manual) < 1e-5,
            "fused vs manual geglu diverge"
        );
    }

    #[test]
    fn logit_softcap_matches_manual() {
        let x = Array::from_slice(&[-60.0_f32, -5.0, 0.0, 5.0, 60.0, 100.0], &[2, 3]);
        let cap = Array::from_f32(30.0);
        let mut cache = LogitSoftcapCache::default();
        let fused = logit_softcap(&mut cache, &x, &cap).unwrap();
        let manual = tanh(x.divide(&cap).unwrap())
            .unwrap()
            .multiply(&cap)
            .unwrap();
        assert!(
            max_abs_diff(&fused, &manual) < 1e-4,
            "fused vs manual logit_softcap diverge"
        );
        // Soft-cap clamps magnitude below the cap.
        let max_mag = fused.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max_mag < 30.0, "softcap did not bound magnitude: {max_mag}");
    }

    #[test]
    fn residual_add_scale_matches_manual() {
        let h = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let ff = Array::from_slice(&[0.5_f32, -0.5, 1.0, -1.0], &[2, 2]);
        let scalar = Array::from_slice(&[2.0_f32], &[1]);
        let mut cache = ResidualAddScaleCache::default();
        let fused = residual_add_scale(&mut cache, &h, &ff, &scalar).unwrap();
        let manual = h.add(&ff).unwrap().multiply(&scalar).unwrap();
        assert!(
            max_abs_diff(&fused, &manual) < 1e-5,
            "fused vs manual residual_add_scale diverge"
        );
    }
}
