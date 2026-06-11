//! Qwen3.6-MoE sparse FFN over the qwen3.5 hybrid GDN + full-attention
//! spine. Reuses the dense decoder skeleton via
//! `Qwen35Model<Qwen35MoeBlock>`; only the FFN differs.
//!
//! `Qwen35MoeBlock` is DeepSeek-style shared+routed MoE: routed experts
//! (silu-gated `SplitSwitchFfn`), one dense shared expert (`SwigluMlp`)
//! with a sigmoid gate, plus a linear router. The loader honours
//! per-tensor quant overrides (`mlp.gate` + `mlp.shared_expert_gate`
//! ship at 8-bit even when the body is 4-bit).

use std::path::Path;
use std::sync::OnceLock;

use mlx_rs::builder::Builder;
use mlx_rs::fast::MetalKernel;
use mlx_rs::macros::{ModuleParameters, Quantizable};
use mlx_rs::module::{Module, ModuleParameters};
use mlx_rs::nn;
use mlx_rs::nn::sigmoid;
use mlx_rs::quantization::{MaybeQuantized, Quantizable as _};
use mlx_rs::transforms::eval_params;
use mlx_rs::Array;

use super::config::TextConfig;
use super::layer::Qwen35Model;
use super::weights::{bucket_key, load_sanitized_weights, Bucketed};
use crate::config::ModelConfig as Config;
use crate::error::Error;
use crate::loader::apply_post_load_memory_policy;
use crate::nn::router_topk::{make_router_topk_kernel, router_topk};
use crate::nn::switch::{SplitSwitchFfn, SwigluActivation};
use crate::nn::SwigluMlp;
use crate::quantization::{requantise_linear, QuantizationConfig};
use crate::qwen3_5::text::config::ModelConfig;

/// Process-wide cached router top-k kernel handle, shared across every
/// MoE layer (mirrors the GDN kernel accessor).
fn router_topk_kernel() -> Result<&'static MetalKernel, Error> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(k) = KERNEL.get() {
        return Ok(k);
    }
    let built = make_router_topk_kernel()?;
    let _ = KERNEL.set(built);
    Ok(KERNEL.get().expect("just set"))
}

/// MoE model alias: the shared decoder skeleton with the MoE FFN.
pub type Qwen35MoeModel = Qwen35Model<Qwen35MoeBlock>;

/// Sparse MoE FFN block.
///
/// Forward: `shared = shared_expert(x) * sigmoid(shared_gate(x))`;
/// `probs = softmax(gate(x))`; topk + renorm; `routed = Σ w·expert`;
/// `out = shared + routed`.
#[derive(Debug, ModuleParameters, Quantizable)]
pub struct Qwen35MoeBlock {
    /// Linear router `[num_experts, hidden]`; 8-bit on every shipped
    /// Qwen3.6-MoE checkpoint.
    #[quantizable]
    #[param]
    pub gate: MaybeQuantized<nn::Linear>,

    /// Routed experts. Field name matches the HF path
    /// (`mlp.switch_mlp.{gate,up,down}_proj.*`).
    #[quantizable]
    #[param]
    pub switch_mlp: SplitSwitchFfn<SwigluActivation>,

    /// Always-on dense shared expert.
    #[quantizable]
    #[param]
    pub shared_expert: SwigluMlp,

    /// Scalar sigmoid gate on the shared-expert output. `[1, hidden]`.
    #[quantizable]
    #[param]
    pub shared_expert_gate: MaybeQuantized<nn::Linear>,

    num_experts: i32,
    num_experts_per_tok: i32,
}

impl Qwen35MoeBlock {
    pub fn new(
        hidden_size: i32,
        moe_intermediate_size: i32,
        shared_expert_intermediate_size: i32,
        num_experts: i32,
        num_experts_per_tok: i32,
    ) -> Result<Self, Error> {
        let gate = nn::LinearBuilder::new(hidden_size, num_experts)
            .bias(false)
            .build()?;
        let switch_mlp = SplitSwitchFfn::<SwigluActivation>::new(
            hidden_size,
            moe_intermediate_size,
            num_experts,
            num_experts_per_tok,
            false,
        )?;
        let shared_expert = SwigluMlp::new(hidden_size, shared_expert_intermediate_size, false)?;
        let shared_expert_gate = nn::LinearBuilder::new(hidden_size, 1).bias(false).build()?;
        Ok(Self {
            gate: MaybeQuantized::Original(gate),
            switch_mlp,
            shared_expert,
            shared_expert_gate: MaybeQuantized::Original(shared_expert_gate),
            num_experts,
            num_experts_per_tok,
        })
    }
}

impl Module<&Array> for Qwen35MoeBlock {
    type Output = Array;
    type Error = Error;

    fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        // Shared branch.
        let shared = self.shared_expert.forward(x)?;
        let sg = self.shared_expert_gate.forward(x)?;
        let shared = shared.multiply(&sigmoid(&sg)?)?;

        // Router: fused top-k selection + renormalised softmax weights in
        // one threadgroup-local kernel (no full-vocab sort). Equivalent to
        // softmax → argpartition → take_along → renormalise.
        let logits = self.gate.forward(x)?;
        let (top_k_indices, top_k_weights) = router_topk(
            router_topk_kernel()?,
            &logits,
            self.num_experts,
            self.num_experts_per_tok,
        )?;

        let routed = self
            .switch_mlp
            .forward_with_combine(x, &top_k_indices, &top_k_weights)?;
        Ok(shared.add(&routed)?)
    }

    fn training_mode(&mut self, _mode: bool) {
        // Routed + shared FFNs are inference-only; no state to propagate.
    }
}

/// Build the MoE model with the shared `Qwen35Model::new` generic,
/// supplying a `Qwen35MoeBlock` factory per layer.
fn make_moe_language_model(cfg: &TextConfig) -> Result<Qwen35MoeModel, Error> {
    Qwen35Model::<Qwen35MoeBlock>::new(cfg.clone(), |c| {
        Qwen35MoeBlock::new(
            c.hidden_size,
            c.moe_intermediate_size,
            c.shared_expert_intermediate_size,
            c.num_experts,
            c.num_experts_per_tok,
        )
    })
}

/// Body quantise, then re-quantise the per-layer override slots
/// (`mlp.gate`, `mlp.shared_expert_gate`) at their declared bit width.
fn quantize_with_overrides(
    model: &mut Qwen35MoeModel,
    q: &QuantizationConfig,
) -> Result<(), Error> {
    let cfg = model.cfg.clone();
    let original = std::mem::replace(model, make_moe_language_model(&cfg)?);
    *model = original
        .try_into_quantized(q.group_size, q.bits)
        .map_err(Error::Exception)?;

    for (layer_idx, layer) in model.model.layers.iter_mut().enumerate() {
        let gate_path = format!("language_model.model.layers.{layer_idx}.mlp.gate");
        let (gs, bits) = q.for_path(&gate_path);
        if (gs, bits) != (q.group_size, q.bits) {
            requantise_linear(&mut layer.mlp.gate, gs, bits)?;
        }
        let sgate_path = format!("language_model.model.layers.{layer_idx}.mlp.shared_expert_gate");
        let (gs, bits) = q.for_path(&sgate_path);
        if (gs, bits) != (q.group_size, q.bits) {
            requantise_linear(&mut layer.mlp.shared_expert_gate, gs, bits)?;
        }
    }
    Ok(())
}

/// End-to-end MoE loader: build (with quant overrides), sanitise + bind
/// weights, hard-error on unbound LM keys (vision-tower keys dropped).
pub(crate) fn load_qwen3_5_moe_model(
    cfg: &Config,
    env: &ModelConfig,
    model_dir: &Path,
) -> Result<Qwen35MoeModel, Error> {
    if !env.text_config.is_moe() {
        return Err(Error::config(format!(
            "qwen3_5_moe loader: num_experts={} is not MoE",
            env.text_config.num_experts
        )));
    }

    let mut model = make_moe_language_model(&env.text_config)?;
    if let Some(q) = cfg.quantization() {
        quantize_with_overrides(&mut model, q)?;
    }

    let weights = load_sanitized_weights(model_dir)?;
    let mut leftover: Vec<String> = Vec::new();
    {
        let mut params = model.parameters_mut().flatten();
        for (k, v) in weights {
            match bucket_key(k) {
                Bucketed::Language(p) => {
                    if let Some(slot) = params.get_mut(&*p) {
                        **slot = v;
                    } else {
                        leftover.push(format!("language_model.{p}"));
                    }
                }
                // Text MoE loader drops any vision-tower keys (VL-MoE is
                // out of scope; a VL checkpoint uses the dense VLM loader).
                Bucketed::Vision(_) => {}
                Bucketed::Other(p) => leftover.push(p),
            }
        }
    }

    if !leftover.is_empty() {
        leftover.sort();
        return Err(Error::config(format!(
            "qwen3_5_moe loader: {} unbound key(s); first 8: {:?}",
            leftover.len(),
            leftover.iter().take(8).collect::<Vec<_>>()
        )));
    }
    eval_params(model.parameters()).map_err(Error::Exception)?;
    apply_post_load_memory_policy();
    Ok(model)
}
