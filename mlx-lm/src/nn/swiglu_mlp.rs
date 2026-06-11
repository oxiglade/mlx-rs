//! Shared SwiGLU MLP used by llama and qwen3.
//!
//! Three Linear projections (gate, up, down): `down(silu(gate(x)) * up(x))`.

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn::{self, Linear, LinearBuilder},
    quantization::MaybeQuantized,
    Array,
};

/// Three-projection SwiGLU MLP. Projection bias is configurable
/// (`true` for llama's `mlp_bias`, `false` for qwen3).
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct SwigluMlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<Linear>,

    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<Linear>,

    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<Linear>,
}

impl SwigluMlp {
    pub fn new(dim: i32, hidden_dim: i32, bias: bool) -> Result<Self, Exception> {
        let gate_proj = LinearBuilder::new(dim, hidden_dim).bias(bias).build()?;
        let down_proj = LinearBuilder::new(hidden_dim, dim).bias(bias).build()?;
        let up_proj = LinearBuilder::new(dim, hidden_dim).bias(bias).build()?;

        Ok(Self {
            gate_proj: MaybeQuantized::Original(gate_proj),
            down_proj: MaybeQuantized::Original(down_proj),
            up_proj: MaybeQuantized::Original(up_proj),
        })
    }
}

impl Module<&Array> for SwigluMlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let gate = self.gate_proj.forward(input)?;
        let up = self.up_proj.forward(input)?;
        let activated = nn::silu(&gate)?.multiply(&up)?;
        self.down_proj.forward(&activated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}
