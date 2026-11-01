use mlx_rs::{error::Exception, nn, ops, transforms::compile, Array};

type Activation = dyn for<'a> FnMut((&'a Array, &'a Array)) -> Result<Array, Exception>;

pub(super) struct GeGlu(Box<Activation>);
impl Default for GeGlu {
    fn default() -> Self {
        Self(Box::new(compile::compile(
            |(gate, up): (&Array, &Array)| nn::gelu_approximate(gate)?.multiply(up),
            true,
        )))
    }
}
impl GeGlu {
    pub(super) fn apply(&mut self, gate: &Array, up: &Array) -> Result<Array, Exception> {
        self.0((gate, up))
    }
}

pub(super) struct SwiGlu(Box<Activation>);
impl Default for SwiGlu {
    fn default() -> Self {
        Self(Box::new(compile::compile(
            |(gate, up): (&Array, &Array)| gate.multiply(ops::sigmoid(gate)?)?.multiply(up),
            true,
        )))
    }
}
impl SwiGlu {
    pub(super) fn apply(&mut self, gate: &Array, up: &Array) -> Result<Array, Exception> {
        self.0((gate, up))
    }
}
