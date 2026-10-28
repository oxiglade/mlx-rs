use mlx_rs::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    random::uniform,
    utils::IntoOption,
    Array,
};

/// A helper model for testing optimizers.
///
/// This is adapted from the swift binding tests in `mlx-swift/Tests/MLXTests/OptimizerTests.swift`.
#[derive(Clone, Debug, ModuleParameters)]
pub struct LinearFunctionModel {
    #[param]
    pub m: Param<Array>,

    #[param]
    pub b: Param<Array>,
}

impl Module<&Array> for LinearFunctionModel {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        self.m.multiply(x)?.add(&self.b)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

impl LinearFunctionModel {
    pub fn new<'a>(shape: impl IntoOption<&'a [i32]>) -> mlx_rs::error::Result<Self> {
        let shape = shape.into_option();
        let m = uniform::<_, f32>(-5.0, 5.0, shape, None)?;
        let b = uniform::<_, f32>(-5.0, 5.0, shape, None)?;
        Ok(Self {
            m: Param::new(m),
            b: Param::new(b),
        })
    }
}
