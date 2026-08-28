use mlx_rs::{
    error::Exception, macros::ModuleParameters, module::Module, nn::Linear, Array, Dtype,
};

#[derive(Debug, ModuleParameters)]
struct M {
    #[param]
    linear: Linear,
}

impl M {
    pub fn new() -> Self {
        Self {
            linear: Linear::new(5, 5).unwrap(),
        }
    }
}

impl Module<&Array> for M {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        self.linear.forward(x)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

#[test]
fn test_nested_module() {
    let mut m = M::new();
    let x = mlx_rs::random::uniform::<_, f32>(1.0, 2.0, &[1, 5], None).unwrap();
    let y = m.forward(&x).unwrap();
    let sum = y.sum(None).unwrap();
    assert_eq!(sum.dtype(), Dtype::Float32);
    assert!(sum.shape().is_empty());
    let sum = sum.item::<f32>();
    assert!(sum.is_finite());
    assert_ne!(sum, 0.0);
}
