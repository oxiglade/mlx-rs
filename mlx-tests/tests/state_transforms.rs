use std::{
    any::Any,
    collections::{BTreeSet, HashMap},
    panic::{catch_unwind, AssertUnwindSafe},
    path::PathBuf,
    rc::Rc,
};

use mlx_rs::{
    macros::ModuleParameters,
    module::{ModuleParameters, Param},
    nn,
    test_utils::assert_array_eq_with_context,
    transforms::{grad_with_argnums, jvp, value_and_grad_with_argnums, vjp},
    Array,
};

const RTOL: f64 = 1.0e-6;
const ATOL: f64 = 1.0e-7;

struct Fixture {
    tensors: HashMap<String, Array>,
}

impl Fixture {
    fn load() -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../conformance/state/state.safetensors");
        let tensors = Array::load_safetensors(path).expect(
            "missing transforms oracle fixture; run conformance/.venv/bin/python conformance/state/generate_state.py",
        );
        Self { tensors }
    }

    fn tensor(&self, name: &str) -> &Array {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("missing fixture tensor {name}"))
    }

    fn nonlinear_inputs(&self) -> [Array; 3] {
        [
            self.tensor("transform.input.x").clone(),
            self.tensor("transform.input.weight").clone(),
            self.tensor("transform.input.bias").clone(),
        ]
    }

    fn directional_primals(&self) -> [Array; 2] {
        [
            self.tensor("transform.input.a").clone(),
            self.tensor("transform.input.c").clone(),
        ]
    }
}

fn assert_named(got: &Array, expected: &Array, context: &str) {
    assert_array_eq_with_context(got, expected, RTOL, ATOL, context);
}

fn nonlinear(args: &[Array]) -> Vec<Array> {
    let affine = args[1].matmul(&args[0]).unwrap().add(&args[2]).unwrap();
    vec![mlx_rs::ops::tanh(&affine)
        .unwrap()
        .square()
        .unwrap()
        .sum(None)
        .unwrap()]
}

fn multi_output(args: &[Array]) -> Vec<Array> {
    let first = mlx_rs::ops::tanh(
        args[0]
            .multiply(&args[1])
            .unwrap()
            .add(&args[0].square().unwrap())
            .unwrap(),
    )
    .unwrap();
    let second = args[0]
        .multiply(&args[1].square().unwrap())
        .unwrap()
        .sum(None)
        .unwrap();
    vec![first, second]
}

#[test]
fn nonlinear_multi_input_grad_and_value_and_grad_match_oracle() {
    let fixture = Fixture::load();
    let inputs = fixture.nonlinear_inputs();
    let argnums = [0, 1, 2];

    let gradients = grad_with_argnums(nonlinear, &argnums)(&inputs).unwrap();
    assert_eq!(gradients.len(), 3, "transform.grad.count");
    for (gradient, name) in gradients.iter().zip(["x", "weight", "bias"]) {
        assert_named(
            gradient,
            fixture.tensor(&format!("transform.nonlinear.gradient.{name}")),
            &format!("transform.grad.{name}"),
        );
    }

    let (values, gradients) = value_and_grad_with_argnums(nonlinear, &argnums)(&inputs).unwrap();
    assert_eq!(values.len(), 1, "transform.value_and_grad.value_count");
    assert_eq!(
        gradients.len(),
        3,
        "transform.value_and_grad.gradient_count"
    );
    assert_named(
        &values[0],
        fixture.tensor("transform.nonlinear.value"),
        "transform.value_and_grad.value",
    );
    for (gradient, name) in gradients.iter().zip(["x", "weight", "bias"]) {
        assert_named(
            gradient,
            fixture.tensor(&format!("transform.nonlinear.gradient.{name}")),
            &format!("transform.value_and_grad.gradient.{name}"),
        );
    }
}

#[test]
fn multi_input_multi_output_jvp_and_vjp_match_oracle() {
    let fixture = Fixture::load();
    let primals = fixture.directional_primals();
    let tangents = [
        fixture.tensor("transform.input.a_tangent").clone(),
        fixture.tensor("transform.input.c_tangent").clone(),
    ];
    let (jvp_values, jvp_tangents) = jvp(multi_output, &primals, &tangents).unwrap();
    assert_eq!(jvp_values.len(), 2, "transform.jvp.value_count");
    assert_eq!(jvp_tangents.len(), 2, "transform.jvp.tangent_count");
    for index in 0..2 {
        assert_named(
            &jvp_values[index],
            fixture.tensor(&format!("transform.jvp.value.{index}")),
            &format!("transform.jvp.value.{index}"),
        );
        assert_named(
            &jvp_tangents[index],
            fixture.tensor(&format!("transform.jvp.tangent.{index}")),
            &format!("transform.jvp.tangent.{index}"),
        );
    }

    let cotangents = [
        fixture.tensor("transform.input.output0_cotangent").clone(),
        fixture.tensor("transform.input.output1_cotangent").clone(),
    ];
    let (vjp_values, vjp_cotangents) = vjp(multi_output, &primals, &cotangents).unwrap();
    assert_eq!(vjp_values.len(), 2, "transform.vjp.value_count");
    assert_eq!(vjp_cotangents.len(), 2, "transform.vjp.cotangent_count");
    for index in 0..2 {
        assert_named(
            &vjp_values[index],
            fixture.tensor(&format!("transform.vjp.value.{index}")),
            &format!("transform.vjp.value.{index}"),
        );
        assert_named(
            &vjp_cotangents[index],
            fixture.tensor(&format!("transform.vjp.cotangent.{index}")),
            &format!("transform.vjp.cotangent.{index}"),
        );
    }
}

#[test]
fn argnums_selects_only_x_and_bias_gradients() {
    let fixture = Fixture::load();
    let inputs = fixture.nonlinear_inputs();
    let argnums = [0, 2];
    let gradients = grad_with_argnums(nonlinear, &argnums)(&inputs).unwrap();
    assert_eq!(gradients.len(), 2, "transform.argnums.gradient_count");
    for (gradient, name) in gradients.iter().zip(["x", "bias"]) {
        assert_named(
            gradient,
            fixture.tensor(&format!("transform.argnums.gradient.{name}")),
            &format!("transform.argnums.gradient.{name}"),
        );
    }
}

#[derive(Debug, ModuleParameters)]
struct TransformModel {
    #[param]
    weight: Param<Array>,
    #[param]
    bias: Param<Array>,
}

fn module_loss(model: &mut TransformModel, x: &Array) -> Array {
    mlx_rs::ops::tanh(model.weight.matmul(x).unwrap().add(&model.bias).unwrap())
        .unwrap()
        .square()
        .unwrap()
        .sum(None)
        .unwrap()
}

#[test]
fn module_keyed_value_and_grad_matches_oracle() {
    let fixture = Fixture::load();
    let mut model = TransformModel {
        weight: Param::new(fixture.tensor("transform.input.weight").clone()),
        bias: Param::new(fixture.tensor("transform.input.bias").clone()),
    };
    let x = fixture.tensor("transform.input.x");
    let mut value_and_grad = nn::value_and_grad(module_loss);
    let (value, gradients) = value_and_grad(&mut model, x).unwrap();

    assert_named(
        &value,
        fixture.tensor("transform.module.value"),
        "transform.module.value",
    );
    let expected_keys = ["bias", "weight"]
        .into_iter()
        .map(Rc::<str>::from)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        gradients.keys().cloned().collect::<BTreeSet<_>>(),
        expected_keys,
        "transform.module.gradient_keys"
    );
    for key in expected_keys {
        assert_named(
            &gradients[&key],
            fixture.tensor(&format!("transform.module.gradient.{key}")),
            &format!("transform.module.gradient.{key}"),
        );
    }

    let parameters = model.parameters().flatten();
    assert_named(
        parameters[&Rc::<str>::from("weight")],
        fixture.tensor("transform.input.weight"),
        "transform.module.parameter.weight",
    );
    assert_named(
        parameters[&Rc::<str>::from("bias")],
        fixture.tensor("transform.input.bias"),
        "transform.module.parameter.bias",
    );
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_owned();
    }
    "non-string panic payload".to_owned()
}

fn expect_named_failure(class: &str, comparison: impl FnOnce()) {
    let failure = catch_unwind(AssertUnwindSafe(comparison))
        .expect_err(&format!("mutation should fail comparison class {class}"));
    let message = panic_message(failure);
    assert!(
        message.contains(class),
        "expected comparison class {class}, got {message}"
    );
}

#[test]
fn fault_perturbed_input_gradient_fails() {
    let fixture = Fixture::load();
    let mut inputs = fixture.nonlinear_inputs();
    inputs[0] = inputs[0].add(&Array::from_f32(0.03125)).unwrap();
    let argnums = [0];
    let gradients = grad_with_argnums(nonlinear, &argnums)(&inputs).unwrap();
    expect_named_failure("transform.gradient.x", || {
        assert_named(
            &gradients[0],
            fixture.tensor("transform.nonlinear.gradient.x"),
            "transform.gradient.x",
        );
    });
}

#[test]
fn fault_swapped_vjp_expectation_for_jvp_fails() {
    let fixture = Fixture::load();
    let primals = fixture.directional_primals();
    let tangents = [
        fixture.tensor("transform.input.a_tangent").clone(),
        fixture.tensor("transform.input.c_tangent").clone(),
    ];
    let (_, jvp_tangents) = jvp(multi_output, &primals, &tangents).unwrap();
    expect_named_failure("transform.jvp.tangent.0", || {
        assert_named(
            &jvp_tangents[0],
            fixture.tensor("transform.vjp.cotangent.0"),
            "transform.jvp.tangent.0",
        );
    });
}
