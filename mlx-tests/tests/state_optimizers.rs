use std::{
    any::Any,
    collections::{BTreeSet, HashMap},
    panic::{catch_unwind, AssertUnwindSafe},
    path::PathBuf,
    rc::Rc,
};

use mlx_rs::{
    array,
    builder::Builder,
    macros::ModuleParameters,
    module::{FlattenedModuleParam, ModuleParameters, Param, Parameter},
    optimizers::{
        AdaDeltaBuilder, AdaGradBuilder, AdafactorBuilder, AdamBuilder, AdamWBuilder,
        AdamaxBuilder, LionBuilder, Optimizer, OptimizerState, RmsPropBuilder, SgdBuilder,
    },
    test_utils::assert_array_eq_with_context,
    transforms::compile::compile_with_state,
    Array,
};

const RTOL: f64 = 1.0e-6;
const ATOL: f64 = 1.0e-7;

#[derive(Clone, Debug, ModuleParameters)]
struct TinyModel {
    #[param]
    weight: Param<Array>,
    #[param]
    bias: Param<Array>,
}

struct Fixture {
    tensors: HashMap<String, Array>,
}

impl Fixture {
    fn load() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../conformance/state");
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("manifest.json")).expect(
                "missing state oracle manifest; run conformance/.venv/bin/python conformance/state/generate_state.py",
            ),
        )
        .expect("invalid state oracle manifest");
        assert_eq!(manifest["tolerance_policy"]["name"], "optimizer_f32_chain");
        assert_eq!(manifest["tolerance_policy"]["rtol"], RTOL);
        assert_eq!(manifest["tolerance_policy"]["atol"], ATOL);
        let tensors = Array::load_safetensors(root.join("state.safetensors")).expect(
            "missing state oracle fixture; run conformance/.venv/bin/python conformance/state/generate_state.py",
        );
        Self { tensors }
    }

    fn tensor(&self, name: &str) -> &Array {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("missing fixture tensor {name}"))
    }

    fn model(&self) -> TinyModel {
        TinyModel {
            weight: Param::new(self.tensor("input.param.weight").clone()),
            bias: Param::new(self.tensor("input.param.bias").clone()),
        }
    }

    fn gradients(&self, step: usize, include_bias: bool) -> FlattenedModuleParam {
        let mut gradients = FlattenedModuleParam::new();
        gradients.insert(
            "weight".into(),
            self.tensor(&format!("input.gradient.step{step}.weight"))
                .clone(),
        );
        if include_bias {
            gradients.insert(
                "bias".into(),
                self.tensor(&format!("input.gradient.step{step}.bias"))
                    .clone(),
            );
        }
        gradients
    }

    fn compiled_gradients(&self, step: usize) -> [Array; 2] {
        [
            self.tensor(&format!("input.gradient.step{step}.weight"))
                .clone(),
            self.tensor(&format!("input.gradient.step{step}.bias"))
                .clone(),
        ]
    }

    fn expected_keys(&self, prefix: &str) -> BTreeSet<String> {
        self.tensors
            .keys()
            .filter_map(|key| key.strip_prefix(prefix).map(str::to_owned))
            .collect()
    }
}

fn optimizer_step<O: Optimizer>(
    (model, optimizer): &mut (TinyModel, O),
    gradients: &[Array],
) -> Vec<Array> {
    let mut gradient_map = FlattenedModuleParam::new();
    gradient_map.insert("weight".into(), gradients[0].clone());
    gradient_map.insert("bias".into(), gradients[1].clone());
    optimizer.update(model, gradient_map).unwrap();
    vec![model.weight.value.clone(), model.bias.value.clone()]
}

fn assert_named(got: &Array, expected: &Array, context: &str) {
    assert_array_eq_with_context(got, expected, RTOL, ATOL, context);
}

fn assert_snapshot<O: Optimizer>(
    fixture: &Fixture,
    case_id: &str,
    step: usize,
    model: &TinyModel,
    optimizer: &O,
) {
    let parameter_prefix = format!("{case_id}.step{step}.param.");
    let parameters = model.parameters().flatten();
    let parameter_keys = parameters
        .keys()
        .map(|key| key.to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        parameter_keys,
        fixture.expected_keys(&parameter_prefix),
        "{case_id}.step{step}.parameter_key_set"
    );
    for key in parameter_keys {
        assert_named(
            parameters[&Rc::<str>::from(key.as_str())],
            fixture.tensor(&format!("{parameter_prefix}{key}")),
            &format!("{case_id}.step{step}.parameter.{key}"),
        );
    }

    let state_prefix = format!("{case_id}.step{step}.state.");
    let state = optimizer
        .state()
        .flatten()
        .map(|(key, value)| (key.to_string(), value))
        .collect::<HashMap<_, _>>();
    let state_keys = state.keys().cloned().collect::<BTreeSet<_>>();
    assert_eq!(
        state_keys,
        fixture.expected_keys(&state_prefix),
        "{case_id}.step{step}.state_key_set"
    );
    for key in state_keys {
        assert_named(
            state[&key],
            fixture.tensor(&format!("{state_prefix}{key}")),
            &format!("{case_id}.step{step}.state.{key}"),
        );
    }
}

fn qualify_oracle_trajectory<O: Optimizer>(case_id: &str, optimizer: O) {
    let fixture = Fixture::load();
    let mut model = fixture.model();
    let mut optimizer = optimizer;
    for step in 1..=3 {
        optimizer
            .update(&mut model, fixture.gradients(step, true))
            .unwrap();
        assert_snapshot(&fixture, case_id, step, &model, &optimizer);
    }
}

fn qualify_compiled_trajectory<O: Optimizer + Clone + 'static>(case_id: &str, optimizer: O) {
    let fixture = Fixture::load();
    let initial_model = fixture.model();
    let mut eager_state = (initial_model.clone(), optimizer.clone());
    let mut compiled_state = (initial_model, optimizer);
    let mut compiled = compile_with_state(optimizer_step::<O>, None);
    for step in 1..=3 {
        let gradients = fixture.compiled_gradients(step);
        optimizer_step(&mut eager_state, &gradients);
        compiled(&mut compiled_state, &gradients).unwrap();
        assert_snapshot(&fixture, case_id, step, &eager_state.0, &eager_state.1);
        assert_snapshot(
            &fixture,
            case_id,
            step,
            &compiled_state.0,
            &compiled_state.1,
        );
    }
}

macro_rules! optimizer_pack {
    ($oracle_test:ident, $compiled_test:ident, $case_id:literal, $optimizer:expr) => {
        #[test]
        fn $oracle_test() {
            qualify_oracle_trajectory($case_id, $optimizer);
        }

        #[test]
        fn $compiled_test() {
            qualify_compiled_trajectory($case_id, $optimizer);
        }
    };
}

optimizer_pack!(
    sgd_oracle_trajectory,
    sgd_eager_compiled_consistency,
    "sgd",
    SgdBuilder::new(0.035_f32)
        .momentum(0.8_f32)
        .weight_decay(0.03_f32)
        .dampening(0.0_f32)
        .nesterov(true)
        .build()
        .unwrap()
);
optimizer_pack!(
    adam_oracle_trajectory,
    adam_eager_compiled_consistency,
    "adam",
    AdamBuilder::new(0.025_f32)
        .betas((0.8_f32, 0.95_f32))
        .eps(1.0e-6_f32)
        .build()
        .unwrap()
);
optimizer_pack!(
    adamw_oracle_trajectory,
    adamw_eager_compiled_consistency,
    "adamw",
    AdamWBuilder::new(0.025_f32)
        .betas((0.8_f32, 0.95_f32))
        .eps(1.0e-6_f32)
        .weight_decay(0.04_f32)
        .build()
        .unwrap()
);
optimizer_pack!(
    adamax_oracle_trajectory,
    adamax_eager_compiled_consistency,
    "adamax",
    AdamaxBuilder::new(0.025_f32)
        .betas((0.8_f32, 0.95_f32))
        .eps(1.0e-6_f32)
        .build()
        .unwrap()
);
optimizer_pack!(
    adagrad_oracle_trajectory,
    adagrad_eager_compiled_consistency,
    "adagrad",
    AdaGradBuilder::new(0.04_f32)
        .eps(1.0e-6_f32)
        .build()
        .unwrap()
);
optimizer_pack!(
    adadelta_oracle_trajectory,
    adadelta_eager_compiled_consistency,
    "adadelta",
    AdaDeltaBuilder::new(0.7_f32)
        .rho(0.9_f32)
        .eps(1.0e-6_f32)
        .build()
        .unwrap()
);
optimizer_pack!(
    rmsprop_oracle_trajectory,
    rmsprop_eager_compiled_consistency,
    "rmsprop",
    RmsPropBuilder::new(0.03_f32)
        .alpha(0.91_f32)
        .epsilon(1.0e-6_f32)
        .build()
        .unwrap()
);
optimizer_pack!(
    lion_oracle_trajectory,
    lion_eager_compiled_consistency,
    "lion",
    LionBuilder::new(0.012_f32)
        .betas((0.82_f32, 0.96_f32))
        .weight_decay(0.07_f32)
        .build()
        .unwrap()
);
optimizer_pack!(
    adafactor_oracle_trajectory,
    adafactor_eager_compiled_consistency,
    "adafactor",
    AdafactorBuilder::new()
        .lr(0.03_f32)
        .eps((1.0e-30_f32, 1.0e-3_f32))
        .clip_threshold(1.0_f32)
        .decay_rate(-0.8_f32)
        .beta1(0.9_f32)
        .weight_decay(0.02_f32)
        .scale_parameter(false)
        .relative_step(false)
        .warmup_init(false)
        .build()
        .unwrap()
);

#[test]
fn adam_frozen_bias_oracle_trajectory() {
    let fixture = Fixture::load();
    let mut model = fixture.model();
    model.bias.freeze(false);
    assert_eq!(model.bias.is_frozen(), Some(true));
    let mut optimizer = AdamBuilder::new(0.025_f32)
        .betas((0.8_f32, 0.95_f32))
        .eps(1.0e-6_f32)
        .build()
        .unwrap();
    for step in 1..=3 {
        optimizer
            .update(&mut model, fixture.gradients(step, false))
            .unwrap();
        assert_snapshot(&fixture, "adam_frozen_bias", step, &model, &optimizer);
    }
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
fn fault_no_op_learning_rate_is_parameter_weight() {
    let fixture = Fixture::load();
    let mut model = fixture.model();
    let mut optimizer = AdamBuilder::new(0.0_f32)
        .betas((0.8_f32, 0.95_f32))
        .eps(1.0e-6_f32)
        .build()
        .unwrap();
    optimizer
        .update(&mut model, fixture.gradients(1, true))
        .unwrap();
    expect_named_failure("adam.step1.parameter.weight", || {
        assert_named(
            &model.weight,
            fixture.tensor("adam.step1.param.weight"),
            "adam.step1.parameter.weight",
        );
    });
}

#[test]
fn fault_stuck_step_counter_is_state_weight_step() {
    let fixture = Fixture::load();
    let mut model = fixture.model();
    let mut optimizer = AdafactorBuilder::new()
        .lr(0.03_f32)
        .eps((1.0e-30_f32, 1.0e-3_f32))
        .clip_threshold(1.0_f32)
        .decay_rate(-0.8_f32)
        .beta1(0.9_f32)
        .weight_decay(0.02_f32)
        .scale_parameter(false)
        .relative_step(false)
        .warmup_init(false)
        .build()
        .unwrap();
    optimizer
        .update(&mut model, fixture.gradients(1, true))
        .unwrap();
    let step = optimizer
        .state_mut()
        .flatten_mut()
        .find_map(|(key, value)| (key.as_ref() == "weight.step").then_some(value))
        .unwrap();
    *step = array!(0_i32);
    expect_named_failure("adafactor.step1.state.weight.step", || {
        assert_named(
            optimizer
                .state()
                .flatten()
                .find_map(|(key, value)| (key.as_ref() == "weight.step").then_some(value))
                .unwrap(),
            fixture.tensor("adafactor.step1.state.weight.step"),
            "adafactor.step1.state.weight.step",
        );
    });
}

#[test]
fn fault_reordered_state_tensors_is_state_weight_zero() {
    let fixture = Fixture::load();
    let mut model = fixture.model();
    let mut optimizer = AdamBuilder::new(0.025_f32)
        .betas((0.8_f32, 0.95_f32))
        .eps(1.0e-6_f32)
        .build()
        .unwrap();
    optimizer
        .update(&mut model, fixture.gradients(1, true))
        .unwrap();
    let first = optimizer
        .state()
        .flatten()
        .find_map(|(key, value)| (key.as_ref() == "weight.0").then_some(value))
        .unwrap();
    expect_named_failure("adam.step1.state.weight.0", || {
        assert_named(
            first,
            fixture.tensor("adam.step1.state.weight.1"),
            "adam.step1.state.weight.0",
        );
    });
}

#[test]
fn fault_frozen_parameter_mutation_is_parameter_bias() {
    let fixture = Fixture::load();
    let mut model = fixture.model();
    model.bias.freeze(false);
    assert_eq!(model.bias.is_frozen(), Some(true));
    let mut optimizer = AdamBuilder::new(0.025_f32)
        .betas((0.8_f32, 0.95_f32))
        .eps(1.0e-6_f32)
        .build()
        .unwrap();
    optimizer
        .update(&mut model, fixture.gradients(1, true))
        .unwrap();
    expect_named_failure("adam_frozen_bias.step1.parameter.bias", || {
        assert_named(
            &model.bias,
            fixture.tensor("adam_frozen_bias.step1.param.bias"),
            "adam_frozen_bias.step1.parameter.bias",
        );
    });
}

#[test]
fn fault_wrong_step_expectation_is_parameter_weight() {
    let fixture = Fixture::load();
    let mut model = fixture.model();
    let mut optimizer = AdamBuilder::new(0.025_f32)
        .betas((0.8_f32, 0.95_f32))
        .eps(1.0e-6_f32)
        .build()
        .unwrap();
    for step in 1..=2 {
        optimizer
            .update(&mut model, fixture.gradients(step, true))
            .unwrap();
    }
    expect_named_failure("adam.step2.parameter.weight", || {
        assert_named(
            &model.weight,
            fixture.tensor("adam.step3.param.weight"),
            "adam.step2.parameter.weight",
        );
    });
}
