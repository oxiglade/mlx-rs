//! Tests for the optimizers. These tests are placed here because the models
//! used for testing make use of `ModuleParameter` macro.

use std::{collections::HashMap, rc::Rc};

use mlx_rs::{
    array,
    builder::Builder,
    losses::{LossReduction, MseLossBuilder},
    macros::ModuleParameters,
    module::{FlattenedModuleParam, Module, ModuleParameters, Param},
    nn,
    ops::{ones, zeros},
    optimizers::{
        AdaDelta, AdaDeltaBuilder, AdaGrad, AdafactorBuilder, Adam, AdamW, Adamax, Lion,
        LionBuilder, Optimizer, RmsProp, RmsPropBuilder, Sgd, SgdBuilder,
    },
    random::uniform,
    test_utils::{assert_array_eq, tolerances},
    transforms::{eval, eval_params},
    Array, Dtype,
};

mod common;

use common::*;

/* -------------------------------------------------------------------------- */
/*                              Convergence tests                             */
/* -------------------------------------------------------------------------- */

pub fn train<F, O>(f: F, steps: usize) -> Result<Array, Box<dyn std::error::Error>>
where
    F: FnOnce() -> O,
    O: Optimizer,
{
    let mut optimizer = f();

    let mse_loss = MseLossBuilder::new()
        .reduction(LossReduction::Mean)
        .build()?;
    let loss = |model: &mut LinearFunctionModel, (x, y): (&Array, &Array)| {
        mse_loss.apply(model.forward(x)?, y)
    };

    // TODO: check compiled model once we have it
    let mut model = LinearFunctionModel::new(None)?;
    eval_params(model.parameters())?;

    let m = array!(0.25);
    let b = array!(7.0);

    let mut lg = nn::value_and_grad(loss);

    let mut last_loss = None;
    for _ in 0..steps {
        // println!("target: b = {}, m = {}", b, m);
        // println!("parameters: {:?}", model.parameters());

        // generate random training data along with the ground truth.
        // notice that the shape is [B, 1] where B is the batch
        // dimension -- this allows us to train on 10 samples simultaneously
        let x = uniform::<_, f32>(-5.0, 5.0, &[10, 1], None)?;
        let y = &m * &x + &b;
        eval([&x, &y])?;

        // compute the loss and gradients.  use the optimizer
        // to adjust the parameters closer to the target
        let (loss, g) = lg(&mut model, (&x, &y))?;
        optimizer.update(&mut model, g)?;

        eval_params(model.parameters())?;

        last_loss = Some(loss);
    }

    Ok(last_loss.unwrap())
}

const NUM_TRIALS: usize = 3;

#[test]
fn test_sgd_converges() {
    let mut total_loss = 0.0;
    for _ in 0..NUM_TRIALS {
        let loss = train(|| Sgd::new(0.1), 30).unwrap();
        total_loss += loss.item::<f32>();
    }
    // It sometimes doesn't converge that fast, so we take the average loss
    // across multiple trials
    let avg_loss = total_loss / NUM_TRIALS as f32;
    assert!(avg_loss < 0.1, "avg loss: {avg_loss}");
}

#[test]
fn test_rmsprop_converges() {
    let mut total_loss = 0.0;
    for _ in 0..NUM_TRIALS {
        // RMSProp doesn't seem to converge as fast as SGD
        let loss = train(|| RmsProp::new(0.1).unwrap(), 100).unwrap();
        total_loss += loss.item::<f32>();
    }
    // It sometimes doesn't converge that fast, so we take the average loss
    // across multiple trials
    let avg_loss = total_loss / NUM_TRIALS as f32;
    assert!(avg_loss < 0.1, "avg loss: {avg_loss}");
}

/* -------------------------------------------------------------------------- */
/*                            Optimizer unit tests                            */
/* -------------------------------------------------------------------------- */

#[derive(Clone, Debug, ModuleParameters)]
struct SimpleModel {
    #[param]
    a: Param<Array>,
}

#[derive(Debug, ModuleParameters)]
struct First {
    #[param]
    pub a: Param<Array>,

    #[param]
    pub b: Param<Array>,
}

#[derive(Debug, ModuleParameters)]
struct NestedModel {
    #[param]
    pub first: First,

    #[param]
    pub second: Param<Array>,
}

type GradsMap = FlattenedModuleParam;

fn assert_tensor_maps_eq<G, E>(
    got: &HashMap<Rc<str>, G>,
    expected: &HashMap<Rc<str>, E>,
    rtol: f64,
    atol: f64,
) where
    G: AsRef<Array>,
    E: AsRef<Array>,
{
    assert_eq!(got.len(), expected.len());
    let mut keys = expected.keys().collect::<Vec<_>>();
    keys.sort();
    for key in keys {
        let got = got
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key}"));
        assert_array_eq(got.as_ref(), expected[key].as_ref(), rtol, atol);
    }
}

fn assert_save_and_load<O>(optimizer: O, new_optimizer: O) -> Result<(), Box<dyn std::error::Error>>
where
    O: Optimizer,
{
    use mlx_rs::optimizers::OptimizerState;

    let tmp_dir = tempfile::tempdir()?;
    let path = tmp_dir.path().join("optimizer.safetensors");

    optimizer.state().save_safetensors(&path)?;

    let mut loaded_optimizer = new_optimizer;
    loaded_optimizer.state_mut().load_safetensors(&path)?;

    let original_state: HashMap<_, _> = optimizer.state().flatten().collect();
    let loaded_state: HashMap<_, _> = loaded_optimizer.state().flatten().collect();

    assert!(!loaded_state.is_empty());
    assert_tensor_maps_eq(
        &loaded_state,
        &original_state,
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );

    Ok(())
}

fn create_default_test_model_and_grads() -> (NestedModel, GradsMap) {
    let first = First {
        a: Param::new(zeros::<f32>(&[10]).unwrap()),
        b: Param::new(zeros::<f32>(&[1]).unwrap()),
    };
    let model = NestedModel {
        first,
        second: Param::new(zeros::<f32>(&[1]).unwrap()),
    };

    let grads_map: GradsMap = model
        .parameters()
        .flatten()
        .iter()
        .map(|(k, v)| {
            let g = ones::<f32>(v.shape()).unwrap();
            (k.clone(), g)
        })
        .collect();

    (model, grads_map)
}

#[test]
fn test_ada_delta() {
    let initial_parameter = array!([1.0, -2.0, 0.5]);
    let initial_state = (
        SimpleModel {
            a: Param::new(initial_parameter.clone()),
        },
        AdaDeltaBuilder::new(0.1_f32).rho(0.99).build().unwrap(),
    );
    let (mut model, mut optimizer) = initial_state.clone();
    let gradients = [
        [0.25_f32, -0.5, 1.0],
        [-0.75, 0.125, -0.25],
        [0.5, 0.25, -0.125],
    ];
    let mut expected_parameter = initial_parameter.as_slice::<f32>().to_vec();
    let mut expected_v = [0.0_f32; 3];
    let mut expected_u = [0.0_f32; 3];
    let rho = 0.99;
    let epsilon = AdaDelta::DEFAULT_EPS;
    let learning_rate = 0.1_f32;

    for gradient in gradients {
        let mut gradient_map = FlattenedModuleParam::new();
        gradient_map.insert("a".into(), Array::from_slice(&gradient, &[3]));
        optimizer.update(&mut model, gradient_map).unwrap();

        for index in 0..gradient.len() {
            let gradient_squared = gradient[index] * gradient[index];
            let v = rho * expected_v[index] + (1.0 - rho) * gradient_squared;
            let numerator = (expected_u[index] + epsilon).sqrt();
            let denominator = (v + epsilon).sqrt();
            let delta = numerator / denominator * gradient[index];
            let delta_squared = delta * delta;
            let u = rho * expected_u[index] + (1.0 - rho) * delta_squared;
            expected_parameter[index] -= learning_rate * delta;
            expected_v[index] = v;
            expected_u[index] = u;
        }

        assert_array_eq(
            model.a.as_ref(),
            Array::from_slice(&expected_parameter, &[3]),
            tolerances::STANDARD.rtol,
            tolerances::STANDARD.atol,
        );
        let (v, u) = optimizer.state.get("a").unwrap();
        assert_array_eq(
            v,
            Array::from_slice(&expected_v, &[3]),
            tolerances::STANDARD.rtol,
            tolerances::STANDARD.atol,
        );
        assert_array_eq(
            u,
            Array::from_slice(&expected_u, &[3]),
            tolerances::STANDARD.rtol,
            tolerances::STANDARD.atol,
        );
    }

    assert_save_and_load(
        optimizer,
        AdaDeltaBuilder::new(0.1_f32).rho(0.99).build().unwrap(),
    )
    .unwrap();
}

// This unit test is adapted from the swift binding unit test `testAdaGrad` in
// `mlx-swift/Tests/MLXTests/IntegrationTests.swift`
#[test]
fn test_adagrad() {
    mlx_rs::random::seed(958).unwrap();
    let a = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq(
        a.mean(None).unwrap(),
        array!(-0.045_843_333),
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );
    assert_array_eq(
        a.sum(None).unwrap(),
        array!(-0.550_12),
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );

    let a_grad = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq(
        a_grad.mean(None).unwrap(),
        array!(0.232_503_94),
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );
    assert_array_eq(
        a_grad.sum(None).unwrap(),
        array!(2.790_047_2),
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = AdaGrad::new(0.1);

    optimizer.update(&mut a_model, a_grad_params).unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq(
        a_model.a.mean(None).unwrap(),
        array!(-0.062_509_984),
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );
    assert_array_eq(
        a_model.a.sum(None).unwrap(),
        array!(-0.750_119_8),
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );

    assert_save_and_load(optimizer, AdaGrad::new(0.1)).unwrap();
}

// This unit test is adapted from the swift binding unit test `testAdam` in
// `mlx-swift/Tests/MLXTests/IntegrationTests.swift`
#[test]
fn test_adam() {
    mlx_rs::random::seed(616).unwrap();
    let a = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq(
        a.mean(None).unwrap(),
        array!(0.112_293_06),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a.sum(None).unwrap(),
        array!(1.347_516_7),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    let a_grad = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq(
        a_grad.mean(None).unwrap(),
        array!(0.305_597_72),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a_grad.sum(None).unwrap(),
        array!(3.667_172_7),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = Adam::new(0.1);

    optimizer.update(&mut a_model, a_grad_params).unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq(
        a_model.a.mean(None).unwrap(),
        array!(0.112_292_78),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a_model.a.sum(None).unwrap(),
        array!(1.347_513_3),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    assert_save_and_load(optimizer, Adam::new(0.1)).unwrap();
}

// This unit test is adapted from the swift binding unit test `testAdamW` in
// `mlx-swift/Tests/MLXTests/IntegrationTests.swift`
#[test]
fn test_adamw() {
    mlx_rs::random::seed(696).unwrap();
    let a = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq(
        a.mean(None).unwrap(),
        array!(-0.363_391_88),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a.sum(None).unwrap(),
        array!(-4.360_702_5),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    let a_grad = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq(
        a_grad.mean(None).unwrap(),
        array!(0.221_754_48),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a_grad.sum(None).unwrap(),
        array!(2.661_053_7),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = AdamW::new(0.1);

    optimizer.update(&mut a_model, a_grad_params).unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq(
        a_model.a.mean(None).unwrap(),
        array!(-0.468_437_6),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a_model.a.sum(None).unwrap(),
        array!(-5.621_251),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    assert_save_and_load(optimizer, AdamW::new(0.1)).unwrap();
}

// This unit test is adapted from the python unit test `test_adamax` in
// `mlx/python/tests/test_optimizers.py`.
#[test]
fn test_adamax() {
    mlx_rs::random::seed(75).unwrap();
    let a = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq(
        a.mean(None).unwrap(),
        array!(-0.303_923_6),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a.sum(None).unwrap(),
        array!(-3.647_083_3),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    let a_grad = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq(
        a_grad.mean(None).unwrap(),
        array!(-0.242_717_24),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a_grad.sum(None).unwrap(),
        array!(-2.912_606_7),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = Adamax::new(0.1);

    optimizer.update(&mut a_model, a_grad_params).unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq(
        a_model.a.mean(None).unwrap(),
        array!(-0.303_923_6),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a_model.a.sum(None).unwrap(),
        array!(-3.647_083_3),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    assert_save_and_load(optimizer, Adamax::new(0.1)).unwrap();
}

// This unit test is adapted from the python unit test `test_rmsprop` in
// `tests/test_optimizer.py`.
#[test]
fn test_rmsprop() {
    const LR: f32 = 1e-2;
    const ALPHA: f32 = 0.99;

    let (mut model, gradients) = create_default_test_model_and_grads();

    let mut optim = RmsPropBuilder::new(LR).alpha(ALPHA).build().unwrap();
    optim.update(&mut model, gradients).unwrap();

    let expected_first_a = ones::<f32>(&[10]).unwrap() * -0.1;
    let expected_first_b = ones::<f32>(&[1]).unwrap() * -0.1;
    let expected_second = ones::<f32>(&[1]).unwrap() * -0.1;

    assert_array_eq(
        model.first.a.as_ref(),
        expected_first_a,
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );
    assert_array_eq(
        model.first.b.as_ref(),
        expected_first_b,
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );
    assert_array_eq(
        model.second.as_ref(),
        expected_second,
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );

    let expected_state_first_a = ones::<f32>(&[10]).unwrap() * 0.01;
    let expected_state_first_b = ones::<f32>(&[1]).unwrap() * 0.01;
    let expected_state_second = ones::<f32>(&[1]).unwrap() * 0.01;

    assert_array_eq(
        optim.state.get("first.a").unwrap(),
        expected_state_first_a,
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );
    assert_array_eq(
        optim.state.get("first.b").unwrap(),
        expected_state_first_b,
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );
    assert_array_eq(
        optim.state.get("second").unwrap(),
        expected_state_second,
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );

    assert_save_and_load(optim, RmsPropBuilder::new(LR).alpha(ALPHA).build().unwrap()).unwrap();
}

// This unit test is adapted from the python unit test `test_sgd` in
// `mlx/python/tests/test_optimizers.py`
#[test]
fn test_sgd() {
    let (mut model, gradients) = create_default_test_model_and_grads();

    let mut optim = SgdBuilder::new(1e-2).momentum(0.9).build().unwrap();
    optim.update(&mut model, gradients).unwrap();

    let expected_first_a = ones::<f32>(&[10]).unwrap() * -0.01;
    let expected_first_b = ones::<f32>(&[1]).unwrap() * -0.01;
    let expected_second = ones::<f32>(&[1]).unwrap() * -0.01;

    assert_array_eq(
        model.first.a.as_ref(),
        expected_first_a,
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );
    assert_array_eq(
        model.first.b.as_ref(),
        expected_first_b,
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );
    assert_array_eq(
        model.second.as_ref(),
        expected_second,
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );

    let expected_state_first_a = ones::<f32>(&[10]).unwrap();
    let expected_state_first_b = ones::<f32>(&[1]).unwrap();
    let expected_state_second = ones::<f32>(&[1]).unwrap();

    assert_array_eq(
        optim.state["first.a"].as_ref(),
        expected_state_first_a,
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );
    assert_array_eq(
        optim.state["first.b"].as_ref(),
        expected_state_first_b,
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );
    assert_array_eq(
        optim.state["second"].as_ref(),
        expected_state_second,
        tolerances::STANDARD.rtol,
        tolerances::STANDARD.atol,
    );
}

// This unit test is adapted from the swift binding unit test `testLion` in
// `mlx-swift/Tests/MLXTests/IntegrationTests.swift`
#[test]
fn test_lion() {
    mlx_rs::random::seed(27).unwrap();
    let a = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq(
        a.mean(None).unwrap(),
        array!(0.177_692_23),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a.sum(None).unwrap(),
        array!(2.132_306_8),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    let a_grad = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq(
        a_grad.mean(None).unwrap(),
        array!(-0.021_187_237),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a_grad.sum(None).unwrap(),
        array!(-0.254_246_83),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = Lion::new(0.1);

    optimizer.update(&mut a_model, a_grad_params).unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq(
        a_model.a.mean(None).unwrap(),
        array!(0.211_025_57),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a_model.a.sum(None).unwrap(),
        array!(2.532_306_7),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    assert_save_and_load(optimizer, Lion::new(0.1)).unwrap();
}

// This unit test is adapted from the swift binding unit test `testLion1` in
// `mlx-swift/Tests/MLXTests/IntegrationTests.swift`
#[test]
fn test_lion1() {
    mlx_rs::random::seed(127).unwrap();
    let a = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq(
        a.mean(None).unwrap(),
        array!(-0.184_610_6),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a.sum(None).unwrap(),
        array!(-2.215_327_3),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    let a_grad = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq(
        a_grad.mean(None).unwrap(),
        array!(-0.036_004_007),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a_grad.sum(None).unwrap(),
        array!(-0.432_048_08),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = LionBuilder::new(0.1).weight_decay(0.1).build().unwrap();

    optimizer.update(&mut a_model, a_grad_params).unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq(
        a_model.a.mean(None).unwrap(),
        array!(-0.182_764_5),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a_model.a.sum(None).unwrap(),
        array!(-2.193_174),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    assert_save_and_load(
        optimizer,
        LionBuilder::new(0.1).weight_decay(0.1).build().unwrap(),
    )
    .unwrap();
}

#[test]
fn test_adafactor() {
    mlx_rs::random::seed(650).unwrap();
    let a = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq(
        a.mean(None).unwrap(),
        array!(-0.520_713_7),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a.sum(None).unwrap(),
        array!(-6.248_564),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    let a_grad = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq(
        a_grad.mean(None).unwrap(),
        array!(0.433_303_65),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a_grad.sum(None).unwrap(),
        array!(5.199_643_6),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = AdafactorBuilder::new().lr(0.1).build().unwrap();

    optimizer.update(&mut a_model, a_grad_params).unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    println!(
        "a_model.a.mean(None).unwrap(): {:?}",
        a_model.a.mean(None).unwrap()
    );
    assert_array_eq(
        a_model.a.mean(None).unwrap(),
        array!(-0.526_828_47),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a_model.a.sum(None).unwrap(),
        array!(-6.321_941_4),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    assert_save_and_load(optimizer, AdafactorBuilder::new().lr(0.1).build().unwrap()).unwrap();
}

#[test]
fn test_adafactor1() {
    mlx_rs::random::seed(193).unwrap();
    let a = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq(
        a.mean(None).unwrap(),
        array!(0.400_818_17),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a.sum(None).unwrap(),
        array!(4.809_818),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    let a_grad = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq(
        a_grad.mean(None).unwrap(),
        array!(0.214_474_72),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a_grad.sum(None).unwrap(),
        array!(2.573_696_6),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = AdafactorBuilder::new().lr(0.1).beta1(0.1).build().unwrap();

    optimizer.update(&mut a_model, a_grad_params).unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq(
        a_model.a.mean(None).unwrap(),
        array!(0.399_430_7),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a_model.a.sum(None).unwrap(),
        array!(4.793_168),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
}

#[test]
fn test_adafactor2() {
    mlx_rs::random::seed(620).unwrap();
    let a = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[10], None).unwrap();
    assert_eq!(a.shape(), &[10]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq(
        a.mean(None).unwrap(),
        array!(0.489_024_55),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a.sum(None).unwrap(),
        array!(4.890_245_4),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    let a_grad = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[10], None).unwrap();
    assert_eq!(a_grad.shape(), &[10]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq(
        a_grad.mean(None).unwrap(),
        array!(0.681_890_2),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a_grad.sum(None).unwrap(),
        array!(6.818_902),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = AdafactorBuilder::new().lr(0.1).build().unwrap();

    optimizer.update(&mut a_model, a_grad_params).unwrap();
    assert_eq!(a_model.a.shape(), &[10]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq(
        a_model.a.mean(None).unwrap(),
        array!(0.483_533_05),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
    assert_array_eq(
        a_model.a.sum(None).unwrap(),
        array!(4.835_330_5),
        tolerances::RANDOM_STATISTIC.rtol,
        tolerances::RANDOM_STATISTIC.atol,
    );
}
