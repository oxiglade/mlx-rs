//! Tests for compilation of modules and optimizers.

mod common;

use std::{collections::HashMap, rc::Rc};

use common::LinearFunctionModel;
use mlx_rs::{
    array,
    error::Exception,
    module::{FlattenedModuleParam, Module, ModuleParameters, Param},
    nn,
    ops::ones,
    optimizers::{Optimizer, OptimizerState, Sgd},
    test_utils::{assert_array_eq, tolerances},
    transforms::compile::compile_with_state,
    Array,
};

fn assert_arrays_eq(got: &[Array], expected: &[Array], rtol: f64, atol: f64) {
    assert_eq!(got.len(), expected.len());
    for (got, expected) in got.iter().zip(expected) {
        assert_array_eq(got, expected, rtol, atol);
    }
}

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

fn assert_optimizer_states_eq<O: Optimizer>(
    got: &(LinearFunctionModel, O),
    expected: &(LinearFunctionModel, O),
    rtol: f64,
    atol: f64,
) {
    let got_parameters = got.0.parameters().flatten();
    let expected_parameters = expected.0.parameters().flatten();
    assert_tensor_maps_eq(&got_parameters, &expected_parameters, rtol, atol);

    let got_optimizer = got.1.state().flatten().collect::<HashMap<_, _>>();
    let expected_optimizer = expected.1.state().flatten().collect::<HashMap<_, _>>();
    assert_tensor_maps_eq(&got_optimizer, &expected_optimizer, rtol, atol);
}

#[test]
fn test_compile_module() {
    let loss = |model: &mut LinearFunctionModel, x: &Array| -> Array {
        let y = model.forward(x).unwrap();
        y.square().unwrap().sum(None).unwrap()
    };
    let mut model = LinearFunctionModel::new(None).unwrap();

    let x = ones::<f32>(&[10, 1]).unwrap();
    let x = vec![x];

    let step = move |model: &mut LinearFunctionModel, x: &[Array]| -> Vec<Array> {
        let mut lg = nn::value_and_grad(loss);
        let x = &x[0];
        let (loss, _grad) = lg(model, x).unwrap();
        vec![loss]
    };

    // Check that the original function works
    let original = step(&mut model, x.as_slice());

    // Make sure the compiled function produces the same result
    let mut compiled = compile_with_state(step, None);
    let result = compiled(&mut model, x.as_slice()).unwrap();
    assert_arrays_eq(
        &result,
        &original,
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    let result = compiled(&mut model, x.as_slice()).unwrap();
    assert_arrays_eq(
        &result,
        &original,
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
}

fn compile_module_and_optimizer<O: Optimizer + Clone>(optimizer: O) {
    let initial_model = LinearFunctionModel {
        m: Param::new(array!(1.25)),
        b: Param::new(array!(-0.75)),
    };
    let mut eager_state = (initial_model.clone(), optimizer.clone());
    let mut compiled_state = (initial_model, optimizer);

    let step = |(model, optimizer): &mut (LinearFunctionModel, O), gradients: &[Array]| {
        let mut gradient_map = FlattenedModuleParam::new();
        gradient_map.insert("m".into(), gradients[0].clone());
        gradient_map.insert("b".into(), gradients[1].clone());
        optimizer.update(model, gradient_map).unwrap();
        vec![model.m.value.clone(), model.b.value.clone()]
    };
    let mut compiled = compile_with_state(step, None);
    let gradients = [
        [array!(0.25), array!(-0.5)],
        [array!(-0.75), array!(0.125)],
        [array!(0.5), array!(0.25)],
    ];

    for gradient in gradients {
        let expected = step(&mut eager_state, &gradient);
        let got = compiled(&mut compiled_state, &gradient).unwrap();
        assert_arrays_eq(
            &got,
            &expected,
            tolerances::STANDARD.rtol,
            tolerances::STANDARD.atol,
        );
        assert_optimizer_states_eq(
            &compiled_state,
            &eager_state,
            tolerances::STANDARD.rtol,
            tolerances::STANDARD.atol,
        );
    }
}

/// A simple sanity check for adafactor optimizer
#[test]
fn test_compile_module_and_adafactor_works() {
    compile_module_and_optimizer(mlx_rs::optimizers::Adafactor::new().unwrap());
}

#[test]
fn test_compile_module_and_sgd_consistency() {
    compile_module_and_optimizer(Sgd::new(0.05_f32));
}

#[test]
fn test_compile_module_and_adam_consistency() {
    compile_module_and_optimizer(mlx_rs::optimizers::Adam::new(0.05_f32));
}

#[test]
fn test_compile_module_and_rmsprop_consistency() {
    compile_module_and_optimizer(mlx_rs::optimizers::RmsProp::new(0.05_f32).unwrap());
}

#[test]
fn test_compile_module_and_adagrad_consistency() {
    compile_module_and_optimizer(mlx_rs::optimizers::AdaGrad::new(0.05_f32));
}

#[test]
fn test_compile_module_and_adadelta_consistency() {
    compile_module_and_optimizer(mlx_rs::optimizers::AdaDelta::new(0.05_f32).unwrap());
}

#[test]
fn test_compile_module_and_adamw_consistency() {
    compile_module_and_optimizer(mlx_rs::optimizers::AdamW::new(0.05_f32));
}

#[test]
fn test_compile_module_and_adamax_consistency() {
    compile_module_and_optimizer(mlx_rs::optimizers::Adamax::new(0.05_f32));
}

#[test]
fn test_compile_module_and_lion_consistency() {
    compile_module_and_optimizer(mlx_rs::optimizers::Lion::new(0.05_f32));
}

#[test]
fn test_compile_module_with_error() {
    let loss = |model: &mut LinearFunctionModel, x: &Array| -> Result<Array, Exception> {
        let y = model.forward(x)?;
        y.square()?.sum(None)
    };
    let mut model = LinearFunctionModel::new(&[10]).unwrap();

    let step =
        move |model: &mut LinearFunctionModel, x: &[Array]| -> Result<Vec<Array>, Exception> {
            let mut lg = nn::value_and_grad(loss);
            let x = &x[0];
            let (loss, _grad) = lg(model, x)?;
            Ok(vec![loss])
        };

    // Make sure the compiled function produces the same result
    let mut compiled = compile_with_state(step, None);

    // input with correct shape
    let x_ok = ones::<f32>(&[10, 1]).unwrap();
    let x_ok = vec![x_ok];
    // input with wrong shape
    let x_err = ones::<f32>(&[1, 2, 3]).unwrap();
    let x_err = vec![x_err];

    // Success case
    // Check that the original function works
    let original = step(&mut model, x_ok.as_slice()).unwrap();

    let result = compiled(&mut model, x_ok.as_slice()).unwrap();
    assert_arrays_eq(
        &result,
        &original,
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    let result = compiled(&mut model, x_ok.as_slice()).unwrap();
    assert_arrays_eq(
        &result,
        &original,
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );

    // Error case

    // Check that the original function returns an error
    let original = step(&mut model, x_err.as_slice());
    assert!(original.is_err());
    // Make sure the compiled function also returns an error
    let result = compiled(&mut model, x_err.as_slice());
    assert!(result.is_err());
}

#[test]
#[ignore = "confirmed defect: a failed compiled call leaves model/optimizer state holding \
tracer arrays without primitives (state corruption on error); un-ignore when fixed"]
fn test_compile_module_and_optimizer_with_error() {
    let initial_state = (
        LinearFunctionModel {
            m: Param::new(array!(1.25)),
            b: Param::new(array!(-0.75)),
        },
        Sgd::new(0.05_f32),
    );
    let mut eager_state = initial_state.clone();
    let mut compiled_state = initial_state;

    let step = |(model, optimizer): &mut (LinearFunctionModel, Sgd),
                inputs: &[Array]|
     -> Result<Vec<Array>, Exception> {
        inputs[0].reshape(&[1])?;
        let mut gradients = FlattenedModuleParam::new();
        gradients.insert("m".into(), inputs[1].clone());
        gradients.insert("b".into(), inputs[2].clone());
        optimizer.update(model, gradients)?;
        Ok(vec![model.m.value.clone(), model.b.value.clone()])
    };
    let mut compiled = compile_with_state(step, None);
    let x_ok = [array!(1.0), array!(0.25), array!(-0.5)];

    for _ in 0..2 {
        let expected = step(&mut eager_state, &x_ok).unwrap();
        let got = compiled(&mut compiled_state, &x_ok).unwrap();
        assert_arrays_eq(
            &got,
            &expected,
            tolerances::STANDARD.rtol,
            tolerances::STANDARD.atol,
        );
        assert_optimizer_states_eq(
            &compiled_state,
            &eager_state,
            tolerances::STANDARD.rtol,
            tolerances::STANDARD.atol,
        );
    }

    let eager_before_error = eager_state.clone();
    let compiled_before_error = compiled_state.clone();
    let x_err = [array!([1.0, 2.0]), array!(0.25), array!(-0.5)];
    assert!(step(&mut eager_state, &x_err).is_err());
    assert!(compiled(&mut compiled_state, &x_err).is_err());
    assert_optimizer_states_eq(
        &eager_state,
        &eager_before_error,
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
    assert_optimizer_states_eq(
        &compiled_state,
        &compiled_before_error,
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
}
