use std::{
    any::Any,
    cell::Cell,
    collections::{BTreeSet, HashMap},
    env, fs,
    panic::{catch_unwind, AssertUnwindSafe},
    path::PathBuf,
    process::Command,
    rc::Rc,
    thread,
    time::{Duration, Instant},
};

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{FlattenedModuleParam, ModuleParameters, Param, Parameter},
    optimizers::{Adam, AdamBuilder, Optimizer, OptimizerState, Sgd, SgdBuilder},
    test_utils::assert_array_eq_with_context,
    transforms::compile::{clear_cache, compile, compile_with_state},
    Array,
};

const RTOL: f64 = 1.0e-6;
const ATOL: f64 = 1.0e-7;
const TINY_LEAVES: usize = 512;

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
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../conformance/state/state.safetensors");
        let tensors = Array::load_safetensors(path).expect(
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

    fn gradients(&self, step: usize) -> [Array; 2] {
        [
            self.tensor(&format!("input.gradient.step{step}.weight"))
                .clone(),
            self.tensor(&format!("input.gradient.step{step}.bias"))
                .clone(),
        ]
    }
}

fn assert_named(got: &Array, expected: &Array, context: &str) {
    assert_array_eq_with_context(got, expected, RTOL, ATOL, context);
}

fn assert_all_leaves(got: &[Array], expected: &[Array], case: &str) {
    assert_eq!(got.len(), expected.len(), "{case}.leaf_count");
    for (index, (got, expected)) in got.iter().zip(expected).enumerate() {
        assert_named(got, expected, &format!("{case}.leaf.{index}"));
    }
}

fn assert_model_optimizer<O: Optimizer>(
    got: &(TinyModel, O),
    expected: &(TinyModel, O),
    case: &str,
) {
    let got_parameters = got.0.parameters().flatten();
    let expected_parameters = expected.0.parameters().flatten();
    let got_keys = got_parameters.keys().cloned().collect::<BTreeSet<_>>();
    let expected_keys = expected_parameters.keys().cloned().collect::<BTreeSet<_>>();
    assert_eq!(got_keys, expected_keys, "{case}.parameter_keys");
    for key in expected_keys {
        assert_named(
            got_parameters[&key],
            expected_parameters[&key],
            &format!("{case}.parameter.{key}"),
        );
    }

    let got_state = got.1.state().flatten().collect::<HashMap<_, _>>();
    let expected_state = expected.1.state().flatten().collect::<HashMap<_, _>>();
    let got_keys = got_state.keys().cloned().collect::<BTreeSet<_>>();
    let expected_keys = expected_state.keys().cloned().collect::<BTreeSet<_>>();
    assert_eq!(got_keys, expected_keys, "{case}.optimizer_state_keys");
    for key in expected_keys {
        assert_named(
            got_state[&key],
            expected_state[&key],
            &format!("{case}.optimizer_state.{key}"),
        );
    }
}

fn adam_frozen_step((model, optimizer): &mut (TinyModel, Adam), gradients: &[Array]) -> Vec<Array> {
    let mut gradient_map = FlattenedModuleParam::new();
    gradient_map.insert("weight".into(), gradients[0].clone());
    optimizer.update(model, gradient_map).unwrap();
    vec![model.weight.value.clone(), model.bias.value.clone()]
}

#[test]
fn frozen_parameter_compiled_updates_match_unfrozen_oracle_on_trainable_keys() {
    let fixture = Fixture::load();
    let mut model = fixture.model();
    model.bias.freeze(false);
    let optimizer = AdamBuilder::new(0.025_f32)
        .betas((0.8_f32, 0.95_f32))
        .eps(1.0e-6_f32)
        .build()
        .unwrap();
    let mut state = (model, optimizer);
    let mut compiled = compile_with_state(adam_frozen_step, None);

    for step in 1..=3 {
        compiled(&mut state, &fixture.gradients(step)).unwrap();
        assert_named(
            &state.0.weight,
            fixture.tensor(&format!("adam.step{step}.param.weight")),
            &format!("compile.frozen.step{step}.parameter.weight"),
        );
        assert_named(
            &state.0.bias,
            fixture.tensor("input.param.bias"),
            &format!("compile.frozen.step{step}.parameter.bias"),
        );

        let optimizer_state = state.1.state().flatten().collect::<HashMap<_, _>>();
        let expected_keys = [Rc::<str>::from("weight.0"), Rc::<str>::from("weight.1")]
            .into_iter()
            .collect::<BTreeSet<_>>();
        assert_eq!(
            optimizer_state.keys().cloned().collect::<BTreeSet<_>>(),
            expected_keys,
            "compile.frozen.step{step}.optimizer_state_keys"
        );
        for key in expected_keys {
            assert_named(
                optimizer_state[&key],
                fixture.tensor(&format!("adam.step{step}.state.{key}")),
                &format!("compile.frozen.step{step}.optimizer_state.{key}"),
            );
        }
    }
}

fn many_suffix_step(state: &mut Vec<Array>, args: &[Array]) -> Vec<Array> {
    for leaf in state.iter_mut().take(64) {
        *leaf = leaf.add(&args[0]).unwrap();
    }
    vec![args[0].square().unwrap()]
}

fn many_interleaved_step(state: &mut Vec<Array>, args: &[Array]) -> Vec<Array> {
    for (index, leaf) in state.iter_mut().enumerate() {
        if index % 3 == 1 {
            *leaf = leaf.add(&args[0]).unwrap();
        }
    }
    vec![args[0].square().unwrap()]
}

fn tiny_state() -> Vec<Array> {
    (0..TINY_LEAVES)
        .map(|index| Array::from_f32(index as f32 / 16.0 - 7.0))
        .collect()
}

#[test]
fn unchanged_state_pruning_with_many_tiny_suffix_leaves() {
    let mut expected = tiny_state();
    let mut got = expected.clone();
    let input = [array!(0.125_f32)];
    let expected_output = many_suffix_step(&mut expected, &input);
    let mut compiled = compile_with_state(many_suffix_step, None);
    let got_output = compiled(&mut got, &input).unwrap();

    assert_all_leaves(&got_output, &expected_output, "compile.pruned.output");
    assert_all_leaves(&got, &expected, "compile.pruned");
}

#[test]
fn partial_pruning_preserves_interleaved_leaf_slots() {
    let mut expected = tiny_state();
    let mut got = expected.clone();
    let input = [array!(-0.375_f32)];
    let expected_output = many_interleaved_step(&mut expected, &input);
    let mut compiled = compile_with_state(many_interleaved_step, None);
    let got_output = compiled(&mut got, &input).unwrap();

    assert_all_leaves(&got_output, &expected_output, "compile.partial.output");
    assert_all_leaves(&got, &expected, "compile.partial");
}

fn nested_step((model, optimizer): &mut (TinyModel, Sgd), gradients: &[Array]) -> Vec<Array> {
    let mut gradient_map = FlattenedModuleParam::new();
    gradient_map.insert("weight".into(), gradients[0].clone());
    gradient_map.insert("bias".into(), gradients[1].clone());
    optimizer.update(model, gradient_map).unwrap();
    vec![model.weight.value.clone(), model.bias.value.clone()]
}

#[test]
fn nested_module_optimizer_tuple_state() {
    let fixture = Fixture::load();
    let optimizer = SgdBuilder::new(0.035_f32)
        .momentum(0.8_f32)
        .weight_decay(0.03_f32)
        .nesterov(true)
        .build()
        .unwrap();
    let initial = (fixture.model(), optimizer);
    let mut expected = initial.clone();
    let mut got = initial;
    let mut compiled = compile_with_state(nested_step, None);

    for step in 1..=3 {
        let gradients = fixture.gradients(step);
        let expected_output = nested_step(&mut expected, &gradients);
        let got_output = compiled(&mut got, &gradients).unwrap();
        assert_all_leaves(
            &got_output,
            &expected_output,
            &format!("compile.nested.step{step}.output"),
        );
        assert_model_optimizer(&got, &expected, &format!("compile.nested.step{step}"));
    }
}

fn repeated_step(state: &mut Vec<Array>, args: &[Array]) -> Vec<Array> {
    state[0] = state[0].add(&args[0]).unwrap();
    state[1] = state[1].multiply(&array!(0.75_f32)).unwrap();
    state[2] = state[2].subtract(&args[0]).unwrap();
    vec![state[0].add(&state[2]).unwrap()]
}

#[test]
fn repeated_calls_advance_state_across_five_cached_calls() {
    let initial = vec![
        array!(0.5_f32),
        array!(-2.0_f32),
        array!(1.25_f32),
        array!(9.0_f32),
    ];
    let mut expected = initial.clone();
    let mut got = initial;
    let mut compiled = compile_with_state(repeated_step, None);

    for call in 1..=5 {
        let input = [array!(call as f32 * 0.2 - 0.35)];
        let expected_output = repeated_step(&mut expected, &input);
        let got_output = compiled(&mut got, &input).unwrap();
        assert_all_leaves(
            &got_output,
            &expected_output,
            &format!("compile.repeated.call{call}.output"),
        );
        assert_all_leaves(&got, &expected, &format!("compile.repeated.call{call}"));
    }
}

#[test]
fn cached_state_rejects_count_and_layout_changes() {
    let args = [array!(0.25_f32)];
    let mut count_state = vec![
        array!(0.5_f32),
        array!(-2.0_f32),
        array!(1.25_f32),
        array!(9.0_f32),
    ];
    let mut count_compiled = compile_with_state(repeated_step, None);
    count_compiled(&mut count_state, &args).unwrap();
    count_state.push(array!(4.0_f32));
    let count_error = count_compiled(&mut count_state, &args).unwrap_err();
    assert!(count_error
        .what()
        .contains("state layout changed at call input"));
    assert_eq!(count_state.len(), 5);

    let mut layout_state = vec![
        array!(0.5_f32),
        array!(-2.0_f32),
        array!(1.25_f32),
        array!(9.0_f32),
    ];
    let mut layout_compiled = compile_with_state(repeated_step, None);
    layout_compiled(&mut layout_state, &args).unwrap();
    layout_state[0] = array!([0.75_f32]);
    let layout_error = layout_compiled(&mut layout_state, &args).unwrap_err();
    assert!(layout_error
        .what()
        .contains("state layout changed at call input"));
    assert_eq!(layout_state[0].shape(), &[1]);
}

#[test]
fn nested_cold_cache_compilation_completes_without_deadlock() {
    const CHILD_ENV: &str = "MLX_RS_NESTED_COLD_CACHE_CHILD";
    const MARKER_ENV: &str = "MLX_RS_NESTED_COLD_CACHE_MARKER";

    if env::var_os(CHILD_ENV).is_some() {
        let marker = PathBuf::from(env::var_os(MARKER_ENV).unwrap());
        let trace_calls = Rc::new(Cell::new(0));
        let trace_calls_inner = Rc::clone(&trace_calls);
        let mut inner = compile(
            move |input: &Array| -> Result<Array, Exception> {
                trace_calls_inner.set(trace_calls_inner.get() + 1);
                fs::write(&marker, b"inner trace entered").unwrap();
                input.square()
            },
            None,
        );
        let mut outer = compile_with_state(
            move |state: &mut Vec<Array>, args: &[Array]| -> Result<Vec<Array>, Exception> {
                let output = inner(&args[0])?;
                state[0] = state[0].add(&output)?;
                Ok(vec![output])
            },
            None,
        );
        let mut state = vec![array!(0.0_f32)];
        let output = outer(&mut state, &[array!(2.0_f32)]).unwrap();
        assert_eq!(trace_calls.get(), 1);
        assert_named(&output[0], &array!(4.0_f32), "compile.nested_cold.output");
        return;
    }

    let directory = tempfile::tempdir().unwrap();
    let marker = directory.path().join("trace-entered");
    let mut child = Command::new(env::current_exe().unwrap())
        .args([
            "--exact",
            "nested_cold_cache_compilation_completes_without_deadlock",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .env(MARKER_ENV, &marker)
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "nested cold-cache child failed: {status}");
            assert!(marker.is_file(), "inner cold trace did not run");
            break;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            let _ = child.wait();
            panic!(
                "nested cold-cache child exceeded 30 seconds; inner trace entered: {}",
                marker.is_file()
            );
        }
        thread::sleep(Duration::from_millis(25));
    }
}

fn fallible_step(state: &mut Vec<Array>, args: &[Array]) -> Result<Vec<Array>, Exception> {
    args[0].reshape(&[1])?;
    state[0] = state[0].add(&args[1])?;
    state[1] = state[1].multiply(&args[1])?;
    Ok(vec![state[0].subtract(&state[1])?])
}

#[test]
fn fallible_success_after_failure_resumes_oracle_trajectory() {
    let initial = vec![array!(0.75_f32), array!(-1.25_f32), array!(4.0_f32)];
    let mut expected = initial.clone();
    let mut got = initial;
    let mut compiled = compile_with_state(fallible_step, None);
    let first = [array!(1.0_f32), array!(0.2_f32)];
    let expected_output = fallible_step(&mut expected, &first).unwrap();
    let got_output = compiled(&mut got, &first).unwrap();
    assert_all_leaves(
        &got_output,
        &expected_output,
        "compile.fallible.before.output",
    );
    assert_all_leaves(&got, &expected, "compile.fallible.before");

    let before_failure = got.clone();
    let failing = [array!([1.0_f32, 2.0]), array!(0.3_f32)];
    assert!(compiled(&mut got, &failing).is_err());
    assert_all_leaves(&got, &before_failure, "compile.fallible.failure_atomicity");

    for call in 1..=3 {
        let input = [array!(1.0_f32), array!(call as f32 * -0.15 + 0.4)];
        let expected_output = fallible_step(&mut expected, &input).unwrap();
        let got_output = compiled(&mut got, &input).unwrap();
        assert_all_leaves(
            &got_output,
            &expected_output,
            &format!("compile.fallible.resumed{call}.output"),
        );
        assert_all_leaves(&got, &expected, &format!("compile.fallible.resumed{call}"));
    }
}

thread_local! {
    static FALLIBLE_CALLS: Cell<usize> = const { Cell::new(0) };
    static TRACE_CALLS: Cell<usize> = const { Cell::new(0) };
}

fn trace_once_step(state: &mut Vec<Array>, _args: &[Array]) -> Result<Vec<Array>, Exception> {
    TRACE_CALLS.with(|counter| counter.set(counter.get() + 1));
    state[0] = state[0].add(&array!(1.0_f32))?;
    Ok(vec![state[0].clone()])
}

fn fail_first_trace_step(
    (model, optimizer): &mut (TinyModel, Adam),
    gradients: &[Array],
) -> Result<Vec<Array>, Exception> {
    let call = FALLIBLE_CALLS.with(|counter| {
        let call = counter.get() + 1;
        counter.set(call);
        call
    });
    let mut gradient_map = FlattenedModuleParam::new();
    gradient_map.insert("weight".into(), gradients[0].clone());
    optimizer.update(model, gradient_map)?;
    if call == 1 {
        array!(1.0_f32).reshape(&[2])?;
    }
    Ok(vec![model.weight.value.clone(), model.bias.value.clone()])
}

fn always_fail_step(state: &mut Vec<Array>, _args: &[Array]) -> Result<Vec<Array>, Exception> {
    FALLIBLE_CALLS.with(|counter| counter.set(counter.get() + 1));
    state[0] = state[0].add(&array!(1.0_f32))?;
    array!(1.0_f32).reshape(&[2])?;
    Ok(Vec::new())
}

#[test]
fn failed_trace_rolls_back_and_caller_retry_recovers_growth() {
    FALLIBLE_CALLS.with(|counter| counter.set(0));
    let fixture = Fixture::load();
    let mut model = fixture.model();
    model.bias.freeze(false);
    let optimizer = AdamBuilder::new(0.025_f32)
        .betas((0.8_f32, 0.95_f32))
        .eps(1.0e-6_f32)
        .build()
        .unwrap();
    let initial = (model, optimizer);
    let mut expected = initial.clone();
    let mut state = initial.clone();
    let args = fixture.gradients(1);
    let mut compiled = compile_with_state(fail_first_trace_step, None);

    assert!(compiled(&mut state, &args).is_err());
    assert_eq!(FALLIBLE_CALLS.with(Cell::get), 1);
    assert_eq!(state.0.parameters().flatten().len(), 2);
    assert_eq!(state.1.state().flatten().count(), 0);
    assert_model_optimizer(&state, &initial, "compile.no_retry.failure");

    let mut trace_calls = 1;
    for call in 1..=4 {
        let expected_output = adam_frozen_step(&mut expected, &args);
        let output = compiled(&mut state, &args).unwrap();
        let calls = FALLIBLE_CALLS.with(Cell::get);
        assert!(
            (trace_calls..=3).contains(&calls),
            "compile.no_retry.call{call}.counter={calls} after {trace_calls}"
        );
        trace_calls = calls;
        assert_all_leaves(
            &output,
            &expected_output,
            &format!("compile.no_retry.call{call}.output"),
        );
        assert_model_optimizer(&state, &expected, &format!("compile.no_retry.call{call}"));
    }
    assert!((2..=3).contains(&trace_calls));
}

#[test]
fn compiled_state_traces_once_and_advances_state_on_cache_hits() {
    TRACE_CALLS.with(|counter| counter.set(0));
    let mut state = vec![array!(0.0_f32)];
    let args: [Array; 0] = [];
    let mut compiled = compile_with_state(trace_once_step, None);

    for call in 1..=4 {
        let output = compiled(&mut state, &args).unwrap();
        assert_eq!(
            TRACE_CALLS.with(Cell::get),
            1,
            "compile.trace_once.call{call}.counter"
        );
        assert_named(
            &output[0],
            &array!(call as f32),
            &format!("compile.trace_once.call{call}.output.0"),
        );
        assert_named(
            &state[0],
            &array!(call as f32),
            &format!("compile.trace_once.call{call}.state.0"),
        );
    }
}

#[test]
fn clear_cache_resolves_the_current_thread_cache() {
    clear_cache();
    TRACE_CALLS.with(|counter| counter.set(0));
    let mut state = vec![array!(0.0_f32)];
    let args: [Array; 0] = [];
    let mut compiled = compile_with_state(trace_once_step, None);

    compiled(&mut state, &args).unwrap();
    assert_eq!(TRACE_CALLS.with(Cell::get), 1);
    clear_cache();
    compiled(&mut state, &args).unwrap();
    assert_eq!(TRACE_CALLS.with(Cell::get), 2);
    assert_named(&state[0], &array!(2.0_f32), "compile.clear_cache.state.0");
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
fn fault_reordered_partial_state_fails_leaf_slot() {
    let mut state = tiny_state();
    let input = [array!(-0.375_f32)];
    many_interleaved_step(&mut state, &input);
    expect_named_failure("compile.partial.leaf.1", || {
        assert_named(&state[1], &state[4], "compile.partial.leaf.1");
    });
}

#[test]
fn fault_shifted_output_split_fails_count() {
    let mut state = vec![
        array!(1.0_f32),
        array!(2.0_f32),
        array!(3.0_f32),
        array!(4.0_f32),
    ];
    let input = [array!(0.25_f32)];
    let mut compiled = compile_with_state(
        |state: &mut Vec<Array>, args: &[Array]| {
            state[0] = state[0].add(&args[0]).unwrap();
            vec![state[0].clone(), state[1].clone()]
        },
        None,
    );
    let actual = compiled(&mut state, &input).unwrap();
    let shifted_expectation = [actual[0].clone()];
    expect_named_failure("compile.output_count", || {
        assert_eq!(
            actual.len(),
            shifted_expectation.len(),
            "compile.output_count"
        );
    });
}

#[test]
fn fault_seeded_duplicate_retry_fails_counter_check() {
    FALLIBLE_CALLS.with(|counter| counter.set(0));
    let mut state = vec![array!(0.0_f32)];
    let args: [Array; 0] = [];
    let mut compiled = compile_with_state(always_fail_step, None);
    assert!(compiled(&mut state, &args).is_err());
    assert_named(&state[0], &array!(0.0_f32), "compile.retry.failed.state.0");
    expect_named_failure("compile.retry.counter", || {
        assert_eq!(FALLIBLE_CALLS.with(Cell::get), 2, "compile.retry.counter");
    });
}
