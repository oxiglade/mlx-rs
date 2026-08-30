use mlx_rs::{
    transforms::{fallible_jvp, jvp},
    Array, Device, Dtype,
};
use std::any::Any;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::process::Command;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

const PANIC_CHILD: &str = "MLX_RS_FFI_PANIC_CHILD";
const ERROR_REGISTRATION_CHILD: &str = "MLX_RS_ERROR_REGISTRATION_CHILD";
const CONCURRENT_ERRORS_CHILD: &str = "MLX_RS_CONCURRENT_ERRORS_CHILD";
const ERROR_WORKERS: usize = 8;

#[test]
fn safetensors_metadata_iteration_does_not_leak() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("metadata.safetensors");
    let arrays: HashMap<String, Array> = HashMap::new();
    let metadata = HashMap::from([("model".to_owned(), "fixture".to_owned())]);
    Array::save_safetensors(&arrays, &Some(metadata), &path).unwrap();

    for _ in 0..200 {
        let (data, metadata) = Array::load_safetensors_with_metadata(&path).unwrap();
        assert!(data.is_empty());
        assert_eq!(metadata.get("model").map(String::as_str), Some("fixture"));
    }
}

#[test]
fn closure_panic_returns_to_rust_before_resuming() {
    if std::env::var_os(PANIC_CHILD).is_some() {
        run_closure_panic_child();
        return;
    }

    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "closure_panic_returns_to_rust_before_resuming",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(PANIC_CHILD, "1")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "child status: {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn run_closure_panic_child() {
    let primal = Array::from_f32(2.0);
    let tangent = Array::from_f32(1.0);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = jvp(
            |_| -> Vec<Array> { panic!("closure panic payload") },
            &[primal],
            &[tangent],
        );
    }))
    .expect_err("the closure panic should resume in Rust");

    assert_eq!(
        panic_message(panic).as_deref(),
        Some("closure panic payload")
    );

    let primal = Array::from_f32(2.0);
    let tangent = Array::from_f32(1.0);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = fallible_jvp(
            |_| -> mlx_rs::error::Result<Vec<Array>> { panic!("fallible closure panic payload") },
            &[primal],
            &[tangent],
        );
    }))
    .expect_err("the fallible closure panic should resume in Rust");

    assert_eq!(
        panic_message(panic).as_deref(),
        Some("fallible closure panic payload")
    );
    assert_eq!((&Array::from_int(20) + 22).item::<i32>(), 42);
}

fn panic_message(payload: Box<dyn Any + Send>) -> Option<String> {
    payload
        .downcast_ref::<&'static str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
}

#[test]
fn float64_is_float_and_inexact() {
    assert!(Dtype::Float64.is_float());
    assert!(Dtype::Float64.is_inexact());
}

#[test]
fn first_contact_error_registration_is_thread_safe() {
    if std::env::var_os(ERROR_REGISTRATION_CHILD).is_some() {
        run_first_contact_error_registration_child();
        return;
    }

    assert_subprocess_success(
        "first_contact_error_registration_is_thread_safe",
        ERROR_REGISTRATION_CHILD,
    );
}

fn run_first_contact_error_registration_child() {
    let arrays = (0..ERROR_WORKERS)
        .map(mismatched_arrays)
        .collect::<Vec<_>>();
    let barrier = Arc::new(Barrier::new(ERROR_WORKERS));
    let handles = arrays
        .into_iter()
        .enumerate()
        .map(|(worker, (lhs, rhs, lhs_len, rhs_len))| {
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                assert_own_broadcast_error(lhs.add(&rhs), worker, lhs_len, rhs_len);
            })
        })
        .collect::<Vec<_>>();

    join_workers(handles);
}

#[test]
fn concurrent_invoke_errors_stay_on_the_calling_thread() {
    if std::env::var_os(CONCURRENT_ERRORS_CHILD).is_some() {
        run_concurrent_invoke_errors_child();
        return;
    }

    assert_subprocess_success(
        "concurrent_invoke_errors_stay_on_the_calling_thread",
        CONCURRENT_ERRORS_CHILD,
    );
}

fn run_concurrent_invoke_errors_child() {
    const OPERATIONS: usize = 20;

    let barrier = Arc::new(Barrier::new(ERROR_WORKERS));
    let handles = (0..ERROR_WORKERS)
        .map(|worker| {
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                for _ in 0..OPERATIONS {
                    let (lhs, rhs, lhs_len, rhs_len) = mismatched_arrays(worker);
                    barrier.wait();
                    assert_own_broadcast_error(lhs.add(&rhs), worker, lhs_len, rhs_len);
                }
            })
        })
        .collect::<Vec<_>>();

    join_workers(handles);
}

fn mismatched_arrays(worker: usize) -> (Array, Array, usize, usize) {
    let lhs_len = worker + 3;
    let rhs_len = worker + ERROR_WORKERS + 3;
    let lhs = Array::from_slice(&vec![worker as i32; lhs_len], &[lhs_len as i32]);
    let rhs = Array::from_slice(&vec![worker as i32; rhs_len], &[rhs_len as i32]);
    (lhs, rhs, lhs_len, rhs_len)
}

fn assert_own_broadcast_error(
    result: mlx_rs::error::Result<Array>,
    worker: usize,
    lhs_len: usize,
    rhs_len: usize,
) {
    let error = result.expect_err("mismatched shapes must fail during invocation");
    let expected = broadcast_error_marker(lhs_len, rhs_len);
    assert!(
        error.what().contains(&expected),
        "worker {worker} received unexpected error: {}",
        error.what()
    );

    for other in 0..ERROR_WORKERS {
        if other != worker {
            let marker = broadcast_error_marker(other + 3, other + ERROR_WORKERS + 3);
            assert!(
                !error.what().contains(&marker),
                "worker {worker} received worker {other}'s error: {}",
                error.what()
            );
        }
    }
}

fn broadcast_error_marker(lhs_len: usize, rhs_len: usize) -> String {
    format!("Shapes ({lhs_len}) and ({rhs_len}) cannot be broadcast.")
}

fn join_workers(handles: Vec<thread::JoinHandle<()>>) {
    for handle in handles {
        handle.join().unwrap();
    }
}

fn assert_subprocess_success(test_name: &str, child_env: &str) {
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
        .env(child_env, "1")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "child status: {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "historical MLX 0.30.6 shared-default reproducer; 0.32.2 is covered by stream_admission"]
fn historical_shared_default_stream_abort_on_mlx_0_30_6() {
    const WORKERS: usize = 8;
    const OPERATIONS: usize = 100;

    Device::set_default(&Device::gpu());
    let barrier = Arc::new(Barrier::new(WORKERS));
    let initialization = Arc::new(Mutex::new(()));
    let handles = (0..WORKERS)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            let initialization = Arc::clone(&initialization);
            thread::spawn(move || {
                let input = {
                    let _guard = initialization.lock().unwrap();
                    let input = Array::from_slice(&[1.0_f32; 4096], &[64, 64]);
                    (&input + 1.0).eval().unwrap();
                    input
                };
                for _ in 0..OPERATIONS {
                    barrier.wait();
                    (&input + 1.0).eval().unwrap();
                }
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        handle.join().unwrap();
    }
}
