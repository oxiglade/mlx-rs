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
#[ignore = "MLX v0.30.6 has no thread-local streams; concurrent GPU operations abort"]
fn concurrent_gpu_operations_abort_without_thread_local_streams() {
    const WORKERS: usize = 8;
    const OPERATIONS: usize = 100;

    Device::set_default(&Device::gpu());
    let barrier = Arc::new(Barrier::new(WORKERS));
    // Serial setup isolates the shared-stream abort from mlx-c's global error-handler race.
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
