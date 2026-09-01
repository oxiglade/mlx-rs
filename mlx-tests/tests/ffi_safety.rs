use mlx_rs::{
    fast,
    io::{GgufError, GgufFile, GgufMetadataKind},
    linalg, memory,
    ops::{
        indexing::{
            ArrayIndexOp, Ellipsis, IndexUpdateError, IntoStrideBy, TryIndexUpdateOp, UpdateMode,
        },
        ContiguousOptions, CountNonzeroOptions, TraceOptions,
    },
    transforms::{fallible_jvp, jvp},
    with_stream, Array, Axes, Device, Dtype, Stream,
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
const MEMORY_CONTROL_CHILD: &str = "MLX_RS_MEMORY_CONTROL_CHILD";
const ERROR_WORKERS: usize = 8;

macro_rules! assert_not_impl_any {
    ($type:ty: $($trait:path),+ $(,)?) => {
        const _: fn() = || {
            trait AmbiguousIfImpl<T: ?Sized> {
                fn check() {}
            }
            impl<T: ?Sized> AmbiguousIfImpl<()> for T {}
            $({
                struct Invalid;
                impl<T: ?Sized + $trait> AmbiguousIfImpl<Invalid> for T {}
            })+
            let _ = <$type as AmbiguousIfImpl<_>>::check;
        };
    };
}

assert_not_impl_any!(GgufFile: Clone, Send, Sync);

fn gguf_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fixture.gguf");
    let mut file = GgufFile::new().unwrap();
    file.insert_array("tensor", &Array::from_slice(&[1_i32, 2, 3], &[3]))
        .unwrap();
    file.insert_metadata("array", Array::from_slice(&[4_i16, 5], &[2]))
        .unwrap();
    file.insert_metadata("string", "value").unwrap();
    file.insert_metadata("strings", vec!["one".to_owned(), "two".to_owned()])
        .unwrap();
    file.save(&path).unwrap();
    (directory, path)
}

#[test]
fn gguf_handles_and_output_vectors_do_not_leak() {
    let (_directory, path) = gguf_fixture();
    for _ in 0..200 {
        drop(GgufFile::new().unwrap());
        let file = GgufFile::load(&path).unwrap();
        assert_eq!(file.array_keys().unwrap(), ["tensor"]);
        assert_eq!(
            file.get_metadata_string("string").unwrap().as_deref(),
            Some("value")
        );
        assert_eq!(
            file.get_metadata_strings("strings").unwrap().unwrap(),
            ["one", "two"]
        );
    }
}

#[test]
fn failed_gguf_load_releases_the_empty_destination() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("invalid.gguf");
    std::fs::write(&path, b"not a GGUF file").unwrap();
    for _ in 0..200 {
        assert!(matches!(
            GgufFile::load(&path),
            Err(GgufError::Exception(_))
        ));
    }
}

#[test]
fn gguf_failed_getters_release_initialized_outputs() {
    let (_directory, path) = gguf_fixture();
    let file = GgufFile::load(path).unwrap();
    for _ in 0..200 {
        assert!(file.get_array("missing").unwrap().is_none());
        assert!(file.get_metadata_array("missing").unwrap().is_none());
        assert!(file.get_metadata_string("missing").unwrap().is_none());
        assert!(file.get_metadata_strings("missing").unwrap().is_none());
        assert!(matches!(
            file.get_metadata_array("string"),
            Err(GgufError::WrongMetadataKind {
                expected: GgufMetadataKind::Array,
                actual: GgufMetadataKind::String,
                ..
            })
        ));
        assert!(matches!(
            file.get_metadata_string("strings"),
            Err(GgufError::WrongMetadataKind {
                expected: GgufMetadataKind::String,
                actual: GgufMetadataKind::Strings,
                ..
            })
        ));
        assert!(matches!(
            file.get_metadata_strings("array"),
            Err(GgufError::WrongMetadataKind {
                expected: GgufMetadataKind::Strings,
                actual: GgufMetadataKind::Array,
                ..
            })
        ));
    }
}

#[test]
fn gguf_extracted_arrays_outlive_the_container() {
    let (_directory, path) = gguf_fixture();
    let file = GgufFile::load(path).unwrap();
    let tensor = file.get_array("tensor").unwrap().unwrap();
    let metadata = file.get_metadata_array("array").unwrap().unwrap();
    drop(file);
    tensor.eval().unwrap();
    metadata.eval().unwrap();
    assert_eq!(tensor.as_slice::<i32>(), &[1, 2, 3]);
    assert_eq!(metadata.as_slice::<i16>(), &[4, 5]);
}

#[test]
fn gguf_container_retains_inserted_arrays() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("owned.gguf");
    let mut file = GgufFile::new().unwrap();
    let tensor = Array::from_slice(&[1_i32, 2], &[2]);
    file.insert_array("tensor", &tensor).unwrap();
    drop(tensor);
    let metadata = Array::from_slice(&[3_i32, 4], &[2]);
    file.insert_metadata("metadata", &metadata).unwrap();
    drop(metadata);
    file.save(&path).unwrap();

    let loaded = GgufFile::load(path).unwrap();
    assert_eq!(
        loaded
            .get_array("tensor")
            .unwrap()
            .unwrap()
            .as_slice::<i32>(),
        &[1, 2]
    );
    assert_eq!(
        loaded
            .get_metadata_array("metadata")
            .unwrap()
            .unwrap()
            .as_slice::<i32>(),
        &[3, 4]
    );
}

#[test]
fn gguf_temporary_vectors_and_early_errors_are_owned_once() {
    for _ in 0..200 {
        let mut file = GgufFile::new().unwrap();
        file.insert_metadata("strings", vec!["a".to_owned(), "b".to_owned()])
            .unwrap();
        assert!(matches!(
            file.insert_metadata("strings", vec!["c".to_owned()]),
            Err(GgufError::MetadataKeyAlreadyExists { .. })
        ));
        assert!(matches!(
            file.insert_metadata("bad\0key", vec!["c".to_owned()]),
            Err(GgufError::InteriorNul)
        ));
        assert!(matches!(
            file.insert_metadata("bad-value", vec!["c\0d".to_owned()]),
            Err(GgufError::InteriorNul)
        ));
        let moved = { file };
        drop(moved);
    }
}

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
fn slogdet_outputs_and_error_guards_release_every_destination() {
    with_stream(&Stream::cpu(), || {
        for _ in 0..200 {
            let matrix = Array::from_slice(&[2.0_f32, 0.0, 0.0, 3.0], &[2, 2]);
            let result = linalg::slogdet(&matrix).unwrap();
            result.sign.eval().unwrap();
            result.log_abs_det.eval().unwrap();

            let non_square = Array::from_slice(&[1.0_f32; 6], &[2, 3]);
            assert!(linalg::slogdet(&non_square).is_err());
        }
    });
}

#[test]
fn unstack_releases_zero_and_nonzero_output_vectors_on_every_path() {
    let values = (0..257).collect::<Vec<i32>>();
    let outputs = Array::from_slice(&values, &[257]).unstack(0).unwrap();
    assert_eq!(outputs.len(), 257);

    for _ in 0..200 {
        let input = Array::from_slice(&[1_i32, 2, 3, 4, 5, 6], &[2, 3]);
        let outputs = input.unstack(0).unwrap();
        assert_eq!(outputs.len(), 2);
        for output in outputs {
            output.eval().unwrap();
        }

        let empty = Array::from_slice::<i32>(&[], &[0, 3]);
        assert!(empty.unstack(0).unwrap().is_empty());

        assert!(input.unstack(2).is_err());
    }
}

#[test]
fn partial_vector_extraction_releases_transferred_arrays_and_container() {
    for _ in 0..200 {
        let arrays = [Array::from_int(1), Array::from_int(2)];
        let handles = [arrays[0].as_ptr(), arrays[1].as_ptr()];
        let vector = unsafe { mlx_sys::mlx_vector_array_new_data(handles.as_ptr(), handles.len()) };

        let mut first = unsafe { mlx_sys::mlx_array_new() };
        assert_eq!(
            unsafe { mlx_sys::mlx_vector_array_get(&mut first, vector, 0) },
            0
        );
        let transferred = unsafe { Array::from_ptr(first) };

        let mut failed = unsafe { mlx_sys::mlx_array_new() };
        assert_ne!(
            unsafe { mlx_sys::mlx_vector_array_get(&mut failed, vector, 2) },
            0
        );
        unsafe { mlx_sys::mlx_array_free(failed) };

        drop(transferred);
        assert_eq!(unsafe { mlx_sys::mlx_vector_array_free(vector) }, 0);
    }
}

#[test]
fn empty_axis_vectors_and_new_error_paths_are_repeatable() {
    for _ in 0..200 {
        let input = Array::from_slice(&[0_i32, 1, 2, 0], &[2, 2]);
        input
            .count_nonzero(CountNonzeroOptions {
                axes: Axes::Axes(Vec::new()),
                keep_dims: false,
            })
            .unwrap()
            .eval()
            .unwrap();
        input.flip(Axes::Axes(Vec::new())).unwrap().eval().unwrap();

        assert!(input.diff(-1, 0).is_err());
        assert!(input
            .search_sorted(Array::from_int(1), mlx_rs::ops::SearchSide::Left)
            .is_err());
        assert!(input
            .trace(TraceOptions {
                axis2: 3,
                ..TraceOptions::default()
            })
            .is_err());
        let complex = Array::from_complex(mlx_rs::complex64::new(1.0, 2.0));
        assert!(complex.trunc().is_err());
        assert!(mlx_rs::ops::vecdot(&input, &input, 3).is_err());
    }
}

#[test]
fn memory_get_set_and_restore_paths_are_repeatable() {
    if std::env::var_os(MEMORY_CONTROL_CHILD).is_none() {
        assert_subprocess_success(
            "memory_get_set_and_restore_paths_are_repeatable",
            MEMORY_CONTROL_CHILD,
        );
        return;
    }

    let original_memory_limit = memory::memory_limit().unwrap();
    for _ in 0..200 {
        assert_eq!(
            memory::set_memory_limit(original_memory_limit).unwrap(),
            original_memory_limit
        );
        assert_eq!(memory::memory_limit().unwrap(), original_memory_limit);

        let cache_limit = memory::set_cache_limit(0).unwrap();
        assert_eq!(memory::set_cache_limit(cache_limit).unwrap(), 0);

        let wired_limit = memory::set_wired_limit(0).unwrap();
        assert_eq!(memory::set_wired_limit(wired_limit).unwrap(), 0);

        let _ = memory::active_memory().unwrap();
        let _ = memory::cache_memory().unwrap();
        let _ = memory::peak_memory().unwrap();
    }
}

#[test]
fn contiguous_and_rms_norm_release_error_outputs() {
    for _ in 0..200 {
        let empty = unsafe { Array::from_ptr(mlx_sys::mlx_array_new()) };
        assert!(empty.contiguous().is_err());
        assert!(empty
            .contiguous_with_options(ContiguousOptions {
                allow_col_major: true,
            })
            .is_err());

        let input = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 4]);
        let wrong_weight = Array::from_slice(&[1.0_f32, 2.0], &[2]);
        let error = fast::rms_norm(&input, Some(&wrong_weight), 1e-5)
            .expect_err("wrong RMSNorm weight length must fail during invocation");
        assert!(error.what().contains("[rms_norm]"));
    }
}

#[test]
fn rms_norm_absence_handles_are_freed_after_each_call() {
    for _ in 0..200 {
        let input = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 4]);
        fast::rms_norm(&input, None, 1e-5).unwrap().eval().unwrap();
    }
}

#[test]
fn static_and_advanced_index_updates_release_outputs_on_every_path() {
    let modes = [
        UpdateMode::Replace,
        UpdateMode::Add,
        UpdateMode::Min,
        UpdateMode::Max,
        UpdateMode::Product,
    ];
    for _ in 0..200 {
        for mode in modes {
            let source = Array::from_slice(&[1_i32, 2, 3, 4, 5], &[5]);
            source
                .try_index_update(1..4, Array::from_int(2), mode)
                .unwrap()
                .eval()
                .unwrap();

            let indices = Array::from_slice(&[0_i32, 2, 4], &[3]);
            source
                .try_index_update(&indices, Array::from_int(2), mode)
                .unwrap()
                .eval()
                .unwrap();

            source
                .try_index_update(3..3, Array::from_slice::<i32>(&[], &[0]), mode)
                .unwrap()
                .eval()
                .unwrap();

            let no_indices: &[ArrayIndexOp<'_>] = &[];
            Array::from_int(1)
                .try_index_update(no_indices, Array::from_int(2), mode)
                .unwrap()
                .eval()
                .unwrap();
        }
    }
}

#[test]
fn index_update_validation_and_broadcast_failures_are_repeatable() {
    for _ in 0..200 {
        let source = Array::from_slice(&[1_i32, 2, 3, 4, 5], &[5]);
        let zero_stride =
            source.try_index_update((..).stride_by(0), Array::from_int(2), UpdateMode::Replace);
        assert!(matches!(
            zero_stride,
            Err(IndexUpdateError::ZeroStride { axis: 0 })
        ));

        assert!(matches!(
            source.try_index_update(1..4, Array::from_slice(&[1_i32, 2], &[2]), UpdateMode::Add,),
            Err(IndexUpdateError::Exception(_))
        ));

        assert!(matches!(
            source.try_index_update((0, 1), Array::from_int(2), UpdateMode::Replace),
            Err(IndexUpdateError::Exception(_))
        ));
        assert!(matches!(
            Array::from_int(1).try_index_update(0, Array::from_int(2), UpdateMode::Replace),
            Err(IndexUpdateError::Exception(_))
        ));
        assert!(matches!(
            source.try_index_update(
                (Ellipsis, Ellipsis),
                Array::from_int(2),
                UpdateMode::Replace,
            ),
            Err(IndexUpdateError::Exception(_))
        ));
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
    assert_eq!((&Array::from_int(20) + 22).item_exact::<i32>(), 42);
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

#[test]
fn implicit_rng_state_is_released_at_thread_exit() {
    const WORKERS: usize = 8;
    const OPERATIONS: usize = 25;

    let handles = (0..WORKERS)
        .map(|_| {
            thread::spawn(|| {
                for _ in 0..OPERATIONS {
                    let result = mlx_rs::random::normal::<f32>(&[2, 3], None, None, None).unwrap();
                    result.eval().unwrap();
                    assert_eq!(result.shape(), &[2, 3]);
                    assert!(result
                        .as_slice::<f32>()
                        .iter()
                        .all(|value| value.is_finite()));
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
