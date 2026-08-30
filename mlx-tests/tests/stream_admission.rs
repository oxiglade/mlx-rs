use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{Arc, Barrier},
    thread,
};

use mlx_rs::{array, task_local_default_stream, with_new_default_stream, Array, Stream};

#[test]
fn scoped_stream_is_identity_preserving_and_passes_through_results() {
    assert!(task_local_default_stream().is_none());
    let selected = Stream::cpu();
    let result = with_new_default_stream(selected.clone(), || {
        assert_eq!(task_local_default_stream(), Some(selected.clone()));
        42
    });
    assert_eq!(result, 42);
    assert!(task_local_default_stream().is_none());
}

#[test]
fn nested_cpu_and_metal_scopes_restore_after_success_and_panic() {
    let cpu = Stream::cpu();
    let metal = Stream::gpu();

    with_new_default_stream(cpu.clone(), || {
        with_new_default_stream(metal.clone(), || {
            assert_eq!(task_local_default_stream(), Some(metal.clone()));
        });
        assert_eq!(task_local_default_stream(), Some(cpu.clone()));

        let panic = catch_unwind(AssertUnwindSafe(|| {
            with_new_default_stream(metal.clone(), || panic!("scoped stream panic"));
        }));
        assert!(panic.is_err());
        assert_eq!(task_local_default_stream(), Some(cpu.clone()));
    });

    assert!(task_local_default_stream().is_none());
}

#[test]
fn scoped_defaults_are_isolated_across_threads() {
    let barrier = Arc::new(Barrier::new(2));
    let handles = [false, true].map(|metal| {
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            let selected = if metal { Stream::gpu() } else { Stream::cpu() };
            with_new_default_stream(selected.clone(), || {
                barrier.wait();
                assert_eq!(task_local_default_stream(), Some(selected.clone()));
            });
            assert!(task_local_default_stream().is_none());
        })
    });

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn cpu_and_metal_defaults_evaluate_on_separate_threads() {
    let barrier = Arc::new(Barrier::new(2));
    let handles = [false, true].map(|metal| {
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            let selected = if metal { Stream::gpu() } else { Stream::cpu() };
            with_new_default_stream(selected, || {
                let input = array!([1.0_f32, 2.0, 3.0]);
                barrier.wait();
                let output = input.add(&array!(2.0_f32)).unwrap();
                output.eval().unwrap();
                output.as_slice::<f32>().to_vec()
            })
        })
    });

    for handle in handles {
        assert_eq!(handle.join().unwrap(), vec![3.0, 4.0, 5.0]);
    }
}

#[test]
fn moved_and_independently_cloned_arrays_use_thread_defaults() {
    let moved = array!([1.0_f32, 2.0, 3.0]);
    let moved = thread::spawn(move || {
        with_new_default_stream(Stream::gpu(), || {
            let output = moved.add(&array!(1.0_f32)).unwrap();
            output.eval().unwrap();
            output
        })
    })
    .join()
    .unwrap();
    assert_eq!(moved.as_slice::<f32>(), &[2.0, 3.0, 4.0]);

    let source = Array::from_slice(&[2.0_f32, 4.0, 6.0], &[3]);
    let cpu_array = source.clone();
    let metal_array = source.clone();
    let cpu = thread::spawn(move || {
        with_new_default_stream(Stream::cpu(), || {
            let output = cpu_array.multiply(&array!(0.5_f32)).unwrap();
            output.eval().unwrap();
            output.as_slice::<f32>().to_vec()
        })
    });
    let metal = thread::spawn(move || {
        with_new_default_stream(Stream::gpu(), || {
            let output = metal_array.multiply(&array!(0.5_f32)).unwrap();
            output.eval().unwrap();
            output.as_slice::<f32>().to_vec()
        })
    });

    assert_eq!(cpu.join().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(metal.join().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(source.as_slice::<f32>(), &[2.0, 4.0, 6.0]);
}

#[test]
fn stream_handles_survive_repeated_create_clone_and_free() {
    for _ in 0..1_000 {
        let cpu = Stream::cpu();
        let metal = Stream::gpu();
        assert_eq!(cpu, cpu.clone());
        assert_eq!(metal, metal.clone());
        assert_ne!(cpu, metal);
    }
}
