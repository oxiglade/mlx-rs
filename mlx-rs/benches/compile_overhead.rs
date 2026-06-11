//! Per-call overhead of `transforms::compile::compile` on a small
//! 3-input op cluster. Cases:
//!  - `ops_inline`: direct call, no compile.
//!  - `compile_per_call`: `compile(...)` constructed + invoked each iter.
//!  - `compile_warm`: closure built once, invoked many times — isolates
//!    construction cost from per-invocation cost.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mlx_rs::{
    error::Exception,
    nn,
    ops::exp,
    transforms::{compile::compile, eval},
    Array, Dtype,
};

fn compute_g_inner((a_log, a, dt_bias): (&Array, &Array, &Array)) -> Result<Array, Exception> {
    let a_log_f32 = a_log.as_dtype(Dtype::Float32)?;
    let inner = a.add(dt_bias)?;
    let s = nn::softplus(&inner)?;
    exp(a_log_f32)?.negative()?.multiply(&s)?.exp()
}

fn make_inputs(hv: i32) -> (Array, Array, Array) {
    let a_log = Array::from_slice(&vec![0.5_f32; hv as usize], &[hv]);
    let a = Array::from_slice(&vec![0.1_f32; hv as usize], &[1, 1, hv]);
    let dt_bias = Array::from_slice(&vec![0.0_f32; hv as usize], &[hv]);
    (a_log, a, dt_bias)
}

fn bench_compile_overhead(c: &mut Criterion) {
    let hv = 32;
    let (a_log, a, dt_bias) = make_inputs(hv);

    // Warm up the MLX kernel/compile cache before measuring.
    let warmup = compute_g_inner((&a_log, &a, &dt_bias)).unwrap();
    eval([&warmup]).unwrap();
    let mut warm_compiled = compile(compute_g_inner, Some(true));
    let warm = warm_compiled((&a_log, &a, &dt_bias)).unwrap();
    eval([&warm]).unwrap();

    let mut group = c.benchmark_group("compute_g");

    group.bench_function("ops_inline", |b| {
        b.iter(|| {
            let r =
                compute_g_inner((black_box(&a_log), black_box(&a), black_box(&dt_bias))).unwrap();
            eval([&r]).unwrap();
        })
    });

    group.bench_function("compile_per_call", |b| {
        b.iter(|| {
            let mut compiled = compile(compute_g_inner, Some(true));
            let r = compiled((black_box(&a_log), black_box(&a), black_box(&dt_bias))).unwrap();
            eval([&r]).unwrap();
        })
    });

    group.bench_function("compile_warm", |b| {
        let mut compiled = compile(compute_g_inner, Some(true));
        b.iter(|| {
            let r = compiled((black_box(&a_log), black_box(&a), black_box(&dt_bias))).unwrap();
            eval([&r]).unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, bench_compile_overhead);
criterion_main!(benches);
