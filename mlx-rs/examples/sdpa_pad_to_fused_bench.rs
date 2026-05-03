//! Benchmark `scaled_dot_product_attention_pad_to_fused` vs the
//! plain `scaled_dot_product_attention` at non-fused head_dims.
//! Demonstrates the speedup the helper unlocks for models like
//! Qwen3-VL (head_dim=72) at long-context prefill.
//!
//! Run with:
//!   cargo run --release --example sdpa_pad_to_fused_bench

use std::time::Instant;

use mlx_rs::random::normal;
use mlx_rs::transforms as mlx_transforms;
use mlx_rs::Array;
use mlx_rs::fast::{
    scaled_dot_product_attention,
    scaled_dot_product_attention_pad_to_fused,
};

fn sync(a: &Array) {
    let _ = mlx_transforms::eval([a]);
}

fn bench(
    label: &str,
    runs: usize,
    f: impl Fn() -> mlx_rs::error::Result<Array>,
) {
    // 10 warmup iterations with sync after each — flushes any
    // benefit from the previous bench's GPU state.
    for _ in 0..10 {
        let r = f().expect("warmup");
        sync(&r);
    }
    // Measured loop: sync after EACH op so we measure end-to-end
    // dispatch + kernel time per call, not graph-batched throughput.
    let t0 = Instant::now();
    for _ in 0..runs {
        let a = f().expect("run");
        sync(&a);
    }
    let elapsed = t0.elapsed();
    println!(
        "  {:<48} {} runs in {:?} ({:?} / run)",
        label, runs, elapsed, elapsed / runs as u32,
    );
}

fn run_case(name: &str, b: i32, h: i32, l: i32, d: i32) {
    let scale = (d as f32).powf(-0.5);
    println!("\n=== {name}: [B={b}, H={h}, L={l}, head_dim={d}] ===");

    let q = normal::<f32>(&[b, h, l, d], None, None, None).expect("q");
    let k = normal::<f32>(&[b, h, l, d], None, None, None).expect("k");
    let v = normal::<f32>(&[b, h, l, d], None, None, None).expect("v");
    sync(&q); sync(&k); sync(&v);

    let runs = 50;
    bench("plain SDPA", runs, || {
        scaled_dot_product_attention(&q, &k, &v, scale, None, None)
    });
    bench("padded SDPA", runs, || {
        scaled_dot_product_attention_pad_to_fused(&q, &k, &v, scale, None, None)
    });
}

fn main() {
    // The hot case: Qwen3-VL vision tower head_dim=72 at long-context
    // prefill — falls outside MLX's fused {64, 80, 128} set, hits the
    // general-purpose fallback. Helper pads to 80 → fused kernel.
    run_case("Qwen3-VL vision @ L=1024",  1, 16, 1024, 72);
    run_case("Qwen3-VL vision @ L=4096",  1, 16, 4096, 72);
    run_case("90-dim transformer @ L=2048", 1, 32, 2048, 90);

    // Passthrough cases: head_dim already in fused set, helper is a
    // no-op. Confirms the helper does NOT regress the fast path.
    run_case("Step-Audio-2 encoder (head_dim=64)", 1, 20, 1500, 64);
    run_case("Llama-style head_dim=128", 1, 32, 2048, 128);
}
