use mlx_rs::error::Exception;
use mlx_rs::{ops, random, transforms, Array};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let num_features: i32 = 100;
    let num_examples: i32 = 1000;
    let num_iterations: i32 = 10000;
    let learning_rate: f32 = 0.01;

    // True weight vector
    let w_star = random::normal::<f32>(&[num_features], None, None, None)?;

    // Input examples (design matrix)
    let x = random::normal::<f32>(&[num_examples, num_features], None, None, None)?;

    // Noisy labels
    let eps = random::normal::<f32>(&[num_examples], None, None, None)? * 1e-2;
    let y = x.matmul(&w_star)? + eps;

    // Initialize random weights
    let w = random::normal::<f32>(&[num_features], None, None, None)? * 1e-2;

    let loss_fn = |inputs: &[Array]| -> Result<Array, Exception> {
        let w = &inputs[0];
        let x = &inputs[1];
        let y = &inputs[2];

        let y_pred = x.matmul(w)?;
        let loss = Array::from_f32(0.5) * ops::mean(&ops::square(y_pred - y)?, None)?;
        Ok(loss)
    };

    let mut grad_fn = transforms::grad(loss_fn);

    let now = std::time::Instant::now();
    let mut inputs = [w, x, y];

    for _ in 0..num_iterations {
        let grad = grad_fn(&inputs)?;
        inputs[0] = &inputs[0] - Array::from_f32(learning_rate) * grad;
        inputs[0].eval()?;
    }

    let elapsed = now.elapsed();

    let loss = loss_fn(&inputs)?;
    let error_norm = ops::sum(&ops::square(&(&inputs[0] - &w_star))?, None)?.sqrt()?;
    let throughput = num_iterations as f32 / elapsed.as_secs_f32();

    println!(
        "Loss {:.5}, L2 distance: |w-w*| = {:.5}, Throughput {:.5} (it/s)",
        loss.item::<f32>(),
        error_norm.item::<f32>(),
        throughput
    );

    Ok(())
}
