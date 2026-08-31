<div align="center">
<h1><b>mlx-rs</b></h1>

Rust bindings for Apple's mlx machine learning library.

[![Discord](https://img.shields.io/discord/1176807732473495552.svg?color=7289da&&logo=discord)](https://discord.gg/jZvTsxDX49)
[![Current Crates.io Version](https://img.shields.io/crates/v/mlx-rs.svg)](https://crates.io/crates/mlx-rs)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://oxiglade.github.io/mlx-rs/mlx_rs/)
[![Test Status](https://github.com/oxiglade/mlx-rs/actions/workflows/validate.yml/badge.svg)](https://github.com/oxiglade/mlx-rs/actions/workflows/validate.yml)
[![Blaze](https://runblaze.dev/gh/307493885959233117281096297203102330146/badge.svg)](https://runblaze.dev)
[![Rust Version](https://img.shields.io/badge/Rust-1.88.0+-blue)](https://releases.rs/docs/1.88.0)
![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)

> **⚠️ Project is in active development - contributors welcome!**

---

<div align="left" valign="middle">
<a href="https://runblaze.dev">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://www.runblaze.dev/logo_dark.png">
   <img align="right" src="https://www.runblaze.dev/logo_light.png" height="102px"/>
 </picture>
</a>

<br style="display: none;"/>

_[Blaze](https://runblaze.dev) supports this project by providing ultra-fast Apple Silicon macOS Github Action Runners. Apply the discount code `AI25` at checkout to enjoy 25% off your first year._

</div>

</div>

## Documentation

Due to known limitation of docsrs, we are hosting the documentation on github pages [here](https://oxiglade.github.io/mlx-rs/mlx_rs/).

## Features

MLX is an array framework for machine learning on Apple Silicon. mlx-rs provides Rust bindings for MLX, allowing you to use MLX in your Rust projects.

Some key features of MLX and `mlx-rs` include:
- **Performance**: MLX is optimized for Apple Silicon, providing fast performance for machine learning tasks.
- **Lazy Evaluation**: MLX uses lazy evaluation to optimize performance and memory usage. Arrays are only materialized when needed.
- **Dynamic Graphs**: Computation graphs in MLX are constructed dynamically, allowing for flexible and efficient computation. Changing the shapes of function arguments does not require recompilation.
- **Mutli-Device Support**: MLX supports running computations on any of the supported devices (for now the CPU and GPU).
- **Unified memory**: MLX provides a unified memory model, meaning arrays live in the same memory space, regardless of the device they are computed on. Operations can be performed on arrays on different devices without copying data between them.

`mlx-rs` is designed to be a safe and idiomatic Rust interface to MLX, providing a seamless experience for Rust developers.

## Examples
The [examples](examples/) directory contains sample projects demonstrating different uses cases of our library.
- [mnist](examples/mnist/): Train a basic neural network on the MNIST digit dataset
- [mistral](examples/mistral/): Text generation using the pre-trained Mistral model

## Installation

Add this to your `Cargo.toml`:
```toml
[dependencies]
mlx-rs = "0.21.0"
```

## Feature Flags

* `metal` - enables metal (GPU) usage in MLX
* `accelerate` - enables using the accelerate framework in MLX

## Important Notes on Automatic Differentiation

When using automatic differentiation in mlx-rs, there's an important difference in how closures work compared to Python's MLX. In Python, variables are implicitly captured and properly traced in the compute graph. However, in Rust, we need to be more explicit about which arrays should be traced.

❌ This approach may cause segfaults:
```rust
// Don't do this
let x = random::normal::<f32>(&[num_examples, num_features], None, None, None)?;
let y = x.matmul(&w_star)? + eps;

let loss_fn = |w: &Array| -> Result<Array, Exception> {
    let y_pred = x.matmul(w)?;  // x and y are captured from outer scope
    let loss = Array::from_f32(0.5) * ops::mean(&ops::square(&(y_pred - &y))?, None, None)?;
    Ok(loss)
};

let grad_fn = transforms::grad(loss_fn, &[0]);
```

✅ Instead, pass all required arrays as inputs to ensure proper tracing:
```rust
let loss_fn = |inputs: &[Array]| -> Result<Array, Exception> {
    let w = &inputs[0];
    let x = &inputs[1];
    let y = &inputs[2];

    let y_pred = x.matmul(w)?;
    let loss = Array::from_f32(0.5) * ops::mean(&ops::square(y_pred - y)?, None, None)?;
    Ok(loss)
};
let argnums = &[0];  // Specify which argument to differentiate with respect to

// Pass all required arrays in the inputs slice
let mut inputs = vec![w, x, y];
let grad = transforms::grad(loss_fn, argnums)(&inputs)?;
```

When using gradients in training loops, remember to update the appropriate array in your inputs:

```rust
let mut inputs = vec![w, x, y];

for _ in 0..num_iterations {
    let grad = transforms::grad(loss_fn, argnums)(&inputs)?;
    inputs[0] = &inputs[0] - Array::from_f32(learning_rate) * grad;  // Update the weight array
    inputs[0].eval()?;
}
```

We are actively working on improving this API to make it more ergonomic and closer to Python's behavior. For now, explicitly passing all required arrays as shown above is the recommended approach.

## Versioning

For simplicity, the main crate `mls-rs` follows MLX’s versioning, allowing you to easily see which MLX version you’re using under the hood. The `mlx-sys` crate follows the versioning of `mlx-c`, as that is the version from which the API is generated.

## Community

If you are excited about the project or want to contribute, don't hesitate to join our [Discord](https://discord.gg/jZvTsxDX49)!
We try to be as welcoming as possible to everybody from any background. We're still building this out, but you can ask your questions there!

## Status

mlx-rs is currently in active development and can be used to run MLX models in Rust.

## MSRV

The minimum supported Rust version is 1.88.0.

The MSRV is the minimum Rust version that can be used to compile each crate.

## License

mlx-rs is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
See [LICENSE-APACHE](./LICENSE-APACHE) and [LICENSE-MIT](./LICENSE-MIT) for details. Opening a pull
request is assumed to signal agreement with these licensing terms.
