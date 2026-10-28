use mlx_rs::{
    builder::Builder,
    error::Exception,
    losses::{CrossEntropyBuilder, LossReduction},
    module::{Module, ModuleParameters},
    nn,
    ops::{eq, indexing::argmax_axis, mean},
    optimizers::{Optimizer, Sgd},
    transforms::eval_params,
    Array,
};

/// MLP model
mod mlp;

/// Retrieves MNIST dataset
mod data;

fn eval_fn(model: &mut mlp::Mlp, (x, y): (&Array, &Array)) -> Result<Array, Exception> {
    let y_pred = model.forward(x)?;
    let accuracy = mean(&eq(&argmax_axis(&y_pred, 1, None)?, y)?, None)?;
    Ok(accuracy)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let num_layers = 2;
    let hidden_dim = 32;
    let num_classes = 10;
    let batch_size = 256;
    let num_epochs = 10;
    let learning_rate = 1e-2;

    let (train_images, train_labels, test_images, test_labels) = data::read_data();
    let loader = data::iterate_data(&train_images, &train_labels, batch_size)?;

    let input_dim = train_images[0].shape()[0];
    let mut model = mlp::Mlp::new(num_layers, input_dim, hidden_dim, num_classes)?;

    let cross_entropy = CrossEntropyBuilder::new()
        .reduction(LossReduction::Mean)
        .build()?;
    let loss_fn = |model: &mut mlp::Mlp, (x, y): (&Array, &Array)| -> Result<Array, Exception> {
        let y_pred = model.forward(x)?;
        cross_entropy.apply(y_pred, y)
    };
    let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);

    let mut optimizer = Sgd::new(learning_rate);

    for e in 0..num_epochs {
        let now = std::time::Instant::now();
        for (x, y) in &loader {
            let (_loss, grad) = loss_and_grad_fn(&mut model, (x, y))?;
            optimizer.update(&mut model, grad).unwrap();
            eval_params(model.parameters())?;
        }

        // Evaluate on test set
        let accuracy = eval_fn(&mut model, (&test_images, &test_labels))?;
        let elapsed = now.elapsed();
        println!(
            "Epoch: {}, Test accuracy: {:.2}, Time: {:.2} s",
            e,
            accuracy.item_exact::<f32>(),
            elapsed.as_secs_f32()
        );
    }

    Ok(())
}
