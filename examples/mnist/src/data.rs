// TODO

use mlx_rs::{error::Exception, ops::stack, Array};
use mnist::{Mnist, MnistBuilder};

const IMAGE_SIZE: usize = 28 * 28;

pub fn read_data() -> (Vec<Array>, Vec<u8>, Array, Array) {
    let Mnist {
        trn_img,
        trn_lbl,
        val_img: _,
        val_lbl: _,
        tst_img,
        tst_lbl,
    } = MnistBuilder::new()
        .label_format_digit()
        .base_path("data")
        .training_images_filename("train-images.idx3-ubyte")
        .training_labels_filename("train-labels.idx1-ubyte")
        .test_images_filename("t10k-images.idx3-ubyte")
        .test_labels_filename("t10k-labels.idx1-ubyte")
        .finalize();

    // Check size
    assert_eq!(trn_img.len(), trn_lbl.len() * IMAGE_SIZE);
    assert_eq!(tst_img.len(), tst_lbl.len() * IMAGE_SIZE);

    // Convert to Array
    let train_images = trn_img
        .chunks_exact(IMAGE_SIZE)
        .map(|chunk| Array::from_slice(chunk, &[IMAGE_SIZE as i32]))
        .collect();

    let test_images = tst_img
        .chunks_exact(IMAGE_SIZE)
        .map(|chunk| Array::from_slice(chunk, &[IMAGE_SIZE as i32]))
        .collect::<Vec<_>>();
    let test_images = stack(&test_images, 0).unwrap();

    let test_labels = Array::from_slice(&tst_lbl, &[tst_lbl.len() as i32]);

    (train_images, trn_lbl, test_images, test_labels)
}

/// The iterator is collected to avoid repeated calls to `stack` in the training loop.
pub fn iterate_data<'a>(
    images: &'a [Array],
    labels: &'a [u8],
    batch_size: usize,
) -> Result<Vec<(Array, Array)>, Exception> {
    images
        .chunks_exact(batch_size)
        .zip(labels.chunks_exact(batch_size))
        .map(move |(images, labels)| {
            let images = stack(images, 0)?;
            let labels = Array::from_slice(labels, &[batch_size as i32]);
            Ok((images, labels))
        })
        .collect()
}
