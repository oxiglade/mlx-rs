use mlx_rs::{
    ops::{
        concatenate,
        indexing::{TryIndexMutOp, TryIndexOp},
    },
    Array,
};

fn prefix() -> Array {
    Array::arange::<_, f32>(0, 4_080, None)
        .unwrap()
        .reshape(&[1, 2, 510, 4])
        .unwrap()
}

fn token() -> Array {
    Array::arange::<_, f32>(9_000, 9_008, None)
        .unwrap()
        .reshape(&[1, 2, 1, 4])
        .unwrap()
}

fn assert_appended_token(buffer: &Array) {
    // Integer indexing into the padded buffer yields a non-contiguous view, so
    // compare arrays instead of extracting a slice.
    let slot = buffer.try_index((.., .., 510, ..)).unwrap();
    let expected = token().try_index((.., .., 0, ..)).unwrap();
    assert!(
        slot.eq_exact(&expected).unwrap(),
        "appended token corrupted"
    );
}

#[test]
fn concatenated_token_survives_capacity_padding() {
    let padding = Array::zeros::<f32>(&[1, 2, 1, 4]).unwrap();
    let padded = concatenate(&[prefix(), token(), padding], -2).unwrap();
    let live = padded.try_index((.., .., ..511, ..)).unwrap();

    assert_eq!(padded.shape(), &[1, 2, 512, 4]);
    assert_appended_token(&live);
}

#[test]
fn indexed_token_survives_capacity_padding() {
    let mut buffer = Array::zeros::<f32>(&[1, 2, 512, 4]).unwrap();
    buffer.try_index_mut((.., .., ..510, ..), prefix()).unwrap();
    buffer
        .try_index_mut((.., .., 510..511, ..), token())
        .unwrap();
    let live = buffer.try_index((.., .., ..511, ..)).unwrap();

    assert_appended_token(&live);
}
