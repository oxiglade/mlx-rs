//! Sanity checks for canonical exported operations.

use mlx_rs::{
    array, complex64,
    error::Exception,
    fast, linalg,
    ops::{self, arange, reshape},
    random,
    test_utils::{assert_array_eq, tolerances},
    with_device, Array, Device, Dtype,
};

// Try two functions that don't have any optional arguments.

#[test]
fn test_ops_arithmetic_abs() {
    let data = array!([1i32, 2, -3, -4, -5]);
    let result = ops::abs(&data).unwrap();

    assert_eq!(result.as_slice::<i32>(), &[1, 2, 3, 4, 5]);

    let result = with_device(Device::cpu(), || ops::abs(data)).unwrap();

    assert_eq!(result.as_slice::<i32>(), &[1, 2, 3, 4, 5]);
}

#[test]
fn test_ops_arithmetic_add() {
    let data1 = array!([1i32, 2, 3, 4, 5]);
    let data2 = array!([1i32, 2, 3, 4, 5]);
    let result = ops::add(&data1, &data2).unwrap();

    assert_eq!(result.as_slice::<i32>(), &[2, 4, 6, 8, 10]);

    let result = with_device(Device::cpu(), || ops::add(data1, data2)).unwrap();

    assert_eq!(result.as_slice::<i32>(), &[2, 4, 6, 8, 10]);
}

// Try a function that has optional arguments.

#[test]
fn test_ops_arithmetic_tensordot() {
    let x = reshape(arange::<_, f32>(None, 60.0, None).unwrap(), &[3, 4, 5]).unwrap();
    let y = reshape(arange::<_, f32>(None, 24.0, None).unwrap(), &[4, 3, 2]).unwrap();
    let axes_x = [1, 0];
    let axes_y = [0, 1];
    let z = ops::tensordot_axes(&x, &y, &axes_x, &axes_y).unwrap();
    let expected = Array::from_slice(
        &[
            4400.0f32, 4730.0, 4532.0, 4874.0, 4664.0, 5018.0, 4796.0, 5162.0, 4928.0, 5306.0,
        ],
        &[5, 2],
    );
    assert_array_eq(z, &expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);

    let z = with_device(Device::cpu(), || {
        ops::tensordot_axes(&x, &y, &axes_x, &axes_y)
    })
    .unwrap();
    assert_array_eq(z, expected, tolerances::EXACT.rtol, tolerances::EXACT.atol);
}

// Test functions defined in `mlx_rs::ops` module.

#[test]
fn test_ops_convolution_conv1d() {
    let input = array!(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        shape = [1, 5, 2]
    );
    let weight = array!(
        [0.5, 0.0, -0.5, 1.0, 0.0, 1.5, 2.0, 0.0, -2.0, 1.5, 0.0, 1.0],
        shape = [2, 3, 2]
    );

    let result = ops::conv1d(&input, &weight, 1, 0, 1, 1).unwrap();

    let expected = array!([12.0, 8.0, 17.0, 13.0, 22.0, 18.0], shape = [1, 3, 2]);
    assert_array_eq(
        result,
        expected,
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
}

#[test]
fn test_ops_factory_arange() {
    // Without specifying start and step
    let array = Array::arange::<_, f32>(None, 50, None).unwrap();
    assert_eq!(array.shape(), &[50]);
    assert_eq!(array.dtype(), Dtype::Float32);

    let data: &[f32] = array.as_slice();
    let expected: Vec<f32> = (0..50).map(|x| x as f32).collect();
    assert_eq!(data, expected.as_slice());

    // With specifying start and step
    let array = Array::arange::<_, f32>(1.0, 50.0, 2.0).unwrap();
    assert_eq!(array.shape(), &[25]);
    assert_eq!(array.dtype(), Dtype::Float32);

    let data: &[f32] = array.as_slice();
    let expected: Vec<f32> = (1..50).step_by(2).map(|x| x as f32).collect();
    assert_eq!(data, expected.as_slice());

    let array = with_device(Device::cpu(), || Array::arange::<_, f32>(1.0, 50.0, 2.0)).unwrap();
    assert_eq!(array.shape(), &[25]);
    assert_eq!(array.dtype(), Dtype::Float32);

    let data: &[f32] = array.as_slice();
    let expected: Vec<f32> = (1..50).step_by(2).map(|x| x as f32).collect();
    assert_eq!(data, expected.as_slice());
}

// Test functions defined in `mlx_rs::fft` module.

#[test]
fn test_fft_fft() {
    const FFT_EXPECTED: &[complex64; 4] = &[
        complex64::new(10.0, 0.0),
        complex64::new(-2.0, 2.0),
        complex64::new(-2.0, 0.0),
        complex64::new(-2.0, -2.0),
    ];

    let data = array!([1.0, 2.0, 3.0, 4.0]);
    let fft = mlx_rs::fft::fft(&data, None, None).unwrap();

    assert_eq!(fft.dtype(), Dtype::Complex64);
    assert_array_eq(
        fft,
        Array::from_slice(FFT_EXPECTED, &[4]),
        tolerances::EXACT.rtol,
        tolerances::EXACT.atol,
    );
}

// Test functions defined in `mlx_rs::linalg` module.

#[test]
fn test_linalg_norm() {
    let a = array!([1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    let norm = linalg::norm_l2(&a, linalg::NormOptions::default()).unwrap();
    assert_eq!(norm.item::<f32>(), 5.477_226);
}

// Test functions defined in `mlx_rs::random` module.

#[test]
fn test_random_uniform() {
    let value = random::uniform::<_, f32>(0.0, 1.0, &[1], None).unwrap();
    assert_eq!(value.shape(), &[1]);
    assert!(value.item::<f32>() >= 0.0 && value.item::<f32>() <= 1.0);
}

#[test]
fn test_random_normal() {
    let value = random::normal::<f32>(&[1], None, None, None).unwrap();
    assert_eq!(value.shape(), &[1]);
    assert!(value.item::<f32>() >= -10.0 && value.item::<f32>() <= 10.0);
}

// Test functions defined in `mlx_rs::fast` module.

#[test]
#[allow(non_snake_case)]
fn test_fast_sdpa() -> Result<(), Exception> {
    // This test just makes sure that `scaled_dot_product_attention` is callable
    // in the various cases, based on the Python test `test_fast_sdpa`.

    let Dk = 64;
    let scale = 1.0 / (Dk as f32).sqrt();
    for seq_len in [63, 129, 400] {
        for dtype in [crate::Dtype::Float32, crate::Dtype::Float16] {
            let B = 2;
            let H = 24;
            let q =
                random::normal::<f32>(&[B, H, seq_len, Dk], None, None, None)?.as_dtype(dtype)?;
            let k =
                random::normal::<f32>(&[B, H, seq_len, Dk], None, None, None)?.as_dtype(dtype)?;
            let v =
                random::normal::<f32>(&[B, H, seq_len, Dk], None, None, None)?.as_dtype(dtype)?;

            let result = fast::scaled_dot_product_attention(q, k, v, scale, None, None)?;
            assert_eq!(result.shape(), [B, H, seq_len, Dk]);
            assert_eq!(result.dtype(), dtype);
        }
    }

    Ok(())
}
