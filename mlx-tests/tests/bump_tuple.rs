use std::ffi::CStr;

use mlx_rs::{linalg::inv_device, Array, Stream};

#[test]
fn linked_mlx_runtime_is_0_32_2() {
    unsafe {
        let mut version = mlx_sys::mlx_string_new();
        assert_eq!(mlx_sys::mlx_version(&mut version), 0);
        let version_text = CStr::from_ptr(mlx_sys::mlx_string_data(version))
            .to_str()
            .unwrap();
        assert_eq!(version_text, "0.32.2");
        assert_eq!(mlx_sys::mlx_string_free(version), 0);
    }
}

#[test]
fn singular_inverse_is_catchable_at_eval() {
    // 0.30.6 aborted the process here (uncaught C++ exception through the
    // bindings); the replay baseline's "invoke" stage includes its recipe's
    // internal eval, so the Rust contract is: lazy construction, catchable
    // eval error.
    let singular = Array::zeros::<f32>(&[2, 2]).unwrap();
    let out = inv_device(&singular, Stream::cpu()).unwrap();
    let error = mlx_rs::transforms::eval([&out]).unwrap_err();
    assert!(!error.what().is_empty());
}
