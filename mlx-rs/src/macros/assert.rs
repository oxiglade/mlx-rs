/// Asserts that two arrays have the same shape and sufficiently close elements.
///
/// This macro ignores dtype. Its optional third argument is passed as both `rtol`
/// and `atol` to [`Array::all_close`](crate::Array::all_close); omitting it uses
/// that method's default tolerances. It panics if the shapes differ, the values
/// are not sufficiently close, or the comparison returns an error.
///
/// Use [`Array::all_close`](crate::Array::all_close) with separate `rtol` and
/// `atol` arguments, plus explicit shape and dtype checks:
///
/// ```
/// use mlx_rs::array;
///
/// let actual = array!([1.0_f32, 2.0]);
/// let expected = array!([1.0_f32, 2.0]);
/// assert_eq!(actual.shape(), expected.shape());
/// assert_eq!(actual.dtype(), expected.dtype());
/// assert!(actual.all_close(&expected, Some(1e-5), Some(1e-8), None).unwrap());
/// ```
#[macro_export]
#[deprecated(
    since = "0.33.0",
    note = "use `Array::all_close` with separate `rtol` and `atol` arguments and an explicit dtype check"
)]
macro_rules! assert_array_eq {
    ($value:expr, $expected:expr) => {
        #[allow(deprecated)]
        {
            assert_array_eq!($value, $expected, None);
        }
    };
    ($value:expr, $expected:expr, $atol:expr) => {
        assert_eq!($value.shape(), $expected.shape(), "Shapes are not equal");
        let assert = $value.all_close(&$expected, $atol, $atol, None);
        assert!(assert.unwrap(), "Values are not sufficiently close");
    };
}
