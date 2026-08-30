/// Asserts that two arrays are equal.
///
/// It checks that the two arrays have the same shape and that all elements are
/// sufficiently close.
///
/// This legacy macro does not compare dtypes and uses its tolerance as both `rtol` and `atol`.
/// Workspace tests should use [`crate::test_utils::assert_array_eq`] for strict comparisons.
#[macro_export]
macro_rules! assert_array_eq {
    ($value:expr, $expected:expr) => {
        assert_array_eq!($value, $expected, None);
    };
    ($value:expr, $expected:expr, $atol:expr) => {
        assert_eq!($value.shape(), $expected.shape(), "Shapes are not equal");
        let assert = $value.all_close(&$expected, $atol, $atol, None);
        assert!(assert.unwrap(), "Values are not sufficiently close");
    };
}
