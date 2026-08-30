#![allow(deprecated, unused_variables)]

use mlx_internal_macros::generate_macro;
use mlx_rs::{with_stream, Stream};

// Test generate_macro for functions with no generic type arguments.
fn foo(
    a: i32,                    // Mandatory argument
    b: i32,                    // Mandatory argument
    c: Option<i32>,            // Optional argument
    d: impl Into<Option<i32>>, // Optional argument but impl Trait
) -> i32 {
    a + b + c.unwrap_or(0) + d.into().unwrap_or(0)
}

#[generate_macro(customize(root = "$crate", forwarding_shim = true))]
#[deprecated(since = "0.26.0", note = "use `foo`")]
fn foo_device(
    a: i32,
    b: i32,
    #[optional] c: Option<i32>,
    #[optional] d: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> i32 {
    with_stream(stream.as_ref(), || foo(a, b, c, d))
}

#[test]
fn test_foo() {
    assert_eq!(foo!(1, 2), 3);
    assert_eq!(foo!(1, 2, c = Some(3)), 6);
    assert_eq!(foo!(1, 2, d = Some(4)), 7);
    assert_eq!(foo!(1, 2, c = Some(3), d = Some(4)), 10);

    let stream = Stream::new();

    assert_eq!(foo!(1, 2, stream = &stream), 3);
    assert_eq!(foo!(1, 2, c = Some(3), stream = &stream), 6);
    assert_eq!(foo!(1, 2, d = Some(4), stream = &stream), 7);
    assert_eq!(foo!(1, 2, c = Some(3), d = Some(4), stream = &stream), 10);
}

// Test generate_macro for functions with generic type arguments.
fn bar<T: Into<i32>>(
    a: T,                    // Mandatory argument
    b: T,                    // Mandatory argument
    c: Option<T>,            // Optional argument
    d: impl Into<Option<T>>, // Optional argument but impl Trait
) -> i32 {
    let a = a.into();
    let b = b.into();
    let c = c.map(Into::into);
    let d = d.into().map(Into::into);
    a + b + c.unwrap_or(0) + d.unwrap_or(0)
}

#[generate_macro(customize(
    root = "$crate",
    default_dtype = i32,
    forwarding_shim = true,
))]
#[deprecated(since = "0.26.0", note = "use `bar`")]
fn bar_device<T: Into<i32>>(
    a: T,
    b: T,
    #[optional] c: Option<T>,
    #[optional] d: impl Into<Option<T>>,
    #[optional] stream: impl AsRef<Stream>,
) -> i32 {
    with_stream(stream.as_ref(), || bar(a, b, c, d))
}

#[test]
fn test_bar() {
    // Without specifying dtype, the default is i32.

    let result = bar!(1, 2);
    assert_eq!(result, 3);

    let result = bar!(1, 2, c = Some(3));
    assert_eq!(result, 6);

    let result = bar!(1, 2, d = Some(4));
    assert_eq!(result, 7);

    let result = bar!(1, 2, c = Some(3), d = Some(4));
    assert_eq!(result, 10);

    // With dtype specified as i16.

    let result = bar!(1, 2, dtype = i16);
    assert_eq!(result, 3);

    let result = bar!(1, 2, c = Some(3), dtype = i16);
    assert_eq!(result, 6);

    let result = bar!(1, 2, d = Some(4), dtype = i16);
    assert_eq!(result, 7);

    let result = bar!(1, 2, c = Some(3), d = Some(4), dtype = i16);
    assert_eq!(result, 10);

    // With stream specified.

    let stream = Stream::new();

    let result = bar!(1, 2, stream = &stream);
    assert_eq!(result, 3);

    let result = bar!(1, 2, c = Some(3), stream = &stream);
    assert_eq!(result, 6);

    let result = bar!(1, 2, d = Some(4), stream = &stream);
    assert_eq!(result, 7);

    let result = bar!(1, 2, c = Some(3), d = Some(4), stream = &stream);
    assert_eq!(result, 10);

    // With dtype and stream specified.

    let result = bar!(1, 2, dtype = i16, stream = &stream);
    assert_eq!(result, 3);

    let result = bar!(1, 2, c = Some(3), dtype = i16, stream = &stream);
    assert_eq!(result, 6);

    let result = bar!(1, 2, d = Some(4), dtype = i16, stream = &stream);
    assert_eq!(result, 7);

    let result = bar!(
        1,
        2,
        c = Some(3),
        d = Some(4),
        dtype = i16,
        stream = &stream
    );
    assert_eq!(result, 10);
}

// Test named mandatory arguments.
fn baz(
    a: Option<i32>, // Optinal argument
    b: i32,         // Mandatory argument
    c: Option<i32>, // Optional argument
) -> i32 {
    a.unwrap_or(0) + b + c.unwrap_or(0)
}

#[generate_macro(customize(root = "$crate", forwarding_shim = true))]
#[deprecated(since = "0.26.0", note = "use `baz`")]
fn baz_device(
    #[optional] a: Option<i32>,
    #[named] b: i32,
    #[optional] c: Option<i32>,
    #[optional] stream: impl AsRef<Stream>,
) -> i32 {
    with_stream(stream.as_ref(), || baz(a, b, c))
}

#[test]
fn test_baz() {
    assert_eq!(baz!(b = 1), 1);
    assert_eq!(baz!(a = Some(2), b = 1), 3);
    assert_eq!(baz!(b = 1, c = Some(3)), 4);
    assert_eq!(baz!(a = Some(2), b = 1, c = Some(3)), 6);

    let stream = Stream::new();

    assert_eq!(baz!(b = 1, stream = &stream), 1);
    assert_eq!(baz!(a = Some(2), b = 1, stream = &stream), 3);
    assert_eq!(baz!(b = 1, c = Some(3), stream = &stream), 4);
    assert_eq!(baz!(a = Some(2), b = 1, c = Some(3), stream = &stream), 6);
}
