#![allow(deprecated)]

use std::cell::Cell;

use mlx_internal_macros::generate_macro;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Device {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy)]
struct Stream(Device);

thread_local! {
    static CURRENT: Cell<Option<Device>> = const { Cell::new(None) };
}

impl Stream {
    fn thread_local_or_default() -> Self {
        Self(CURRENT.get().unwrap_or(Device::Gpu))
    }

    fn thread_local_or_cpu() -> Self {
        Self(CURRENT.get().unwrap_or(Device::Cpu))
    }
}

impl AsRef<Stream> for Stream {
    fn as_ref(&self) -> &Stream {
        self
    }
}

fn with_stream<T>(stream: &Stream, f: impl FnOnce() -> T) -> T {
    let previous = CURRENT.replace(Some(stream.0));
    let result = f();
    CURRENT.set(previous);
    result
}

fn cpu_only() -> Device {
    Stream::thread_local_or_cpu().0
}

#[generate_macro(customize(root = "$crate"))]
#[deprecated(since = "0.26.0", note = "use `cpu_only`")]
fn cpu_only_device(#[optional] stream: impl AsRef<Stream>) -> Device {
    with_stream(stream.as_ref(), cpu_only)
}

fn typed<T>(value: T) -> &'static str {
    let _ = value;
    std::any::type_name::<T>()
}

#[generate_macro(customize(
    root = "$crate",
    default_dtype = i32,
    forwarding_shim = true,
))]
#[deprecated(since = "0.26.0", note = "use `typed`")]
fn typed_device<T>(value: T, #[optional] stream: impl AsRef<Stream>) -> &'static str {
    with_stream(stream.as_ref(), || typed(value))
}

#[test]
fn no_stream_form_preserves_cpu_fallback() {
    assert_eq!(cpu_only!(), Device::Cpu);
}

#[test]
fn explicit_stream_form_preserves_override() {
    let stream = Stream(Device::Gpu);
    assert_eq!(cpu_only!(stream = &stream), Device::Gpu);
}

#[test]
fn generic_forms_compile_with_default_and_selected_dtype() {
    assert_eq!(typed!(1), "i32");
    assert_eq!(typed!(1, dtype = i16), "i16");

    let stream = Stream(Device::Cpu);
    assert_eq!(typed!(1, stream = &stream), "i32");
    assert_eq!(typed!(1, dtype = i16, stream = &stream), "i16");
}
