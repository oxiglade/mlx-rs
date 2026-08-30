use std::{cell::RefCell, ffi::CStr, thread::LocalKey};

use crate::{
    device::Device,
    error::Result,
    utils::{guard::Guarded, SUCCESS},
};

thread_local! {
    static THREAD_LOCAL_DEFAULT_STREAM: RefCell<Option<Stream>> = const { RefCell::new(None) };
}

struct ScopedValueGuard<T: 'static> {
    local: &'static LocalKey<RefCell<Option<T>>>,
    previous: Option<T>,
}

impl<T: 'static> Drop for ScopedValueGuard<T> {
    fn drop(&mut self) {
        self.local.with_borrow_mut(|stream| {
            *stream = self.previous.take();
        });
    }
}

fn with_scoped_value<T: 'static, R>(
    local: &'static LocalKey<RefCell<Option<T>>>,
    value: T,
    f: impl FnOnce() -> R,
) -> R {
    let previous = local.with_borrow_mut(|current| current.replace(value));
    let _guard = ScopedValueGuard { local, previous };
    f()
}

/// Gets the thread-local scoped default stream.
///
/// The value does not propagate across asynchronous task suspension or between operating-system
/// threads.
pub fn thread_local_default_stream() -> Option<Stream> {
    THREAD_LOCAL_DEFAULT_STREAM.with_borrow(|s| s.clone())
}

/// Uses `stream` for operations constructed during `f`.
///
/// Scopes are synchronous, thread-local, nestable, and restore the previous stream if `f` panics.
/// To select a stream for one operation, put only that operation in the closure:
///
/// ```rust
/// use mlx_rs::{fft, with_stream, Array, Stream};
///
/// let input = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
/// let stream = Stream::cpu();
/// let output = with_stream(&stream, || fft::fft(&input, None, None)).unwrap();
/// assert_eq!(output.shape(), &[4]);
/// ```
pub fn with_stream<F, T>(stream: &Stream, f: F) -> T
where
    F: FnOnce() -> T,
{
    with_scoped_value(&THREAD_LOCAL_DEFAULT_STREAM, stream.clone(), f)
}

/// Uses the default stream on `device` for operations constructed during `f`.
///
/// This is equivalent to creating a stream for the device and passing it to [`with_stream`].
/// Scopes are synchronous, thread-local, nestable, and panic-safe.
pub fn with_device<F, T>(device: Device, f: F) -> T
where
    F: FnOnce() -> T,
{
    let stream = Stream::new_with_device(&device);
    with_stream(&stream, f)
}

/// Gets the thread-local scoped default stream.
#[deprecated(since = "0.26.0", note = "use `thread_local_default_stream`")]
pub fn task_local_default_stream() -> Option<Stream> {
    thread_local_default_stream()
}

/// Uses a given default stream for the duration of `f`.
#[deprecated(since = "0.26.0", note = "use `with_stream(&stream, f)`")]
pub fn with_new_default_stream<F, T>(default_stream: Stream, f: F) -> T
where
    F: FnOnce() -> T,
{
    with_stream(&default_stream, f)
}

/// Parameter type for all MLX operations.
///
/// Use this to control where operations are evaluated:
///
/// If omitted it will use the [Default::default()], which will be [Device::gpu()] unless
/// set otherwise.
#[derive(PartialEq)]
pub struct StreamOrDevice {
    pub(crate) stream: Stream,
}

impl StreamOrDevice {
    /// Create a new [`StreamOrDevice`] with a [`Stream`].
    pub fn new(stream: Stream) -> StreamOrDevice {
        StreamOrDevice { stream }
    }

    /// Create a new [`StreamOrDevice`] with a [`Device`].
    pub fn new_with_device(device: &Device) -> StreamOrDevice {
        StreamOrDevice {
            stream: Stream::new_with_device(device),
        }
    }

    /// Current default CPU stream.
    pub fn cpu() -> StreamOrDevice {
        StreamOrDevice {
            stream: Stream::cpu(),
        }
    }

    /// Current default GPU stream.
    pub fn gpu() -> StreamOrDevice {
        StreamOrDevice {
            stream: Stream::gpu(),
        }
    }
}

impl Default for StreamOrDevice {
    /// The default stream on the default device.
    ///
    /// This will be [Device::gpu()] unless [Device::set_default()]
    /// sets it otherwise.
    fn default() -> Self {
        Self {
            stream: Stream::new(),
        }
    }
}

impl AsRef<Stream> for StreamOrDevice {
    fn as_ref(&self) -> &Stream {
        &self.stream
    }
}

impl std::fmt::Debug for StreamOrDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.stream)
    }
}

impl std::fmt::Display for StreamOrDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.stream)
    }
}

/// A stream of evaluation attached to a particular device.
///
/// Typically, this is used via the `stream:` parameter on a method with a [StreamOrDevice]:
pub struct Stream {
    pub(crate) c_stream: mlx_sys::mlx_stream,
}

impl AsRef<Stream> for Stream {
    fn as_ref(&self) -> &Stream {
        self
    }
}

impl Clone for Stream {
    fn clone(&self) -> Self {
        Stream::try_from_op(|res| unsafe { mlx_sys::mlx_stream_set(res, self.c_stream) })
            .expect("Failed to clone stream")
    }
}

impl Stream {
    /// Create a new stream on the default device, or return the thread-local
    /// default stream if present.
    pub fn thread_local_or_default() -> Self {
        thread_local_default_stream().unwrap_or_default()
    }

    /// Create a new stream on the default cpu device, or return the thread-local
    /// default stream if present.
    pub fn thread_local_or_cpu() -> Self {
        thread_local_default_stream().unwrap_or_else(Stream::cpu)
    }

    /// Create a new stream on the default gpu device, or return the thread-local
    /// default stream if present.
    pub fn thread_local_or_gpu() -> Self {
        thread_local_default_stream().unwrap_or_else(Stream::gpu)
    }

    /// Returns the thread-local scoped stream or the default stream.
    #[deprecated(since = "0.26.0", note = "use `Stream::thread_local_or_default`")]
    pub fn task_local_or_default() -> Self {
        Self::thread_local_or_default()
    }

    /// Returns the thread-local scoped stream or the CPU stream.
    #[deprecated(since = "0.26.0", note = "use `Stream::thread_local_or_cpu`")]
    pub fn task_local_or_cpu() -> Self {
        Self::thread_local_or_cpu()
    }

    /// Returns the thread-local scoped stream or the GPU stream.
    #[deprecated(since = "0.26.0", note = "use `Stream::thread_local_or_gpu`")]
    pub fn task_local_or_gpu() -> Self {
        Self::thread_local_or_gpu()
    }

    /// Create a new stream on the default device. Panics if fails.
    pub fn new() -> Stream {
        unsafe {
            let mut dev = mlx_sys::mlx_device_new();
            // SAFETY: mlx_get_default_device internally never throws an error
            mlx_sys::mlx_get_default_device(&mut dev as *mut _);

            let mut c_stream = mlx_sys::mlx_stream_new();
            // SAFETY: mlx_get_default_stream internally never throws if dev is valid
            mlx_sys::mlx_get_default_stream(&mut c_stream as *mut _, dev);

            mlx_sys::mlx_device_free(dev);
            Stream { c_stream }
        }
    }

    /// Try to get the default stream on the given device.
    pub fn try_default_on_device(device: &Device) -> Result<Stream> {
        Stream::try_from_op(|res| unsafe { mlx_sys::mlx_get_default_stream(res, device.c_device) })
    }

    /// Create a new stream on the given device
    pub fn new_with_device(device: &Device) -> Stream {
        unsafe {
            let c_stream = mlx_sys::mlx_stream_new_device(device.c_device);
            Stream { c_stream }
        }
    }

    /// Get the underlying C pointer.
    pub fn as_ptr(&self) -> mlx_sys::mlx_stream {
        self.c_stream
    }

    /// Current default CPU stream.
    pub fn cpu() -> Self {
        unsafe {
            let c_stream = mlx_sys::mlx_default_cpu_stream_new();
            Stream { c_stream }
        }
    }

    /// Current default GPU stream.
    pub fn gpu() -> Self {
        unsafe {
            let c_stream = mlx_sys::mlx_default_gpu_stream_new();
            Stream { c_stream }
        }
    }

    /// Get the index of the stream.
    pub fn get_index(&self) -> Result<i32> {
        i32::try_from_op(|res| unsafe { mlx_sys::mlx_stream_get_index(res, self.c_stream) })
    }

    fn describe(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        unsafe {
            let mut mlx_str = mlx_sys::mlx_string_new();
            let result = match mlx_sys::mlx_stream_tostring(&mut mlx_str as *mut _, self.c_stream) {
                SUCCESS => {
                    let ptr = mlx_sys::mlx_string_data(mlx_str);
                    let c_str = CStr::from_ptr(ptr);
                    write!(f, "{}", c_str.to_string_lossy())
                }
                _ => Err(std::fmt::Error),
            };
            mlx_sys::mlx_string_free(mlx_str);
            result
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_stream_free(self.c_stream) };
    }
}

impl Default for Stream {
    fn default() -> Self {
        Stream::new()
    }
}

impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.describe(f)
    }
}

impl std::fmt::Display for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.describe(f)
    }
}

impl PartialEq for Stream {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlx_sys::mlx_stream_equal(self.c_stream, other.c_stream) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_scopes_nest_and_restore_after_panic() {
        let outer = Stream::cpu();
        with_stream(&outer, || {
            assert_eq!(thread_local_default_stream(), Some(outer.clone()));

            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                with_device(Device::cpu(), || panic!("scope panic"));
            }));

            assert!(panic.is_err());
            assert_eq!(thread_local_default_stream(), Some(outer.clone()));
        });

        assert!(thread_local_default_stream().is_none());
    }

    #[test]
    fn test_scoped_default_stream() {
        // First set default stream to CPU
        let cpu_device = Device::cpu();
        Device::set_default(&cpu_device);
        let cpu_stream = Stream::default();

        let task_default_stream = Stream::gpu();
        with_stream(&task_default_stream, || {
            let task_local_stream_0 = Stream::thread_local_or_default();
            let task_local_stream_1 = Stream::thread_local_or_default();
            assert_eq!(task_local_stream_0, task_local_stream_1);
            assert_ne!(task_local_stream_0, cpu_stream);
        });
    }

    #[test]
    fn test_scoped_default_stream_restored_after_panic() {
        let cpu = Device::cpu();
        let outer_stream = Stream::new_with_device(&cpu);
        with_stream(&outer_stream, || {
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let inner = Stream::new_with_device(&cpu);
                with_stream(&inner, || panic!("stream panic"));
            }));

            assert!(panic.is_err());
            assert_eq!(thread_local_default_stream(), Some(outer_stream.clone()));
        });

        assert!(thread_local_default_stream().is_none());
    }

    #[test]
    fn test_stream_clone() {
        let stream = Stream::new();
        let cloned_stream = stream.clone();
        assert_eq!(stream, cloned_stream);
    }

    #[test]
    fn test_cpu_gpu_stream_not_equal() {
        let cpu_device = Device::cpu();
        let gpu_device = Device::gpu();

        // First set default stream to CPU
        Device::set_default(&cpu_device);
        let cpu_stream = Stream::default();

        // Then set default stream to GPU
        Device::set_default(&gpu_device);
        let gpu_stream = Stream::default();

        // Assert that CPU and GPU streams are not equal
        assert_ne!(cpu_stream, gpu_stream);
    }
}
