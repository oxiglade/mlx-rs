use super::{dimension, CacheError};
use mlx_rs::{error::Exception, Array, Stream};
use std::ffi::{c_char, c_void, CStr};

unsafe extern "C" {
    fn mlxrs_kv_append(
        out: *mut mlx_sys::mlx_array,
        cache: mlx_sys::mlx_array,
        update: mlx_sys::mlx_array,
        offset: i32,
        stream: mlx_sys::mlx_stream,
        error: extern "C" fn(*const c_char, *mut c_void),
        context: *mut c_void,
    ) -> i32;
    fn mlxrs_kv_synchronize(
        stream: mlx_sys::mlx_stream,
        error: extern "C" fn(*const c_char, *mut c_void),
        context: *mut c_void,
    ) -> i32;
}

extern "C" fn record_error(message: *const c_char, context: *mut c_void) {
    // The shim calls synchronously with a live message and the caller's error slot.
    unsafe {
        *context.cast::<Option<Exception>>() =
            Some(Exception::custom(CStr::from_ptr(message).to_string_lossy()));
    }
}

pub(super) fn append(
    cache: &Array,
    update: &Array,
    offset: usize,
    stream: &Stream,
) -> Result<Array, CacheError> {
    let offset = dimension(offset)?;
    let mut out = mlx_sys::mlx_array {
        ctx: std::ptr::null_mut(),
    };
    let mut error: Option<Exception> = None;
    let status = unsafe {
        mlxrs_kv_append(
            &mut out,
            cache.as_ptr(),
            update.as_ptr(),
            offset,
            stream.as_ptr(),
            record_error,
            (&mut error as *mut Option<Exception>).cast(),
        )
    };
    if status != 0 {
        return Err(error
            .unwrap_or_else(|| Exception::custom("KvAppend construction failed"))
            .into());
    }
    Ok(unsafe { Array::from_ptr(out) })
}

pub(super) fn synchronize(stream: &Stream) -> Result<(), CacheError> {
    let mut error: Option<Exception> = None;
    let status = unsafe {
        mlxrs_kv_synchronize(
            stream.as_ptr(),
            record_error,
            (&mut error as *mut Option<Exception>).cast(),
        )
    };
    if status != 0 {
        return Err(error
            .unwrap_or_else(|| Exception::custom("KV stream synchronization failed"))
            .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::{ops::indexing::TryIndexOp, Dtype};

    #[test]
    fn append_shares_live_storage_and_copies_strided_updates(
    ) -> Result<(), Box<dyn std::error::Error>> {
        mlx_rs::with_device(mlx_rs::Device::gpu(), || {
            for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
                let cache = mlx_rs::ops::zeros_dtype(&[2, 3, 11, 4], dtype)?;
                cache.eval()?;
                let data: Vec<_> = (1..=120).map(|n| n as f32).collect();
                let update = Array::from_slice(&data, &[2, 3, 4, 5])
                    .as_dtype(dtype)?
                    .swap_axes(2, 3)?;
                let stream = Stream::thread_local_or_default();
                let out = append(&cache, &update, 3, &stream)?;
                out.eval()?;
                assert_eq!(
                    out.contiguous()?.to_vec_cast::<f32>()?,
                    cache.contiguous()?.to_vec_cast::<f32>()?
                );
                assert_eq!(
                    out.try_index((.., .., 3..8, ..))?
                        .contiguous()?
                        .to_vec_cast::<f32>()?,
                    update.contiguous()?.to_vec_cast::<f32>()?
                );
                assert!(out
                    .try_index((.., .., ..3, ..))?
                    .contiguous()?
                    .to_vec_cast::<f32>()?
                    .iter()
                    .all(|&v| v == 0.));
                assert!(out
                    .try_index((.., .., 8.., ..))?
                    .contiguous()?
                    .to_vec_cast::<f32>()?
                    .iter()
                    .all(|&v| v == 0.));
                let error = append(&cache, &update, 7, &stream).unwrap_err();
                assert!(
                    matches!(error, CacheError::Exception(e) if e.what() == "Invalid KvAppend inputs")
                );
            }
            Ok(())
        })
    }
}
