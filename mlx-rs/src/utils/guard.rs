use half::{bf16, f16};
use mlx_sys::{__BindgenComplex, bfloat16_t, float16_t, mlx_array};

use crate::{complex64, error::Exception, Array};

use super::{VectorArray, SUCCESS};

type Status = i32;

pub trait Guard<T>: Default {
    type MutRawPtr;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr;

    fn set_init_success(&mut self, success: bool);

    fn try_into_guarded(self) -> Result<T, Exception>;
}

pub(crate) trait Guarded: Sized {
    type Guard: Guard<Self>;

    #[track_caller]
    fn try_from_op<F>(f: F) -> Result<Self, Exception>
    where
        F: FnOnce(<Self::Guard as Guard<Self>>::MutRawPtr) -> Status,
    {
        crate::error::INIT_ERR_HANDLER
            .with(|init| init.call_once(crate::error::setup_mlx_error_handler));

        let mut guard = Self::Guard::default();
        let status = f(guard.as_mut_raw_ptr());
        crate::error::resume_closure_panic();
        match status {
            SUCCESS => {
                guard.set_init_success(true);
                guard.try_into_guarded()
            }
            _ => {
                // Err(crate::error::get_and_clear_last_mlx_error()
                // .expect("MLX operation failed but no error was set"))
                let what = crate::error::get_and_clear_last_mlx_error()
                    .expect("MLX operation failed but no error was set")
                    .what;
                let location = std::panic::Location::caller();
                Err(Exception { what, location })
            }
        }
    }
}

pub(crate) struct MaybeUninitArray {
    pub(crate) ptr: mlx_array,
    pub(crate) init_success: bool,
}

impl Default for MaybeUninitArray {
    fn default() -> Self {
        Self::new()
    }
}

impl MaybeUninitArray {
    pub fn new() -> Self {
        unsafe {
            Self {
                ptr: mlx_sys::mlx_array_new(),
                init_success: false,
            }
        }
    }
}

impl Drop for MaybeUninitArray {
    fn drop(&mut self) {
        if !self.init_success {
            unsafe {
                mlx_sys::mlx_array_free(self.ptr);
            }
        }
    }
}

impl Guard<Array> for MaybeUninitArray {
    type MutRawPtr = *mut mlx_array;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        &mut self.ptr
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }

    fn try_into_guarded(self) -> Result<Array, Exception> {
        debug_assert!(self.init_success);
        unsafe { Ok(Array::from_ptr(self.ptr)) }
    }
}

impl Guarded for Array {
    type Guard = MaybeUninitArray;
}

pub(crate) struct MaybeUninitVectorArray {
    pub(crate) ptr: mlx_sys::mlx_vector_array,
    pub(crate) init_success: bool,
}

impl Default for MaybeUninitVectorArray {
    fn default() -> Self {
        Self::new()
    }
}

impl MaybeUninitVectorArray {
    pub fn new() -> Self {
        unsafe {
            Self {
                ptr: mlx_sys::mlx_vector_array_new(),
                init_success: false,
            }
        }
    }
}

impl Drop for MaybeUninitVectorArray {
    fn drop(&mut self) {
        if !self.init_success {
            unsafe {
                mlx_sys::mlx_vector_array_free(self.ptr);
            }
        }
    }
}

impl Guard<Vec<Array>> for MaybeUninitVectorArray {
    type MutRawPtr = *mut mlx_sys::mlx_vector_array;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        &mut self.ptr
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }

    fn try_into_guarded(mut self) -> Result<Vec<Array>, Exception> {
        debug_assert!(self.init_success);
        self.init_success = false; // mlx_vector_array still needs to be freed after we extracted its elements
        unsafe {
            let size = mlx_sys::mlx_vector_array_size(self.ptr);
            (0..size)
                .map(|i| Array::try_from_op(|res| mlx_sys::mlx_vector_array_get(res, self.ptr, i)))
                .collect()
        }
    }
}

impl Guarded for Vec<Array> {
    type Guard = MaybeUninitVectorArray;
}

impl Guard<VectorArray> for MaybeUninitVectorArray {
    type MutRawPtr = *mut mlx_sys::mlx_vector_array;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        &mut self.ptr
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }

    fn try_into_guarded(self) -> Result<VectorArray, Exception> {
        Ok(VectorArray { c_vec: self.ptr })
    }
}

impl Guarded for VectorArray {
    type Guard = MaybeUninitVectorArray;
}

impl Guard<(Array, Array)> for (MaybeUninitArray, MaybeUninitArray) {
    type MutRawPtr = (*mut mlx_array, *mut mlx_array);

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        (self.0.as_mut_raw_ptr(), self.1.as_mut_raw_ptr())
    }

    fn set_init_success(&mut self, success: bool) {
        self.0.set_init_success(success);
        self.1.set_init_success(success);
    }

    fn try_into_guarded(self) -> Result<(Array, Array), Exception> {
        Ok((self.0.try_into_guarded()?, self.1.try_into_guarded()?))
    }
}

impl Guarded for (Array, Array) {
    type Guard = (MaybeUninitArray, MaybeUninitArray);
}

impl Guard<(Array, Array, Array)> for (MaybeUninitArray, MaybeUninitArray, MaybeUninitArray) {
    type MutRawPtr = (*mut mlx_array, *mut mlx_array, *mut mlx_array);

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        (
            self.0.as_mut_raw_ptr(),
            self.1.as_mut_raw_ptr(),
            self.2.as_mut_raw_ptr(),
        )
    }

    fn set_init_success(&mut self, success: bool) {
        self.0.set_init_success(success);
        self.1.set_init_success(success);
        self.2.set_init_success(success);
    }

    fn try_into_guarded(self) -> Result<(Array, Array, Array), Exception> {
        Ok((
            self.0.try_into_guarded()?,
            self.1.try_into_guarded()?,
            self.2.try_into_guarded()?,
        ))
    }
}

impl Guarded for (Array, Array, Array) {
    type Guard = (MaybeUninitArray, MaybeUninitArray, MaybeUninitArray);
}

impl Guard<(Vec<Array>, Vec<Array>)> for (MaybeUninitVectorArray, MaybeUninitVectorArray) {
    type MutRawPtr = (
        *mut mlx_sys::mlx_vector_array,
        *mut mlx_sys::mlx_vector_array,
    );

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        (
            <MaybeUninitVectorArray as Guard<Vec<Array>>>::as_mut_raw_ptr(&mut self.0),
            <MaybeUninitVectorArray as Guard<Vec<Array>>>::as_mut_raw_ptr(&mut self.1),
        )
    }

    fn set_init_success(&mut self, success: bool) {
        <MaybeUninitVectorArray as Guard<Vec<Array>>>::set_init_success(&mut self.0, success);
        <MaybeUninitVectorArray as Guard<Vec<Array>>>::set_init_success(&mut self.1, success);
    }

    fn try_into_guarded(self) -> Result<(Vec<Array>, Vec<Array>), Exception> {
        Ok((self.0.try_into_guarded()?, self.1.try_into_guarded()?))
    }
}

impl Guarded for (Vec<Array>, Vec<Array>) {
    type Guard = (MaybeUninitVectorArray, MaybeUninitVectorArray);
}

pub(crate) struct MaybeUninitDevice {
    pub(crate) ptr: mlx_sys::mlx_device,
    pub(crate) init_success: bool,
}

impl Default for MaybeUninitDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl MaybeUninitDevice {
    pub fn new() -> Self {
        unsafe {
            Self {
                ptr: mlx_sys::mlx_device_new(),
                init_success: false,
            }
        }
    }
}

impl Drop for MaybeUninitDevice {
    fn drop(&mut self) {
        if !self.init_success {
            unsafe {
                mlx_sys::mlx_device_free(self.ptr);
            }
        }
    }
}

impl Guard<crate::Device> for MaybeUninitDevice {
    type MutRawPtr = *mut mlx_sys::mlx_device;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        &mut self.ptr
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }

    fn try_into_guarded(self) -> Result<crate::Device, Exception> {
        debug_assert!(self.init_success);
        Ok(crate::Device { c_device: self.ptr })
    }
}

impl Guarded for crate::DeviceType {
    type Guard = mlx_sys::mlx_device_type;
}

impl Guard<crate::DeviceType> for mlx_sys::mlx_device_type {
    type MutRawPtr = *mut mlx_sys::mlx_device_type;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        self
    }

    fn set_init_success(&mut self, _: bool) {}

    fn try_into_guarded(self) -> Result<crate::DeviceType, Exception> {
        match self {
            mlx_sys::mlx_device_type__MLX_CPU => Ok(crate::DeviceType::Cpu),
            mlx_sys::mlx_device_type__MLX_GPU => Ok(crate::DeviceType::Gpu),
            _ => Err(Exception {
                what: "Unknown device type".to_string(),
                location: std::panic::Location::caller(),
            }),
        }
    }
}

impl Guarded for crate::Device {
    type Guard = MaybeUninitDevice;
}

pub(crate) struct MaybeUninitStream {
    pub(crate) ptr: mlx_sys::mlx_stream,
    pub(crate) init_success: bool,
}

impl Default for MaybeUninitStream {
    fn default() -> Self {
        Self::new()
    }
}

impl MaybeUninitStream {
    pub fn new() -> Self {
        unsafe {
            Self {
                ptr: mlx_sys::mlx_stream_new(),
                init_success: false,
            }
        }
    }
}

impl Drop for MaybeUninitStream {
    fn drop(&mut self) {
        if !self.init_success {
            unsafe {
                mlx_sys::mlx_stream_free(self.ptr);
            }
        }
    }
}

impl Guard<crate::Stream> for MaybeUninitStream {
    type MutRawPtr = *mut mlx_sys::mlx_stream;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        &mut self.ptr
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }

    fn try_into_guarded(self) -> Result<crate::Stream, Exception> {
        debug_assert!(self.init_success);
        Ok(crate::Stream { c_stream: self.ptr })
    }
}

impl Guarded for crate::Stream {
    type Guard = MaybeUninitStream;
}

pub(crate) struct MaybeUninitSafeTensors {
    pub(crate) c_data: mlx_sys::mlx_map_string_to_array,
    pub(crate) c_metadata: mlx_sys::mlx_map_string_to_string,
    pub(crate) init_success: bool,
}

impl Default for MaybeUninitSafeTensors {
    fn default() -> Self {
        Self::new()
    }
}

impl MaybeUninitSafeTensors {
    pub fn new() -> Self {
        unsafe {
            Self {
                c_metadata: mlx_sys::mlx_map_string_to_string_new(),
                c_data: mlx_sys::mlx_map_string_to_array_new(),
                init_success: false,
            }
        }
    }
}

impl Drop for MaybeUninitSafeTensors {
    fn drop(&mut self) {
        if !self.init_success {
            unsafe {
                mlx_sys::mlx_map_string_to_string_free(self.c_metadata);
                mlx_sys::mlx_map_string_to_array_free(self.c_data);
            }
        }
    }
}

impl Guard<crate::utils::io::SafeTensors> for MaybeUninitSafeTensors {
    type MutRawPtr = (
        *mut mlx_sys::mlx_map_string_to_array,
        *mut mlx_sys::mlx_map_string_to_string,
    );

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        (&mut self.c_data, &mut self.c_metadata)
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }

    fn try_into_guarded(self) -> Result<crate::utils::io::SafeTensors, Exception> {
        debug_assert!(self.init_success);
        Ok(crate::utils::io::SafeTensors {
            c_metadata: self.c_metadata,
            c_data: self.c_data,
        })
    }
}

impl Guarded for crate::utils::io::SafeTensors {
    type Guard = MaybeUninitSafeTensors;
}

pub(crate) struct MaybeUninitClosure {
    pub(crate) ptr: mlx_sys::mlx_closure,
    pub(crate) init_success: bool,
}

impl Default for MaybeUninitClosure {
    fn default() -> Self {
        Self::new()
    }
}

impl MaybeUninitClosure {
    pub fn new() -> Self {
        unsafe {
            Self {
                ptr: mlx_sys::mlx_closure_new(),
                init_success: false,
            }
        }
    }
}

impl Drop for MaybeUninitClosure {
    fn drop(&mut self) {
        if !self.init_success {
            unsafe {
                mlx_sys::mlx_closure_free(self.ptr);
            }
        }
    }
}

impl<'a> Guard<crate::utils::Closure<'a>> for MaybeUninitClosure {
    type MutRawPtr = *mut mlx_sys::mlx_closure;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        &mut self.ptr
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }

    fn try_into_guarded(self) -> Result<crate::utils::Closure<'a>, Exception> {
        debug_assert!(self.init_success);
        Ok(crate::utils::Closure {
            c_closure: self.ptr,
            lt_marker: std::marker::PhantomData,
        })
    }
}

impl Guarded for crate::utils::Closure<'_> {
    type Guard = MaybeUninitClosure;
}

pub(crate) struct MaybeUninitClosureValueAndGrad {
    pub(crate) ptr: mlx_sys::mlx_closure_value_and_grad,
    pub(crate) init_success: bool,
}

impl Default for MaybeUninitClosureValueAndGrad {
    fn default() -> Self {
        Self::new()
    }
}

impl MaybeUninitClosureValueAndGrad {
    pub fn new() -> Self {
        unsafe {
            Self {
                ptr: mlx_sys::mlx_closure_value_and_grad_new(),
                init_success: false,
            }
        }
    }
}

impl Drop for MaybeUninitClosureValueAndGrad {
    fn drop(&mut self) {
        if !self.init_success {
            unsafe {
                mlx_sys::mlx_closure_value_and_grad_free(self.ptr);
            }
        }
    }
}

impl Guard<crate::transforms::ClosureValueAndGrad> for MaybeUninitClosureValueAndGrad {
    type MutRawPtr = *mut mlx_sys::mlx_closure_value_and_grad;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        &mut self.ptr
    }

    fn set_init_success(&mut self, success: bool) {
        self.init_success = success;
    }

    fn try_into_guarded(self) -> Result<crate::transforms::ClosureValueAndGrad, Exception> {
        debug_assert!(self.init_success);
        Ok(crate::transforms::ClosureValueAndGrad {
            c_closure_value_and_grad: self.ptr,
        })
    }
}

impl Guarded for crate::transforms::ClosureValueAndGrad {
    type Guard = MaybeUninitClosureValueAndGrad;
}

macro_rules! impl_guarded_for_primitive {
    ($type:ty) => {
        impl Guarded for $type {
            type Guard = $type;
        }

        impl Guard<$type> for $type {
            type MutRawPtr = *mut $type;

            fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
                self
            }

            fn set_init_success(&mut self, _: bool) { }

            fn try_into_guarded(self) -> Result<$type, Exception> {
                Ok(self)
            }
        }
    };

    ($($type:ty),*) => {
        $(impl_guarded_for_primitive!($type);)*
    };
}

impl_guarded_for_primitive!(bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, ());

impl Guarded for f16 {
    type Guard = float16_t;
}

impl Guard<f16> for float16_t {
    type MutRawPtr = *mut float16_t;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        self
    }

    fn set_init_success(&mut self, _: bool) {}

    fn try_into_guarded(self) -> Result<f16, Exception> {
        Ok(f16::from_bits(self.0))
    }
}

impl Guarded for bf16 {
    type Guard = bfloat16_t;
}

impl Guard<bf16> for bfloat16_t {
    type MutRawPtr = *mut bfloat16_t;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        self
    }

    fn set_init_success(&mut self, _: bool) {}

    fn try_into_guarded(self) -> Result<bf16, Exception> {
        Ok(bf16::from_bits(self))
    }
}

impl Guarded for complex64 {
    type Guard = __BindgenComplex<f32>;
}

impl Guard<complex64> for __BindgenComplex<f32> {
    type MutRawPtr = *mut __BindgenComplex<f32>;

    fn as_mut_raw_ptr(&mut self) -> Self::MutRawPtr {
        self
    }

    fn set_init_success(&mut self, _: bool) {}

    fn try_into_guarded(self) -> Result<complex64, Exception> {
        Ok(complex64::new(self.re, self.im))
    }
}
