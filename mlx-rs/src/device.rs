use std::ffi::CStr;

use crate::{
    error::Result,
    utils::{guard::Guarded, SUCCESS},
};

///Type of device.
#[derive(num_enum::IntoPrimitive, Debug, Clone, Copy)]
#[repr(u32)]
pub enum DeviceType {
    /// CPU device
    Cpu = mlx_sys::mlx_device_type__MLX_CPU,

    /// GPU device
    Gpu = mlx_sys::mlx_device_type__MLX_GPU,
}

/// Representation of a Device in MLX.
pub struct Device {
    pub(crate) c_device: mlx_sys::mlx_device,
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlx_sys::mlx_device_equal(self.c_device, other.c_device) }
    }
}

impl Device {
    /// Create a new [`Device`]
    pub fn new(device_type: DeviceType, index: i32) -> Device {
        let c_device = unsafe { mlx_sys::mlx_device_new_type(device_type.into(), index) };
        Device { c_device }
    }

    /// Try to get the default device.
    pub fn try_default() -> Result<Self> {
        Device::try_from_op(|res| unsafe { mlx_sys::mlx_get_default_device(res) })
    }

    /// Create a default CPU device.
    pub fn cpu() -> Device {
        Device::new(DeviceType::Cpu, 0)
    }

    /// Create a default GPU device.
    pub fn gpu() -> Device {
        Device::new(DeviceType::Gpu, 0)
    }

    /// Get the device index
    pub fn get_index(&self) -> Result<i32> {
        i32::try_from_op(|res| unsafe { mlx_sys::mlx_device_get_index(res, self.c_device) })
    }

    /// Get the device type
    pub fn get_type(&self) -> Result<DeviceType> {
        DeviceType::try_from_op(|res| unsafe { mlx_sys::mlx_device_get_type(res, self.c_device) })
    }

    /// Set the default device.
    ///
    /// # Example:
    ///
    /// ```rust
    /// use mlx_rs::{Device, DeviceType};
    /// Device::set_default(&Device::new(DeviceType::Cpu, 1));
    /// ```
    ///
    /// By default, this is `gpu()`.
    pub fn set_default(device: &Device) {
        unsafe { mlx_sys::mlx_set_default_device(device.c_device) };
    }

    fn describe(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        unsafe {
            let mut mlx_str = mlx_sys::mlx_string_new();
            let result = match mlx_sys::mlx_device_tostring(&mut mlx_str as *mut _, self.c_device) {
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

impl Drop for Device {
    fn drop(&mut self) {
        let status = unsafe { mlx_sys::mlx_device_free(self.c_device) };
        debug_assert_eq!(status, SUCCESS);
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::try_default().unwrap()
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.describe(f)
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.describe(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmt() {
        let device = Device::default();
        let description = format!("{device}");
        #[cfg(feature = "metal")]
        assert_eq!(description, "Device(gpu, 0)");
        #[cfg(not(feature = "metal"))]
        assert_eq!(description, "Device(cpu, 0)");
    }
}
