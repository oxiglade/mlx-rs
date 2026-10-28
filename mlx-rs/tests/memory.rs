use mlx_rs::memory;

struct MemoryLimitGuard(usize);

impl Drop for MemoryLimitGuard {
    fn drop(&mut self) {
        let _ = memory::set_memory_limit(self.0);
    }
}

struct CacheLimitGuard(usize);

impl Drop for CacheLimitGuard {
    fn drop(&mut self) {
        let _ = memory::set_cache_limit(self.0);
    }
}

struct WiredLimitGuard(usize);

impl Drop for WiredLimitGuard {
    fn drop(&mut self) {
        let _ = memory::set_wired_limit(self.0);
    }
}

#[test]
fn process_global_memory_controls_restore_their_limits() {
    let original_memory_limit = memory::memory_limit().unwrap();
    let temporary_memory_limit = if original_memory_limit == 0 {
        1
    } else {
        original_memory_limit - 1
    };
    assert_eq!(
        memory::set_memory_limit(temporary_memory_limit).unwrap(),
        original_memory_limit
    );
    let memory_guard = MemoryLimitGuard(original_memory_limit);
    assert_eq!(memory::memory_limit().unwrap(), temporary_memory_limit);
    assert_eq!(
        memory::set_memory_limit(original_memory_limit).unwrap(),
        temporary_memory_limit
    );
    std::mem::forget(memory_guard);

    let original_cache_limit = memory::set_cache_limit(0).unwrap();
    let cache_guard = CacheLimitGuard(original_cache_limit);
    assert_eq!(memory::set_cache_limit(original_cache_limit).unwrap(), 0);
    std::mem::forget(cache_guard);

    let original_wired_limit = memory::set_wired_limit(0).unwrap();
    let wired_guard = WiredLimitGuard(original_wired_limit);
    assert_eq!(memory::set_wired_limit(original_wired_limit).unwrap(), 0);
    std::mem::forget(wired_guard);

    memory::clear_cache().unwrap();
    assert_eq!(memory::cache_memory().unwrap(), 0);
    memory::reset_peak_memory().unwrap();
    let _ = memory::active_memory().unwrap();
    let _ = memory::cache_memory().unwrap();
    let _ = memory::peak_memory().unwrap();
}
