const LEAKED_ITERATOR_HANDLES: usize = 16;

#[test]
#[ignore = "run only by cargo run -p xtask -- verify-ffi --calibrate"]
fn deliberate_iterator_handle_leak() {
    unsafe {
        let map = mlx_sys::mlx_map_string_to_string_new();
        for _ in 0..LEAKED_ITERATOR_HANDLES {
            let _iterator = mlx_sys::mlx_map_string_to_string_iterator_new(map);
        }
        assert_eq!(mlx_sys::mlx_map_string_to_string_free(map), 0);
    }
}
