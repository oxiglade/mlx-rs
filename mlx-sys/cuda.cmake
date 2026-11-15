# project(mlx.data) runs before mlx-c declares mlxc; patch the target once it exists.
function(mlx_sys_cuda_half)
  target_compile_options(mlxc PRIVATE "SHELL:-include ${MLX_SYS_CUDA_HALF}")
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL mlx_sys_cuda_half)
