# project(mlx) runs after CMake loads the cache and any toolchain file.
if(MLX_RS_METAL_BASE)
  set(target "${CMAKE_OSX_DEPLOYMENT_TARGET}")
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND target STREQUAL "")
    execute_process(
      COMMAND sw_vers -productVersion
      OUTPUT_VARIABLE target
      OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  endif()
  set(sdk "${CMAKE_OSX_SYSROOT}")
  if(NOT IS_ABSOLUTE "${sdk}")
    if(sdk STREQUAL "")
      set(sdk macosx)
    endif()
    execute_process(
      COMMAND xcrun --sdk "${sdk}" --show-sdk-path
      OUTPUT_VARIABLE sdk
      OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  endif()
  string(SHA256 key "${CMAKE_SYSTEM_NAME}\n${target}\n${sdk}\n${CMAKE_BUILD_TYPE}")
  set(MLX_METAL_PATH
      "${MLX_RS_METAL_BASE}-${CMAKE_SYSTEM_NAME}-${target}-${CMAKE_BUILD_TYPE}-${key}"
      CACHE PATH "MLX metallib directory" FORCE)
endif()
file(MAKE_DIRECTORY "${MLX_METAL_PATH}")
file(WRITE "${CMAKE_BINARY_DIR}/mlx-metal-path.txt" "${MLX_METAL_PATH}")
