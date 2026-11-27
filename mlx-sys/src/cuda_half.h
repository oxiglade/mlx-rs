/* Forced into mlx-c and bindgen by the `cuda` feature. mlx-c's half.h only
 * declares half types on aarch64, so x86_64 (MLX's CUDA backend) would lose
 * the float16/bfloat16 array functions. */
#ifndef MLX_SYS_CUDA_HALF_H
#define MLX_SYS_CUDA_HALF_H

#if defined(__x86_64__) && !defined(HAS_FLOAT16)
#define HAS_FLOAT16
#define HAS_BFLOAT16
#ifdef __cplusplus
/* mlx-c's C++ calls item<float16_t>(), which needs MLX's own half types. */
#include "mlx/types/half_types.h"
typedef mlx::core::float16_t float16_t;
typedef mlx::core::bfloat16_t bfloat16_t;
#else
/* Same bindgen types as aarch64: __BindgenFloat16 and u16. */
typedef _Float16 float16_t;
typedef __bf16 bfloat16_t;
#endif
#endif

#endif
