#ifndef ABI_FIXTURE_TYPES_H
#define ABI_FIXTURE_TYPES_H

typedef unsigned int mlx_size;
typedef enum mlx_mode_ { mlx_read = 1, mlx_write = 4 } mlx_mode;
typedef struct mlx_handle_ { void *ctx; } mlx_handle;
typedef int (*mlx_callback)(void *ctx, mlx_size count);
typedef struct mlx_io_vtable_ {
    mlx_callback read;
    void (*close)(void *ctx);
} mlx_io_vtable;
typedef union mlx_value_ { int integer; double real; } mlx_value;
#define mlx_limit 16

#endif
