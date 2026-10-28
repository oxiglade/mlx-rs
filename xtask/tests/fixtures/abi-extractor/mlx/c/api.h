#ifndef ABI_FIXTURE_API_H
#define ABI_FIXTURE_API_H
#include "mlx/c/types.h"

mlx_handle mlx_handle_new(void);
int mlx_handle_free(mlx_handle handle);
int mlx_apply(mlx_handle handle, const mlx_io_vtable *io, mlx_size count);

#endif
