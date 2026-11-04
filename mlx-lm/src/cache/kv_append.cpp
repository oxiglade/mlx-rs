#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include "mlx/primitives.h"

#ifdef MLX_LM_METAL
#include "mlx/backend/gpu/copy.h"
#endif

namespace {
using namespace mlx::core;

class KvAppend : public UnaryPrimitive {
 public:
  KvAppend(Stream stream, int offset) : UnaryPrimitive(stream), offset_(offset) {}

  const char* name() const override { return "KvAppend"; }

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::invalid_argument("KvAppend requires a GPU stream");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
#ifdef MLX_LM_METAL
    const auto& cache = inputs[0];
    const auto& update = inputs[1];
    if (!cache.flags().row_contiguous || cache.data_size() != cache.size()) {
      throw std::invalid_argument("KvAppend requires full contiguous storage");
    }
    out.copy_shared_buffer(cache);
    copy_gpu_inplace(
        update,
        out,
        update.shape(),
        update.strides(),
        out.strides(),
        0,
        static_cast<int64_t>(offset_) * out.strides()[2],
        CopyType::GeneralGeneral,
        stream());
#else
    throw std::invalid_argument("KvAppend was built without Metal");
#endif
  }

 private:
  int offset_;
};
} // namespace

extern "C" int mlxrs_kv_append(
    mlx_array* out, mlx_array cache, mlx_array update, int offset, mlx_stream stream,
    void (*error)(const char*, void*), void* context) {
  try {
    const auto& c = mlx_array_get_(cache);
    const auto& u = mlx_array_get_(update);
    const auto& s = mlx_stream_get_(stream);
    if (!out || s.device != Device::gpu || c.ndim() != 4 || u.ndim() != 4 ||
        c.dtype() != u.dtype() ||
        (c.dtype() != float32 && c.dtype() != float16 && c.dtype() != bfloat16) ||
        c.shape(0) != u.shape(0) || c.shape(1) != u.shape(1) || c.shape(3) != u.shape(3) ||
        u.size() == 0 || u.size() > UINT32_MAX || offset < 0 ||
        u.shape(2) > c.shape(2) || offset > c.shape(2) - u.shape(2)) {
      throw std::invalid_argument("Invalid KvAppend inputs");
    }
    mlx_array_set_(*out, array(c.shape(), c.dtype(), std::make_shared<KvAppend>(s, offset), {c, u}));
    return 0;
  } catch (const std::exception& e) {
    error(e.what(), context);
    return 1;
  } catch (...) {
    error("Unknown KvAppend construction exception", context);
    return 1;
  }
}

extern "C" int mlxrs_kv_synchronize(
    mlx_stream stream, void (*error)(const char*, void*), void* context) {
  try {
    synchronize(mlx_stream_get_(stream));
    return 0;
  } catch (const std::exception& e) {
    error(e.what(), context);
    return 1;
  } catch (...) {
    error("Unknown KV stream synchronization exception", context);
    return 1;
  }
}
