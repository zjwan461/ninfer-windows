// ninfer::ops - head_rmsnorm wrapper: public validation and launcher dispatch.
#include "ninfer/ops/head_rmsnorm.h"

#include "ops/launcher/head_rmsnorm.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include "ops/op_counters.h"

namespace ninfer::ops {

void head_rmsnorm(const Tensor& x, float eps, Tensor& out, cudaStream_t stream) {
    ninfer::ops::op_counters().norm.fetch_add(1, std::memory_order_relaxed);
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("head_rmsnorm: x/out must be BF16");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("head_rmsnorm: x/out must be contiguous");
    }
    if (x.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("head_rmsnorm: x/out data must be non-null");
    }
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument("head_rmsnorm: eps must be positive and finite");
    }
    if (x.numel() != out.numel()) {
        throw std::invalid_argument("head_rmsnorm: x/out shapes must match");
    }
    if (x.ne[0] <= 0) { throw std::invalid_argument("head_rmsnorm: row width must be positive"); }
    if (x.numel() == 0) { return; }

    detail::head_rmsnorm_launch(x, eps, out, stream);
}

} // namespace ninfer::ops
