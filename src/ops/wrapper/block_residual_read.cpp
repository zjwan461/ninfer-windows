// ninfer::ops - block_residual_read wrapper: public validation and launcher dispatch.
#include "ninfer/ops/block_residual_read.h"

#include "ops/launcher/block_residual_read.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

void block_residual_read(const Tensor& query, const Tensor& states, float eps, Tensor& out,
                         cudaStream_t stream) {
    if (query.dtype != DType::FP32 || !query.is_contiguous() || query.data == nullptr) {
        throw std::invalid_argument("block_residual_read: query must be contiguous FP32");
    }
    if (states.dtype != DType::BF16 || !states.is_contiguous() || states.data == nullptr) {
        throw std::invalid_argument("block_residual_read: states must be contiguous BF16");
    }
    if (out.dtype != DType::BF16 || !out.is_contiguous() || out.data == nullptr) {
        throw std::invalid_argument("block_residual_read: out must be contiguous BF16");
    }
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument("block_residual_read: eps must be positive and finite");
    }
    // ne[0] is the fastest dimension, so the row width is hidden and the row count is ne[1].
    if (query.ne[0] <= 0 || query.ne[1] <= 0 || states.ne[0] != query.ne[0] ||
        states.ne[1] <= 0 || out.ne[0] != query.ne[0] || out.ne[1] != query.ne[1]) {
        throw std::invalid_argument(
            "block_residual_read: query [hidden, q], states [hidden, s], out [hidden, q]");
    }

    detail::block_residual_read_launch(query, states, eps, out, stream);
}

} // namespace ninfer::ops
