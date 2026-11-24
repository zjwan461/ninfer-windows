// ninfer::ops - moe_router wrapper: public validation and launcher dispatch.
#include "ninfer/ops/moe_router.h"

#include "ops/launcher/moe_router.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

void moe_router(const Tensor& x, const Tensor& weight, const Tensor& bias, int top_k, float scale,
                Tensor& indices, Tensor& weights, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || !x.is_contiguous() || x.data == nullptr) {
        throw std::invalid_argument("moe_router: x must be contiguous BF16");
    }
    if (weight.dtype != DType::FP32 || !weight.is_contiguous() || weight.data == nullptr) {
        throw std::invalid_argument("moe_router: weight must be contiguous FP32");
    }
    if (bias.dtype != DType::FP32 || !bias.is_contiguous() || bias.data == nullptr) {
        throw std::invalid_argument("moe_router: bias must be contiguous FP32");
    }
    if (indices.dtype != DType::I32 || !indices.is_contiguous() || indices.data == nullptr) {
        throw std::invalid_argument("moe_router: indices must be contiguous INT32");
    }
    if (weights.dtype != DType::FP32 || !weights.is_contiguous() || weights.data == nullptr) {
        throw std::invalid_argument("moe_router: weights must be contiguous FP32");
    }
    if (!(scale > 0.0F) || !std::isfinite(scale)) {
        throw std::invalid_argument("moe_router: scale must be positive and finite");
    }

    const auto experts = std::int64_t(weight.ne[0]);
    const auto hidden = std::int64_t(x.ne[0]);
    const auto tokens = std::int64_t(x.ne[1]);
    if (hidden <= 0 || tokens <= 0 || experts <= 0) {
        throw std::invalid_argument("moe_router: invalid geometry");
    }
    if (weight.ne[1] != hidden || bias.numel() != experts) {
        throw std::invalid_argument("moe_router: weight must be [experts, hidden] and bias [experts]");
    }
    if (top_k <= 0 || top_k > experts) {
        throw std::invalid_argument("moe_router: top_k must be in [1, experts]");
    }
    if (indices.numel() != experts * 0 + std::int64_t(top_k) * tokens ||
        weights.numel() != std::int64_t(top_k) * tokens) {
        throw std::invalid_argument("moe_router: outputs must be [top_k, tokens]");
    }

    detail::moe_router_launch(x, weight, bias, top_k, scale, indices, weights, stream);
}

} // namespace ninfer::ops
