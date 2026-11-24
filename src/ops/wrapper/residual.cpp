// ninfer::ops - residual helpers wrapper: public validation and launcher dispatch.
#include "ninfer/ops/residual.h"

#include "ops/launcher/residual.h"

#include <stdexcept>
#include <string>
#include "ops/op_counters.h"

namespace ninfer::ops {
namespace {

void require(const Tensor& t, DType dtype, const char* label) {
    if (t.dtype != dtype) {
        throw std::invalid_argument(std::string("residual: ") + label + " has the wrong dtype");
    }
    if (!t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("residual: ") + label + " must be contiguous");
    }
}

void require_same(const Tensor& a, const Tensor& b, const Tensor& out) {
    if (a.numel() != b.numel() || a.numel() != out.numel()) {
        throw std::invalid_argument("residual: operands must have the same element count");
    }
}

} // namespace

void cast_bf16_to_f32(Tensor& out, const Tensor& in, cudaStream_t stream) {
    require(in, DType::BF16, "in");
    require(out, DType::FP32, "out");
    require_same(in, in, out);
    detail::cast_bf16_to_f32_launch(out, in, stream);
}

void cast_f32_to_bf16(Tensor& out, const Tensor& in, cudaStream_t stream) {
    require(in, DType::FP32, "in");
    require(out, DType::BF16, "out");
    require_same(in, in, out);
    detail::cast_f32_to_bf16_launch(out, in, stream);
}

void residual_add_f32(Tensor& out, const Tensor& base, const Tensor& addend, cudaStream_t stream) {
    require(base, DType::FP32, "base");
    require(addend, DType::BF16, "addend");
    require(out, DType::FP32, "out");
    require_same(base, addend, out);
    detail::residual_add_f32_launch(out, base, addend, stream);
}

void residual_store_bf16(Tensor& out, const Tensor& base, const Tensor& addend,
                         cudaStream_t stream) {
    require(base, DType::FP32, "base");
    require(addend, DType::BF16, "addend");
    require(out, DType::BF16, "out");
    require_same(base, addend, out);
    detail::residual_store_bf16_launch(out, base, addend, stream);
}

void axpy_bf16_into_f32_device_scale(Tensor& acc, const Tensor& x, const float* scales, int slot,
                                     cudaStream_t stream) {
    ninfer::ops::op_counters().accumulate.fetch_add(1, std::memory_order_relaxed);
    if (scales == nullptr) { throw std::invalid_argument("axpy: scales must be non-null"); }
    if (slot < 0) { throw std::invalid_argument("axpy: slot must be non-negative"); }
    if (acc.numel() != x.numel()) { throw std::invalid_argument("axpy: acc and x must match"); }
    detail::axpy_bf16_into_f32_device_launch(acc, x, scales, slot, stream);
}

void axpy_bf16_into_f32(Tensor& acc, const Tensor& x, float scale, cudaStream_t stream) {
    require(acc, DType::FP32, "acc");
    require(x, DType::BF16, "x");
    require_same(acc, x, acc);
    detail::axpy_bf16_into_f32_launch(acc, x, scale, stream);
}

} // namespace ninfer::ops
