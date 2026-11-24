// ninfer::ops - causal_dconv_residual wrapper: public validation and launcher dispatch.
#include "ninfer/ops/causal_dconv_residual.h"

#include "ops/launcher/causal_dconv_residual.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_bf16(const Tensor& t, const char* label) {
    if (t.dtype != DType::BF16) {
        throw std::invalid_argument(std::string("causal_dconv_residual: ") + label +
                                    " must be BF16");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string("causal_dconv_residual: ") + label +
                                    " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string("causal_dconv_residual: ") + label +
                                    " data must be non-null");
    }
}

} // namespace

void causal_dconv_residual(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                           Tensor& conv_state_out, Tensor& out, cudaStream_t stream) {
    require_bf16(x, "x");
    require_bf16(weight, "weight");
    require_bf16(conv_state_in, "conv_state_in");
    require_bf16(conv_state_out, "conv_state_out");
    require_bf16(out, "out");

    const auto channels = std::int64_t(x.ne[0]);
    const auto tokens = std::int64_t(x.ne[1]);
    if (channels <= 0 || tokens <= 0) {
        throw std::invalid_argument("causal_dconv_residual: x must be [channels, tokens]");
    }
    if (weight.numel() != channels * 4) {
        throw std::invalid_argument("causal_dconv_residual: weight must be [channels, 4]");
    }
    if (conv_state_in.numel() != channels * 3 || conv_state_out.numel() != channels * 3) {
        throw std::invalid_argument("causal_dconv_residual: state must be [channels, 3]");
    }
    if (out.numel() != channels * tokens) {
        throw std::invalid_argument("causal_dconv_residual: out must match x");
    }
    if (reinterpret_cast<const void*>(conv_state_in.data) != conv_state_out.data &&
        reinterpret_cast<const void*>(x.data) == out.data) {
        throw std::invalid_argument("causal_dconv_residual: out must not alias x");
    }

    detail::causal_dconv_residual_launch(x, weight, conv_state_in, conv_state_out, out, stream);
}

} // namespace ninfer::ops
