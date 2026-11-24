// ninfer::ops - dynamic_short_conv wrapper: public validation and launcher dispatch.
#include "ninfer/ops/dynamic_short_conv.h"

#include "ops/launcher/dynamic_short_conv.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_bf16(const Tensor& t, const char* label) {
    if (t.dtype != DType::BF16) {
        throw std::invalid_argument(std::string("dynamic_short_conv: ") + label + " must be BF16");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string("dynamic_short_conv: ") + label +
                                    " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string("dynamic_short_conv: ") + label +
                                    " data must be non-null");
    }
}

} // namespace

void dynamic_short_conv(const Tensor& x, const Tensor& static_weight, const Tensor& dynamic_weight,
                        const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                        int heads, cudaStream_t stream) {
    require_bf16(x, "x");
    require_bf16(static_weight, "static_weight");
    require_bf16(dynamic_weight, "dynamic_weight");
    require_bf16(conv_state_in, "conv_state_in");
    require_bf16(conv_state_out, "conv_state_out");
    require_bf16(out, "out");

    const auto channels = std::int64_t(x.ne[0]);
    const auto tokens = std::int64_t(x.ne[1]);
    if (channels <= 0 || tokens <= 0) {
        throw std::invalid_argument("dynamic_short_conv: x must be [channels, tokens]");
    }
    if (heads <= 0 || channels % heads != 0) {
        throw std::invalid_argument("dynamic_short_conv: channels must divide by heads");
    }
    if (static_weight.numel() != channels * 4) {
        throw std::invalid_argument("dynamic_short_conv: static_weight must be [4, channels]");
    }
    if (dynamic_weight.numel() != tokens * heads * 4) {
        throw std::invalid_argument("dynamic_short_conv: dynamic_weight must be [tokens, heads*4]");
    }
    if (conv_state_in.numel() != channels * 3 || conv_state_out.numel() != channels * 3) {
        throw std::invalid_argument("dynamic_short_conv: state must be [channels, 3]");
    }
    if (out.numel() != channels * tokens) {
        throw std::invalid_argument("dynamic_short_conv: out must match x");
    }

    detail::dynamic_short_conv_launch(x, static_weight, dynamic_weight, conv_state_in,
                                      conv_state_out, out, heads, stream);
}

} // namespace ninfer::ops
