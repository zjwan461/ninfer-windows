// ninfer::ops - message_merge wrapper: public validation and launcher dispatch.
#include "ninfer/ops/message_merge.h"

#include "ops/launcher/message_merge.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_bf16(const Tensor& t, const char* label) {
    if (t.dtype != DType::BF16) {
        throw std::invalid_argument(std::string("message_merge: ") + label + " must be BF16");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string("message_merge: ") + label + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string("message_merge: ") + label + " data must be non-null");
    }
}

} // namespace

void message_merge(const Tensor& x, const Tensor& message, const Tensor& forget, const Tensor& gain,
                   Tensor& out, cudaStream_t stream) {
    require_bf16(x, "x");
    require_bf16(message, "message");
    require_bf16(forget, "forget");
    require_bf16(gain, "gain");
    require_bf16(out, "out");

    const auto count = x.numel();
    if (message.numel() != count || forget.numel() != count || gain.numel() != count ||
        out.numel() != count) {
        throw std::invalid_argument("message_merge: every operand must have the same element count");
    }
    if (count == 0) { return; }

    detail::message_merge_launch(x, message, forget, gain, out, stream);
}

} // namespace ninfer::ops
