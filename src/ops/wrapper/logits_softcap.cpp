// ninfer::ops - logits softcap wrapper.
#include "ninfer/ops/logits_softcap.h"

#include "ops/launcher/logits_softcap.h"

#include <cmath>
#include <stdexcept>

namespace ninfer::ops {

void logits_softcap(Tensor& logits, float cap, cudaStream_t stream) {
    if (logits.dtype != DType::BF16 || !logits.is_contiguous() || logits.data == nullptr) {
        throw std::invalid_argument("logits_softcap: logits must be contiguous BF16");
    }
    if (!(cap > 0.0F) || !std::isfinite(cap)) {
        throw std::invalid_argument("logits_softcap: cap must be positive and finite");
    }
    detail::logits_softcap_launch(logits, cap, stream);
}

} // namespace ninfer::ops
