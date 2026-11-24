// ninfer::ops - softcapped_swiglu wrapper: public validation and launcher dispatch.
#include "ninfer/ops/softcapped_swiglu.h"

#include "ops/launcher/softcapped_swiglu.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include "ops/op_counters.h"

namespace ninfer::ops {

void softcapped_swiglu(const Tensor& gate_up, float gate_cap, float up_cap, Tensor& out,
                       cudaStream_t stream) {
    ninfer::ops::op_counters().swiglu.fetch_add(1, std::memory_order_relaxed);
    if (gate_up.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("softcapped_swiglu: gate_up/out must be BF16");
    }
    if (!gate_up.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("softcapped_swiglu: gate_up/out must be contiguous");
    }
    if (gate_up.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("softcapped_swiglu: data must be non-null");
    }
    if (!(gate_cap > 0.0F) || !std::isfinite(gate_cap) || !(up_cap > 0.0F) ||
        !std::isfinite(up_cap)) {
        throw std::invalid_argument("softcapped_swiglu: caps must be positive and finite");
    }
    if (out.ne[0] <= 0 || out.ne[1] <= 0) {
        throw std::invalid_argument("softcapped_swiglu: out must be [inter, tokens]");
    }
    if (gate_up.numel() != 2 * out.numel()) {
        throw std::invalid_argument("softcapped_swiglu: gate_up must be twice the size of out");
    }

    detail::softcapped_swiglu_launch(gate_up, gate_cap, up_cap, out, stream);
}

} // namespace ninfer::ops
