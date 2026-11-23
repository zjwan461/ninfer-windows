#pragma once

#include "ninfer/ops/softcapped_swiglu.h"

namespace ninfer::ops::detail {

void softcapped_swiglu_launch(const Tensor& gate_up, float gate_cap, float up_cap, Tensor& out,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
