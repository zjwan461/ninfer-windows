#pragma once

#include "ninfer/ops/dynamic_short_conv.h"

namespace ninfer::ops::detail {

void dynamic_short_conv_launch(const Tensor& x, const Tensor& static_weight,
                               const Tensor& dynamic_weight, const Tensor& conv_state_in,
                               Tensor& conv_state_out, Tensor& out, int heads,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
