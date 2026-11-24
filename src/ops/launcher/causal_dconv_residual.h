#pragma once

#include "ninfer/ops/causal_dconv_residual.h"

namespace ninfer::ops::detail {

void causal_dconv_residual_launch(const Tensor& x, const Tensor& weight,
                                  const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
