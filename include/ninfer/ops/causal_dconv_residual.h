#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Inkling 2's static short convolution: a depthwise causal width-4 convolution, then the input
// added back and the sum scaled by 1/sqrt(kernel + 1):
//
//   ideal[c, t] = (sum_{j=0..3} weight[c, j] * u[c, t - 3 + j] + x[c, t]) / sqrt(5)
//
// where `u[c, -3..-1]` are the three values held in `conv_state` and `u[c, t] = x[c, t]` for t >= 0.
// There is no activation: unlike the GDN convolution this one only conditions the stream, and the
// residual is part of the Op so the caller never has to re-add.
//
// `x` and `out` are contiguous BF16 [C, T], `weight` is contiguous BF16 [C, 4] and the state is
// contiguous BF16 [C, 3] ordered oldest to newest. `conv_state_in` and `conv_state_out` may be
// disjoint or exactly the same storage; the output receives the trailing three values of
// concat(state, x). No operand may overlap another.
void causal_dconv_residual(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                           Tensor& conv_state_out, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
