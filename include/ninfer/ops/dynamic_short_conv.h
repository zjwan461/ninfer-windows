#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Inkling 2's dynamic short convolution, used for keys and values in the global-attention layers.
// The taps are the sum of a static per-channel filter and a per-token filter projected from the
// layer input, shared by every channel of a key/value head:
//
//   tap[c, t, j] = static_weight[j, c] + dynamic_weight[t, (c / head_width) * kernel + j]
//   ideal[c, t]  = (sum_{j=0..3} tap[c, t, j] * u[c, t - 3 + j] + x[c, t]) / 3
//
// where `u` is the causal window (three state values, then x) and the final divisor is
// `sqrt(2 * kernel + 1)` with kernel = 4. The caller produces `dynamic_weight` with the projection
// it already owns; this Op only applies the convolution and carries the state.
//
// `x` and `out` are contiguous BF16 [C, T]; `static_weight` is [4, C] (tap-major, unlike the static
// convolution's [C, 4]); `dynamic_weight` is [T, heads * 4]; the state is [C, 3] oldest to newest.
void dynamic_short_conv(const Tensor& x, const Tensor& static_weight, const Tensor& dynamic_weight,
                        const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                        int heads, cudaStream_t stream);

} // namespace ninfer::ops
