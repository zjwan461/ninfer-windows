#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Inkling 2's cross-pass message merge. A later loop pass does not replace its input with the
// previous pass's state; it blends the two through two per-channel gates:
//
//   ideal[i] = (1 - tanh(forget[i])) * x[i] + tanh(gain[i]) * message[i]
//
// `x`, `message`, `forget` and `gain` are contiguous BF16 of the same element count, and `out`
// matches them. The blend runs in FP32 end to end, matching the reference, because the gates are
// trained as continuous multipliers rather than as probabilities.
void message_merge(const Tensor& x, const Tensor& message, const Tensor& forget, const Tensor& gain,
                   Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
