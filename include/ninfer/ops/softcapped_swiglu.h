#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Inkling 2's routed-expert activation. Both branches are bounded first, and the sigmoid sees the
// raw gate rather than the bounded one:
//
//   softcap(z, c) = tanh(z / c) * c
//   ideal[i, t]   = softcap(gate[i, t], gate_cap) * sigmoid(gate[i, t]) * softcap(up[i, t], up_cap)
//
// with gate_cap 4 and up_cap 25 for this checkpoint. `gate_up` is the fused projection,
// contiguous BF16 [2 * inter, tokens], gate in the first half and up in the second; `out` is
// [inter, tokens]. The activation runs in FP32 and the store is BF16.
void softcapped_swiglu(const Tensor& gate_up, float gate_cap, float up_cap, Tensor& out,
                       cudaStream_t stream);

} // namespace ninfer::ops
