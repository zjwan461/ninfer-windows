#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

using Bf16Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// Runtime-extent fallback for shapes without a tuned route. Same contract as a tuned launch:
// y[n, t] = W[n, k] . x[k, t] with BF16 operands and an FP32 accumulator.
void launch_bf16_generic(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

// Runtime-extent BF16 decode GEMV (one token), used when no tuned shape covers the geometry.
void launch_bf16_generic_gemv(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
