#pragma once
#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
using Nvfp4Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// Runtime-extent NVFP4 decode route for shapes without a tuned kernel: y[n] = W[n, k] . x[k] /
// weight_scale_divisor, FP32 accumulation, BF16 store.
void launch_nvfp4_generic_gemv(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail


