#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Root-mean-square normalisation without a learned gain:
//
//   ideal[d, r] = x[d, r] / sqrt(mean_d(x[.., r]^2) + eps)
//
// Inkling 2 normalises the value projection this way (`use_value_norm_no_gamma`), and its rows are
// one head wide, so the reduction is over the fastest dimension of a `[D, rows]` contiguous BF16
// tensor. The reduction and the reciprocal run in FP32; the store is BF16. `rmsnorm` is the same
// reduction with a gain vector and a tuned dispatch for the 27B shapes; this Op exists because the
// v path has no gain to read and its width is 128 rather than 5120.
void head_rmsnorm(const Tensor& x, float eps, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
