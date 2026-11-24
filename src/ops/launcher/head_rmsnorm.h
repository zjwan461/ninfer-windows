#pragma once

#include "ninfer/ops/head_rmsnorm.h"

namespace ninfer::ops::detail {

void head_rmsnorm_launch(const Tensor& x, float eps, Tensor& out, cudaStream_t stream);

// Two RMSNorm passes into one contiguous buffer: the first stream fills [0, width), the second
// [width, 2*width).
void rmsnorm_dual_launch(const Tensor& first, const Tensor& second, Tensor& out, float eps,
                         cudaStream_t stream);

} // namespace ninfer::ops::detail
