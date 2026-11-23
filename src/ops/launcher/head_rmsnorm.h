#pragma once

#include "ninfer/ops/head_rmsnorm.h"

namespace ninfer::ops::detail {

void head_rmsnorm_launch(const Tensor& x, float eps, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
