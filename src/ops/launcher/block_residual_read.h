#pragma once

#include "ninfer/ops/block_residual_read.h"

namespace ninfer::ops::detail {

void block_residual_read_launch(const Tensor& query, const Tensor& states, float eps, Tensor& out,
                                cudaStream_t stream);

} // namespace ninfer::ops::detail
