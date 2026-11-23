#pragma once

#include "ninfer/ops/moe_router.h"

namespace ninfer::ops::detail {

void moe_router_launch(const Tensor& x, const Tensor& weight, const Tensor& bias, int top_k,
                       float scale, Tensor& indices, Tensor& weights, cudaStream_t stream);

} // namespace ninfer::ops::detail
