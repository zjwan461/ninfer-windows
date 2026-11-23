#pragma once

#include "ninfer/ops/message_merge.h"

namespace ninfer::ops::detail {

void message_merge_launch(const Tensor& x, const Tensor& message, const Tensor& forget,
                          const Tensor& gain, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
