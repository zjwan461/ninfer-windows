#pragma once

#include "ninfer/ops/scaled_l2_clip.h"

namespace ninfer::ops::detail {

void scaled_l2_clip_launch(const Tensor& x, const Tensor& scale, const Tensor* positions,
                           const ScaledL2ClipArgs& args, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
