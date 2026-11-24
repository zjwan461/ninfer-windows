#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void logits_softcap_launch(Tensor& logits, float cap, cudaStream_t stream);

} // namespace ninfer::ops::detail
