#pragma once

// ninfer::ops::detail - private launch prototypes for the KV block pool ops.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void kv_block_mean_keys_launch(const Tensor& keys, std::int32_t block_tokens, Tensor& means,
                               cudaStream_t stream);

void kv_block_scores_launch(const Tensor& query, const Tensor& means, Tensor& scores,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail