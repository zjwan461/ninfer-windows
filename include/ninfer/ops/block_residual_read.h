#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Inkling 2's block attention residual: a sublayer input is read out of the block-state stack by
// one or more learned queries instead of being copied.
//
//   s_norm[s, :]  = state[s, :] / sqrt(mean(state[s, :]^2) + eps)      per state, no gain
//   logits[q, s]  = dot(query[q, :], s_norm[s, :]) / sqrt(hidden)
//   out[q, :]     = sum_s softmax(logits[q, :])[s] * s_norm[s, :]
//
// `query` is contiguous FP32 with `hidden` as the fastest dimension and the query count on ne[1];
// `states` is the same shape convention (hidden fastest, one row per block state) and `out`
// matches `query`. The reference
// runs this in FP32 over the FP32 queries, so the queries are FP32 here and the states are the
// BF16 stack. `states` may be small (one entry per finished block), which is why this is a
// separate Op rather than a call into the causal attention kernel.
void block_residual_read(const Tensor& query, const Tensor& states, float eps, Tensor& out,
                         cudaStream_t stream);

} // namespace ninfer::ops
