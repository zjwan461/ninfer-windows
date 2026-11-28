#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Captures one position-independent key summary per block of history.
 *
 * `keys` is contiguous BF16 [D,H,T] holding the keys as they are produced before
 * rotation is applied, so every element is already free of the position phase.
 * `block_tokens` consecutive positions form one block; the trailing block may be
 * shorter. `means` is contiguous FP32 [D,H,blocks] with blocks = ceil(T/block_tokens):
 *
 *   means[d,h,b] = (1/n) * sum_{p in block b} float(keys[d,h,p])
 *
 * with n the number of positions in that block. Blocks are accumulated in FP32 and
 * the divisor is the exact block length, so the result is the arithmetic mean of the
 * stored BF16 keys in the order they appear. The Op has no workspace, reads only
 * `keys`, and writes all of `means`.
 */
void kv_block_mean_keys(const Tensor& keys, std::int32_t block_tokens, Tensor& means,
                        cudaStream_t stream);

/**
 * Scores history blocks against one pooled query by mean-key cosine similarity.
 *
 * `query` is contiguous BF16 [D,Q] holding Q query heads and `means` is contiguous
 * FP32 [D,H,B] from kv_block_mean_keys, with Q a positive multiple of H. For block b
 * and KV head h the query vector is the mean of that KV head's group of query heads
 * (query heads [h*Q/H, (h+1)*Q/H)), and
 *
 *   score[b] = (1/H) * sum_h cos(query_mean_h, means[:,h,b])
 *   cos(a,b) = dot(a,b) / (|a| * |b|),   cos = 0 when |a| = 0 or |b| = 0
 *
 * `scores` is contiguous FP32 [B]. The Op has no workspace, reads only its inputs,
 * and writes all of `scores`.
 */
void kv_block_scores(const Tensor& query, const Tensor& means, Tensor& scores,
                     cudaStream_t stream);

} // namespace ninfer::ops