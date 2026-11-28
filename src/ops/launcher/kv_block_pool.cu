// Implements: include/ninfer/ops/kv_block_pool.h
// Match: validated contiguous BF16 keys/query and FP32 means/scores.
// Algorithm assumptions: one thread strides over the head dimension, so both kernels
// keep one block of history and one KV head resident at a time.
#include "ops/launcher/kv_block_pool.h"

#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kMeanKeysThreads     = 256;
constexpr int kScoresLanesPerHead  = 32;

__global__ void kv_block_mean_keys_kernel(const __nv_bfloat16* __restrict__ keys,
                                          float* __restrict__ means, int dim, int heads,
                                          int tokens, int block_tokens) {
    const int block  = static_cast<int>(blockIdx.x);
    const int head   = static_cast<int>(blockIdx.y);
    const int first  = block * block_tokens;
    const int last   = min(tokens, first + block_tokens);
    const int count  = last - first;
    if (count <= 0) { return; }

    const std::size_t head_base  = static_cast<std::size_t>(head) * dim;
    const int         blocks     = gridDim.x;
    const float       inv_count  = 1.0F / static_cast<float>(count);
    for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
        const std::size_t row = (head_base + static_cast<std::size_t>(d)) * tokens;
        float             acc = 0.0F;
        for (int p = first; p < last; ++p) { acc += __bfloat162float(keys[row + p]); }
        means[(head_base + static_cast<std::size_t>(d)) * blocks + block] = acc * inv_count;
    }
}

__global__ void kv_block_scores_kernel(const __nv_bfloat16* __restrict__ query,
                                       const float* __restrict__ means,
                                       float* __restrict__ scores, int dim, int kv_heads,
                                       int q_heads, int blocks) {
    const int        block = static_cast<int>(blockIdx.x);
    const int        warp  = static_cast<int>(threadIdx.x) >> 5;
    const int        lane  = static_cast<int>(threadIdx.x) & 31;
    if (warp >= kv_heads) { return; }

    const int group = q_heads / kv_heads;
    float     dot   = 0.0F;
    float     q_norm = 0.0F;
    float     m_norm = 0.0F;
    for (int d = lane; d < dim; d += kScoresLanesPerHead) {
        float pooled = 0.0F;
        for (int r = 0; r < group; ++r) {
            const std::size_t q_row =
                (static_cast<std::size_t>(warp * group + r)) * dim + static_cast<std::size_t>(d);
            pooled += __bfloat162float(query[q_row]);
        }
        pooled /= static_cast<float>(group);
        const float mean =
            means[(static_cast<std::size_t>(warp) * dim + static_cast<std::size_t>(d)) * blocks +
                  block];
        dot += pooled * mean;
        q_norm += pooled * pooled;
        m_norm += mean * mean;
    }

    for (int offset = kScoresLanesPerHead / 2; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffffU, dot, offset);
        q_norm += __shfl_down_sync(0xffffffffU, q_norm, offset);
        m_norm += __shfl_down_sync(0xffffffffU, m_norm, offset);
    }

    __shared__ float per_head[32];
    if (lane == 0) {
        const float denominator = sqrtf(q_norm) * sqrtf(m_norm);
        per_head[warp] = denominator > 0.0F ? dot / denominator : 0.0F;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0F;
        for (int h = 0; h < kv_heads; ++h) { sum += per_head[h]; }
        scores[block] = sum / static_cast<float>(kv_heads);
    }
}

} // namespace

void kv_block_mean_keys_launch(const Tensor& keys, std::int32_t block_tokens, Tensor& means,
                               cudaStream_t stream) {
    const int dim    = keys.ne[0];
    const int heads  = keys.ne[1];
    const int tokens = keys.ne[2];
    const int blocks = means.ne[2];
    if (dim <= 0 || heads <= 0 || tokens <= 0 || blocks <= 0) { return; }

    const dim3 grid(static_cast<unsigned int>(blocks), static_cast<unsigned int>(heads));
    const int  threads = dim < kMeanKeysThreads ? dim : kMeanKeysThreads;
    kv_block_mean_keys_kernel<<<grid, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(keys.data), static_cast<float*>(means.data), dim, heads,
        tokens, block_tokens);
    CUDA_CHECK(cudaGetLastError());
}

void kv_block_scores_launch(const Tensor& query, const Tensor& means, Tensor& scores,
                            cudaStream_t stream) {
    const int dim      = means.ne[0];
    const int kv_heads = means.ne[1];
    const int blocks   = means.ne[2];
    const int q_heads  = query.ne[1];
    if (dim <= 0 || kv_heads <= 0 || blocks <= 0) { return; }

    const int threads = kScoresLanesPerHead * kv_heads;
    kv_block_scores_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads),
                             0, stream>>>(static_cast<const __nv_bfloat16*>(query.data),
                                          static_cast<const float*>(means.data),
                                          static_cast<float*>(scores.data), dim, kv_heads,
                                          q_heads, blocks);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail