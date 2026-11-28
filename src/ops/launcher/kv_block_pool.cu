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

namespace {

constexpr int kAccumulateThreads = 256;

// One launch adds one chunk of history to the running per-block sums. Columns of the chunk land in
// whichever block their absolute position falls into, so a chunk may span blocks and need not be
// block aligned; only the caller decides when a block is complete.
__global__ void kv_block_accumulate_kernel(const __nv_bfloat16* __restrict__ keys,
                                            float* __restrict__ sums, int dim, int heads, int tokens,
                                            int blocks, int block_tokens, int first_position) {
    const int block       = static_cast<int>(blockIdx.x);
    const int head        = static_cast<int>(blockIdx.y);
    const int block_first = block * block_tokens;
    const int lo          = max(first_position, block_first);
    const int hi          = min(first_position + tokens, block_first + block_tokens);
    if (hi <= lo) { return; }

    const std::size_t head_base = static_cast<std::size_t>(head) * dim;
    for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
        float acc = 0.0F;
        for (int position = lo; position < hi; ++position) {
            const int column = position - first_position;
            acc += __bfloat162float(keys[(head_base + static_cast<std::size_t>(d)) * tokens + column]);
        }
        sums[(head_base + static_cast<std::size_t>(d)) * blocks + block] += acc;
    }
}

// The exact column count per block, incremented once per touched column (not once per thread).
__global__ void kv_block_counts_kernel(int* __restrict__ counts, int tokens, int blocks,
                                       int block_tokens, int first_position) {
    const int block = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                      static_cast<int>(threadIdx.x);
    if (block >= blocks) { return; }
    const int lo = max(first_position, block * block_tokens);
    const int hi = min(first_position + tokens, (block + 1) * block_tokens);
    if (hi > lo) { counts[block] += hi - lo; }
}

__global__ void kv_block_finalize_kernel(const float* __restrict__ sums,
                                          const int* __restrict__ counts, float* __restrict__ means,
                                          int total, int blocks) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                      static_cast<int>(threadIdx.x);
    if (index >= total) { return; }
    const int count = counts[index % blocks];
    means[index]    = count > 0 ? sums[index] / static_cast<float>(count) : 0.0F;
}

} // namespace

void kv_block_accumulate_launch(const Tensor& keys, std::int32_t block_tokens,
                                std::int32_t first_position, Tensor& sums, Tensor& counts,
                                cudaStream_t stream) {
    const int dim      = keys.ne[0];
    const int heads    = keys.ne[1];
    const int tokens   = keys.ne[2];
    const int blocks   = sums.ne[2];
    if (dim <= 0 || heads <= 0 || tokens <= 0 || blocks <= 0) { return; }

    const dim3 grid(static_cast<unsigned int>(blocks), static_cast<unsigned int>(heads));
    const int  threads = dim < kAccumulateThreads ? dim : kAccumulateThreads;
    kv_block_accumulate_kernel<<<grid, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(keys.data), static_cast<float*>(sums.data), dim, heads,
        tokens, blocks, block_tokens, first_position);
    CUDA_CHECK(cudaGetLastError());

    const int count_threads = block_tokens < kAccumulateThreads ? block_tokens : kAccumulateThreads;
    const int count_blocks  = (blocks + count_threads - 1) / count_threads;
    kv_block_counts_kernel<<<static_cast<unsigned int>(count_blocks),
                             static_cast<unsigned int>(count_threads), 0, stream>>>(
        static_cast<int*>(counts.data), tokens, blocks, block_tokens, first_position);
    CUDA_CHECK(cudaGetLastError());
}

void kv_block_finalize_launch(const Tensor& sums, const Tensor& counts, Tensor& means,
                              cudaStream_t stream) {
    const int blocks = sums.ne[2];
    const int total  = sums.ne[0] * sums.ne[1] * blocks;
    if (total <= 0 || blocks <= 0) { return; }

    const int threads = total < kAccumulateThreads ? total : kAccumulateThreads;
    const int grid     = (total + threads - 1) / threads;
    kv_block_finalize_kernel<<<static_cast<unsigned int>(grid), static_cast<unsigned int>(threads),
                               0, stream>>>(static_cast<const float*>(sums.data),
                                            static_cast<const int*>(counts.data),
                                            static_cast<float*>(means.data), total, blocks);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail