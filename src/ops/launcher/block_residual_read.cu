// ninfer::ops - Inkling 2 block attention residual read.
#include "ops/launcher/block_residual_read.h"

#include <cuda_bf16.h>

#include <cstddef>

namespace ninfer::ops::detail {
namespace {

// Sums `value` across the block. Every thread in the block must call it; the trailing barrier
// keeps the next call from overwriting the scratch slot before everyone has read the result.
__device__ __forceinline__ float block_reduce_sum(float value, float* scratch) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int warps = blockDim.x / 32;
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0F;
        for (int i = 0; i < warps; ++i) { total += scratch[i]; }
        scratch[0] = total;
    }
    __syncthreads();
    const float total = scratch[0];
    __syncthreads();
    return total;
}

// One block per query row: normalize every state, score it against the query, then a softmax over
// the (small) state stack.
__global__ void block_residual_read_kernel(const float* __restrict__ query,
                                           const __nv_bfloat16* __restrict__ states,
                                           __nv_bfloat16* __restrict__ out, int queries, int count,
                                           int hidden, float eps, float scale) {
    __shared__ float scratch[32];
    __shared__ float logits[8];
    __shared__ float state_norm[8];

    const int q = blockIdx.x;
    if (q >= queries) { return; }
    const float* q_row = query + std::size_t(q) * hidden;

    // Normalise each state and score it.
    for (int s = 0; s < count; ++s) {
        const __nv_bfloat16* row = states + std::size_t(s) * hidden;
        float square_sum = 0.0F;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
            const float value = __bfloat162float(row[i]);
            square_sum += value * value;
        }
        square_sum = block_reduce_sum(square_sum, scratch);
        __syncthreads();
        const float inverse = rsqrtf(square_sum / float(hidden) + eps);
        float dot = 0.0F;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
            dot += q_row[i] * (__bfloat162float(row[i]) * inverse);
        }
        dot = block_reduce_sum(dot, scratch);
        __syncthreads();
        if (threadIdx.x == 0) {
            logits[s] = dot * scale;
            state_norm[s] = inverse;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float maximum = logits[0];
        for (int s = 1; s < count; ++s) { maximum = fmaxf(maximum, logits[s]); }
        float total = 0.0F;
        for (int s = 0; s < count; ++s) {
            logits[s] = __expf(logits[s] - maximum);
            total += logits[s];
        }
        for (int s = 0; s < count; ++s) { logits[s] /= total; }
    }
    __syncthreads();

    __nv_bfloat16* o_row = out + std::size_t(q) * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float accumulator = 0.0F;
        for (int s = 0; s < count; ++s) {
            const __nv_bfloat16* row = states + std::size_t(s) * hidden;
            accumulator += logits[s] * (__bfloat162float(row[i]) * state_norm[s]);
        }
        o_row[i] = __float2bfloat16(accumulator);
    }
}

} // namespace

void block_residual_read_launch(const Tensor& query, const Tensor& states, float eps, Tensor& out,
                                cudaStream_t stream) {
    const int queries = query.ne[0];
    const int hidden = query.ne[1];
    const int count = states.ne[0];
    if (queries <= 0 || hidden <= 0 || count <= 0) { return; }
    const float scale = 1.0F / sqrtf(float(hidden));
    block_residual_read_kernel<<<queries, 256, 0, stream>>>(
        static_cast<const float*>(query.data), static_cast<const __nv_bfloat16*>(states.data),
        static_cast<__nv_bfloat16*>(out.data), queries, count, hidden, eps, scale);
}

} // namespace ninfer::ops::detail
