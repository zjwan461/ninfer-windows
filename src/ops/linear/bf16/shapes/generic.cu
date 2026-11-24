// Generic BF16 x BF16 GEMM with runtime extents.
//
// Every tuned route in this directory is templated on a compile-time geometry and covers exactly
// one (n, k). A model whose widths are not in that table still has to run, so this route computes
// y[n, t] = W[n, k] . x[k, t] for any extents. It is deliberately simple: one output element per
// thread, the activation tile staged in shared memory, the weight row walked once. That makes it
// several times slower than a tuned route, which is the right trade for shapes that exist only to
// check correctness before they earn a kernel.

#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/common/math.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kTileRows   = 8;  // output rows per block
constexpr int kTileTokens = 32; // tokens per block
constexpr int kChunk      = 64; // k elements staged per iteration

__global__ void bf16_generic_kernel(const __nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ weight,
                                    __nv_bfloat16* __restrict__ out, int n, int k, int tokens) {
    __shared__ __nv_bfloat16 tile[kChunk][kTileTokens + 1];

    const int row_base = blockIdx.y * kTileRows;
    const int token_base = blockIdx.x * kTileTokens;
    const int row = row_base + threadIdx.x / kTileTokens;
    const int token = token_base + threadIdx.x % kTileTokens;
    const int lane_token = threadIdx.x % kTileTokens;

    float accumulator = 0.0F;
    for (int offset = 0; offset < k; offset += kChunk) {
        const int remaining = k - offset < kChunk ? k - offset : kChunk;
        for (int i = threadIdx.x; i < remaining * kTileTokens; i += blockDim.x) {
            const int kk = i / kTileTokens;
            const int tt = token_base + i % kTileTokens;
            tile[kk][i % kTileTokens] =
                (offset + kk < k && tt < tokens) ? x[std::size_t(offset + kk) * tokens + tt]
                                                 : __float2bfloat16(0.0F);
        }
        __syncthreads();
        if (row < n && token < tokens) {
            const __nv_bfloat16* w = weight + std::size_t(row) * k + offset;
            for (int kk = 0; kk < remaining; ++kk) {
                accumulator += __bfloat162float(w[kk]) * __bfloat162float(tile[kk][lane_token]);
            }
        }
        __syncthreads();
    }
    if (row < n && token < tokens) {
        out[std::size_t(row) * tokens + token] = __float2bfloat16(accumulator);
    }
}

} // namespace

void launch_bf16_generic(const Tensor& x, const Weight& weight, Tensor& out,
                         cudaStream_t stream) {
    const int k = weight.k;
    const int n = weight.n;
    const int tokens = x.ne[1];
    if (k <= 0 || n <= 0 || tokens <= 0) { return; }
    constexpr int kThreads = kTileRows * kTileTokens; // 256
    const dim3 grid(static_cast<unsigned>((tokens + kTileTokens - 1) / kTileTokens),
                    static_cast<unsigned>((n + kTileRows - 1) / kTileRows));
    bf16_generic_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<__nv_bfloat16*>(out.data), n, k, tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
