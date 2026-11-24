// BF16 decode GEMV with runtime extents.
//
// The generic tile kernel below stages a 32-token activation tile, which is the right shape for
// prefill but leaves 31 of every 32 threads idle for a single token. Decode is the common case for
// this model, so it gets its own route: one warp per output row, 16 bytes per load, a warp
// reduction at the end.

#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/common/math.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kValuesPerLoad = 8;  // 16 bytes of BF16, one uint4

__global__ void bf16_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                 const __nv_bfloat16* __restrict__ weight,
                                 __nv_bfloat16* __restrict__ out, int n, int k) {
    const int warps_per_block = blockDim.x / 32;
    const int row = blockIdx.x * warps_per_block + threadIdx.x / 32;
    if (row >= n) { return; }
    const int lane = threadIdx.x % 32;
    const __nv_bfloat16* w = weight + std::size_t(row) * k;

    float accumulator = 0.0F;
    const int stride = 32 * kValuesPerLoad;
    int offset = lane * kValuesPerLoad;
    for (; offset + kValuesPerLoad <= k; offset += stride) {
        const auto packed = *reinterpret_cast<const uint4*>(w + offset);
        const auto* halves = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 w2 = __bfloat1622float2(halves[pair]);
            const float2 x2 = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(x + offset + pair * 2));
            accumulator += w2.x * x2.x + w2.y * x2.y;
        }
    }
    for (; offset < k; ++offset) {
        accumulator += __bfloat162float(w[offset]) * __bfloat162float(x[offset]);
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        accumulator += __shfl_xor_sync(0xffffffffu, accumulator, delta);
    }
    if (lane == 0) { out[row] = __float2bfloat16(accumulator); }
}

} // namespace

void launch_bf16_generic_gemv(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
    const int n = weight.n;
    const int k = weight.k;
    if (n <= 0 || k <= 0) { return; }
    constexpr int kThreads = 256;
    const int warps = kThreads / 32;
    const auto blocks = static_cast<unsigned>((n + warps - 1) / warps);
    bf16_gemv_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.qdata), static_cast<__nv_bfloat16*>(out.data),
        n, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
