// Two RMSNorm passes into one contiguous buffer, which is what a controller gate needs.
//
// Each message gate normalises two streams and then projects the pair side by side. Done as two calls
// the first result has to be written where the projection expects it anyway; done as one, both halves
// of that buffer are filled by one launch.

#include "core/device.h"
#include "core/tensor.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

__device__ __forceinline__ float dual_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void rmsnorm_dual_kernel(const __nv_bfloat16* __restrict__ first,
                                    const __nv_bfloat16* __restrict__ second,
                                    __nv_bfloat16* __restrict__ out, int width, float eps) {
    __shared__ float warp_sums[32];
    // Block index below `width` handles the first stream, above it the second.
    const int rows = gridDim.x / 2;
    const bool is_second = blockIdx.x >= rows;
    const int row = is_second ? blockIdx.x - rows : blockIdx.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const __nv_bfloat16* source = is_second ? second : first;
    const __nv_bfloat16* x_row = source + std::size_t(row) * width;
    __nv_bfloat16* o_row =
        out + (std::size_t(is_second ? rows * width : 0) + std::size_t(row) * width);

    float local = 0.0F;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float value = __bfloat162float(x_row[i]);
        local += value * value;
    }
    local = dual_sum(local);
    if (lane == 0) { warp_sums[warp] = local; }
    __syncthreads();

    const int warps = blockDim.x / 32;
    float total = (threadIdx.x < warps) ? warp_sums[threadIdx.x] : 0.0F;
    if (warp == 0) { total = dual_sum(total); }
    __shared__ float inverse;
    if (threadIdx.x == 0) { inverse = rsqrtf(total / float(width) + eps); }
    __syncthreads();

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        o_row[i] = __float2bfloat16(__bfloat162float(x_row[i]) * inverse);
    }
}

} // namespace

void rmsnorm_dual_launch(const Tensor& first, const Tensor& second, Tensor& out, float eps,
                         cudaStream_t stream) {
    const int width = out.ne[0] / 2;
    const int rows = first.ne[1] > 0 ? first.ne[1] : 1;
    if (width <= 0) { return; }
    constexpr int kThreads = 256;
    rmsnorm_dual_kernel<<<2 * rows, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(first.data),
        static_cast<const __nv_bfloat16*>(second.data), static_cast<__nv_bfloat16*>(out.data), width,
        eps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
