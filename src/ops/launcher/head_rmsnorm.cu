// ninfer::ops - RMSNorm without a gain vector (Inkling 2 value projection).
#include "ops/launcher/head_rmsnorm.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// One block per row: a warp-level reduction over the row followed by a cross-warp reduction in
// shared memory. Rows are head width (128) in practice, where only four warps do any work.
__global__ void head_rmsnorm_kernel(const __nv_bfloat16* __restrict__ x,
                                    __nv_bfloat16* __restrict__ out, int width, float eps) {
    __shared__ float warp_sums[32];
    const int row = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const __nv_bfloat16* x_row = x + std::size_t(row) * width;
    __nv_bfloat16* o_row = out + std::size_t(row) * width;

    float local = 0.0F;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float value = __bfloat162float(x_row[i]);
        local += value * value;
    }
    local = warp_reduce_sum(local);
    if (lane == 0) { warp_sums[warp] = local; }
    __syncthreads();

    if (warp == 0) {
        const int warps = blockDim.x / 32;
        float total = lane < warps ? warp_sums[lane] : 0.0F;
        total = warp_reduce_sum(total);
        if (lane == 0) { warp_sums[0] = rsqrtf(total / float(width) + eps); }
    }
    __syncthreads();

    const float inverse = warp_sums[0];
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        o_row[i] = __float2bfloat16(__bfloat162float(x_row[i]) * inverse);
    }
}

} // namespace

void head_rmsnorm_launch(const Tensor& x, float eps, Tensor& out, cudaStream_t stream) {
    const int width = x.ne[0];
    const std::int64_t rows = x.numel() / width;
    if (width <= 0 || rows <= 0) { return; }
    const int block = width >= 512 ? 512 : (width >= 128 ? 128 : 32);
    head_rmsnorm_kernel<<<static_cast<unsigned>(rows), block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data), width,
        eps);
}

} // namespace ninfer::ops::detail
