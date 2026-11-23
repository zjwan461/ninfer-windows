// ninfer::ops - Inkling 2 scaled L2 clip (query and relative-state normalisation).
#include "ops/launcher/scaled_l2_clip.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kWarp = 32;

__device__ __forceinline__ float warp_sum_scaled(float value) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// One warp per (row, head) pair; each lane owns four head dimensions, so head_dim 128 fits.
__global__ void scaled_l2_clip_kernel(const __nv_bfloat16* __restrict__ x,
                                      const __nv_bfloat16* __restrict__ scale,
                                      const int* __restrict__ positions,
                                      __nv_bfloat16* __restrict__ out, int rows, int heads,
                                      int head_dim, float threshold, float tau_alpha,
                                      float tau_floor) {
    const int warp = blockIdx.x * (blockDim.x / kWarp) + (threadIdx.x / kWarp);
    const int total = rows * heads;
    if (warp >= total) { return; }
    const int lane = threadIdx.x % kWarp;
    const int row = warp / heads;
    const int head = warp % heads;
    const int dims = (head_dim + kWarp - 1) / kWarp; // 4 for 128, 1 for 16

    const std::size_t base = std::size_t(row) * heads * head_dim + std::size_t(head) * head_dim;
    const __nv_bfloat16* x_row = x + base + lane * dims;
    const __nv_bfloat16* s_row = scale + std::size_t(head) * head_dim + lane * dims;
    __nv_bfloat16* o_row = out + base + lane * dims;

    float value[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    float square_sum = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (i < dims && lane * dims + i < head_dim) {
            value[i] = __bfloat162float(x_row[i]) * __bfloat162float(s_row[i]);
            square_sum += value[i] * value[i];
        }
    }
    const float norm = sqrtf(warp_sum_scaled(square_sum));
    float gain = threshold / (norm + 1e-6F);
    gain = gain > 1.0F ? 1.0F : gain;

    if (tau_alpha > 0.0F) {
        const float position = float(positions[row]) + 1.0F;
        const float ratio = position / tau_floor;
        gain *= 1.0F + tau_alpha * logf(ratio > 1.0F ? ratio : 1.0F);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (i < dims && lane * dims + i < head_dim) {
            o_row[i] = __float2bfloat16(value[i] * gain);
        }
    }
}

} // namespace

void scaled_l2_clip_launch(const Tensor& x, const Tensor& scale, const Tensor* positions,
                           const ScaledL2ClipArgs& args, Tensor& out, cudaStream_t stream) {
    if (args.head_dim <= 0 || args.head_dim > 128) {
        throw std::invalid_argument("scaled_l2_clip: head_dim must be in (0, 128]");
    }
    if (args.rows <= 0 || args.heads <= 0) { return; }
    const int total = args.rows * args.heads;
    constexpr int kBlock = 128; // four warps, four heads per block
    const auto blocks = static_cast<unsigned int>((total + 3) / 4);
    const int* position_data = positions == nullptr ? nullptr : static_cast<const int*>(positions->data);
    scaled_l2_clip_kernel<<<blocks, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(scale.data),
        position_data, static_cast<__nv_bfloat16*>(out.data), args.rows, args.heads,
        args.head_dim, args.threshold, args.tau_alpha, args.tau_floor);
}

} // namespace ninfer::ops::detail
