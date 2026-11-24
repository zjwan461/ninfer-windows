// Generic NVFP4 x BF16 decode GEMV with runtime extents.
//
// The tuned NVFP4 routes cover five (n, k) pairs. A model whose widths are not in that table still
// has to run, so this route computes y[n] = W[n, k] . x[k] / weight_scale_divisor for any extents,
// one warp per output row.
//
// Storage follows the repository container (validate_nvfp4_weight): a code plane of n*k/2 bytes
// holding row-major packed E2M1 pairs, then a scale plane at align_up(n*k/2, 256) with one E4M3
// scale per 16 values, swizzled BlockScaleK16M128x4, then the FP32 divisor.

#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/math.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

__device__ __forceinline__ std::int64_t generic_scale_offset(int row, int group, int groups_per_row) {
    const int m_tile = row / 128;
    const int row_inner = row - m_tile * 128;
    const int scale_tile = group / 4;
    const int scale_lane = group & 3;
    const int tiles_per_row = groups_per_row / 4;
    return (std::int64_t(m_tile) * tiles_per_row + scale_tile) * 512 + (row_inner & 31) * 16 +
           (row_inner >> 5) * 4 + scale_lane;
}

__global__ void nvfp4_generic_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                          const std::uint8_t* __restrict__ codes,
                                          const std::uint8_t* __restrict__ scales,
                                          __nv_bfloat16* __restrict__ out, int rows, int k,
                                          float divisor) {
    const int warps_per_block = blockDim.x / 32;
    const int row = blockIdx.x * warps_per_block + threadIdx.x / 32;
    if (row >= rows) { return; }
    const int lane = threadIdx.x % 32;
    const int groups = k / 16;
    const std::int64_t code_base = std::int64_t(row) * (k / 2);

    float accumulator = 0.0F;
    for (int group = lane; group < groups; group += 32) {
        const std::uint32_t* words =
            reinterpret_cast<const std::uint32_t*>(codes + code_base + group * 8);
        const std::uint32_t lo = words[0];
        const std::uint32_t hi = words[1];
        const float scale = decode_nvfp4_e4m3(scales[generic_scale_offset(row, group, groups)]);
        const int base = group * 16;
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 values = decode_nvfp4_e2m1x2(std::uint8_t(lo >> (8 * pair)));
            accumulator += scale * (values.x * __bfloat162float(x[base + 2 * pair]) +
                                    values.y * __bfloat162float(x[base + 2 * pair + 1]));
        }
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 values = decode_nvfp4_e2m1x2(std::uint8_t(hi >> (8 * pair)));
            accumulator += scale * (values.x * __bfloat162float(x[base + 8 + 2 * pair]) +
                                    values.y * __bfloat162float(x[base + 8 + 2 * pair + 1]));
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor_sync(0xffffffffu, accumulator, offset);
    }
    if (lane == 0) { out[row] = __float2bfloat16(accumulator / divisor); }
}

// The same arithmetic with the weight read from a device-side table by slot, so the choice of expert
// never has to reach the host. One thread per row keeps it simple: it is the reference shape, and the
// point of the route is that a captured graph can contain it.
__global__ void nvfp4_gather_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                         const Weight* __restrict__ table, int slot,
                                         __nv_bfloat16* __restrict__ out, int rows, int k,
                                         float divisor) {
    const int warps_per_block = blockDim.x / 32;
    const int row = blockIdx.x * warps_per_block + threadIdx.x / 32;
    if (row >= rows) { return; }
    const Weight weight = table[slot];
    const auto* codes = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales);
    const int lane = threadIdx.x % 32;
    const int groups = k / 16;
    const std::int64_t code_base = std::int64_t(row) * (k / 2);

    float accumulator = 0.0F;
    for (int group = lane; group < groups; group += 32) {
        const std::uint32_t* words =
            reinterpret_cast<const std::uint32_t*>(codes + code_base + group * 8);
        const std::uint32_t lo = words[0];
        const std::uint32_t hi = words[1];
        const float scale = decode_nvfp4_e4m3(scales[generic_scale_offset(row, group, groups)]);
        const int base = group * 16;
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 values = decode_nvfp4_e2m1x2(std::uint8_t(lo >> (8 * pair)));
            accumulator += scale * (values.x * __bfloat162float(x[base + 2 * pair]) +
                                    values.y * __bfloat162float(x[base + 2 * pair + 1]));
        }
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 values = decode_nvfp4_e2m1x2(std::uint8_t(hi >> (8 * pair)));
            accumulator += scale * (values.x * __bfloat162float(x[base + 8 + 2 * pair]) +
                                    values.y * __bfloat162float(x[base + 8 + 2 * pair + 1]));
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor_sync(0xffffffffu, accumulator, offset);
    }
    if (lane == 0) { out[row] = __float2bfloat16(accumulator / divisor); }
}

} // namespace

// `extents` is a host copy of the same entry: the table lives on the device, so the launcher cannot
// read it, and the only thing the kernel needs from it is the pair of data pointers.
void launch_nvfp4_gather_gemv(const Tensor& x, const Weight* table, int slot, const Weight& extents,
                              Tensor& out, cudaStream_t stream) {
    const int rows = extents.n;
    const int k = extents.k;
    if (rows <= 0 || k <= 0) { return; }
    constexpr int kThreads = 256;
    const int warps = kThreads / 32;
    const auto blocks = static_cast<unsigned>((rows + warps - 1) / warps);
    nvfp4_gather_gemv_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), table, slot,
        static_cast<__nv_bfloat16*>(out.data), rows, k, extents.weight_scale_divisor);
}

void launch_nvfp4_generic_gemv(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream) {
    const int rows = weight.n;
    const int k = weight.k;
    if (rows <= 0 || k <= 0) { return; }
    const auto* codes = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales);
    const float divisor = weight.weight_scale_divisor;
    constexpr int kThreads = 256;
    const int warps = kThreads / 32;
    const auto blocks = static_cast<unsigned>((rows + warps - 1) / warps);
    nvfp4_generic_gemv_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), codes, scales,
        static_cast<__nv_bfloat16*>(out.data), rows, k, divisor);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
