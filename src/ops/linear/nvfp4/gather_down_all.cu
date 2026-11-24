// The down projections of all routed experts, summed in one launch.
//
// A warp owns one output row and walks the experts in slot order, accumulating in the same order the
// separate launches did, so the sum is bit-identical rather than merely close. Because every row is
// written once, the accumulator needs no clearing either.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/math.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/op_counters.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

__device__ __forceinline__ std::int64_t down_scale_offset(int row, int group, int groups_per_row) {
    constexpr int kTileRows = 128;
    constexpr int kTileGroups = 4;
    const int m_tile = row / kTileRows;
    const int row_inner = row % kTileRows;
    const int tiles_per_row = (groups_per_row + kTileGroups - 1) / kTileGroups;
    const int scale_tile = group / kTileGroups;
    const int scale_lane = group % kTileGroups;
    return (std::int64_t(m_tile) * tiles_per_row + scale_tile) * 512 + (row_inner & 31) * 16 +
           (row_inner >> 5) * 4 + scale_lane;
}

__device__ __forceinline__ float down_row_dot(const std::uint8_t* codes,
                                              const std::uint8_t* scales,
                                              const __nv_bfloat16* x, int row, int k, int lane) {
    const int groups = k / 16;
    const std::int64_t code_base = std::int64_t(row) * (k / 2);
    float accumulator = 0.0F;
    for (int group = lane; group < groups; group += 32) {
        const std::uint32_t* words =
            reinterpret_cast<const std::uint32_t*>(codes + code_base + group * 8);
        const std::uint32_t lo = words[0];
        const std::uint32_t hi = words[1];
        const float scale = decode_nvfp4_e4m3(scales[down_scale_offset(row, group, groups)]);
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
    return accumulator;
}

__global__ void nvfp4_gather_down_all_kernel(const __nv_bfloat16* __restrict__ acts,
                                             const Weight* __restrict__ table,
                                             const std::int32_t* __restrict__ slots, int top_k,
                                             const float* __restrict__ scales,
                                             float* __restrict__ acc, int rows, int k, int inter) {
    const int warps_per_block = blockDim.x / 32;
    const int row = blockIdx.x * warps_per_block + threadIdx.x / 32;
    if (row >= rows) { return; }
    const int lane = threadIdx.x % 32;

    float total = 0.0F;
    for (int slot = 0; slot < top_k; ++slot) {
        const Weight weight = table[slots[slot]];
        const auto* codes = static_cast<const std::uint8_t*>(weight.qdata);
        const auto* block_scales = static_cast<const std::uint8_t*>(weight.scales);
        const __nv_bfloat16* act = acts + std::size_t(slot) * inter;
        const float value = down_row_dot(codes, block_scales, act, row, k, lane) /
                            weight.weight_scale_divisor;
        total += scales[slot] * value;
    }
    if (lane == 0) { acc[row] = total; }
}

} // namespace

void launch_nvfp4_gather_down_all(const Tensor& acts, const Weight* table,
                                  const std::int32_t* slots, int top_k, const Weight& extents,
                                  const float* scales, Tensor& acc, cudaStream_t stream) {
    ninfer::ops::op_counters().expert_gemv.fetch_add(1, std::memory_order_relaxed);
    const int rows = extents.n;
    const int k = extents.k;
    const int inter = k;
    if (rows <= 0 || k <= 0 || top_k <= 0) { return; }
    constexpr int kThreads = 256;
    const int warps = kThreads / 32;
    const auto blocks = static_cast<unsigned>((rows + warps - 1) / warps);
    nvfp4_gather_down_all_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(acts.data), table, slots, top_k, scales,
        static_cast<float*>(acc.data), rows, k, inter);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
