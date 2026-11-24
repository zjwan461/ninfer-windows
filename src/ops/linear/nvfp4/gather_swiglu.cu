// Gate/up projection and its SwiGLU in one launch.
//
// The two halves of an expert's gate/up matrix feed a single output row each, so a warp that reads
// both halves produces one finished activation. Done as two operations it is a projection into a
// 2*inter buffer and then a separate pass over it; done as one it is half the launches for the same
// arithmetic and one buffer fewer.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/math.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/op_counters.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Same cap as the standalone op: tanh-bounded above, so the activation stays in range.
__device__ __forceinline__ float capped(float value, float cap) {
    return cap * tanhf(value / cap);
}

__device__ __forceinline__ std::int64_t swiglu_scale_offset(int row, int group, int groups_per_row) {
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

__device__ __forceinline__ float dot_row(const std::uint8_t* codes, const std::uint8_t* scales,
                                         const __nv_bfloat16* x, int row, int k, int lane) {
    const int groups = k / 16;
    const std::int64_t code_base = std::int64_t(row) * (k / 2);
    float accumulator = 0.0F;
    for (int group = lane; group < groups; group += 32) {
        const std::uint32_t* words =
            reinterpret_cast<const std::uint32_t*>(codes + code_base + group * 8);
        const std::uint32_t lo = words[0];
        const std::uint32_t hi = words[1];
        const float scale = decode_nvfp4_e4m3(scales[swiglu_scale_offset(row, group, groups)]);
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

__global__ void nvfp4_gather_swiglu_kernel(const __nv_bfloat16* __restrict__ x,
                                           const Weight* __restrict__ table,
                                           const std::int32_t* __restrict__ slots, int slot,
                                           __nv_bfloat16* __restrict__ out, int inter, int k,
                                           float gate_cap, float up_cap) {
    const int warps_per_block = blockDim.x / 32;
    const int row = blockIdx.x * warps_per_block + threadIdx.x / 32;
    if (row >= inter) { return; }
    const int lane = threadIdx.x % 32;
    const Weight weight = table[slots[slot]];
    const auto* codes = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales);
    const float divisor = weight.weight_scale_divisor;

    const float gate = dot_row(codes, scales, x, row, k, lane) / divisor;
    const float up = dot_row(codes, scales, x, inter + row, k, lane) / divisor;
    if (lane == 0) {
        const float value = capped(gate, gate_cap) * (1.0F / (1.0F + __expf(-gate))) *
                            capped(up, up_cap);
        out[row] = __float2bfloat16(value);
    }
}

} // namespace

void launch_nvfp4_gather_swiglu(const Tensor& x, const Weight* table, const std::int32_t* slots,
                                int slot, const Weight& extents, float gate_cap, float up_cap,
                                Tensor& out, cudaStream_t stream) {
    ninfer::ops::op_counters().expert_gemv.fetch_add(1, std::memory_order_relaxed);
    const int inter = extents.n / 2;
    const int k = extents.k;
    if (inter <= 0 || k <= 0) { return; }
    constexpr int kThreads = 256;
    const int warps = kThreads / 32;
    const auto blocks = static_cast<unsigned>((inter + warps - 1) / warps);
    nvfp4_gather_swiglu_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), table, slots, slot,
        static_cast<__nv_bfloat16*>(out.data), inter, k, gate_cap, up_cap);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
