// Generic q4_g64_fp16 x BF16 decode GEMV with runtime extents.
//
// The tuned routes cover the shapes this repository's other models use; Inkling 2's 2304x2304
// experts are not among them, and a 16 GB card needs those experts below eight bits. Storage is
// the row-split layout: k/2 packed bytes per row (low nibble = even column) then one fp16 scale
// per 64 values.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/math.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kGroup = 64;

__device__ __forceinline__ float q4_value(std::uint8_t packed, int parity) {
    const int nibble = parity == 0 ? (packed & 0x0F) : (packed >> 4);
    // The row-split codec stores values in two's complement nibbles.
    const int signed_value = (nibble & 0x08) != 0 ? nibble - 16 : nibble;
    return float(signed_value);
}

__global__ void q4_generic_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                       const std::uint8_t* __restrict__ codes,
                                       const std::uint16_t* __restrict__ scales,
                                       __nv_bfloat16* __restrict__ out, int rows, int k) {
    const int warps_per_block = blockDim.x / 32;
    const int row = blockIdx.x * warps_per_block + threadIdx.x / 32;
    if (row >= rows) { return; }
    const int lane = threadIdx.x % 32;
    const int groups = k / kGroup;
    const std::uint8_t* code_row = codes + std::int64_t(row) * (k / 2);
    const std::uint16_t* scale_row = scales + std::int64_t(row) * groups;

    float accumulator = 0.0F;
    for (int group = lane; group < groups; group += 32) {
        const float scale = __half2float(__ushort_as_half(scale_row[group]));
        float partial = 0.0F;
        for (int i = 0; i < kGroup; i += 2) {
            const int index = group * kGroup + i;
            const std::uint8_t packed = code_row[index / 2];
            partial += q4_value(packed, 0) * __bfloat162float(x[index]);
            partial += q4_value(packed, 1) * __bfloat162float(x[index + 1]);
        }
        accumulator += scale * partial;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor_sync(0xffffffffu, accumulator, offset);
    }
    if (lane == 0) { out[row] = __float2bfloat16(accumulator); }
}

} // namespace

void launch_q4_generic_gemv(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
    const int rows = weight.n;
    const int k = weight.k;
    if (rows <= 0 || k <= 0 || (k % kGroup) != 0) { return; }
    constexpr int kThreads = 256;
    const int warps = kThreads / 32;
    const auto blocks = static_cast<unsigned>((rows + warps - 1) / warps);
    q4_generic_gemv_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint16_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        rows, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
