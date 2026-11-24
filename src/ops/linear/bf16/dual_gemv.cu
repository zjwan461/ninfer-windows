// Two BF16 projections from one input in a single launch.
//
// The key and value projections read the same normalised stream and write the same number of rows,
// so a grid over both is one launch instead of two, with no arithmetic change.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/op_counters.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

__device__ __forceinline__ float dot_bf16(const __nv_bfloat16* __restrict__ x,
                                          const __nv_bfloat16* __restrict__ row, int k, int lane,
                                          int stride) {
    float accumulator = 0.0F;
    for (int index = lane; index < k; index += stride) {
        accumulator += __bfloat162float(row[index]) * __bfloat162float(x[index]);
    }
    return accumulator;
}

__global__ void bf16_dual_gemv_kernel(const __nv_bfloat16* __restrict__ x_a,
                                      const __nv_bfloat16* __restrict__ x_b,
                                      const __nv_bfloat16* __restrict__ matrix_a,
                                      const __nv_bfloat16* __restrict__ matrix_b,
                                      __nv_bfloat16* __restrict__ out_a,
                                      __nv_bfloat16* __restrict__ out_b, int rows_a, int rows_b,
                                      int k) {
    // One grid over both outputs, so the two projections can have different heights.
    const int warps_per_block = blockDim.x / 32;
    const int index = blockIdx.x * warps_per_block + threadIdx.x / 32;
    if (index >= rows_a + rows_b) { return; }
    const int lane = threadIdx.x % 32;
    const bool first = index < rows_a;
    const int row = first ? index : index - rows_a;
    const __nv_bfloat16* x = first ? x_a : x_b;
    const __nv_bfloat16* matrix = first ? matrix_a : matrix_b;

    float accumulator = dot_bf16(x, matrix + std::int64_t(row) * k, k, lane, 32);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor_sync(0xffffffffu, accumulator, offset);
    }
    if (lane == 0) {
        if (first) {
            out_a[row] = __float2bfloat16(accumulator);
        } else {
            out_b[row] = __float2bfloat16(accumulator);
        }
    }
}

} // namespace

void launch_bf16_dual_gemv(const Tensor& x_a, const Tensor& x_b, const Weight& weight_a,
                           const Weight& weight_b,
                           Tensor& out_a, Tensor& out_b, cudaStream_t stream) {
    ninfer::ops::op_counters().linear.fetch_add(1, std::memory_order_relaxed);
    const int rows_a = weight_a.n;
    const int rows_b = weight_b.n;
    const int k = weight_a.k;
    if (rows_a <= 0 || rows_b <= 0 || k <= 0) { return; }
    constexpr int kThreads = 256;
    const int warps = kThreads / 32;
    const auto blocks = static_cast<unsigned>((rows_a + rows_b + warps - 1) / warps);
    bf16_dual_gemv_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x_a.data), static_cast<const __nv_bfloat16*>(x_b.data),
        static_cast<const __nv_bfloat16*>(weight_a.qdata),
        static_cast<const __nv_bfloat16*>(weight_b.qdata),
        static_cast<__nv_bfloat16*>(out_a.data), static_cast<__nv_bfloat16*>(out_b.data), rows_a,
        rows_b, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
