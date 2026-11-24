// Three gained RMSNorm passes in one launch: the attention streams share a geometry and differ only in
// their input, their gain and their output.
//
// The three outputs are passed separately rather than as one buffer, because nothing here should depend
// on where the arena happened to put them.

#include "core/device.h"
#include "core/tensor.h"
#include "ops/op_counters.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

__device__ __forceinline__ float triple_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void rmsnorm_triple_kernel(const __nv_bfloat16* __restrict__ a,
                                      const __nv_bfloat16* __restrict__ b,
                                      const __nv_bfloat16* __restrict__ c,
                                      const __nv_bfloat16* __restrict__ gain_a,
                                      const __nv_bfloat16* __restrict__ gain_b,
                                      const __nv_bfloat16* __restrict__ gain_c,
                                      __nv_bfloat16* __restrict__ out_a,
                                      __nv_bfloat16* __restrict__ out_b,
                                      __nv_bfloat16* __restrict__ out_c, int width, float eps) {
    __shared__ float warp_sums[32];
    __shared__ float inverse;
    const int rows = gridDim.x / 3;
    const int stream = blockIdx.x / rows;
    const int row = blockIdx.x - stream * rows;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const __nv_bfloat16* source = stream == 0 ? a : (stream == 1 ? b : c);
    const __nv_bfloat16* gain = stream == 0 ? gain_a : (stream == 1 ? gain_b : gain_c);
    __nv_bfloat16* destination = stream == 0 ? out_a : (stream == 1 ? out_b : out_c);
    const __nv_bfloat16* x_row = source + std::size_t(row) * width;
    __nv_bfloat16* o_row = destination + std::size_t(row) * width;

    float local = 0.0F;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float value = __bfloat162float(x_row[i]);
        local += value * value;
    }
    local = triple_sum(local);
    if (lane == 0) { warp_sums[warp] = local; }
    __syncthreads();
    const int warps = blockDim.x / 32;
    float total = (threadIdx.x < warps) ? warp_sums[threadIdx.x] : 0.0F;
    if (warp == 0) { total = triple_sum(total); }
    if (threadIdx.x == 0) { inverse = rsqrtf(total / float(width) + eps); }
    __syncthreads();
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float scaled = __bfloat162float(x_row[i]) * inverse;
        o_row[i] = __float2bfloat16(scaled * __bfloat162float(gain[i]));
    }
}

} // namespace

void rmsnorm_triple_launch(const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& gain_a,
                           const Tensor& gain_b, const Tensor& gain_c, Tensor& out_a, Tensor& out_b,
                           Tensor& out_c, float eps, cudaStream_t stream) {
    ninfer::ops::op_counters().norm.fetch_add(1, std::memory_order_relaxed);
    const int width = a.ne[0];
    const int rows = a.ne[1] > 0 ? a.ne[1] : 1;
    if (width <= 0) { return; }
    constexpr int kThreads = 256;
    rmsnorm_triple_kernel<<<3 * rows, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(a.data), static_cast<const __nv_bfloat16*>(b.data),
        static_cast<const __nv_bfloat16*>(c.data), static_cast<const __nv_bfloat16*>(gain_a.data),
        static_cast<const __nv_bfloat16*>(gain_b.data),
        static_cast<const __nv_bfloat16*>(gain_c.data), static_cast<__nv_bfloat16*>(out_a.data),
        static_cast<__nv_bfloat16*>(out_b.data), static_cast<__nv_bfloat16*>(out_c.data), width,
        eps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
