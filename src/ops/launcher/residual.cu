// ninfer::ops - residual-stream helpers that keep the FP32 accumulator the reference uses.
#include "ops/launcher/residual.h"

#include <cuda_bf16.h>

#include <cstddef>

namespace ninfer::ops::detail {
namespace {

__global__ void cast_bf16_to_f32_kernel(const __nv_bfloat16* __restrict__ in,
                                        float* __restrict__ out, std::int64_t count) {
    const std::int64_t index = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) { out[index] = __bfloat162float(in[index]); }
}

__global__ void cast_f32_to_bf16_kernel(const float* __restrict__ in,
                                        __nv_bfloat16* __restrict__ out, std::int64_t count) {
    const std::int64_t index = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) { out[index] = __float2bfloat16(in[index]); }
}

__global__ void residual_add_f32_kernel(const float* __restrict__ base,
                                        const __nv_bfloat16* __restrict__ addend,
                                        float* __restrict__ out, std::int64_t count) {
    const std::int64_t index = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) { out[index] = base[index] + __bfloat162float(addend[index]); }
}

__global__ void residual_store_bf16_kernel(const float* __restrict__ base,
                                           const __nv_bfloat16* __restrict__ addend,
                                           __nv_bfloat16* __restrict__ out, std::int64_t count) {
    const std::int64_t index = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = __float2bfloat16(base[index] + __bfloat162float(addend[index]));
    }
}

unsigned blocks(std::int64_t count) { return static_cast<unsigned>((count + 255) / 256); }

} // namespace

void cast_bf16_to_f32_launch(Tensor& out, const Tensor& in, cudaStream_t stream) {
    const std::int64_t count = out.numel();
    if (count <= 0) { return; }
    cast_bf16_to_f32_kernel<<<blocks(count), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(in.data), static_cast<float*>(out.data), count);
}

void cast_f32_to_bf16_launch(Tensor& out, const Tensor& in, cudaStream_t stream) {
    const std::int64_t count = out.numel();
    if (count <= 0) { return; }
    cast_f32_to_bf16_kernel<<<blocks(count), 256, 0, stream>>>(
        static_cast<const float*>(in.data), static_cast<__nv_bfloat16*>(out.data), count);
}

void residual_add_f32_launch(Tensor& out, const Tensor& base, const Tensor& addend,
                             cudaStream_t stream) {
    const std::int64_t count = out.numel();
    if (count <= 0) { return; }
    residual_add_f32_kernel<<<blocks(count), 256, 0, stream>>>(
        static_cast<const float*>(base.data), static_cast<const __nv_bfloat16*>(addend.data),
        static_cast<float*>(out.data), count);
}

void residual_store_bf16_launch(Tensor& out, const Tensor& base, const Tensor& addend,
                                cudaStream_t stream) {
    const std::int64_t count = out.numel();
    if (count <= 0) { return; }
    residual_store_bf16_kernel<<<blocks(count), 256, 0, stream>>>(
        static_cast<const float*>(base.data), static_cast<const __nv_bfloat16*>(addend.data),
        static_cast<__nv_bfloat16*>(out.data), count);
}

} // namespace ninfer::ops::detail
