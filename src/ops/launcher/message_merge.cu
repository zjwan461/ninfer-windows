// ninfer::ops - Inkling 2 cross-pass message merge.
#include "ops/launcher/message_merge.h"

#include <cuda_bf16.h>

#include <cstddef>

namespace ninfer::ops::detail {
namespace {

__global__ void message_merge_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ message,
                                     const __nv_bfloat16* __restrict__ forget,
                                     const __nv_bfloat16* __restrict__ gain,
                                     __nv_bfloat16* __restrict__ out, std::int64_t count) {
    const std::int64_t index = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) { return; }
    const float keep = 1.0F - tanhf(__bfloat162float(forget[index]));
    const float add = tanhf(__bfloat162float(gain[index]));
    out[index] = __float2bfloat16(keep * __bfloat162float(x[index]) +
                                  add * __bfloat162float(message[index]));
}

} // namespace

void message_merge_launch(const Tensor& x, const Tensor& message, const Tensor& forget,
                          const Tensor& gain, Tensor& out, cudaStream_t stream) {
    const std::int64_t count = out.numel();
    if (count <= 0) { return; }
    const auto blocks = static_cast<unsigned>((count + 255) / 256);
    message_merge_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(message.data),
        static_cast<const __nv_bfloat16*>(forget.data), static_cast<const __nv_bfloat16*>(gain.data),
        static_cast<__nv_bfloat16*>(out.data), count);
}

} // namespace ninfer::ops::detail
