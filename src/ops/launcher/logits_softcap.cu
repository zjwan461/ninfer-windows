// ninfer::ops - the tanh softcap a model applies to its logits before sampling.
#include "ops/launcher/logits_softcap.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

__global__ void logits_softcap_kernel(__nv_bfloat16* __restrict__ logits, std::int64_t count,
                                      float cap) {
    const std::int64_t index = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) { return; }
    const float value = __bfloat162float(logits[index]);
    logits[index] = __float2bfloat16(cap * tanhf(value / cap));
}

} // namespace

void logits_softcap_launch(Tensor& logits, float cap, cudaStream_t stream) {
    const std::int64_t count = logits.numel();
    if (count <= 0) { return; }
    const auto blocks = static_cast<unsigned>((count + 255) / 256);
    logits_softcap_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(logits.data), count, cap);
}

} // namespace ninfer::ops::detail
