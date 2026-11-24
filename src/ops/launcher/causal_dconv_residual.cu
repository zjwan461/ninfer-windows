// ninfer::ops - Inkling 2 static short convolution with residual and 1/sqrt(5) scaling.
#include "ops/launcher/causal_dconv_residual.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kKernel = 4;
constexpr int kState = kKernel - 1;

// One thread per output element: the tensors are narrow (channels x a few tokens) and the op is
// bandwidth bound, so a flat mapping keeps the state prefix and the residual trivially ordered.
__global__ void causal_dconv_residual_kernel(const __nv_bfloat16* __restrict__ x,
                                             const __nv_bfloat16* __restrict__ weight,
                                             const __nv_bfloat16* __restrict__ state_in,
                                             __nv_bfloat16* __restrict__ state_out,
                                             __nv_bfloat16* __restrict__ out, int channels,
                                             int tokens) {
    const int channel = blockIdx.y;
    if (channel >= channels) { return; }
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens) { return; }

    const __nv_bfloat16* w = weight + std::size_t(channel) * kKernel;
    const __nv_bfloat16* s = state_in + std::size_t(channel) * kState;
    const std::size_t base = std::size_t(channel) * tokens;

    float accumulator = 0.0F;
#pragma unroll
    for (int j = 0; j < kKernel; ++j) {
        const int offset = token - (kKernel - 1) + j; // tap j reads `offset` steps back from token
        float sample = 0.0F;
        if (offset >= 0) {
            sample = __bfloat162float(x[base + offset]);
        } else {
            const int state_index = kState + offset; // offset -3 -> state 0
            if (state_index >= 0) { sample = __bfloat162float(s[state_index]); }
        }
        accumulator += __bfloat162float(w[j]) * sample;
    }
    out[base + token] = __float2bfloat16((accumulator + __bfloat162float(x[base + token])) *
                                         (1.0F / 2.2360679F)); // 1/sqrt(5)
}

__global__ void causal_dconv_state_kernel(const __nv_bfloat16* __restrict__ x,
                                          const __nv_bfloat16* __restrict__ state_in,
                                          __nv_bfloat16* __restrict__ state_out, int channels,
                                          int tokens) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) { return; }
    const std::size_t base = std::size_t(channel) * tokens;
    __nv_bfloat16 window[kState];
#pragma unroll
    for (int i = 0; i < kState; ++i) { window[i] = state_in[std::size_t(channel) * kState + i]; }
    for (int t = 0; t < tokens; ++t) {
#pragma unroll
        for (int i = 0; i + 1 < kState; ++i) { window[i] = window[i + 1]; }
        window[kState - 1] = x[base + t];
    }
#pragma unroll
    for (int i = 0; i < kState; ++i) { state_out[std::size_t(channel) * kState + i] = window[i]; }
}

} // namespace

void causal_dconv_residual_launch(const Tensor& x, const Tensor& weight,
                                  const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                                  cudaStream_t stream) {
    const int channels = x.ne[0];
    const int tokens = x.ne[1];
    if (channels <= 0 || tokens <= 0) { return; }

    const dim3 grid(static_cast<unsigned>((tokens + 127) / 128), static_cast<unsigned>(channels));
    causal_dconv_residual_kernel<<<grid, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), static_cast<__nv_bfloat16*>(out.data),
        channels, tokens);

    const auto state_blocks = static_cast<unsigned>((channels + 255) / 256);
    causal_dconv_state_kernel<<<state_blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), channels, tokens);
}

} // namespace ninfer::ops::detail
