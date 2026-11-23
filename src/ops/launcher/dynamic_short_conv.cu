// ninfer::ops - Inkling 2 dynamic short convolution (global-attention keys and values).
#include "ops/launcher/dynamic_short_conv.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kKernel = 4;
constexpr int kState = kKernel - 1;

__global__ void dynamic_short_conv_kernel(const __nv_bfloat16* __restrict__ x,
                                          const __nv_bfloat16* __restrict__ static_weight,
                                          const __nv_bfloat16* __restrict__ dynamic_weight,
                                          const __nv_bfloat16* __restrict__ state_in,
                                          __nv_bfloat16* __restrict__ state_out,
                                          __nv_bfloat16* __restrict__ out, int channels,
                                          int tokens, int head_width, int heads) {
    const int channel = blockIdx.y;
    if (channel >= channels) { return; }
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens) { return; }

    const int head = channel / head_width;
    const __nv_bfloat16* dynamic = dynamic_weight + std::size_t(token) * heads * kKernel +
                                   std::size_t(head) * kKernel;
    const __nv_bfloat16* state = state_in + std::size_t(channel) * kState;
    const std::size_t base = std::size_t(channel) * tokens;

    float accumulator = 0.0F;
#pragma unroll
    for (int lag = 0; lag < kKernel; ++lag) {
        // The tap index is the backward distance: tap 0 weights the current token, tap 3 the token
        // three steps back. The static convolution in this model uses the opposite (nn.Conv1d)
        // order, which is why its current-token tap is the last one.
        const int offset = token - lag;
        float sample = 0.0F;
        if (offset >= 0) {
            sample = __bfloat162float(x[base + offset]);
        } else {
            const int state_index = kState + offset;
            if (state_index >= 0) { sample = __bfloat162float(state[state_index]); }
        }
        const float tap = __bfloat162float(static_weight[std::size_t(lag) * channels + channel]) +
                          __bfloat162float(dynamic[lag]);
        accumulator += tap * sample;
    }
    out[base + token] = __float2bfloat16((accumulator + __bfloat162float(x[base + token])) *
                                         (1.0F / 3.0F)); // 1/sqrt(2 * 4 + 1)
}

__global__ void dynamic_short_conv_state_kernel(const __nv_bfloat16* __restrict__ x,
                                                const __nv_bfloat16* __restrict__ state_in,
                                                __nv_bfloat16* __restrict__ state_out,
                                                int channels, int tokens) {
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

void dynamic_short_conv_launch(const Tensor& x, const Tensor& static_weight,
                               const Tensor& dynamic_weight, const Tensor& conv_state_in,
                               Tensor& conv_state_out, Tensor& out, int heads,
                               cudaStream_t stream) {
    const int channels = x.ne[0];
    const int tokens = x.ne[1];
    if (channels <= 0 || tokens <= 0) { return; }
    const int head_width = channels / heads;
    if (head_width * heads != channels) {
        throw std::invalid_argument("dynamic_short_conv: channels must divide by heads");
    }

    const dim3 grid(static_cast<unsigned>((tokens + 127) / 128), static_cast<unsigned>(channels));
    dynamic_short_conv_kernel<<<grid, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(static_weight.data),
        static_cast<const __nv_bfloat16*>(dynamic_weight.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), static_cast<__nv_bfloat16*>(out.data),
        channels, tokens, head_width, heads);

    const auto state_blocks = static_cast<unsigned>((channels + 255) / 256);
    dynamic_short_conv_state_kernel<<<state_blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), channels, tokens);
}

} // namespace ninfer::ops::detail
