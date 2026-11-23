// ninfer::ops - Inkling 2 expert router.
#include "ops/launcher/moe_router.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// One block per token: thread e scores expert e, then thread 0 selects the top-k. The scoring is a
// row-major dot over the hidden axis, which is a memory-bound reduction this size; a GEMM route
// replaces it once the router is fused into the layer.
__global__ void moe_router_kernel(const __nv_bfloat16* __restrict__ x,
                                  const float* __restrict__ weight, const float* __restrict__ bias,
                                  std::int32_t* __restrict__ indices, float* __restrict__ weights,
                                  int experts, int hidden, int tokens, int top_k,
                                  float scale) {
    extern __shared__ float shared[]; // scores[experts] + slack for the selection
    float* scores = shared;

    const int token = blockIdx.x;
    const int expert = threadIdx.x;
    if (expert < experts) {
        const float* row = weight + std::size_t(expert) * hidden;
        float accumulator = 0.0F;
        for (int h = 0; h < hidden; ++h) {
            accumulator += __bfloat162float(x[std::size_t(h) * tokens + token]) * row[h];
        }
        const float logit = accumulator;
        scores[expert] = 1.0F / (1.0F + __expf(-logit));
    }
    __syncthreads();

    if (threadIdx.x != 0) { return; }
    // Selection uses score + bias; the weight uses the raw score. Ties resolve to the lower index.
    bool used[128];
    for (int e = 0; e < experts; ++e) { used[e] = false; }
    float selected[32];
    for (int j = 0; j < top_k; ++j) {
        int best = -1;
        float best_value = -1e30F;
        for (int e = 0; e < experts; ++e) {
            if (used[e]) { continue; }
            const float value = scores[e] + bias[e];
            if (value > best_value) { best_value = value; best = e; }
        }
        used[best] = true;
        indices[std::size_t(j) * tokens + token] = best;
        selected[j] = scores[best];
    }
    float total = 0.0F;
    for (int j = 0; j < top_k; ++j) { total += selected[j]; }
    const float denominator = total + 1e-20F;
    for (int j = 0; j < top_k; ++j) {
        weights[std::size_t(j) * tokens + token] = selected[j] / denominator * scale;
    }
}

} // namespace

void moe_router_launch(const Tensor& x, const Tensor& weight, const Tensor& bias, int top_k,
                       float scale, Tensor& indices, Tensor& weights, cudaStream_t stream) {
    const int hidden = x.ne[0];
    const int tokens = x.ne[1];
    const int experts = weight.ne[0];
    if (hidden <= 0 || tokens <= 0 || experts <= 0) { return; }
    if (experts > 128 || top_k <= 0 || top_k > 32 || top_k > experts) {
        throw std::invalid_argument("moe_router: expert or top-k geometry is out of range");
    }
    const std::size_t shared_bytes = std::size_t(experts + top_k) * sizeof(float);
    moe_router_kernel<<<tokens, 128, shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const float*>(weight.data),
        static_cast<const float*>(bias.data), static_cast<std::int32_t*>(indices.data),
        static_cast<float*>(weights.data), experts, hidden, tokens, top_k, scale);
}

} // namespace ninfer::ops::detail
