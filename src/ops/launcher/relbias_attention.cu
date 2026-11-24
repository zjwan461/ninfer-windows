// ninfer::ops - Inkling 2 decode attention: relative bias, sink and output gate.
#include "ops/launcher/relbias_attention.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kWarp = 32;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// One warp per query head. Each lane owns four of the head's dimensions, so a head dimension of
// 128 maps exactly onto the warp and the per-key dot product is a shuffle reduction.
__global__ void relbias_attention_kernel(const __nv_bfloat16* __restrict__ q,
                                         const __nv_bfloat16* __restrict__ k,
                                         const __nv_bfloat16* __restrict__ v,
                                         const __nv_bfloat16* __restrict__ rel,
                                         const __nv_bfloat16* __restrict__ proj,
                                         const __nv_bfloat16* __restrict__ sink,
                                         const __nv_bfloat16* __restrict__ gate,
                                         __nv_bfloat16* __restrict__ out, int heads,
                                         int kv_heads, int head_dim, int q_pos, int kv_len,
                                         int window, int rel_extent, int d_rel, float scale) {
    const int head = blockIdx.x * (blockDim.x / kWarp) + (threadIdx.x / kWarp);
    if (head >= heads) { return; }
    const int lane = threadIdx.x % kWarp;
    const int dims = head_dim / kWarp; // 4 for head_dim 128
    const int kv_head = head / (heads / kv_heads);

    float q_reg[4];
    const __nv_bfloat16* q_head = q + head * head_dim + lane * dims;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q_reg[i] = (i < dims) ? __bfloat162float(q_head[i]) : 0.0F;
    }

    float rel_reg[16];
    const int rel_count = d_rel < 16 ? d_rel : 16;
    for (int r = 0; r < rel_count; ++r) { rel_reg[r] = __bfloat162float(rel[head * d_rel + r]); }
    const __nv_bfloat16* proj_base = proj;

    const float sink_value = __bfloat162float(sink[head]);
    const int first = (window > 0 && q_pos - window + 1 > 0) ? q_pos - window + 1 : 0;

    float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    float running_max = sink_value;
    float running_sum = 1.0F; // exp(sink - running_max) with running_max == sink

    const __nv_bfloat16* k_head = k + (std::size_t(kv_head) * kv_len + first) * head_dim;
    const __nv_bfloat16* v_head = v + (std::size_t(kv_head) * kv_len + first) * head_dim;
    const int count = q_pos - first + 1;

    for (int step = 0; step < count; ++step) {
        const int position = first + step;
        float dot = 0.0F;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (i < dims) {
                dot += q_reg[i] * __bfloat162float(k_head[std::size_t(step) * head_dim + lane * dims + i]);
            }
        }
        dot = warp_sum(dot) * scale;

        const int distance = q_pos - position;
        float bias = 0.0F;
        if (distance >= 0 && distance < rel_extent) {
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                if (r < rel_count) { bias += rel_reg[r] * __bfloat162float(proj_base[r * rel_extent + distance]); }
            }
        }

        const float logit = dot + bias;
        const float new_max = fmaxf(running_max, logit);
        const float correction = __expf(running_max - new_max);
        const float weight = __expf(logit - new_max);
        running_sum = running_sum * correction + weight;
        running_max = new_max;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (i < dims) {
                acc[i] = acc[i] * correction +
                         weight * __bfloat162float(v_head[std::size_t(step) * head_dim + lane * dims + i]);
            }
        }
    }

    // The sink entered the running normaliser as exp(sink - running_max) with no numerator term,
    // so dividing by running_sum already excludes its value and includes its probability mass.
    __nv_bfloat16* out_head = out + head * head_dim + lane * dims;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (i < dims) {
            const float gated = (acc[i] / running_sum) *
                                 (1.0F / (1.0F + __expf(-__bfloat162float(
                                                     gate[head * head_dim + lane * dims + i]))));
            out_head[i] = __float2bfloat16(gated);
        }
    }
}

} // namespace

void relbias_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& rel,
                              const Tensor& proj, const Tensor& sink, const Tensor& gate,
                              const RelBiasAttentionArgs& args, Tensor& out,
                              cudaStream_t stream) {
    if (args.heads <= 0 || args.kv_heads <= 0 || args.kv_heads > args.heads) {
        throw std::invalid_argument("relbias_attention: invalid head geometry");
    }
    if (args.heads % args.kv_heads != 0) {
        throw std::invalid_argument("relbias_attention: heads must be a multiple of kv heads");
    }
    if (args.head_dim != 128) {
        throw std::invalid_argument("relbias_attention: only head_dim 128 is implemented");
    }
    if (args.q_pos < 0 || args.q_pos >= args.kv_len) {
        throw std::invalid_argument("relbias_attention: q_pos outside the cache");
    }
    if (args.d_rel <= 0 || args.d_rel > 16 || args.rel_extent <= 0) {
        throw std::invalid_argument("relbias_attention: invalid relative-bias geometry");
    }

    constexpr int kWarpBlock = 128; // four warps, four heads per block
    const auto blocks = static_cast<unsigned int>((args.heads + 3) / 4);
    relbias_attention_kernel<<<blocks, kWarpBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const __nv_bfloat16*>(rel.data),
        static_cast<const __nv_bfloat16*>(proj.data), static_cast<const __nv_bfloat16*>(sink.data),
        static_cast<const __nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(out.data),
        args.heads, args.kv_heads, args.head_dim, args.q_pos, args.kv_len, args.window,
        args.rel_extent, args.d_rel, args.scale);
}

} // namespace ninfer::ops::detail
