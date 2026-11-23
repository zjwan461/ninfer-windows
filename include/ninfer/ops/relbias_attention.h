#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Decode-step attention with Inkling 2's additions: a learned relative-position bias over a
// bounded distance, a per-head sink logit, and the output gate.
//
//   logits[j] = scale * dot(q[h], k[kvh][j]) + bias(h, q_pos - j)
//   bias(h, d) = dot(rel[h], proj[:, d]) for 0 <= d < rel_extent, else 0
//   p = softmax([logits..., sink[h]])[..., :-1]
//   out[h, :] = (sum_j p[j] v[kvh][j]) * sigmoid(gate[h, :])
//
// `window` bounds the visible distance (0 means the whole cache), `q` is the already-clipped
// query of Inkling 2's projection chain, and the sigmoid is applied in FP32 before the output is
// stored. The kernel reads one query position; prefill arrives with the chunked variant.
struct RelBiasAttentionArgs {
    int heads      = 0;
    int kv_heads   = 0;
    int head_dim   = 0;
    int q_pos      = 0;
    int kv_len     = 0;
    int window     = 0;
    int rel_extent = 0;
    int d_rel      = 0;
    float scale    = 0.0F;
};

void relbias_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& rel,
                       const Tensor& proj, const Tensor& sink, const Tensor& gate,
                       const RelBiasAttentionArgs& args, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
