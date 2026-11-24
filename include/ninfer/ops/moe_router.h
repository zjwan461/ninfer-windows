#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Inkling 2's expert router: sigmoid scoring, selection bias, top-k, renormalise, route scale.
//
//   logits[e]  = sum_h x[h, t] * weight[e, h]          FP32
//   scores[e]  = sigmoid(logits[e])
//   selected   = the top-k entries of scores[e] + bias[e]   (the bias steers selection only)
//   weights[j] = scores[selected[j]] / (sum_j scores + 1e-20) * scale
//
// `x` is contiguous BF16 [hidden, tokens], `weight` FP32 [experts, hidden], `bias` FP32 [experts];
// `indices` receives INT32 [top_k, tokens] and `weights` FP32 [top_k, tokens]. The selection order
// is unspecified because the MoE accumulates its top-k outputs.
void moe_router(const Tensor& x, const Tensor& weight, const Tensor& bias, int top_k, float scale,
                Tensor& indices, Tensor& weights, cudaStream_t stream);

} // namespace ninfer::ops
