#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Inkling 2's query and relative-state normalisation:
//
//   v[h, d]  = x[h, d] * scale[h, d]
//   norm     = ||v[h, :]||_2
//   v        = v * min(threshold / (norm + 1e-6), 1)
//   v        = v * tau(pos)        only when tau_alpha > 0 (global attention layers)
//
// where `tau(pos) = 1 + tau_alpha * ln(max((pos + 1) / tau_floor, 1))`. The learned `scale` is
// per head and per head dimension, the clip never scales a short vector up, and both the norm and
// the scaling run in FP32 before the BF16 store.
//
// `x` and `out` are `[rows, heads * head_dim]` contiguous BF16 tokens, where a row's head `h`
// occupies `[h * head_dim, (h + 1) * head_dim)`. `positions` carries one position per row and is
// read only when tau_alpha > 0.
struct ScaledL2ClipArgs {
    int rows       = 0;
    int heads      = 0;
    int head_dim   = 0;
    float threshold = 0.0F;
    float tau_alpha = 0.0F;
    float tau_floor = 0.0F;
};

void scaled_l2_clip(const Tensor& x, const Tensor& scale, const Tensor* positions,
                    const ScaledL2ClipArgs& args, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
