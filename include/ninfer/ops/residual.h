#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// The reference accumulates Inkling 2's residual stream in FP32 and rounds to BF16 once, at the
// layer output. These three helpers keep that shape: promote a BF16 value, add a BF16 tensor into
// an FP32 accumulator, and store an FP32 accumulator plus a BF16 addend with a single rounding.

void cast_bf16_to_f32(Tensor& out, const Tensor& in, cudaStream_t stream);
void cast_f32_to_bf16(Tensor& out, const Tensor& in, cudaStream_t stream);

void residual_add_f32(Tensor& out, const Tensor& base, const Tensor& addend, cudaStream_t stream);

// out = to_bf16(base + addend), with the add in FP32 and one rounding on the store.
void residual_store_bf16(Tensor& out, const Tensor& base, const Tensor& addend, cudaStream_t stream);

// acc += scale * x, accumulating in FP32. The routed experts use this so the mixture stays on the
// device instead of round-tripping through the host once per expert.
void axpy_bf16_into_f32(Tensor& acc, const Tensor& x, float scale, cudaStream_t stream);

} // namespace ninfer::ops
