#pragma once

#include "ninfer/ops/residual.h"

namespace ninfer::ops::detail {

void cast_bf16_to_f32_launch(Tensor& out, const Tensor& in, cudaStream_t stream);
void cast_f32_to_bf16_launch(Tensor& out, const Tensor& in, cudaStream_t stream);
void residual_add_f32_launch(Tensor& out, const Tensor& base, const Tensor& addend,
                             cudaStream_t stream);
void residual_store_bf16_launch(Tensor& out, const Tensor& base, const Tensor& addend,
                                cudaStream_t stream);

} // namespace ninfer::ops::detail
