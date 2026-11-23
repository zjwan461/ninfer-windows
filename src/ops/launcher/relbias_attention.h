#pragma once

#include "ninfer/ops/relbias_attention.h"

namespace ninfer::ops::detail {

void relbias_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& rel, const Tensor& proj, const Tensor& sink,
                              const Tensor& gate, const RelBiasAttentionArgs& args,
                              Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
