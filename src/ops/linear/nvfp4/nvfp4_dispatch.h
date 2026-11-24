#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                                std::int32_t input_rows,
                                                                LinearPolicy policy,
                                                                std::int32_t min_tokens,
                                                                std::int32_t max_tokens);

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream);

// True when a tuned NVFP4 route exists for this geometry.
[[nodiscard]] bool has_nvfp4_shape(std::int32_t n, std::int32_t k);

} // namespace ninfer::ops::detail
