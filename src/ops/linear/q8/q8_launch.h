#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q8Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q8_gemv_n2048_k16384(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_simt_r8_c4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_simt_r8_c8(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r32_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r32_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r32_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r48_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r64_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r64_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r96_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r128_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r128_c80(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r64x16_c48_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_mma_r64x32_c64_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);


// Runtime-extent q8 decode route for shapes without a tuned kernel.
void launch_q8_gather_gemv(const Tensor& x, const Weight* table, const std::int32_t* slots,
                           int slot, const Weight& extents, Tensor& out, cudaStream_t stream);

void launch_q8_gather_accumulate(const Tensor& x, const Weight* table, const std::int32_t* slots,
                                 int slot, const Weight& extents, const float* scales,
                                 Tensor& acc, cudaStream_t stream);

void launch_q8_generic_gemv(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail
