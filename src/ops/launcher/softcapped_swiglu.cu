// ninfer::ops - Inkling 2 softcapped SwiGLU.
#include "ops/launcher/softcapped_swiglu.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

__device__ __forceinline__ float softcap(float value, float cap) {
    return tanhf(value / cap) * cap;
}

// One thread per output element; the op is elementwise across the fused halves.
__global__ void softcapped_swiglu_kernel(const __nv_bfloat16* __restrict__ gate_up,
                                         __nv_bfloat16* __restrict__ out, int inter, int tokens,
                                         float gate_cap, float up_cap) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= tokens) { return; }
    const int row = blockIdx.y;
    if (row >= inter) { return; }

    const std::size_t gate_index = std::size_t(row) * tokens + column;
    const std::size_t up_index = std::size_t(inter + row) * tokens + column;
    const float gate = __bfloat162float(gate_up[gate_index]);
    const float up = __bfloat162float(gate_up[up_index]);
    const float value = softcap(gate, gate_cap) * (1.0F / (1.0F + __expf(-gate))) *
                        softcap(up, up_cap);
    out[std::size_t(row) * tokens + column] = __float2bfloat16(value);
}

} // namespace

void softcapped_swiglu_launch(const Tensor& gate_up, float gate_cap, float up_cap, Tensor& out,
                              cudaStream_t stream) {
    const int inter = out.ne[0];
    const int tokens = out.ne[1];
    if (inter <= 0 || tokens <= 0) { return; }
    if (gate_up.ne[0] != 2 * inter) {
        throw std::invalid_argument("softcapped_swiglu: gate_up must carry twice the rows of out");
    }
    const dim3 grid(static_cast<unsigned>((tokens + 127) / 128), static_cast<unsigned>(inter));
    softcapped_swiglu_kernel<<<grid, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up.data), static_cast<__nv_bfloat16*>(out.data),
        inter, tokens, gate_cap, up_cap);
}

} // namespace ninfer::ops::detail
