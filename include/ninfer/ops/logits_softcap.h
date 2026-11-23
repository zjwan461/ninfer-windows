#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// logits <- cap * tanh(logits / cap), in place. Inkling 2 bounds its logits this way (cap 30)
// before sampling, and doing it on the device keeps a decode step free of a host round trip.
void logits_softcap(Tensor& logits, float cap, cudaStream_t stream);

} // namespace ninfer::ops
