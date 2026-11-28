#pragma once
#include "core/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::core {

// Receives the keys of one attention layer at the only moment they still are what a recall index
// wants: before the rotation. What is stored on the device is post-rotation, so a reader of the KV
// cache can only recover a rotated key; the caller here hands over the tensor that is about to be
// rotated in place, and a sink that keeps its own copy is unaffected by the rotation that follows.
//
// Columns arrive in the order they were written, one layer at a time, and a sink is expected to
// track its own position. Only committed history is offered: the prefill path calls this, the
// verify path (whose extra columns may be discarded) does not.
class KvRecallSink {
public:
    virtual ~KvRecallSink()                                      = default;
    KvRecallSink(const KvRecallSink&)                            = delete;
    KvRecallSink& operator=(const KvRecallSink&)                 = delete;
    KvRecallSink(KvRecallSink&&)                                 = delete;
    KvRecallSink& operator=(KvRecallSink&&)                      = delete;

    // pre_rope_keys: bf16 [head_dim, kv_heads, columns], contiguous, valid until the stream reaches
    // the work the sink enqueues. layer is the index of the attention layer in the model.
    virtual void capture_layer(std::int32_t layer, const Tensor& pre_rope_keys,
                               cudaStream_t stream) = 0;

    // The queries of the same columns, also before the rotation: the recall score compares a query
    // with a block mean, so a sink needs both sides in the same space. Offered right after
    // capture_layer for the same layer and the same columns.
    virtual void capture_query(std::int32_t, const Tensor&, cudaStream_t) {}

    // The canonical prefill consumed the whole prompt for this turn. A sink that publishes derived
    // state (means, counts, a summary) finishes it here; the default sink has nothing to do.
    virtual void end_of_prefill(cudaStream_t) {}

protected:
    KvRecallSink() = default;
};

} // namespace ninfer::core