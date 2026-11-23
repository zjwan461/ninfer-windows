#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "models/inkling2/config.h"
#include "models/inkling2/materialize.h"
#include "models/inkling2/parameters.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace ninfer::models::inkling2::execution {

// Inkling-2-Mini decodes 24 layers as 48 visits, and every visit owns its attention cache and its
// convolution window: the loop core's three passes must not see each other's keys, values or
// convolution history. `max_context` sizes those caches.
struct SessionOptions {
    std::int32_t max_context = 4096;
};

// One token per step. Buffers and caches are allocated once in the constructor: nothing in the
// decode path allocates, because a cudaMalloc here synchronises the device.
class Session {
public:
    Session(TextWeights& weights, const Config& config, DeviceContext& device, SessionOptions options);
    ~Session();
    Session(const Session&)            = delete;
    Session& operator=(const Session&) = delete;

    void reset();

    // Runs the token through every visit and returns the softcapped logits as BF16 [vocabulary, 1].
    // The returned view stays valid until the next step. Nothing synchronises here.
    Tensor step(std::uint32_t token, cudaStream_t stream);

    [[nodiscard]] std::int32_t position() const;
    [[nodiscard]] std::int32_t max_context() const;

    // Called with a stage name and its buffer after each intermediate is produced, so the parity
    // test can compare a live session against the recorded reference.
    std::function<void(int layer, const char*, void*, int)> stage_sink;

    // Called with each visit's routed selection before the experts run. A parity test uses it to
    // pin the selection to the reference's, because this model's router has near-ties and a flip
    // would otherwise be indistinguishable from an arithmetic error.
    std::function<void(int visit, std::vector<std::int32_t>& indices,
                       std::vector<float>& weights)> routing_hook;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::inkling2::execution
