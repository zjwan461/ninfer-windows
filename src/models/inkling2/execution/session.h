#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "models/inkling2/config.h"
#include "models/inkling2/parameters.h"
#include "models/inkling2/schedule.h"

#include "artifact/reader.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace ninfer::models::inkling2::execution {

// Inkling-2-Mini runs 48 layer visits over 24 layers, and every visit owns its own attention cache
// and convolution window: the loop core's three passes must not see each other's keys, values or
// convolution history. `max_context` sizes those caches.
struct SessionOptions {
    std::int32_t max_context = 4096;
};

// One decode step for a single token. Prefill arrives later; this path is what the benchmark and
// the parity test exercise.
class Session {
public:
    Session(Weights& weights, const Config& config, DeviceContext& device, SessionOptions options);
    ~Session();
    Session(const Session&)            = delete;
    Session& operator=(const Session&) = delete;

    void reset();

    // Runs the token through all 48 visits and returns the softcapped logits (BF16, vocabulary on
    // ne[1]). Nothing synchronises: the caller decides when to wait.
    Tensor step(std::uint32_t token, cudaStream_t stream);

    [[nodiscard]] std::int32_t position() const { return position_; }
    [[nodiscard]] std::int32_t max_context() const { return options_.max_context; }

    // Called with each intermediate stage's name and buffer after it is produced, so the parity
    // test can compare a running session against the recorded reference.
    std::function<void(const char*, void*, int)> stage_sink;

private:
    struct Scratch;

    void run_visit(const Visit& visit, std::int32_t stack_length, void* previous_block,
                   void* previous_h, cudaStream_t stream);
    void finish(cudaStream_t stream);

    Weights& weights_;
    const Config& config_;
    Schedule schedule_;
    SessionOptions options_;
    std::unique_ptr<Scratch> s_;
    std::vector<int> visit_cache_;  // visit index -> cache slot

    std::int32_t position_ = 0;
};

} // namespace ninfer::models::inkling2::execution
