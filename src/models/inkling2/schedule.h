#pragma once

#include "models/inkling2/config.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::inkling2 {

// One layer application. `message_from` is the visit whose hidden state this one reads from: the
// same layer one pass earlier, or -1 when the visit starts from the prelude output.
struct Visit {
    std::uint32_t index          = 0;
    std::uint32_t layer          = 0;
    std::uint32_t pass           = 0;
    std::int32_t message_from    = -1;
    bool core                    = false;
    bool sliding                 = false;
};

// The static 48-visit plan. A visit owns one KV cache and one conv-state cache, which is why
// these are cache indices and not layer indices.
struct Schedule {
    std::vector<Visit> visits;
    // Visits that read a message, in execution order: the second and later passes of the core.
    std::vector<std::uint32_t> message_visits;
};

[[nodiscard]] Schedule build_schedule(const Config& config);

} // namespace ninfer::models::inkling2
