#include "models/inkling2/schedule.h"

#include "artifact/schema.h"

#include <string>

namespace ninfer::models::inkling2 {

Schedule build_schedule(const Config& config) {
    Schedule out;
    const auto layers = config.num_hidden_layers;
    const auto core_start = config.loop.core_start;
    const auto core_end   = config.loop.core_end;
    out.visits.reserve(config.visits());

    // Prelude: runs once, reads nothing from a previous pass.
    for (std::uint32_t layer = 0; layer < core_start; ++layer) {
        out.visits.push_back(Visit{std::uint32_t(out.visits.size()), layer, 0, -1, false,
                                   config.sliding_layer[layer]});
    }

    // Core: every pass starts from the prelude output; a later pass reads the previous pass's
    // visit of the same layer as its message.
    std::vector<std::uint32_t> previous_pass;
    for (std::uint32_t pass = 0; pass < config.loop.num_passes; ++pass) {
        std::vector<std::uint32_t> this_pass;
        this_pass.reserve(core_end - core_start);
        for (std::uint32_t layer = core_start; layer < core_end; ++layer) {
            const auto index = std::uint32_t(out.visits.size());
            const auto from = pass == 0 ? -1 : std::int32_t(
                previous_pass[layer - core_start]);
            out.visits.push_back(Visit{index, layer, pass, from, true,
                                       config.sliding_layer[layer]});
            if (from >= 0) { out.message_visits.push_back(index); }
            this_pass.push_back(index);
        }
        previous_pass = std::move(this_pass);
    }

    // Coda: continues from the last pass and runs once.
    for (std::uint32_t layer = core_end; layer < layers; ++layer) {
        out.visits.push_back(Visit{std::uint32_t(out.visits.size()), layer, 0, -1, false,
                                   config.sliding_layer[layer]});
    }

    if (out.visits.size() != config.visits()) {
        throw artifact::ArtifactError("Inkling2: schedule length " +
                                      std::to_string(out.visits.size()) + " differs from " +
                                      std::to_string(config.visits()));
    }
    if (out.message_visits.size() !=
        std::size_t(config.loop.core_layers()) * (config.loop.num_passes - 1)) {
        throw artifact::ArtifactError("Inkling2: message visit count is wrong");
    }
    return out;
}

} // namespace ninfer::models::inkling2
