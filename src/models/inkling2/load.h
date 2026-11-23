#pragma once

#include "models/inkling2/config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models::inkling2 {

// One logical parameter the text mathematics consumes, with the shape and stored format the
// checkpoint fixes. Names follow the artifact binding convention (`text/layers/6/attn/wq_du`).
struct Slot {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::string format;
};

// Every text slot for this config. Experts are separate parameters so each keeps its own NVFP4
// weight divisor; layers below `kPassThroughExperts` carry the checkpoint's NVFP4 payload, the
// rest were requantised at the same 8-bit granularity.
inline constexpr std::uint32_t kPassThroughExperts = 20;

[[nodiscard]] std::vector<Slot> expected_slots(const Config& config);

// Throw when the artifact does not bind exactly these slots with these shapes and formats.
void validate_parameters(const artifact::Directory& directory, const Config& config);

} // namespace ninfer::models::inkling2
