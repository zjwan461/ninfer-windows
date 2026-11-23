#pragma once

#include "artifact/materializer.h"
#include "models/inkling2/config.h"
#include "models/inkling2/load.h"
#include "models/inkling2/parameters.h"

#include <cstdint>
#include <vector>

namespace ninfer {
struct DeviceContext;
struct Weight;
} // namespace ninfer

namespace ninfer::artifact {
class Reader;
}

namespace ninfer::models::inkling2 {

// Text weights on the device, addressed by role. `slots` is parallel to the order
// `bind_parameters` produced, so `index` positions index into it directly.
struct TextWeights {
    artifact::MaterializedArtifact backing;
    std::vector<Weight> slots;
    Index index;
    std::uint64_t device_bytes   = 0;
    std::uint64_t reserved_bytes = 0;
};

// Materialise every text slot the artifact carries. The artifact stays the source of truth for
// shapes and formats: the binder validates each declaration before any byte moves.
[[nodiscard]] TextWeights materialize_text(const artifact::Reader& reader, const Config& config,
                                            DeviceContext& device,
                                            const StartupObserver* observer = nullptr);

} // namespace ninfer::models::inkling2
