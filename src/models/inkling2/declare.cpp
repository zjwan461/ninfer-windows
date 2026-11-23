#include "models/inkling2/load.h"

#include "artifact/binder.h"

namespace ninfer::models::inkling2 {

// Separate translation unit on purpose: declaring parameters is the only text-loading step that
// needs the artifact binder and, through it, CUDA. Everything else in `load` stays host-only and
// unit-testable without a device.
void declare_parameters(artifact::Binder& binder, const Config& config) {
    for (const auto& slot : expected_slots(config)) {
        (void)binder.parameter(slot.name, slot.shape, artifact::Residency::Device,
                               weight_type(slot.format));
    }
}

} // namespace ninfer::models::inkling2
