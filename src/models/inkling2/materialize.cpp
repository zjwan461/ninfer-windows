#include "models/inkling2/materialize.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/types.h"

#include <utility>

namespace ninfer::models::inkling2 {

TextWeights materialize_text(const artifact::Reader& reader, const Config& config,
                             DeviceContext& device, const StartupObserver* observer) {
    // Resolve first: a name, shape or format disagreement fails before the allocator runs.
    const auto bound = bind_parameters(reader.directory(), config);

    artifact::Binder binder(reader);
    std::vector<artifact::ParameterReference> references;
    references.reserve(bound.size());
    for (const auto& slot : bound) {
        references.push_back(binder.parameter(slot.name, slot.shape, artifact::Residency::Device,
                                              weight_type(slot.format)));
    }

    TextWeights out;
    out.index = build_index(bound, config);
    out.backing = artifact::materialize(reader, std::move(binder).finish(), device, observer);
    out.device_bytes   = out.backing.stats().h2d_bytes;
    out.reserved_bytes = out.backing.stats().device_capacity_bytes;

    out.slots.reserve(references.size());
    for (const auto& reference : references) {
        const auto view = artifact::bind_view(reference, out.backing);
        // NVFP4 and q8 arrive as complete parents, so the native Weight ABI covers them; a
        // parameter that ever binds a partial plane must be handled explicitly here instead.
        if (!is_complete_weight(view)) {
            throw artifact::ArtifactError("Inkling2: " + reference.name +
                                          " does not bind one complete weight");
        }
        out.slots.push_back(native_weight(view));
    }
    return out;
}

} // namespace ninfer::models::inkling2
