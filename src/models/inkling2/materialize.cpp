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

    out.views.reserve(references.size());
    for (const auto& reference : references) {
        auto view = artifact::bind_view(reference, out.backing);
        // Every slot is one whole object today; a partial plane would need an explicit consumer,
        // so fail here instead of handing a half weight to a kernel.
        if (!is_complete_weight(view)) {
            throw artifact::ArtifactError("Inkling2: " + reference.name +
                                          " does not bind one complete weight");
        }
        out.views.push_back(std::move(view));
    }
    return out;
}

Weight matrix_weight(const TextWeights& weights, std::int32_t slot) {
    if (slot < 0 || static_cast<std::size_t>(slot) >= weights.views.size()) {
        throw artifact::ArtifactError("Inkling2: weight slot out of range");
    }
    return native_weight(weights.views[static_cast<std::size_t>(slot)]);
}

Tensor dense_tensor(const TextWeights& weights, std::int32_t slot) {
    if (slot < 0 || static_cast<std::size_t>(slot) >= weights.views.size()) {
        throw artifact::ArtifactError("Inkling2: weight slot out of range");
    }
    const auto& view = weights.views[static_cast<std::size_t>(slot)];
    const auto region = contiguous_weight_region(view);
    const auto& geometry = region.parent->geometry;
    if (!region.parent->data) {
        throw artifact::ArtifactError("Inkling2: dense slot is not device resident");
    }
    if (geometry.format != QType::BF16 && geometry.format != QType::FP32) {
        throw artifact::ArtifactError("Inkling2: dense slot is not a direct format");
    }
    const auto word = geometry.format == QType::FP32 ? 4u : 2u;
    auto* data = const_cast<std::byte*>(region.parent->data) + region.begin * word;
    const auto dtype = geometry.format == QType::FP32 ? DType::FP32 : DType::BF16;
    switch (view.shape.size()) {
    case 1:
        return Tensor(data, dtype, {std::int32_t(view.shape[0])});
    case 2:
        return Tensor(data, dtype, {std::int32_t(view.shape[0]), std::int32_t(view.shape[1])});
    case 3:
        return Tensor(data, dtype, {std::int32_t(view.shape[0]), std::int32_t(view.shape[1]),
                                    std::int32_t(view.shape[2])});
    default:
        throw artifact::ArtifactError("Inkling2: dense slot has an unsupported rank");
    }
}

} // namespace ninfer::models::inkling2
