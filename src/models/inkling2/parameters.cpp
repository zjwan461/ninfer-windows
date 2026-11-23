#include "models/inkling2/parameters.h"

#include <array>
#include <charconv>
#include <map>
#include <string>
#include <string_view>

namespace ninfer::models::inkling2 {
namespace {

using artifact::ArtifactError;

struct RoleEntry {
    std::string_view suffix;
    Role role;
};

// Longest suffix wins, so `k_sconv/static_w` is never read as `k_sconv`.
constexpr RoleEntry kLayerRoles[] = {
    {"/attn/wq_du", Role::WqDu},
    {"/attn/wk_dv", Role::WkDv},
    {"/attn/wv_dv", Role::WvDv},
    {"/attn/wo_ud", Role::WoUd},
    {"/attn/wr_du", Role::WrDu},
    {"/attn/gate_attn", Role::GateAttn},
    {"/attn/attn_sink_proj", Role::AttnSink},
    {"/attn/rel_logits_proj/proj", Role::RelLogits},
    {"/attn/q_l2_clip/scale", Role::QkClip},
    {"/attn/rel_l2_clip/scale", Role::RelClip},
    {"/attn/k_sconv/static_w", Role::KSconvStatic},
    {"/attn/k_sconv/weight_proj", Role::KSconvProj},
    {"/attn/v_sconv/static_w", Role::VSconvStatic},
    {"/attn/v_sconv/weight_proj", Role::VSconvProj},
    {"/attn/k_sconv", Role::KSconv},
    {"/attn/v_sconv", Role::VSconv},
    {"/attn_sconv", Role::AttnSconv},
    {"/attn_norm", Role::AttnNorm},
    {"/q_input_norm", Role::QInputNorm},
    {"/k_input_norm", Role::KInputNorm},
    {"/v_input_norm", Role::VInputNorm},
    {"/mlp_norm", Role::MlpNorm},
    {"/mlp_sconv", Role::MlpSconv},
    {"/mlp/gate/bias", Role::MlpGateBias},
    {"/mlp/gate", Role::MlpGate},
};

std::uint32_t parse_layer(std::string_view segment, std::string_view name) {
    std::uint32_t value = 0;
    const auto result = std::from_chars(segment.data(), segment.data() + segment.size(), value);
    if (result.ec != std::errc{} || result.ptr != segment.data() + segment.size()) {
        throw ArtifactError("Inkling2: " + std::string(name) + " has a non-numeric layer index");
    }
    return value;
}

std::uint32_t parse_expert(std::string_view segment, std::string_view name) {
    std::uint32_t value = 0;
    const auto result = std::from_chars(segment.data(), segment.data() + segment.size(), value);
    if (result.ec != std::errc{} || result.ptr != segment.data() + segment.size()) {
        throw ArtifactError("Inkling2: " + std::string(name) + " has a non-numeric expert index");
    }
    return value;
}

void assign(std::int32_t& slot, std::int32_t value, std::string_view name) {
    if (slot >= 0) {
        throw ArtifactError("Inkling2: two parameters claim one role (" + std::string(name) + ")");
    }
    slot = value;
}

std::int32_t controller_input(std::string_view name) {
    constexpr std::string_view kInputs[] = {"/controllers/attn/", "/controllers/k/",
                                            "/controllers/v/", "/controllers/mlp/"};
    for (std::size_t i = 0; i < std::size(kInputs); ++i) {
        if (name.find(kInputs[i]) != std::string_view::npos) { return std::int32_t(i); }
    }
    return -1;
}

} // namespace

Index build_index(const std::vector<BoundSlot>& slots, const Config& config) {
    Index index;
    index.layers.resize(config.num_hidden_layers);
    for (auto& layer : index.layers) {
        layer.expert_w13.assign(config.moe.num_experts, -1);
        layer.expert_w2.assign(config.moe.num_experts, -1);
    }

    for (std::size_t position = 0; position < slots.size(); ++position) {
        const auto& name = slots[position].name;
        const auto here = std::int32_t(position);

        if (name == "text/embed") { assign(index.embed, here, name); continue; }
        if (name == "text/embed_norm") { assign(index.embed_norm, here, name); continue; }
        if (name == "text/norm") { assign(index.norm, here, name); continue; }
        if (name == "text/unembed") { assign(index.unembed, here, name); continue; }
        if (name == "text/attn_res/output_query") { assign(index.output_query, here, name); continue; }
        if (name == "text/middle_cycle_cross_pass_queries") {
            assign(index.cross_pass_queries, here, name); continue;
        }
        if (name == "text/middle_cycle_cross_pass_kv_queries") {
            assign(index.cross_pass_kv_queries, here, name); continue;
        }

        constexpr std::string_view kAttnRes = "text/attn_res/";
        if (name.starts_with(kAttnRes)) {
            const auto rest = std::string_view(name).substr(kAttnRes.size());
            const auto slash = rest.find('/');
            if (slash == std::string_view::npos) {
                throw ArtifactError("Inkling2: " + name + " has no layer index");
            }
            const auto kind = rest.substr(0, slash);
            const auto layer = parse_layer(rest.substr(slash + 1), name);
            if (layer >= index.layers.size()) {
                throw ArtifactError("Inkling2: " + name + " exceeds the layer count");
            }
            if (kind == "attn_queries") {
                // The checkpoint stores one entry per layer *with* an attention residual read,
                // i.e. layers 1..N-1, so its index is one behind the layer it belongs to.
                index.attn_queries.resize(config.num_hidden_layers, -1);
                if (layer + 1 >= index.attn_queries.size()) {
                    throw ArtifactError("Inkling2: " + name + " exceeds the layer count");
                }
                assign(index.attn_queries[layer + 1], here, name);
                continue;
            }
            if (kind == "mlp_queries") { index.mlp_queries.resize(config.num_hidden_layers, -1);
                                         assign(index.mlp_queries[layer], here, name); continue; }
            throw ArtifactError("Inkling2: unknown residual query " + name);
        }

        constexpr std::string_view kLayers = "text/layers/";
        if (!name.starts_with(kLayers)) {
            throw ArtifactError("Inkling2: " + name + " is not a text parameter");
        }
        const auto rest = std::string_view(name).substr(kLayers.size());
        const auto slash = rest.find('/');
        if (slash == std::string_view::npos) {
            throw ArtifactError("Inkling2: " + name + " has no sublayer");
        }
        const auto layer = parse_layer(rest.substr(0, slash), name);
        if (layer >= index.layers.size()) {
            throw ArtifactError("Inkling2: " + name + " exceeds the layer count");
        }
        auto& slots_of_layer = index.layers[layer];
        const auto tail = rest.substr(slash);

        constexpr std::string_view kExperts = "/mlp/experts/";
        if (tail.starts_with(kExperts)) {
            const auto after = tail.substr(kExperts.size());
            const auto next = after.find('/');
            if (next == std::string_view::npos) {
                throw ArtifactError("Inkling2: " + name + " has no expert weight");
            }
            const auto expert = parse_expert(after.substr(0, next), name);
            if (expert >= config.moe.num_experts) {
                throw ArtifactError("Inkling2: " + name + " exceeds the expert count");
            }
            const auto weight = after.substr(next + 1);
            if (weight == "w13") { assign(slots_of_layer.expert_w13[expert], here, name); continue; }
            if (weight == "w2") { assign(slots_of_layer.expert_w2[expert], here, name); continue; }
            throw ArtifactError("Inkling2: unknown expert weight " + name);
        }

        constexpr std::string_view kControllers = "/controllers/";
        if (tail.starts_with(kControllers)) {
            const auto input = controller_input(name);
            if (input < 0) { throw ArtifactError("Inkling2: unknown controller input " + name); }
            std::int32_t which = -1;
            if (tail.ends_with("/down_weight")) { which = 0; }
            else if (tail.ends_with("/forget_weight")) { which = 1; }
            else if (tail.ends_with("/gain_weight")) { which = 2; }
            else { throw ArtifactError("Inkling2: unknown controller role " + name); }
            const std::array<std::array<std::int32_t, 4>*, 3> rows = {
                &slots_of_layer.controller_down, &slots_of_layer.controller_forget,
                &slots_of_layer.controller_gain};
            assign((*rows[static_cast<std::size_t>(which)])[static_cast<std::size_t>(input)],
                   here, name);
            continue;
        }

        bool matched = false;
        for (const auto& entry : kLayerRoles) {
            if (tail == entry.suffix) {
                assign(slots_of_layer.role[static_cast<std::size_t>(entry.role)], here, name);
                matched = true;
                break;
            }
        }
        if (!matched) { throw ArtifactError("Inkling2: " + name + " has no known role"); }
    }

    return index;
}

} // namespace ninfer::models::inkling2
