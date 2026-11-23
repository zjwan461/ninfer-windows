#pragma once

#include "models/inkling2/load.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace ninfer::models::inkling2 {

// Roles a text slot can take. The execution code addresses weights by role, never by string, so a
// typo in a parameter name fails at load time instead of producing a silently wrong tensor.
enum class Role : std::uint8_t {
    WqDu, WkDv, WvDv, WoUd, WrDu, GateAttn, AttnSink, RelLogits,
    QkClip, RelClip, KSconv, VSconv, KSconvStatic, VSconvStatic, KSconvProj, VSconvProj,
    AttnSconv, AttnNorm, QInputNorm, KInputNorm, VInputNorm,
    MlpNorm, MlpSconv, MlpGate, MlpGateBias,
    ExpertW13, ExpertW2,
    ControllerAttnDown, ControllerAttnForget, ControllerAttnGain,
    ControllerKDown, ControllerKForget, ControllerKGain,
    ControllerVDown, ControllerVForget, ControllerVGain,
    ControllerMlpDown, ControllerMlpForget, ControllerMlpGain,
    Count
};

inline constexpr std::size_t kRoleCount = static_cast<std::size_t>(Role::Count);
inline constexpr std::string_view kRoleName[] = {
    "wq_du", "wk_dv", "wv_dv", "wo_ud", "wr_du", "gate_attn", "attn_sink", "rel_logits",
    "q_l2_clip", "rel_l2_clip", "k_sconv", "v_sconv",
    "k_sconv_static", "v_sconv_static", "k_sconv_proj", "v_sconv_proj",
    "attn_sconv", "attn_norm", "q_input_norm", "k_input_norm", "v_input_norm",
    "mlp_norm", "mlp_sconv", "mlp_gate", "mlp_gate_bias",
    "expert_w13", "expert_w2",
    "ctrl_attn_down", "ctrl_attn_forget", "ctrl_attn_gain",
    "ctrl_k_down", "ctrl_k_forget", "ctrl_k_gain",
    "ctrl_v_down", "ctrl_v_forget", "ctrl_v_gain",
    "ctrl_mlp_down", "ctrl_mlp_forget", "ctrl_mlp_gain",
};
static_assert(std::size(kRoleName) == kRoleCount, "role names must cover every role");

// One layer's slots. `role` holds the artifact index for the roles the layer actually has;
// experts and controllers are dense arrays because both are indexed by expert or input.
struct LayerSlots {
    std::array<std::int32_t, kRoleCount> role{};
    std::vector<std::int32_t> expert_w13;
    std::vector<std::int32_t> expert_w2;
    std::array<std::int32_t, 4> controller_down{};   // attn, k, v, mlp
    std::array<std::int32_t, 4> controller_forget{};
    std::array<std::int32_t, 4> controller_gain{};

    LayerSlots() { role.fill(-1); expert_w13.reserve(0); controller_down.fill(-1);
                   controller_forget.fill(-1); controller_gain.fill(-1); }

    [[nodiscard]] std::int32_t operator[](Role index) const {
        return role[static_cast<std::size_t>(index)];
    }
    [[nodiscard]] bool has(Role index) const {
        return role[static_cast<std::size_t>(index)] >= 0;
    }
};

struct Index {
    std::vector<LayerSlots> layers;
    std::int32_t embed = -1;
    std::int32_t embed_norm = -1;
    std::int32_t norm = -1;
    std::int32_t unembed = -1;
    std::vector<std::int32_t> attn_queries;
    std::vector<std::int32_t> mlp_queries;
    std::int32_t output_query = -1;
    std::int32_t cross_pass_queries = -1;
    std::int32_t cross_pass_kv_queries = -1;
};

// Map every bound slot to its role. Throws when a slot fits no role or two slots claim one.
[[nodiscard]] Index build_index(const std::vector<BoundSlot>& slots, const Config& config);

} // namespace ninfer::models::inkling2
