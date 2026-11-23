#include "models/inkling2/config.h"

#include "artifact/schema.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <string_view>

namespace ninfer::models::inkling2 {
namespace {

using artifact::ArtifactError;
using artifact::Json;
using artifact::require_members;

std::uint32_t integer(const Json& value, std::string_view label, bool positive = true) {
    const auto n = artifact::require_u64(value, label, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(label) + ": config dimension exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

// Every field access goes through here so a malformed config names the member it was reading.
const Json& member(const Json& value, const char* name) {
    if (!value.is_object()) {
        throw ArtifactError(std::string(name) + ": containing block is not an object");
    }
    const auto found = value.find(name);
    if (found == value.end()) {
        throw ArtifactError(std::string("missing member ") + name);
    }
    return *found;
}

std::uint32_t dimension(const Json& config, const char* name, bool positive = true) {
    return integer(member(config, name), name, positive);
}

float positive_number(const Json& value, std::string_view label) {
    if (!value.is_number()) {
        throw ArtifactError(std::string(label) + " must be a real value");
    }
    const float result = value.get<float>();
    if (!std::isfinite(result) || result <= 0) {
        throw ArtifactError(std::string(label) + " must be positive finite FP32");
    }
    return result;
}

float positive_float(const Json& config, const char* name) {
    return positive_number(member(config, name), name);
}

bool flag(const Json& config, const char* name) {
    const auto& value = member(config, name);
    if (!value.is_boolean()) {
        throw ArtifactError(std::string(name) + " must be a boolean");
    }
    return value.get<bool>();
}

std::string architecture(const Json& config) {
    const auto& names = member(config, "architectures");
    if (!names.is_array() || names.size() != 1) {
        throw ArtifactError("architectures must name one implementation");
    }
    return artifact::require_id(names[0], "architecture");
}

AttentionConfig attention(const Json& value, std::uint32_t layers) {
    AttentionConfig out;
    out.num_attention_heads = dimension(value, "num_attention_heads");
    out.num_key_value_heads = dimension(value, "num_key_value_heads");
    out.head_dim            = dimension(value, "head_dim");
    out.d_rel               = dimension(value, "d_rel");
    out.rel_extent          = dimension(value, "rel_extent");
    out.sconv_kernel        = dimension(value, "sconv_kernel_size");
    out.sliding_window      = dimension(value, "sliding_window_size");
    out.qk_l2_clip_threshold  = positive_float(value, "qk_l2_clip_threshold");
    out.rel_l2_clip_threshold = positive_float(value, "rel_l2_clip_threshold");
    out.log_scaling_floor     = positive_float(value, "log_scaling_n_floor");
    out.log_scaling_alpha     = positive_float(value, "log_scaling_alpha");
    out.value_norm_without_gamma = flag(value, "use_value_norm_no_gamma");
    if (out.num_attention_heads % out.num_key_value_heads) {
        throw ArtifactError("attention query heads must be divisible by KV heads");
    }
    if (out.qk_l2_clip_threshold > 1e3f || out.rel_l2_clip_threshold > 1e3f) {
        throw ArtifactError("attention clip thresholds are implausible");
    }
    (void)layers;
    return out;
}

// One entry per layer; `local_layer_ids` lists the sliding-window layers, the rest are global.
std::vector<bool> sliding_layers(const Json& value, std::uint32_t layers) {
    const auto& ids = member(value, "local_layer_ids");
    if (!ids.is_array() || ids.size() > layers) {
        throw ArtifactError("local_layer_ids must list at most one id per layer");
    }
    std::vector<bool> sliding(layers, false);
    std::set<std::uint32_t> seen;
    for (const auto& id : ids) {
        const auto layer = integer(id, "local_layer_ids entry", false);
        if (layer >= layers) { throw ArtifactError("local_layer_ids entry exceeds layer count"); }
        if (!seen.insert(layer).second) {
            throw ArtifactError("local_layer_ids repeats a layer");
        }
        sliding[layer] = true;
    }
    return sliding;
}

LoopConfig loop(const Json& value, std::uint32_t layers) {
    require_members(value, {"core_start", "core_end", "num_passes", "gate_rank"}, {"is_varloop"},
                    "loop_config");
    LoopConfig out;
    out.core_start = dimension(value, "core_start", false);
    out.core_end   = dimension(value, "core_end", false);
    out.num_passes = dimension(value, "num_passes");
    out.gate_rank  = dimension(value, "gate_rank");
    if (out.core_start >= out.core_end || out.core_end > layers) {
        throw ArtifactError("loop core must be a nonempty layer range inside the model");
    }
    if (out.gate_rank == 0) { throw ArtifactError("loop gate_rank must be positive"); }
    if (out.core_layers() % 6 != 0) {
        throw ArtifactError("loop core must contain whole attention-residual blocks");
    }
    return out;
}

AttnResConfig attn_res(const Json& value) {
    AttnResConfig out;
    out.layers_per_block = dimension(value, "n_layers_per_attn_res_block");
    out.normalize_values = flag(value, "attn_res_normalize_values");
    if (out.layers_per_block == 0) { throw ArtifactError("attention residual block is empty"); }
    return out;
}

MoeConfig moe(const Json& value) {
    MoeConfig out;
    out.num_experts         = dimension(value, "n_routed_experts");
    out.num_experts_per_tok = dimension(value, "num_experts_per_tok");
    out.intermediate_size   = dimension(value, "intermediate_size");
    out.route_scale         = positive_float(value, "route_scale");
    out.gate_softcap        = positive_float(value, "mlp_gate_softcap");
    out.up_softcap          = positive_float(value, "mlp_up_softcap");
    out.gate_bias           = flag(value, "use_gate_bias");
    if (dimension(value, "n_shared_experts", false) != 0) {
        throw ArtifactError("Inkling 2 Mini has no shared expert");
    }
    if (out.num_experts_per_tok > out.num_experts) {
        throw ArtifactError("selected experts exceed expert count");
    }
    return out;
}

VisionConfig vision(const Json& value) {
    require_members(value,
                    {"text_hidden_size", "projection_hidden_dim", "hidden_size",
                     "num_hidden_layers", "num_attention_heads", "native_patch_size",
                     "adapter_merge_size", "temporal_patch_size", "posemb_grid_size", "mlp_ratio",
                     "layer_norm_eps"},
                    {"model_type", "projector_bias"}, "vision");
    VisionConfig out;
    out.text_hidden_size    = dimension(value, "text_hidden_size");
    out.num_hidden_layers   = dimension(value, "num_hidden_layers");
    out.hidden_size         = dimension(value, "hidden_size");
    out.num_attention_heads = dimension(value, "num_attention_heads");
    out.native_patch_size   = dimension(value, "native_patch_size");
    out.adapter_merge_size  = dimension(value, "adapter_merge_size");
    out.temporal_patch_size = dimension(value, "temporal_patch_size");
    out.posemb_grid_size    = dimension(value, "posemb_grid_size");
    out.projection_hidden   = dimension(value, "projection_hidden_dim");
    out.mlp_ratio           = positive_float(value, "mlp_ratio");
    out.layer_norm_eps      = positive_float(value, "layer_norm_eps");
    if (out.hidden_size % out.num_attention_heads) {
        throw ArtifactError("vision width must divide by its head count");
    }
    return out;
}

MtpConfig mtp(const Json& value) {
    require_members(value,
                    {"num_hidden_layers", "block_size", "markov_rank", "intermediate_size",
                     "target_layer_ids", "local_layer_ids", "sliding_window_size"},
                    {"hidden_size", "head_dim", "num_attention_heads", "num_key_value_heads",
                     "max_position_embeddings", "q_bias", "o_bias", "rms_norm_eps", "rope_theta",
                     "markov_head_type", "torch_dtype"},
                    "DSpark config");
    MtpConfig out;
    out.num_hidden_layers = dimension(value, "num_hidden_layers");
    out.block_size        = dimension(value, "block_size");
    out.markov_rank       = dimension(value, "markov_rank");
    out.intermediate_size = dimension(value, "intermediate_size");
    if (dimension(value, "sliding_window_size") == 0) {
        throw ArtifactError("DSpark sliding window must be positive");
    }
    const auto& taps = member(value, "target_layer_ids");
    if (!taps.is_array() || taps.empty()) {
        throw ArtifactError("DSpark must read at least one target visit");
    }
    std::uint32_t previous = 0;
    for (const auto& tap : taps) {
        const auto visit = integer(tap, "DSpark target visit", false);
        if (visit < previous) { throw ArtifactError("DSpark target visits must ascend"); }
        previous = visit;
        out.target_visits.push_back(visit);
    }
    const auto& locals = member(value, "local_layer_ids");
    if (!locals.is_array()) { throw ArtifactError("DSpark local_layer_ids must be an array"); }
    for (const auto& id : locals) {
        const auto layer = integer(id, "DSpark local layer", false);
        if (layer >= out.num_hidden_layers) {
            throw ArtifactError("DSpark local layer exceeds its depth");
        }
        out.local_layers.push_back(layer);
    }
    return out;
}

} // namespace

std::uint32_t Config::visits() const noexcept {
    const std::uint32_t core = loop.core_layers();
    return (num_hidden_layers - core) + core * loop.num_passes;
}

std::uint32_t Config::global_layers() const noexcept {
    return std::uint32_t(std::count(sliding_layer.begin(), sliding_layer.end(), false));
}

Config parse_config(const artifact::Directory& directory) {
    try {
        const auto& component = directory.component("text");
        // `target` is optional in the container and means "this component consumes that one";
        // the text backbone is the target, so it either omits it or names itself.
        if (component.target && *component.target != "text") {
            throw ArtifactError("text component must not target another component");
        }
        const Json& config = component.config;
        // The six optional members are accepted because the checkpoint carries them, but nothing
        // in this architecture reads them: the sliding-window layers share the main head geometry
        // (`swa_*`), the projections are bias-free (`q_bias`, `o_bias`), and `torch_dtype` is the
        // training dtype, not an inference contract.
        require_members(config,
                        {"architectures", "model_type", "hidden_size", "num_hidden_layers",
                         "vocab_size", "unpadded_vocab_size", "model_max_length", "rms_norm_eps",
                         "final_logit_softcapping", "num_attention_heads", "num_key_value_heads",
                         "head_dim", "d_rel", "rel_extent", "sconv_kernel_size",
                         "sliding_window_size", "qk_l2_clip_threshold", "rel_l2_clip_threshold",
                         "log_scaling_n_floor", "log_scaling_alpha", "use_value_norm_no_gamma",
                         "local_layer_ids", "loop_config", "n_layers_per_attn_res_block",
                         "attn_res_normalize_values", "n_routed_experts", "num_experts_per_tok",
                         "n_shared_experts", "intermediate_size", "route_scale",
                         "mlp_gate_softcap", "mlp_up_softcap", "use_gate_bias"},
                        {"swa_head_dim", "swa_num_attention_heads", "swa_num_key_value_heads",
                         "q_bias", "o_bias", "torch_dtype"},
                        "text config");
        if (architecture(config) != "Inkling2LoopForConditionalGeneration") {
            throw ArtifactError("architecture is not Inkling2LoopForConditionalGeneration");
        }
        if (member(config, "model_type").get<std::string>() != "inkling2_loop_text") {
            throw ArtifactError("model_type is not inkling2_loop_text");
        }
        Config out;
        out.hidden_size       = dimension(config, "hidden_size");
        out.num_hidden_layers = dimension(config, "num_hidden_layers");
        out.vocab_size        = dimension(config, "vocab_size");
        out.unpadded_vocab    = dimension(config, "unpadded_vocab_size");
        out.max_context       = artifact::require_u64(member(config, "model_max_length"), "model_max_length", true);
        out.rms_norm_eps        = positive_float(config, "rms_norm_eps");
        out.final_logit_softcap = positive_float(config, "final_logit_softcapping");
        if (out.unpadded_vocab > out.vocab_size) {
            throw ArtifactError("padded vocabulary is smaller than the unpadded one");
        }
        // Each phase carries its own label so a malformed member names the block it belongs to.
        const auto phase = [](const char* label, auto&& read) {
            try {
                return read();
            } catch (const std::exception& error) {
                throw ArtifactError(std::string(label) + ": " + error.what());
            }
        };
        out.attention     = phase("attention", [&] { return attention(config, out.num_hidden_layers); });
        out.sliding_layer = phase("attention", [&] { return sliding_layers(config, out.num_hidden_layers); });
        out.loop     = phase("loop", [&] { return loop(member(config, "loop_config"), out.num_hidden_layers); });
        out.attn_res = phase("attention residual", [&] { return attn_res(config); });
        out.moe      = phase("MoE", [&] { return moe(config); });
        if (out.loop.core_layers() % out.attn_res.layers_per_block != 0) {
            throw ArtifactError("loop core does not split into whole attention-residual blocks");
        }
        if (out.visits() != 48) {
            throw ArtifactError("Inkling 2 Mini must produce 48 layer visits");
        }
        if (directory.components.contains("vision")) {
            const auto& vision_component = directory.components.at("vision");
            out.vision = vision(vision_component.config);
            if (out.vision.text_hidden_size != out.hidden_size) {
                throw ArtifactError("vision text width differs from the text hidden size");
            }
        }
        if (directory.components.contains("mtp")) {
            const auto& mtp_component = directory.components.at("mtp");
            out.mtp = mtp(mtp_component.config);
            if (!out.mtp.target_visits.empty() &&
                out.mtp.target_visits.back() > out.visits()) {
                throw ArtifactError("DSpark target visit exceeds the visit count");
            }
        }
        return out;
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("Inkling2 config: ") + error.what());
    }
}

} // namespace ninfer::models::inkling2
