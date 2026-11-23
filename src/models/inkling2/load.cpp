#include "models/inkling2/load.h"

#include "artifact/schema.h"

#include <set>
#include <sstream>
#include <string_view>

namespace ninfer::models::inkling2 {
namespace {

using artifact::ArtifactError;
using artifact::Binding;
using artifact::Directory;
using artifact::TensorObject;

constexpr std::string_view kBf16 = "bf16";
constexpr std::string_view kFp32 = "fp32";
constexpr std::string_view kNvfp4 = "nvfp4";
constexpr std::string_view kQ8    = "q8_g32_fp16";

std::string layer_prefix(std::uint32_t layer) {
    return "text/layers/" + std::to_string(layer) + "/";
}

std::string join_shape(const std::vector<std::uint64_t>& shape) {
    std::ostringstream text;
    text << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) { text << ','; }
        text << shape[i];
    }
    text << ']';
    return text.str();
}

struct Builder {
    std::vector<Slot> slots;

    void add(std::string name, std::vector<std::uint64_t> shape, std::string_view format) {
        slots.push_back(Slot{std::move(name), std::move(shape), std::string(format)});
    }
};

} // namespace

std::vector<Slot> expected_slots(const Config& config) {
    const std::uint64_t hidden = config.hidden_size;
    const std::uint64_t query  = config.attention.query_width();
    const std::uint64_t key    = config.attention.key_width();
    const std::uint64_t heads  = config.attention.num_attention_heads;
    const std::uint64_t d_rel  = config.attention.d_rel;
    const std::uint64_t kernel = config.attention.sconv_kernel;
    const std::uint64_t inter  = config.moe.intermediate_size;
    const std::uint64_t rank   = config.loop.gate_rank;
    const std::uint64_t experts = config.moe.num_experts;

    Builder out;

    // Text embeddings and the two end-of-model norms.
    out.add("text/embed", {config.vocab_size, hidden}, kBf16);
    out.add("text/embed_norm", {hidden}, kBf16);
    out.add("text/norm", {hidden}, kBf16);
    out.add("text/unembed", {config.vocab_size, hidden}, kBf16);

    // Block attention residuals: one query per layer, read by the block's residual mix.
    for (std::uint32_t layer = 0; layer + 1 < config.num_hidden_layers; ++layer) {
        // The checkpoint gives the layer before each global layer three rows instead of one.
        const std::uint64_t rows = (layer % config.attn_res.layers_per_block + 1 ==
                                    config.attn_res.layers_per_block - 1)
                                       ? 3
                                       : 1;
        out.add("text/attn_res/attn_queries/" + std::to_string(layer),
                rows == 1 ? std::vector<std::uint64_t>{hidden}
                          : std::vector<std::uint64_t>{rows, hidden},
                kFp32);
    }
    for (std::uint32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        out.add("text/attn_res/mlp_queries/" + std::to_string(layer), {hidden}, kFp32);
    }
    out.add("text/attn_res/output_query", {hidden}, kFp32);

    // Cross-pass message reads: two tensors of two input roles each, one row per core layer.
    const std::uint64_t core = config.loop.core_layers();
    out.add("text/middle_cycle_cross_pass_queries", {2 * core, hidden}, kFp32);
    out.add("text/middle_cycle_cross_pass_kv_queries", {2 * core, hidden}, kFp32);

    for (std::uint32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = layer_prefix(layer);

        // Attention: u/d-factorised projections, relative-bias path, sink, gate, clips, convolutions.
        out.add(prefix + "attn/wq_du", {query, hidden}, kBf16);
        out.add(prefix + "attn/wk_dv", {key, hidden}, kBf16);
        out.add(prefix + "attn/wv_dv", {key, hidden}, kBf16);
        out.add(prefix + "attn/wo_ud", {hidden, query}, kBf16);
        out.add(prefix + "attn/wr_du", {heads * d_rel, hidden}, kBf16);
        out.add(prefix + "attn/gate_attn", {query, hidden}, kBf16);
        out.add(prefix + "attn/attn_sink_proj", {heads, hidden}, kBf16);
        out.add(prefix + "attn/rel_logits_proj/proj", {d_rel, config.attention.rel_extent}, kBf16);
        out.add(prefix + "attn/q_l2_clip/scale", {heads, config.attention.head_dim}, kBf16);
        out.add(prefix + "attn/rel_l2_clip/scale", {heads, d_rel}, kBf16);
        // The short convolutions differ by attention kind. A sliding-window layer filters k and v
        // directly with a depthwise kernel; a global layer runs the same convolution in a learned
        // 32-wide space (`weight_proj`) and applies a fixed 4-tap filter (`static_w`), which is why
        // it also carries separate q/k/v input norms instead of one `attn_norm`.
        if (config.sliding_layer[layer]) {
            out.add(prefix + "attn/k_sconv", {key, 1, kernel}, kBf16);
            out.add(prefix + "attn/v_sconv", {key, 1, kernel}, kBf16);
            out.add(prefix + "attn_norm", {hidden}, kBf16);
        } else {
            out.add(prefix + "attn/k_sconv/static_w", {kernel, key}, kBf16);
            out.add(prefix + "attn/k_sconv/weight_proj", {config.attention.num_key_value_heads * kernel, hidden}, kBf16);
            out.add(prefix + "attn/v_sconv/static_w", {kernel, key}, kBf16);
            out.add(prefix + "attn/v_sconv/weight_proj", {config.attention.num_key_value_heads * kernel, hidden}, kBf16);
            out.add(prefix + "q_input_norm", {hidden}, kBf16);
            out.add(prefix + "k_input_norm", {hidden}, kBf16);
            out.add(prefix + "v_input_norm", {hidden}, kBf16);
        }
        out.add(prefix + "attn_sconv", {hidden, 1, kernel}, kBf16);

        out.add(prefix + "mlp_norm", {hidden}, kBf16);
        out.add(prefix + "mlp_sconv", {hidden, 1, kernel}, kBf16);
        out.add(prefix + "mlp/gate", {experts, hidden}, kBf16);
        out.add(prefix + "mlp/gate/bias", {experts}, kFp32);

        // Routed experts: fused gate+up and down, one parameter each.
        const std::string_view expert_format = layer < kPassThroughExperts ? kNvfp4 : kQ8;
        for (std::uint32_t expert = 0; expert < experts; ++expert) {
            const std::string base = prefix + "mlp/experts/" + std::to_string(expert) + "/";
            out.add(base + "w13", {2 * inter, hidden}, expert_format);
            out.add(base + "w2", {hidden, inter}, expert_format);
        }

        // Loop controllers exist only inside the core, one per sublayer input.
        if (config.loop.in_core(layer)) {
            for (std::string_view input : {"attn", "k", "v", "mlp"}) {
                const std::string base = prefix + "controllers/" + std::string(input) + "/";
                out.add(base + "down_weight", {rank, 2 * hidden}, kBf16);
                out.add(base + "forget_weight", {hidden, rank}, kBf16);
                out.add(base + "gain_weight", {hidden, rank}, kBf16);
            }
        }
    }
    return out.slots;
}

std::vector<BoundSlot> bind_parameters(const Directory& directory, const Config& config) {
    const auto slots = expected_slots(config);
    std::vector<BoundSlot> bound;
    bound.reserve(slots.size());
    for (const auto& slot : slots) {
        const auto binding = directory.bindings.find(slot.name);
        if (binding == directory.bindings.end()) {
            throw ArtifactError("Inkling2: missing parameter " + slot.name);
        }
        const Binding& entry = binding->second;
        if (!entry.whole_object || entry.parts.size() != 1) {
            throw ArtifactError("Inkling2: " + slot.name +
                                " must bind exactly one whole object");
        }
        const TensorObject& object = directory.tensor(entry.parts.front().object);
        if (object.shape != slot.shape) {
            throw ArtifactError("Inkling2: " + slot.name + " stores " + join_shape(object.shape) +
                                ", expected " + join_shape(slot.shape));
        }
        if (object.format != slot.format) {
            throw ArtifactError("Inkling2: " + slot.name + " is stored as " + object.format +
                                ", expected " + slot.format);
        }
        bound.push_back(BoundSlot{slot.name, slot.shape, slot.format, entry.parts.front().object});
    }
    return bound;
}

QType weight_type(std::string_view format) {
    if (format == kNvfp4) { return QType::NVFP4; }
    if (format == kQ8) { return QType::Q8_G32_FP16; }
    if (format == kBf16) { return QType::BF16; }
    if (format == kFp32) { return QType::FP32; }
    throw ArtifactError("Inkling2: unsupported stored format " + std::string(format));
}

void validate_parameters(const Directory& directory, const Config& config) {
    const auto bound = bind_parameters(directory, config);
    std::set<std::string, std::less<>> known;
    for (const auto& slot : bound) { known.insert(slot.name); }
    for (const auto& [name, binding] : directory.bindings) {
        (void)binding;
        if (name.starts_with("text/") && !known.contains(name)) {
            throw ArtifactError("Inkling2: unexpected text parameter " + name);
        }
    }
}

} // namespace ninfer::models::inkling2
