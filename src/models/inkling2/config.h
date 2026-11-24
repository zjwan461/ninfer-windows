#pragma once

#include "models/registry.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models::inkling2 {

// Attention geometry shared by every layer visit. Inkling 2 Mini is not RoPE: q and k carry an
// L2 clip, and the position signal enters as a learned relative-bias logit of rank `d_rel` over a
// `rel_extent` window. The projections are factorised (`wq_du`, `wk_dv`, ...); that factorisation
// is execution, so only the widths the kernels need live here.
struct AttentionConfig {
    std::uint32_t num_attention_heads = 0;
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim            = 0;
    std::uint32_t d_rel               = 0;
    std::uint32_t rel_extent          = 0;
    std::uint32_t sconv_kernel        = 0;
    std::uint32_t sliding_window      = 0;
    float qk_l2_clip_threshold        = 0;
    float rel_l2_clip_threshold       = 0;
    float log_scaling_floor           = 0;
    float log_scaling_alpha           = 0;
    bool value_norm_without_gamma     = false;

    [[nodiscard]] std::uint64_t query_width() const noexcept {
        return std::uint64_t(num_attention_heads) * head_dim;
    }

    [[nodiscard]] std::uint64_t key_width() const noexcept {
        return std::uint64_t(num_key_value_heads) * head_dim;
    }
};

// The loop core runs `num_passes` times with shared weights; every pass starts from the prelude
// output and sees earlier passes only through the rank-`gate_rank` controllers.
struct LoopConfig {
    std::uint32_t core_start = 0;
    std::uint32_t core_end   = 0; // exclusive
    std::uint32_t num_passes = 0;
    std::uint32_t gate_rank  = 0;

    [[nodiscard]] std::uint32_t core_layers() const noexcept { return core_end - core_start; }
    [[nodiscard]] bool in_core(std::uint32_t layer) const noexcept {
        return layer >= core_start && layer < core_end;
    }
};

// Residual-stream renormalisation every `layers_per_block` layers (block attention residuals).
struct AttnResConfig {
    std::uint32_t layers_per_block = 0;
    bool normalize_values          = false;
};

// 128 routed experts, top-8, no shared expert, gated SwiGLU with soft caps and a route scale.
struct MoeConfig {
    std::uint32_t num_experts         = 0;
    std::uint32_t num_experts_per_tok = 0;
    std::uint32_t intermediate_size   = 0;
    float route_scale                 = 0;
    float gate_softcap                = 0;
    float up_softcap                  = 0;
    bool gate_bias                    = false;

    [[nodiscard]] std::uint64_t w13_rows(std::uint64_t hidden) const noexcept {
        return 3 * std::uint64_t(intermediate_size) * hidden; // gate and up, fused
    }
};

struct VisionConfig {
    std::uint32_t text_hidden_size     = 0;
    std::uint32_t num_hidden_layers    = 0;
    std::uint32_t hidden_size          = 0;
    std::uint32_t num_attention_heads  = 0;
    std::uint32_t native_patch_size    = 0;
    std::uint32_t adapter_merge_size   = 0;
    std::uint32_t temporal_patch_size  = 0;
    std::uint32_t posemb_grid_size     = 0;
    std::uint32_t projection_hidden    = 0;
    float mlp_ratio                    = 0;
    float layer_norm_eps               = 0;
};

// DSpark drafter: three layers reading four target visits, with a rank-`markov_rank` head.
struct MtpConfig {
    std::uint32_t num_hidden_layers   = 0;
    std::uint32_t block_size          = 0;
    std::uint32_t markov_rank         = 0;
    std::uint32_t intermediate_size   = 0;
    std::vector<std::uint32_t> target_visits;
    std::vector<std::uint32_t> local_layers;
};

struct Config {
    std::uint32_t hidden_size       = 0;
    std::uint32_t num_hidden_layers = 0;
    std::uint32_t vocab_size        = 0;
    std::uint32_t unpadded_vocab    = 0;
    std::uint64_t max_context       = 0;
    float rms_norm_eps              = 0;
    float final_logit_softcap       = 0;
    AttentionConfig attention;
    LoopConfig loop;
    AttnResConfig attn_res;
    MoeConfig moe;
    VisionConfig vision;
    MtpConfig mtp;
    // One entry per layer: true when the layer attends with the sliding window.
    std::vector<bool> sliding_layer;

    // Layer applications per token: prelude + coda once, the loop core `num_passes` times.
    [[nodiscard]] std::uint32_t visits() const noexcept;
    // One KV cache and one conv-state cache per visit, not per layer.
    [[nodiscard]] std::uint32_t kv_caches() const noexcept { return visits(); }
    [[nodiscard]] std::uint32_t global_layers() const noexcept;
};

// Validate the `text` component of an artifact against this architecture.
[[nodiscard]] Config parse_config(const artifact::Directory& directory);

} // namespace ninfer::models::inkling2
