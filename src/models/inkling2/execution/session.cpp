// Inkling-2-Mini decode: the 48-visit chain over 24 layers, with a cache per visit.
#include "models/inkling2/execution/session.h"

#include "models/inkling2/execution/weight_table.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/q8/q8_launch.h"

#include "models/inkling2/materialize.h"
#include "models/inkling2/schedule.h"

#include "ninfer/ops/block_residual_read.h"
#include "ninfer/ops/causal_dconv_residual.h"
#include "ninfer/ops/dynamic_short_conv.h"
#include "ninfer/ops/head_rmsnorm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/logits_softcap.h"
#include "ninfer/ops/message_merge.h"
#include "ninfer/ops/moe_router.h"
#include "ninfer/ops/relbias_attention.h"
#include "ninfer/ops/residual.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scaled_l2_clip.h"
#include "ninfer/ops/softcapped_swiglu.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::inkling2::execution {

using namespace ninfer::ops;

namespace {

constexpr int kHidden  = 2304;
constexpr int kQ       = 3072;  // 24 heads of 128
constexpr int kKv      = 1024;  // 8 heads of 128
constexpr int kHeads   = 24;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kRel     = 384;   // 24 heads of 16
constexpr int kDRel    = 16;
constexpr int kInter   = 1152;
constexpr int kExperts = 128;
constexpr int kTopK    = 8;
constexpr int kStackLimit = 8;
constexpr int kConvState  = 3;  // a kernel of four keeps three previous values
constexpr float kEps          = 1e-6F;
constexpr float kQkThreshold  = 22.627416997969522F;  // sqrt(512)
constexpr float kRelThreshold = 8.0F;
constexpr float kLogAlpha     = 0.1F;
constexpr float kLogFloor     = 8192.0F;

std::uint16_t to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    return std::uint16_t((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

float from_bf16(std::uint16_t value) {
    const std::uint32_t bits = std::uint32_t(value) << 16;
    float out = 0.0F;
    std::memcpy(&out, &bits, 4);
    return out;
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling2 session: ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

void* allocate(std::size_t bytes) {
    void* device = nullptr;
    check(cudaMalloc(&device, bytes), "cudaMalloc");
    check(cudaMemset(device, 0, bytes), "cudaMemset");
    return device;
}

Tensor vec(void* data, int width, DType dtype = DType::BF16) {
    return Tensor(data, dtype, {width, 1});
}

} // namespace

struct Session::Impl {
    TextWeights* weights = nullptr;
    const Config* config = nullptr;
    Schedule schedule;
    SessionOptions options;
    int width = 0;  // number of visits, the number of caches

    // Working buffers, allocated once.
    void* h = nullptr;
    void* h_in = nullptr;
    void* attn_in = nullptr;
    void* q_in = nullptr;
    void* k_in = nullptr;
    void* v_in = nullptr;
    void* qn = nullptr;
    void* kn = nullptr;
    void* vn = nullptr;
    void* q = nullptr;
    void* q_clip = nullptr;
    void* k = nullptr;
    void* v = nullptr;
    void* k_conv = nullptr;
    void* v_conv = nullptr;
    void* k_clip = nullptr;
    void* v_clip = nullptr;
    void* dyn = nullptr;
    void* rel = nullptr;
    void* rel_clip = nullptr;
    void* sink = nullptr;
    void* gate = nullptr;
    void* attn = nullptr;
    void* proj = nullptr;
    void* attn_conv = nullptr;
    void* mlp_in = nullptr;
    void* gated = nullptr;
    void* forget = nullptr;
    void* gain = nullptr;
    void* controller = nullptr;
    void* gate_up = nullptr;
    void* act = nullptr;
    void* down = nullptr;
    void* mlp_conv = nullptr;
    void* moe_acc = nullptr;
    void* residual = nullptr;
    void* stack = nullptr;
    void* message = nullptr;
    void* msg_stack = nullptr;
    void* rms_x = nullptr;
    void* rms_m = nullptr;
    void* ones = nullptr;
    void* indices = nullptr;
    void* router_weights = nullptr;
    void* routing_host = nullptr;  // pinned: the readback is 64 bytes and pageable copies stage
    // Device-side expert tables and the host templates that describe their geometry. The template
    // is a real `matrix_weight` result rather than hand-filled fields, so layout, grouping and the
    // container's divisor come from the same place the direct route reads them.
    std::vector<void*> expert_w13_table;
    std::vector<void*> expert_w2_table;
    std::vector<Weight> expert_w13_template;
    std::vector<Weight> expert_w2_template;
    void* position = nullptr;
    void* previous_h = nullptr;
    void* previous_block = nullptr;
    void* pre_loop_h = nullptr;
    void* pre_loop_stack = nullptr;
    void* msg_query = nullptr;   // four FP32 message query rows
    void* concat = nullptr;      // the message gate's two normalised streams, side by side
    void* logits = nullptr;
    void* embed_row = nullptr;

    // Caches, one slot per visit.
    void* k_cache = nullptr;
    void* v_cache = nullptr;
    void* k_state = nullptr;     // conv windows
    void* v_state = nullptr;
    void* attn_state = nullptr;
    void* mlp_state = nullptr;
    void* state_scratch = nullptr;
    std::vector<void*> router_f32;  // per layer: the router widened to FP32 once

    int position_now = 0;
    int stack_length = 0;
    std::string context_label;
};



Session::Session(TextWeights& weights, const Config& config, DeviceContext& device,
                 SessionOptions options) {
    (void)device;
    if (options.max_context <= 0) {
        throw std::invalid_argument("inkling2 session: max_context must be positive");
    }
    impl_ = std::make_unique<Impl>();
    Impl& s = *impl_;
    s.weights = &weights;
    s.config = &config;
    s.schedule = build_schedule(config);
    s.options = options;
    s.width = static_cast<int>(s.schedule.visits.size());

    const auto context = std::size_t(options.max_context);
    const auto hidden_bytes = std::size_t(kHidden) * 2;

    s.h = allocate(hidden_bytes);
    s.h_in = allocate(hidden_bytes);
    s.attn_in = allocate(hidden_bytes);
    s.q_in = allocate(hidden_bytes);
    s.k_in = allocate(hidden_bytes);
    s.v_in = allocate(hidden_bytes);
    s.qn = allocate(hidden_bytes);
    s.kn = allocate(hidden_bytes);
    s.vn = allocate(hidden_bytes);
    s.q = allocate(std::size_t(kQ) * 2);
    s.q_clip = allocate(std::size_t(kQ) * 2);
    s.k = allocate(std::size_t(kKv) * 2);
    s.v = allocate(std::size_t(kKv) * 2);
    s.k_conv = allocate(std::size_t(kKv) * 2);
    s.v_conv = allocate(std::size_t(kKv) * 2);
    s.k_clip = allocate(std::size_t(kKv) * 2);
    s.v_clip = allocate(std::size_t(kKv) * 2);
    s.dyn = allocate(std::size_t(kKv) * 2);
    s.rel = allocate(std::size_t(kRel) * 2);
    s.rel_clip = allocate(std::size_t(kRel) * 2);
    s.sink = allocate(std::size_t(kHeads) * 2);
    s.gate = allocate(std::size_t(kQ) * 2);
    s.attn = allocate(std::size_t(kQ) * 2);
    s.proj = allocate(hidden_bytes);
    s.attn_conv = allocate(hidden_bytes);
    s.mlp_in = allocate(hidden_bytes);
    s.gated = allocate(hidden_bytes);
    s.forget = allocate(hidden_bytes);
    s.gain = allocate(hidden_bytes);
    s.controller = allocate(std::size_t(64) * 2);
    s.gate_up = allocate(std::size_t(2 * kInter) * 2);
    s.act = allocate(std::size_t(kInter) * 2);
    s.down = allocate(hidden_bytes);
    s.mlp_conv = allocate(hidden_bytes);
    s.moe_acc = allocate(std::size_t(kHidden) * 4);
    s.residual = allocate(std::size_t(kHidden) * 4);
    s.stack = allocate(std::size_t(kStackLimit) * hidden_bytes);
    s.message = allocate(std::size_t(4) * hidden_bytes);
    s.msg_stack = allocate(std::size_t(2) * hidden_bytes);
    s.msg_query = allocate(std::size_t(4) * kHidden * 4);
    s.concat = allocate(std::size_t(2) * kHidden * 2);
    s.rms_x = allocate(hidden_bytes);
    s.rms_m = allocate(hidden_bytes);
    s.ones = allocate(std::size_t(kKv) * 2);
    s.indices = allocate(std::size_t(kTopK) * sizeof(std::int32_t));
    s.router_weights = allocate(std::size_t(kTopK) * sizeof(float));
    // The selection is the one readback a step keeps, so its host buffer is pinned: a pageable
    // device-to-host copy stages through the driver and cannot overlap the transfer.
    {
        const int layers = static_cast<int>(weights.index.layers.size());
        s.expert_w13_table.assign(static_cast<std::size_t>(layers), nullptr);
        s.expert_w2_table.assign(static_cast<std::size_t>(layers), nullptr);
        s.expert_w13_template.assign(static_cast<std::size_t>(layers), Weight{});
        s.expert_w2_template.assign(static_cast<std::size_t>(layers), Weight{});
        for (int layer_index = 0; layer_index < layers; ++layer_index) {
            const auto& layer = weights.index.layers[static_cast<std::size_t>(layer_index)];
            std::vector<Weight> w13(static_cast<std::size_t>(kExperts));
            std::vector<Weight> w2(static_cast<std::size_t>(kExperts));
            for (int expert = 0; expert < kExperts; ++expert) {
                w13[static_cast<std::size_t>(expert)] =
                    matrix_weight(weights, layer.expert_w13[static_cast<std::size_t>(expert)]);
                w2[static_cast<std::size_t>(expert)] =
                    matrix_weight(weights, layer.expert_w2[static_cast<std::size_t>(expert)]);
            }
            s.expert_w13_template[static_cast<std::size_t>(layer_index)] = w13.front();
            s.expert_w2_template[static_cast<std::size_t>(layer_index)] = w2.front();
            s.expert_w13_table[static_cast<std::size_t>(layer_index)] =
                upload_weight_table(w13, nullptr);
            s.expert_w2_table[static_cast<std::size_t>(layer_index)] =
                upload_weight_table(w2, nullptr);
        }
    }
    check(cudaMallocHost(&s.routing_host, std::size_t(kTopK) * (sizeof(std::int32_t) + sizeof(float))),
          "routing host");
    s.position = allocate(sizeof(std::int32_t));
    s.previous_h = allocate(hidden_bytes);
    s.previous_block = allocate(hidden_bytes);
    s.pre_loop_h = allocate(hidden_bytes);
    s.pre_loop_stack = allocate(std::size_t(kStackLimit) * hidden_bytes);
    s.embed_row = allocate(hidden_bytes);

    const auto unembed = matrix_weight(weights, weights.index.unembed);
    s.logits = allocate(std::size_t(unembed.n) * 2);

    // One cache per visit: the loop core's passes must not share keys, values or conv windows.
    const auto kv_slot = context * std::size_t(kKvHeads) * std::size_t(kHeadDim) * 2;
    s.k_cache = allocate(std::size_t(s.width) * kv_slot);
    s.v_cache = allocate(std::size_t(s.width) * kv_slot);
    const auto state_bytes = context == 0 ? 0 : 1;

    s.k_state = allocate(std::size_t(s.width) * kKv * kConvState * 2);
    s.v_state = allocate(std::size_t(s.width) * kKv * kConvState * 2);
    s.attn_state = allocate(std::size_t(s.width) * kHidden * kConvState * 2);
    s.mlp_state = allocate(std::size_t(s.width) * kHidden * kConvState * 2);
    s.state_scratch = allocate(std::size_t(std::max(kKv, kHidden)) * kConvState * 2);
    (void)state_bytes;

    // The ones vector is the key clip's scale: a single token's key is clipped against unit scale.
    std::vector<std::uint16_t> ones(std::size_t(kKv), to_bf16(1.0F));
    check(cudaMemcpy(s.ones, ones.data(), ones.size() * 2, cudaMemcpyHostToDevice), "ones");

    // The router is BF16 in the checkpoint and FP32 in the reference; widen once per layer.
    s.router_f32.assign(weights.index.layers.size(), nullptr);
    for (std::size_t index = 0; index < weights.index.layers.size(); ++index) {
        const auto slot = weights.index.layers[index][Role::MlpGate];
        if (slot < 0) { continue; }
        const auto gate = dense_tensor(weights, slot);
        std::vector<std::uint16_t> host(std::size_t(kExperts) * kHidden);
        check(cudaMemcpy(host.data(), gate.data, host.size() * 2, cudaMemcpyDeviceToHost), "router");
        std::vector<float> wide(host.size());
        for (std::size_t i = 0; i < host.size(); ++i) { wide[i] = from_bf16(host[i]); }
        void* device_gate = allocate(wide.size() * 4);
        check(cudaMemcpy(device_gate, wide.data(), wide.size() * 4, cudaMemcpyHostToDevice),
              "router");
        s.router_f32[index] = device_gate;
    }
}

Session::~Session() {
    if (!impl_) { return; }
    Impl& s = *impl_;
    for (void* pointer : {s.h, s.h_in, s.attn_in, s.q_in, s.k_in, s.v_in, s.qn, s.kn, s.vn, s.q,
                          s.q_clip, s.k, s.v, s.k_conv, s.v_conv, s.k_clip, s.v_clip, s.dyn, s.rel,
                          s.rel_clip, s.sink, s.gate, s.attn, s.proj, s.attn_conv, s.mlp_in,
                          s.gated, s.forget, s.gain, s.controller, s.gate_up, s.act, s.down,
                          s.mlp_conv, s.moe_acc, s.residual, s.stack, s.message, s.msg_stack,
                          s.rms_x, s.rms_m, s.msg_query, s.concat, s.ones, s.indices, s.router_weights, s.position,
                          s.previous_h, s.previous_block, s.pre_loop_h, s.pre_loop_stack,
                          s.logits, s.embed_row, s.k_cache, s.v_cache, s.k_state, s.v_state,
                          s.attn_state, s.mlp_state, s.state_scratch}) {
        if (pointer != nullptr) { cudaFree(pointer); }
    }
    if (s.routing_host != nullptr) { cudaFreeHost(s.routing_host); }
    for (void* pointer : s.router_f32) {
        if (pointer != nullptr) { cudaFree(pointer); }
    }
}

std::int32_t Session::position() const { return impl_->position_now; }
std::int32_t Session::max_context() const { return impl_->options.max_context; }

void Session::reset() {
    Impl& s = *impl_;
    s.position_now = 0;
    s.stack_length = 0;
    const auto kv_slot = std::size_t(s.options.max_context) * kKvHeads * kHeadDim * 2;
    check(cudaMemset(s.k_cache, 0, std::size_t(s.width) * kv_slot), "reset k");
    check(cudaMemset(s.v_cache, 0, std::size_t(s.width) * kv_slot), "reset v");
    check(cudaMemset(s.k_state, 0, std::size_t(s.width) * kKv * kConvState * 2), "reset ks");
    check(cudaMemset(s.v_state, 0, std::size_t(s.width) * kKv * kConvState * 2), "reset vs");
    check(cudaMemset(s.attn_state, 0, std::size_t(s.width) * kHidden * kConvState * 2), "reset as");
    check(cudaMemset(s.mlp_state, 0, std::size_t(s.width) * kHidden * kConvState * 2), "reset ms");
    check(cudaMemset(s.stack, 0, std::size_t(kStackLimit) * kHidden * 2), "reset stack");
}


Tensor Session::step(std::uint32_t token, cudaStream_t stream) {
    Impl& s = *impl_;
    TextWeights& weights = *s.weights;
    const Config& config = *s.config;
    const int stack_limit = kStackLimit;
    const int position_now = s.position_now;
    if (position_now >= s.options.max_context) {
        throw std::out_of_range("inkling2 session: context exhausted");
    }

    // The ops take their outputs by non-const reference, so the view has to outlive the call:
    // this holds it for the duration of the full expression.
    struct View {
        Tensor tensor;
        operator Tensor&() { return tensor; }
    };
    const auto vec = [](void* data, int width, DType dtype = DType::BF16) {
        return View{Tensor(data, dtype, {width, 1})};
    };
    const auto linear_role = [&](const LayerSlots& layer, Role role, void* input, int in_width,
                                 int out_width, void* out) {
        linear(vec(input, in_width), matrix_weight(weights, layer[role]), vec(out, out_width),
               stream);
    };
    // (1 - tanh(forget)) * x + tanh(gain) * message, with both gates read from both streams.
    const auto message_gate = [&](const LayerSlots& layer, Role down, Role forget_proj,
                                  Role gain_proj, void* x, void* message, void* out) {
        head_rmsnorm(vec(x, kHidden), kEps, vec(s.rms_x, kHidden), stream);
        head_rmsnorm(vec(message, kHidden), kEps, vec(s.rms_m, kHidden), stream);
        // The gate reads both normalised streams side by side, so this needs a buffer of
        // 2 * hidden: borrowing a hidden-sized one overruns it and quietly poisons its neighbour.
        cudaMemcpyAsync(s.concat, s.rms_x, kHidden * 2, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(static_cast<char*>(s.concat) + kHidden * 2, s.rms_m, kHidden * 2,
                        cudaMemcpyDeviceToDevice, stream);
        linear(vec(s.concat, 2 * kHidden), matrix_weight(weights, layer[down]),
               vec(s.controller, 64), stream);
        linear_role(layer, forget_proj, s.controller, 64, kHidden, s.forget);
        linear_role(layer, gain_proj, s.controller, 64, kHidden, s.gain);
        message_merge(vec(x, kHidden), vec(message, kHidden), vec(s.forget, kHidden),
                      vec(s.gain, kHidden), vec(out, kHidden), stream);
    };

    // ---- embedding -------------------------------------------------------------------------
    {
        auto embed = matrix_weight(weights, weights.index.embed);
        if (embed.qtype != QType::BF16) {
            throw std::runtime_error("inkling2 session: the embedding matrix must be BF16");
        }
        if (token >= static_cast<std::uint32_t>(embed.n)) {
            throw std::out_of_range("inkling2 session: token outside the vocabulary");
        }
        Tensor row(const_cast<void*>(static_cast<const void*>(
                       static_cast<const char*>(embed.qdata) + std::size_t(token) * embed.k * 2)),
                   DType::BF16, {embed.k, 1});
        rmsnorm(row, dense_tensor(weights, weights.index.embed_norm), kEps, false,
                vec(s.h, kHidden), stream);
    }

    int stack_length = 0;
    int pass_seen = -1;
    bool pre_loop_saved = false;

    const auto run_visit = [&](const Visit& visit, int stack_length_now) {
        const auto layer_index = static_cast<int>(visit.layer);
        const auto& layer = weights.index.layers[static_cast<std::size_t>(layer_index)];
        const bool sliding = config.sliding_layer[static_cast<std::size_t>(layer_index)];
        const bool boundary = (layer_index % 6) == 0;
        const bool uses_messages = visit.pass > 0 && visit.core;
        const int slot = static_cast<int>(visit.index);

        // The convolution ops write the shifted window to a scratch buffer; these views name it.
        Tensor state_kv(s.state_scratch, DType::BF16, {kKv, kConvState});
        Tensor state_hidden(s.state_scratch, DType::BF16, {kHidden, kConvState});
        cudaMemcpyAsync(s.h_in, s.h, kHidden * 2, cudaMemcpyDeviceToDevice, stream);
        if (stage_sink && boundary) {
            stage_sink(layer_index, "boundary_in", s.h_in, kHidden);
        }

        // ---- attention input: read out of the block stack -----------------------------------
        const int queries = sliding ? 1 : 3;
        if (layer_index > 0) {
            cudaMemcpyAsync(static_cast<char*>(s.stack) + stack_length_now * kHidden * 2, s.h_in,
                            kHidden * 2, cudaMemcpyDeviceToDevice, stream);
            const auto query_slot =
                weights.index.attn_queries[static_cast<std::size_t>(layer_index)];
            Tensor query(dense_tensor(weights, query_slot).data, DType::FP32, {kHidden, queries});
            Tensor states(s.stack, DType::BF16, {kHidden, stack_length_now + 1});
            if (stage_sink) {
                stage_sink(layer_index, "block_states", s.stack, kHidden * (stack_length_now + 1));
            }
            Tensor read_out(s.message, DType::BF16, {kHidden, queries});
            block_residual_read(query, states, kEps, read_out, stream);
            cudaMemcpyAsync(s.attn_in, s.message, kHidden * 2, cudaMemcpyDeviceToDevice, stream);
            if (queries == 3) {
                cudaMemcpyAsync(s.k_in, static_cast<char*>(s.message) + kHidden * 2, kHidden * 2,
                                cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(s.v_in, static_cast<char*>(s.message) + 2 * kHidden * 2,
                                kHidden * 2, cudaMemcpyDeviceToDevice, stream);
            }
        } else {
            cudaMemcpyAsync(s.attn_in, s.h_in, kHidden * 2, cudaMemcpyDeviceToDevice, stream);
        }
        if (stage_sink) { stage_sink(layer_index, "attn_in", s.attn_in, kHidden); }

        void* k_base = queries == 3 ? s.k_in : s.attn_in;
        void* v_base = queries == 3 ? s.v_in : s.attn_in;
        if (uses_messages) {
            // The messages come from the pass that just ended: its last block entry and its final
            // hidden state.
            cudaMemcpyAsync(s.msg_stack, s.previous_block, kHidden * 2, cudaMemcpyDeviceToDevice,
                            stream);
            cudaMemcpyAsync(static_cast<char*>(s.msg_stack) + kHidden * 2, s.previous_h,
                            kHidden * 2, cudaMemcpyDeviceToDevice, stream);
            const auto cp = dense_tensor(weights, weights.index.cross_pass_queries);
            const auto kv = dense_tensor(weights, weights.index.cross_pass_kv_queries);
            // The checkpoint keeps the four queries in two tensors of two rows per loop position,
            // merged as [query, key, value, mlp].
            const auto row = std::size_t(layer_index - 6) * 2;
            const auto row_bytes = std::size_t(kHidden) * 4;
            void* merged = s.msg_query;
            const auto* cp_base = static_cast<const char*>(cp.data);
            const auto* kv_base = static_cast<const char*>(kv.data);
            char* dst = static_cast<char*>(merged);
            cudaMemcpyAsync(dst, cp_base + row * row_bytes, row_bytes, cudaMemcpyDeviceToDevice,
                            stream);
            cudaMemcpyAsync(dst + row_bytes, kv_base + row * row_bytes, row_bytes,
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(dst + 2 * row_bytes, kv_base + (row + 1) * row_bytes, row_bytes,
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(dst + 3 * row_bytes, cp_base + (row + 1) * row_bytes, row_bytes,
                            cudaMemcpyDeviceToDevice, stream);
            Tensor message_query(merged, DType::FP32, {kHidden, 4});
            Tensor message_states(s.msg_stack, DType::BF16, {kHidden, 2});
            Tensor message_out(s.message, DType::BF16, {kHidden, 4});
            block_residual_read(message_query, message_states, kEps, message_out, stream);
            char* rows = static_cast<char*>(s.message);
            message_gate(layer, Role::ControllerAttnDown, Role::ControllerAttnForget,
                         Role::ControllerAttnGain, s.attn_in, rows, s.q_in);
            message_gate(layer, Role::ControllerKDown, Role::ControllerKForget,
                         Role::ControllerKGain, k_base, rows + kHidden * 2, s.k_in);
            message_gate(layer, Role::ControllerVDown, Role::ControllerVForget,
                         Role::ControllerVGain, v_base, rows + 2 * kHidden * 2, s.v_in);
        } else {
            cudaMemcpyAsync(s.q_in, s.attn_in, kHidden * 2, cudaMemcpyDeviceToDevice, stream);
            if (queries == 1) {
                cudaMemcpyAsync(s.k_in, s.attn_in, kHidden * 2, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(s.v_in, s.attn_in, kHidden * 2, cudaMemcpyDeviceToDevice, stream);
            }
        }

        // ---- per-stream norms ---------------------------------------------------------------
        if (sliding) {
            const auto norm = dense_tensor(weights, layer[Role::AttnNorm]);
            rmsnorm(vec(s.q_in, kHidden), norm, kEps, false, vec(s.qn, kHidden), stream);
            rmsnorm(vec(s.k_in, kHidden), norm, kEps, false, vec(s.kn, kHidden), stream);
            rmsnorm(vec(s.v_in, kHidden), norm, kEps, false, vec(s.vn, kHidden), stream);
        } else {
            rmsnorm(vec(s.q_in, kHidden), dense_tensor(weights, layer[Role::QInputNorm]), kEps,
                    false, vec(s.qn, kHidden), stream);
            rmsnorm(vec(s.k_in, kHidden), dense_tensor(weights, layer[Role::KInputNorm]), kEps,
                    false, vec(s.kn, kHidden), stream);
            rmsnorm(vec(s.v_in, kHidden), dense_tensor(weights, layer[Role::VInputNorm]), kEps,
                    false, vec(s.vn, kHidden), stream);
        }
        if (stage_sink) { stage_sink(layer_index, "ln", s.qn, kHidden); }

        // ---- query, key and value --------------------------------------------------------
        Tensor position_t(s.position, DType::I32, {1});
        cudaMemcpyAsync(s.position, &position_now, sizeof(std::int32_t), cudaMemcpyHostToDevice,
                        stream);
        ScaledL2ClipArgs q_args;
        q_args.rows = 1; q_args.heads = kHeads; q_args.head_dim = kHeadDim;
        q_args.threshold = kQkThreshold;
        q_args.tau_alpha = sliding ? 0.0F : kLogAlpha;
        q_args.tau_floor = kLogFloor;
        linear_role(layer, Role::WqDu, s.qn, kHidden, kQ, s.q);
        scaled_l2_clip(Tensor(s.q, DType::BF16, {kQ, 1}),
                       dense_tensor(weights, layer[Role::QkClip]),
                       sliding ? nullptr : &position_t, q_args, vec(s.q_clip, kQ), stream);
        cudaMemcpyAsync(s.q, s.q_clip, kQ * 2, cudaMemcpyDeviceToDevice, stream);
        if (stage_sink) { stage_sink(layer_index, "q", s.q, kQ); }

        linear_role(layer, Role::WkDv, s.kn, kHidden, kKv, s.k);
        linear_role(layer, Role::WvDv, s.vn, kHidden, kKv, s.v);
        for (int stream_index = 0; stream_index < 2; ++stream_index) {
            const bool is_key = stream_index == 0;
            void* base = is_key ? s.k : s.v;
            void* out = is_key ? s.k_conv : s.v_conv;
            void* state = is_key ? static_cast<char*>(s.k_state) + std::size_t(slot) * kKv * kConvState * 2
                                 : static_cast<char*>(s.v_state) + std::size_t(slot) * kKv * kConvState * 2;
            if (sliding) {
                auto conv = dense_tensor(weights, layer[is_key ? Role::KSconv : Role::VSconv]);
                Tensor conv2d(conv.data, DType::BF16, {kKv, 4});
                causal_dconv_residual(vec(base, kKv), conv2d, Tensor(state, DType::BF16, {kKv, kConvState}),
                                      state_kv, vec(out, kKv), stream);
            } else {
                const auto proj = matrix_weight(weights, layer[is_key ? Role::KSconvProj : Role::VSconvProj]);
                linear(vec(is_key ? s.kn : s.vn, kHidden), proj, vec(s.dyn, 32), stream);
                auto static_w = dense_tensor(weights, layer[is_key ? Role::KSconvStatic : Role::VSconvStatic]);
                Tensor static_4c(static_w.data, DType::BF16, {4, kKv});
                dynamic_short_conv(vec(base, kKv), static_4c, vec(s.dyn, 32),
                                   Tensor(state, DType::BF16, {kKv, kConvState}),
                                   state_kv, vec(out, kKv), kKvHeads, stream);
            }
            cudaMemcpyAsync(state, s.state_scratch, std::size_t(kKv) * kConvState * 2,
                            cudaMemcpyDeviceToDevice, stream);
        }
        ScaledL2ClipArgs k_args;
        k_args.rows = 1; k_args.heads = kKvHeads; k_args.head_dim = kHeadDim;
        k_args.threshold = kQkThreshold; k_args.tau_alpha = 0.0F; k_args.tau_floor = kLogFloor;
        scaled_l2_clip(Tensor(s.k_conv, DType::BF16, {kKv, 1}), Tensor(s.ones, DType::BF16, {kKvHeads, kHeadDim}),
                       nullptr, k_args, vec(s.k_clip, kKv), stream);
        cudaMemcpyAsync(s.k_conv, s.k_clip, kKv * 2, cudaMemcpyDeviceToDevice, stream);
        // The value stream is normalised per head, not per vector.
        {
            Tensor v_heads_in(s.v_conv, DType::BF16, {kHeadDim, kKvHeads});
            Tensor v_heads_out(s.v_clip, DType::BF16, {kHeadDim, kKvHeads});
            head_rmsnorm(v_heads_in, kEps, v_heads_out, stream);
        }
        cudaMemcpyAsync(s.v_conv, s.v_clip, kKv * 2, cudaMemcpyDeviceToDevice, stream);
        if (stage_sink) {
            stage_sink(layer_index, "k", s.k_conv, kKv);
            stage_sink(layer_index, "v", s.v_conv, kKv);
        }

        // ---- relative states, sink, gate -------------------------------------------------
        ScaledL2ClipArgs rel_args;
        rel_args.rows = 1; rel_args.heads = kHeads; rel_args.head_dim = kDRel;
        rel_args.threshold = kRelThreshold;
        rel_args.tau_alpha = sliding ? 0.0F : kLogAlpha;
        rel_args.tau_floor = kLogFloor;
        linear_role(layer, Role::WrDu, s.qn, kHidden, kRel, s.rel);
        scaled_l2_clip(Tensor(s.rel, DType::BF16, {kRel, 1}),
                       dense_tensor(weights, layer[Role::RelClip]),
                       sliding ? nullptr : &position_t, rel_args, vec(s.rel_clip, kRel), stream);
        cudaMemcpyAsync(s.rel, s.rel_clip, kRel * 2, cudaMemcpyDeviceToDevice, stream);
        linear_role(layer, Role::AttnSink, s.qn, kHidden, kHeads, s.sink);
        linear_role(layer, Role::GateAttn, s.qn, kHidden, kQ, s.gate);
        if (stage_sink) { stage_sink(layer_index, "sink", s.sink, kHeads); }

        // ---- attention over this visit's cache -------------------------------------------
        {
            const auto kv_slot = std::size_t(s.options.max_context) * kKvHeads * kHeadDim;
            char* k_base = static_cast<char*>(s.k_cache) + std::size_t(slot) * kv_slot * 2;
            char* v_base = static_cast<char*>(s.v_cache) + std::size_t(slot) * kv_slot * 2;
            for (int head = 0; head < kKvHeads; ++head) {
                const auto offset = (std::size_t(head) * s.options.max_context +
                                     std::size_t(position_now)) * kHeadDim * 2;
                cudaMemcpyAsync(k_base + offset,
                                static_cast<char*>(s.k_conv) + std::size_t(head) * kHeadDim * 2,
                                kHeadDim * 2, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(v_base + offset,
                                static_cast<char*>(s.v_conv) + std::size_t(head) * kHeadDim * 2,
                                kHeadDim * 2, cudaMemcpyDeviceToDevice, stream);
            }
            RelBiasAttentionArgs args;
            args.heads = kHeads; args.kv_heads = kKvHeads; args.head_dim = kHeadDim;
            args.q_pos = position_now;
            args.kv_len = s.options.max_context;
            args.window = sliding ? 1024 : 0;
            args.rel_extent = 1024; args.d_rel = kDRel;
            args.scale = 1.0F / std::sqrt(float(kHeadDim));
            relbias_attention(vec(s.q, kQ), Tensor(k_base, DType::BF16, {kHeadDim, kKvHeads * s.options.max_context}),
                              Tensor(v_base, DType::BF16, {kHeadDim, kKvHeads * s.options.max_context}),
                              vec(s.rel, kRel), dense_tensor(weights, layer[Role::RelLogits]),
                              vec(s.sink, kHeads), vec(s.gate, kQ), args, vec(s.attn, kQ), stream);
        }
        linear_role(layer, Role::WoUd, s.attn, kQ, kHidden, s.proj);
        if (stage_sink) { stage_sink(layer_index, "attnpre", s.proj, kHidden); }
        {
            char* state = static_cast<char*>(s.attn_state) + std::size_t(slot) * kHidden * kConvState * 2;
            auto conv = dense_tensor(weights, layer[Role::AttnSconv]);
            Tensor conv2d(conv.data, DType::BF16, {kHidden, 4});
            causal_dconv_residual(vec(s.proj, kHidden), conv2d,
                                  Tensor(state, DType::BF16, {kHidden, kConvState}),
                                  state_hidden, vec(s.attn_conv, kHidden), stream);
            cudaMemcpyAsync(state, s.state_scratch, std::size_t(kHidden) * kConvState * 2,
                            cudaMemcpyDeviceToDevice, stream);
        }
        if (stage_sink) { stage_sink(layer_index, "attn", s.attn_conv, kHidden); }

        // ---- residual, in FP32 as the reference keeps it ---------------------------------
        Tensor residual(s.residual, DType::FP32, {kHidden, 1});
        // A boundary pushes its pre-layer state, which shifts everything the MLP read sees: the
        // residual then sits one entry above it rather than on top of it.
        const int mlp_depth = boundary ? stack_length_now + 1 : stack_length_now;
        if (boundary) {
            // A boundary starts a new block: it pushes the pre-layer state and does not add the
            // attention output to anything.
            cast_bf16_to_f32(residual, vec(s.attn_conv, kHidden), stream);
            cudaMemcpyAsync(static_cast<char*>(s.stack) + std::size_t(stack_length_now) * kHidden * 2,
                            s.h_in, kHidden * 2, cudaMemcpyDeviceToDevice, stream);
        } else {
            cast_bf16_to_f32(residual, vec(s.h_in, kHidden), stream);
            residual_add_f32(residual, residual, vec(s.attn_conv, kHidden), stream);
        }
        cast_f32_to_bf16(vec(s.gated, kHidden), residual, stream);

        // ---- MLP input, read from the stack plus the residual -----------------------------
        cudaMemcpyAsync(static_cast<char*>(s.stack) + std::size_t(mlp_depth) * kHidden * 2,
                        s.gated, kHidden * 2, cudaMemcpyDeviceToDevice, stream);
        {
            const auto query_slot = weights.index.mlp_queries[static_cast<std::size_t>(layer_index)];
            Tensor query(dense_tensor(weights, query_slot).data, DType::FP32, {kHidden, 1});
            Tensor states(s.stack, DType::BF16, {kHidden, mlp_depth + 1});
            block_residual_read(query, states, kEps, vec(s.mlp_in, kHidden), stream);
        }
        if (uses_messages) {
            message_gate(layer, Role::ControllerMlpDown, Role::ControllerMlpForget,
                         Role::ControllerMlpGain, s.mlp_in,
                         static_cast<char*>(s.message) + 3 * kHidden * 2, s.gated);
            cudaMemcpyAsync(s.mlp_in, s.gated, kHidden * 2, cudaMemcpyDeviceToDevice, stream);
        }
        rmsnorm(vec(s.mlp_in, kHidden), dense_tensor(weights, layer[Role::MlpNorm]), kEps, false,
                vec(s.mlp_in, kHidden), stream);
        if (stage_sink) { stage_sink(layer_index, "mlp_in", s.mlp_in, kHidden); }

        // ---- routed experts --------------------------------------------------------------
        Tensor gate_matrix(s.router_f32[static_cast<std::size_t>(layer_index)], DType::FP32,
                           {kExperts, kHidden});
        Tensor gate_bias(dense_tensor(weights, layer[Role::MlpGateBias]).data, DType::FP32,
                         {kExperts, 1});
        Tensor router_out(s.indices, DType::I32, {kTopK, 1});
        Tensor router_weights(s.router_weights, DType::FP32, {kTopK, 1});
        moe_router(vec(s.mlp_in, kHidden), gate_matrix, gate_bias, kTopK, 8.0F, router_out,
                   router_weights, stream);
        // The selection picks weight slots on the host, so this is the one readback a step keeps.
        auto* routing_host = static_cast<char*>(s.routing_host);
        check(cudaMemcpyAsync(routing_host, s.indices, std::size_t(kTopK) * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost, stream), "routing indices");
        check(cudaMemcpyAsync(routing_host + std::size_t(kTopK) * sizeof(std::int32_t),
                              s.router_weights, std::size_t(kTopK) * sizeof(float),
                              cudaMemcpyDeviceToHost, stream), "routing weights");
        // The experts cannot start until the host knows which ones, so this drain stays; the copy
        // itself no longer stages through a pageable buffer.
        //
        // `NINFER_INKLING2_NO_ROUTING_READBACK=1` skips the drain and reuses the previous visit's
        // selection. It is a probe, not a mode: it changes what the model computes, and exists so
        // that "how much does the readback cost" can be answered by measurement instead of by
        // argument.
        static const bool skip_readback = std::getenv("NINFER_INKLING2_NO_ROUTING_READBACK") != nullptr;
        if (!skip_readback) { check(cudaStreamSynchronize(stream), "routing wait"); }
        static std::vector<std::int32_t> probe_indices(static_cast<std::size_t>(kTopK), 0);
        static std::vector<float> probe_routing(static_cast<std::size_t>(kTopK), 1.0F);
        std::vector<std::int32_t> indices(static_cast<std::size_t>(kTopK));
        std::vector<float> routing(static_cast<std::size_t>(kTopK));
        if (skip_readback) {
            indices = probe_indices;
            routing = probe_routing;
        } else {
            std::memcpy(indices.data(), routing_host, indices.size() * sizeof(std::int32_t));
            std::memcpy(routing.data(), routing_host + std::size_t(kTopK) * sizeof(std::int32_t),
                        routing.size() * sizeof(float));
            probe_indices = indices;
            probe_routing = routing;
        }
        if (routing_hook) {
            routing_hook(slot, indices, routing);
            // The hook may have replaced the selection (the parity harness pins it to the oracle),
            // and the device path reads the selection from device memory, so a pinned choice has to
            // be written back or the two disagree.
            check(cudaMemcpyAsync(s.indices, indices.data(),
                                  indices.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                  stream), "pinned indices");
            check(cudaMemcpyAsync(s.router_weights, routing.data(), routing.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream), "pinned weights");
        }
        Tensor accumulator(s.moe_acc, DType::FP32, {kHidden, 1});
        check(cudaMemsetAsync(s.moe_acc, 0, std::size_t(kHidden) * 4, stream), "moe clear");
        // The matching probe for the other half of the question: skip the routed experts entirely
        // and measure what the rest of the visit costs. Same caveat - it changes what is computed.
        static const bool skip_experts = std::getenv("NINFER_INKLING2_NO_EXPERTS") != nullptr;
        // `NINFER_INKLING2_DEVICE_EXPERTS=1` takes the selection off the host entirely: both gather
        // routes read the weight and the slot from device memory. Off by default while it is being
        // proven, so the two paths can be compared on the same run.
        static const bool device_experts =
            std::getenv("NINFER_INKLING2_DEVICE_EXPERTS") != nullptr;
        const auto* expert_slots = static_cast<const std::int32_t*>(s.indices);
        const auto* routing_scales = static_cast<const float*>(s.router_weights);
        const auto* w13_table =
            static_cast<const Weight*>(s.expert_w13_table[static_cast<std::size_t>(layer_index)]);
        const auto* w2_table =
            static_cast<const Weight*>(s.expert_w2_table[static_cast<std::size_t>(layer_index)]);
        const Weight& w13_extents =
            s.expert_w13_template[static_cast<std::size_t>(layer_index)];
        const Weight& w2_extents = s.expert_w2_template[static_cast<std::size_t>(layer_index)];
        const bool nvfp4_experts = w13_extents.qtype == QType::NVFP4;
        for (int j = 0; j < kTopK && !skip_experts; ++j) {
            if (device_experts) {
                if (nvfp4_experts) {
                    ops::detail::launch_nvfp4_gather_gemv(vec(s.mlp_in, kHidden), w13_table,
                                                          expert_slots, j, w13_extents,
                                                          vec(s.gate_up, 2 * kInter), stream);
                } else {
                    ops::detail::launch_q8_gather_gemv(vec(s.mlp_in, kHidden), w13_table,
                                                       expert_slots, j, w13_extents,
                                                       vec(s.gate_up, 2 * kInter), stream);
                }
                softcapped_swiglu(vec(s.gate_up, 2 * kInter), 4.0F, 25.0F, vec(s.act, kInter),
                                  stream);
                if (nvfp4_experts) {
                    ops::detail::launch_nvfp4_gather_gemv(vec(s.act, kInter), w2_table, expert_slots,
                                                          j, w2_extents, vec(s.down, kHidden),
                                                          stream);
                } else {
                    ops::detail::launch_q8_gather_gemv(vec(s.act, kInter), w2_table, expert_slots, j,
                                                       w2_extents, vec(s.down, kHidden), stream);
                }
                axpy_bf16_into_f32_device_scale(accumulator, vec(s.down, kHidden), routing_scales, j,
                                                stream);
                continue;
            }
            const auto expert = static_cast<std::size_t>(indices[static_cast<std::size_t>(j)]);
            linear(vec(s.mlp_in, kHidden), matrix_weight(weights, layer.expert_w13[expert]),
                   vec(s.gate_up, 2 * kInter), stream);
            softcapped_swiglu(vec(s.gate_up, 2 * kInter), 4.0F, 25.0F, vec(s.act, kInter), stream);
            linear(vec(s.act, kInter), matrix_weight(weights, layer.expert_w2[expert]),
                   vec(s.down, kHidden), stream);
            axpy_bf16_into_f32(accumulator, vec(s.down, kHidden),
                               routing[static_cast<std::size_t>(j)], stream);
        }
        cast_f32_to_bf16(vec(s.down, kHidden), accumulator, stream);
        if (stage_sink) { stage_sink(layer_index, "mlp_pre", s.down, kHidden); }

        {
            char* state = static_cast<char*>(s.mlp_state) + std::size_t(slot) * kHidden * kConvState * 2;
            auto conv = dense_tensor(weights, layer[Role::MlpSconv]);
            Tensor conv2d(conv.data, DType::BF16, {kHidden, 4});
            causal_dconv_residual(vec(s.down, kHidden), conv2d,
                                  Tensor(state, DType::BF16, {kHidden, kConvState}),
                                  state_hidden, vec(s.mlp_conv, kHidden), stream);
            cudaMemcpyAsync(state, s.state_scratch, std::size_t(kHidden) * kConvState * 2,
                            cudaMemcpyDeviceToDevice, stream);
        }
        if (stage_sink) { stage_sink(layer_index, "mlp", s.mlp_conv, kHidden); }
        residual_store_bf16(vec(s.h, kHidden), residual, vec(s.mlp_conv, kHidden), stream);
        if (stage_sink) { stage_sink(layer_index, "out", s.h, kHidden); }
        (void)stack_limit;
    };

    for (const auto& visit : s.schedule.visits) {
        if (static_cast<int>(visit.pass) != pass_seen) {
            if (visit.pass > 0) {
                // A later pass restarts from the pre-loop snapshot; the pass that just ended is the
                // message source.
                cudaMemcpyAsync(s.previous_h, s.h, kHidden * 2, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(s.previous_block,
                                static_cast<char*>(s.stack) + std::size_t(stack_length - 1) * kHidden * 2,
                                kHidden * 2, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(s.h, s.pre_loop_h, kHidden * 2, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(s.stack, s.pre_loop_stack,
                                std::size_t(stack_length) * kHidden * 2, cudaMemcpyDeviceToDevice,
                                stream);
                stack_length = 1;
                if (stage_sink) {
                    stage_sink(6, "previous_block", s.previous_block, kHidden);
                    stage_sink(6, "previous_h", s.previous_h, kHidden);
                }
            }
            pass_seen = static_cast<int>(visit.pass);
        }
        if (visit.pass == 0 && visit.layer == 6 && !pre_loop_saved) {
            cudaMemcpyAsync(s.pre_loop_h, s.h, kHidden * 2, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(s.pre_loop_stack, s.stack,
                            std::size_t(stack_length) * kHidden * 2, cudaMemcpyDeviceToDevice,
                            stream);
            pre_loop_saved = true;
        }
        run_visit(visit, stack_length);
        if (visit.layer % 6 == 0) { ++stack_length; }
    }

    // ---- final read, norm and unembed ------------------------------------------------------
    {
        cudaMemcpyAsync(static_cast<char*>(s.stack) + std::size_t(stack_length) * kHidden * 2, s.h,
                        kHidden * 2, cudaMemcpyDeviceToDevice, stream);
        Tensor query(dense_tensor(weights, weights.index.output_query).data, DType::FP32,
                     {kHidden, 1});
        Tensor states(s.stack, DType::BF16, {kHidden, stack_length + 1});
        if (stage_sink) { stage_sink(23, "final_states", s.stack, kHidden * (stack_length + 1)); }
        block_residual_read(query, states, kEps, vec(s.mlp_in, kHidden), stream);
        if (stage_sink) { stage_sink(23, "final_read", s.mlp_in, kHidden); }
        rmsnorm(vec(s.mlp_in, kHidden), dense_tensor(weights, weights.index.norm), kEps, false,
                vec(s.gated, kHidden), stream);
        if (stage_sink) { stage_sink(23, "final", s.gated, kHidden); }
        const auto unembed = matrix_weight(weights, weights.index.unembed);
        Tensor output(s.logits, DType::BF16, {unembed.n, 1});
        linear(vec(s.gated, kHidden), unembed, output, stream);
        logits_softcap(output, 30.0F, stream);
    }

    ++s.position_now;
    s.stack_length = stack_length;
    return Tensor(s.logits, DType::BF16, {matrix_weight(weights, weights.index.unembed).n, 1});
}
} // namespace ninfer::models::inkling2::execution
