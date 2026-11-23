// The 48 layer visits of Inkling 2 Mini against the reference, for one token.
//
//   ninfer_inkling2_visits_test --artifact <artifact> --oracle <dir>
//
// The oracle comes from `dump_visits.py --token-id N --raw <dir>`, which writes the hidden state
// entering (visit_in_<i>.bin) and leaving (visit_out_<i>.bin) every visit as float32 [2304].

#include "artifact/reader.h"
#include "core/device.h"
#include "models/inkling2/config.h"
#include "models/inkling2/materialize.h"
#include "models/inkling2/parameters.h"
#include "models/inkling2/schedule.h"
#include "ninfer/ops/block_residual_read.h"
#include "ninfer/ops/causal_dconv_residual.h"
#include "ninfer/ops/dynamic_short_conv.h"
#include "ninfer/ops/head_rmsnorm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/message_merge.h"
#include "ninfer/ops/moe_router.h"
#include "ninfer/ops/relbias_attention.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scaled_l2_clip.h"
#include "ninfer/ops/softcapped_swiglu.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::models::inkling2;

constexpr int kHidden = 2304;
constexpr int kQ = 3072;
constexpr int kKv = 1024;
constexpr int kRel = 384;
constexpr int kInter = 1152;
constexpr int kExperts = 128;
constexpr int kTopK = 8;
constexpr float kEps = 1.0e-6F;
constexpr float kQkThreshold = 22.627416997969522F;
constexpr float kRelThreshold = 8.0F;
constexpr float kLogAlpha = 0.1F;
constexpr float kLogFloor = 8192.0F;

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

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("cannot open " + path.string()); }
    stream.seekg(0, std::ios::end);
    const auto bytes = stream.tellg();
    stream.seekg(0);
    std::vector<float> data(static_cast<std::size_t>(bytes) / sizeof(float));
    stream.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("cannot open " + path.string()); }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

void* alloc(std::size_t count, std::size_t element) {
    void* device = nullptr;
    cudaMalloc(&device, count * element);
    return device;
}

struct Scratch {
    void* h;
    void* attn_in;
    void* q_in;
    void* k_in;
    void* v_in;
    void* qn;
    void* kn;
    void* vn;
    void* q;
    void* k;
    void* v;
    void* k_conv;
    void* v_conv;
    void* dyn_k;
    void* dyn_v;
    void* rel;
    void* sink;
    void* gate;
    void* attn;
    void* proj;
    void* attn_conv;
    void* mlp_in;
    void* gated_x;
    void* gated_m;
    void* forget;
    void* gain;
    void* controller_h;
    void* gate_up;
    void* act;
    void* down;
    void* mlp_conv;
    void* stack;      // [8, hidden] block states
    void* message;    // [4, hidden]
    void* ones;
    void* indices;
    void* router_weights;
    void* position;
    void* concat;     // [2 * hidden] for the message gate
    void* rms_x;
    void* rms_m;
    void* msg_stack;  // [2, hidden] for the cross-pass message read
    void* q_clip;
    void* k_clip;
    void* v_norm;
    void* rel_clip;
    void* k_state;
    void* v_state;
    void* attn_state;
    void* mlp_state;
    void* k_state_out;
    void* v_state_out;
    void* attn_state_out;
    void* mlp_state_out;
};

} // namespace


namespace {

struct Runner {
    const TextWeights& weights;
    const Config& config;
    Scratch s{};
    cudaStream_t stream = nullptr;
    double worst_visit = 0.0;
    void* gate_f32 = nullptr;
    std::filesystem::path route_dir;      // optional: reference routing per layer
    std::filesystem::path stage_fixture;  // per-stage reference buffers
    bool stage = false;
    int stage_layer = 0;
    void stage_compare(void* device, const char* suffix, int width = kHidden) {
        if (!stage) { return; }
        const std::string name = "layer" + std::to_string(stage_layer) + "_" + suffix;
        compare(device, stage_fixture / name, name.c_str(), width);
    }

    void init() {
        auto make = [](std::size_t count) { return alloc(count, 2); };
        s.h = make(kHidden); s.attn_in = make(kHidden);
        s.q_in = make(kHidden); s.k_in = make(kHidden); s.v_in = make(kHidden);
        s.qn = make(kHidden); s.kn = make(kHidden); s.vn = make(kHidden);
        s.q = make(kQ); s.k = make(kKv); s.v = make(kKv);
        s.k_conv = make(kKv); s.v_conv = make(kKv);
        s.dyn_k = make(kKv); s.dyn_v = make(kKv);
        s.rel = make(kRel); s.sink = make(24); s.gate = make(kQ); s.attn = make(kQ);
        s.proj = make(kHidden); s.attn_conv = make(kHidden);
        s.mlp_in = make(kHidden); s.gated_x = make(kHidden); s.gated_m = make(kHidden);
        s.forget = make(kHidden); s.gain = make(kHidden); s.controller_h = make(64);
        s.gate_up = make(2 * kInter); s.act = make(kInter); s.down = make(kHidden);
        s.mlp_conv = make(kHidden);
        s.stack = make(8 * kHidden); s.message = make(4 * kHidden); s.msg_stack = make(2 * kHidden);
        s.concat = make(2 * kHidden); s.rms_x = make(kHidden); s.rms_m = make(kHidden);
        s.q_clip = make(kQ); s.k_clip = make(kKv); s.v_norm = make(kKv); s.rel_clip = make(kRel);
        s.ones = make(kKv);
        {
            std::vector<std::uint16_t> ones(static_cast<std::size_t>(kKv), to_bf16(1.0F));
            cudaMemcpy(s.ones, ones.data(), ones.size() * 2, cudaMemcpyHostToDevice);
        }
        s.position = alloc(1, sizeof(int));
        const int position = 0;
        cudaMemcpy(s.position, &position, sizeof(int), cudaMemcpyHostToDevice);
        s.indices = alloc(kTopK, sizeof(std::int32_t));
        s.router_weights = alloc(kTopK, sizeof(float));
        s.k_state = alloc(kKv * 3, 2); s.v_state = alloc(kKv * 3, 2);
        s.attn_state = alloc(kHidden * 3, 2); s.mlp_state = alloc(kHidden * 3, 2);
        s.k_state_out = alloc(kKv * 3, 2); s.v_state_out = alloc(kKv * 3, 2);
        s.attn_state_out = alloc(kHidden * 3, 2); s.mlp_state_out = alloc(kHidden * 3, 2);
        for (void* p : {s.k_state, s.v_state, s.attn_state, s.mlp_state}) {
            cudaMemset(p, 0, (p == s.k_state || p == s.v_state ? kKv : kHidden) * 3 * 2);
        }
    }

    // Ops take their output as `Tensor&`, so the shape helpers hand back a reference-carrying
    // wrapper instead of a temporary.
    struct TRef {
        Tensor t;
        operator Tensor&() { return t; }
        operator const Tensor&() const { return t; }
    };
    [[nodiscard]] TRef vec(void* device, int width) const {
        return TRef{Tensor(device, DType::BF16, {width, 1})};
    }
    [[nodiscard]] TRef grid(void* device, int rows, int width) const {
        return TRef{Tensor(device, DType::BF16, {width, rows})};
    }

    void linear_role(const LayerSlots& layer, Role role, void* input, int in_width, int out_width,
                     void* out) {
        linear(vec(input, in_width), matrix_weight(weights, layer[role]), vec(out, out_width),
               stream);
    }

    // (1 - tanh(forget)) * x + tanh(gain) * message, with the gates produced from both streams.
    void message_gate(const LayerSlots& layer, Role down, Role forget_proj, Role gain_proj,
                      void* x, void* message, void* out) {
        head_rmsnorm(vec(x, kHidden), kEps, vec(s.rms_x, kHidden), stream);
        head_rmsnorm(vec(message, kHidden), kEps, vec(s.rms_m, kHidden), stream);
        cudaMemcpyAsync(s.concat, s.rms_x, kHidden * 2, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(static_cast<char*>(s.concat) + kHidden * 2, s.rms_m, kHidden * 2,
                        cudaMemcpyDeviceToDevice, stream);
        linear(vec(s.concat, 2 * kHidden), matrix_weight(weights, layer[down]),
               vec(s.controller_h, 64), stream);
        linear_role(layer, forget_proj, s.controller_h, 64, kHidden, s.forget);
        linear_role(layer, gain_proj, s.controller_h, 64, kHidden, s.gain);
        message_merge(vec(x, kHidden), vec(message, kHidden), vec(s.forget, kHidden),
                      vec(s.gain, kHidden), vec(out, kHidden), stream);
    }

    void run_visit(int layer_index, int pass, int loop_index, int stack_length,
                   int previous_stack_length, void* previous_h) {
        // Every visit starts a fresh three-value window: this run feeds one token, so each
        // convolution sees zeros and the state is scratch rather than persistent.
        cudaMemset(s.k_state, 0, kKv * 3 * 2);
        cudaMemset(s.v_state, 0, kKv * 3 * 2);
        cudaMemset(s.attn_state, 0, kHidden * 3 * 2);
        cudaMemset(s.mlp_state, 0, kHidden * 3 * 2);

        const auto& layer = weights.index.layers[static_cast<std::size_t>(layer_index)];
        const bool sliding = config.sliding_layer[static_cast<std::size_t>(layer_index)];
        const bool boundary = (layer_index % 6) == 0;

        // The pre-layer state is what a boundary appends to the block stack.
        void* h_in = alloc(kHidden, 2);
        cudaMemcpy(h_in, s.h, kHidden * 2, cudaMemcpyDeviceToDevice);

        // ---- attention input: read out of the block stack -------------------------------
        // The block residual read yields one stream for a sliding layer and three for a full
        // layer, where q, k and v read the stack with their own query.
        const int queries = sliding ? 1 : 3;
        if (layer_index > 0) {
            const auto slot = weights.index.attn_queries[static_cast<std::size_t>(layer_index)];
            if (slot < 0) { throw std::runtime_error("missing attn_queries for layer"); }
            cudaMemcpyAsync(static_cast<char*>(s.stack) + stack_length * kHidden * 2, h_in,
                            kHidden * 2, cudaMemcpyDeviceToDevice, stream);
            Tensor query(dense_tensor(weights, slot).data, DType::FP32, {kHidden, queries});
            Tensor state_grid(s.stack, DType::BF16, {kHidden, stack_length + 1});
            if (stage && stage_layer <= 2) {
                std::vector<float> q(static_cast<std::size_t>(kHidden) * queries);
                cudaMemcpy(q.data(), dense_tensor(weights, slot).data, q.size() * 4,
                           cudaMemcpyDeviceToHost);
                std::cout << "      attn query first3: " << q[0] << " " << q[1] << " " << q[2]
                          << "\n";
            }
            if (stage && stage_layer <= 2) {
                std::vector<std::uint16_t> st(static_cast<std::size_t>(kHidden) * (stack_length + 1));
                cudaMemcpy(st.data(), s.stack, st.size() * 2, cudaMemcpyDeviceToHost);
                std::ofstream out("/tmp/stage_stack_" + std::to_string(stage_layer) + ".bin",
                                  std::ios::binary);
                out.write(reinterpret_cast<const char*>(st.data()), std::streamsize(st.size() * 2));
            }
            if (stage && stage_layer <= 2) {
                std::vector<std::uint8_t> gamma(static_cast<std::size_t>(kHidden) * 2);
                cudaMemcpy(gamma.data(), dense_tensor(weights, layer[Role::AttnNorm]).data,
                           gamma.size(), cudaMemcpyDeviceToHost);
                std::ofstream g(std::string("/tmp/stage_gamma_") +
                                std::to_string(stage_layer) + ".bin", std::ios::binary);
                g.write(reinterpret_cast<const char*>(gamma.data()), std::streamsize(gamma.size()));
            }
            Tensor read_out(s.message, DType::BF16, {kHidden, queries});
            block_residual_read(query, state_grid, kEps, read_out, stream);
            cudaMemcpy(s.attn_in, s.message, kHidden * 2, cudaMemcpyDeviceToDevice);
            if (queries == 3) {
                cudaMemcpy(s.k_in, static_cast<char*>(s.message) + kHidden * 2, kHidden * 2,
                           cudaMemcpyDeviceToDevice);
                cudaMemcpy(s.v_in, static_cast<char*>(s.message) + 2 * kHidden * 2, kHidden * 2,
                           cudaMemcpyDeviceToDevice);
            }
        } else {
            cudaMemcpy(s.attn_in, h_in, kHidden * 2, cudaMemcpyDeviceToDevice);
        }

        // ---- cross-pass messages --------------------------------------------------------
        const bool uses_messages = pass > 0;
        void* k_in_base = queries == 3 ? s.k_in : s.attn_in;
        void* v_in_base = queries == 3 ? s.v_in : s.attn_in;
        auto one = [](void*) {};
        if (uses_messages) {
            cudaMemcpyAsync(s.msg_stack, static_cast<char*>(s.stack) +
                                                (previous_stack_length - 1) * kHidden * 2,
                            kHidden * 2, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(static_cast<char*>(s.msg_stack) + kHidden * 2, previous_h, kHidden * 2,
                            cudaMemcpyDeviceToDevice, stream);
            Tensor message_query(dense_tensor(weights, weights.index.cross_pass_queries).data,
                                DType::FP32, {kHidden, 4});
            Tensor message_states(s.msg_stack, DType::BF16, {kHidden, 2});
            Tensor message_out(s.message, DType::BF16, {kHidden, 4});
            block_residual_read(message_query, message_states, kEps, message_out, stream);
            // The queries are indexed by loop position; slice the [4, hidden] rows for this one.
            char* rows = static_cast<char*>(s.message);
            // The base streams are the block read's rows when the layer is a full one; a sliding
            // layer has a single base stream and blends it with each message.
            one(k_in_base);
            message_gate(layer, Role::ControllerAttnDown, Role::ControllerAttnForget,
                         Role::ControllerAttnGain, s.attn_in, rows, s.q_in);
            message_gate(layer, Role::ControllerKDown, Role::ControllerKForget,
                         Role::ControllerKGain, k_in_base, rows + kHidden * 2, s.k_in);
            message_gate(layer, Role::ControllerVDown, Role::ControllerVForget,
                         Role::ControllerVGain, v_in_base, rows + 2 * kHidden * 2, s.v_in);
        } else {
            cudaMemcpy(s.q_in, s.attn_in, kHidden * 2, cudaMemcpyDeviceToDevice);
            if (queries == 1) {
                cudaMemcpy(s.k_in, s.attn_in, kHidden * 2, cudaMemcpyDeviceToDevice);
                cudaMemcpy(s.v_in, s.attn_in, kHidden * 2, cudaMemcpyDeviceToDevice);
            }
        }

        // ---- per-stream norms -----------------------------------------------------------
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

        // ---- queries, keys, values ------------------------------------------------------
        ScaledL2ClipArgs q_args;
        q_args.rows = 1; q_args.heads = 24; q_args.head_dim = 128;
        q_args.threshold = kQkThreshold;
        q_args.tau_alpha = sliding ? 0.0F : kLogAlpha;
        q_args.tau_floor = kLogFloor;
        if (stage && stage_layer <= 2) {
            std::vector<std::uint16_t> a(static_cast<std::size_t>(kHidden));
            cudaMemcpy(a.data(), s.attn_in, a.size() * 2, cudaMemcpyDeviceToHost);
            std::ofstream out(std::string("/tmp/stage_attn_in_") +
                              std::to_string(stage_layer) + ".bin", std::ios::binary);
            out.write(reinterpret_cast<const char*>(a.data()), std::streamsize(a.size() * 2));
        }
        stage_compare(s.qn, "ln.bin");
        linear_role(layer, Role::WqDu, s.qn, kHidden, kQ, s.q);
        Tensor position(s.position, DType::I32, {1});
        Tensor q_heads(s.q, DType::BF16, {kQ, 1});
        scaled_l2_clip(q_heads, dense_tensor(weights, layer[Role::QkClip]),
                       sliding ? nullptr : &position, q_args, vec(s.q_clip, kQ), stream);
        cudaMemcpy(s.q, s.q_clip, kQ * 2, cudaMemcpyDeviceToDevice);

        stage_compare(s.q, "q.bin", kQ);
        linear_role(layer, Role::WkDv, s.kn, kHidden, kKv, s.k);
        if (sliding) {
            auto conv = dense_tensor(weights, layer[Role::KSconv]);
            Tensor conv2d(conv.data, DType::BF16, {kKv, 4});
            Tensor state_in(s.k_state, DType::BF16, {kKv, 3});
            Tensor state_out(s.k_state_out, DType::BF16, {kKv, 3});
            causal_dconv_residual(vec(s.k, kKv), conv2d, state_in, state_out, vec(s.k_conv, kKv),
                                  stream);
        } else {
            // dynamic: taps = static[4, C] + weight_proj(hidden) per head
            auto proj = matrix_weight(weights, layer[Role::KSconvProj]);
            Tensor dyn_k(s.dyn_k, DType::BF16, {32, 1});
            linear(vec(s.kn, kHidden), proj, dyn_k, stream);
            auto static_w = dense_tensor(weights, layer[Role::KSconvStatic]);
            Tensor static_4c(static_w.data, DType::BF16, {4, kKv});
            Tensor state_in(s.k_state, DType::BF16, {kKv, 3});
            Tensor state_out(s.k_state_out, DType::BF16, {kKv, 3});
            Tensor dyn_k_row(s.dyn_k, DType::BF16, {32, 1});
            dynamic_short_conv(vec(s.k, kKv), static_4c, dyn_k_row, state_in, state_out,
                               vec(s.k_conv, kKv), 8, stream);
        }
        ScaledL2ClipArgs k_args;
        k_args.rows = 1; k_args.heads = 8; k_args.head_dim = 128;
        k_args.threshold = kQkThreshold; k_args.tau_alpha = 0.0F; k_args.tau_floor = kLogFloor;
        scaled_l2_clip(grid(s.k_conv, 1, kKv), Tensor(s.ones, DType::BF16, {8, 128}), nullptr,
                       k_args, vec(s.k_clip, kKv), stream);
        cudaMemcpy(s.k_conv, s.k_clip, kKv * 2, cudaMemcpyDeviceToDevice);
        stage_compare(s.k_conv, "k.bin", kKv);

        linear_role(layer, Role::WvDv, s.vn, kHidden, kKv, s.v);
        if (sliding) {
            auto conv = dense_tensor(weights, layer[Role::VSconv]);
            Tensor conv2d(conv.data, DType::BF16, {kKv, 4});
            Tensor state_in(s.v_state, DType::BF16, {kKv, 3});
            Tensor state_out(s.v_state_out, DType::BF16, {kKv, 3});
            causal_dconv_residual(vec(s.v, kKv), conv2d, state_in, state_out, vec(s.v_conv, kKv),
                                  stream);
        } else {
            auto proj = matrix_weight(weights, layer[Role::VSconvProj]);
            Tensor dyn_v(s.dyn_v, DType::BF16, {32, 1});
            linear(vec(s.vn, kHidden), proj, dyn_v, stream);
            auto static_w = dense_tensor(weights, layer[Role::VSconvStatic]);
            Tensor static_4c(static_w.data, DType::BF16, {4, kKv});
            Tensor state_in(s.v_state, DType::BF16, {kKv, 3});
            Tensor state_out(s.v_state_out, DType::BF16, {kKv, 3});
            Tensor dyn_v_row(s.dyn_v, DType::BF16, {32, 1});
            dynamic_short_conv(vec(s.v, kKv), static_4c, dyn_v_row, state_in, state_out,
                               vec(s.v_conv, kKv), 8, stream);
        }
        Tensor v_in_t(s.v_conv, DType::BF16, {128, 8});
        Tensor v_out_t(s.v_norm, DType::BF16, {128, 8});
        head_rmsnorm(v_in_t, kEps, v_out_t, stream);
        cudaMemcpy(s.v_conv, s.v_norm, kKv * 2, cudaMemcpyDeviceToDevice);

        stage_compare(s.v_conv, "v.bin", kKv);

        // ---- relative states, sink and gate ---------------------------------------------
        ScaledL2ClipArgs rel_args;
        rel_args.rows = 1; rel_args.heads = 24; rel_args.head_dim = 16;
        rel_args.threshold = kRelThreshold;
        rel_args.tau_alpha = sliding ? 0.0F : kLogAlpha;
        rel_args.tau_floor = kLogFloor;
        linear_role(layer, Role::WrDu, s.qn, kHidden, kRel, s.rel);
        Tensor rel_heads(s.rel, DType::BF16, {kRel, 1});
        scaled_l2_clip(rel_heads, dense_tensor(weights, layer[Role::RelClip]),
                       sliding ? nullptr : &position, rel_args, vec(s.rel_clip, kRel), stream);
        cudaMemcpy(s.rel, s.rel_clip, kRel * 2, cudaMemcpyDeviceToDevice);
        linear_role(layer, Role::AttnSink, s.qn, kHidden, 24, s.sink);
        linear_role(layer, Role::GateAttn, s.qn, kHidden, kQ, s.gate);

        stage_compare(s.sink, "sink.bin", 24);
        RelBiasAttentionArgs args;
        args.heads = 24; args.kv_heads = 8; args.head_dim = 128;
        args.q_pos = 0; args.kv_len = 1; args.window = sliding ? 1024 : 0;
        args.rel_extent = 1024; args.d_rel = 16;
        args.scale = 1.0F / std::sqrt(128.0F);
        relbias_attention(grid(s.q, 1, kQ), grid(s.k_conv, 1, kKv), grid(s.v_conv, 1, kKv),
                          grid(s.rel, 1, kRel),
                          dense_tensor(weights, layer[Role::RelLogits]),
                          vec(s.sink, 24), vec(s.gate, kQ), args, vec(s.attn, kQ), stream);

        linear_role(layer, Role::WoUd, s.attn, kQ, kHidden, s.proj);
        stage_compare(s.proj, "attnpre.bin");
        {
            auto conv = dense_tensor(weights, layer[Role::AttnSconv]);
            Tensor conv2d(conv.data, DType::BF16, {kHidden, 4});
            Tensor state_in(s.attn_state, DType::BF16, {kHidden, 3});
            Tensor state_out(s.attn_state_out, DType::BF16, {kHidden, 3});
            causal_dconv_residual(vec(s.proj, kHidden), conv2d, state_in, state_out,
                                  vec(s.attn_conv, kHidden), stream);
        }

        // ---- residual and the MLP read ---------------------------------------------------
        stage_compare(s.attn_conv, "attn.bin");
        if (boundary) {
            // A boundary layer does not add its attention output to anything: it starts a new
            // block by pushing the pre-layer state, and that state is also the MLP's residual.
            cudaMemcpy(s.gated_x, s.attn_conv, kHidden * 2, cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(static_cast<char*>(s.stack) + stack_length * kHidden * 2, h_in,
                            kHidden * 2, cudaMemcpyDeviceToDevice, stream);
            ++stack_length;
        } else {
            cudaMemcpy(s.gated_x, h_in, kHidden * 2, cudaMemcpyDeviceToDevice);
            accumulate(s.gated_x, s.attn_conv);
        }
        // The MLP input is read from the block stack plus the residual, which sits on top of it.
        cudaMemcpyAsync(static_cast<char*>(s.stack) + stack_length * kHidden * 2, s.gated_x,
                        kHidden * 2, cudaMemcpyDeviceToDevice, stream);
        Tensor mlp_query(dense_tensor(weights, weights.index.mlp_queries[
                                 static_cast<std::size_t>(layer_index)]).data,
                         DType::FP32, {kHidden, 1});
        Tensor mlp_states(s.stack, DType::BF16, {kHidden, stack_length + 1});
        block_residual_read(mlp_query, mlp_states, kEps, vec(s.mlp_in, kHidden), stream);
        if (uses_messages) {
            message_gate(layer, Role::ControllerMlpDown, Role::ControllerMlpForget,
                         Role::ControllerMlpGain, s.mlp_in,
                         static_cast<char*>(s.message) + 3 * kHidden * 2, s.gated_m);
            cudaMemcpy(s.mlp_in, s.gated_m, kHidden * 2, cudaMemcpyDeviceToDevice);
        }
        rmsnorm(vec(s.mlp_in, kHidden), dense_tensor(weights, layer[Role::MlpNorm]), kEps,
                false, vec(s.mlp_in, kHidden), stream);

        if (stage && stage_layer == 0) {
            auto dump = [](void* device, const char* name, int width, int element) {
                std::vector<std::uint8_t> raw(std::size_t(width) * element);
                cudaMemcpy(raw.data(), device, raw.size(), cudaMemcpyDeviceToHost);
                std::ofstream out(std::string("/tmp/stage_") + name, std::ios::binary);
                out.write(reinterpret_cast<const char*>(raw.data()), std::streamsize(raw.size()));
            };
            dump(s.mlp_in, "mlp_in_pre.bin", kHidden, 2);
            dump(mlp_query.data, "query.bin", kHidden, 4);
            dump(s.stack, "stack.bin", 2 * kHidden, 2);
        }
        stage_compare(s.mlp_in, "mlp_in.bin");

        // ---- routed experts ---------------------------------------------------------------
        auto gate_w = dense_tensor(weights, layer[Role::MlpGate]);
        auto gate_bias = dense_tensor(weights, layer[Role::MlpGateBias]);
        // The checkpoint stores the router in BF16 while the reference runs it in FP32.
        if (gate_f32 == nullptr) { gate_f32 = alloc(std::size_t(kExperts) * kHidden, 4); }
        {
            std::vector<std::uint16_t> host_gate(std::size_t(kExperts) * kHidden);
            std::vector<float> wide(host_gate.size());
            cudaMemcpy(host_gate.data(), gate_w.data, host_gate.size() * 2, cudaMemcpyDeviceToHost);
            for (std::size_t i = 0; i < host_gate.size(); ++i) { wide[i] = from_bf16(host_gate[i]); }
            cudaMemcpy(gate_f32, wide.data(), wide.size() * 4, cudaMemcpyHostToDevice);
        }
        Tensor gate_matrix(gate_f32, DType::FP32, {kExperts, kHidden});
        Tensor gate_bias_vec(gate_bias.data, DType::FP32, {kExperts});
        Tensor router_out(s.indices, DType::I32, {kTopK, 1});
        Tensor router_weights(s.router_weights, DType::FP32, {kTopK, 1});
        moe_router(vec(s.mlp_in, kHidden), gate_matrix, gate_bias_vec, kTopK, 8.0F, router_out,
                   router_weights, stream);
        // Does the routing have a near-tie? A hidden one would flip an expert under BF16 noise
        // and change the MoE output far more than rounding.
        if (stage) {
            std::vector<std::uint16_t> xh(kHidden);
            cudaMemcpy(xh.data(), s.mlp_in, kHidden * 2, cudaMemcpyDeviceToHost);
            const std::vector<float>* gate_host = nullptr;
            std::vector<float> gate_copy;
            if (gate_f32 != nullptr) {
                gate_copy.resize(std::size_t(kExperts) * kHidden);
                cudaMemcpy(gate_copy.data(), gate_f32, gate_copy.size() * 4,
                           cudaMemcpyDeviceToHost);
                gate_host = &gate_copy;
            }
            if (gate_host != nullptr) {
                std::vector<double> scores(kExperts);
                // The selection bias is FP32 in the checkpoint, unlike the BF16 gate matrix.
                std::vector<float> bias_host(kExperts);
                cudaMemcpy(bias_host.data(), dense_tensor(weights, layer[Role::MlpGateBias]).data,
                           kExperts * 4, cudaMemcpyDeviceToHost);
                for (int e = 0; e < kExperts; ++e) {
                    double dot = 0.0;
                    for (int i = 0; i < kHidden; ++i) {
                        dot += double((*gate_host)[std::size_t(e) * kHidden + i]) *
                               double(from_bf16(xh[std::size_t(i)]));
                    }
                    scores[std::size_t(e)] = 1.0 / (1.0 + std::exp(-dot)) + double(bias_host[std::size_t(e)]);
                }
                std::vector<double> sorted = scores;
                std::sort(sorted.begin(), sorted.end(), std::greater<double>());
                std::cout << "      router margin (8th-9th) " << (sorted[7] - sorted[8])
                          << " top " << sorted[0] << "\n";
            }
        }
        std::vector<std::int32_t> indices(kTopK);
        std::vector<float> routing(kTopK);
        cudaMemcpy(indices.data(), s.indices, kTopK * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(routing.data(), s.router_weights, kTopK * sizeof(float), cudaMemcpyDeviceToHost);
        if (!route_dir.empty()) {
            // The reference's own selection: routing is a discrete function and this model has
            // near-ties, so the MoE arithmetic has to be checked on a fixed selection.
            const auto path = route_dir / ("ref_route_" + std::to_string(layer_index) + ".json");
            if (std::filesystem::exists(path)) {
                const auto json = artifact::parse_json(read_text(path), "route");
                for (int j = 0; j < kTopK; ++j) {
                    indices[static_cast<std::size_t>(j)] = json.at("indices")[j].get<std::int32_t>();
                    routing[static_cast<std::size_t>(j)] =
                        static_cast<float>(json.at("weights")[j].get<double>());
                }
            }
        }
        if (stage) {
            std::cout << "      my experts:";
            for (int j = 0; j < kTopK; ++j) { std::cout << " " << indices[static_cast<std::size_t>(j)]; }
            std::cout << "\n      my weights:";
            for (int j = 0; j < kTopK; ++j) { std::cout << " " << routing[static_cast<std::size_t>(j)]; }
            std::cout << "\n";
            std::vector<std::uint16_t> a(3);
            cudaMemcpy(a.data(), s.mlp_in, 3 * 2, cudaMemcpyDeviceToHost);
            std::cout << "      my mlp_in[0:3] " << from_bf16(a[0]) << " " << from_bf16(a[1]) << " "
                      << from_bf16(a[2]) << "\n";
        }
        std::vector<double> accumulator(kHidden, 0.0);
        for (int j = 0; j < kTopK; ++j) {
            const auto expert = static_cast<std::size_t>(indices[static_cast<std::size_t>(j)]);
            linear(vec(s.mlp_in, kHidden), matrix_weight(weights, layer.expert_w13[expert]),
                   vec(s.gate_up, 2 * kInter), stream);
            softcapped_swiglu(vec(s.gate_up, 2 * kInter), 4.0F, 25.0F, vec(s.act, kInter), stream);
            linear(vec(s.act, kInter), matrix_weight(weights, layer.expert_w2[expert]),
                   vec(s.down, kHidden), stream);
            std::vector<std::uint16_t> host(kHidden);
            cudaMemcpy(host.data(), s.down, kHidden * 2, cudaMemcpyDeviceToHost);
            for (int i = 0; i < kHidden; ++i) {
                accumulator[static_cast<std::size_t>(i)] +=
                    double(routing[static_cast<std::size_t>(j)]) *
                    double(from_bf16(host[static_cast<std::size_t>(i)]));
            }
        }
        std::vector<std::uint16_t> host_moe(kHidden);
        for (int i = 0; i < kHidden; ++i) {
            host_moe[static_cast<std::size_t>(i)] = to_bf16(float(accumulator[static_cast<std::size_t>(i)]));
        }
        cudaMemcpy(s.down, host_moe.data(), kHidden * 2, cudaMemcpyHostToDevice);

        {
            const auto path = stage_fixture /
                ("layer" + std::to_string(stage_layer) + "_mlp_pre.bin");
            if (stage && std::filesystem::exists(path)) {
                std::vector<std::uint16_t> host(kHidden);
                cudaMemcpy(host.data(), s.down, kHidden * 2, cudaMemcpyDeviceToHost);
                const auto expected = read_f32(path);
                double worst = 0.0;
                for (std::size_t i = 0; i < host.size(); ++i) {
                    worst = std::max(worst, std::fabs(double(from_bf16(host[i])) - expected[i]));
                }
                std::cout << "      " << path.filename().string() << ": max|err| " << worst
                          << "\n";
            }
        }
        {
            auto conv = dense_tensor(weights, layer[Role::MlpSconv]);
            Tensor conv2d(conv.data, DType::BF16, {kHidden, 4});
            Tensor state_in(s.mlp_state, DType::BF16, {kHidden, 3});
            Tensor state_out(s.mlp_state_out, DType::BF16, {kHidden, 3});
            causal_dconv_residual(vec(s.down, kHidden), conv2d, state_in, state_out,
                                  vec(s.mlp_conv, kHidden), stream);
        }
        if (stage) {
            std::vector<std::uint16_t> a(3);
            cudaMemcpy(a.data(), s.down, 3 * 2, cudaMemcpyDeviceToHost);
            std::cout << "      my moe[0:3] " << from_bf16(a[0]) << " " << from_bf16(a[1]) << " "
                      << from_bf16(a[2]) << "\n";
        }
        stage_compare(s.mlp_conv, "mlp.bin");
        accumulate(s.gated_x, s.mlp_conv);
        cudaMemcpy(s.h, s.gated_x, kHidden * 2, cudaMemcpyDeviceToDevice);
        stage_compare(s.h, "out.bin");
        cudaFree(h_in);
    }

    // Compares one device buffer against a float32 fixture and prints the worst element.
    void compare(void* device, const std::filesystem::path& path, const char* label,
                 int width = kHidden) {
        if (!std::filesystem::exists(path)) { return; }
        const auto expected = read_f32(path);
        std::vector<std::uint16_t> got(static_cast<std::size_t>(width));
        cudaMemcpy(got.data(), device, got.size() * 2, cudaMemcpyDeviceToHost);
        double worst = 0.0;
        std::size_t where = 0;
        for (std::size_t i = 0; i < expected.size() && i < got.size(); ++i) {
            const double error = std::fabs(double(from_bf16(got[i])) - expected[i]);
            if (error > worst) { worst = error; where = i; }
        }
        std::cout << "      " << label << ": max|err| " << worst << " at " << where << " (got "
                  << from_bf16(got[where]) << ", ref " << expected[where] << ")\n";
    }

    // Accumulates `add` into `base` through the host, in float32.
    void accumulate(void* base, void* add) {
        std::vector<std::uint16_t> a(kHidden), b(kHidden);
        cudaMemcpy(a.data(), base, kHidden * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), add, kHidden * 2, cudaMemcpyDeviceToHost);
        for (int i = 0; i < kHidden; ++i) {
            a[static_cast<std::size_t>(i)] = to_bf16(from_bf16(a[static_cast<std::size_t>(i)]) +
                                                     from_bf16(b[static_cast<std::size_t>(i)]));
        }
        cudaMemcpy(base, a.data(), kHidden * 2, cudaMemcpyHostToDevice);
    }
};

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path artifact_path;
        std::filesystem::path oracle;
        std::filesystem::path stages;
        std::filesystem::path routes;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--artifact" && i + 1 < argc) { artifact_path = argv[++i]; }
            else if (arg == "--oracle" && i + 1 < argc) { oracle = argv[++i]; }
            else if (arg == "--stages" && i + 1 < argc) { stages = argv[++i]; }
            else if (arg == "--routes" && i + 1 < argc) { routes = argv[++i]; }
            else { throw std::invalid_argument("unknown argument " + arg); }
        }
        if (artifact_path.empty() || oracle.empty()) {
            std::cout << "SKIP: supply --artifact and --oracle\n";
            return 77;
        }

        artifact::Reader reader(artifact_path);
        const auto config = parse_config(reader.directory());
        DeviceContext device;
        auto weights = materialize_text(reader, config, device);
        const auto schedule = build_schedule(config);

        Runner runner{weights, config};
        runner.init();
        if (!routes.empty()) { runner.route_dir = routes; }
        if (!stages.empty()) {
            runner.stage_fixture = stages;
            runner.stage = true;
        }

        // The token enters layer 0 as the embedded row the oracle recorded.
        const auto first_in = read_f32(oracle / "visit_in_0.bin");
        std::vector<std::uint16_t> host_h(first_in.size());
        for (std::size_t i = 0; i < first_in.size(); ++i) { host_h[i] = to_bf16(first_in[i]); }
        cudaMemcpy(runner.s.h, host_h.data(), host_h.size() * 2, cudaMemcpyHostToDevice);

        int stack_length = 0;
        int previous_stack_length = 0;
        void* previous_h = alloc(kHidden, 2);
        // Pass 0 ends with the states later passes read their messages from.
        std::vector<std::uint16_t> previous_hidden;
        int pass_seen = 0;

        double worst = 0.0;
        int worst_visit = -1;
        for (std::size_t index = 0; index < schedule.visits.size(); ++index) {
            const auto& visit = schedule.visits[index];
            runner.stage = !stages.empty() && visit.layer <= 2;
            runner.stage_layer = visit.layer;
            if (visit.pass != pass_seen) {
                // A new pass restarts from the pre-loop states; the previous pass's last hidden
                // and block stack become the message source.
                pass_seen = visit.pass;
            }
            runner.run_visit(visit.layer, visit.pass, visit.layer - 6,
                             stack_length, previous_stack_length, previous_h);
            const auto status = cudaDeviceSynchronize();
            if (status != cudaSuccess) {
                std::cerr << "visit " << index << ": " << cudaGetErrorString(status) << "\n";
                return 1;
            }
            const auto expected_out = read_f32(oracle / ("visit_out_" + std::to_string(index) + ".bin"));
            std::vector<std::uint16_t> got(kHidden);
            cudaMemcpy(got.data(), runner.s.h, kHidden * 2, cudaMemcpyDeviceToHost);
            double local = 0.0;
            for (std::size_t i = 0; i < got.size(); ++i) {
                local = std::max(local, std::fabs(double(from_bf16(got[i])) - expected_out[i]));
            }
            if (local > worst) { worst = local; worst_visit = static_cast<int>(index); }
            std::cout << "visit " << index << " layer " << visit.layer << " pass " << visit.pass
                      << (config.sliding_layer[static_cast<std::size_t>(visit.layer)] ? " sliding" : " full   ")
                      << " max|err| " << local << "\n";
            if (local > 5.0) {
                std::cerr << "FAIL: visit " << index << " diverged\n";
                return 1;
            }
            if (visit.message_from >= 0) {
                cudaMemcpy(previous_h, runner.s.h, kHidden * 2, cudaMemcpyDeviceToDevice);
            }
            if ((visit.layer % 6) == 0) { ++stack_length; }
        }
        std::cout << "worst visit " << worst_visit << " max|err| " << worst << "\n";
        if (worst > 5.0e-2) {
            std::cerr << "FAIL: tolerance exceeded\n";
            return 1;
        }
        (void)previous_hidden;
        (void)previous_stack_length;
        std::cout << "ok: all visits match the reference\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
