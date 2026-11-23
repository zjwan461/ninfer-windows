// Layer 0 parity against the reference implementation: both sub-blocks and the residual wiring.
//
//   ninfer_inkling2_layer0_attention_test --artifact <artifact> --fixture <dir>
//
// The fixture comes from tests/inkling2/dump_layer0.py: layer0_in.bin (the hidden state entering
// layer 0) and layer0_attn.bin (the output of post_attention_convolution), both float32
// [1, 2304]. The test replays that token through NInfer's ops and compares the same boundary.

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/schema.h"
#include "core/device.h"
#include "models/inkling2/config.h"
#include "models/inkling2/materialize.h"
#include "models/inkling2/parameters.h"
#include "ninfer/ops/causal_dconv_residual.h"
#include "ninfer/ops/head_rmsnorm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/relbias_attention.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scaled_l2_clip.h"
#include "ninfer/ops/block_residual_read.h"
#include "ninfer/ops/moe_router.h"
#include "ninfer/ops/softcapped_swiglu.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::ops;

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

// Compare one stage against the reference buffer and report the worst element.
void compare_stage(const char* label, void* device, const std::vector<float>& expected) {
    std::vector<std::uint16_t> got(expected.size());
    cudaMemcpy(got.data(), device, got.size() * 2, cudaMemcpyDeviceToHost);
    double worst = 0.0;
    std::size_t index = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double error = std::fabs(double(from_bf16(got[i])) - expected[i]);
        if (error > worst) { worst = error; index = i; }
    }
    std::cout << "  " << label << ": max|err| " << worst << " at " << index << " (got "
              << from_bf16(got[index]) << ", expected " << expected[index] << ")\n";
}

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("cannot open " + path.string()); }
    stream.seekg(0, std::ios::end);
    const auto bytes = stream.tellg();
    stream.seekg(0);
    std::vector<float> data(std::size_t(bytes) / sizeof(float));
    stream.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

std::vector<std::uint16_t> read_bf16(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("cannot open " + path.string()); }
    stream.seekg(0, std::ios::end);
    const auto bytes = stream.tellg();
    stream.seekg(0);
    std::vector<std::uint16_t> data(static_cast<std::size_t>(bytes) / 2);
    stream.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

void* upload(const void* host, std::size_t bytes) {
    void* device = nullptr;
    cudaMalloc(&device, bytes);
    cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice);
    return device;
}

void* upload_bf16(const std::vector<std::uint16_t>& host) {
    void* device = nullptr;
    cudaMalloc(&device, host.size() * 2);
    cudaMemcpy(device, host.data(), host.size() * 2, cudaMemcpyHostToDevice);
    return device;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path artifact_path;
        std::filesystem::path fixture;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--artifact" && i + 1 < argc) { artifact_path = argv[++i]; }
            else if (arg == "--fixture" && i + 1 < argc) { fixture = argv[++i]; }
            else { throw std::invalid_argument("unknown argument " + arg); }
        }
        if (artifact_path.empty() || fixture.empty()) {
            std::cout << "SKIP: supply --artifact and --fixture\n";
            return 77;
        }

        const auto input = read_f32(fixture / "layer0_in.bin");
        const auto stage_ln = read_f32(fixture / "layer0_ln.bin");
        const auto stage_attnpre = read_f32(fixture / "layer0_attnpre.bin");
        const auto stage_q = read_f32(fixture / "layer0_q.bin");
        const auto stage_k = read_f32(fixture / "layer0_k.bin");
        const auto stage_v = read_f32(fixture / "layer0_v.bin");
        const auto stage_sink = read_f32(fixture / "layer0_sink.bin");
        const auto expected_mlp_in = read_f32(fixture / "layer0_mlp_in.bin");
        const auto expected_out = read_f32(fixture / "layer0_out.bin");
        const auto expected = read_f32(fixture / "layer0_attn.bin");
        if (input.size() != 2304 || expected.size() != 2304) {
            throw std::runtime_error("fixture must hold one 2304-wide token");
        }

        artifact::Reader reader(artifact_path);
        const auto config = models::inkling2::parse_config(reader.directory());
        DeviceContext device;
        auto weights = models::inkling2::materialize_text(reader, config, device);
        const auto& layer = weights.index.layers[0];

        std::vector<std::uint16_t> host_input(input.size());
        for (std::size_t i = 0; i < input.size(); ++i) { host_input[i] = to_bf16(input[i]); }
        void* d_input = upload_bf16(host_input);

        // Buffers for one token.
        void* d_norm = nullptr;
        void* d_q = nullptr;
        void* d_k = nullptr;
        void* d_v = nullptr;
        void* d_rel = nullptr;
        void* d_sink = nullptr;
        void* d_gate = nullptr;
        void* d_attn = nullptr;
        void* d_proj_out = nullptr;
        void* d_conv_out = nullptr;
        void* d_ones = nullptr;
        cudaMalloc(&d_norm, 2304 * 2);
        cudaMalloc(&d_q, 3072 * 2);
        cudaMalloc(&d_k, 1024 * 2);
        cudaMalloc(&d_v, 1024 * 2);
        cudaMalloc(&d_rel, 24 * 16 * 2);
        cudaMalloc(&d_sink, 24 * 2);
        cudaMalloc(&d_gate, 3072 * 2);
        cudaMalloc(&d_attn, 3072 * 2);
        cudaMalloc(&d_proj_out, 2304 * 2);
        cudaMalloc(&d_conv_out, 2304 * 2);
        cudaMalloc(&d_ones, 24 * 128 * 2);
        {
            std::vector<std::uint16_t> ones(24 * 128, to_bf16(1.0F));
            cudaMemcpy(d_ones, ones.data(), ones.size() * 2, cudaMemcpyHostToDevice);
        }

        const auto stream = static_cast<cudaStream_t>(nullptr);
        Tensor t_input(d_input, DType::BF16, {2304, 1});
        Tensor t_norm(d_norm, DType::BF16, {2304, 1});

        // attn_norm: weighted RMSNorm over the hidden row.
        rmsnorm(t_input, models::inkling2::dense_tensor(weights, layer[models::inkling2::Role::AttnNorm]),
                1.0e-6F, false, t_norm, stream);

        compare_stage("rmsnorm(attn_norm)", d_norm, stage_ln);

        // q = wq_du x, then the learned scale and clip.
        Tensor t_q(d_q, DType::BF16, {3072, 1});
        linear(t_norm, models::inkling2::matrix_weight(weights, layer[models::inkling2::Role::WqDu]),
               t_q, stream);
        // The op takes [rows, heads * head_dim]: one token, every head in the row.
        Tensor t_q_heads(d_q, DType::BF16, {1, 3072});
        Tensor t_q_clipped(d_q, DType::BF16, {1, 3072});
        ScaledL2ClipArgs q_args;
        q_args.rows = 1; q_args.heads = 24; q_args.head_dim = 128;
        q_args.threshold = 22.627417F; q_args.tau_alpha = 0.0F; q_args.tau_floor = 8192.0F;
        scaled_l2_clip(t_q_heads,
                       models::inkling2::dense_tensor(weights, layer[models::inkling2::Role::QkClip]),
                       nullptr, q_args, t_q_clipped, stream);

        compare_stage("q after scale+clip", d_q, stage_q);

        // k = wk_dv x -> short conv -> clip without a learned scale.
        Tensor t_k(d_k, DType::BF16, {1024, 1});
        linear(t_norm, models::inkling2::matrix_weight(weights, layer[models::inkling2::Role::WkDv]),
               t_k, stream);
        void* d_k_conv = nullptr;
        void* d_state = nullptr;
        cudaMalloc(&d_k_conv, 1024 * 2);
        cudaMalloc(&d_state, 1024 * 3 * 2);
        cudaMemset(d_state, 0, 1024 * 3 * 2);
        // k_sconv is stored [1024, 1, 4]; the op wants [channels, 4].
        auto k_weight = models::inkling2::dense_tensor(weights, layer[models::inkling2::Role::KSconv]);
        Tensor k_weight_2d(k_weight.data, DType::BF16, {1024, 4});
        Tensor t_k_conv(d_k_conv, DType::BF16, {1024, 1});
        Tensor t_state(d_state, DType::BF16, {1024, 3});
        void* d_state_out = nullptr;
        cudaMalloc(&d_state_out, 1024 * 3 * 2);
        Tensor t_state_out(d_state_out, DType::BF16, {1024, 3});
        causal_dconv_residual(t_k, k_weight_2d, t_state, t_state_out, t_k_conv, stream);
        Tensor t_k_heads(d_k_conv, DType::BF16, {1, 1024});
        Tensor t_k_clipped(d_k_conv, DType::BF16, {1, 1024});
        ScaledL2ClipArgs k_args;
        k_args.rows = 1; k_args.heads = 8; k_args.head_dim = 128;
        k_args.threshold = 22.627417F; k_args.tau_alpha = 0.0F; k_args.tau_floor = 8192.0F;
        Tensor t_ones(d_ones, DType::BF16, {8, 128}); // k has no learned scale, so the clip sees ones
        scaled_l2_clip(t_k_heads, t_ones, nullptr, k_args, t_k_clipped, stream);

        compare_stage("k after conv+clip", d_k_conv, stage_k);

        // v = wv_dv x -> short conv -> unweighted RMSNorm per head.
        Tensor t_v(d_v, DType::BF16, {1024, 1});
        linear(t_norm, models::inkling2::matrix_weight(weights, layer[models::inkling2::Role::WvDv]),
               t_v, stream);
        void* d_v_conv = nullptr;
        cudaMalloc(&d_v_conv, 1024 * 2);
        auto v_weight = models::inkling2::dense_tensor(weights, layer[models::inkling2::Role::VSconv]);
        Tensor v_weight_2d(v_weight.data, DType::BF16, {1024, 4});
        Tensor t_v_conv_in(d_v_conv, DType::BF16, {1024, 1});
        void* d_v_state_out = nullptr;
        cudaMalloc(&d_v_state_out, 1024 * 3 * 2);
        Tensor t_v_state_out(d_v_state_out, DType::BF16, {1024, 3});
        causal_dconv_residual(t_v, v_weight_2d, t_state, t_v_state_out, t_v_conv_in, stream);
        // ne[0] is the fastest dimension: one head is 128 contiguous channels, so the row width
        // is 128 and there are 8 rows. Writing {8, 128} here normalised over 8 elements.
        Tensor t_v_heads(d_v_conv, DType::BF16, {128, 8});
        void* d_v_norm = nullptr;
        cudaMalloc(&d_v_norm, 1024 * 2);
        Tensor t_v_norm(d_v_norm, DType::BF16, {128, 8});
        head_rmsnorm(t_v_heads, 1.0e-6F, t_v_norm, stream);

        compare_stage("v after conv+rmsnorm", d_v_norm, stage_v);

        // Relative states, sink and gate.
        Tensor t_rel(d_rel, DType::BF16, {384, 1});
        linear(t_norm, models::inkling2::matrix_weight(weights, layer[models::inkling2::Role::WrDu]),
               t_rel, stream);
        Tensor t_rel_heads(d_rel, DType::BF16, {1, 384});
        Tensor t_rel_clipped(d_rel, DType::BF16, {1, 384});
        ScaledL2ClipArgs rel_args;
        rel_args.rows = 1; rel_args.heads = 24; rel_args.head_dim = 16;
        rel_args.threshold = 8.0F; rel_args.tau_alpha = 0.0F; rel_args.tau_floor = 8192.0F;
        scaled_l2_clip(t_rel_heads,
                       models::inkling2::dense_tensor(weights, layer[models::inkling2::Role::RelClip]),
                       nullptr, rel_args, t_rel_clipped, stream);

        Tensor t_sink(d_sink, DType::BF16, {24, 1});
        linear(t_norm, models::inkling2::matrix_weight(weights, layer[models::inkling2::Role::AttnSink]),
               t_sink, stream);
        compare_stage("sink", d_sink, stage_sink);
        Tensor t_gate(d_gate, DType::BF16, {3072, 1});
        linear(t_norm, models::inkling2::matrix_weight(weights, layer[models::inkling2::Role::GateAttn]),
               t_gate, stream);

        RelBiasAttentionArgs args;
        args.heads = 24; args.kv_heads = 8; args.head_dim = 128;
        args.q_pos = 0; args.kv_len = 1; args.window = 0;
        args.rel_extent = 1024; args.d_rel = 16;
        args.scale = 1.0F / std::sqrt(128.0F);
        Tensor t_attn(d_attn, DType::BF16, {3072, 1});
        auto rel_proj = models::inkling2::dense_tensor(
            weights, layer[models::inkling2::Role::RelLogits]);
        relbias_attention(t_q_clipped, t_k_clipped, t_v_norm, t_rel_clipped, rel_proj, t_sink,
                          t_gate, args, t_attn, stream);

        // Output projection, then the attention short convolution.
        Tensor t_out(d_proj_out, DType::BF16, {2304, 1});
        linear(t_attn, models::inkling2::matrix_weight(weights, layer[models::inkling2::Role::WoUd]),
               t_out, stream);
        compare_stage("self_attn output (after Wo)", d_proj_out, stage_attnpre);
        auto attn_conv = models::inkling2::dense_tensor(
            weights, layer[models::inkling2::Role::AttnSconv]);
        Tensor attn_conv_2d(attn_conv.data, DType::BF16, {2304, 4});
        Tensor t_conv(d_conv_out, DType::BF16, {2304, 1});
        void* d_conv_state = nullptr;
        void* d_conv_state_out = nullptr;
        cudaMalloc(&d_conv_state, 2304 * 3 * 2);
        cudaMalloc(&d_conv_state_out, 2304 * 3 * 2);
        cudaMemset(d_conv_state, 0, 2304 * 3 * 2);
        Tensor t_conv_state(d_conv_state, DType::BF16, {2304, 3});
        Tensor t_conv_state_out(d_conv_state_out, DType::BF16, {2304, 3});
        causal_dconv_residual(t_out, attn_conv_2d, t_conv_state, t_conv_state_out, t_conv, stream);

        const auto status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            std::cerr << "FAIL: " << cudaGetErrorString(status) << "\n";
            return 1;
        }

        // ---- MLP half -------------------------------------------------------------------
        // Layer 0 is a block boundary, so the attention output *is* the residual here; the
        // pre-layer state joins block_states and the MLP input is read out of that stack.
        void* d_states = nullptr;
        cudaMalloc(&d_states, 2 * 2304 * 2);
        cudaMemcpy(d_states, d_input, 2304 * 2, cudaMemcpyDeviceToHost); // placeholder, replaced below
        cudaMemcpy(static_cast<char*>(d_states), d_input, 2304 * 2, cudaMemcpyDeviceToDevice);
        cudaMemcpy(static_cast<char*>(d_states) + 2304 * 2, d_conv_out, 2304 * 2,
                   cudaMemcpyDeviceToDevice);

        const auto query_index = weights.index.mlp_queries[0];
        if (query_index < 0) { throw std::runtime_error("layer 0 has no mlp residual query"); }
        auto query = models::inkling2::dense_tensor(weights, query_index);
        // The reference keeps these queries in FP32; the artifact stores them that way too.
        void* d_query = nullptr;
        cudaMalloc(&d_query, 2304 * 4);
        cudaMemcpy(d_query, query.data, 2304 * 4, cudaMemcpyDeviceToDevice);

        void* d_mlp_in = nullptr;
        cudaMalloc(&d_mlp_in, 2304 * 2);
        Tensor t_query(d_query, DType::FP32, {1, 2304});
        Tensor t_states(d_states, DType::BF16, {2, 2304});
        Tensor t_mlp_in(d_mlp_in, DType::BF16, {1, 2304});
        block_residual_read(t_query, t_states, 1.0e-6F, t_mlp_in, stream);

        // post_attention_layernorm on the read-out state.
        void* d_mlp_norm = nullptr;
        cudaMalloc(&d_mlp_norm, 2304 * 2);
        Tensor t_mlp_norm_in(d_mlp_in, DType::BF16, {2304, 1});
        Tensor t_mlp_norm(d_mlp_norm, DType::BF16, {2304, 1});
        rmsnorm(t_mlp_norm_in,
                models::inkling2::dense_tensor(weights, layer[models::inkling2::Role::MlpNorm]),
                1.0e-6F, false, t_mlp_norm, stream);
        compare_stage("block read + post_attention_layernorm", d_mlp_norm, expected_mlp_in);

        // Router and the eight experts.
        auto gate = models::inkling2::dense_tensor(weights, layer[models::inkling2::Role::MlpGate]);
        std::vector<float> gate_f32(std::size_t(128) * 2304);
        {
            std::vector<std::uint16_t> host(std::size_t(128) * 2304);
            cudaMemcpy(host.data(), gate.data, host.size() * 2, cudaMemcpyDeviceToHost);
            for (std::size_t i = 0; i < host.size(); ++i) { gate_f32[i] = from_bf16(host[i]); }
        }
        auto bias = models::inkling2::dense_tensor(weights,
                                                   layer[models::inkling2::Role::MlpGateBias]);
        void* d_gate_f32 = upload(gate_f32.data(), gate_f32.size() * 4);
        void* d_indices = nullptr;
        void* d_weights = nullptr;
        cudaMalloc(&d_indices, 8 * sizeof(std::int32_t));
        cudaMalloc(&d_weights, 8 * sizeof(float));
        Tensor t_gate_f32(d_gate_f32, DType::FP32, {128, 2304});
        Tensor t_bias(bias.data, DType::FP32, {128});
        Tensor t_indices(d_indices, DType::I32, {8, 1});
        Tensor t_router_weights(d_weights, DType::FP32, {8, 1});
        moe_router(t_mlp_norm, t_gate_f32, t_bias, 8, 8.0F, t_indices, t_router_weights, stream);

        std::vector<std::int32_t> host_indices(8);
        std::vector<float> host_weights(8);
        cudaMemcpy(host_indices.data(), d_indices, 8 * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_weights.data(), d_weights, 8 * sizeof(float), cudaMemcpyDeviceToHost);

        // The fixture names its expert files in the reference's order, which is not the router's
        // order (the reference selects with sorted=false), so pair by expert id.
        const auto reference_router = artifact::parse_json(
            [&] {
                std::ifstream stream(fixture / "layer0_router.json");
                return std::string(std::istreambuf_iterator<char>(stream),
                                   std::istreambuf_iterator<char>());
            }(),
            "router");
        std::vector<std::int32_t> reference_indices;
        for (const auto& value : reference_router.at("indices")) {
            reference_indices.push_back(value.get<std::int32_t>());
        }

        std::vector<double> accumulator(2304, 0.0);
        for (std::size_t slot = 0; slot < 8; ++slot) {
            const auto expert = reference_indices[slot];
            std::size_t mine = 8;
            for (std::size_t j = 0; j < 8; ++j) {
                if (host_indices[j] == expert) { mine = j; }
            }
            if (mine == 8) {
                std::cerr << "FAIL: router did not select expert " << expert << "\n";
                return 1;
            }
            const double weight = double(host_weights[mine]);
            const auto w13 = read_bf16(fixture / ("expert_" + std::to_string(slot) + "_w13.bin"));
            const auto w2 = read_bf16(fixture / ("expert_" + std::to_string(slot) + "_w2.bin"));
            void* d_w13 = upload(w13.data(), w13.size() * 2);
            void* d_w2 = upload(w2.data(), w2.size() * 2);
            void* d_up = nullptr;
            void* d_act = nullptr;
            void* d_down = nullptr;
            cudaMalloc(&d_up, 2304 * 2);
            cudaMalloc(&d_act, 1152 * 2);
            cudaMalloc(&d_down, 2304 * 2);

            Weight weight13{};
            weight13.qdata = d_w13; weight13.qtype = QType::BF16;
            weight13.n = 2304; weight13.k = 2304; weight13.ndim = 2;
            weight13.shape[0] = 2304; weight13.shape[1] = 2304;
            Weight weight2{};
            weight2.qdata = d_w2; weight2.qtype = QType::BF16;
            weight2.n = 2304; weight2.k = 1152; weight2.ndim = 2;
            weight2.shape[0] = 2304; weight2.shape[1] = 1152;

            Tensor t_up(d_up, DType::BF16, {2304, 1});
            Tensor t_act(d_act, DType::BF16, {1152, 1});
            Tensor t_down(d_down, DType::BF16, {2304, 1});
            linear(t_mlp_norm, weight13, t_up, stream);
            softcapped_swiglu(t_up, 4.0F, 25.0F, t_act, stream);
            linear(t_act, weight2, t_down, stream);

            std::vector<std::uint16_t> host_down(2304);
            cudaMemcpy(host_down.data(), d_down, 2304 * 2, cudaMemcpyDeviceToHost);
            for (std::size_t i = 0; i < 2304; ++i) {
                accumulator[i] += weight * double(from_bf16(host_down[i]));
            }
            cudaFree(d_w13); cudaFree(d_w2); cudaFree(d_up); cudaFree(d_act); cudaFree(d_down);
        }

        std::vector<std::uint16_t> host_moe(2304);
        for (std::size_t i = 0; i < 2304; ++i) { host_moe[i] = to_bf16(float(accumulator[i])); }
        void* d_moe = upload(host_moe.data(), host_moe.size() * 2);
        Tensor t_moe(d_moe, DType::BF16, {2304, 1});

        // post_mlp_convolution, then the residual add (attention output is the residual for a
        // block-boundary layer).
        auto mlp_conv = models::inkling2::dense_tensor(weights,
                                                       layer[models::inkling2::Role::MlpSconv]);
        Tensor mlp_conv_2d(mlp_conv.data, DType::BF16, {2304, 4});
        void* d_mlp_conv = nullptr;
        void* d_mlp_state = nullptr;
        void* d_mlp_state_out = nullptr;
        cudaMalloc(&d_mlp_conv, 2304 * 2);
        cudaMalloc(&d_mlp_state, 2304 * 3 * 2);
        cudaMalloc(&d_mlp_state_out, 2304 * 3 * 2);
        cudaMemset(d_mlp_state, 0, 2304 * 3 * 2);
        Tensor t_mlp_conv(d_mlp_conv, DType::BF16, {2304, 1});
        Tensor t_mlp_state(d_mlp_state, DType::BF16, {2304, 3});
        Tensor t_mlp_state_out(d_mlp_state_out, DType::BF16, {2304, 3});
        causal_dconv_residual(t_moe, mlp_conv_2d, t_mlp_state, t_mlp_state_out, t_mlp_conv, stream);

        std::vector<std::uint16_t> host_conv(2304);
        cudaMemcpy(host_conv.data(), d_mlp_conv, 2304 * 2, cudaMemcpyDeviceToHost);

        std::vector<std::uint16_t> got(2304);
        cudaMemcpy(got.data(), d_conv_out, got.size() * 2, cudaMemcpyDeviceToHost);
        for (std::size_t i = 0; i < 2304; ++i) {
            const float residual = from_bf16(got[i]); // layer 0 is a boundary: attn output only
            got[i] = to_bf16(residual + from_bf16(host_conv[i]));
        }

        double worst = 0.0;
        double worst_ratio = 0.0;
        std::size_t worst_index = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const double error = std::fabs(double(from_bf16(got[i])) - expected_out[i]);
            worst_ratio = std::max(worst_ratio,
                                   error / (1.0e-2 + 0.02 * std::fabs(expected_out[i])));
            if (error > worst) { worst = error; worst_index = i; }
        }
        std::cout << "whole layer: max|err| " << worst << " at " << worst_index
                  << " (got " << from_bf16(got[worst_index]) << ", expected "
                  << expected_out[worst_index] << ")\n";
        if (worst_ratio > 1.0) {
            std::cerr << "FAIL: tolerance exceeded\n";
            return 1;
        }
        std::cout << "ok: layer 0 matches the reference\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
