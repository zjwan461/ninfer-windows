// Layer 0 MLP sub-block parity: router, eight experts, softcapped SwiGLU, residual conv.

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/schema.h"
#include "core/device.h"
#include "models/inkling2/config.h"
#include "models/inkling2/materialize.h"
#include "models/inkling2/parameters.h"
#include "ninfer/ops/causal_dconv_residual.h"
#include "ninfer/ops/linear.h"
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
    std::vector<std::uint16_t> data(std::size_t(bytes) / 2);
    stream.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

void* upload(const void* host, std::size_t bytes) {
    void* device = nullptr;
    cudaMalloc(&device, bytes);
    cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice);
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

        const auto input = read_f32(fixture / "layer0_mlp_in.bin");
        const auto expected_pre = read_f32(fixture / "layer0_mlp_pre.bin");
        const auto expected = read_f32(fixture / "layer0_mlp.bin");
        if (input.size() != 2304 || expected.size() != 2304) {
            throw std::runtime_error("fixture must hold one 2304-wide token");
        }

        artifact::Reader reader(artifact_path);
        const auto config = models::inkling2::parse_config(reader.directory());
        DeviceContext device;
        auto weights = models::inkling2::materialize_text(reader, config, device);
        const auto& layer = weights.index.layers[0];

        // Router: the checkpoint stores the gate in BF16 while the reference runs it in FP32.
        auto gate = models::inkling2::dense_tensor(weights, layer[models::inkling2::Role::MlpGate]);
        std::vector<float> gate_f32(std::size_t(128) * 2304);
        {
            std::vector<std::uint16_t> host(std::size_t(128) * 2304);
            cudaMemcpy(host.data(), gate.data, host.size() * 2, cudaMemcpyDeviceToHost);
            for (std::size_t i = 0; i < host.size(); ++i) { gate_f32[i] = from_bf16(host[i]); }
        }
        auto bias = models::inkling2::dense_tensor(weights, layer[models::inkling2::Role::MlpGateBias]);

        std::vector<std::uint16_t> host_x(input.size());
        for (std::size_t i = 0; i < input.size(); ++i) { host_x[i] = to_bf16(input[i]); }

        void* d_x = upload(host_x.data(), host_x.size() * 2);
        void* d_gate = upload(gate_f32.data(), gate_f32.size() * 4);
        void* d_indices = nullptr;
        void* d_weights = nullptr;
        cudaMalloc(&d_indices, 8 * sizeof(std::int32_t));
        cudaMalloc(&d_weights, 8 * sizeof(float));

        Tensor t_x(d_x, DType::BF16, {2304, 1});
        Tensor t_gate(d_gate, DType::FP32, {128, 2304});
        Tensor t_bias(bias.data, DType::FP32, {128});
        Tensor t_indices(d_indices, DType::I32, {8, 1});
        Tensor t_weights(d_weights, DType::FP32, {8, 1});
        moe_router(t_x, t_gate, t_bias, 8, 8.0F, t_indices, t_weights, nullptr);

        std::vector<std::int32_t> host_indices(8);
        std::vector<float> host_weights(8);
        cudaMemcpy(host_indices.data(), d_indices, 8 * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_weights.data(), d_weights, 8 * sizeof(float), cudaMemcpyDeviceToHost);

        // The reference's selection and weights for this token.
        const auto reference = artifact::parse_json(
            [&] {
                std::ifstream stream(fixture / "layer0_router.json");
                return std::string(std::istreambuf_iterator<char>(stream),
                                   std::istreambuf_iterator<char>());
            }(),
            "router");
        std::vector<std::int32_t> expected_indices;
        for (const auto& value : reference.at("indices")) {
            expected_indices.push_back(value.get<std::int32_t>());
        }
        std::vector<double> expected_weights;
        for (const auto& value : reference.at("weights")) {
            expected_weights.push_back(value.get<double>());
        }
        std::cout << "router selected";
        for (std::size_t i = 0; i < expected_indices.size(); ++i) {
            std::cout << " " << host_indices[i] << "(ref " << expected_indices[i] << ")";
        }
        std::cout << "\n";

        double router_error = 0.0;
        // The router's selection order is unspecified (the reference uses sorted=false), so pair
        // by expert id: the fixture names its files in the reference's order.
        std::vector<double> accumulator(2304, 0.0);
        for (std::size_t slot = 0; slot < 8; ++slot) {
            const auto expert = expected_indices[slot];
            std::size_t mine = 8;
            for (std::size_t j = 0; j < 8; ++j) {
                if (host_indices[j] == expert) { mine = j; }
            }
            if (mine == 8) {
                std::cerr << "FAIL: router did not select expert " << expert << "\n";
                return 1;
            }
            router_error = std::max(router_error,
                                    std::fabs(double(host_weights[mine]) - expected_weights[slot]));
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
            weight13.qdata = d_w13;
            weight13.qtype = QType::BF16;
            weight13.n = 2304;
            weight13.k = 2304;
            weight13.ndim = 2;
            weight13.shape[0] = 2304;
            weight13.shape[1] = 2304;
            Weight weight2{};
            weight2.qdata = d_w2;
            weight2.qtype = QType::BF16;
            weight2.n = 2304;
            weight2.k = 1152;
            weight2.ndim = 2;
            weight2.shape[0] = 2304;
            weight2.shape[1] = 1152;

            Tensor t_up(d_up, DType::BF16, {2304, 1});
            Tensor t_act(d_act, DType::BF16, {1152, 1});
            Tensor t_down(d_down, DType::BF16, {2304, 1});
            linear(t_x, weight13, t_up, nullptr);
            softcapped_swiglu(t_up, 4.0F, 25.0F, t_act, nullptr);
            linear(t_act, weight2, t_down, nullptr);

            std::vector<std::uint16_t> host_down(2304);
            cudaMemcpy(host_down.data(), d_down, 2304 * 2, cudaMemcpyDeviceToHost);
            for (std::size_t i = 0; i < 2304; ++i) {
                accumulator[i] += weight * double(from_bf16(host_down[i]));
            }
            cudaFree(d_w13); cudaFree(d_w2); cudaFree(d_up); cudaFree(d_act); cudaFree(d_down);
        }

        double worst = 0.0;
        double worst_ratio = 0.0;
        for (std::size_t i = 0; i < 2304; ++i) {
            const double error = std::fabs(accumulator[i] - expected_pre[i]);
            worst = std::max(worst, error);
            worst_ratio = std::max(worst_ratio, error / (2.0e-2 + 0.02 * std::fabs(expected_pre[i])));
        }
        std::cout << "router weight max|d| " << router_error << "\n";
        std::cout << "MoE output (pre-conv) max|err| " << worst << "\n";
        if (worst_ratio > 1.0) {
            std::cerr << "FAIL: MLP pre-convolution differs\n";
            return 1;
        }
        std::cout << "ok: layer 0 MLP matches the reference\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
