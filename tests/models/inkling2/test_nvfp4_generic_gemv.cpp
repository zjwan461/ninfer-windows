// The runtime-extent NVFP4 decode route against the artifact's own expert weights.
//
// The fixture's expert_<slot>_w13.bin is the same expert dequantised to BF16 by the reference, so
// running the quantised artifact weight through the generic route and comparing against the
// dequantised matrix times the same token checks the code plane, the swizzled scale plane and the
// per-expert divisor at once.

#include "artifact/reader.h"
#include "artifact/schema.h"
#include "core/device.h"
#include "models/inkling2/config.h"
#include "models/inkling2/materialize.h"
#include "models/inkling2/parameters.h"
#include "ninfer/ops/linear.h"

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

        artifact::Reader reader(artifact_path);
        const auto config = models::inkling2::parse_config(reader.directory());
        DeviceContext device;
        auto weights = models::inkling2::materialize_text(reader, config, device);
        const auto& layer = weights.index.layers[0];

        const auto reference = artifact::parse_json(
            [&] {
                std::ifstream stream(fixture / "layer0_router.json");
                return std::string(std::istreambuf_iterator<char>(stream),
                                   std::istreambuf_iterator<char>());
            }(),
            "router");
        std::vector<std::int32_t> indices;
        for (const auto& value : reference.at("indices")) {
            indices.push_back(value.get<std::int32_t>());
        }

        std::vector<std::uint16_t> host_x(2304);
        for (std::size_t i = 0; i < host_x.size(); ++i) {
            host_x[i] = to_bf16(0.01F * float(static_cast<int>(i * 37 % 101) - 50));
        }
        void* d_x = nullptr;
        cudaMalloc(&d_x, host_x.size() * 2);
        cudaMemcpy(d_x, host_x.data(), host_x.size() * 2, cudaMemcpyHostToDevice);
        Tensor t_x(d_x, DType::BF16, {2304, 1});

        double worst_ratio = 0.0;
        double worst = 0.0;
        for (std::size_t slot = 0; slot < 4; ++slot) {
            const auto expert = indices[slot];
            const auto w13 = read_bf16(fixture / ("expert_" + std::to_string(slot) + "_w13.bin"));

            std::vector<double> expected(2304, 0.0);
            for (int row = 0; row < 2304; ++row) {
                double accumulator = 0.0;
                for (int kk = 0; kk < 2304; ++kk) {
                    accumulator += double(from_bf16(w13[std::size_t(row) * 2304 + kk])) *
                                   double(from_bf16(host_x[std::size_t(kk)]));
                }
                expected[std::size_t(row)] = accumulator;
            }

            void* d_out = nullptr;
            cudaMalloc(&d_out, 2304 * 2);
            Tensor t_out(d_out, DType::BF16, {2304, 1});
            auto weight = models::inkling2::matrix_weight(weights, layer.expert_w13[expert]);
            linear(t_x, weight, t_out, nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess) {
                std::cerr << "FAIL: " << cudaGetErrorString(cudaGetLastError()) << "\n";
                return 1;
            }
            std::vector<std::uint16_t> got(2304);
            cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost);
            cudaFree(d_out);

            double local = 0.0;
            for (std::size_t i = 0; i < got.size(); ++i) {
                const double error = std::fabs(double(from_bf16(got[i])) - expected[i]);
                local = std::max(local, error);
                worst_ratio = std::max(worst_ratio,
                                       error / (2.0e-2 + 0.02 * std::fabs(expected[i])));
            }
            worst = std::max(worst, local);
            std::cout << "expert " << expert << " (slot " << slot << "): max|err| " << local
                      << "\n";
        }
        if (worst_ratio > 1.0) {
            std::cerr << "FAIL: worst " << worst << "\n";
            return 1;
        }
        std::cout << "ok: generic NVFP4 route reproduces the dequantised experts\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
