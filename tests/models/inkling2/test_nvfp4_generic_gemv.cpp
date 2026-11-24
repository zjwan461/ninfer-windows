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
#include "ops/linear/nvfp4/nvfp4_launch.h"

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

            // The gather route reads the same weight from a device table by slot; it must produce
            // the same bits, because the indirection is the only thing that differs.
            {
                void* table = nullptr;
                cudaMalloc(&table, sizeof(Weight));
                cudaMemcpy(table, &weight, sizeof(Weight), cudaMemcpyHostToDevice);
                void* d_gathered = nullptr;
                cudaMalloc(&d_gathered, got.size() * 2);
                ninfer::Tensor t_gathered(d_gathered, ninfer::DType::BF16, {2304, 1});
                void* d_slots = nullptr;
                cudaMalloc(&d_slots, sizeof(std::int32_t));
                cudaMemset(d_slots, 0, sizeof(std::int32_t));
                ninfer::ops::detail::launch_nvfp4_gather_gemv(
                    t_x, static_cast<const Weight*>(table), static_cast<const std::int32_t*>(d_slots),
                    0, weight, t_gathered, nullptr);
                cudaFree(d_slots);
                if (cudaDeviceSynchronize() != cudaSuccess) {
                    std::cerr << "FAIL: gather route: " << cudaGetErrorString(cudaGetLastError())
                              << "\n";
                    return 1;
                }
                std::vector<std::uint16_t> gathered(got.size());
                cudaMemcpy(gathered.data(), d_gathered, gathered.size() * 2,
                           cudaMemcpyDeviceToHost);
                // Against the reference dequantisation, not against the other route: the direct
                // route may use a shape-tuned kernel with a different reduction order, so the last
                // bit can legitimately differ. The reference is the thing both must reproduce.
                double gather_error = 0.0;
                double gather_ratio = 0.0;
                for (std::size_t i = 0; i < gathered.size(); ++i) {
                    const double error = std::fabs(double(from_bf16(gathered[i])) - expected[i]);
                    gather_error = std::max(gather_error, error);
                    gather_ratio = std::max(gather_ratio,
                                            error / (2.0e-2 + 0.02 * std::fabs(expected[i])));
                }
                if (gather_ratio > 1.0) {
                    std::cerr << "FAIL: gather route max|err| " << gather_error << "\n";
                    return 1;
                }
                worst_ratio = std::max(worst_ratio, gather_ratio);
                worst = std::max(worst, gather_error);
                cudaFree(d_gathered);
                cudaFree(table);
            }

            // The down matrix is a different geometry (hidden x inter, not 2*inter x hidden) and it
            // was the one the session's device path got wrong, so it is checked against its own
            // fixture rather than assumed to follow the gate/up case.
            {
                const auto w2 = read_bf16(fixture / ("expert_" + std::to_string(slot) + "_w2.bin"));
                std::vector<double> expected_down(2304, 0.0);
                std::vector<std::uint16_t> ones_input(1152, 0x3F80);  // 1.0 in BF16
                for (int row = 0; row < 2304; ++row) {
                    double accumulator = 0.0;
                    for (int kk = 0; kk < 1152; ++kk) {
                        accumulator += double(from_bf16(w2[std::size_t(row) * 1152 + kk]));
                    }
                    expected_down[std::size_t(row)] = accumulator;
                }
                void* d_ones = nullptr;
                cudaMalloc(&d_ones, 1152 * 2);
                cudaMemcpy(d_ones, ones_input.data(), 1152 * 2, cudaMemcpyHostToDevice);
                void* d_down = nullptr;
                cudaMalloc(&d_down, 2304 * 2);
                void* down_table = nullptr;
                cudaMalloc(&down_table, sizeof(ninfer::Weight));
                const auto down_weight =
                    models::inkling2::matrix_weight(weights, layer.expert_w2[expert]);
                cudaMemcpy(down_table, &down_weight, sizeof(ninfer::Weight), cudaMemcpyHostToDevice);
                void* d_zero = nullptr;
                cudaMalloc(&d_zero, sizeof(std::int32_t));
                cudaMemset(d_zero, 0, sizeof(std::int32_t));
                ninfer::Tensor t_ones(d_ones, ninfer::DType::BF16, {1152, 1});
                ninfer::Tensor t_down(d_down, ninfer::DType::BF16, {2304, 1});
                ninfer::ops::detail::launch_nvfp4_gather_gemv(
                    t_ones, static_cast<const ninfer::Weight*>(down_table),
                    static_cast<const std::int32_t*>(d_zero), 0, down_weight, t_down, nullptr);
                if (cudaDeviceSynchronize() != cudaSuccess) {
                    std::cerr << "FAIL: down gather: " << cudaGetErrorString(cudaGetLastError())
                              << "\n";
                    return 1;
                }
                std::vector<std::uint16_t> down_out(2304);
                cudaMemcpy(down_out.data(), d_down, 2304 * 2, cudaMemcpyDeviceToHost);
                double down_error = 0.0;
                for (std::size_t i = 0; i < down_out.size(); ++i) {
                    down_error = std::max(down_error, std::fabs(double(from_bf16(down_out[i])) -
                                                                expected_down[i]));
                }
                std::cout << "expert " << expert << " down max|err| " << down_error << "\n";
                if (down_error > 0.05) {
                    std::cerr << "FAIL: down gather max|err| " << down_error << "\n";
                    return 1;
                }
                cudaFree(d_zero);
                cudaFree(down_table);
                cudaFree(d_down);
                cudaFree(d_ones);
            }

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
