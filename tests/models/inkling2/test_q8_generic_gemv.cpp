// The runtime-extent q8 decode route against a host dequantisation of the same artifact weight.

#include "artifact/reader.h"
#include "core/device.h"
#include "models/inkling2/config.h"
#include "models/inkling2/materialize.h"
#include "models/inkling2/parameters.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

std::uint16_t to_bf16(float v) {
    std::uint32_t b = 0;
    std::memcpy(&b, &v, 4);
    return std::uint16_t((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
}
float from_bf16(std::uint16_t v) {
    const std::uint32_t b = std::uint32_t(v) << 16;
    float o = 0.0F;
    std::memcpy(&o, &b, 4);
    return o;
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path artifact_path;
        int layer_index = 20;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--artifact" && i + 1 < argc) { artifact_path = argv[++i]; }
            else if (arg == "--layer" && i + 1 < argc) { layer_index = std::stoi(argv[++i]); }
            else { throw std::invalid_argument("unknown argument " + arg); }
        }
        if (artifact_path.empty()) {
            std::cout << "SKIP: supply --artifact\n";
            return 77;
        }
        artifact::Reader reader(artifact_path);
        const auto config = models::inkling2::parse_config(reader.directory());
        DeviceContext device;
        auto weights = models::inkling2::materialize_text(reader, config, device);
        const auto& layer = weights.index.layers[static_cast<std::size_t>(layer_index)];

        std::vector<std::uint16_t> host_x(2304);
        for (std::size_t i = 0; i < host_x.size(); ++i) {
            host_x[i] = to_bf16(0.01F * float(static_cast<int>(i * 37 % 101) - 50));
        }
        void* d_x = nullptr;
        cudaMalloc(&d_x, host_x.size() * 2);
        cudaMemcpy(d_x, host_x.data(), host_x.size() * 2, cudaMemcpyHostToDevice);
        Tensor t_x(d_x, DType::BF16, {2304, 1});

        auto weight = models::inkling2::matrix_weight(weights, layer.expert_w13[0]);
        std::cout << "qtype " << int(weight.qtype) << " n " << weight.n << " k " << weight.k
                  << " group " << weight.group << " divisor " << weight.weight_scale_divisor
                  << "\n";
        if (weight.qtype != QType::Q8_G32_FP16) {
            std::cout << "SKIP: layer " << layer_index << " is not q8\n";
            return 77;
        }

        const int k = weight.k;
        const int groups = k / weight.group;
        std::vector<std::int8_t> codes(std::size_t(weight.n) * k);
        std::vector<std::uint16_t> scales(std::size_t(weight.n) * groups);
        cudaMemcpy(codes.data(), weight.qdata, codes.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(scales.data(), weight.scales, scales.size() * 2, cudaMemcpyDeviceToHost);

        void* d_out = nullptr;
        cudaMalloc(&d_out, std::size_t(weight.n) * 2);
        Tensor t_out(d_out, DType::BF16, {weight.n, 1});
        linear(t_x, weight, t_out, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::cerr << "FAIL: " << cudaGetErrorString(cudaGetLastError()) << "\n";
            return 1;
        }
        std::vector<std::uint16_t> got(std::size_t(weight.n));
        cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost);

        double worst = 0.0;
        double worst_ratio = 0.0;
        for (int row = 0; row < weight.n; ++row) {
            double expected = 0.0;
            for (int g = 0; g < groups; ++g) {
                const float scale = __half2float(*reinterpret_cast<const __half*>(
                    &scales[std::size_t(row) * groups + g]));
                double partial = 0.0;
                for (int i = 0; i < weight.group; ++i) {
                    const int index = g * weight.group + i;
                    partial += double(codes[std::size_t(row) * k + index]) *
                               double(from_bf16(host_x[std::size_t(index)]));
                }
                expected += double(scale) * partial;
            }
            const double error = std::fabs(double(from_bf16(got[std::size_t(row)])) - expected);
            worst = std::max(worst, error);
            worst_ratio = std::max(worst_ratio, error / (2.0e-2 + 0.02 * std::fabs(expected)));
        }
        std::cout << "layer " << layer_index << " expert 0: max|err| " << worst << "\n";
        if (worst_ratio > 1.0) {
            std::cerr << "FAIL: q8 route disagrees with the host dequant\n";
            return 1;
        }
        std::cout << "ok: generic q8 route matches the host dequantisation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
