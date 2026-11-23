// Inkling 2 softcapped SwiGLU against an FP64 oracle.

#include "ninfer/ops/softcapped_swiglu.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;

namespace {

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

int run_case(int inter, int tokens, float sigma, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, sigma);
    std::vector<std::uint16_t> fused(std::size_t(2) * inter * tokens);
    for (auto& v : fused) { v = to_bf16(normal(rng)); }

    std::vector<double> reference(std::size_t(inter) * tokens);
    for (int i = 0; i < inter; ++i) {
        for (int t = 0; t < tokens; ++t) {
            const double gate = from_bf16(fused[std::size_t(i) * tokens + t]);
            const double up = from_bf16(fused[std::size_t(inter + i) * tokens + t]);
            const double capped_gate = std::tanh(gate / 4.0) * 4.0;
            const double capped_up = std::tanh(up / 25.0) * 25.0;
            const double sigmoid = 1.0 / (1.0 + std::exp(-gate));
            reference[std::size_t(i) * tokens + t] = capped_gate * sigmoid * capped_up;
        }
    }

    void* din = nullptr;
    void* dout = nullptr;
    cudaMalloc(&din, fused.size() * 2);
    cudaMalloc(&dout, std::size_t(inter) * tokens * 2);
    cudaMemcpy(din, fused.data(), fused.size() * 2, cudaMemcpyHostToDevice);

    Tensor tin(din, DType::BF16, {2 * inter, tokens});
    Tensor tout(dout, DType::BF16, {inter, tokens});
    softcapped_swiglu(tin, 4.0F, 25.0F, tout, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "FAIL: kernel error\n";
        return 1;
    }

    std::vector<std::uint16_t> got(std::size_t(inter) * tokens);
    cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(din);
    cudaFree(dout);

    double worst = 0.0;
    double worst_ratio = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double error = std::fabs(double(from_bf16(got[i])) - reference[i]);
        worst = std::max(worst, error);
        worst_ratio = std::max(worst_ratio, error / (2.0e-2 + 0.01 * std::fabs(reference[i])));
    }
    if (worst_ratio > 1.0) {
        std::cerr << "FAIL: inter " << inter << " tokens " << tokens << " max|err| " << worst
                  << "\n";
        return 1;
    }
    std::cout << "case inter=" << inter << " tokens=" << tokens << ": max|err| " << worst << "\n";
    return 0;
}

} // namespace

int main() {
    if (run_case(1152, 6, 0.5F, 1) || run_case(1152, 1, 8.0F, 2) || run_case(64, 16, 3.0F, 3)) {
        return 1;
    }
    std::cout << "ok: softcapped SwiGLU matches the FP64 oracle\n";
    return 0;
}
