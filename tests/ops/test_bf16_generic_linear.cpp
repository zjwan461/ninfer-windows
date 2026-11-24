// The runtime-extent BF16 fallback: every Inkling 2 projection width against an FP64 oracle.

#include "ninfer/ops/linear.h"

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

int run_case(int n, int k, int tokens, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 0.35F);

    std::vector<std::uint16_t> weight(std::size_t(n) * k);
    for (auto& v : weight) { v = to_bf16(normal(rng) * 0.05F); }
    std::vector<std::uint16_t> x(std::size_t(k) * tokens);
    for (auto& v : x) { v = to_bf16(normal(rng)); }

    std::vector<double> reference(std::size_t(n) * tokens);
    for (int row = 0; row < n; ++row) {
        for (int t = 0; t < tokens; ++t) {
            double accumulator = 0.0;
            for (int kk = 0; kk < k; ++kk) {
                accumulator += double(from_bf16(weight[std::size_t(row) * k + kk])) *
                               double(from_bf16(x[std::size_t(kk) * tokens + t]));
            }
            reference[std::size_t(row) * tokens + t] = accumulator;
        }
    }

    void* dw = nullptr;
    void* dx = nullptr;
    void* dout = nullptr;
    cudaMalloc(&dw, weight.size() * 2);
    cudaMalloc(&dx, x.size() * 2);
    cudaMalloc(&dout, reference.size() * 2);
    cudaMemcpy(dw, weight.data(), weight.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);

    Weight w{};
    w.qdata = dw;
    w.qtype = QType::BF16;
    w.n = n;
    w.k = k;
    w.ndim = 2;
    w.shape[0] = n;
    w.shape[1] = k;

    Tensor tx(dx, DType::BF16, {k, tokens});
    Tensor tout(dout, DType::BF16, {n, tokens});
    linear(tx, w, tout, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "FAIL: kernel error\n";
        return 1;
    }

    std::vector<std::uint16_t> got(reference.size());
    cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(dw); cudaFree(dx); cudaFree(dout);

    double worst_ratio = 0.0;
    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double error = std::fabs(double(from_bf16(got[i])) - reference[i]);
        worst = std::max(worst, error);
        worst_ratio = std::max(worst_ratio, error / (1.0e-2 + 0.02 * std::fabs(reference[i])));
    }
    if (worst_ratio > 1.0) {
        std::cerr << "FAIL: n=" << n << " k=" << k << " tokens=" << tokens << " max|err| " << worst
                  << "\n";
        return 1;
    }
    std::cout << "case n=" << n << " k=" << k << " tokens=" << tokens << ": max|err| " << worst
              << "\n";
    return 0;
}

} // namespace

int main() {
    if (run_case(3072, 2304, 4, 1) || run_case(1024, 2304, 4, 2) || run_case(2304, 3072, 2, 3) ||
        run_case(2304, 1152, 3, 4) || run_case(384, 2304, 1, 5) || run_case(2304, 2304, 1, 6)) {
        return 1;
    }
    std::cout << "ok: generic BF16 GEMM matches the FP64 oracle on every Inkling width\n";
    return 0;
}
