// Inkling 2 value normalisation: RMS without a gain vector, FP64 oracle.

#include "ninfer/ops/head_rmsnorm.h"

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

constexpr float kEps = 1.0e-6F;

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

int run_case(int width, int rows, float sigma, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, sigma);
    std::vector<std::uint16_t> x(std::size_t(width) * rows);
    for (auto& v : x) { v = to_bf16(normal(rng)); }

    std::vector<double> reference(x.size());
    for (int r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int d = 0; d < width; ++d) {
            // ne[0] is the fastest dimension, so element (d, r) lives at r * width + d.
            const double value = from_bf16(x[std::size_t(r) * width + d]);
            sum += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum / double(width) + double(kEps));
        for (int d = 0; d < width; ++d) {
            reference[std::size_t(r) * width + d] = from_bf16(x[std::size_t(r) * width + d]) * inverse;
        }
    }

    void* dx = nullptr;
    void* dout = nullptr;
    cudaMalloc(&dx, x.size() * 2);
    cudaMalloc(&dout, x.size() * 2);
    cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);

    Tensor tx(dx, DType::BF16, {width, rows});
    Tensor tout(dout, DType::BF16, {width, rows});
    head_rmsnorm(tx, kEps, tout, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "FAIL: kernel error\n";
        return 1;
    }

    std::vector<std::uint16_t> got(x.size());
    cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dout);

    // A row scaled to unit RMS is order one, and BF16 keeps one part in 256 of the value, so the
    // bound is relative with a floor for values near zero.
    double worst = 0.0;
    double worst_ratio = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double error = std::fabs(double(from_bf16(got[i])) - reference[i]);
        const double limit = 5.0e-3 + 8.0e-3 * std::fabs(reference[i]);
        worst = std::max(worst, error);
        worst_ratio = std::max(worst_ratio, error / limit);
    }
    if (worst_ratio > 1.0) {
        std::cerr << "FAIL: width " << width << " rows " << rows << " max|err| " << worst << "\n";
        return 1;
    }
    std::cout << "case width=" << width << " rows=" << rows << ": max|err| " << worst << "\n";
    return 0;
}

} // namespace

int main() {
    if (run_case(128, 24, 0.7F, 1) || run_case(128, 1, 3.0F, 2) ||
        run_case(2304, 8, 0.5F, 3) || run_case(512, 5, 1.5F, 4)) {
        return 1;
    }
    std::cout << "ok: unweighted RMSNorm matches the FP64 oracle\n";
    return 0;
}
