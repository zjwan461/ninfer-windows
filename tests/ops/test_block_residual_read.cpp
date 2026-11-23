// Inkling 2 block attention residual: learned queries reading the block-state stack.

#include "ninfer/ops/block_residual_read.h"

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

int run_case(int queries, int states, int hidden, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);

    std::vector<float> query(std::size_t(queries) * hidden);
    for (auto& v : query) { v = normal(rng) * 0.02F; }
    std::vector<std::uint16_t> stack(std::size_t(states) * hidden);
    for (auto& v : stack) { v = to_bf16(normal(rng)); }

    // FP64 oracle over the represented inputs.
    std::vector<double> normalized(std::size_t(states) * hidden);
    for (int s = 0; s < states; ++s) {
        double sum = 0.0;
        for (int h = 0; h < hidden; ++h) {
            const double value = from_bf16(stack[std::size_t(s) * hidden + h]);
            sum += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum / hidden + double(kEps));
        for (int h = 0; h < hidden; ++h) {
            normalized[std::size_t(s) * hidden + h] =
                from_bf16(stack[std::size_t(s) * hidden + h]) * inverse;
        }
    }
    std::vector<double> reference(std::size_t(queries) * hidden);
    const double scale = 1.0 / std::sqrt(double(hidden));
    for (int q = 0; q < queries; ++q) {
        std::vector<double> logits(static_cast<std::size_t>(states));
        for (int s = 0; s < states; ++s) {
            double dot = 0.0;
            for (int h = 0; h < hidden; ++h) {
                dot += double(query[std::size_t(q) * hidden + h]) *
                       normalized[std::size_t(s) * hidden + h];
            }
            logits[std::size_t(s)] = dot * scale;
        }
        double maximum = logits[0];
        for (int s = 1; s < states; ++s) { maximum = std::max(maximum, logits[std::size_t(s)]); }
        double total = 0.0;
        for (int s = 0; s < states; ++s) {
            logits[std::size_t(s)] = std::exp(logits[std::size_t(s)] - maximum);
            total += logits[std::size_t(s)];
        }
        for (int h = 0; h < hidden; ++h) {
            double accumulator = 0.0;
            for (int s = 0; s < states; ++s) {
                accumulator += logits[std::size_t(s)] / total * normalized[std::size_t(s) * hidden + h];
            }
            reference[std::size_t(q) * hidden + h] = accumulator;
        }
    }

    void* d_query = nullptr;
    void* d_states = nullptr;
    void* d_out = nullptr;
    cudaMalloc(&d_query, query.size() * 4);
    cudaMalloc(&d_states, stack.size() * 2);
    cudaMalloc(&d_out, reference.size() * 2);
    cudaMemcpy(d_query, query.data(), query.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, stack.data(), stack.size() * 2, cudaMemcpyHostToDevice);

    Tensor t_query(d_query, DType::FP32, {queries, hidden});
    Tensor t_states(d_states, DType::BF16, {states, hidden});
    Tensor t_out(d_out, DType::BF16, {queries, hidden});
    block_residual_read(t_query, t_states, kEps, t_out, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "FAIL: kernel error\n";
        return 1;
    }

    std::vector<std::uint16_t> got(reference.size());
    cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_query); cudaFree(d_states); cudaFree(d_out);

    double worst = 0.0;
    double worst_ratio = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double error = std::fabs(double(from_bf16(got[i])) - reference[i]);
        worst = std::max(worst, error);
        worst_ratio = std::max(worst_ratio, error / (2.0e-3 + 0.01 * std::fabs(reference[i])));
    }
    if (worst_ratio > 1.0) {
        std::cerr << "FAIL: queries " << queries << " states " << states << " max|err| " << worst
                  << "\n";
        return 1;
    }
    std::cout << "case queries=" << queries << " states=" << states << ": max|err| " << worst
              << "\n";
    return 0;
}

} // namespace

int main() {
    if (run_case(1, 1, 2304, 1) || run_case(1, 2, 2304, 2) || run_case(3, 4, 2304, 3) ||
        run_case(3, 1, 2304, 4) || run_case(1, 3, 512, 5)) {
        return 1;
    }
    std::cout << "ok: block residual read matches the FP64 oracle\n";
    return 0;
}
