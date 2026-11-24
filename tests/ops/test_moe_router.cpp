// Inkling 2 expert router: sigmoid scoring, selection bias, top-k, renormalise, route scale.

#include "ninfer/ops/moe_router.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
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

int run_case(int experts, int hidden, int tokens, int top_k, float scale, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 0.3F);

    std::vector<std::uint16_t> x(std::size_t(hidden) * tokens);
    for (auto& v : x) { v = to_bf16(normal(rng)); }
    std::vector<float> weight(std::size_t(experts) * hidden);
    for (auto& v : weight) { v = normal(rng) * 0.05F; }
    std::vector<float> bias(static_cast<std::size_t>(experts));
    for (auto& v : bias) { v = normal(rng) * 0.1F; }

    std::vector<std::int32_t> expected_indices(std::size_t(top_k) * tokens);
    std::vector<float> expected_weights(std::size_t(top_k) * tokens);
    for (int t = 0; t < tokens; ++t) {
        std::vector<double> scores(static_cast<std::size_t>(experts));
        for (int e = 0; e < experts; ++e) {
            double dot = 0.0;
            for (int h = 0; h < hidden; ++h) {
                dot += double(from_bf16(x[std::size_t(h) * tokens + t])) *
                       double(weight[std::size_t(e) * hidden + h]);
            }
            scores[std::size_t(e)] = 1.0 / (1.0 + std::exp(-dot));
        }
        std::vector<int> order(static_cast<std::size_t>(experts));
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                          [&](int a, int b) {
                              const double sa = scores[std::size_t(a)] + bias[std::size_t(a)];
                              const double sb = scores[std::size_t(b)] + bias[std::size_t(b)];
                              if (sa != sb) { return sa > sb; }
                              return a < b;
                          });
        double total = 0.0;
        for (int j = 0; j < top_k; ++j) { total += scores[std::size_t(order[std::size_t(j)])]; }
        for (int j = 0; j < top_k; ++j) {
            const int e = order[std::size_t(j)];
            expected_indices[std::size_t(j) * tokens + t] = e;
            expected_weights[std::size_t(j) * tokens + t] =
                float(scores[std::size_t(e)] / (total + 1e-20) * scale);
        }
    }

    void* dx = nullptr;
    void* dw = nullptr;
    void* db = nullptr;
    void* di = nullptr;
    void* dweights = nullptr;
    cudaMalloc(&dx, x.size() * 2);
    cudaMalloc(&dw, weight.size() * 4);
    cudaMalloc(&db, bias.size() * 4);
    cudaMalloc(&di, expected_indices.size() * 4);
    cudaMalloc(&dweights, expected_weights.size() * 4);
    cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, weight.data(), weight.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice);

    Tensor tx(dx, DType::BF16, {hidden, tokens});
    Tensor tw(dw, DType::FP32, {experts, hidden});
    Tensor tb(db, DType::FP32, {experts});
    Tensor ti(di, DType::I32, {top_k, tokens});
    Tensor to(dweights, DType::FP32, {top_k, tokens});

    moe_router(tx, tw, tb, top_k, scale, ti, to, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "FAIL: kernel error\n";
        return 1;
    }

    std::vector<std::int32_t> got_indices(expected_indices.size());
    std::vector<float> got_weights(expected_weights.size());
    cudaMemcpy(got_indices.data(), di, got_indices.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(got_weights.data(), dweights, got_weights.size() * 4, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dw); cudaFree(db); cudaFree(di); cudaFree(dweights);

    std::size_t wrong = 0;
    double worst = 0.0;
    for (std::size_t i = 0; i < got_indices.size(); ++i) {
        if (got_indices[i] != expected_indices[i]) { ++wrong; }
        worst = std::max(worst, std::fabs(double(got_weights[i]) - double(expected_weights[i])));
    }
    if (wrong != 0 || !(worst <= 1e-5)) {
        std::cerr << "FAIL: " << wrong << " index mismatches, max|dweight| " << worst << "\n";
        return 1;
    }
    std::cout << "case experts=" << experts << " hidden=" << hidden << " tokens=" << tokens
              << " top_k=" << top_k << ": indices exact, max|dweight| " << worst << "\n";
    return 0;
}

} // namespace

int main() {
    if (run_case(128, 2304, 4, 8, 8.0F, 1) || run_case(128, 2304, 1, 8, 8.0F, 2) ||
        run_case(16, 64, 8, 4, 2.5F, 3)) {
        return 1;
    }
    std::cout << "ok: router selection and weights match the FP64 oracle\n";
    return 0;
}
