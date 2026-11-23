// Inkling 2 static short convolution: residual, 1/sqrt(5) scaling, causal state carry.

#include "ninfer/ops/causal_dconv_residual.h"

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
    const std::uint32_t rounded = (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
    return static_cast<std::uint16_t>(rounded);
}

float from_bf16(std::uint16_t value) {
    const std::uint32_t bits = std::uint32_t(value) << 16;
    float out = 0.0F;
    std::memcpy(&out, &bits, 4);
    return out;
}

// FP64 oracle: state holds the three values before the chunk, output and new state are returned.
std::vector<double> oracle(const std::vector<std::uint16_t>& x,
                           const std::vector<std::uint16_t>& weight,
                           const std::vector<std::uint16_t>& state, int channels, int tokens,
                           std::vector<std::uint16_t>& state_out) {
    std::vector<double> out(std::size_t(channels) * tokens);
    std::vector<double> tail(std::size_t(channels) * 3);
    const double norm = 1.0 / std::sqrt(5.0);
    for (int c = 0; c < channels; ++c) {
        std::vector<double> stream;
        for (int i = 0; i < 3; ++i) {
            stream.push_back(from_bf16(state[std::size_t(c) * 3 + i]));
        }
        for (int t = 0; t < tokens; ++t) {
            stream.push_back(from_bf16(x[std::size_t(c) * tokens + t]));
        }
        for (int t = 0; t < tokens; ++t) {
            double acc = 0.0;
            for (int j = 0; j < 4; ++j) {
                acc += from_bf16(weight[std::size_t(c) * 4 + j]) * stream[std::size_t(t) + j];
            }
            acc += stream[std::size_t(t) + 3]; // the residual is x itself
            out[std::size_t(c) * tokens + t] = acc * norm;
        }
        for (int i = 0; i < 3; ++i) {
            tail[std::size_t(c) * 3 + i] = stream[stream.size() - 3 + std::size_t(i)];
            state_out[std::size_t(c) * 3 + i] = to_bf16(float(tail[std::size_t(c) * 3 + i]));
        }
    }
    return out;
}

int run_case(int channels, int tokens, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 0.5F);

    std::vector<std::uint16_t> x(std::size_t(channels) * tokens);
    for (auto& v : x) { v = to_bf16(normal(rng)); }
    std::vector<std::uint16_t> weight(std::size_t(channels) * 4);
    for (auto& v : weight) { v = to_bf16(normal(rng)); }
    std::vector<std::uint16_t> state(std::size_t(channels) * 3);
    for (auto& v : state) { v = to_bf16(normal(rng)); }

    std::vector<std::uint16_t> expected_state(state.size());
    const auto reference = oracle(x, weight, state, channels, tokens, expected_state);

    const auto upload = [](const std::vector<std::uint16_t>& host) {
        void* device = nullptr;
        cudaMalloc(&device, host.size() * 2);
        cudaMemcpy(device, host.data(), host.size() * 2, cudaMemcpyHostToDevice);
        return device;
    };
    void* dx = upload(x);
    void* dw = upload(weight);
    void* ds = upload(state);
    void* dstate = nullptr;
    void* dout = nullptr;
    cudaMalloc(&dstate, state.size() * 2);
    cudaMalloc(&dout, x.size() * 2);

    Tensor tx(dx, DType::BF16, {channels, tokens});
    Tensor tw(dw, DType::BF16, {channels, 4});
    Tensor ts(ds, DType::BF16, {channels, 3});
    Tensor tstate(dstate, DType::BF16, {channels, 3});
    Tensor tout(dout, DType::BF16, {channels, tokens});

    causal_dconv_residual(tx, tw, ts, tstate, tout, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "FAIL: kernel error\n";
        return 1;
    }

    std::vector<std::uint16_t> got(x.size());
    std::vector<std::uint16_t> got_state(state.size());
    cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(got_state.data(), dstate, got_state.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dw); cudaFree(ds); cudaFree(dstate); cudaFree(dout);

    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        worst = std::max(worst, std::fabs(double(from_bf16(got[i])) - reference[i]));
    }
    std::size_t state_mismatch = 0;
    for (std::size_t i = 0; i < got_state.size(); ++i) {
        if (got_state[i] != expected_state[i]) { ++state_mismatch; }
    }
    if (!(worst <= 2.0e-2) || state_mismatch != 0) {
        std::cerr << "FAIL: max|err| " << worst << ", state mismatches " << state_mismatch
                  << " (channels " << channels << ", tokens " << tokens << ")\n";
        return 1;
    }
    std::cout << "case channels=" << channels << " tokens=" << tokens << ": max|err| " << worst
              << ", state exact\n";
    return 0;
}

} // namespace

int main() {
    if (run_case(1024, 8, 1) || run_case(1024, 1, 2) || run_case(2304, 5, 3) ||
        run_case(64, 32, 4)) {
        return 1;
    }
    std::cout << "ok: short convolution matches the FP64 oracle and carries the exact state\n";
    return 0;
}
