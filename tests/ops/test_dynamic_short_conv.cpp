// Inkling 2 dynamic short convolution: static plus per-token taps, residual, causal state.

#include "ninfer/ops/dynamic_short_conv.h"

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

std::vector<double> oracle(const std::vector<std::uint16_t>& x,
                           const std::vector<std::uint16_t>& static_weight,
                           const std::vector<std::uint16_t>& dynamic_weight,
                           const std::vector<std::uint16_t>& state, int channels, int tokens,
                           int heads, std::vector<std::uint16_t>& state_out) {
    std::vector<double> out(std::size_t(channels) * tokens);
    const int head_width = channels / heads;
    for (int c = 0; c < channels; ++c) {
        std::vector<double> stream;
        for (int i = 0; i < 3; ++i) {
            stream.push_back(from_bf16(state[std::size_t(c) * 3 + i]));
        }
        for (int t = 0; t < tokens; ++t) {
            stream.push_back(from_bf16(x[std::size_t(c) * tokens + t]));
        }
        const int head = c / head_width;
        for (int t = 0; t < tokens; ++t) {
            double acc = 0.0;
            for (int j = 0; j < 4; ++j) {
                const double tap =
                    from_bf16(static_weight[std::size_t(j) * channels + c]) +
                    from_bf16(dynamic_weight[(std::size_t(t) * heads + head) * 4 + std::size_t(j)]);
                acc += tap * stream[std::size_t(t) + std::size_t(j)];
            }
            acc += stream[std::size_t(t) + 3];
            out[std::size_t(c) * tokens + t] = acc / 3.0;
        }
        for (int i = 0; i < 3; ++i) {
            state_out[std::size_t(c) * 3 + i] =
                to_bf16(float(stream[stream.size() - 3 + std::size_t(i)]));
        }
    }
    return out;
}

int run_case(int channels, int tokens, int heads, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 0.4F);
    const auto fill = [&](std::size_t count, float sigma) {
        std::vector<std::uint16_t> data(count);
        for (auto& v : data) {
            std::uint32_t bits = 0;
            const float value = sigma * normal(rng);
            std::memcpy(&bits, &value, 4);
            v = std::uint16_t((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
        }
        return data;
    };

    const auto x = fill(std::size_t(channels) * tokens, 1.0F);
    const auto static_weight = fill(std::size_t(4) * channels, 0.5F);
    const auto dynamic_weight = fill(std::size_t(tokens) * heads * 4, 0.3F);
    const auto state = fill(std::size_t(channels) * 3, 1.0F);

    std::vector<std::uint16_t> expected_state(state.size());
    const auto reference = oracle(x, static_weight, dynamic_weight, state, channels, tokens, heads,
                                  expected_state);

    const auto upload = [](const std::vector<std::uint16_t>& host) {
        void* device = nullptr;
        cudaMalloc(&device, host.size() * 2);
        cudaMemcpy(device, host.data(), host.size() * 2, cudaMemcpyHostToDevice);
        return device;
    };
    void* dx = upload(x);
    void* dsw = upload(static_weight);
    void* ddw = upload(dynamic_weight);
    void* ds = upload(state);
    void* dstate = nullptr;
    void* dout = nullptr;
    cudaMalloc(&dstate, state.size() * 2);
    cudaMalloc(&dout, x.size() * 2);

    Tensor tx(dx, DType::BF16, {channels, tokens});
    Tensor tsw(dsw, DType::BF16, {4, channels});
    Tensor tdw(ddw, DType::BF16, {tokens, heads * 4});
    Tensor ts(ds, DType::BF16, {channels, 3});
    Tensor tstate(dstate, DType::BF16, {channels, 3});
    Tensor tout(dout, DType::BF16, {channels, tokens});

    dynamic_short_conv(tx, tsw, tdw, ts, tstate, tout, heads, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "FAIL: kernel error\n";
        return 1;
    }

    std::vector<std::uint16_t> got(x.size());
    std::vector<std::uint16_t> got_state(state.size());
    cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(got_state.data(), dstate, got_state.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dsw); cudaFree(ddw); cudaFree(ds); cudaFree(dstate); cudaFree(dout);

    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        worst = std::max(worst, std::fabs(double(from_bf16(got[i])) - reference[i]));
    }
    std::size_t mismatch = 0;
    for (std::size_t i = 0; i < got_state.size(); ++i) {
        if (got_state[i] != expected_state[i]) { ++mismatch; }
    }
    if (!(worst <= 2.0e-2) || mismatch != 0) {
        std::cerr << "FAIL: max|err| " << worst << ", state mismatches " << mismatch << "\n";
        return 1;
    }
    std::cout << "case channels=" << channels << " tokens=" << tokens << " heads=" << heads
              << ": max|err| " << worst << ", state exact\n";
    return 0;
}

} // namespace

int main() {
    if (run_case(1024, 8, 8, 1) || run_case(1024, 1, 8, 2) || run_case(256, 5, 2, 3)) {
        return 1;
    }
    std::cout << "ok: dynamic short convolution matches the FP64 oracle\n";
    return 0;
}
