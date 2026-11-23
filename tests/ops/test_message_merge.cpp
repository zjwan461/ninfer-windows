// Inkling 2 cross-pass message merge against an FP64 oracle.

#include "ninfer/ops/message_merge.h"

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

int run_case(std::int64_t count, float sigma, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, sigma);
    const auto fill = [&] {
        std::vector<std::uint16_t> data(static_cast<std::size_t>(count));
        for (auto& v : data) { v = to_bf16(normal(rng)); }
        return data;
    };
    const auto x = fill();
    const auto message = fill();
    const auto forget = fill();
    const auto gain = fill();

    std::vector<double> reference(static_cast<std::size_t>(count));
    for (std::size_t i = 0; i < std::size_t(count); ++i) {
        const double keep = 1.0 - std::tanh(double(from_bf16(forget[i])));
        const double add = std::tanh(double(from_bf16(gain[i])));
        reference[i] = keep * double(from_bf16(x[i])) + add * double(from_bf16(message[i]));
    }

    const auto upload = [](const std::vector<std::uint16_t>& host) {
        void* device = nullptr;
        cudaMalloc(&device, host.size() * 2);
        cudaMemcpy(device, host.data(), host.size() * 2, cudaMemcpyHostToDevice);
        return device;
    };
    void* dx = upload(x);
    void* dm = upload(message);
    void* df = upload(forget);
    void* dg = upload(gain);
    void* dout = nullptr;
    cudaMalloc(&dout, std::size_t(count) * 2);

    Tensor tx(dx, DType::BF16, {static_cast<std::int32_t>(count)});
    Tensor tm(dm, DType::BF16, {static_cast<std::int32_t>(count)});
    Tensor tf(df, DType::BF16, {static_cast<std::int32_t>(count)});
    Tensor tg(dg, DType::BF16, {static_cast<std::int32_t>(count)});
    Tensor tout(dout, DType::BF16, {static_cast<std::int32_t>(count)});

    message_merge(tx, tm, tf, tg, tout, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "FAIL: kernel error\n";
        return 1;
    }

    std::vector<std::uint16_t> got(static_cast<std::size_t>(count));
    cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dm); cudaFree(df); cudaFree(dg); cudaFree(dout);

    double worst = 0.0;
    double worst_ratio = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double error = std::fabs(double(from_bf16(got[i])) - reference[i]);
        worst = std::max(worst, error);
        worst_ratio = std::max(worst_ratio, error / (5.0e-3 + 8.0e-3 * std::fabs(reference[i])));
    }
    if (worst_ratio > 1.0) {
        std::cerr << "FAIL: count " << count << " max|err| " << worst << "\n";
        return 1;
    }
    std::cout << "case count=" << count << " max|err| " << worst << "\n";
    return 0;
}

} // namespace

int main() {
    if (run_case(2304, 0.8F, 1) || run_case(2304, 3.0F, 2) || run_case(1, 0.5F, 3) ||
        run_case(2304 * 16, 1.2F, 4)) {
        return 1;
    }
    std::cout << "ok: message merge matches the FP64 oracle\n";
    return 0;
}
