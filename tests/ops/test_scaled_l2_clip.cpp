// Inkling 2 scaled L2 clip: learned per-head scale, clip, optional log scaling, FP64 oracle.

#include "ninfer/ops/scaled_l2_clip.h"

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

struct Case {
    int rows;
    int heads;
    int head_dim;
    float threshold;
    float tau_alpha;
    float tau_floor;
};

std::vector<double> oracle(const std::vector<std::uint16_t>& x,
                           const std::vector<std::uint16_t>& scale,
                           const std::vector<int>& positions, const Case& c) {
    std::vector<double> out(x.size());
    for (int row = 0; row < c.rows; ++row) {
        for (int head = 0; head < c.heads; ++head) {
            const std::size_t base =
                (std::size_t(row) * c.heads + head) * std::size_t(c.head_dim);
            double square_sum = 0.0;
            std::vector<double> value(std::size_t(c.head_dim));
            for (int d = 0; d < c.head_dim; ++d) {
                value[std::size_t(d)] = double(from_bf16(x[base + d])) *
                                        double(from_bf16(scale[std::size_t(head) * c.head_dim + d]));
                square_sum += value[std::size_t(d)] * value[std::size_t(d)];
            }
            const double norm = std::sqrt(square_sum);
            double gain = double(c.threshold) / (norm + 1e-6);
            gain = gain > 1.0 ? 1.0 : gain;
            if (c.tau_alpha > 0.0F) {
                const double ratio = (double(positions[std::size_t(row)]) + 1.0) / c.tau_floor;
                gain *= 1.0 + double(c.tau_alpha) * std::log(ratio > 1.0 ? ratio : 1.0);
            }
            for (int d = 0; d < c.head_dim; ++d) {
                out[base + std::size_t(d)] = value[std::size_t(d)] * gain;
            }
        }
    }
    return out;
}

int run_case(const Case& c, float value_scale, std::uint32_t seed, bool in_place = false) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, value_scale);

    std::vector<std::uint16_t> x(std::size_t(c.rows) * c.heads * c.head_dim);
    for (auto& v : x) { v = to_bf16(normal(rng)); }
    std::vector<std::uint16_t> scale(std::size_t(c.heads) * c.head_dim);
    for (auto& v : scale) { v = to_bf16(0.5F + std::fabs(normal(rng))); }
    std::vector<int> positions(std::size_t(c.rows));
    for (int i = 0; i < c.rows; ++i) { positions[std::size_t(i)] = 100 * i + 8192; }

    const auto reference = oracle(x, scale, positions, c);

    const auto upload = [](const void* host, std::size_t bytes) {
        void* device = nullptr;
        cudaMalloc(&device, bytes);
        cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice);
        return device;
    };
    void* dx = upload(x.data(), x.size() * 2);
    void* ds = upload(scale.data(), scale.size() * 2);
    void* dp = upload(positions.data(), positions.size() * sizeof(int));
    void* dout = nullptr;
    cudaMalloc(&dout, x.size() * 2);

    Tensor tx(dx, DType::BF16, {c.rows, c.heads * c.head_dim});
    Tensor ts(ds, DType::BF16, {c.heads, c.head_dim});
    Tensor tp(dp, DType::I32, {c.rows});
    Tensor tout(dout, DType::BF16, {c.rows, c.heads * c.head_dim});
    if (in_place) {
        // The Op writes each head's elements after reading them, so exact aliasing is allowed and
        // the result must not depend on it.
        cudaMemcpy(dout, dx, x.size() * 2, cudaMemcpyDeviceToDevice);
        tout = Tensor(dout, DType::BF16, {c.rows, c.heads * c.head_dim});
    }

    ScaledL2ClipArgs args;
    args.rows = c.rows;
    args.heads = c.heads;
    args.head_dim = c.head_dim;
    args.threshold = c.threshold;
    args.tau_alpha = c.tau_alpha;
    args.tau_floor = c.tau_floor;

    scaled_l2_clip(tx, ts, args.tau_alpha > 0.0F ? &tp : nullptr, args, tout, nullptr);
    const auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::cerr << "FAIL: kernel returned " << cudaGetErrorString(status) << "\n";
        return 1;
    }

    std::vector<std::uint16_t> got(x.size());
    cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(ds); cudaFree(dp); cudaFree(dout);

    // The output is stored in BF16, so the bound is storage rounding: one part in 256 of the
    // value, plus a floor for values near zero.
    double worst_ratio = 0.0;
    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double error = std::fabs(double(from_bf16(got[i])) - reference[i]);
        const double limit = 2.0e-2 + 0.01 * std::fabs(reference[i]);
        worst = std::max(worst, error);
        worst_ratio = std::max(worst_ratio, error / limit);
    }
    if (worst_ratio > 1.0) {
        std::cerr << "FAIL: max |error| " << worst << " exceeds its BF16 bound (rows " << c.rows
                  << ", head_dim " << c.head_dim << ")\n";
        return 1;
    }
    std::cout << "case rows=" << c.rows << " head_dim=" << c.head_dim
              << " tau=" << (c.tau_alpha > 0.0F ? "on" : "off")
              << (in_place ? " in-place" : "") << ": max|err| " << worst << "\n";
    return 0;
}

} // namespace

int main() {
    if (run_case(Case{3, 24, 128, 22.627417F, 0.0F, 0.0F}, 0.6F, 1) ||
        run_case(Case{5, 24, 128, 22.627417F, 0.1F, 8192.0F}, 2.5F, 2) ||
        run_case(Case{4, 24, 16, 8.0F, 0.0F, 0.0F}, 0.8F, 3) ||
        run_case(Case{4, 24, 16, 8.0F, 0.1F, 8192.0F}, 0.8F, 4) ||
        run_case(Case{1, 8, 128, 22.627417F, 0.0F, 0.0F}, 0.01F, 5) ||
        run_case(Case{2, 8, 128, 22.627417F, 0.0F, 0.0F}, 4.0F, 6, true) ||
        run_case(Case{2, 24, 16, 8.0F, 0.1F, 8192.0F}, 3.0F, 7, true)) {
        return 1;
    }
    std::cout << "ok: scaled L2 clip matches the FP64 oracle\n";
    return 0;
}
