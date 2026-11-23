// Inkling 2 decode attention: relative bias, sink and gate against an FP64 oracle.

#include "ninfer/ops/relbias_attention.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
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

struct Geometry {
    int heads;
    int kv_heads;
    int head_dim;
    int kv_len;
    int q_pos;
    int window;
    int rel_extent;
    int d_rel;
};

std::vector<double> oracle(const std::vector<std::uint16_t>& q,
                           const std::vector<std::uint16_t>& k,
                           const std::vector<std::uint16_t>& v,
                           const std::vector<std::uint16_t>& rel,
                           const std::vector<std::uint16_t>& proj,
                           const std::vector<std::uint16_t>& sink,
                           const std::vector<std::uint16_t>& gate, const Geometry& g,
                           float scale) {
    std::vector<double> out(std::size_t(g.heads) * g.head_dim);
    const int group = g.heads / g.kv_heads;
    for (int head = 0; head < g.heads; ++head) {
        const int kv_head = head / group;
        const int first = (g.window > 0 && g.q_pos - g.window + 1 > 0) ? g.q_pos - g.window + 1 : 0;

        std::vector<double> logits;
        for (int position = first; position <= g.q_pos; ++position) {
            double dot = 0.0;
            for (int d = 0; d < g.head_dim; ++d) {
                const std::size_t qi = std::size_t(head) * g.head_dim + d;
                const std::size_t ki = (std::size_t(kv_head) * g.kv_len + position) * g.head_dim + d;
                dot += double(from_bf16(q[qi])) * double(from_bf16(k[ki]));
            }
            double bias = 0.0;
            const int distance = g.q_pos - position;
            if (distance >= 0 && distance < g.rel_extent) {
                for (int r = 0; r < g.d_rel; ++r) {
                    bias += double(from_bf16(rel[std::size_t(head) * g.d_rel + r])) *
                            double(from_bf16(proj[std::size_t(r) * g.rel_extent + distance]));
                }
            }
            logits.push_back(dot * double(scale) + bias);
        }

        // The sink is one extra logit with a zero value.
        double maximum = double(from_bf16(sink[head]));
        for (const double logit : logits) { maximum = std::max(maximum, logit); }
        double denominator = std::exp(double(from_bf16(sink[head])) - maximum);
        std::vector<double> weights(logits.size());
        for (std::size_t i = 0; i < logits.size(); ++i) {
            weights[i] = std::exp(logits[i] - maximum);
            denominator += weights[i];
        }
        for (int d = 0; d < g.head_dim; ++d) {
            double acc = 0.0;
            for (std::size_t i = 0; i < weights.size(); ++i) {
                const int position = first + int(i);
                const std::size_t vi = (std::size_t(kv_head) * g.kv_len + position) * g.head_dim + d;
                acc += weights[i] * double(from_bf16(v[vi]));
            }
            const std::size_t oi = std::size_t(head) * g.head_dim + d;
            const double gated = 1.0 / (1.0 + std::exp(-double(from_bf16(gate[oi]))));
            out[oi] = acc / denominator * gated;
        }
    }
    return out;
}

int run_case(const Geometry& g, float scale, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 0.5F);

    const auto fill = [&](std::size_t count) {
        std::vector<std::uint16_t> data(count);
        for (auto& value : data) { value = to_bf16(normal(rng)); }
        return data;
    };

    const auto q = fill(std::size_t(g.heads) * g.head_dim);
    const auto k = fill(std::size_t(g.kv_heads) * g.kv_len * g.head_dim);
    const auto v = fill(std::size_t(g.kv_heads) * g.kv_len * g.head_dim);
    const auto rel = fill(std::size_t(g.heads) * g.d_rel);
    const auto proj = fill(std::size_t(g.d_rel) * g.rel_extent);
    const auto sink = fill(std::size_t(g.heads));
    const auto gate = fill(std::size_t(g.heads) * g.head_dim);

    const auto reference = oracle(q, k, v, rel, proj, sink, gate, g, scale);

    const auto upload = [](const std::vector<std::uint16_t>& host) {
        void* device = nullptr;
        const std::size_t bytes = host.size() * sizeof(std::uint16_t);
        cudaMalloc(&device, bytes);
        cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice);
        return device;
    };
    void* dq = upload(q);
    void* dk = upload(k);
    void* dv = upload(v);
    void* drel = upload(rel);
    void* dproj = upload(proj);
    void* dsink = upload(sink);
    void* dgate = upload(gate);
    void* dout = nullptr;
    cudaMalloc(&dout, q.size() * sizeof(std::uint16_t));

    Tensor tq(dq, DType::BF16, {g.heads * g.head_dim});
    Tensor tk(dk, DType::BF16, {g.kv_heads, g.kv_len, g.head_dim});
    Tensor tv(dv, DType::BF16, {g.kv_heads, g.kv_len, g.head_dim});
    Tensor trel(drel, DType::BF16, {g.heads, g.d_rel});
    Tensor tproj(dproj, DType::BF16, {g.d_rel, g.rel_extent});
    Tensor tsink(dsink, DType::BF16, {g.heads});
    Tensor tgate(dgate, DType::BF16, {g.heads * g.head_dim});
    Tensor tout(dout, DType::BF16, {g.heads * g.head_dim});

    RelBiasAttentionArgs args;
    args.heads = g.heads;
    args.kv_heads = g.kv_heads;
    args.head_dim = g.head_dim;
    args.q_pos = g.q_pos;
    args.kv_len = g.kv_len;
    args.window = g.window;
    args.rel_extent = g.rel_extent;
    args.d_rel = g.d_rel;
    args.scale = scale;

    relbias_attention(tq, tk, tv, trel, tproj, tsink, tgate, args, tout, nullptr);
    const auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::cerr << "FAIL: kernel returned " << cudaGetErrorString(status) << "\n";
        return 1;
    }

    std::vector<std::uint16_t> got(q.size());
    cudaMemcpy(got.data(), dout, got.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(drel); cudaFree(dproj);
    cudaFree(dsink); cudaFree(dgate); cudaFree(dout);

    double worst = 0.0;
    std::size_t worst_index = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double value = double(from_bf16(got[i]));
        const double error = std::fabs(value - reference[i]);
        if (error > worst) { worst = error; worst_index = i; }
    }
    // BF16 storage of the output dominates the error budget; the arithmetic runs in FP32.
    const double limit = 2.0e-2;
    if (!(worst <= limit)) {
        std::cerr << "FAIL: max |error| " << worst << " at " << worst_index << " (got "
                  << from_bf16(got[worst_index]) << ", expected " << reference[worst_index]
                  << ")\n";
        return 1;
    }
    std::cout << "case heads=" << g.heads << " kv=" << g.kv_heads << " len=" << g.kv_len
              << " pos=" << g.q_pos << " window=" << g.window << ": max|err| " << worst << "\n";
    return 0;
}

} // namespace

int main() {
    const float scale = 1.0F / std::sqrt(128.0F);
    Geometry full{8, 2, 128, 64, 40, 0, 1024, 16};
    Geometry windowed{8, 2, 128, 64, 40, 16, 1024, 16};
    Geometry single{4, 1, 128, 1, 0, 0, 1024, 16};
    Geometry long_distance{4, 4, 128, 2048, 1500, 0, 1024, 16};
    if (run_case(full, scale, 1) || run_case(windowed, scale, 2) ||
        run_case(single, scale, 3) || run_case(long_distance, scale, 4)) {
        return 1;
    }
    std::cout << "ok: relative bias, sink and gate match the FP64 oracle\n";
    return 0;
}
