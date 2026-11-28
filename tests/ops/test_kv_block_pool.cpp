// Tests for include/ninfer/ops/kv_block_pool.h.
//
// Oracle: an independent FP64 implementation of the documented mathematics over the
// same BF16 inputs the kernels read. Both Ops are reductions whose inputs arrive in
// FP32, so the tolerance is a small absolute bound rather than a bit-exact verdict.
#include "ninfer/ops/kv_block_pool.h"

#include "ops/op_check.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr double kMeanKeysTolerance = 1e-6;
constexpr double kScoresTolerance   = 1e-6;

std::vector<double> reference_mean_keys(const std::vector<float>& keys, int dim, int heads,
                                        int tokens, int block_tokens) {
    const int blocks = (tokens + block_tokens - 1) / block_tokens;
    std::vector<double> means(static_cast<std::size_t>(dim) * heads * blocks, 0.0);
    for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < dim; ++d) {
            const std::size_t row = (static_cast<std::size_t>(h) * dim + d) * tokens;
            for (int b = 0; b < blocks; ++b) {
                const int first = b * block_tokens;
                const int last  = std::min(tokens, first + block_tokens);
                double    acc   = 0.0;
                for (int p = first; p < last; ++p) { acc += static_cast<double>(keys[row + p]); }
                means[(static_cast<std::size_t>(h) * dim + d) * blocks + b] =
                    acc / static_cast<double>(last - first);
            }
        }
    }
    return means;
}

std::vector<double> reference_scores(const std::vector<float>& query, const std::vector<double>& means,
                                     int dim, int kv_heads, int q_heads, int blocks) {
    const int group = q_heads / kv_heads;
    std::vector<double> scores(static_cast<std::size_t>(blocks), 0.0);
    for (int b = 0; b < blocks; ++b) {
        double sum = 0.0;
        for (int h = 0; h < kv_heads; ++h) {
            double dot = 0.0;
            double qn  = 0.0;
            double mn  = 0.0;
            for (int d = 0; d < dim; ++d) {
                double pooled = 0.0;
                for (int r = 0; r < group; ++r) {
                    pooled += static_cast<double>(
                        query[static_cast<std::size_t>(h * group + r) * dim + d]);
                }
                pooled /= static_cast<double>(group);
                const double mean = means[(static_cast<std::size_t>(h) * dim + d) * blocks + b];
                dot += pooled * mean;
                qn += pooled * pooled;
                mn += mean * mean;
            }
            const double denominator = std::sqrt(qn) * std::sqrt(mn);
            sum += denominator > 0.0 ? dot / denominator : 0.0;
        }
        scores[static_cast<std::size_t>(b)] = sum / static_cast<double>(kv_heads);
    }
    return scores;
}

int mean_keys_case(int dim, int heads, int tokens, int block_tokens) {
    std::vector<float> keys(static_cast<std::size_t>(dim) * heads * tokens);
    fill_uniform(keys, 0x51edU + static_cast<std::uint32_t>(dim + heads + tokens + block_tokens),
                 -1.0F, 1.0F);
    round_to_bf16(keys);

    const int          blocks = (tokens + block_tokens - 1) / block_tokens;
    DeviceBuffer       device_keys  = to_device_bf16(keys);
    DeviceBuffer       device_means(static_cast<std::size_t>(dim) * heads * blocks *
                              sizeof(float));
    Tensor keys_tensor(device_keys.p, DType::BF16, {dim, heads, tokens});
    Tensor means_tensor(device_means.p, DType::FP32, {dim, heads, blocks});
    ops::kv_block_mean_keys(keys_tensor, block_tokens, means_tensor, nullptr);
    cuda_synchronize();

    const std::vector<double> got =
        from_device_f32(device_means, static_cast<std::size_t>(dim) * heads * blocks);
    const std::vector<double> expected =
        reference_mean_keys(keys, dim, heads, tokens, block_tokens);
    const std::string label = "kv_block_mean_keys D=" + std::to_string(dim) +
                              " H=" + std::to_string(heads) + " T=" + std::to_string(tokens) +
                              " block=" + std::to_string(block_tokens);
    return verify_pointwise(label, got, expected,
                            PointwiseCriterion{kMeanKeysTolerance, kMeanKeysTolerance});
}

int scores_case(int dim, int kv_heads, int q_heads, int blocks, bool zero_block) {
    std::vector<float> query(static_cast<std::size_t>(dim) * q_heads);
    fill_uniform(query, 0x7a11U + static_cast<std::uint32_t>(dim + kv_heads + q_heads + blocks),
                 0.5F, 1.5F);
    round_to_bf16(query);

    std::vector<float> means_f32(static_cast<std::size_t>(dim) * kv_heads * blocks);
    fill_uniform(means_f32, 0x2c0fU, -1.0F, 1.0F);
    if (zero_block) {
        for (int h = 0; h < kv_heads; ++h) {
            for (int d = 0; d < dim; ++d) {
                means_f32[(static_cast<std::size_t>(h) * dim + d) * blocks] = 0.0F;
            }
        }
    }
    const std::vector<double> means(means_f32.begin(), means_f32.end());

    DeviceBuffer device_query = to_device_bf16(query);
    DeviceBuffer device_means(means_f32.size() * sizeof(float));
    device_means.copy_from_host(means_f32.data(), means_f32.size() * sizeof(float));
    DeviceBuffer device_scores(static_cast<std::size_t>(blocks) * sizeof(float));

    Tensor query_tensor(device_query.p, DType::BF16, {dim, q_heads});
    Tensor means_tensor(device_means.p, DType::FP32, {dim, kv_heads, blocks});
    Tensor scores_tensor(device_scores.p, DType::FP32, {blocks});
    ops::kv_block_scores(query_tensor, means_tensor, scores_tensor, nullptr);
    cuda_synchronize();

    const std::vector<double> got     = from_device_f32(device_scores, blocks);
    const std::vector<double> expected = reference_scores(query, means, dim, kv_heads, q_heads, blocks);
    const std::string label = "kv_block_scores D=" + std::to_string(dim) +
                              " H=" + std::to_string(kv_heads) + "/Q=" + std::to_string(q_heads) +
                              " B=" + std::to_string(blocks) + (zero_block ? " zero-block" : "");
    return verify_pointwise(label, got, expected,
                            PointwiseCriterion{kScoresTolerance, kScoresTolerance});
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "kv_block_pool: CUDA unavailable, skipped\n";
        return 0;
    }

    int failures = 0;
    failures += mean_keys_case(8, 2, 40, 16);
    failures += mean_keys_case(16, 3, 128, 32);
    failures += mean_keys_case(64, 4, 257, 64);
    failures += scores_case(8, 2, 2, 4, false);
    failures += scores_case(16, 4, 8, 3, false);
    failures += scores_case(64, 8, 16, 6, false);
    failures += scores_case(16, 4, 4, 3, true);
    // Real model geometry: head_dim 256, 4 KV heads over 24 query heads, 128-token blocks.
    failures += mean_keys_case(256, 4, 1024, 128);
    failures += scores_case(256, 4, 24, 8, false);

    if (failures == 0) { std::cout << "kv_block_pool: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
