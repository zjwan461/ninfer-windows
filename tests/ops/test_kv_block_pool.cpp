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

// Accumulates one history in chunks that do not line up with the blocks, then finalizes. The
// oracle divides each block by its own column count, which is exactly what a device finalize does;
// that is why a partial trailing block would have to be ignored by the caller, not by the kernel.
int accumulate_case(int dim, int heads, int block_tokens, int total_columns, int chunk) {
    std::vector<float> keys(static_cast<std::size_t>(dim) * heads * total_columns);
    fill_uniform(keys, 0x51abU + static_cast<std::uint32_t>(total_columns + chunk), -1.0F, 1.0F);
    round_to_bf16(keys);

    const int blocks = (total_columns + block_tokens - 1) / block_tokens;

    DeviceBuffer device_sums(static_cast<std::size_t>(dim) * heads * blocks * sizeof(float));
    std::vector<float> zero_sums(static_cast<std::size_t>(dim) * heads * blocks, 0.0F);
    device_sums.copy_from_host(zero_sums.data(), zero_sums.size() * sizeof(float));

    DeviceBuffer device_counts(static_cast<std::size_t>(blocks) * sizeof(std::int32_t));
    std::vector<std::int32_t> zero_counts(static_cast<std::size_t>(blocks), 0);
    device_counts.copy_from_host(zero_counts.data(), zero_counts.size() * sizeof(std::int32_t));

    Tensor sums_tensor(device_sums.p, DType::FP32, {dim, heads, blocks});
    Tensor counts_tensor(device_counts.p, DType::I32, {blocks});

    for (int first = 0; first < total_columns; first += chunk) {
        const int columns = std::min(chunk, total_columns - first);
        std::vector<float> slice(static_cast<std::size_t>(dim) * heads * columns);
        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < dim; ++d) {
                for (int c = 0; c < columns; ++c) {
                    slice[(static_cast<std::size_t>(h) * dim + d) * columns + c] =
                        keys[(static_cast<std::size_t>(h) * dim + d) * total_columns + first + c];
                }
            }
        }
        DeviceBuffer device_slice = to_device_bf16(slice);
        Tensor slice_tensor(device_slice.p, DType::BF16, {dim, heads, columns});
        ops::kv_block_accumulate(slice_tensor, block_tokens, first, sums_tensor, counts_tensor,
                                 nullptr);
    }

    DeviceBuffer device_means(static_cast<std::size_t>(dim) * heads * blocks * sizeof(float));
    Tensor means_tensor(device_means.p, DType::FP32, {dim, heads, blocks});
    ops::kv_block_finalize(sums_tensor, counts_tensor, means_tensor, nullptr);
    cuda_synchronize();

    const std::vector<double> got = from_device_f32(device_means, dim * heads * blocks);
    const std::vector<double> expected =
        reference_mean_keys(keys, dim, heads, total_columns, block_tokens);
    const std::string label = "kv_block_accumulate/finalize D=" + std::to_string(dim) +
                              " H=" + std::to_string(heads) + " T=" +
                              std::to_string(total_columns) + " block=" +
                              std::to_string(block_tokens) + " chunk=" + std::to_string(chunk);
    return verify_pointwise(label, got, expected,
                            PointwiseCriterion{kMeanKeysTolerance, kMeanKeysTolerance});
}

// End to end over the three ops: keys streamed in chunks (as prefill delivers them) -> block sums ->
// means -> cosine scores against a query. The planted block repeats the query direction, so if the
// chunk-to-block mapping or the published layout were wrong, the wrong block would score highest.
int recall_chain_case(int dim, int kv_heads, int q_heads, int block_tokens, int blocks, int chunk) {
    const int total_columns = block_tokens * blocks;
    const std::vector<float> query(static_cast<std::size_t>(dim) * q_heads, 0.0F);
    std::vector<float> query_vec(query.begin(), query.end());
    fill_uniform(query_vec, 0x6d31U, 0.5F, 1.5F);
    round_to_bf16(query_vec);

    const int planted = 3;
    std::vector<float> keys(static_cast<std::size_t>(dim) * kv_heads * total_columns, 0.0F);
    fill_uniform(keys, 0x77c1U, -0.25F, 0.25F);
    // Every column of the planted block repeats the query direction, so the block mean is exactly
    // that direction and the cosine must come out at 1.
    for (int h = 0; h < kv_heads; ++h) {
        for (int d = 0; d < dim; ++d) {
            const float value =
                [&] {
                    const int group = q_heads / kv_heads;
                    float     pooled = 0.0F;
                    for (int r = 0; r < group; ++r) {
                        pooled += query_vec[static_cast<std::size_t>(h * group + r) * dim + d];
                    }
                    return pooled / static_cast<float>(group);
                }();
            for (int c = 0; c < block_tokens; ++c) {
                keys[(static_cast<std::size_t>(h) * dim + d) * total_columns + planted * block_tokens +
                     c] = value;
            }
        }
    }
    round_to_bf16(keys);

    DeviceBuffer device_sums(static_cast<std::size_t>(dim) * kv_heads * blocks * sizeof(float));
    std::vector<float> zero_sums(static_cast<std::size_t>(dim) * kv_heads * blocks, 0.0F);
    device_sums.copy_from_host(zero_sums.data(), zero_sums.size() * sizeof(float));
    DeviceBuffer device_counts(static_cast<std::size_t>(blocks) * sizeof(std::int32_t));
    std::vector<std::int32_t> zero_counts(static_cast<std::size_t>(blocks), 0);
    device_counts.copy_from_host(zero_counts.data(), zero_counts.size() * sizeof(std::int32_t));

    Tensor sums_tensor(device_sums.p, DType::FP32, {dim, kv_heads, blocks});
    Tensor counts_tensor(device_counts.p, DType::I32, {blocks});

    for (int first = 0; first < total_columns; first += chunk) {
        const int columns = std::min(chunk, total_columns - first);
        std::vector<float> slice(static_cast<std::size_t>(dim) * kv_heads * columns);
        for (int h = 0; h < kv_heads; ++h) {
            for (int d = 0; d < dim; ++d) {
                for (int c = 0; c < columns; ++c) {
                    slice[(static_cast<std::size_t>(h) * dim + d) * columns + c] =
                        keys[(static_cast<std::size_t>(h) * dim + d) * total_columns + first + c];
                }
            }
        }
        DeviceBuffer device_slice = to_device_bf16(slice);
        Tensor slice_tensor(device_slice.p, DType::BF16, {dim, kv_heads, columns});
        ops::kv_block_accumulate(slice_tensor, block_tokens, first, sums_tensor, counts_tensor,
                                 nullptr);
    }

    DeviceBuffer device_means(static_cast<std::size_t>(dim) * kv_heads * blocks * sizeof(float));
    Tensor means_tensor(device_means.p, DType::FP32, {dim, kv_heads, blocks});
    ops::kv_block_finalize(sums_tensor, counts_tensor, means_tensor, nullptr);

    DeviceBuffer device_query = to_device_bf16(query_vec);
    DeviceBuffer device_scores(static_cast<std::size_t>(blocks) * sizeof(float));
    Tensor query_tensor(device_query.p, DType::BF16, {dim, q_heads});
    Tensor scores_tensor(device_scores.p, DType::FP32, {blocks});
    ops::kv_block_scores(query_tensor, means_tensor, scores_tensor, nullptr);
    cuda_synchronize();

    const std::vector<double> scores = from_device_f32(device_scores, blocks);
    int best = 0;
    for (int b = 1; b < blocks; ++b) {
        if (scores[static_cast<std::size_t>(b)] > scores[static_cast<std::size_t>(best)]) { best = b; }
    }
    const std::string label = "kv_block_pool recall chain D=" + std::to_string(dim) + " H=" +
                              std::to_string(kv_heads) + " blocks=" + std::to_string(blocks) +
                              " chunk=" + std::to_string(chunk);
    if (best != planted) {
        std::cout << "FAIL: " << label << "\n  planted block " << planted << " scored "
                  << scores[static_cast<std::size_t>(planted)] << " but block " << best << " scored "
                  << scores[static_cast<std::size_t>(best)] << " (highest)\n";
        return 1;
    }
    const double planted_score = scores[static_cast<std::size_t>(planted)];
    if (std::fabs(planted_score - 1.0) > 1e-3) {
        std::cout << "FAIL: " << label << "\n  planted block score " << planted_score
                  << " should be ~1\n";
        return 1;
    }
    return 0;
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
    // Incremental capture: 256-token prefill chunks over 128-token blocks, and chunks that straddle
    // block boundaries (3-column chunks over 4-column blocks).
    failures += accumulate_case(256, 4, 128, 1024, 256);
    failures += accumulate_case(8, 2, 4, 9, 3);
    failures += accumulate_case(8, 2, 4, 12, 5);
    // The three ops composed the way prefill will call them: streamed keys, block means, cosine.
    failures += recall_chain_case(64, 4, 8, 128, 8, 256);
    failures += recall_chain_case(256, 4, 24, 128, 16, 256);

    if (failures == 0) { std::cout << "kv_block_pool: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
