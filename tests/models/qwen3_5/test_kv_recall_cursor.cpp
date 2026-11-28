// Tests for the device recall cursor: pre-rope keys in, published block means out, checked against an
// independent double-precision oracle on the host.
#include "models/qwen3_5/program/storage/kv_recall_cursor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::Tensor;
using ninfer::models::qwen3_5::detail::KvRecallCursor;

int expect(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        return 1;
    }
    return 0;
}

template <typename Call>
int expect_throws(Call&& call, const char* label) {
    try {
        call();
    } catch (const std::exception&) {
        return 0;
    }
    std::cerr << "FAIL: " << label << " (nothing thrown)\n";
    return 1;
}

// One column of keys laid out [head_dim, kv_heads, columns]; the value depends on layer, position and
// coordinate so that a misplaced column or a swapped layer cannot pass the comparison.
float value_at(int layer, int position, int head, int dim, int head_dim) {
    return static_cast<float>(layer * 1000 + position * 10 + head * head_dim + dim);
}

struct Keys {
    std::vector<__nv_bfloat16> host;
    void* device = nullptr;
    int head_dim = 0;
    int kv_heads = 0;
    int columns  = 0;

    Keys(int layer, int first_position, int columns_in, int head_dim_in, int kv_heads_in)
        : head_dim(head_dim_in), kv_heads(kv_heads_in), columns(columns_in) {
        host.resize(static_cast<std::size_t>(head_dim) * kv_heads * columns);
        for (int c = 0; c < columns; ++c) {
            for (int h = 0; h < kv_heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    const std::size_t index = static_cast<std::size_t>(h) * head_dim + d +
                                              static_cast<std::size_t>(head_dim) * kv_heads * c;
                    host[index] =
                        __float2bfloat16(value_at(layer, first_position + c, h, d, head_dim));
                }
            }
        }
        cudaMalloc(&device, host.size() * sizeof(__nv_bfloat16));
        cudaMemcpy(device, host.data(), host.size() * sizeof(__nv_bfloat16),
                   cudaMemcpyHostToDevice);
    }
    ~Keys() { cudaFree(device); }
    [[nodiscard]] Tensor tensor() const { return Tensor(device, DType::BF16, {head_dim, kv_heads, columns}); }
};

// Oracle: the mean over each block's columns, in the layout the cursor publishes.
double oracle_mean(int layer, int block, int block_tokens, int head, int dim, int head_dim) {
    double sum = 0.0;
    for (int c = 0; c < block_tokens; ++c) {
        // The captured keys are stored as bf16, so the oracle must average the represented values,
        // not the real ones.
        sum += static_cast<double>(__bfloat162float(
            __float2bfloat16(value_at(layer, block * block_tokens + c, head, dim, head_dim))));
    }
    return sum / static_cast<double>(block_tokens);
}

int accumulate_and_read_case() {
    int failures = 0;
    constexpr int kLayers = 2;
    constexpr int kHeadDim = 2;
    constexpr int kKvHeads = 1;
    constexpr int kBlock = 2;
    constexpr int kCapacity = 4;

    KvRecallCursor cursor(kLayers, kHeadDim, kKvHeads, kBlock, kCapacity);
    failures += expect(cursor.complete_blocks() == 0, "nothing captured yet");

    // Two turns of two columns each: each turn completes one block.
    for (int turn = 0; turn < 2; ++turn) {
        Keys keys(0, turn * 2, 2, kHeadDim, kKvHeads);
        for (int layer = 0; layer < kLayers; ++layer) {
            Keys layer_keys(layer, turn * 2, 2, kHeadDim, kKvHeads);
            cursor.capture_layer(layer, layer_keys.tensor(), nullptr);
        }
        failures += expect(cursor.complete_blocks() == turn + 1, "a completed turn publishes a block");
    }
    failures += expect(cursor.captured_columns() == 4, "four columns captured");

    failures += expect_throws([&] { (void)cursor.block_means(0); },
                              "means are unavailable before finish()");

    cursor.finish(nullptr);
    for (int layer = 0; layer < kLayers; ++layer) {
        const std::vector<float>& means = cursor.block_means(layer);
        failures += expect(means.size() == static_cast<std::size_t>(kHeadDim) * kKvHeads * kCapacity,
                           "means keep the block-capacity stride");
        for (int block = 0; block < 2; ++block) {
            for (int h = 0; h < kKvHeads; ++h) {
                for (int d = 0; d < kHeadDim; ++d) {
                    const std::size_t index = static_cast<std::size_t>(h) * kHeadDim + d +
                                              static_cast<std::size_t>(kHeadDim) * kKvHeads * block;
                    const double want = oracle_mean(layer, block, kBlock, h, d, kHeadDim);
                    if (std::fabs(static_cast<double>(means[index]) - want) > 1e-5) {
                        std::cerr << "FAIL: layer " << layer << " block " << block << " head " << h
                                  << " dim " << d << ": got " << means[index] << " want " << want
                                  << '\n';
                        ++failures;
                    }
                }
            }
        }
    }

    cursor.reset();
    failures += expect(cursor.complete_blocks() == 0, "reset forgets every block");
    failures += expect(cursor.captured_columns() == 0, "reset forgets the column count");
    return failures;
}

// Queries are offered as [head_dim, query_heads, columns] and pooled over each KV head's group and
// over the tail of the columns (at most kQueryTokens of them).
float query_value_at(int layer, int position, int head, int dim, int head_dim) {
    return static_cast<float>(layer * 100 + position * 3 + head * head_dim + dim);
}

struct Queries {
    std::vector<__nv_bfloat16> host;
    void* device = nullptr;
    int head_dim = 0;
    int query_heads = 0;
    int columns = 0;

    Queries(int layer, int first_position, int columns_in, int head_dim_in, int query_heads_in)
        : head_dim(head_dim_in), query_heads(query_heads_in), columns(columns_in) {
        host.resize(static_cast<std::size_t>(head_dim) * query_heads * columns);
        for (int c = 0; c < columns; ++c) {
            for (int h = 0; h < query_heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    const std::size_t index = static_cast<std::size_t>(h) * head_dim + d +
                                              static_cast<std::size_t>(head_dim) * query_heads * c;
                    host[index] = __float2bfloat16(query_value_at(
                        layer, first_position + c, h, d, head_dim));
                }
            }
        }
        cudaMalloc(&device, host.size() * sizeof(__nv_bfloat16));
        cudaMemcpy(device, host.data(), host.size() * sizeof(__nv_bfloat16),
                   cudaMemcpyHostToDevice);
    }
    ~Queries() { cudaFree(device); }
    [[nodiscard]] Tensor tensor() const {
        return Tensor(device, DType::BF16, {head_dim, query_heads, columns});
    }
};

int query_case() {
    int failures = 0;
    constexpr int kHeadDim = 2;
    constexpr int kKvHeads = 2;
    constexpr int kGroup = 2;
    constexpr int kQueryHeads = kKvHeads * kGroup;
    constexpr int kColumns = 5;

    KvRecallCursor cursor(1, kHeadDim, kKvHeads, 4, 4);
    failures += expect_throws([&] { (void)cursor.query(0); }, "no query before capture");

    Queries queries(0, 0, kColumns, kHeadDim, kQueryHeads);
    cursor.capture_query(0, queries.tensor(), nullptr);
    const float* pooled = cursor.query(0);
    for (int kv = 0; kv < kKvHeads; ++kv) {
        for (int d = 0; d < kHeadDim; ++d) {
            double sum = 0.0;
            for (int c = 0; c < kColumns; ++c) {
                for (int r = 0; r < kGroup; ++r) {
                    const int head = kv * kGroup + r;
                    sum += static_cast<double>(__bfloat162float(__float2bfloat16(
                        query_value_at(0, c, head, d, kHeadDim))));
                }
            }
            const double want = sum / static_cast<double>(kGroup * kColumns);
            const double got = static_cast<double>(pooled[kv * kHeadDim + d]);
            if (std::fabs(got - want) > 1e-5) {
                std::cerr << "FAIL: query kv " << kv << " dim " << d << ": got " << got
                          << " want " << want << '\n';
                ++failures;
            }
        }
    }

    // A prompt longer than the pool keeps only its tail: the head of the columns must not be seen.
    constexpr int kLongColumns = 600;
    KvRecallCursor capped(1, 1, 1, 4, 4);
    Queries long_queries(0, 0, kLongColumns, 1, 1);
    capped.capture_query(0, long_queries.tensor(), nullptr);
    double tail_sum = 0.0;
    const int pooled_columns = KvRecallCursor::kQueryTokens;
    for (int c = kLongColumns - pooled_columns; c < kLongColumns; ++c) {
        tail_sum += static_cast<double>(
            __bfloat162float(__float2bfloat16(query_value_at(0, c, 0, 0, 1))));
    }
    const double tail_want = tail_sum / static_cast<double>(pooled_columns);
    failures += expect(std::fabs(static_cast<double>(capped.query(0)[0]) - tail_want) < 1e-5,
                       "only the tail of a long prompt is pooled");

    failures += expect_throws([&] { (void)capped.query(1); }, "a layer outside the cursor");
    cursor.reset();
    failures += expect_throws([&] { (void)cursor.query(0); }, "reset forgets the query");
    return failures;
}

int block_scores_case() {
    int failures = 0;
    constexpr int kHeadDim = 2;
    constexpr int kKvHeads = 1;
    constexpr int kQueryHeads = 2;
    constexpr int kBlock = 2;
    constexpr int kCapacity = 3;
    constexpr int kColumns = 6;

    KvRecallCursor cursor(1, kHeadDim, kKvHeads, kBlock, kCapacity);
    // The query pools to (1, 0) because both query heads carry the same direction.
    std::vector<__nv_bfloat16> query_host(static_cast<std::size_t>(kHeadDim) * kQueryHeads * 2);
    for (int c = 0; c < 2; ++c) {
        for (int h = 0; h < kQueryHeads; ++h) {
            query_host[static_cast<std::size_t>(h) * kHeadDim + 0] = __float2bfloat16(3.0F);
            query_host[static_cast<std::size_t>(h) * kHeadDim + 1] = __float2bfloat16(0.0F);
        }
    }
    void* query_device = nullptr;
    cudaMalloc(&query_device, query_host.size() * sizeof(__nv_bfloat16));
    cudaMemcpy(query_device, query_host.data(), query_host.size() * sizeof(__nv_bfloat16),
               cudaMemcpyHostToDevice);
    cursor.capture_query(0, Tensor(query_device, DType::BF16, {kHeadDim, kQueryHeads, 2}), nullptr);

    // Block 0 points along the second axis, block 1 along the query direction, block 2 against it:
    // the cosines are exactly 0, 1 and -1.
    std::vector<__nv_bfloat16> keys_host(static_cast<std::size_t>(kHeadDim) * kKvHeads * kColumns);
    auto put = [&](int column, int d, float value) {
        keys_host[static_cast<std::size_t>(d) + static_cast<std::size_t>(kHeadDim) * column] =
            __float2bfloat16(value);
    };
    for (int c = 0; c < 2; ++c) {
        put(c, 0, 0.0F);
        put(c, 1, 2.0F);
    }
    for (int c = 2; c < 4; ++c) {
        put(c, 0, 2.0F);
        put(c, 1, 0.0F);
    }
    for (int c = 4; c < 6; ++c) {
        put(c, 0, -1.0F);
        put(c, 1, 0.0F);
    }
    void* keys_device = nullptr;
    cudaMalloc(&keys_device, keys_host.size() * sizeof(__nv_bfloat16));
    cudaMemcpy(keys_device, keys_host.data(), keys_host.size() * sizeof(__nv_bfloat16),
               cudaMemcpyHostToDevice);
    cursor.capture_layer(0, Tensor(keys_device, DType::BF16, {kHeadDim, kKvHeads, kColumns}),
                         nullptr);

    failures += expect_throws([&] { (void)cursor.block_scores(); }, "scores need finish() first");
    cursor.finish(nullptr);
    const std::vector<float> scores = cursor.block_scores();
    failures += expect(scores.size() == 3, "one score per complete block");
    const float want[3] = {0.0F, 1.0F, -1.0F};
    for (int block = 0; block < 3; ++block) {
        if (std::fabs(scores[static_cast<std::size_t>(block)] - want[block]) > 1e-4F) {
            std::cerr << "FAIL: block " << block << " score " << scores[block] << " want "
                      << want[block] << '\n';
            ++failures;
        }
    }
    cudaFree(query_device);
    cudaFree(keys_device);
    return failures;
}

int partial_block_case() {
    int failures = 0;
    constexpr int kHeadDim = 1;
    constexpr int kKvHeads = 1;
    constexpr int kBlock = 4;
    constexpr int kCapacity = 2;
    KvRecallCursor cursor(1, kHeadDim, kKvHeads, kBlock, kCapacity);

    Keys keys(0, 0, 3, kHeadDim, kKvHeads);
    cursor.capture_layer(0, keys.tensor(), nullptr);
    failures += expect(cursor.captured_columns() == 3, "three columns captured");
    failures += expect(cursor.complete_blocks() == 0, "a partial block publishes nothing");

    // A chunk that is not block aligned is still usable: the later turn completes the block.
    Keys more(0, 3, 1, kHeadDim, kKvHeads);
    cursor.capture_layer(0, more.tensor(), nullptr);
    failures += expect(cursor.complete_blocks() == 1, "the block completes across turns");
    cursor.finish(nullptr);

    const std::vector<float>& means = cursor.block_means(0);
    const double want = (static_cast<double>(value_at(0, 0, 0, 0, kHeadDim)) +
                         static_cast<double>(value_at(0, 1, 0, 0, kHeadDim)) +
                         static_cast<double>(value_at(0, 2, 0, 0, kHeadDim)) +
                         static_cast<double>(value_at(0, 3, 0, 0, kHeadDim))) /
                        4.0;
    failures += expect(std::fabs(static_cast<double>(means[0]) - want) < 1e-5,
                       "the completed block averages across turns");
    return failures;
}

int invalid_case() {
    int failures = 0;
    failures += expect_throws([] { KvRecallCursor cursor(0, 2, 1, 2, 2); }, "zero layers");
    failures += expect_throws([] { KvRecallCursor cursor(1, 0, 1, 2, 2); }, "zero head_dim");
    failures += expect_throws([] { KvRecallCursor cursor(1, 2, 0, 2, 2); }, "zero kv_heads");
    failures += expect_throws([] { KvRecallCursor cursor(1, 2, 1, 0, 2); }, "zero block_tokens");
    failures += expect_throws([] { KvRecallCursor cursor(1, 2, 1, 2, 0); }, "zero capacity");

    KvRecallCursor cursor(2, 2, 1, 2, 2);
    Keys keys(0, 0, 2, 2, 1);
    failures += expect_throws([&] { cursor.capture_layer(2, keys.tensor(), nullptr); },
                              "a layer outside the cursor");
    failures += expect_throws([&] { cursor.capture_layer(-1, keys.tensor(), nullptr); },
                              "a negative layer");
    failures += expect_throws(
        [&] {
            cursor.capture_layer(0, Tensor(keys.device, DType::FP32, {2, 1, 2}), nullptr);
        },
        "float keys");

    // Layer 0 offers two columns; layer 1 offering four means the caller skipped a layer.
    cursor.capture_layer(0, keys.tensor(), nullptr);
    Keys wider(0, 2, 4, 2, 1);
    failures += expect_throws([&] { cursor.capture_layer(1, wider.tensor(), nullptr); },
                              "layers that disagree on the column count");

    // Capacity: two blocks of two columns in a cursor that holds two, fed one more turn.
    KvRecallCursor tight(1, 2, 1, 2, 2);
    for (int turn = 0; turn < 3; ++turn) {
        Keys chunk(0, turn * 2, 2, 2, 1);
        if (turn < 2) {
            tight.capture_layer(0, chunk.tensor(), nullptr);
        } else {
            failures += expect_throws([&] { tight.capture_layer(0, chunk.tensor(), nullptr); },
                                      "a block past the capacity");
        }
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += accumulate_and_read_case();
    failures += query_case();
    failures += block_scores_case();
    failures += partial_block_case();
    failures += invalid_case();

    if (failures == 0) { std::cout << "kv_recall_cursor: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}