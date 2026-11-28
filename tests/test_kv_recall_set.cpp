// Tests for src/core/kv_recall_set.h: per-layer block summaries and the single ranking they fold to.
#include "core/kv_recall_set.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::core::KvRecallSet;

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

// One column of keys for a given layer: the value encodes the layer so that a set that mixed the
// layers up would not produce the expected means.
std::vector<float> column(int head_dim, int kv_heads, int layer, int position) {
    std::vector<float> keys(static_cast<std::size_t>(head_dim) * kv_heads);
    for (int h = 0; h < kv_heads; ++h) {
        for (int d = 0; d < head_dim; ++d) {
            keys[static_cast<std::size_t>(h) * head_dim + d] =
                static_cast<float>(layer * 1000 + position * 10 + d);
        }
    }
    return keys;
}

void capture_columns(KvRecallSet& set, int layer, int head_dim, int kv_heads, int first, int count) {
    std::vector<std::vector<float>> columns;
    for (int c = 0; c < count; ++c) { columns.push_back(column(head_dim, kv_heads, layer, first + c)); }
    for (const auto& one : columns) { set.capture(layer, one, 1); }
}

int layers_kept_apart_case() {
    int failures = 0;
    constexpr int kHeadDim = 2;
    constexpr int kKvHeads = 1;
    constexpr int kBlock    = 2;
    KvRecallSet set(2, kHeadDim, kKvHeads, kBlock);

    capture_columns(set, 0, kHeadDim, kKvHeads, 0, 2);
    capture_columns(set, 1, kHeadDim, kKvHeads, 0, 2);
    failures += expect(set.complete_blocks() == 1, "two columns per layer is one block");

    // Layer 0 columns are positions 0 and 1 of layer 0: means 5 and 6 for d=0,1.
    const auto& layer0 = set.block_means(0);
    const auto& layer1 = set.block_means(1);
    failures += expect(layer0.size() == 2, "one block of two dimensions");
    failures += expect(layer0[0] == 5.0F && layer0[1] == 6.0F, "layer 0 mean follows its own keys");
    failures += expect(layer1[0] == 1005.0F && layer1[1] == 1006.0F,
                       "layer 1 mean follows its own keys");
    return failures;
}

int ragged_layers_case() {
    int failures = 0;
    constexpr int kHeadDim = 1;
    constexpr int kKvHeads = 1;
    constexpr int kBlock    = 2;
    KvRecallSet set(3, kHeadDim, kKvHeads, kBlock);

    // Layer 0 finishes two blocks, layers 1 and 2 only one: the set may only score one.
    capture_columns(set, 0, kHeadDim, kKvHeads, 0, 4);
    capture_columns(set, 1, kHeadDim, kKvHeads, 0, 2);
    capture_columns(set, 2, kHeadDim, kKvHeads, 0, 2);
    failures += expect(set.complete_blocks() == 1, "the slowest layer caps the scored blocks");

    capture_columns(set, 1, kHeadDim, kKvHeads, 2, 2);
    capture_columns(set, 2, kHeadDim, kKvHeads, 2, 2);
    failures += expect(set.complete_blocks() == 2, "catching up raises the cap");

    set.reset();
    failures += expect(set.complete_blocks() == 0, "reset forgets every block");
    failures += expect(set.block_means(0).empty(), "reset empties the means");
    return failures;
}

int mean_scores_case() {
    int failures = 0;
    const std::vector<std::vector<float>> per_layer{{1.0F, 0.0F, 0.5F}, {0.0F, 1.0F, 1.5F}};
    const std::vector<float> mean = KvRecallSet::mean_scores(per_layer);
    failures += expect(mean.size() == 3, "one score per block");
    failures += expect(mean[0] == 0.5F && mean[1] == 0.5F && mean[2] == 1.0F,
                       "the mean is over layers, per block");

    failures += expect_throws([] {
        (void)KvRecallSet::mean_scores({});
    }, "an empty set of layers is rejected");
    failures += expect_throws([] {
        (void)KvRecallSet::mean_scores({{1.0F, 2.0F}, {1.0F}});
    }, "layers that disagree on the block count are rejected");
    return failures;
}

int invalid_case() {
    int failures = 0;
    failures += expect_throws([] { KvRecallSet set(0, 2, 1, 2); }, "zero layers is rejected");
    failures += expect_throws([] { KvRecallSet set(2, 0, 1, 2); }, "zero head_dim is rejected");
    failures += expect_throws([] { KvRecallSet set(2, 2, 0, 2); }, "zero kv_heads is rejected");
    failures += expect_throws([] { KvRecallSet set(2, 2, 1, 0); }, "zero block_tokens is rejected");

    KvRecallSet set(1, 1, 1, 2);
    failures += expect_throws([&] { set.capture(1, std::vector<float>(1, 1.0F), 1); },
                              "a layer outside the set is rejected");
    failures += expect_throws([&] { set.capture(-1, std::vector<float>(1, 1.0F), 1); },
                              "a negative layer is rejected");
    failures += expect_throws([&] { set.capture(0, std::vector<float>(2, 1.0F), 1); },
                              "the wrong column count is rejected");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += layers_kept_apart_case();
    failures += ragged_layers_case();
    failures += mean_scores_case();
    failures += invalid_case();

    if (failures == 0) { std::cout << "kv_recall_set: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}