// Implements: src/core/kv_recall_set.h
#include "core/kv_recall_set.h"

#include "core/kv_recall_index.h"

#include <stdexcept>
#include <string>

namespace ninfer::core {

KvRecallSet::KvRecallSet(std::int32_t layers, std::int32_t head_dim, std::int32_t kv_heads,
                         std::int32_t block_tokens)
    : layers_count_(layers), block_tokens_(block_tokens) {
    if (layers <= 0 || head_dim <= 0 || kv_heads <= 0 || block_tokens <= 0) {
        throw std::invalid_argument("kv_recall_set: geometry must be positive");
    }
    layers_.reserve(static_cast<std::size_t>(layers));
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        layers_.push_back(std::make_unique<KvRecallIndex>(head_dim, kv_heads, block_tokens));
    }
}

KvRecallSet::~KvRecallSet() = default;
KvRecallSet::KvRecallSet(KvRecallSet&&) noexcept            = default;
KvRecallSet& KvRecallSet::operator=(KvRecallSet&&) noexcept = default;

void KvRecallSet::capture(std::int32_t layer, std::span<const float> keys, std::int32_t columns) {
    if (layer < 0 || layer >= layers_count_) {
        throw std::out_of_range("kv_recall_set: layer " + std::to_string(layer) +
                                " is outside [0," + std::to_string(layers_count_) + ")");
    }
    layers_[static_cast<std::size_t>(layer)]->capture(keys, columns);
}

std::int32_t KvRecallSet::complete_blocks() const {
    std::int32_t blocks = -1;
    for (const auto& layer : layers_) {
        const std::int32_t layer_blocks = layer->complete_blocks();
        if (blocks < 0 || layer_blocks < blocks) { blocks = layer_blocks; }
    }
    return blocks < 0 ? 0 : blocks;
}

std::int32_t KvRecallSet::layers() const { return layers_count_; }

std::int32_t KvRecallSet::block_tokens() const { return block_tokens_; }

const std::vector<float>& KvRecallSet::block_means(std::int32_t layer) const {
    if (layer < 0 || layer >= layers_count_) {
        throw std::out_of_range("kv_recall_set: layer " + std::to_string(layer) +
                                " is outside [0," + std::to_string(layers_count_) + ")");
    }
    return layers_[static_cast<std::size_t>(layer)]->block_means();
}

void KvRecallSet::reset() {
    for (auto& layer : layers_) { layer->reset(); }
}

std::vector<float> KvRecallSet::mean_scores(const std::vector<std::vector<float>>& per_layer) {
    if (per_layer.empty()) {
        throw std::invalid_argument("kv_recall_set: at least one layer score vector is required");
    }
    const std::size_t blocks = per_layer.front().size();
    for (const auto& scores : per_layer) {
        if (scores.size() != blocks) {
            throw std::invalid_argument("kv_recall_set: every layer must score every block");
        }
    }
    std::vector<float> mean(blocks, 0.0F);
    for (const auto& scores : per_layer) {
        for (std::size_t block = 0; block < blocks; ++block) { mean[block] += scores[block]; }
    }
    const float divisor = static_cast<float>(per_layer.size());
    for (float& value : mean) { value /= divisor; }
    return mean;
}

} // namespace ninfer::core