#include "core/kv_recall_index.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

[[nodiscard]] std::string detail(const char* what, long long value) {
    return std::string(what) + std::to_string(value);
}

} // namespace

KvRecallIndex::KvRecallIndex(std::int32_t head_dim, std::int32_t kv_heads, std::int32_t block_tokens) {
    if (head_dim <= 0 || kv_heads <= 0 || block_tokens <= 0) {
        throw std::invalid_argument("kv_recall_index: geometry must be positive");
    }
    head_dim_     = head_dim;
    kv_heads_     = kv_heads;
    block_tokens_ = block_tokens;
}

void KvRecallIndex::reset() noexcept {
    next_position_   = 0;
    complete_blocks_ = 0;
    sums_.clear();
    counts_.clear();
    means_.clear();
}

void KvRecallIndex::grow_to_blocks(std::int32_t blocks) {
    if (blocks <= 0) { return; }
    const std::size_t per_block = static_cast<std::size_t>(head_dim_) * static_cast<std::size_t>(kv_heads_);
    if (static_cast<std::size_t>(blocks) * per_block <= sums_.size()) { return; }
    sums_.resize(static_cast<std::size_t>(blocks) * per_block, 0.0F);
    counts_.resize(static_cast<std::size_t>(blocks), 0U);
}

void KvRecallIndex::finish_block(std::int32_t block) {
    const std::size_t per_block = static_cast<std::size_t>(head_dim_) * static_cast<std::size_t>(kv_heads_);
    if (counts_[static_cast<std::size_t>(block)] != static_cast<std::uint32_t>(block_tokens_)) {
        throw std::logic_error("kv_recall_index: block finished with the wrong column count");
    }
    means_.resize(static_cast<std::size_t>(complete_blocks_ + 1) * per_block);
    const float divisor = static_cast<float>(block_tokens_);
    for (std::int32_t head = 0; head < kv_heads_; ++head) {
        for (std::int32_t dim = 0; dim < head_dim_; ++dim) {
            const std::size_t source = (static_cast<std::size_t>(block) * per_block) +
                                       static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim_) +
                                       static_cast<std::size_t>(dim);
            const std::size_t target =
                (static_cast<std::size_t>(complete_blocks_) * per_block) +
                static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim_) +
                static_cast<std::size_t>(dim);
            means_[target] = sums_[source] / divisor;
        }
    }
    ++complete_blocks_;
}

void KvRecallIndex::capture(std::span<const float> keys, std::int32_t columns) {
    if (columns <= 0) { throw std::invalid_argument("kv_recall_index: a chunk needs columns"); }
    const std::size_t expected =
        static_cast<std::size_t>(head_dim_) * static_cast<std::size_t>(kv_heads_) * static_cast<std::size_t>(columns);
    if (keys.size() != expected) {
        throw std::invalid_argument(detail("kv_recall_index: chunk holds ", static_cast<long long>(keys.size())) +
                                    ", expected " + std::to_string(expected));
    }

    const std::int32_t per_block = block_tokens_;
    const std::int32_t first_block = static_cast<std::int32_t>(next_position_ / static_cast<std::uint32_t>(per_block));
    const std::int32_t last_block =
        static_cast<std::int32_t>((next_position_ + static_cast<std::uint32_t>(columns) - 1U) /
                                  static_cast<std::uint32_t>(per_block));
    grow_to_blocks(last_block + 1);

    const std::size_t per_block_elements =
        static_cast<std::size_t>(head_dim_) * static_cast<std::size_t>(kv_heads_);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::uint32_t position = next_position_ + static_cast<std::uint32_t>(column);
        const std::int32_t block = static_cast<std::int32_t>(position / static_cast<std::uint32_t>(per_block));
        float* sum = sums_.data() + static_cast<std::size_t>(block) * per_block_elements;
        for (std::int32_t head = 0; head < kv_heads_; ++head) {
            for (std::int32_t dim = 0; dim < head_dim_; ++dim) {
                const std::size_t source =
                    (static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim_) +
                     static_cast<std::size_t>(dim)) * static_cast<std::size_t>(columns) +
                    static_cast<std::size_t>(column);
                sum[static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim_) +
                    static_cast<std::size_t>(dim)] += keys[source];
            }
        }
        ++counts_[static_cast<std::size_t>(block)];
    }
    next_position_ += static_cast<std::uint32_t>(columns);

    for (std::int32_t block = first_block; block <= last_block; ++block) {
        if (block >= complete_blocks_ &&
            counts_[static_cast<std::size_t>(block)] == static_cast<std::uint32_t>(block_tokens_)) {
            finish_block(block);
        }
    }
}

} // namespace ninfer