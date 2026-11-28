#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer {

// Incremental per-block summary of the keys a prefill pass writes, used to decide which blocks of a
// long request stay in the executed KV view. The summary is deliberately built from keys *before*
// RoPE: a mean over post-rotation keys would depend on where the block happened to sit, which is the
// one property recall blocks must not have.
//
// Keys arrive chunk by chunk, and a block boundary can fall inside a chunk, so the index keeps a
// running sum and an exact count per block instead of finishing a block in one call.
class KvRecallIndex {
public:
    KvRecallIndex(std::int32_t head_dim, std::int32_t kv_heads, std::int32_t block_tokens);

    // Drops every summary and expects the first captured column at position 0.
    void reset() noexcept;

    [[nodiscard]] std::int32_t head_dim() const noexcept { return head_dim_; }
    [[nodiscard]] std::int32_t kv_heads() const noexcept { return kv_heads_; }
    [[nodiscard]] std::int32_t block_tokens() const noexcept { return block_tokens_; }

    // Next position capture() will accept.
    [[nodiscard]] std::uint32_t next_position() const noexcept { return next_position_; }

    // Number of blocks whose columns have all been captured.
    [[nodiscard]] std::int32_t complete_blocks() const noexcept { return complete_blocks_; }

    // Columns captured so far; the trailing partial block is not part of any summary yet.
    [[nodiscard]] std::uint32_t captured_columns() const noexcept { return next_position_; }

    // Appends one chunk of pre-rotation keys laid out as [head_dim, kv_heads, columns]. The chunk
    // must start exactly where the previous one ended; a gap would silently weight a block by a
    // subset of its columns.
    void capture(std::span<const float> keys, std::int32_t columns);

    // Means of the complete blocks, laid out as [head_dim, kv_heads, complete_blocks]. Materialized
    // on demand: the layout depends on the block count, so it cannot be filled in one block at a
    // time without rewriting what is already there.
    [[nodiscard]] const std::vector<float>& block_means() const;

private:
    void grow_to_blocks(std::int32_t blocks);
    void finish_block(std::int32_t block);

    std::int32_t head_dim_   = 0;
    std::int32_t kv_heads_   = 0;
    std::int32_t block_tokens_ = 0;
    std::uint32_t next_position_ = 0;
    std::int32_t complete_blocks_ = 0;
    std::vector<float> sums_;  // [blocks][head_dim, kv_heads], block-major so growth only appends
    std::vector<std::uint32_t> counts_;
    mutable std::vector<float> means_; // [head_dim, kv_heads, complete_blocks]
    mutable bool means_valid_ = false;
};

} // namespace ninfer