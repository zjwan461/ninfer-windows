#pragma once
#include "core/arena.h"
#include "core/kv_recall_sink.h"
#include "core/tensor.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// Keeps a per-layer summary of the history keys on the device: the sums and the exact column counts
// of every recall block, which the block-mean op turns into one mean per block. Only whole blocks
// are published, and the mean of a block is what a boundary decision compares a query with.
//
// Device state, so it lives with the program rather than in core: the accumulator is a request
// resource, sized once and reset per workspace. The layout follows the op — sums are
// [head_dim, kv_heads, block_capacity] and the block stride is the capacity, not the number of
// blocks written so far, so finishing a block never moves the blocks already published.
//
// One cursor is fed by the prefill path, layer by layer, turn by turn. A turn offers the same number
// of columns to every layer; a layer that is offered a different count means the caller skipped a
// layer, and the cursor refuses rather than rank a partial index.
class KvRecallCursor final : public core::KvRecallSink {
public:
    KvRecallCursor(std::int32_t layers, std::int32_t head_dim, std::int32_t kv_heads,
                   std::int32_t block_tokens, std::int32_t block_capacity);
    ~KvRecallCursor() override;

    void capture_layer(std::int32_t layer, const Tensor& pre_rope_keys,
                       cudaStream_t stream) override;

    // Pools the offered queries per KV head and keeps them for the boundary score. Queries arrive in
    // the same space as the keys (before the rotation); only the last `kQueryTokens` columns of the
    // chunk are pooled, because a boundary only asks about the end of the prompt.
    void capture_query(std::int32_t layer, const Tensor& pre_rope_queries,
                       cudaStream_t stream) override;

    // Pooled query of one layer, layout [head_dim, kv_heads]: one vector per KV head, already
    // averaged over its query-head group and over the pooled columns. Valid after capture_query().
    [[nodiscard]] const float* query(std::int32_t layer) const;

    // Finishes the turn for a sink that was handed the canonical prefill: publishes the means and
    // reports what was captured, so a live run shows the seam ran.
    void end_of_prefill(cudaStream_t stream) override;

    // Turns the device sums into the published block means and copies them to the host. Called at the
    // boundary, once, after the prefill that fed the cursor.
    void finish(cudaStream_t stream);

    // Mean of every block of one layer, layout [head_dim, kv_heads, block_capacity]; only the first
    // complete_blocks() entries of the last dimension are meaningful. Valid after finish().
    [[nodiscard]] const std::vector<float>& block_means(std::int32_t layer) const;

    [[nodiscard]] std::int32_t layers() const noexcept { return layers_count_; }
    [[nodiscard]] std::int32_t block_tokens() const noexcept { return block_tokens_; }
    [[nodiscard]] std::int32_t block_capacity() const noexcept { return block_capacity_; }
    [[nodiscard]] std::int32_t captured_columns() const noexcept { return captured_columns_; }
    // Whole blocks only: a trailing block that is still growing has no mean yet.
    [[nodiscard]] std::int32_t complete_blocks() const noexcept;

    void reset();

    // How many trailing columns of a chunk a boundary pools into one query. The reference
    // implementation uses the last user message, capped at 512 columns.
    static constexpr std::int32_t kQueryTokens = 512;

private:
    [[nodiscard]] std::size_t means_per_layer() const noexcept;

    std::int32_t layers_count_   = 0;
    std::int32_t head_dim_       = 0;
    std::int32_t kv_heads_       = 0;
    std::int32_t block_tokens_   = 0;
    std::int32_t block_capacity_ = 0;

    std::vector<DeviceBuffer> sums_;
    std::vector<DeviceBuffer> counts_;
    std::vector<DeviceBuffer> means_;
    std::vector<std::vector<float>> block_means_;
    // One pooled query per layer, [head_dim, kv_heads]; the shape is fixed, so a plain vector holds
    // it across layers.
    std::vector<float> queries_;
    std::vector<bool> query_valid_;

    std::int32_t captured_columns_ = 0;
    std::int32_t turn_columns_     = 0;
    std::int32_t layers_seen_      = 0;
    bool finished_                 = false;
};

} // namespace ninfer::models::qwen3_5::detail