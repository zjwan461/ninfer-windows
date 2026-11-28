#pragma once

// ninfer::core - the per-layer set of block summaries a request keeps for KV recall.
//
// One KvRecallIndex per standard-attention layer, because the mean key lives in that layer's key
// space: layer 3's keys cannot be scored against layer 7's query. Recurrent layers (GDN) are not
// represented at all - they keep O(1) state rather than token-wise KV, so there is nothing to
// summarize and nothing to recall.
//
// The set owns no memory policy and no device state. It answers "what does block b look like in
// layer l" and "which blocks look alike across the layers that agree", which is all a boundary
// decision needs once the per-layer scores exist.

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer {
class KvRecallIndex;
}

namespace ninfer::core {

class KvRecallSet {
public:
    /**
     * `layers` standard-attention layers, each with `head_dim` x `kv_heads` keys and
     * `block_tokens` columns per block. Throws std::invalid_argument on non-positive geometry.
     */
    KvRecallSet(std::int32_t layers, std::int32_t head_dim, std::int32_t kv_heads,
                std::int32_t block_tokens);
    ~KvRecallSet();

    KvRecallSet(KvRecallSet&&) noexcept;
    KvRecallSet& operator=(KvRecallSet&&) noexcept;
    KvRecallSet(const KvRecallSet&) = delete;
    KvRecallSet& operator=(const KvRecallSet&) = delete;

    /**
     * Appends `columns` pre-RoPE key columns of one layer. Throws std::out_of_range for a layer
     * outside [0, layers) and std::invalid_argument when the column count does not match the
     * layer's geometry.
     */
    void capture(std::int32_t layer, std::span<const float> keys, std::int32_t columns);

    /**
     * Columns that may be scored: the whole blocks every layer has finished. A layer that is one
     * block ahead of the others cannot vote with the block it alone has, because the ranking is a
     * single decision shared by all layers.
     */
    [[nodiscard]] std::int32_t complete_blocks() const;

    [[nodiscard]] std::int32_t layers() const;
    [[nodiscard]] std::int32_t block_tokens() const;

    /** Means of one layer's complete blocks, laid out [head_dim, kv_heads, complete_blocks]. */
    [[nodiscard]] const std::vector<float>& block_means(std::int32_t layer) const;

    void reset();

    /**
     * Folds per-layer block scores into the single ranking the boundary acts on: the mean over
     * layers. Every vector must hold one score per complete block; a layer that disagrees is an
     * error rather than a silently skewed average. Throws std::invalid_argument otherwise.
     */
    [[nodiscard]] static std::vector<float> mean_scores(
        const std::vector<std::vector<float>>& per_layer);

private:
    std::vector<std::unique_ptr<KvRecallIndex>> layers_;
    std::int32_t layers_count_  = 0;
    std::int32_t block_tokens_  = 0;
};

} // namespace ninfer::core