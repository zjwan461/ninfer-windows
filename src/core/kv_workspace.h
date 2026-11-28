#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer {

// A logical block of KV history: a fixed span of tokens that is resident either in the device
// window or in the host tier. Block ids follow chronology: a larger id covers a later span.
enum class KvBlockTier : std::uint8_t { Device, Host };

struct KvBlock {
    std::int32_t id             = 0;
    std::int32_t orig_pos_start = 0;
    std::int32_t tokens         = 0;
    KvBlockTier tier            = KvBlockTier::Host;
    // Sink and recent blocks are always visible; a plan that drops one is a policy bug.
    bool pinned                 = false;

    friend bool operator==(const KvBlock&, const KvBlock&) = default;
};

struct WorkspaceBudget {
    std::int32_t device_tokens  = 0;
    std::int32_t reserve_tokens = 0;

    friend bool operator==(const WorkspaceBudget&, const WorkspaceBudget&) = default;
};

// Where a selected block lands in the compacted device window.
struct KvRemap {
    std::int32_t id            = 0;
    std::int32_t compact_start = 0;

    friend bool operator==(const KvRemap&, const KvRemap&) = default;
};

// The movement and the final device layout for one recall point.
//
// The device window is a *compact* prefix: selected blocks occupy [0, visible_tokens) in ascending
// id order, so the attention Op keeps reading rows [0, p] and needs no new kernel. Movement is
// ordered: every stage_out must complete before any stage_in starts, because the destination pool
// is bounded.
struct WorkspacePlan {
    std::vector<KvBlock> stage_out;
    std::vector<KvBlock> stage_in;
    std::vector<KvRemap> remaps;
    std::int32_t visible_tokens = 0;
    std::int32_t device_tokens  = 0;
};

// Compute the movement and the compacted layout for one recall point.
//
// `blocks` is the whole history in ascending id order with the tier each block occupies now.
// `selected_ids` is the set to make visible: pinned blocks plus the scorer's choices, in any order
// and possibly with duplicates. Selecting nothing is legal; selecting a pinned block is required.
//
// Throws std::invalid_argument for a malformed request (unknown id, negative or zero length,
// non-ascending blocks, a dropped pinned block) and std::runtime_error when the selected set does
// not fit the device window. The planner never invents a selection: shrinking the set is policy.
[[nodiscard]] WorkspacePlan plan_workspace(const WorkspaceBudget& budget,
                                           std::span<const KvBlock> blocks,
                                           std::span<const std::int32_t> selected_ids);

// Which blocks to make visible when only a page budget is affordable.
struct KvSelectSpec {
    // Pages that form one block; the last block may be shorter.
    std::int32_t block_pages = 0;
    // Pages at the start of the request that are always visible.
    std::int32_t sink_pages = 0;
    // Pages at the end of the request that are always visible.
    std::int32_t recent_pages = 0;
    // Pages the executed view may occupy, sink and recent included.
    std::int32_t budget_pages = 0;
};

/**
 * Picks the recall blocks to execute for one request.
 *
 * `scores` holds one relevance score per block, in ascending block id; `page_count` is the
 * request's page count, so the block count is ceil(page_count / block_pages). The result is the
 * ascending set of block ids to make visible: every block that touches the sink prefix or the
 * recent tail, plus the highest scoring remaining blocks that still fit `budget_pages`. Ties are
 * broken toward the lower block id, and a block is only added when its whole span fits.
 *
 * The page cost of a block is its own page span, so a short trailing block costs less. When the
 * mandatory blocks alone exceed the budget, or when no block can be afforded, the function throws
 * std::runtime_error: shrinking the request is the caller's decision, not the selector's.
 */
[[nodiscard]] std::vector<std::int32_t> select_kv_blocks(std::span<const float> scores,
                                                         std::int32_t page_count,
                                                         const KvSelectSpec& spec);

} // namespace ninfer