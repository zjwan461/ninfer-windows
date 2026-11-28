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

} // namespace ninfer