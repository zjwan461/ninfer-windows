#include "core/kv_workspace.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

[[nodiscard]] bool selected_contains(const std::vector<std::int32_t>& sorted,
                                     std::int32_t id) {
    return std::binary_search(sorted.begin(), sorted.end(), id);
}

} // namespace

WorkspacePlan plan_workspace(const WorkspaceBudget& budget, std::span<const KvBlock> blocks,
                             std::span<const std::int32_t> selected_ids) {
    if (budget.device_tokens <= 0) {
        throw std::invalid_argument("working set: device window must be positive");
    }
    if (budget.reserve_tokens < 0 || budget.reserve_tokens >= budget.device_tokens) {
        throw std::invalid_argument("working set: reserve must be within the device window");
    }

    std::int32_t previous_id = -1;
    for (const KvBlock& block : blocks) {
        if (block.tokens <= 0) {
            throw std::invalid_argument("working set: block " + std::to_string(block.id) +
                                        " has no tokens");
        }
        if (block.orig_pos_start < 0) {
            throw std::invalid_argument("working set: block " + std::to_string(block.id) +
                                        " has a negative start");
        }
        if (block.id <= previous_id) {
            throw std::invalid_argument("working set: blocks must be ordered by ascending id");
        }
        previous_id = block.id;
    }

    std::vector<std::int32_t> selection(selected_ids.begin(), selected_ids.end());

    // Reject unknown ids before deduplicating, so a typo is a failure and not an empty plan.
    for (std::int32_t id : selection) {
        const auto found = std::lower_bound(blocks.begin(), blocks.end(), id,
                                            [](const KvBlock& block, std::int32_t value) {
                                                return block.id < value;
                                            });
        if (found == blocks.end() || found->id != id) {
            throw std::invalid_argument("working set: selected block " + std::to_string(id) +
                                        " is not in the history");
        }
    }

    std::sort(selection.begin(), selection.end());
    selection.erase(std::unique(selection.begin(), selection.end()), selection.end());

    for (const KvBlock& block : blocks) {
        if (block.pinned && !selected_contains(selection, block.id)) {
            throw std::invalid_argument("working set: pinned block " + std::to_string(block.id) +
                                        " was not selected");
        }
    }

    // The compacted window is the selection in chronology, and it has to leave room to generate.
    WorkspacePlan plan;
    plan.device_tokens = budget.device_tokens;
    for (const KvBlock& block : blocks) {
        if (!selected_contains(selection, block.id)) { continue; }
        KvRemap remap;
        remap.id            = block.id;
        remap.compact_start = plan.visible_tokens;
        plan.remaps.push_back(remap);
        plan.visible_tokens += block.tokens;
    }

    if (plan.visible_tokens + budget.reserve_tokens > budget.device_tokens) {
        throw std::runtime_error("working set: selection of " +
                                 std::to_string(plan.visible_tokens) + " tokens plus " +
                                 std::to_string(budget.reserve_tokens) +
                                 " reserve exceeds the " + std::to_string(budget.device_tokens) +
                                 " token device window");
    }

    for (const KvBlock& block : blocks) {
        const bool selected = selected_contains(selection, block.id);
        if (block.tier == KvBlockTier::Device && !selected) {
            plan.stage_out.push_back(block);
        } else if (block.tier == KvBlockTier::Host && selected) {
            plan.stage_in.push_back(block);
        }
    }

    return plan;
}

} // namespace ninfer