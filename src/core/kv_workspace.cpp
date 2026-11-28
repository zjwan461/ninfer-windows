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

std::vector<std::int32_t> select_kv_blocks(std::span<const float> scores, std::int32_t page_count,
                                           const KvSelectSpec& spec) {
    if (page_count <= 0) { throw std::invalid_argument("selection: page_count must be positive"); }
    if (spec.block_pages <= 0) {
        throw std::invalid_argument("selection: block_pages must be positive");
    }
    if (spec.sink_pages < 0 || spec.recent_pages < 0) {
        throw std::invalid_argument("selection: sink_pages and recent_pages must be nonnegative");
    }
    if (spec.sink_pages + spec.recent_pages > page_count) {
        throw std::invalid_argument("selection: sink_pages + recent_pages exceeds the request");
    }

    const std::int32_t block_count = (page_count + spec.block_pages - 1) / spec.block_pages;
    if (static_cast<std::int32_t>(scores.size()) != block_count) {
        throw std::invalid_argument("selection: scores must hold one value per block");
    }
    if (spec.budget_pages < 0 || spec.budget_pages > page_count) {
        throw std::invalid_argument("selection: budget_pages must be in [0,page_count]");
    }

    auto first_page = [&](std::int32_t block) { return block * spec.block_pages; };
    auto block_cost = [&](std::int32_t block) {
        const std::int32_t first = first_page(block);
        return std::min(spec.block_pages, page_count - first);
    };

    std::vector<bool> mandatory(static_cast<std::size_t>(block_count), false);
    std::int32_t mandatory_pages = 0;
    for (std::int32_t page = 0; page < page_count; ++page) {
        const bool sink   = page < spec.sink_pages;
        const bool recent = page >= page_count - spec.recent_pages;
        if (!sink && !recent) { continue; }
        const std::int32_t block = page / spec.block_pages;
        if (mandatory[static_cast<std::size_t>(block)]) { continue; }
        mandatory[static_cast<std::size_t>(block)] = true;
        mandatory_pages += block_cost(block);
    }

    if (mandatory_pages > spec.budget_pages) {
        throw std::runtime_error("selection: the mandatory blocks alone exceed the page budget");
    }

    std::vector<std::int32_t> candidates;
    candidates.reserve(static_cast<std::size_t>(block_count));
    for (std::int32_t block = 0; block < block_count; ++block) {
        const float score = scores[static_cast<std::size_t>(block)];
        if (!(score == score)) { throw std::invalid_argument("selection: scores must be finite"); }
        if (!mandatory[static_cast<std::size_t>(block)]) { candidates.push_back(block); }
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [&](std::int32_t a, std::int32_t b) {
                         const float sa = scores[static_cast<std::size_t>(a)];
                         const float sb = scores[static_cast<std::size_t>(b)];
                         if (sa != sb) { return sa > sb; }
                         return a < b;
                     });

    std::vector<std::int32_t> chosen;
    for (std::int32_t block = 0; block < block_count; ++block) {
        if (mandatory[static_cast<std::size_t>(block)]) { chosen.push_back(block); }
    }
    std::int32_t used = mandatory_pages;
    for (const std::int32_t block : candidates) {
        const std::int32_t cost = block_cost(block);
        if (used + cost > spec.budget_pages) { continue; }
        chosen.push_back(block);
        used += cost;
    }

    std::sort(chosen.begin(), chosen.end());
    if (chosen.empty()) {
        throw std::runtime_error("selection: no block fits the page budget");
    }
    return chosen;
}

} // namespace ninfer