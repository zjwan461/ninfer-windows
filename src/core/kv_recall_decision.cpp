#include "core/kv_recall_decision.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::core {
namespace {

[[nodiscard]] std::vector<std::int32_t> pinned_blocks(std::span<const KvBlock> blocks) {
    std::vector<std::int32_t> pinned;
    for (const KvBlock& block : blocks) {
        if (block.pinned) { pinned.push_back(block.id); }
    }
    return pinned;
}

} // namespace

KvRecallDecision plan_recall(const KvRecallRequest& request, std::span<const float> scores,
                             std::span<const std::int32_t> pages,
                             std::span<const KvBlock> blocks) {
    if (request.page_tokens <= 0) {
        throw std::invalid_argument("kv_recall_decision: page_tokens must be positive");
    }
    if (request.committed_frontier < 0) {
        throw std::invalid_argument("kv_recall_decision: committed_frontier must not be negative");
    }
    if (pages.empty() && request.committed_frontier != 0) {
        throw std::invalid_argument("kv_recall_decision: a non-empty history needs pages");
    }
    const std::int64_t covered_pages =
        (static_cast<std::int64_t>(request.committed_frontier) + request.page_tokens - 1) /
        request.page_tokens;
    if (covered_pages != static_cast<std::int64_t>(pages.size())) {
        throw std::invalid_argument("kv_recall_decision: pages " + std::to_string(pages.size()) +
                                    " do not cover " + std::to_string(request.committed_frontier) +
                                    " committed tokens");
    }
    if (request.select.block_pages <= 0) {
        throw std::invalid_argument("kv_recall_decision: block_pages must be positive");
    }
    const std::int64_t block_count =
        (static_cast<std::int64_t>(pages.size()) + request.select.block_pages - 1) /
        request.select.block_pages;
    if (static_cast<std::int64_t>(scores.size()) != block_count) {
        throw std::invalid_argument("kv_recall_decision: " + std::to_string(scores.size()) +
                                    " scores for " + std::to_string(block_count) + " blocks");
    }

    KvRecallDecision decision{};
    decision.selected_blocks = select_kv_blocks(scores, static_cast<std::int32_t>(pages.size()),
                                                request.select);

    // A pinned block is a lifetime, not a preference: the selector cannot see it, so it is added
    // here and the workspace planner is what refuses the request if the budget is too small.
    for (const std::int32_t id : pinned_blocks(blocks)) {
        if (id < 0 || id >= block_count) {
            throw std::invalid_argument("kv_recall_decision: pinned block " + std::to_string(id) +
                                        " is outside the page order");
        }
        decision.selected_blocks.push_back(id);
    }
    std::sort(decision.selected_blocks.begin(), decision.selected_blocks.end());
    decision.selected_blocks.erase(
        std::unique(decision.selected_blocks.begin(), decision.selected_blocks.end()),
        decision.selected_blocks.end());

    const KvViewSpec view_spec{request.select.block_pages, request.select.sink_pages,
                               request.select.recent_pages};
    decision.view = build_kv_view(std::vector<std::int32_t>(pages.begin(), pages.end()), view_spec,
                                  decision.selected_blocks);

    // A page is dropped whole, so the executed frontier is the committed one minus those pages.
    // Using visible_pages * page_tokens instead would overstate a frontier that ends mid-page.
    const std::int64_t dropped_pages =
        static_cast<std::int64_t>(pages.size()) - decision.view.visible_pages;
    const std::int64_t visible_tokens =
        static_cast<std::int64_t>(request.committed_frontier) - dropped_pages * request.page_tokens;
    if (visible_tokens < 0) {
        throw std::invalid_argument("kv_recall_decision: the view drops more tokens than exist");
    }
    decision.boundary = plan_view_boundary(
        request.committed_frontier, static_cast<std::int32_t>(visible_tokens),
        request.current_rope_delta);

    const std::int32_t window_pages = request.select.budget_pages + request.reserve_pages;
    const WorkspaceBudget budget{window_pages * request.page_tokens,
                                 request.reserve_pages * request.page_tokens};
    decision.residency = plan_workspace(budget, blocks, decision.selected_blocks);
    return decision;
}

} // namespace ninfer::core