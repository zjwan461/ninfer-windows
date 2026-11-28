#pragma once

// One recall point, decided end to end: which blocks to execute, the page order they produce, the
// positions that order hands to decode, and the residency movement it costs.
//
// Every part of this exists already and is tested on its own:
//
//   select_kv_blocks   chooses blocks under a page budget,
//   build_kv_view      turns the choice into the executed page order,
//   plan_view_boundary turns the page order into the two position numbers,
//   plan_workspace     turns the choice into stage-out / stage-in and compact starts.
//
// What was missing is a single place that runs them in that order on the same inputs and refuses
// the combinations that would silently misplace a position. The boundary site in the prefill path
// calls this; nothing else should have to re-derive the arithmetic.

#include "core/kv_view.h"
#include "core/kv_workspace.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::core {

struct KvRecallRequest {
    // Block geometry and the executed-page budget, shared with the selector and the view.
    KvSelectSpec select;
    // Tokens in one page; the engine uses 64 (src/core/paged_kv_cache.h).
    std::int32_t page_tokens = 64;
    // Pages kept free inside the device window for the columns this turn will append.
    std::int32_t reserve_pages = 0;
    // Tokens the sequence has executed. May end mid-page.
    std::int32_t committed_frontier = 0;
    // The sequence's running RoPE delta before this view.
    std::int32_t current_rope_delta = 0;
};

struct KvRecallDecision {
    // Blocks the executed view contains, ascending.
    std::vector<std::int32_t> selected_blocks;
    // Executed page order and its length.
    KvView view;
    // Position bookkeeping the decode path needs.
    KvViewBoundary boundary;
    // Residency movement to reach that order.
    WorkspacePlan residency;
};

/**
 * Decides one recall point.
 *
 * `scores` holds one score per block of `pages` (block count is ceil(pages / block_pages)).
 * `pages` is the request's physical page order in time order, covering the whole committed
 * prefix, so its length must equal ceil(committed_frontier / page_tokens). `blocks` describes
 * the whole history in ascending id order with the tier each block occupies now; a block marked
 * pinned is always selected, whatever it scored.
 *
 * Throws std::invalid_argument when the request cannot describe a real view (mismatched page
 * and score counts, pages that do not cover the committed prefix, a non-positive page size,
 * a pinned block that the page order does not contain) and std::runtime_error when the budget
 * cannot hold the blocks that must be visible.
 */
[[nodiscard]] KvRecallDecision plan_recall(const KvRecallRequest& request,
                                           std::span<const float> scores,
                                           std::span<const std::int32_t> pages,
                                           std::span<const KvBlock> blocks);

} // namespace ninfer::core