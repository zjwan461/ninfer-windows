// One recall point decided end to end, over a history the size of a real workspace.
//
// The pieces were tested separately; this test pins the arithmetic that joins them, because the
// failure mode of a wrong join is silent: a frontier that does not shrink, a RoPE delta that does
// not grow by the same amount, or a view that is longer than the history it came from.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/kv_recall_decision.h"
#include "core/kv_view.h"
#include "core/kv_workspace.h"

namespace {

using ninfer::KvBlock;
using ninfer::KvBlockTier;
using ninfer::core::KvRecallDecision;
using ninfer::core::KvRecallRequest;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

template <class Fn>
void expect_throws(Fn&& fn, const std::string& what) {
    try {
        fn();
        std::fprintf(stderr, "FAIL: %s did not throw\n", what.c_str());
        ++failures;
    } catch (const std::exception&) {
    }
}

constexpr std::int32_t kPageTokens = 64;
constexpr std::int32_t kBlockPages = 2; // 128-token blocks, the recall block of the port plan

KvRecallRequest request_for(std::int32_t pages, std::int32_t budget_pages,
                            std::int32_t reserve_pages = 0) {
    KvRecallRequest request;
    request.select.block_pages  = kBlockPages;
    request.select.sink_pages   = 1;
    request.select.recent_pages = 1;
    request.select.budget_pages = budget_pages;
    request.page_tokens         = kPageTokens;
    request.reserve_pages       = reserve_pages;
    request.committed_frontier  = pages * kPageTokens;
    return request;
}

std::vector<std::int32_t> identity_pages(std::int32_t pages) {
    std::vector<std::int32_t> ids(static_cast<std::size_t>(pages));
    for (std::int32_t page = 0; page < pages; ++page) { ids[page] = 1000 + page; }
    return ids;
}

std::vector<KvBlock> resident_blocks(std::int32_t pages) {
    std::vector<KvBlock> blocks;
    for (std::int32_t page = 0; page + kBlockPages <= pages; page += kBlockPages) {
        KvBlock block;
        block.id             = page / kBlockPages;
        block.orig_pos_start = page * kPageTokens;
        block.tokens         = kBlockPages * kPageTokens;
        block.tier           = KvBlockTier::Device;
        blocks.push_back(block);
    }
    return blocks;
}

std::vector<float> flat_scores(std::int32_t blocks) {
    std::vector<float> scores(static_cast<std::size_t>(blocks), 0.05F);
    return scores;
}

void whole_history_case() {
    constexpr std::int32_t kPages = 8;
    const auto request            = request_for(kPages, kPages);
    const auto pages              = identity_pages(kPages);
    const auto blocks             = resident_blocks(kPages);
    const auto scores             = flat_scores(static_cast<std::int32_t>(blocks.size()));

    const KvRecallDecision decision =
        ninfer::core::plan_recall(request, scores, pages, blocks);
    check(decision.view.pages == pages, "selecting everything keeps the page order");
    check(decision.boundary.dropped_tokens == 0, "nothing is dropped when everything fits");
    check(decision.boundary.execution_frontier == request.committed_frontier,
          "the executed frontier is the committed one");
    check(decision.boundary.rope_delta == 0, "no phase shift without a drop");
    check(decision.residency.stage_in.empty(), "a whole device history stages nothing in");
}

void dropped_prefix_case() {
    constexpr std::int32_t kPages  = 16;
    constexpr std::int32_t kBudget = 6;
    auto request                   = request_for(kPages, kBudget);
    request.current_rope_delta     = 7;
    const auto pages               = identity_pages(kPages);
    const auto blocks              = resident_blocks(kPages);
    auto scores                    = flat_scores(static_cast<std::int32_t>(blocks.size()));
    scores[3]                      = 0.95F; // the needle sits in block 3

    const KvRecallDecision decision =
        ninfer::core::plan_recall(request, scores, pages, blocks);
    check(static_cast<std::int32_t>(decision.view.visible_pages) <= kBudget,
          "the executed view stays inside the budget");
    check(std::find(decision.selected_blocks.begin(), decision.selected_blocks.end(), 3) !=
              decision.selected_blocks.end(),
          "the highest-scoring block is selected");
    check(decision.view.visible_pages < kPages, "a tight budget really drops pages");
    const std::int32_t dropped_pages = kPages - decision.view.visible_pages;
    check(decision.boundary.dropped_tokens == dropped_pages * kPageTokens,
          "dropped tokens equal the dropped pages");
    check(decision.boundary.execution_frontier + decision.boundary.dropped_tokens ==
              request.committed_frontier,
          "the two frontiers differ by exactly the dropped tokens");
    check(decision.boundary.rope_delta == request.current_rope_delta + dropped_pages * kPageTokens,
          "the RoPE delta grows by exactly the dropped tokens");
    check(decision.residency.visible_tokens == decision.boundary.execution_frontier,
          "the residency plan charges the same frontier the boundary reports");
}

void mid_page_frontier_case() {
    constexpr std::int32_t kPages = 8;
    auto request                  = request_for(kPages, kPages);
    request.committed_frontier -= 20; // the frontier ends inside the last page
    const auto pages  = identity_pages(kPages);
    const auto blocks = resident_blocks(kPages);
    const auto scores = flat_scores(static_cast<std::int32_t>(blocks.size()));

    const KvRecallDecision decision =
        ninfer::core::plan_recall(request, scores, pages, blocks);
    check(decision.boundary.dropped_tokens == 0, "a mid-page frontier drops nothing");
    check(decision.boundary.execution_frontier == request.committed_frontier,
          "a mid-page frontier is not rounded up to the page end");
}

void pinned_case() {
    constexpr std::int32_t kPages  = 16;
    constexpr std::int32_t kBudget = 6; // sink block + recent block + the pinned block + the needle
    const auto request             = request_for(kPages, kBudget);
    const auto pages               = identity_pages(kPages);
    auto blocks                    = resident_blocks(kPages);
    blocks[0].pinned               = true; // the sink block is a lifetime, not a preference
    auto scores                    = flat_scores(static_cast<std::int32_t>(blocks.size()));
    scores[5]                      = 0.99F;

    const KvRecallDecision decision =
        ninfer::core::plan_recall(request, scores, pages, blocks);
    check(std::find(decision.selected_blocks.begin(), decision.selected_blocks.end(), 0) !=
              decision.selected_blocks.end(),
          "a pinned block is always selected");
    check(std::find(decision.selected_blocks.begin(), decision.selected_blocks.end(), 5) !=
              decision.selected_blocks.end(),
          "the needle survives alongside the pinned block");
}

void rejection_cases() {
    constexpr std::int32_t kPages = 8;
    const auto pages              = identity_pages(kPages);
    const auto blocks             = resident_blocks(kPages);

    // Pages must cover the committed prefix exactly.
    expect_throws(
        [&] {
            auto request = request_for(kPages, kPages);
            request.committed_frontier += kPageTokens;
            (void)ninfer::core::plan_recall(request, flat_scores(4), pages, blocks);
        },
        "pages that do not cover the frontier");

    // One score per block, no more and no less.
    expect_throws(
        [&] {
            const auto request = request_for(kPages, kPages);
            (void)ninfer::core::plan_recall(request, flat_scores(3), pages, blocks);
        },
        "a score count that does not match the blocks");

    // A budget smaller than the compulsory blocks cannot produce a view.
    expect_throws(
        [&] {
            const auto request = request_for(kPages, 1);
            (void)ninfer::core::plan_recall(request, flat_scores(4), pages, blocks);
        },
        "a budget below the compulsory blocks");

    // A pinned block outside the page order is a caller bug, not a smaller selection.
    expect_throws(
        [&] {
            auto broken       = blocks;
            broken[0].pinned  = true;
            broken[0].id      = 99;
            const auto request = request_for(kPages, kPages);
            (void)ninfer::core::plan_recall(request, flat_scores(4), pages, broken);
        },
        "a pinned block outside the page order");
}

} // namespace

int main() {
    whole_history_case();
    dropped_prefix_case();
    mid_page_frontier_case();
    pinned_case();
    rejection_cases();
    if (failures != 0) {
        std::fprintf(stderr, "kv_recall_decision: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("kv_recall_decision: all checks passed\n");
    return 0;
}