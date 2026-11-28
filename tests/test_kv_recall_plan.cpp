// End-to-end accounting of a recall step: score, select, assemble, and plan the residency move.
//
// The three halves were tested separately (kv_block_pool on the device, select_kv_blocks and
// plan_workspace on the host). This test composes them over a history the size of a real
// workspace and asserts the invariants that must hold before any of it touches the execution
// path: the budget is never exceeded, the compulsory blocks are always present, the view has no
// holes, and the residency delta describes exactly the traffic the host tier will see.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/kv_view.h"
#include "core/kv_workspace.h"

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// 64 tokens per physical page is what the paged cache uses; a recall block spans whole pages.
constexpr std::int32_t kPageTokens = 64;
constexpr std::int32_t kBlockTokens = 128;
constexpr std::int32_t kBlockPages = kBlockTokens / kPageTokens;

std::int32_t blocks_for(std::int64_t tokens) {
    return static_cast<std::int32_t>(tokens / kBlockTokens);
}

// Scores that put a single block far above the rest, the way a real query does with a needle.
std::vector<float> needle_scores(std::int32_t blocks, std::int32_t needle) {
    std::vector<float> scores(static_cast<std::size_t>(blocks), 0.05F);
    for (std::int32_t block = 0; block < blocks; ++block) {
        scores[static_cast<std::size_t>(block)] =
            0.05F + 0.01F * static_cast<float>(block % 7);
    }
    scores[static_cast<std::size_t>(needle)] = 0.95F;
    return scores;
}

std::vector<ninfer::KvBlock> history(std::int32_t blocks, std::int32_t resident_blocks) {
    std::vector<ninfer::KvBlock> out;
    out.reserve(static_cast<std::size_t>(blocks));
    for (std::int32_t block = 0; block < blocks; ++block) {
        out.push_back(ninfer::KvBlock{
            block, block * kBlockTokens, kBlockTokens,
            block < resident_blocks ? ninfer::KvBlockTier::Device
                                    : ninfer::KvBlockTier::Host,
            false});
    }
    return out;
}

void scenario(const char* name, std::int64_t tokens, std::int32_t budget_tokens,
              std::int32_t sink_tokens, std::int32_t recent_tokens) {
    const std::int32_t blocks = blocks_for(tokens);
    const std::int32_t needle = blocks / 2;
    const auto scores = needle_scores(blocks, needle);

    ninfer::KvSelectSpec select_spec{};
    select_spec.block_pages = kBlockPages;
    select_spec.sink_pages = sink_tokens / kPageTokens;
    select_spec.recent_pages = recent_tokens / kPageTokens;
    select_spec.budget_pages = budget_tokens / kPageTokens;

    const auto selected = ninfer::select_kv_blocks(scores, blocks * kBlockPages, select_spec);
    const auto selected_tokens =
        static_cast<std::int64_t>(selected.size()) * kBlockPages * kPageTokens;
    check(selected_tokens <= budget_tokens,
          std::string(name) + ": selection stays inside the budget");
    check(std::find(selected.begin(), selected.end(), needle) != selected.end(),
          std::string(name) + ": the needle's block is selected");
    for (std::int32_t block = 0; block < select_spec.sink_pages / kBlockPages; ++block) {
        check(std::find(selected.begin(), selected.end(), block) != selected.end(),
              std::string(name) + ": a sink block is selected");
    }
    for (std::int32_t block = blocks - select_spec.recent_pages / kBlockPages; block < blocks;
         ++block) {
        check(std::find(selected.begin(), selected.end(), block) != selected.end(),
              std::string(name) + ": a recent block is selected");
    }

    // The view is built from the request's own page order; the selection only names blocks.
    std::vector<std::int32_t> request_pages;
    request_pages.reserve(static_cast<std::size_t>(blocks * kBlockPages));
    for (std::int32_t page = 0; page < blocks * kBlockPages; ++page) {
        request_pages.push_back(page);
    }
    std::vector<std::int32_t> expected_pages;
    expected_pages.reserve(static_cast<std::size_t>(selected.size() * kBlockPages));
    for (const std::int32_t block : selected) {
        for (std::int32_t page = 0; page < kBlockPages; ++page) {
            expected_pages.push_back(block * kBlockPages + page);
        }
    }
    ninfer::core::KvViewSpec view_spec{};
    view_spec.block_pages = kBlockPages;
    view_spec.sink_pages = select_spec.sink_pages;
    view_spec.recent_pages = select_spec.recent_pages;
    const auto view = ninfer::core::build_kv_view(request_pages, view_spec, selected);
    check(view.pages == expected_pages,
          std::string(name) + ": the view is exactly the selected pages, in order");
    check(view.visible_pages == static_cast<std::int32_t>(expected_pages.size()),
          std::string(name) + ": the executed rows are the selected pages");
    check(static_cast<std::int64_t>(view.visible_pages) * kPageTokens <= budget_tokens,
          std::string(name) + ": the view fits the budget");
    for (std::size_t index = 1; index < view.pages.size(); ++index) {
        check(view.pages[index] > view.pages[index - 1],
              std::string(name) + ": the view is ordered and free of holes");
    }

    // First turn: nothing is resident, so every selected block comes up from the host tier.
    // The device window is the recall budget plus the generation reserve, the way the reference
    // server sizes it: the selector may fill B, and R stays free for the tokens a step appends.
    const auto cold = history(blocks, 0);
    ninfer::WorkspaceBudget budget{};
    budget.device_tokens = budget_tokens + recent_tokens;
    budget.reserve_tokens = recent_tokens;
    const auto cold_plan = ninfer::plan_workspace(budget, cold, selected);
    check(cold_plan.stage_out.empty(), std::string(name) + ": a cold turn evicts nothing");
    check(cold_plan.stage_in.size() == selected.size(),
          std::string(name) + ": a cold turn stages in every selected block");
    check(static_cast<std::int64_t>(cold_plan.visible_tokens) == selected_tokens,
          std::string(name) + ": the plan sees the same tokens as the view");
    check(cold_plan.device_tokens == budget.device_tokens,
          std::string(name) + ": the plan keeps the device window it was given");
    check(cold_plan.remaps.size() == selected.size(),
          std::string(name) + ": every selected block gets a remap");
    for (std::size_t index = 0; index < cold_plan.remaps.size(); ++index) {
        if (index == 0) {
            check(cold_plan.remaps[index].compact_start == 0,
                  std::string(name) + ": the compacted prefix starts at zero");
        } else {
            check(cold_plan.remaps[index].compact_start > cold_plan.remaps[index - 1].compact_start,
                  std::string(name) + ": remaps advance with the block order");
        }
    }

    // Second turn: the view from the cold turn is resident, the query picks a different needle.
    const auto warm = history(blocks, static_cast<std::int32_t>(selected.size()));
    const std::int32_t other = needle / 2;
    const auto warm_scores = needle_scores(blocks, other);
    const auto warm_selected =
        ninfer::select_kv_blocks(warm_scores, blocks * kBlockPages, select_spec);
    const auto warm_plan = ninfer::plan_workspace(budget, warm, warm_selected);
    std::int32_t staged_in = 0;
    for (const auto& block : warm_plan.stage_in) {
        check(std::find(warm_selected.begin(), warm_selected.end(), block.id) !=
                  warm_selected.end(),
              std::string(name) + ": every staged-in block is selected");
        check(block.tier == ninfer::KvBlockTier::Host,
              std::string(name) + ": only host blocks are staged in");
        ++staged_in;
    }
    for (const auto& block : warm_plan.stage_out) {
        check(block.tier == ninfer::KvBlockTier::Device,
              std::string(name) + ": only device blocks are staged out");
        check(std::find(warm_selected.begin(), warm_selected.end(), block.id) ==
                  warm_selected.end(),
              std::string(name) + ": no selected block is staged out");
    }
    const auto warm_tokens =
        static_cast<std::int64_t>(warm_selected.size()) * kBlockPages * kPageTokens;
    check(warm_plan.device_tokens == budget.device_tokens,
          std::string(name) + ": the warm plan keeps the device window it was given");
    check(static_cast<std::int64_t>(warm_plan.visible_tokens) == warm_tokens,
          std::string(name) + ": the warm plan counts the visible tokens");
    check(staged_in <= static_cast<std::int32_t>(warm_selected.size()),
          std::string(name) + ": the warm turn never stages in more than it selects");
}

} // namespace

int main() {
    try {
        // A 637k-token workspace with the budget measured on the 32 GB card, and the same history
        // against a budget too small to hold it: the compulsory blocks must still be served.
        scenario("637k", 637 * 1024, 98'304, 12'288, 1'024);
        scenario("262k", 262'144, 98'304, 12'288, 1'024);
        scenario("tight", 262'144, 16'384, 1'024, 1'024);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "kv_recall_plan: threw %s\n", error.what());
        return 2;
    }

    if (failures != 0) {
        std::fprintf(stderr, "kv_recall_plan: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("kv_recall_plan: all checks passed\n");
    return 0;
}