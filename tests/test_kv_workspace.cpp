#include "core/kv_workspace.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int expect(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        return 1;
    }
    return 0;
}

int expect_throws(auto&& call, const char* label) {
    try {
        call();
    } catch (const std::invalid_argument&) {
        return 0;
    } catch (const std::runtime_error&) {
        return 0;
    }
    std::cerr << "FAIL: " << label << " did not throw\n";
    return 1;
}

using ninfer::KvBlock;
using ninfer::KvBlockTier;
using ninfer::KvRemap;
using ninfer::WorkspaceBudget;

KvBlock block(std::int32_t id, std::int32_t tokens, KvBlockTier tier, bool pinned = false) {
    return KvBlock{id, id * tokens, tokens, tier, pinned};
}

} // namespace

int main() {
    int failures = 0;

    // Everything resident and selected: no movement, contiguous remaps.
    {
        const std::vector<KvBlock> blocks{block(0, 128, KvBlockTier::Device),
                                          block(1, 128, KvBlockTier::Device)};
        const std::vector<std::int32_t> selected{0, 1};
        const auto plan = ninfer::plan_workspace(WorkspaceBudget{1024, 128}, blocks, selected);
        failures += expect(plan.stage_in.empty() && plan.stage_out.empty(), "no-op has no movement");
        failures += expect(plan.visible_tokens == 256, "no-op visible tokens");
        failures += expect(plan.remaps.size() == 2 && plan.remaps[0].compact_start == 0 &&
                               plan.remaps[1].compact_start == 128,
                           "no-op remaps assemble from zero");
    }

    // A middle block is dropped: it spills, and the remainder compacts with no hole.
    {
        const std::vector<KvBlock> blocks{block(0, 128, KvBlockTier::Device, true),
                                          block(1, 128, KvBlockTier::Device),
                                          block(2, 128, KvBlockTier::Device, true)};
        const std::vector<std::int32_t> selected{0, 2};
        const auto plan = ninfer::plan_workspace(WorkspaceBudget{1024, 128}, blocks, selected);
        failures += expect(plan.stage_out.size() == 1 && plan.stage_out[0].id == 1,
                           "dropped block spills");
        failures += expect(plan.stage_in.empty(), "nothing restored in this plan");
        failures += expect(plan.remaps.size() == 2 && plan.remaps[0].compact_start == 0 &&
                               plan.remaps[1].compact_start == 128,
                           "the surviving blocks close the hole");
        failures += expect(plan.visible_tokens == 256, "compaction keeps the live length");
    }

    // Host blocks come back, and both directions are ordered by chronology.
    {
        const std::vector<KvBlock> blocks{block(0, 128, KvBlockTier::Host, true),
                                          block(1, 128, KvBlockTier::Device),
                                          block(2, 128, KvBlockTier::Host)};
        const std::vector<std::int32_t> selected{2, 0, 1};
        const auto plan = ninfer::plan_workspace(WorkspaceBudget{1024, 128}, blocks, selected);
        failures += expect(plan.stage_in.size() == 2 && plan.stage_in[0].id == 0 &&
                               plan.stage_in[1].id == 2,
                           "restores follow chronology");
        failures += expect(plan.stage_out.empty(), "nothing spills when everything is selected");
        failures += expect(plan.remaps.size() == 3 && plan.remaps[2].compact_start == 256,
                           "three blocks assemble contiguously");
    }

    // Duplicate selections collapse.
    {
        const std::vector<KvBlock> blocks{block(0, 128, KvBlockTier::Device)};
        const std::vector<std::int32_t> selected{0, 0, 0};
        const auto plan = ninfer::plan_workspace(WorkspaceBudget{1024, 0}, blocks, selected);
        failures += expect(plan.remaps.size() == 1 && plan.visible_tokens == 128,
                           "duplicates collapse to one block");
    }

    // Pinned blocks may not be dropped, and the request is validated before any planning.
    {
        const std::vector<KvBlock> blocks{block(0, 128, KvBlockTier::Device, true),
                                          block(1, 128, KvBlockTier::Device)};
        const std::vector<std::int32_t> selected{1};
        failures += expect_throws(
            [&] { (void)ninfer::plan_workspace(WorkspaceBudget{1024, 0}, blocks, selected); },
            "dropping a pinned block");
    }

    // Unknown ids, bad geometry and over-capacity are all refused.
    {
        const std::vector<KvBlock> blocks{block(0, 128, KvBlockTier::Device)};
        const std::vector<std::int32_t> unknown{7};
        failures += expect_throws(
            [&] { (void)ninfer::plan_workspace(WorkspaceBudget{1024, 0}, blocks, unknown); },
            "unknown id");

        const std::vector<KvBlock> unordered{block(1, 128, KvBlockTier::Device),
                                             block(0, 128, KvBlockTier::Device)};
        const std::vector<std::int32_t> all{0, 1};
        failures += expect_throws(
            [&] { (void)ninfer::plan_workspace(WorkspaceBudget{1024, 0}, unordered, all); },
            "unordered history");

        failures += expect_throws(
            [&] { (void)ninfer::plan_workspace(WorkspaceBudget{1024, 1024}, blocks, all); },
            "reserve equal to the window");

        const std::vector<KvBlock> big{block(0, 1024, KvBlockTier::Device)};
        const std::vector<std::int32_t> one{0};
        failures += expect_throws(
            [&] { (void)ninfer::plan_workspace(WorkspaceBudget{1024, 128}, big, one); },
            "selection plus reserve over the window");
    }

    // The plan is deterministic under permuted input: the selection is a set, not a sequence.
    {
        const std::vector<KvBlock> blocks{block(0, 128, KvBlockTier::Host, true),
                                          block(1, 128, KvBlockTier::Host),
                                          block(2, 128, KvBlockTier::Device, true)};
        const std::vector<std::int32_t> a{2, 1, 0};
        const std::vector<std::int32_t> b{0, 1, 2};
        const auto plan_a = ninfer::plan_workspace(WorkspaceBudget{1024, 128}, blocks, a);
        const auto plan_b = ninfer::plan_workspace(WorkspaceBudget{1024, 128}, blocks, b);
        failures += expect(plan_a.remaps == plan_b.remaps, "selection order is irrelevant");
        failures += expect(plan_a.stage_in.size() == plan_b.stage_in.size() &&
                               plan_a.stage_in.front().id == plan_b.stage_in.front().id,
                           "restore order is canonical");
    }

    // Block selection under a page budget.
    {
        const ninfer::KvSelectSpec spec{4, 4, 4, 12};
        // 16 pages, 4 pages per block, so four blocks; sink block 0 and recent block 3 are mandatory.
        const std::vector<float> scores{0.1F, 0.9F, 0.5F, 0.2F};
        const std::vector<std::int32_t> chosen = ninfer::select_kv_blocks(scores, 16, spec);
        failures += expect(chosen == std::vector<std::int32_t>({0, 1, 3}),
                           "highest scoring block is added to the mandatory ones");

        const ninfer::KvSelectSpec tight{4, 4, 4, 8};
        failures += expect(ninfer::select_kv_blocks(scores, 16, tight) ==
                               std::vector<std::int32_t>({0, 3}),
                           "a budget of two blocks keeps only the mandatory ones");

        const ninfer::KvSelectSpec all{4, 4, 4, 16};
        failures += expect(ninfer::select_kv_blocks(scores, 16, all) ==
                               std::vector<std::int32_t>({0, 1, 2, 3}),
                           "a full budget keeps every block");

        const std::vector<float> tied{0.5F, 0.5F, 0.5F, 0.5F};
        failures += expect(ninfer::select_kv_blocks(tied, 16, ninfer::KvSelectSpec{4, 4, 4, 12}) ==
                               std::vector<std::int32_t>({0, 1, 3}),
                           "ties go to the lower block id");

        // 10 pages: the trailing block holds 2 pages, so it is cheaper than a full one.
        const std::vector<float> tail_scores{0.1F, 0.2F, 0.9F};
        failures += expect(ninfer::select_kv_blocks(tail_scores, 10, ninfer::KvSelectSpec{4, 4, 2,
                                                                                         6}) ==
                               std::vector<std::int32_t>({0, 2}),
                           "a short trailing block costs its own pages");
    }

    // Selection rejects geometry it cannot honour instead of silently shrinking the request.
    {
        const std::vector<float> scores{0.5F, 0.5F, 0.5F, 0.5F};
        failures += expect_throws(
            [&] {
                (void)ninfer::select_kv_blocks(scores, 16, ninfer::KvSelectSpec{4, 4, 4, 4});
            },
            "mandatory blocks exceeding the budget");
        failures += expect_throws(
            [&] { (void)ninfer::select_kv_blocks(scores, 16, ninfer::KvSelectSpec{0, 4, 4, 16}); },
            "zero block_pages");
        failures += expect_throws(
            [&] {
                (void)ninfer::select_kv_blocks(scores, 16, ninfer::KvSelectSpec{4, 16, 16, 16});
            },
            "sink and recent exceeding the request");
        failures += expect_throws(
            [&] {
                (void)ninfer::select_kv_blocks(std::vector<float>{0.5F}, 16,
                                               ninfer::KvSelectSpec{4, 4, 4, 16});
            },
            "a score per block is required");
    }

    if (failures == 0) {
        std::cout << "kv_workspace: all checks passed\n";
    }
    return failures == 0 ? 0 : 1;
}