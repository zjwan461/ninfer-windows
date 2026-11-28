// Tests for src/core/kv_view.h: the logical compaction of a request's pages.
#include "core/kv_view.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::core::build_kv_view;
using ninfer::core::KvView;
using ninfer::core::KvViewSpec;

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
    }
    std::cerr << "FAIL: " << label << " did not throw\n";
    return 1;
}

std::vector<std::int32_t> pages_of(std::int32_t count, std::int32_t first = 100) {
    std::vector<std::int32_t> pages(static_cast<std::size_t>(count));
    for (std::int32_t i = 0; i < count; ++i) {
        pages[static_cast<std::size_t>(i)] = first + i * 7;
    }
    return pages;
}

int identity_case() {
    const std::vector<std::int32_t> pages = pages_of(12);
    const KvView view = build_kv_view(pages, KvViewSpec{4, 0, 0}, {0, 1, 2});
    int failures = 0;
    failures += expect(view.pages == pages, "selecting every block returns the page order");
    failures += expect(view.visible_pages == 12, "selecting every block keeps every row");
    failures += expect(view.blocks == std::vector<std::int32_t>({0, 1, 2}),
                       "every selected block is reported");
    return failures;
}

int middle_eviction_case() {
    const std::vector<std::int32_t> pages = pages_of(12);
    const KvView view = build_kv_view(pages, KvViewSpec{4, 0, 0}, {0, 2});
    int failures = 0;
    failures += expect(view.visible_pages == 8, "evicting the middle block drops four rows");
    failures += expect(view.pages.size() == 8, "the executed page order has eight entries");
    for (int i = 0; i < 4; ++i) {
        failures += expect(view.pages[static_cast<std::size_t>(i)] == pages[static_cast<std::size_t>(i)],
                           "the first block keeps its pages in order");
        failures += expect(view.pages[static_cast<std::size_t>(4 + i)] ==
                               pages[static_cast<std::size_t>(8 + i)],
                           "the last block follows the first without a hole");
    }
    return failures;
}

int sink_and_recent_case() {
    const std::vector<std::int32_t> pages = pages_of(12);
    const KvView view = build_kv_view(pages, KvViewSpec{4, 2, 2}, {});
    int failures = 0;
    failures += expect(view.visible_pages == 4, "an empty selection keeps only sink and recent");
    failures += expect(view.pages[0] == pages[0] && view.pages[1] == pages[1],
                       "the sink prefix is kept from the front");
    failures += expect(view.pages[2] == pages[10] && view.pages[3] == pages[11],
                       "the recent tail is kept from the back");
    failures += expect(view.blocks.empty(), "no block is reported as selected");
    return failures;
}

int sink_recent_override_case() {
    const std::vector<std::int32_t> pages = pages_of(12);
    // Block 1 is not selected, but its first page is a sink page and its last page is recent.
    const KvView view = build_kv_view(pages, KvViewSpec{4, 5, 2}, {});
    int failures = 0;
    failures += expect(view.visible_pages == 7, "sink and recent pages survive their blocks");
    failures += expect(view.pages[4] == pages[4], "the last sink page is present");
    failures += expect(view.pages[5] == pages[10] && view.pages[6] == pages[11],
                       "the recent tail is present");
    return failures;
}

int partial_trailing_block_case() {
    const std::vector<std::int32_t> pages = pages_of(10);
    const KvView view = build_kv_view(pages, KvViewSpec{4, 0, 0}, {2});
    int failures = 0;
    failures += expect(view.visible_pages == 2, "the short trailing block keeps its two pages");
    failures += expect(view.pages[0] == pages[8] && view.pages[1] == pages[9],
                       "the short trailing block keeps identity and order");
    failures += expect(view.blocks == std::vector<std::int32_t>({2}), "the short block is reported");
    return failures;
}

int unsorted_selection_case() {
    const std::vector<std::int32_t> pages = pages_of(12);
    const KvView a = build_kv_view(pages, KvViewSpec{4, 0, 0}, {2, 1});
    const KvView b = build_kv_view(pages, KvViewSpec{4, 0, 0}, {1, 2, 1});
    int failures = 0;
    failures += expect(a.pages == b.pages, "selection order does not affect the page order");
    failures += expect(a.blocks == std::vector<std::int32_t>({1, 2}),
                       "blocks are reported in time order");
    return failures;
}

int invalid_case() {
    const std::vector<std::int32_t> pages = pages_of(12);
    int failures = 0;
    failures += expect_throws([&] { build_kv_view({}, KvViewSpec{4, 0, 0}, {}); },
                              "an empty request");
    failures += expect_throws([&] { build_kv_view(pages, KvViewSpec{0, 0, 0}, {}); },
                              "zero block_pages");
    failures += expect_throws([&] { build_kv_view(pages, KvViewSpec{4, -1, 0}, {}); },
                              "negative sink_pages");
    failures += expect_throws([&] { build_kv_view(pages, KvViewSpec{4, 7, 7}, {}); },
                              "sink and recent exceeding the request");
    failures += expect_throws([&] { build_kv_view(pages, KvViewSpec{4, 0, 0}, {3}); },
                              "a selected block out of range");
    failures += expect_throws([&] { build_kv_view(pages, KvViewSpec{4, 0, 0}, {-1}); },
                              "a negative block index");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += identity_case();
    failures += middle_eviction_case();
    failures += sink_and_recent_case();
    failures += sink_recent_override_case();
    failures += partial_trailing_block_case();
    failures += unsorted_selection_case();
    failures += invalid_case();

    if (failures == 0) { std::cout << "kv_view: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}