// Implements: src/core/kv_view.h
#include "core/kv_view.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::core {
namespace {

std::int32_t block_of(std::int32_t page, std::int32_t block_pages) {
    return page / block_pages;
}

} // namespace

KvView build_kv_view(const std::vector<std::int32_t>& pages, const KvViewSpec& spec,
                     const std::vector<std::int32_t>& selected_blocks) {
    if (pages.empty()) { throw std::invalid_argument("kv_view: the request has no pages"); }
    if (spec.block_pages <= 0) {
        throw std::invalid_argument("kv_view: block_pages must be positive");
    }
    if (spec.sink_pages < 0 || spec.recent_pages < 0) {
        throw std::invalid_argument("kv_view: sink_pages and recent_pages must be nonnegative");
    }

    const std::int32_t page_count = static_cast<std::int32_t>(pages.size());
    if (spec.sink_pages + spec.recent_pages > page_count) {
        throw std::invalid_argument("kv_view: sink_pages + recent_pages exceeds the request");
    }
    const std::int32_t block_count = (page_count + spec.block_pages - 1) / spec.block_pages;

    std::vector<bool> keep(static_cast<std::size_t>(page_count), false);
    for (std::int32_t page = 0; page < spec.sink_pages; ++page) { keep[page] = true; }
    for (std::int32_t page = page_count - spec.recent_pages; page < page_count; ++page) {
        keep[static_cast<std::size_t>(page)] = true;
    }

    std::vector<bool> selected(static_cast<std::size_t>(block_count), false);
    for (const std::int32_t block : selected_blocks) {
        if (block < 0 || block >= block_count) {
            throw std::invalid_argument("kv_view: selected block " + std::to_string(block) +
                                        " is outside [0," + std::to_string(block_count) + ")");
        }
        selected[static_cast<std::size_t>(block)] = true;
    }

    KvView view;
    view.blocks.reserve(selected_blocks.size());
    view.pages.reserve(static_cast<std::size_t>(page_count));
    for (std::int32_t page = 0; page < page_count; ++page) {
        const std::int32_t block = block_of(page, spec.block_pages);
        if (!keep[static_cast<std::size_t>(page)] && !selected[static_cast<std::size_t>(block)]) {
            continue;
        }
        view.pages.push_back(pages[static_cast<std::size_t>(page)]);
    }

    for (std::int32_t block = 0; block < block_count; ++block) {
        if (!selected[static_cast<std::size_t>(block)]) { continue; }
        view.blocks.push_back(block);
    }

    view.visible_pages = static_cast<std::int32_t>(view.pages.size());
    return view;
}

KvViewBoundary plan_view_boundary(std::int32_t original_frontier, std::int32_t visible_tokens,
                                  std::int32_t current_rope_delta) {
    if (original_frontier < 0 || visible_tokens < 0) {
        throw std::invalid_argument("kv_view: boundary positions must be nonnegative");
    }
    if (current_rope_delta < 0) {
        throw std::invalid_argument("kv_view: rope delta must be nonnegative");
    }
    if (visible_tokens > original_frontier) {
        throw std::invalid_argument("kv_view: a view cannot be longer than the history it views");
    }

    KvViewBoundary boundary;
    boundary.dropped_tokens = original_frontier - visible_tokens;
    boundary.rope_delta = current_rope_delta + boundary.dropped_tokens;
    boundary.execution_frontier = visible_tokens;
    boundary.committed_frontier = original_frontier;
    return boundary;
}

} // namespace ninfer::core