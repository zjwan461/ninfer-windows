#pragma once

// Compaction of a request's logical KV pages into the executed prefix.
//
// The attention path of this engine reads cache rows [0,p] through one execution
// table, so a recall view has to be a *contiguous* run of pages in time order; the
// table is the place where selecting a subset of history becomes cheap. This header
// owns that purely logical step: it takes the request's pages in time order plus a
// set of selected recall blocks and returns the page order to execute, with the
// unpinned sink prefix and the recent tail always present.
//
// Nothing here touches storage, RoPE or the device. Pages keep their physical
// identity, so keys already written keep the position phase they were written with;
// only the row they are read back through changes.

#include <cstdint>
#include <vector>

namespace ninfer::core {

struct KvViewSpec {
    // Pages that form one recall block. The selection granularity is a whole block.
    std::int32_t block_pages = 0;
    // Pages at the start of the request that are never dropped (attention sinks).
    std::int32_t sink_pages = 0;
    // Pages at the end of the request that are never dropped (the live frontier).
    std::int32_t recent_pages = 0;
};

struct KvView {
    // Physical page identity of each executed row, in time order.
    std::vector<std::int32_t> pages;
    // Executed rows. The attention bound is this many pages.
    std::int32_t visible_pages = 0;
    // Blocks that survived selection, in time order.
    std::vector<std::int32_t> blocks;
};

/**
 * Builds the executed page order for one request.
 *
 * `pages` holds the request's physical page identities in time order. Blocks are the
 * consecutive runs of `block_pages` pages; the trailing block may be short.
 * `selected_blocks` holds block indices in any order and may repeat. The result is
 * always in ascending time order and contains, for every page of `pages`:
 *
 *   - pages of a selected block,
 *   - the sink prefix [0, sink_pages),
 *   - the recent tail [pages.size() - recent_pages, pages.size()).
 *
 * A page outside every rule is dropped. Selection never reorders pages, so a request
 * whose every block is selected yields its own page order unchanged.
 *
 * Throws std::invalid_argument on inconsistent geometry (non-positive `block_pages`,
 * negative sink/recent, sink + recent larger than the request, a selected block index
 * out of range, or an empty request).
 */
[[nodiscard]] KvView build_kv_view(const std::vector<std::int32_t>& pages, const KvViewSpec& spec,
                                   const std::vector<std::int32_t>& selected_blocks);

} // namespace ninfer::core