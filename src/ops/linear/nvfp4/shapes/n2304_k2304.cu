// Inkling-2-Mini's routed experts are both 2304 x 2304: the fused gate/up projection (1152 rows
// expanded twice) and the down projection. The decode route is the same tuned GEMV schedule the
// other shapes use; only the geometry differs, and 2304 needs a 256-value phase, so this shape
// reads 8 values per lane instead of the 16 the 5120-wide shapes use.
//
// The schedule is inherited from the 5120-wide shapes rather than tuned for this geometry. Two
// alternatives were tried and measured nothing: three runs of the *same* binary on the shared slice
// used here gave 15.2, 17.1 and 18.2 tok/s, so its run-to-run spread is about twenty percent and
// the 14.7 - 16.4 the alternatives reported is inside it. Tuning this belongs on the target cards,
// where the noise floor is a dedicated GPU.
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_shapes.h"

namespace ninfer::ops::detail {
namespace {
using Geometry = Nvfp4Geometry<2304, 2304>;
using Gemv =
    Nvfp4GemvSchedule<8, 2, 8, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
using FullChunk = Nvfp4SimtSchedule<4, 1, 2, 8, 32, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                    Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                    Nvfp4SimtBlockOrder::RowsContiguous, 1>;

Nvfp4Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) { return launch_nvfp4_gemv<Geometry, Gemv>; }
    if (tokens <= 32) { return launch_nvfp4_simt<Geometry, 32, FullChunk, true>; }
    throw std::logic_error("nvfp4 N2304K2304: prefill beyond 32 tokens has no route");
}

Nvfp4A4Route select_a4(std::int32_t) {
    throw std::logic_error("nvfp4 N2304K2304: no A4 route");
}

bool uses_a4(std::int32_t, std::int32_t) { return false; }
} // namespace

const Nvfp4LinearShape kNvfp4N2304K2304{2304, 2304, launch_nvfp4_a16_chunks<32, select_a16>, launch_nvfp4_a4<select_a4>,
                                        uses_a4};
} // namespace ninfer::ops::detail
