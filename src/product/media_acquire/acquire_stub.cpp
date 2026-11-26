// Headerless builds only: the real translation unit needs libcurl, so this one refuses every
// acquisition. A text-only session never fetches a URL; a request that does fails loudly.
#include "product/media_acquire/acquire.h"

namespace ninfer::product::media_acquire {

std::vector<std::uint8_t> acquire_bytes(const Source&, const Policy&) {
    throw Error(ErrorKind::RemoteUnavailable,
                "media acquisition is not available in this build (NINFER_WITH_MEDIA=OFF)");
}

} // namespace ninfer::product::media_acquire
