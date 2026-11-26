// Offline builds only: the real translation unit needs libcurl. Local files, inline data and
// already-acquired bytes never need the network, so those three kinds are served here and only a
// URL is refused - which keeps local image input usable in a build without curl.
#include "product/media_acquire/acquire.h"

#include <fstream>
#include <iterator>

namespace ninfer::product::media_acquire {

std::vector<std::uint8_t> acquire_bytes(const Source& source, const Policy& policy) {
    switch (source.kind) {
    case SourceKind::Bytes:
        return source.bytes;
    case SourceKind::Data:
        return source.bytes;
    case SourceKind::Path: {
        std::ifstream file(source.value, std::ios::binary);
        if (!file.is_open()) {
            throw Error(ErrorKind::RemoteUnavailable, "media file is unavailable: " + source.value);
        }
        std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)),
                                        std::istreambuf_iterator<char>());
        if (bytes.size() > policy.max_bytes) {
            throw Error(ErrorKind::BudgetExceeded, "media file exceeds the byte budget");
        }
        return bytes;
    }
    case SourceKind::Url:
        throw Error(ErrorKind::RemoteUnavailable,
                    "URL media acquisition is not available in this build "
                    "(NINFER_WITH_REMOTE_FETCH=OFF)");
    }
    throw Error(ErrorKind::RemoteUnavailable, "unknown media source");
}

} // namespace ninfer::product::media_acquire
