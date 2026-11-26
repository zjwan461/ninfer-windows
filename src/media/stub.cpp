// Headerless builds only: the real translation unit needs the FFmpeg headers, so this one provides
// the same symbols and refuses every call. Vision input is unavailable in such a build; nothing
// that only decodes text can reach these.
#include "decode/decode.h"

namespace ninfer::media::decode {
namespace {

[[noreturn]] void unavailable() {
    throw Error(ErrorKind::InvalidInput,
                "media decode is not available in this build (NINFER_WITH_MEDIA=OFF)");
}

} // namespace

ImageInfo inspect_image(std::span<const std::uint8_t>, const Policy&) { unavailable(); }

VideoInfo inspect_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    unavailable();
}

Image decode_image(std::span<const std::uint8_t>, const Policy&) { unavailable(); }

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) { unavailable(); }

} // namespace ninfer::media::decode
