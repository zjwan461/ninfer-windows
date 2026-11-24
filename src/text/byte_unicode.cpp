// ninfer::text - the GPT-2 byte-to-unicode bijection used by byte-level BPE vocabularies.
#include "text/byte_unicode.h"

#include <array>
#include <cstdint>

namespace ninfer::text {
namespace {

constexpr bool printable(std::int32_t byte) {
    return (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || (byte >= 174 && byte <= 255);
}

struct Table {
    std::array<std::string, 256> encode;  // byte -> UTF-8 of the mapped codepoint
    std::array<int, 0x200> decode{};      // mapped codepoint -> byte, -1 when unused
};

const Table& table() {
    static const Table built = [] {
        Table out;
        for (auto& entry : out.decode) { entry = -1; }
        std::int32_t extra = 0;
        for (std::int32_t byte = 0; byte < 256; ++byte) {
            const std::int32_t mapped = printable(byte) ? byte : 256 + extra++;
            std::string encoded;
            if (mapped < 0x80) {
                encoded.push_back(static_cast<char>(mapped));
            } else if (mapped < 0x800) {
                encoded.push_back(static_cast<char>(0xC0 | (mapped >> 6)));
                encoded.push_back(static_cast<char>(0x80 | (mapped & 0x3F)));
            } else {
                encoded.push_back(static_cast<char>(0xE0 | (mapped >> 12)));
                encoded.push_back(static_cast<char>(0x80 | ((mapped >> 6) & 0x3F)));
                encoded.push_back(static_cast<char>(0x80 | (mapped & 0x3F)));
            }
            out.encode[static_cast<std::size_t>(byte)] = std::move(encoded);
            out.decode[static_cast<std::size_t>(mapped)] = byte;
        }
        return out;
    }();
    return built;
}

// Decodes one codepoint from the mapped string, returning its length in bytes.
std::size_t next_codepoint(std::string_view text, std::size_t offset, std::int32_t& value) {
    const auto lead = static_cast<unsigned char>(text[offset]);
    if (lead < 0x80) {
        value = lead;
        return 1;
    }
    if ((lead & 0xE0) == 0xC0 && offset + 1 < text.size()) {
        value = ((lead & 0x1F) << 6) | (static_cast<unsigned char>(text[offset + 1]) & 0x3F);
        return 2;
    }
    if ((lead & 0xF0) == 0xE0 && offset + 2 < text.size()) {
        value = ((lead & 0x0F) << 12) |
                ((static_cast<unsigned char>(text[offset + 1]) & 0x3F) << 6) |
                (static_cast<unsigned char>(text[offset + 2]) & 0x3F);
        return 3;
    }
    value = 0xFFFD;
    return 1;
}

} // namespace

std::string bytes_to_unicode(std::string_view bytes) {
    std::string out;
    out.reserve(bytes.size() * 2);
    const auto& map = table();
    for (const auto raw : bytes) {
        out += map.encode[static_cast<unsigned char>(raw)];
    }
    return out;
}

std::string unicode_to_bytes(std::string_view mapped) {
    std::string out;
    out.reserve(mapped.size());
    const auto& map = table();
    std::size_t offset = 0;
    while (offset < mapped.size()) {
        std::int32_t codepoint = 0;
        offset += next_codepoint(mapped, offset, codepoint);
        if (codepoint >= 0 && codepoint < 0x200) {
            const int byte = map.decode[static_cast<std::size_t>(codepoint)];
            if (byte >= 0) {
                out.push_back(static_cast<char>(byte));
                continue;
            }
        }
        out.push_back(static_cast<char>(0xEF));  // not part of the alphabet: a replacement character
        out.push_back(static_cast<char>(0xBF));
        out.push_back(static_cast<char>(0xBD));
    }
    return out;
}

} // namespace ninfer::text
