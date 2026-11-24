// The GPT-2 byte-to-unicode bijection: exact in both directions, and the sizes they expect.

#include "text/byte_unicode.h"

#include <iostream>
#include <string>

namespace {
int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cout << "FAIL " << label << "\n";
    ++failures;
}
} // namespace

int main() {
    // Every byte must survive a round trip, and the mapping must be a bijection over 256 entries.
    std::string all_bytes;
    for (int byte = 0; byte < 256; ++byte) { all_bytes.push_back(static_cast<char>(byte)); }
    const auto mapped = ninfer::text::bytes_to_unicode(all_bytes);
    expect(mapped.size() > all_bytes.size(), "mapping expands the non-ASCII bytes");
    expect(ninfer::text::unicode_to_bytes(mapped) == all_bytes, "round trip over all 256 bytes");

    // The two the vocabulary actually leans on: a space is a single visible character and a newline
    // is another, which is why byte-level vocabularies look like they contain punctuation.
    expect(ninfer::text::bytes_to_unicode(" ") == "\xC4\xA0", "space maps to U+0120");
    expect(ninfer::text::bytes_to_unicode("\n") == "\xC4\x8A", "newline maps to U+010A");
    expect(ninfer::text::bytes_to_unicode("a") == "a", "printable ASCII maps to itself");
    expect(ninfer::text::unicode_to_bytes("\xC4\xA0") == " ", "U+0120 maps back to a space");

    if (failures != 0) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "ok: the byte mapping is exact in both directions\n";
    return 0;
}
