#pragma once

#include <string>
#include <string_view>

namespace ninfer::text {

// GPT-2's byte-to-unicode bijection, which is the alphabet a byte-level BPE vocabulary is written
// in: printable ASCII and the Latin-1 range map to themselves, and every other byte maps to
// U+0100 + its rank among those, so `text` becomes a string of visible characters with no
// ambiguity. The inverse is exact, so a token can always be turned back into its bytes.
std::string bytes_to_unicode(std::string_view bytes);
std::string unicode_to_bytes(std::string_view mapped);

} // namespace ninfer::text
