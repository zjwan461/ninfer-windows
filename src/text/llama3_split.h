#pragma once

#include <string_view>
#include <vector>

namespace ninfer::text {

// Splits text into the chunks a Llama-3 style pre-tokeniser produces before BPE. The pattern has
// seven alternatives and the order decides the result, so this is a scanner rather than a
// general-purpose regex: `tests/text/llama3_pretokenizer_reference.py` evaluates the same pattern
// with a Unicode-aware engine and provides the vectors this is checked against.
std::vector<std::string_view> llama3_split(std::string_view text);

} // namespace ninfer::text
