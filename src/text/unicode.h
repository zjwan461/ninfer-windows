#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::text::unicode_internal {

struct CodepointSpan {
    std::int32_t value = 0;
    std::size_t offset = 0;
    std::size_t length = 0;
};

std::string normalize_nfc(std::string_view text);
CodepointSpan utf8_codepoint_at(std::string_view text, std::size_t offset,
                                std::string_view context);
std::vector<CodepointSpan> utf8_codepoints(std::string_view text, std::string_view context);
std::string codepoint_to_utf8(std::int32_t codepoint);

bool is_letter(std::int32_t codepoint) noexcept;
// The Llama-3 pre-tokeniser splits on letter case, so it needs the categories apart rather than
// the single "is a letter" the normaliser is content with.
bool is_uppercase_letter(std::int32_t codepoint) noexcept;
bool is_titlecase_letter(std::int32_t codepoint) noexcept;
bool is_modifier_letter(std::int32_t codepoint) noexcept;
bool is_other_letter(std::int32_t codepoint) noexcept;
bool is_lowercase_letter(std::int32_t codepoint) noexcept;
bool is_mark(std::int32_t codepoint) noexcept;
bool is_number(std::int32_t codepoint) noexcept;
bool is_whitespace(std::int32_t codepoint) noexcept;

} // namespace ninfer::text::unicode_internal
