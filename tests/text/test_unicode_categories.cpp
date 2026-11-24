// The letter-case categories the Llama-3 pre-tokeniser splits on.

#include "text/unicode.h"

#include <cstdint>
#include <iostream>
#include <string>

namespace {
using namespace ninfer::text::unicode_internal;

int failures = 0;

void expect(bool actual, bool expected, const std::string& label) {
    if (actual == expected) { return; }
    std::cout << "FAIL " << label << ": got " << (actual ? "true" : "false") << ", expected "
              << (expected ? "true" : "false") << "\n";
    ++failures;
}

void check(std::int32_t codepoint, const std::string& label, bool upper, bool title, bool modifier,
           bool other, bool lower) {
    expect(is_uppercase_letter(codepoint), upper, label + " uppercase");
    expect(is_titlecase_letter(codepoint), title, label + " titlecase");
    expect(is_modifier_letter(codepoint), modifier, label + " modifier");
    expect(is_other_letter(codepoint), other, label + " other");
    expect(is_lowercase_letter(codepoint), lower, label + " lowercase");
}
} // namespace

int main() {
    check('A', "A", true, false, false, false, false);
    check('a', "a", false, false, false, false, true);
    check(0x01C5, "Dz titlecase", false, true, false, false, false);
    check(0x02B0, "modifier h", false, false, true, false, false);
    check(0x4E2D, "CJK middle", false, false, false, true, false);
    check(0x05D0, "hebrew alef", false, false, false, true, false);
    check('1', "digit", false, false, false, false, false);
    check(' ', "space", false, false, false, false, false);

    // Every category predicate must agree with the coarse one it refines.
    for (std::int32_t codepoint = 0x20; codepoint < 0x3000; ++codepoint) {
        const bool any = is_uppercase_letter(codepoint) || is_titlecase_letter(codepoint) ||
                         is_modifier_letter(codepoint) || is_other_letter(codepoint) ||
                         is_lowercase_letter(codepoint);
        if (any != is_letter(codepoint)) {
            std::cout << "FAIL category mismatch at U+" << std::hex << codepoint << std::dec
                      << "\n";
            ++failures;
            break;
        }
    }

    if (failures != 0) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "ok: letter categories match their definitions\n";
    return 0;
}
