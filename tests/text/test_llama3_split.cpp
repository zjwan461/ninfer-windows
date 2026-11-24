// The Llama-3 split scanner against chunks produced by the pattern itself.

#include "text/llama3_split.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(const std::string& text, const std::vector<std::string>& expected) {
    const auto actual = ninfer::text::llama3_split(text);
    bool same = actual.size() == expected.size();
    for (std::size_t i = 0; same && i < actual.size(); ++i) { same = actual[i] == expected[i]; }
    if (same) { return; }
    std::cout << "FAIL " << text << "\n  expected:";
    for (const auto& chunk : expected) { std::cout << " [" << chunk << "]"; }
    std::cout << "\n  actual  :";
    for (const auto& chunk : actual) { std::cout << " [" << chunk << "]"; }
    std::cout << "\n";
    ++failures;
}
} // namespace

int main() {
    // The vectors are the ones llama3_pretokenizer_reference.py produces for these inputs.
    expect("hello world", {"hello", " world"});
    expect("Hello World", {"Hello", " World"});
    expect("  leading spaces", {" ", " leading", " spaces"});
    expect("trailing   ", {"trailing", "   "});
    expect("don't stop", {"don't", " stop"});
    expect("I'll've", {"I'll", "'ve"});
    expect("MACD2000", {"MACD", "200", "0"});
    expect("2024", {"202", "4"});
    expect("a1b2", {"a", "1", "b", "2"});
    expect("code: x = 1;", {"code", ":", " x", " =", " ", "1", ";"});
    expect("line\nbreak\r\nwindows", {"line", "\n", "break", "\r\n", "windows"});
    expect("", {});

    if (failures != 0) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "ok: the split matches the reference chunks\n";
    return 0;
}
