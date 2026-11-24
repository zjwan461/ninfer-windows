// ninfer::text - the Llama-3 pre-tokeniser split, as a scanner rather than a regex engine.
#include "text/llama3_split.h"

#include "text/unicode.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::text {
namespace {

using unicode_internal::CodepointSpan;
using unicode_internal::is_lowercase_letter;
using unicode_internal::is_mark;
using unicode_internal::is_modifier_letter;
using unicode_internal::is_number;
using unicode_internal::is_other_letter;
using unicode_internal::is_titlecase_letter;
using unicode_internal::is_uppercase_letter;
using unicode_internal::is_whitespace;

struct Scanner {
    std::string_view text;
    std::vector<CodepointSpan> codepoints;

    std::size_t size() const { return codepoints.size(); }
    std::int32_t value(std::size_t index) const { return codepoints[index].value; }

    // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
    bool upper_class(std::size_t index) const {
        const auto cp = value(index);
        return is_uppercase_letter(cp) || is_titlecase_letter(cp) || is_modifier_letter(cp) ||
               is_other_letter(cp) || is_mark(cp);
    }
    // [\p{Ll}\p{Lm}\p{Lo}\p{M}]
    bool lower_class(std::size_t index) const {
        const auto cp = value(index);
        return is_lowercase_letter(cp) || is_modifier_letter(cp) || is_other_letter(cp) ||
               is_mark(cp);
    }
    bool letter(std::size_t index) const {
        const auto cp = value(index);
        return is_uppercase_letter(cp) || is_titlecase_letter(cp) || is_modifier_letter(cp) ||
               is_other_letter(cp) || is_lowercase_letter(cp);
    }
    bool digit(std::size_t index) const { return is_number(value(index)); }
    bool space(std::size_t index) const { return is_whitespace(value(index)); }
    bool newline(std::size_t index) const {
        const auto cp = value(index);
        return cp == '\r' || cp == '\n';
    }
    // [^\r\n\p{L}\p{N}]
    bool prefix_class(std::size_t index) const {
        return !newline(index) && !letter(index) && !digit(index);
    }
    // ?[^\s\p{L}\p{N}]
    bool punctuation(std::size_t index) const {
        return !space(index) && !letter(index) && !digit(index);
    }

    // (?i:'s|'t|'re|'ve|'m|'ll|'d), in the pattern's order.
    std::size_t contraction(std::size_t at) const {
        struct Entry { const char* suffix; };
        static constexpr Entry kEntries[] = {{"'s"}, {"'t"}, {"'re"}, {"'ve"}, {"'m"}, {"'ll"},
                                             {"'d"}};
        for (const auto& entry : kEntries) {
            std::string_view suffix(entry.suffix);
            if (at + suffix.size() > size()) { continue; }
            bool matches = true;
            for (std::size_t i = 0; i < suffix.size(); ++i) {
                const auto cp = value(at + i);
                const auto wanted = static_cast<char>(suffix[i]);
                const auto lower = (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp;
                if (lower != wanted) { matches = false; break; }
            }
            if (matches) { return at + suffix.size(); }
        }
        return at;
    }

    // The two word alternatives: A* B+ (first) and A+ B* (second), each with an optional
    // contraction. Greedy A* is honoured by trying the longest A run first.
    std::size_t word_match(std::size_t at, bool first_alternative) const {
        std::size_t pos = at;
        if (pos < size() && prefix_class(pos)) { ++pos; }
        std::size_t upper_run = pos;
        while (upper_run < size() && upper_class(upper_run)) { ++upper_run; }
        std::size_t end = 0;
        if (first_alternative) {
            for (std::size_t split = upper_run; split + 1 > pos; --split) {
                std::size_t run = split;
                while (run < size() && lower_class(run)) { ++run; }
                if (run > split) { end = run; break; }
                if (split == pos) { break; }
            }
        } else {
            if (upper_run > pos) {
                end = upper_run;
                while (end < size() && lower_class(end)) { ++end; }
            }
        }
        if (end == 0) { return at; }
        return contraction(end);
    }

    //  ?[^\s\p{L}\p{N}]+[\r\n/]*
    std::size_t punctuation_match(std::size_t at) const {
        std::size_t pos = at;
        if (pos < size() && value(pos) == ' ') { ++pos; }
        std::size_t run = pos;
        while (run < size() && punctuation(run)) { ++run; }
        if (run == pos) { return at; }
        std::size_t end = run;
        while (end < size() && (newline(end) || value(end) == '/')) { ++end; }
        return end;
    }
};
} // namespace

std::vector<std::string_view> llama3_split(std::string_view text) {
    Scanner scanner{text, unicode_internal::utf8_codepoints(text, "llama3_split")};
    std::vector<std::string_view> chunks;
    std::size_t at = 0;
    while (at < scanner.size()) {
        std::size_t end = scanner.word_match(at, true);
        if (end == at) { end = scanner.word_match(at, false); }
        if (end == at) {
            // \p{N}{1,3}
            std::size_t run = at;
            while (run < scanner.size() && scanner.digit(run) && run - at < 3) { ++run; }
            end = run;
        }
        if (end == at) { end = scanner.punctuation_match(at); }
        if (end == at) {
            // \s*[\r\n]+ : a whitespace run that ends on a newline.
            std::size_t run = at;
            while (run < scanner.size() && scanner.space(run)) { ++run; }
            while (run > at && !scanner.newline(run - 1)) { --run; }
            end = run;
        }
        if (end == at) {
            // \s+(?!\S) : a whitespace run whose next codepoint is whitespace or the end. The
            // quantifier is greedy, so it takes as much of the run as the lookahead allows - which
            // for a run that does not reach the end is all but its last codepoint.
            std::size_t run = at;
            while (run < scanner.size() && scanner.space(run)) { ++run; }
            for (std::size_t candidate = run; candidate > at; --candidate) {
                const bool next_is_space_or_end =
                    candidate == scanner.size() || scanner.space(candidate);
                if (next_is_space_or_end) { end = candidate; break; }
            }
        }
        if (end == at) {
            // \s+
            std::size_t run = at;
            while (run < scanner.size() && scanner.space(run)) { ++run; }
            end = run;
        }
        if (end == at) { ++end; }  // a codepoint no alternative claims still has to advance
        const auto begin = scanner.codepoints[at].offset;
        const auto finish = end < scanner.size() ? scanner.codepoints[end].offset : text.size();
        chunks.push_back(text.substr(begin, finish - begin));
        at = end;
    }
    return chunks;
}

} // namespace ninfer::text
