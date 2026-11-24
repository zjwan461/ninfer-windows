// The byte-level BPE encoder against ids produced by the library that owns the tokenizer file.

#include "text/byte_bpe.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
int failures = 0;

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string tokenizer_path = "/workspaces/models/inkling2/tokenizer.json";
        std::string fixture_path = "tests/text/fixtures/inkling2_tokens.json";
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--tokenizer" && i + 1 < argc) { tokenizer_path = argv[++i]; }
            else if (arg == "--fixture" && i + 1 < argc) { fixture_path = argv[++i]; }
            else { throw std::invalid_argument("unknown argument " + arg); }
        }

        const ninfer::text::ByteBpe encoder(read_file(tokenizer_path));
        const auto fixture = nlohmann::json::parse(read_file(fixture_path));
        std::cout << "vocabulary " << encoder.vocab_size() << "\n";
        for (const auto& vector : fixture) {
            const auto text = vector["text"].get<std::string>();
            const auto expected = vector["ids"].get<std::vector<std::int32_t>>();
            const auto actual = encoder.encode(text);
            if (actual != expected) {
                std::cout << "FAIL " << text << "\n  expected:";
                for (auto id : expected) { std::cout << " " << id; }
                std::cout << "\n  actual  :";
                for (auto id : actual) { std::cout << " " << id; }
                std::cout << "\n";
                ++failures;
                continue;
            }
            // Decoding is the inverse, which is what makes a round trip a real check.
            const auto decoded = encoder.decode(actual);
            if (decoded != text) {
                std::cout << "FAIL round trip " << text << " -> " << decoded << "\n";
                ++failures;
            }
        }
        if (failures != 0) {
            std::cerr << failures << " failures\n";
            return 1;
        }
        std::cout << "ok: the encoder reproduces the reference token ids\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
