#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::text {

// A byte-level BPE encoder driven by a `tokenizer.json`, with the pre-tokeniser Inkling 2 uses.
// The tokenization is: extract the special tokens, split the rest with the Llama-3 pattern, map
// each chunk's bytes into the vocabulary's alphabet, then merge by rank.
class ByteBpe {
public:
    explicit ByteBpe(std::string_view tokenizer_json);
    ~ByteBpe();
    ByteBpe(const ByteBpe&);
    ByteBpe& operator=(const ByteBpe&);
    ByteBpe(ByteBpe&&) noexcept;
    ByteBpe& operator=(ByteBpe&&) noexcept;

    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text) const;
    [[nodiscard]] std::string decode(std::span<const std::int32_t> ids) const;
    [[nodiscard]] std::size_t vocab_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::text
