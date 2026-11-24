// ninfer::text - byte-level BPE with the Llama-3 pre-tokeniser.
#include "text/byte_bpe.h"

#include "text/byte_unicode.h"
#include "text/llama3_split.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::text {
namespace {

using Json = nlohmann::json;

// One merge rule, keyed by the pair of token strings it joins.
struct PairKey {
    std::string left;
    std::string right;
    bool operator==(const PairKey& other) const {
        return left == other.left && right == other.right;
    }
};

struct PairHash {
    std::size_t operator()(const PairKey& key) const {
        return std::hash<std::string>{}(key.left) * 1'000'003u ^ std::hash<std::string>{}(key.right);
    }
};

} // namespace

struct ByteBpe::Impl {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, std::int32_t> token_to_id;
    std::unordered_map<PairKey, std::int32_t, PairHash> pair_rank;
    std::vector<std::pair<std::string, std::int32_t>> specials;  // content -> id, longest first
};

ByteBpe::ByteBpe(std::string_view tokenizer_json) : impl_(std::make_unique<Impl>()) {
    Json root;
    try {
        root = Json::parse(tokenizer_json);
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument(std::string("malformed tokenizer.json: ") + error.what());
    }
    if (!root.contains("model") || !root["model"].is_object()) {
        throw std::invalid_argument("tokenizer.json has no model");
    }
    const auto& model = root["model"];
    if (!model.contains("vocab") || !model["vocab"].is_object()) {
        throw std::invalid_argument("tokenizer.json has no model.vocab");
    }
    const auto& vocab = model["vocab"];
    for (auto entry = vocab.begin(); entry != vocab.end(); ++entry) {
        const auto id = entry.value().get<std::int32_t>();
        if (id < 0) { continue; }
        if (static_cast<std::size_t>(id) >= impl_->id_to_token.size()) {
            impl_->id_to_token.resize(static_cast<std::size_t>(id) + 1);
        }
        impl_->id_to_token[static_cast<std::size_t>(id)] = entry.key();
        impl_->token_to_id.emplace(entry.key(), id);
    }
    if (model.contains("merges") && model["merges"].is_array()) {
        std::int32_t rank = 0;
        for (const auto& merge : model["merges"]) {
            std::string left;
            std::string right;
            if (merge.is_array() && merge.size() == 2) {
                left = merge[0].get<std::string>();
                right = merge[1].get<std::string>();
            } else if (merge.is_string()) {
                const auto text = merge.get<std::string>();
                const auto split = text.find(' ');
                if (split == std::string::npos) { continue; }
                left = text.substr(0, split);
                right = text.substr(split + 1);
            } else {
                continue;
            }
            impl_->pair_rank.emplace(PairKey{std::move(left), std::move(right)}, rank++);
        }
    }
    if (root.contains("added_tokens") && root["added_tokens"].is_array()) {
        for (const auto& token : root["added_tokens"]) {
            if (!token.is_object() || !token.contains("content") || !token.contains("id")) {
                continue;
            }
            const auto content = token["content"].get<std::string>();
            const auto id = token["id"].get<std::int32_t>();
            if (content.empty()) { continue; }
            impl_->specials.emplace_back(content, id);
            if (static_cast<std::size_t>(id) >= impl_->id_to_token.size()) {
                impl_->id_to_token.resize(static_cast<std::size_t>(id) + 1);
            }
            impl_->id_to_token[static_cast<std::size_t>(id)] = content;
        }
        // Longest first, so `<|endoftext|>` wins over a prefix of it.
        std::stable_sort(impl_->specials.begin(), impl_->specials.end(),
                         [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    }
}

ByteBpe::~ByteBpe() = default;
ByteBpe::ByteBpe(const ByteBpe& other) : impl_(std::make_unique<Impl>(*other.impl_)) {}
ByteBpe& ByteBpe::operator=(const ByteBpe& other) {
    impl_ = std::make_unique<Impl>(*other.impl_);
    return *this;
}
ByteBpe::ByteBpe(ByteBpe&&) noexcept = default;
ByteBpe& ByteBpe::operator=(ByteBpe&&) noexcept = default;

std::size_t ByteBpe::vocab_size() const { return impl_->id_to_token.size(); }

std::vector<std::int32_t> ByteBpe::encode(std::string_view text) const {
    std::vector<std::int32_t> ids;
    std::size_t offset = 0;
    while (offset < text.size()) {
        // A special token wins wherever it starts, because the pipeline extracts them before the
        // split; when none matches here, the run up to the next one is tokenised normally.
        const std::pair<std::string, std::int32_t>* matched = nullptr;
        for (const auto& special : impl_->specials) {
            if (text.compare(offset, special.first.size(), special.first) == 0) {
                matched = &special;
                break;
            }
        }
        if (matched != nullptr) {
            ids.push_back(matched->second);
            offset += matched->first.size();
            continue;
        }
        // How far the ordinary run reaches: up to the next special token, or the end.
        std::size_t run_end = text.size();
        for (const auto& special : impl_->specials) {
            const auto found = text.find(special.first, offset);
            if (found != std::string_view::npos && found < run_end) { run_end = found; }
        }
        const auto run = text.substr(offset, run_end - offset);
        for (const auto chunk : llama3_split(run)) {
            const auto mapped = bytes_to_unicode(chunk);
            // Symbols start as single characters of the mapped alphabet, which is why the merges
            // are keyed by strings rather than by ids.
            std::vector<std::string> symbols;
            symbols.reserve(mapped.size());
            std::size_t at = 0;
            while (at < mapped.size()) {
                const auto lead = static_cast<unsigned char>(mapped[at]);
                const std::size_t width = lead < 0x80 ? 1 : (lead < 0xE0 ? 2 : 3);
                symbols.emplace_back(mapped.substr(at, width));
                at += width;
            }
            while (symbols.size() > 1) {
                std::int32_t best_rank = -1;
                std::size_t best_at = 0;
                for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
                    const auto found = impl_->pair_rank.find(PairKey{symbols[i], symbols[i + 1]});
                    if (found != impl_->pair_rank.end() &&
                        (best_rank < 0 || found->second < best_rank)) {
                        best_rank = found->second;
                        best_at = i;
                    }
                }
                if (best_rank < 0) { break; }
                symbols[best_at] += symbols[best_at + 1];
                symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_at) + 1);
            }
            for (const auto& symbol : symbols) {
                const auto found = impl_->token_to_id.find(symbol);
                if (found == impl_->token_to_id.end()) {
                    throw std::runtime_error("tokenizer.json has no id for a produced symbol");
                }
                ids.push_back(found->second);
            }
        }
        offset = run_end;
    }
    return ids;
}

std::string ByteBpe::decode(std::span<const std::int32_t> ids) const {
    std::string out;
    for (const auto id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= impl_->id_to_token.size()) { continue; }
        const auto& token = impl_->id_to_token[static_cast<std::size_t>(id)];
        out += unicode_to_bytes(token);
    }
    return out;
}

} // namespace ninfer::text
