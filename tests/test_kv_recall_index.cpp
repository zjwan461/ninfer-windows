#include "core/kv_recall_index.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

[[nodiscard]] double key_value(std::int32_t position, std::int32_t head, std::int32_t dim) {
    return static_cast<double>((position + 1) * 3) * 0.25 + static_cast<double>(head) * 100.0 +
           static_cast<double>(dim) * 10.0;
}

// Independent oracle: a block mean is the plain average of the block's columns, computed in double.
[[nodiscard]] double oracle_mean(std::int32_t block, std::int32_t block_tokens, std::int32_t head,
                                 std::int32_t dim) {
    double sum = 0.0;
    for (std::int32_t column = 0; column < block_tokens; ++column) {
        sum += key_value(block * block_tokens + column, head, dim);
    }
    return sum / static_cast<double>(block_tokens);
}

void test_straddling_chunks() {
    constexpr std::int32_t kHeadDim = 4;
    constexpr std::int32_t kHeads   = 2;
    constexpr std::int32_t kBlock   = 4;
    constexpr std::int32_t kColumns = 9;

    ninfer::KvRecallIndex index(kHeadDim, kHeads, kBlock);
    expect(index.complete_blocks() == 0 && index.next_position() == 0,
           "a fresh index holds no summary");

    std::int32_t position = 0;
    for (const std::int32_t chunk : {3, 3, 3}) {
        std::vector<float> keys(static_cast<std::size_t>(kHeadDim * kHeads * chunk));
        for (std::int32_t head = 0; head < kHeads; ++head) {
            for (std::int32_t dim = 0; dim < kHeadDim; ++dim) {
                for (std::int32_t column = 0; column < chunk; ++column) {
                    keys[static_cast<std::size_t>((head * kHeadDim + dim) * chunk + column)] =
                        static_cast<float>(key_value(position + column, head, dim));
                }
            }
        }
        index.capture(keys, chunk);
        position += chunk;
        expect(index.next_position() == static_cast<std::uint32_t>(position),
               "capture advances the expected position by the chunk width");
    }

    expect(index.captured_columns() == static_cast<std::uint32_t>(kColumns),
           "captured columns count every appended column");
    // Nine columns of four complete two blocks; the ninth column is a partial block and must not
    // contribute a summary of its own.
    expect(index.complete_blocks() == 2, "only whole blocks produce a summary");

    const std::vector<float>& means = index.block_means();
    expect(means.size() == static_cast<std::size_t>(kHeadDim * kHeads * index.complete_blocks()),
           "means cover every complete block and no partial one");
    for (std::int32_t block = 0; block < index.complete_blocks(); ++block) {
        for (std::int32_t head = 0; head < kHeads; ++head) {
            for (std::int32_t dim = 0; dim < kHeadDim; ++dim) {
                const double expected = oracle_mean(block, kBlock, head, dim);
                const double actual =
                    static_cast<double>(means[static_cast<std::size_t>((block * kHeads + head) * kHeadDim + dim)]);
                if (std::fabs(actual - expected) > 1e-5) {
                    expect(false, "block mean matches the column average");
                    std::cerr << "  block " << block << " head " << head << " dim " << dim << ": got "
                              << actual << " want " << expected << '\n';
                    return;
                }
            }
        }
    }
    expect(true, "block means match the column average");
}

void test_rejects_bad_input() {
    ninfer::KvRecallIndex index(2, 1, 2);
    bool threw = false;
    try {
        index.capture(std::vector<float>(3, 1.0F), 2);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "a chunk of the wrong size is rejected");

    threw = false;
    try {
        index.capture(std::vector<float>(4, 1.0F), 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "an empty chunk is rejected");

    threw = false;
    try {
        ninfer::KvRecallIndex bad(0, 1, 2);
        (void)bad;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "a zero head width is rejected");
}

void test_reset() {
    ninfer::KvRecallIndex index(2, 1, 2);
    const std::array<float, 4> keys{1.0F, 2.0F, 3.0F, 4.0F};
    index.capture(keys, 2);
    expect(index.complete_blocks() == 1 && !index.block_means().empty(),
           "a full chunk completes a block");
    index.reset();
    expect(index.complete_blocks() == 0 && index.next_position() == 0 && index.block_means().empty(),
           "reset drops every summary and the expected position");
}

} // namespace

int main() {
    try {
        test_straddling_chunks();
        test_rejects_bad_input();
        test_reset();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "kv_recall_index: all checks passed\n";
    return 0;
}