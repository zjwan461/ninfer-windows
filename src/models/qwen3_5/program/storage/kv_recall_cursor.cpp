// Implements: src/models/qwen3_5/program/storage/kv_recall_cursor.h
#include "models/qwen3_5/program/storage/kv_recall_cursor.h"

#include "ninfer/ops/kv_block_pool.h"

#include "core/device.h"

#include <algorithm>
#include <cmath>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

#include <cstdio>

namespace ninfer::models::qwen3_5::detail {

using core::KvRecallSink;

KvRecallCursor::KvRecallCursor(std::int32_t layers, std::int32_t head_dim, std::int32_t kv_heads,
                               std::int32_t block_tokens, std::int32_t block_capacity)
    : layers_count_(layers), head_dim_(head_dim), kv_heads_(kv_heads), block_tokens_(block_tokens),
      block_capacity_(block_capacity) {
    if (layers <= 0 || head_dim <= 0 || kv_heads <= 0 || block_tokens <= 0 ||
        block_capacity <= 0) {
        throw std::invalid_argument("kv_recall_cursor: geometry must be positive");
    }
    const std::size_t sum_bytes =
        static_cast<std::size_t>(head_dim) * kv_heads * block_capacity * sizeof(float);
    const std::size_t count_bytes = static_cast<std::size_t>(block_capacity) * sizeof(std::int32_t);
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        sums_.emplace_back(sum_bytes);
        counts_.emplace_back(count_bytes);
        means_.emplace_back(sum_bytes);
        sums_.back().fill(0);
        counts_.back().fill(0);
        means_.back().fill(0);
        block_means_.emplace_back(static_cast<std::size_t>(head_dim) * kv_heads * block_capacity,
                                  0.0F);
    }
    queries_.assign(static_cast<std::size_t>(layers) * head_dim * kv_heads, 0.0F);
    query_valid_.assign(static_cast<std::size_t>(layers), false);
}

KvRecallCursor::~KvRecallCursor() = default;

std::size_t KvRecallCursor::means_per_layer() const noexcept {
    return static_cast<std::size_t>(head_dim_) * kv_heads_ * block_capacity_;
}

std::int32_t KvRecallCursor::complete_blocks() const noexcept {
    const std::int32_t whole = captured_columns_ / block_tokens_;
    return whole < block_capacity_ ? whole : block_capacity_;
}

void KvRecallCursor::capture_layer(std::int32_t layer, const Tensor& pre_rope_keys,
                                   cudaStream_t stream) {
    if (layer < 0 || layer >= layers_count_) {
        throw std::out_of_range("kv_recall_cursor: layer " + std::to_string(layer) +
                                " is outside [0," + std::to_string(layers_count_) + ")");
    }
    if (pre_rope_keys.dtype != DType::BF16) {
        throw std::invalid_argument("kv_recall_cursor: keys must be bf16");
    }
    if (pre_rope_keys.ne[0] != head_dim_ || pre_rope_keys.ne[1] != kv_heads_ ||
        pre_rope_keys.ne[2] <= 0) {
        throw std::invalid_argument("kv_recall_cursor: keys must be [head_dim, kv_heads, columns]");
    }
    const std::int32_t columns = pre_rope_keys.ne[2];
    if (layers_seen_ == 0) {
        turn_columns_ = columns;
    } else if (columns != turn_columns_) {
        throw std::logic_error("kv_recall_cursor: layer " + std::to_string(layer) +
                               " was offered " + std::to_string(columns) + " columns, expected " +
                               std::to_string(turn_columns_));
    }

    const std::int32_t required =
        (captured_columns_ + turn_columns_ + block_tokens_ - 1) / block_tokens_;
    if (required > block_capacity_) {
        throw std::length_error("kv_recall_cursor: " + std::to_string(required) +
                                " blocks do not fit a capacity of " +
                                std::to_string(block_capacity_));
    }

    const std::size_t index = static_cast<std::size_t>(layer);
    Tensor sums(sums_[index].p, DType::FP32, {head_dim_, kv_heads_, block_capacity_});
    Tensor counts(counts_[index].p, DType::I32, {block_capacity_});
    ops::kv_block_accumulate(pre_rope_keys, block_tokens_, captured_columns_, sums, counts,
                             stream);
    finished_ = false;

    if (++layers_seen_ == layers_count_) {
        layers_seen_ = 0;
        captured_columns_ += turn_columns_;
    }
}

void KvRecallCursor::finish(cudaStream_t stream) {
    for (std::int32_t layer = 0; layer < layers_count_; ++layer) {
        const std::size_t index = static_cast<std::size_t>(layer);
        Tensor sums(sums_[index].p, DType::FP32, {head_dim_, kv_heads_, block_capacity_});
        Tensor counts(counts_[index].p, DType::I32, {block_capacity_});
        Tensor means(means_[index].p, DType::FP32, {head_dim_, kv_heads_, block_capacity_});
        ops::kv_block_finalize(sums, counts, means, stream);
        means_[index].copy_to_host(block_means_[index].data(), means_per_layer() * sizeof(float));
    }
    finished_ = true;
}

void KvRecallCursor::capture_query(std::int32_t layer, const Tensor& pre_rope_queries,
                                   cudaStream_t stream) {
    if (layer < 0 || layer >= layers_count_) {
        throw std::out_of_range("kv_recall_cursor: layer " + std::to_string(layer) +
                                " is outside [0," + std::to_string(layers_count_) + ")");
    }
    if (pre_rope_queries.dtype != DType::BF16) {
        throw std::invalid_argument("kv_recall_cursor: queries must be bf16");
    }
    if (pre_rope_queries.ne[0] != head_dim_ || pre_rope_queries.ne[2] <= 0) {
        throw std::invalid_argument("kv_recall_cursor: queries must be [head_dim, heads, columns]");
    }
    const std::int32_t query_heads = pre_rope_queries.ne[1];
    if (query_heads <= 0 || query_heads % kv_heads_ != 0) {
        throw std::invalid_argument("kv_recall_cursor: query heads must divide into the KV heads");
    }
    const std::int32_t group   = query_heads / kv_heads_;
    const std::int32_t columns = pre_rope_queries.ne[2];
    const std::int32_t pooled  = std::min(columns, kQueryTokens);
    const std::size_t row = static_cast<std::size_t>(head_dim_) * query_heads * sizeof(std::uint16_t);
    std::vector<std::uint16_t> host(static_cast<std::size_t>(head_dim_) * query_heads * pooled);
    const auto* base = static_cast<const std::uint8_t*>(pre_rope_queries.data) +
                       static_cast<std::size_t>(columns - pooled) * row;
    CUDA_CHECK(cudaMemcpy2DAsync(host.data(), row, base, row, row, pooled,
                                 cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto* values = reinterpret_cast<const __nv_bfloat16*>(host.data());
    // The question sits at the end of the prompt, so the query pool covers only its last columns:
    // averaging a whole 512-column tail mixes the ask into the haystack it is asking about.
    std::int32_t asked = pooled;
    if (const char* raw = std::getenv("NINFER_KV_RECALL_QUERY_TOKENS")) {
        const long parsed = std::strtol(raw, nullptr, 10);
        if (parsed > 0 && parsed < pooled) { asked = static_cast<std::int32_t>(parsed); }
    }
    const std::int32_t first_asked = pooled - asked;
    float* out         = queries_.data() + static_cast<std::size_t>(layer) * head_dim_ * kv_heads_;
    const float inverse = 1.0F / static_cast<float>(group * asked);
    for (std::int32_t kv = 0; kv < kv_heads_; ++kv) {
        for (std::int32_t d = 0; d < head_dim_; ++d) {
            float sum = 0.0F;
            for (std::int32_t r = 0; r < group; ++r) {
                const std::int32_t head = kv * group + r;
                for (std::int32_t c = first_asked; c < pooled; ++c) {
                    // The copied block is one contiguous run of [head_dim, query_heads] per column,
                    // so a column's head sits at (c * query_heads + head) * head_dim + d.
                    sum += __bfloat162float(
                        values[(static_cast<std::size_t>(c) * query_heads + head) * head_dim_ + d]);
                }
            }
            out[static_cast<std::size_t>(kv) * head_dim_ + d] = sum * inverse;
        }
    }
    query_valid_[static_cast<std::size_t>(layer)] = true;
}

const float* KvRecallCursor::query(std::int32_t layer) const {
    if (layer < 0 || layer >= layers_count_) {
        throw std::out_of_range("kv_recall_cursor: layer " + std::to_string(layer) +
                                " is outside [0," + std::to_string(layers_count_) + ")");
    }
    if (!query_valid_[static_cast<std::size_t>(layer)]) {
        throw std::logic_error("kv_recall_cursor: no query was captured for this layer");
    }
    return queries_.data() + static_cast<std::size_t>(layer) * head_dim_ * kv_heads_;
}

std::vector<float> KvRecallCursor::block_scores() const {
    if (!finished_) {
        throw std::logic_error("kv_recall_cursor: block_scores needs finish() first");
    }
    for (std::int32_t layer = 0; layer < layers_count_; ++layer) {
        if (!query_valid_[static_cast<std::size_t>(layer)]) {
            throw std::logic_error("kv_recall_cursor: a layer is missing its query");
        }
    }
    const std::int32_t blocks = complete_blocks();
    std::vector<float> scores(static_cast<std::size_t>(blocks), 0.0F);
    if (blocks == 0) { return scores; }
    const float inverse = 1.0F / static_cast<float>(layers_count_ * kv_heads_);
    for (std::int32_t layer = 0; layer < layers_count_; ++layer) {
        const std::vector<float>& means = block_means_[static_cast<std::size_t>(layer)];
        const float* q = queries_.data() + static_cast<std::size_t>(layer) * head_dim_ * kv_heads_;
        for (std::int32_t kv = 0; kv < kv_heads_; ++kv) {
            const float* qh = q + static_cast<std::size_t>(kv) * head_dim_;
            double q_norm   = 0.0;
            for (std::int32_t d = 0; d < head_dim_; ++d) {
                q_norm += static_cast<double>(qh[d]) * static_cast<double>(qh[d]);
            }
            q_norm = std::sqrt(q_norm);
            for (std::int32_t block = 0; block < blocks; ++block) {
                const float* mh = means.data() + static_cast<std::size_t>(kv) * head_dim_ +
                                  static_cast<std::size_t>(head_dim_) * kv_heads_ * block;
                double dot = 0.0;
                double m_norm = 0.0;
                for (std::int32_t d = 0; d < head_dim_; ++d) {
                    dot += static_cast<double>(qh[d]) * static_cast<double>(mh[d]);
                    m_norm += static_cast<double>(mh[d]) * static_cast<double>(mh[d]);
                }
                m_norm = std::sqrt(m_norm);
                const double denom = q_norm * m_norm;
                if (denom > 0.0) {
                    scores[static_cast<std::size_t>(block)] +=
                        static_cast<float>(dot / denom) * inverse;
                }
            }
        }
    }
    return scores;
}

void KvRecallCursor::end_of_prefill(cudaStream_t stream) {
    finish(stream);
    std::fprintf(stderr,
                 "kv recall: captured %d columns over %d layers -> %d complete blocks of %d\n",
                 captured_columns_, layers_count_, complete_blocks(), block_tokens_);
    std::fflush(stderr);
}

const std::vector<float>& KvRecallCursor::block_means(std::int32_t layer) const {
    if (layer < 0 || layer >= layers_count_) {
        throw std::out_of_range("kv_recall_cursor: layer " + std::to_string(layer) +
                                " is outside [0," + std::to_string(layers_count_) + ")");
    }
    if (!finished_) {
        throw std::logic_error("kv_recall_cursor: finish() has not run since the last capture");
    }
    return block_means_[static_cast<std::size_t>(layer)];
}

void KvRecallCursor::reset() {
    for (auto& sum : sums_) { sum.fill(0); }
    for (auto& count : counts_) { count.fill(0); }
    for (auto& mean : means_) { mean.fill(0); }
    for (auto& stored : block_means_) { stored.assign(stored.size(), 0.0F); }
    std::fill(queries_.begin(), queries_.end(), 0.0F);
    std::fill(query_valid_.begin(), query_valid_.end(), false);
    captured_columns_ = 0;
    turn_columns_     = 0;
    layers_seen_      = 0;
    finished_         = false;
}

} // namespace ninfer::models::qwen3_5::detail