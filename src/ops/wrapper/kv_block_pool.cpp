// ninfer::ops - KV block pool wrappers: public api validation and launcher dispatch.
#include "ninfer/ops/kv_block_pool.h"

#include "ops/launcher/kv_block_pool.h" // detail::kv_block_*_launch

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_contiguous(const Tensor& tensor, const char* label, DType dtype) {
    if (tensor.dtype != dtype) {
        throw std::invalid_argument(std::string("kv_block_pool: ") + label + " has the wrong dtype");
    }
    for (int d = 0; d < 4; ++d) {
        if (tensor.ne[d] < 0) {
            throw std::invalid_argument(std::string("kv_block_pool: ") + label +
                                        " dimensions must be nonnegative");
        }
    }
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("kv_block_pool: ") + label + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("kv_block_pool: ") + label +
                                    " data must be non-null");
    }
}

} // namespace

void kv_block_mean_keys(const Tensor& keys, std::int32_t block_tokens, Tensor& means,
                        cudaStream_t stream) {
    require_contiguous(keys, "keys", DType::BF16);
    require_contiguous(means, "means", DType::FP32);

    if (keys.ne[3] != 1) {
        throw std::invalid_argument("kv_block_pool: keys must be rank-3 [D,H,T]");
    }
    if (means.ne[3] != 1) {
        throw std::invalid_argument("kv_block_pool: means must be rank-3 [D,H,B]");
    }
    if (block_tokens <= 0) {
        throw std::invalid_argument("kv_block_pool: block_tokens must be positive");
    }
    if (keys.ne[0] != means.ne[0] || keys.ne[1] != means.ne[1]) {
        throw std::invalid_argument("kv_block_pool: means must share keys' D and H");
    }

    const std::int32_t tokens = keys.ne[2];
    const std::int32_t blocks = tokens > 0 ? (tokens + block_tokens - 1) / block_tokens : 0;
    if (means.ne[2] != blocks) {
        throw std::invalid_argument("kv_block_pool: means must be [D,H,ceil(T/block_tokens)]");
    }
    if (blocks == 0) { return; }

    detail::kv_block_mean_keys_launch(keys, block_tokens, means, stream);
}

void kv_block_scores(const Tensor& query, const Tensor& means, Tensor& scores,
                     cudaStream_t stream) {
    require_contiguous(query, "query", DType::BF16);
    require_contiguous(means, "means", DType::FP32);
    require_contiguous(scores, "scores", DType::FP32);

    if (query.ne[2] != 1 || query.ne[3] != 1) {
        throw std::invalid_argument("kv_block_pool: query must be rank-2 [D,Q]");
    }
    if (means.ne[3] != 1) {
        throw std::invalid_argument("kv_block_pool: means must be rank-3 [D,H,B]");
    }
    if (scores.ne[1] != 1 || scores.ne[2] != 1 || scores.ne[3] != 1) {
        throw std::invalid_argument("kv_block_pool: scores must be rank-1 [B]");
    }
    if (query.ne[0] != means.ne[0]) {
        throw std::invalid_argument("kv_block_pool: query and means must share D");
    }
    if (query.ne[1] <= 0 || means.ne[1] <= 0) {
        throw std::invalid_argument("kv_block_pool: query and means head counts must be positive");
    }
    if (query.ne[1] % means.ne[1] != 0) {
        throw std::invalid_argument(
            "kv_block_pool: query head count must be a multiple of the KV head count");
    }
    if (scores.ne[0] != means.ne[2]) {
        throw std::invalid_argument("kv_block_pool: scores must have one entry per block");
    }
    if (means.ne[2] == 0) { return; }

    detail::kv_block_scores_launch(query, means, scores, stream);
}

} // namespace ninfer::ops