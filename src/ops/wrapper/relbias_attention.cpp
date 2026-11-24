// ninfer::ops - relbias_attention wrapper: public validation and launcher dispatch.
#include "ninfer/ops/relbias_attention.h"

#include "ops/launcher/relbias_attention.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_bf16(const Tensor& t, const char* label) {
    if (t.dtype != DType::BF16) {
        throw std::invalid_argument(std::string("relbias_attention: ") + label + " must be BF16");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string("relbias_attention: ") + label +
                                    " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string("relbias_attention: ") + label +
                                    " data must be non-null");
    }
}

void require_shape(const Tensor& t, const char* label, std::int64_t expected) {
    if (t.numel() != expected) {
        throw std::invalid_argument(std::string("relbias_attention: ") + label +
                                    " has an unexpected number of elements");
    }
}

} // namespace

void relbias_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& rel,
                       const Tensor& proj, const Tensor& sink, const Tensor& gate,
                       const RelBiasAttentionArgs& args, Tensor& out, cudaStream_t stream) {
    if (args.heads <= 0 || args.kv_heads <= 0 || args.head_dim <= 0) {
        throw std::invalid_argument("relbias_attention: head geometry must be positive");
    }
    if (!(args.scale > 0.0F) || !std::isfinite(args.scale)) {
        throw std::invalid_argument("relbias_attention: scale must be positive and finite");
    }
    require_bf16(q, "q");
    require_bf16(k, "k");
    require_bf16(v, "v");
    require_bf16(rel, "rel");
    require_bf16(proj, "proj");
    require_bf16(sink, "sink");
    require_bf16(gate, "gate");
    require_bf16(out, "out");

    const auto heads = std::int64_t(args.heads);
    const auto kv_heads = std::int64_t(args.kv_heads);
    const auto head_dim = std::int64_t(args.head_dim);
    require_shape(q, "q", heads * head_dim);
    require_shape(k, "k", kv_heads * std::int64_t(args.kv_len) * head_dim);
    require_shape(v, "v", kv_heads * std::int64_t(args.kv_len) * head_dim);
    require_shape(rel, "rel", heads * std::int64_t(args.d_rel));
    require_shape(proj, "proj", std::int64_t(args.d_rel) * std::int64_t(args.rel_extent));
    require_shape(sink, "sink", heads);
    require_shape(gate, "gate", heads * head_dim);
    require_shape(out, "out", heads * head_dim);

    detail::relbias_attention_launch(q, k, v, rel, proj, sink, gate, args, out, stream);
}

} // namespace ninfer::ops
