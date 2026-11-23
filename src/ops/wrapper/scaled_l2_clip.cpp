// ninfer::ops - scaled_l2_clip wrapper: public validation and launcher dispatch.
#include "ninfer/ops/scaled_l2_clip.h"

#include "ops/launcher/scaled_l2_clip.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_bf16(const Tensor& t, const char* label) {
    if (t.dtype != DType::BF16) {
        throw std::invalid_argument(std::string("scaled_l2_clip: ") + label + " must be BF16");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string("scaled_l2_clip: ") + label + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string("scaled_l2_clip: ") + label +
                                    " data must be non-null");
    }
}

} // namespace

void scaled_l2_clip(const Tensor& x, const Tensor& scale, const Tensor* positions,
                    const ScaledL2ClipArgs& args, Tensor& out, cudaStream_t stream) {
    if (args.rows < 0 || args.heads <= 0 || args.head_dim <= 0) {
        throw std::invalid_argument("scaled_l2_clip: invalid geometry");
    }
    if (!(args.threshold > 0.0F) || !std::isfinite(args.threshold)) {
        throw std::invalid_argument("scaled_l2_clip: threshold must be positive and finite");
    }
    if (args.tau_alpha < 0.0F || !std::isfinite(args.tau_alpha)) {
        throw std::invalid_argument("scaled_l2_clip: tau_alpha must be nonnegative and finite");
    }
    if (args.tau_alpha > 0.0F && (!(args.tau_floor > 0.0F) || !std::isfinite(args.tau_floor))) {
        throw std::invalid_argument("scaled_l2_clip: tau_floor must be positive when tau is used");
    }
    if (args.tau_alpha > 0.0F && positions == nullptr) {
        throw std::invalid_argument("scaled_l2_clip: positions are required when tau is used");
    }
    require_bf16(x, "x");
    require_bf16(scale, "scale");
    require_bf16(out, "out");

    const auto rows = std::int64_t(args.rows);
    const auto heads = std::int64_t(args.heads);
    const auto head_dim = std::int64_t(args.head_dim);
    if (x.numel() != rows * heads * head_dim || out.numel() != rows * heads * head_dim) {
        throw std::invalid_argument("scaled_l2_clip: x/out must be [rows, heads * head_dim]");
    }
    if (scale.numel() != heads * head_dim) {
        throw std::invalid_argument("scaled_l2_clip: scale must be [heads, head_dim]");
    }
    if (positions != nullptr && positions->numel() != rows) {
        throw std::invalid_argument("scaled_l2_clip: positions must have one entry per row");
    }
    if (rows == 0) { return; }

    detail::scaled_l2_clip_launch(x, scale, positions, args, out, stream);
}

} // namespace ninfer::ops
