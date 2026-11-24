#pragma once

#include <atomic>

namespace ninfer::ops {

// How many of each kind of operation a decode step issues. Wall time on a shared machine is not
// usable for judging a change, but a count is: it does not depend on the load, and it is the number
// that decides how much launch overhead a step pays.
struct OpCounters {
    std::atomic<long> linear{0};
    std::atomic<long> expert_gemv{0};
    std::atomic<long> accumulate{0};
};

inline OpCounters& op_counters() {
    static OpCounters counters;
    return counters;
}

} // namespace ninfer::ops
