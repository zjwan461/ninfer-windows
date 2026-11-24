// ninfer::models::inkling2 - device-side weight tables for on-device expert selection.
#include "models/inkling2/execution/weight_table.h"

#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2 {
namespace {
void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling2 weight table: ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}
} // namespace

void* upload_weight_table(std::span<const Weight> weights, cudaStream_t stream) {
    if (weights.empty()) { return nullptr; }
    void* table = nullptr;
    check(cudaMalloc(&table, weights.size_bytes()), "cudaMalloc");
    check(cudaMemcpyAsync(table, weights.data(), weights.size_bytes(), cudaMemcpyHostToDevice,
                          stream), "cudaMemcpyAsync");
    return table;
}

std::vector<Weight> read_weight_table(const void* table, std::size_t count, cudaStream_t stream) {
    std::vector<Weight> host(count);
    if (count == 0) { return host; }
    check(cudaMemcpyAsync(host.data(), table, count * sizeof(Weight), cudaMemcpyDeviceToHost, stream),
          "cudaMemcpyAsync");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    return host;
}

void free_weight_table(void* table) {
    if (table != nullptr) { cudaFree(table); }
}

} // namespace ninfer::models::inkling2
