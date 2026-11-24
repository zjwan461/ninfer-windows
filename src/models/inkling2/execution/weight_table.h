#pragma once

#include "core/weight.h"

#include <cstddef>
#include <span>
#include <vector>

#include <cuda_runtime.h>

namespace ninfer::models::inkling2 {

// Copies a set of weights into one device-side array, so a kernel can index it with a slot number
// instead of the host choosing a handle. `Weight` is a flat POD, which is what makes this possible
// at all: 256 experts times two matrices is a few tens of kilobytes.
void* upload_weight_table(std::span<const Weight> weights, cudaStream_t stream);

// Reads a table back. Only the tests use it, to prove the copy is field-for-field.
std::vector<Weight> read_weight_table(const void* table, std::size_t count, cudaStream_t stream);

void free_weight_table(void* table);

} // namespace ninfer::models::inkling2
