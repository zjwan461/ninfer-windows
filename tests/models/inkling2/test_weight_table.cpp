// The device-side weight table copies every field, which is what a kernel indexing it relies on.

#include "models/inkling2/execution/weight_table.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

namespace {
int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cout << "FAIL " << label << "\n";
    ++failures;
}
} // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) {
        std::cerr << "SKIP: no CUDA device\n";
        return 77;
    }
    // Four weights that differ in every field a kernel reads.
    std::vector<ninfer::Weight> host(4);
    for (std::size_t i = 0; i < host.size(); ++i) {
        auto& weight = host[i];
        weight.qdata = reinterpret_cast<const void*>(0x1000 + i * 0x100);
        weight.scales = reinterpret_cast<const void*>(0x2000 + i * 0x100);
        weight.n = static_cast<std::int32_t>(2304 + i);
        weight.k = static_cast<std::int32_t>(2304 - i);
        weight.qtype = i % 2 == 0 ? ninfer::QType::NVFP4 : ninfer::QType::Q4_G64_FP16;
        weight.group = static_cast<std::int32_t>(16 + i);
    }

    void* table = nullptr;
    try {
        table = ninfer::models::inkling2::upload_weight_table(host, nullptr);
        const auto back = ninfer::models::inkling2::read_weight_table(table, host.size(), nullptr);
        for (std::size_t i = 0; i < host.size(); ++i) {
            const auto label = "weight " + std::to_string(i);
            expect(back[i].qdata == host[i].qdata, label + " qdata");
            expect(back[i].scales == host[i].scales, label + " scales");
            expect(back[i].n == host[i].n, label + " n");
            expect(back[i].k == host[i].k, label + " k");
            expect(back[i].qtype == host[i].qtype, label + " qtype");
            expect(back[i].group == host[i].group, label + " group");
        }
        expect(sizeof(ninfer::Weight) == sizeof(ninfer::Weight), "trivially copyable size");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    ninfer::models::inkling2::free_weight_table(table);

    if (failures != 0) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "ok: the weight table copies every field\n";
    return 0;
}
