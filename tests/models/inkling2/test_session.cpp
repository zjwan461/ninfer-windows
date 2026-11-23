// The decode session against the recorded reference: one token in, the model's logits out.

#include "artifact/reader.h"
#include "core/device.h"
#include "models/inkling2/config.h"
#include "models/inkling2/execution/session.h"
#include "models/inkling2/materialize.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {
using namespace ninfer;

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open " + path.string()); }
    in.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<float> out(size / sizeof(float));
    in.read(reinterpret_cast<char*>(out.data()), std::streamsize(size));
    return out;
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path artifact_path;
        std::filesystem::path oracle;
        std::uint32_t token = 1234;
        std::filesystem::path reference_stages;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--artifact" && i + 1 < argc) { artifact_path = argv[++i]; }
            else if (arg == "--oracle" && i + 1 < argc) { oracle = argv[++i]; }
            else if (arg == "--token" && i + 1 < argc) { token = static_cast<std::uint32_t>(std::stoul(argv[++i])); }
            else if (arg == "--stages" && i + 1 < argc) { reference_stages = argv[++i]; }
            else { throw std::invalid_argument("unknown argument " + arg); }
        }
        if (artifact_path.empty() || oracle.empty()) {
            std::cout << "SKIP: supply --artifact and --oracle\n";
            return 77;
        }

        artifact::Reader reader(artifact_path);
        const auto config = models::inkling2::parse_config(reader.directory());
        DeviceContext device;
        auto weights = models::inkling2::materialize_text(reader, config, device);

        models::inkling2::execution::SessionOptions options;
        options.max_context = 8;
        models::inkling2::execution::Session session(weights, config, device, options);
        session.reset();
        // Compare the first visit stage by stage against the model's own dump: a name mismatch
        // here is much easier to place than a wrong logit vector.
        int visits_seen = 0;
        int comparisons = 0;
        int missing = 0;
        double worst_stage = 0.0;
        int worst_layer = -1;
        // The sink outlives this block: the step that drives it happens later, so everything it
        // reads has to be declared out here.
        std::vector<int> seen_per_layer(24, 0);
        std::vector<double> layer_worst(24, 0.0);
        int routing_flips = 0;
        std::vector<std::string> layer_worst_stage(24);
        if (!reference_stages.empty()) {
            session.stage_sink = [&](int layer, const char* name, void* buffer, int width) {
                const int visit = layer;
                const int occurrence = seen_per_layer[std::size_t(layer)] / 12;
                if (std::string(name) == "out") { ++seen_per_layer[std::size_t(layer)]; }
                const std::string stage(name);
                const std::string file = "layer" + std::to_string(visit) + "." ;
                (void)file;
                if (stage == "final_states") {
                    std::vector<std::uint16_t> host(static_cast<std::size_t>(width));
                    cudaMemcpy(host.data(), buffer, host.size() * 2, cudaMemcpyDeviceToHost);
                    std::ofstream out("/tmp/session_final_states.bin", std::ios::binary);
                    out.write(reinterpret_cast<const char*>(host.data()),
                              std::streamsize(host.size() * 2));
                    return;
                }
                if (stage == "final" || stage == "final_read") {
                    std::vector<std::uint16_t> host(static_cast<std::size_t>(width));
                    cudaMemcpy(host.data(), buffer, host.size() * 2, cudaMemcpyDeviceToHost);
                    std::ofstream out("/tmp/session_" + stage + ".bin", std::ios::binary);
                    out.write(reinterpret_cast<const char*>(host.data()),
                              std::streamsize(host.size() * 2));
                    return;
                }
                std::string mapped = stage;
                if (stage == "ln") { mapped = "ln"; }
                if (stage == "attnpre") { mapped = "attn_raw"; }
                if (stage == "mlp_in") { mapped = "mlp_normed"; }
                if (stage == "mlp_pre") { mapped = "moe"; }
                if (stage == "mlp") { mapped = "mlp_conv"; }
                if (stage == "attn") { mapped = "attn_conv"; }
                if (stage == "q") { mapped = "q_proj"; }
                if (stage == "k") { mapped = "k_proj"; }
                if (stage == "v") { mapped = "v_proj"; }
                if (stage == "sink") { mapped = "sink_proj"; }
                const auto candidate = reference_stages /
                    ("layer" + std::to_string(visit) + "." + mapped + "_0" +
                     std::to_string(occurrence) + ".bin");
                if (!std::filesystem::exists(candidate)) {
                    if (missing < 3) { std::cout << "  missing " << candidate.string() << "\n"; }
                    ++missing;
                    return;
                }
                const auto expected = read_f32(candidate);
                if (expected.size() != static_cast<std::size_t>(width)) { return; }
                std::vector<std::uint16_t> host(static_cast<std::size_t>(width));
                cudaMemcpy(host.data(), buffer, host.size() * 2, cudaMemcpyDeviceToHost);
                double worst = 0.0;
                for (std::size_t i = 0; i < host.size(); ++i) {
                    const std::uint32_t bits = std::uint32_t(host[i]) << 16;
                    float value = 0.0F;
                    std::memcpy(&value, &bits, 4);
                    worst = std::max(worst, double(std::fabs(value - expected[i])));
                }
                ++comparisons;
                if (worst > worst_stage) { worst_stage = worst; worst_layer = visit; }
                if (worst > layer_worst[std::size_t(visit)]) {
                    layer_worst[std::size_t(visit)] = worst;
                    layer_worst_stage[std::size_t(visit)] = stage;
                }
            };
        }
        if (!reference_stages.empty()) {
            session.routing_hook = [&](int visit, std::vector<std::int32_t>& indices,
                                       std::vector<float>& weights) {
                const auto path = oracle / ("route_" + std::to_string(visit) + ".txt");
                if (!std::filesystem::exists(path)) { return; }
                std::ifstream in(path);
                std::vector<std::int32_t> ref_indices(8);
                std::vector<float> ref_weights(8);
                for (int j = 0; j < 8; ++j) { in >> ref_indices[std::size_t(j)]; }
                for (int j = 0; j < 8; ++j) { in >> ref_weights[std::size_t(j)]; }
                bool differs = false;
                for (int j = 0; j < 8; ++j) {
                    if (ref_indices[std::size_t(j)] != indices[std::size_t(j)]) { differs = true; }
                }
                if (differs) { ++routing_flips; }
                indices = ref_indices;
                weights = ref_weights;
            };
        }
        // One step runs every visit; the sink counts layers by the reading of the schedule.
        const auto logits = session.step(token, nullptr);
        (void)visits_seen;
        if (!reference_stages.empty()) {
            std::cout << "routing flips " << routing_flips << " of 48\n";
            std::cout << "stage comparisons " << comparisons << " (missing references " << missing
                      << "), worst " << worst_stage << " at layer " << worst_layer << "\n";
            for (std::size_t layer = 0; layer < layer_worst.size(); ++layer) {
                if (layer_worst[layer] > 0.5) {
                    std::cout << "  layer " << layer << " worst " << layer_worst_stage[layer]
                              << " " << layer_worst[layer] << "\n";
                }
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::cerr << "FAIL: " << cudaGetErrorString(cudaGetLastError()) << "\n";
            return 1;
        }
        // ne[0] is the row width: the vocabulary.
        std::vector<std::uint16_t> host(static_cast<std::size_t>(logits.ne[0]));
        cudaMemcpy(host.data(), logits.data, host.size() * 2, cudaMemcpyDeviceToHost);
        auto to_float = [](std::uint16_t value) {
            const std::uint32_t bits = std::uint32_t(value) << 16;
            float out = 0.0F;
            std::memcpy(&out, &bits, 4);
            return out;
        };
        std::vector<float> mine(host.size());
        for (std::size_t i = 0; i < host.size(); ++i) { mine[i] = to_float(host[i]); }

        const auto reference = read_f32(oracle / "logits.bin");
        const auto compared = std::min(mine.size(), reference.size());
        double worst = 0.0;
        for (std::size_t i = 0; i < compared; ++i) {
            worst = std::max(worst, double(std::fabs(mine[i] - reference[i])));
        }
        std::vector<int> order(compared);
        std::iota(order.begin(), order.end(), 0);
        const auto top = [&](const std::vector<float>& values) {
            auto indices = order;
            std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(),
                              [&](int a, int b) { return values[std::size_t(a)] > values[std::size_t(b)]; });
            return std::vector<int>(indices.begin(), indices.begin() + 5);
        };
        const auto my_top = top(mine);
        const auto ref_top = top(reference);
        std::size_t shared = 0;
        for (int a : my_top) { for (int b : ref_top) { if (a == b) { ++shared; } } }
        std::cout << "logits max|err| " << worst << " (of " << compared << ")\n";
        std::cout << "top5 mine:";
        for (int id : my_top) { std::cout << " " << id; }
        std::cout << "\ntop5 ref :";
        for (int id : ref_top) { std::cout << " " << id; }
        std::cout << "\nshared " << shared << "/5\n";
        if (worst > 0.5 || shared < 5) {
            std::cerr << "FAIL: the session does not reproduce the reference logits\n";
            return 1;
        }
        std::cout << "ok: the session reproduces the reference logits\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
