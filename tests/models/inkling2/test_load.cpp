// Loads a real Inkling 2 Mini artifact and checks the loader against it.
//
//   ninfer_inkling2_load_real_test --artifact <path/to/inkling2-mini.ninfer>
//
// Without --artifact the test reports SKIP (return 77), like the Qwen real-artifact test.

#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/reader.h"
#include "models/inkling2/load.h"
#include "models/inkling2/materialize.h"
#include "models/inkling2/parameters.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <iostream>
#include <map>
#include <string>

namespace {

using namespace ninfer;
using ninfer::test::artifact_fixture::require;

std::string value_of(int argc, char** argv, int& index, const std::string& flag) {
    if (++index >= argc) { throw std::invalid_argument("missing value for " + flag); }
    return argv[index];
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path path;
        bool materialize = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--artifact") {
                path = value_of(argc, argv, i, arg);
            } else if (arg == "--materialize") {
                materialize = true;
            } else {
                throw std::invalid_argument("unknown argument " + arg);
            }
        }
        if (path.empty()) {
            std::cout << "SKIP: supply --artifact <inkling2-mini.ninfer>\n";
            return 77;
        }

        artifact::Reader reader(path);
        const auto& directory = reader.directory();
        const auto config = models::inkling2::parse_config(directory);

        require(config.visits() == 48, "the loop must produce 48 visits");
        require(config.kv_caches() == 48, "48 KV caches must follow from 48 visits");
        require(config.global_layers() == 4, "four global attention layers");

        models::inkling2::validate_parameters(directory, config);
        const auto bound = models::inkling2::bind_parameters(directory, config);
        require(bound.size() == 6790, "text slot count differs from the artifact");

        std::map<std::string, std::size_t> by_format;
        for (const auto& slot : bound) { by_format[slot.format]++; }
        require(by_format["nvfp4"] == 5120, "NVFP4 expert count differs");
        require(by_format["q8_g32_fp16"] == 1024, "q8 expert count differs");
        require(by_format["bf16"] == 572, "bf16 slot count differs");
        require(by_format["fp32"] == 74, "fp32 slot count differs");

        const auto index = models::inkling2::build_index(bound, config);
        std::size_t assigned = 0;
        for (const auto& layer : index.layers) {
            for (auto slot : layer.role) { if (slot >= 0) { ++assigned; } }
            for (auto slot : layer.expert_w13) { if (slot >= 0) { ++assigned; } }
            for (auto slot : layer.expert_w2) { if (slot >= 0) { ++assigned; } }
            for (auto slot : layer.controller_down) { if (slot >= 0) { ++assigned; } }
            for (auto slot : layer.controller_forget) { if (slot >= 0) { ++assigned; } }
            for (auto slot : layer.controller_gain) { if (slot >= 0) { ++assigned; } }
        }
        for (auto slot : index.attn_queries) { if (slot >= 0) { ++assigned; } }
        for (auto slot : index.mlp_queries) { if (slot >= 0) { ++assigned; } }
        require(assigned + 7 == bound.size(), "every slot must take exactly one role");
        require(index.layers[0].has(models::inkling2::Role::AttnNorm),
                "a sliding-window layer normalises the attention input once");
        require(!index.layers[5].has(models::inkling2::Role::AttnNorm),
                "a global layer has no attn_norm");
        require(index.layers[5].has(models::inkling2::Role::QInputNorm),
                "a global layer carries q/k/v input norms");
        require(index.layers[5].controller_gain[3] < 0,
                "controllers exist only inside the loop core");

        require(models::inkling2::weight_type("nvfp4") == QType::NVFP4, "nvfp4 maps to NVFP4");
        require(models::inkling2::weight_type("q8_g32_fp16") == QType::Q8_G32_FP16,
                "q8 maps to Q8_G32_FP16");
        bool unmapped = false;
        try {
            (void)models::inkling2::weight_type("q4_g64_fp16");
        } catch (const artifact::ArtifactError&) {
            unmapped = true;
        }
        require(unmapped, "a format this architecture does not execute must be rejected");

        // A config that disagrees with the architecture must be rejected rather than loaded.
        auto broken = directory;
        broken.components.at("text").config["loop_config"]["num_passes"] = 2;
        bool rejected = false;
        try {
            (void)models::inkling2::parse_config(broken);
        } catch (const artifact::ArtifactError&) {
            rejected = true;
        }
        require(rejected, "a two-pass config must be rejected");
        if (materialize) {
            // Allocates the whole text model on the device; opt-in because a 16 GB card cannot
            // hold it and the default test run must stay device-light.
            std::uint64_t expected = 0;
            for (const auto& slot : bound) {
                expected += directory.tensor(slot.handle).bytes;
            }
            DeviceContext device;
            auto weights = models::inkling2::materialize_text(reader, config, device);
            require(weights.views.size() == bound.size(), "materialised slot count differs");
            require(weights.device_bytes == expected, "H2D did not cover every slot byte");
            require(weights.index.layers.size() == config.num_hidden_layers,
                    "role index does not follow the config");
            std::size_t matrices = 0;
            std::size_t dense = 0;
            for (std::size_t position = 0; position < bound.size(); ++position) {
                if (bound[position].shape.size() == 2) {
                    const auto weight =
                        models::inkling2::matrix_weight(weights, std::int32_t(position));
                    require(weight.shape[0] > 0 && weight.shape[1] > 0,
                            "a matrix weight came back without shape");
                    ++matrices;
                } else {
                    const auto tensor =
                        models::inkling2::dense_tensor(weights, std::int32_t(position));
                    require(tensor.numel() > 0, "a dense slot came back empty");
                    ++dense;
                }
            }
            std::cout << "materialised " << weights.device_bytes / (1024.0 * 1024.0 * 1024.0)
                      << " GiB, matrices " << matrices << ", dense " << dense << ", reserved "
                      << weights.reserved_bytes / (1024.0 * 1024.0 * 1024.0) << " GiB\n";
        }
        std::cout << "ok: 6790 slots, 48 visits, config validated\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
