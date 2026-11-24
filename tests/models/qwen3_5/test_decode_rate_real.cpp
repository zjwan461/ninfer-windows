// Decode rate of a real artifact through the engine, which is the number a user feels.
//
//   NINFER_TEST_ARTIFACT=/path/to/model.ninfer ./ninfer_qwen3_5_decode_rate_real_test

#include "ninfer/engine.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
ninfer::RequestOptions request(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}
} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    const auto outputs = argc > 1 ? static_cast<std::uint32_t>(std::stoul(argv[1])) : 64U;
    try {
        ninfer::EngineOptions options;
        options.artifact_path   = artifact;
        options.max_context     = 4096;
        options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(4096);
        options.prefill_chunk   = 4096;
        options.max_concurrency = 1;
        options.use_cuda_graph  = true;
        options.kv_cache        = ninfer::KvCacheStorage::BFloat16;
        ninfer::Engine engine(options);

        const auto prompt = engine.tokenize_text("Count from one to twenty: one, two, three,");
        // One warm-up generation, then the timed one: the first call pays for module load and the
        // first touch of every weight, which is a second and not part of a rate.
        (void)engine.generate(engine.prepare_tokens(prompt), request(4));

        const auto begin = std::chrono::steady_clock::now();
        const auto result = engine.generate(engine.prepare_tokens(prompt), request(outputs));
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        const auto generated = result.generated_token_ids.size();
        std::cout << generated << " tokens in " << seconds << " s -> "
                  << (double(generated) / seconds) << " tok/s\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
