// Host-only: the 48-visit schedule is fixed by the config, so it can be checked without a device.

#include "artifact/fixture.h"
#include "models/inkling2/config.h"
#include "models/inkling2/schedule.h"

#include <array>
#include <iostream>
#include <string>

namespace {

using ninfer::test::artifact_fixture::require;

// The text component of the published checkpoint, verbatim.
constexpr char kTextConfig[] = R"TEXT(
{"architectures":["Inkling2LoopForConditionalGeneration"],"attn_res_normalize_values":true,"d_rel":16,"final_logit_softcapping":30.0,"head_dim":128,"hidden_size":2304,"intermediate_size":1152,"local_layer_ids":[0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],"log_scaling_alpha":0.1,"log_scaling_n_floor":8192,"loop_config":{"core_end":18,"core_start":6,"gate_rank":64,"is_varloop":false,"num_passes":3},"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,"model_max_length":1048576,"model_type":"inkling2_loop_text","n_layers_per_attn_res_block":6,"n_routed_experts":128,"n_shared_experts":0,"num_attention_heads":24,"num_experts_per_tok":8,"num_hidden_layers":24,"num_key_value_heads":8,"o_bias":false,"q_bias":false,"qk_l2_clip_threshold":22.627416997969522,"rel_extent":1024,"rel_l2_clip_threshold":8.0,"rms_norm_eps":1e-06,"route_scale":8.0,"sconv_kernel_size":4,"sliding_window_size":1024,"swa_head_dim":128,"swa_num_attention_heads":24,"swa_num_key_value_heads":8,"torch_dtype":"bfloat16","unpadded_vocab_size":200058,"use_gate_bias":true,"use_value_norm_no_gamma":true,"vocab_size":201024}
)TEXT";

} // namespace

int main() {
    try {
        auto root = ninfer::artifact::parse_json(kTextConfig, "text config");
        ninfer::artifact::Directory directory;
        ninfer::artifact::Component text;
        text.target = "text";
        text.config = root;
        directory.components["text"] = text;

        const auto config = ninfer::models::inkling2::parse_config(directory);
        const auto schedule = ninfer::models::inkling2::build_schedule(config);
        require(schedule.visits.size() == 48, "schedule must have 48 visits");

        // The visit order the reference implementation produced for one forward pass.
        constexpr std::array<std::uint32_t, 48> kOracleOrder = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23};
        for (std::size_t i = 0; i < kOracleOrder.size(); ++i) {
            require(schedule.visits[i].layer == kOracleOrder[i], "visit order differs from the reference");
            require(schedule.visits[i].index == i, "visit index must be its position");
        }

        // Every pass starts from the prelude: only core visits after the first pass read a message.
        require(schedule.message_visits.size() == 24, "12 core layers x 2 later passes read a message");
        require(schedule.visits[6].message_from == -1, "the first core pass reads nothing");
        require(schedule.visits[18].message_from == 6, "pass two of layer 6 reads visit 6");
        require(schedule.visits[41].message_from == 29, "pass three of layer 17 reads visit 29");
        require(schedule.visits[47].message_from == -1, "the coda reads nothing");
        for (const auto index : schedule.message_visits) {
            const auto& visit = schedule.visits[index];
            const auto& source = schedule.visits[std::size_t(visit.message_from)];
            require(source.layer == visit.layer, "a message must come from the same layer");
            require(source.pass + 1 == visit.pass, "a message must come from the pass before");
        }

        // Sliding-window layers are exactly local_layer_ids.
        for (const auto& visit : schedule.visits) {
            require(visit.sliding == config.sliding_layer[visit.layer], "sliding flag differs");
        }
        require(config.sliding_layer[5] == false, "layer 5 is global");
        require(config.sliding_layer[6] == true, "layer 6 is sliding");

        std::cout << "ok: 48 visits, 24 message reads, order matches the reference\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
