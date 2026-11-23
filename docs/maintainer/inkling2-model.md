# Inkling 2 Mini (`inkling2_loop`) — support spec

Status: **stage 1 (artifact)**. A `.ninfer` artifact is built from the published checkpoint and
verified against it; no C++ model module or kernel exists yet. This document is the contract the
later stages are measured against; every claim below is either read
from the checkpoint's own metadata (`config.json`, `model.safetensors.index.json`,
`hf_quant_config.json`), from the vLLM day-0 write-up, or marked **unverified** where only the
reference implementation can settle it.

Upstream references: [model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini),
[vLLM loop write-up](https://vllm.ai/blog/2026-11-20-inkling2-mini). NInfer conventions follow
[qwen3_5-model.md](qwen3_5-model.md): code fixes the mathematics, config supplies the dimensions.

## 1. Target profiles

The fork keeps three build profiles; the two cards this work is *verified* on are the first two.

| Preset | Card | Compute capability | SM count | RAM budget | Role |
|---|---|---|---:|---:|---|
| `release-4080s` | RTX 4080 SUPER 32 GB | `sm_89` | 80 | 32 GB | primary target, full quality |
| `release-5060ti` | RTX 5060 Ti 16 GB | `sm_120a` | 36 | 16 GB | fits only at reduced precision |
| `release` | RTX 4090 24 GB | `sm_89` | 128 | 24 GB | kept buildable, upstream's card |

Nothing in Inkling support is allowed to regress the three existing 27B models; the new
architecture lands as an additive module (see §11).

## 2. Model shape

| Quantity | Value | Source |
|---|---|---|
| Total / active parameters | 26.4 B / 3.4 B, 5.4 B effective per token after three passes | model card |
| Text hidden width | 2304 | `config.json` |
| Layers | 24 = prelude 0–5, loop core 6–17, coda 18–23 | `loop_config` |
| Layer applications per token | **48** (core runs `num_passes = 3`) | derived |
| Attention | 24 Q heads / 8 KV heads / head_dim 128 | `config.json` |
| Attention pattern | 20 sliding-window layers (window 1024), 4 global (5, 11, 17, 23) | `local_layer_ids` |
| MoE | 128 routed experts, top-8, intermediate 1152, no shared expert | `n_routed_experts` |
| Vocabulary | 201 024 (unpadded 200 058), untied embed/unembed | `vocab_size`, `unembed.weight` |
| Position capacity | 1 048 576 | `model_max_length` |
| Vision | 23-layer CLIP-style ViT, patch 20, merge 2, temporal 2, 32×32 position grid | `vision_config` |
| Drafter | DSpark, 3 layers, block 8, markov rank 256, taps at visits 5/23/41/48 | `mtp_config` |
| Checkpoint | 19.53 GB single file + 636 MB `mtp.safetensors` | index |

## 3. The loop

- Prelude and coda run once. The core runs three times **with shared weights**; every pass starts
  from the prelude's output, not from the previous pass's output.
- Later passes see earlier passes only through **messages**. Each core sublayer input — attention
  query, key, value and MLP — gets its own message read from the previous pass, so one token reads
  **12 layers × 4 inputs = 48 messages**.
- A rank-64 controller merges each message (FP32, no FMA contraction):

  ```
  h   = W_down · [rms(input), rms(message)]
  out = (1 − tanh(W_forget·h)) · input + tanh(W_gain·h) · message
  ```

  Checkpoint tensors: `layers.{6..17}.controllers.{attn,k,v,mlp}.{down,forget,gain}_weight`
  (48 controllers, 144 tensors), plus the learned read queries
  `middle_cycle_cross_pass_queries`, `middle_cycle_cross_pass_kv_queries` and the per-layer
  `attn_res.{attn,mlp}_queries.{0..23}`, `attn_res.output_query`.
- **Each pass owns its KV cache and its conv state.** The model has 24 layers but 48 KV caches and
  48 conv-state caches. This is the single most expensive consequence of the loop and it is the
  reason the cache manager has to key on *visit index*, not layer index.

Visit schedule (index = position in the 48-visit sequence):

| Visits | Layers | Passes | Cache count |
|---|---|---|---|
| 0–5 | 0–5 (prelude) | 1 | 6 |
| 6–17 | 6–17 (core) | 1 | 12 |
| 18–29 | 6–17 (core) | 2 | 12 |
| 30–41 | 6–17 (core) | 3 | 12 |
| 42–47 | 18–23 (coda) | 1 | 6 |

DSpark taps the hidden state at visits **5, 23, 41, 48** — i.e. end of prelude, end of pass 2, end
of pass 3, and the final output. `mtp_config.target_layer_ids = [5, 23, 41, 48]` confirms the
visit numbering used here.

## 4. Attention

Per layer, from the checkpoint's tensor names:

| Tensor | Meaning |
|---|---|
| `wq_du`, `wk_dv`, `wv_dv`, `wo_ud`, `wr_du` | attention projections factorised through shared "u"/"d" and per-head "d"/"v" spaces, plus a separate relative-bias path |
| `rel_logits_proj.proj`, `rel_l2_clip.scale`, `d_rel = 16`, `rel_extent = 1024` | learned relative-position bias of rank 16 over a 1024-token extent |
| `q_l2_clip.scale` | L2 clip on q (threshold 22.627…), K clipped separately |
| `attn_sink_proj.weight`, `gate_attn.weight` | attention sink logit and output gate |
| `k_sconv.weight`, `v_sconv.weight`, `attn_sconv.weight`, `mlp_sconv.weight` | short convolutions, kernel 4 (`sconv_kernel_size`), one per sublayer stream |
| `use_value_norm_no_gamma = true` | value normalisation without a learned gain |
| `log_scaling_floor = 8192`, `alpha = 0.1` | log-scaled attention on the global layers |
| `final_logit_softcapping = 30.0` | logit soft cap at the head |

**Unverified**: the exact u/d factorisation (whether `wq_du` maps d→u or u→d, and how `d_rel`
enters `wr_du`/`rel_logits_proj`). Settle against `transformers`' `inkling2_loop` modelling code
before writing the kernel; do not infer it from shapes alone.

`attn_res` block residuals (`n_layers_per_attn_res_block = 6`,
`attn_res_normalize_values = true`) run once per group of six layers, so one core pass produces
exactly two blocks. Combined with the loop this means the residual stream is renormalised six
times per token, not four.

## 5. MoE and head

| Quantity | Value |
|---|---|
| Experts / selected | 128 / 8 |
| Expert intermediate width | 1152 |
| Gate bias | yes (`mlp.gate.bias`) |
| Route scale | 8.0 |
| Gate / up soft caps | 4.0 / 25.0 |
| Expert weights | fused gate+up (`w13_weight`) and down (`w2_weight`) |

`route_scale = 8` with `top-8` of 128 and no shared expert is a materially different router from
the Qwen MoE path (which has a gated shared expert and no route scale). The existing
`sparse_moe` kernels take the routing as input, so the router itself is a small addition; the
soft caps belong inside the expert activation, not the router.

## 6. Numerics in the checkpoint

`hf_quant_config.json` groups 1116 of the model's 1244 tensors into five classes; the 128 it
omits are exactly the block scales and amax metadata listed below.

| Group | Count | Contents | Activation quant |
|---|---:|---|---|
| fp32 | 74 | controllers, `attn_res` queries, clip scales | none |
| bf16 | 993 | norms, attention, sconv, embeddings, vision | none |
| NVFP4 | 40 | experts of layers 0–19 (`w13`, `w2`) | per-token amax, block 16, e4m3 scales, fp8 outer scale |
| MXFP8 | 8 | experts of layers 20–23 | block 32, `ue8m0` scales |
| bf16 | 1 | final norm | none |

Auxiliary NVFP4 tensors (40 `*_scale`, 40 `*_amax`, 40 `*_outer_scale_denominator`) and the 8
MXFP8 `*_scale` tensors are **not** listed in the quant config; a converter that trusts that
file alone will silently drop every scale. `tools/convert/inkling2.py --check` counts them:
40 NVFP4 payloads, 8 MXFP8 payloads, 48 block scales, 80 metadata tensors, 1068 bf16. This is the first trap, and `tools/convert/inkling2.py
--check` exists to catch it.

The dequantisation contract is settled, and it was verified on the real shard, not inferred:

```
NVFP4  w = e2m1(low nibble = even column, high nibble = odd column)
           * e4m3(scale word, one per 16 columns)
           * amax[expert] / outer_scale_denominator      (denominator = 2688 = 448 x 6)
MXFP8  w = e4m3(code) * 2**(scale_byte - 127)           (one scale per 32 columns)
```

The check that makes this trustworthy is scale-free: every expert matrix must come out at
`rms = 1/sqrt(fan_in)`. Measured over all 24 layers, `w13` lands on `0.02083 = 1/sqrt(2304)` and
`w2` on `0.02946 = 1/sqrt(1152)`, uniform to five decimals. A missing, doubled or 448-scaled outer
factor cannot hit both numbers.

`sm_89` has no FP4 tensor cores and no block-scaled FP8 MMA, so NVFP4 there is a **storage** format
only: codes and scales are read at 4.5 bits and dequantised in-kernel against bf16 tensor cores —
the same mechanism the Q4 groupwise path already uses. Ada does have FP8 MMA, so the four MXFP8
layers can run on fp8 tensor cores once the per-32 `ue8m0` scale is folded in-kernel.

**Pass-through needs no container change.** NInfer stores experts as separate logical parameters
(`experts/{e}/...`), and each expert carries its own FP32 weight divisor, so
`divisor = denominator / amax[e]` reproduces the stored numerics exactly. Round-tripping one expert
through `tools/artifact/codecs/nvfp4.py` preserves the code, scale and divisor words bit-for-bit
and matches the direct dequantisation to 3e-8 (FP32 multiply-versus-divide). Converting the same
expert to Q4 groupwise instead measures cosine 0.9935 against the stored values — a second
quantisation the pass-through avoids.

## 7. Tensor naming

Prefixes are `model.llm.*` (822 tensors) and `model.visual.*` (422). Layers are
`model.llm.layers.{0..23}.<sublayer>.<tensor>`; the vision tower is
`model.visual.backbone.{embeddings,layers.{0..22}}.*` with `ln_1`/`ln_2`, `ls_1`/`ls_2.gamma`,
`mlp.c_fc`/`c_proj`, `proj_in`/`proj_out`. Any name outside these two trees is a converter error.

## 8. Memory budget

Weights at the two candidate store formats, using the measured 26.4 B parameters:

| Store format | Bits/param | Weights | Fits 16 GB? | Fits 32 GB? | Extra loss vs the checkpoint |
|---|---:|---:|---|---|---|
| NVFP4 pass-through (as shipped) | 4.5 | 14.9 GB | no | yes | **none** |
| Q4 groupwise, per-64 fp16 scales | 4.25 | 14.0 GB | no | yes | cosine 0.9935 per matrix |
| Q3 groupwise (new recipe) | 3.25 | 10.7 GB | yes | yes | not yet measured |

Decision: the 4080 SUPER carries the checkpoint's NVFP4 payload unchanged (zero quantisation loss,
2x the scale traffic versus Q4); the 5060 Ti needs Q3 and that is the only deliberately lossy step.
Q4 stays available as a middle option to measure, not as the default.

KV per token, counting only the four global layers and their visits (5 once, 11 ×3, 17 ×3, 23
once = 8 visit-caches × 8 KV heads × 128 dim × 2 tensors):

| KV dtype | Bytes/token | 32 GB card, ~256K ctx | 16 GB card, ~32K ctx |
|---|---:|---:|---:|
| bf16 | 32 KiB | 8.0 GB | 1.0 GB |
| fp8 / int8 | 16 KiB | 4.0 GB | 0.5 GB |
| rotated 4-bit (`rk4v4`) | ~9 KiB | 2.3 GB | 0.3 GB |

Sliding-window layers cost a fixed window each (1024 × 4 KiB × 40 visit-caches ≈ 160 MB), not a
per-token term.

Practical consequence: **the 1M context in the model card is not reachable on either card.** 256K
on the 4080 SUPER at `rk4v4` is the realistic target; the 5060 Ti gets ~32K and only with Q3
experts.

## 8a. Stage-1 artifact (built)

`tools/convert/inkling2.py --emit` produced `inkling2-mini.ninfer`: **19.68 GB**, **7212 bound
parameters**, 7215 objects, 1760 s on one H200-class card with 192 host cores.

| Format | Parameters | What it holds |
|---|---:|---|
| `nvfp4` | 5120 | experts of layers 0-19, E2M1 codes + E4M3 block scales + per-expert divisor, untouched |
| `q8_g32_fp16` | 1024 | experts of layers 20-23, requantised from MXFP8 at the same 8-bit, per-32 granularity |
| `bf16` | 994 | attention, controllers, `attn_res`, sconv, embeddings, vision tower, norms |
| `fp32` | 74 | the controllers' arithmetic and `attn_res` queries, as stored |

Read-back against the source, not against the converter's own state:

| Parameter | Check |
|---|---|
| `text/layers/0/mlp/experts/0/w13` (nvfp4) | cos 0.99986, max abs diff 7.5e-09 — the FP32 multiply/divide rounding only |
| `text/layers/20/mlp/experts/0/w13` (q8) | cos 1.00000, median relative error 0.0078 |
| `text/embed/weight` (bf16) | bit-exact |

The artifact is a **slot bag**, not yet a model: every stored tensor is one logical parameter named
after its source path (`text/layers/6/attn/wq_du.weight`, `text/layers/6/mlp/experts/7/w13`). The
C++ model module of stage 2 binds those slots to the mathematics; nothing here presumes how the
u/d factorisation composes.

## 9. Drafter and vision (stage 4)

`mtp.safetensors` is a separate 636 MB bf16 file with 54 tensors: `layers.{0..2}`, `markov_head.*`
(6), `mask_embed`, `fuse_weight`, `norm`. It is *not* the Qwen MTP shape — it is a 3-layer DSpark
draft that reads four target visits and carries an RNN-style markov head of rank 256. Do not
attempt to reuse `mtp.cpp`; start from `docs/maintainer/dflash.md` conventions instead.

The vision tower is CLIP-style, not Qwen-VL style: `class_embedding`, separate
`cls_position_embedding`, per-block `ls_1`/`ls_2.gamma` layer scales, GELU `c_fc`/`c_proj`. The
existing `vision` execution path assumes MRoPE and a different block layout, so the tower is a new
module that may reuse the ops but not the wiring.

## 10. Acceptance criteria

| Stage | Gate |
|---|---|
| 0 | this document; `tools/convert/inkling2.py --check` passes on the real checkpoint metadata |
| 1 | **met**: artifact written, NVFP4 slots read back at cos 0.99986 (9e-3 relative to FP32 rounding), bf16 slots bit-exact |
| 2 | text-only, greedy, 8K: all 48 visit hidden states within the tolerance the vLLM write-up measured (≈ FlashAttention-vs-PyTorch divergence), short prompts token-identical to the reference |
| 3 | 48 KV + conv-state caches, 128K context on the 4080 SUPER, prefix reuse |
| 4 | vision + DSpark; GSM8K and 32K perplexity within noise of the reference |
| 5 | serving parity: chat template, reasoning effort, tool calls, `/v1/messages` |

## 11. Where the code goes

Additive only; the shared files are `src/models/registry.cpp`, `src/models/CMakeLists.txt`,
`src/ops/CMakeLists.txt` and the root `CMakeLists.txt`, and each gains lines, not restructuring.

```
src/models/inkling2/          config, load, program, execution, state, frontend
src/ops/relbias_attn/         relative bias + sink + gate + log scaling
src/ops/sconv/                depthwise short convolution, kernel 4
src/ops/attn_res/             message read, rank-64 controller, block residual
tools/convert/inkling2.py     converter (stage 1)
tests/inkling2/               oracle dumps and parity checks
```

## 12. Getting the weights under a 20 MB/s link

Only three files are needed for stages 1–3 (the 19.5 GB checkpoint, `mtp.safetensors`, the
tokenizer). At 20 MB/s the checkpoint alone is ~16 minutes; `hf download` resumes, so a dropped
connection costs nothing. Sequence: metadata first (a few hundred KB), then the checkpoint while
the converter is written, then `mtp.safetensors` only when stage 4 starts.

```bash
hf download thinkingmachines/Inkling-2-Mini config.json model.safetensors.index.json \
  hf_quant_config.json tokenizer.json tokenizer_config.json --local-dir E:\LLM\inkling2
hf download thinkingmachines/Inkling-2-Mini model-00001-of-00001.safetensors --local-dir E:\LLM\inkling2
```
