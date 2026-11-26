# Ternary Bonsai 2.5 27B: what the current support already covers

Status: research note, 2026-11-26, branch `feat/bonsai2.5-research`, based on `main` @ `a5b49ac`.
Nothing here is implemented. The question it answers is narrow: **can the code that already
supports Bonsai 2 load Bonsai 2.5, and if not, what is actually missing?**

Sources: the published GGUF metadata and tensor tables (read straight out of the files with
`tools/convert/sources/gguf_reader.py` through HTTP range requests, no full download), and the three
existing documents - [bonsai-ternary-design.md](bonsai-ternary-design.md),
[bonsai-ternary-conversion.md](bonsai-ternary-conversion.md),
[bonsai-gguf-inventory.md](bonsai-gguf-inventory.md).

## 1. What Prism shipped

| file | size | note |
|---|---:|---|
| `Ternary-Bonsai-2.5-27B-PTQ1_0.gguf` | 5.99 GB | dense trits, the pick on NVIDIA |
| `Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf` | 6.23 GB | same weights **plus a draft head** |
| `Ternary-Bonsai-2.5-27B-PQ2_0[-mtp].gguf` | 7.25 / 7.49 GB | 2-bit slot packing |
| `Ternary-Bonsai-2.5-27B-Q2_0[-mtp].gguf` | 7.67 / 7.91 GB | stock llama.cpp |
| `Ternary-Bonsai-2.5-27B-mmproj-BF16.gguf` | 0.93 GB | vision tower |
| `Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf` | 0.63 GB | vision tower |

Two things changed against Bonsai 2, and neither is the weight format: the **MTP draft head now ships
in the GGUF** (Bonsai 2 had none, and the recipe borrowed Qwen3.8-27B's), and the **release is trained
for agentic work** with `xhigh`/`medium` reasoning effort. The card states the ternary format, the
5.93 GB size and the Hadamard basis are unchanged.

## 2. The text model is the same file shape

Bonsai 2 and 2.5 both carry **851 tensors and the same architecture metadata**. Diffing the two
inventories:

| | Bonsai 2 | Bonsai 2.5 |
|---|---:|---:|
| tensors | 851 | 851 |
| ternary (`type143` / `PTQ1_0`) | 402 | 402 |
| F32 | 353 | **449** |
| BF16 | 96 | **0** |

Every tensor name, shape and offset pattern matches. The only content change is that the 48 GDN
layers' `ssm_alpha.weight` and `ssm_beta.weight` - 2 x 48 = 96 tensors - are **F32 instead of BF16**.

Architecture metadata is identical to Bonsai 2 (`qwen35`, 64 blocks, 5120 hidden, 17408 FF,
24/4 heads, head_dim 256, ssm 6144/128/16/48, conv 4, `full_attention_interval` 4, 262144 context),
as are every `prism.hadamard.*` key, the tokenizer (248320 tokens, 247587 merges, eos 248046) and the
rope sections. The four metadata keys that differ are `general.name`, `general.version`,
`general.basename` and a new `general.sampling.min_p`, none of which the converter reads.

**`_raw_rows` in `tools/convert/sources/prism_checkpoint.py` already accepts both dtypes**:

    if info.ggml_type not in (_F32, _BF16):
        raise ValueError(...)
    word = 4 if info.ggml_type == _F32 else 2
    dtype = torch.float32 if info.ggml_type == _F32 else torch.bfloat16

So the 96 retyped tensors are handled by the code as it stands. `_bonsai_geometry` compares
exactly the keys that did not change, so it passes unchanged.

Conclusion: **the text path needs no code change** for a PTQ1_0 or PQ2_0 conversion. What it needs is a
run: the claim above is from reading the files and the source, not from a completed conversion.

## 3. The vision tower is byte-for-byte the same shape

The 2.5 mmproj metadata against what `bonsai-ternary-conversion.md` section "Vision tower" documents
for Bonsai 2:

| key | documented for 2 | 2.5 mmproj |
|---|---|---|
| `clip.vision.block_count` | 27 | 27 |
| `clip.vision.embedding_length` | 1152 | 1152 |
| `clip.vision.feed_forward_length` | 4304 | 4304 |
| `clip.vision.attention.head_count` | 16 | 16 |
| `clip.vision.patch_size` | 16 | 16 |
| `clip.projector_type` | `qwen3vl_merger` | `qwen3vl_merger` |
| `clip.vision.projection_dim` | 5120 | 5120 |

The existing `--source mmproj=` path therefore applies unchanged, including the same quantisation of
the tower to the official Vision formats. `mmproj-Q8_0` keeps `ffn_down` in F16, which
`tools/convert/sources/mmproj.py` already decodes.

## 4. The MTP head is the one real gap

Bonsai 2.5's `-mtp` GGUF adds **15 tensors** and `qwen35.nextn_predict_layers: 1`: one complete
`blk.64` nextn block - `attn_q/k/v/output` (Q4_0), `attn_q_norm/k_norm` (F32, width 256),
`attn_norm`/`post_attention_norm` (F32), `ffn_gate/up/down` (Q4_0), plus
`nextn.eh_proj` [10240, 5120] (Q4_0), `nextn.enorm`, `nextn.hnorm` and `nextn.shared_head_norm` (F32).

Today's recipe takes the draft head from an **existing Qwen3.8-27B `.ninfer`** (`--source mtp=...`)
and copies it, optionally requantising its layer projections (`bonsai2_27b_mtp_q4|q5|q4q5`). That
works for 2.5 without any change, because:

- the draft head belongs to the same base model (2.5 is quantised from Qwen3.8-27B),
- the residual stream is never rotated, so the unrotated Qwen3.8 head applies as-is (design doc
  section 0, and the reason the Bonsai 2 recipe copies it),
- the artifact's `mtp/` layout already matches this geometry.

What the shipped head buys is different: it is **Q4_0 quantised from the same base**, where the
existing path copies Qwen3.8's **Q8/BF16** head. The existing path is therefore the higher-precision
of the two, and cheaper to run (fewer weight bytes per draft step is the only thing the Q4_0 head
would win). Reading `blk.64` natively is a convenience, not a capability.

**Nothing in the current code reads `blk.64` from a GGUF**, so a native path is new work: a mapping
from the GGUF nextn names to the artifact's `mtp/` names, the `nextn.*` norms and `eh_proj`, and the
`nextn_predict_layers` metadata check. Estimate: 1-2 days including a synthetic fixture.

## 4b. Covering every packing: PTQ1_0, PQ2_0 and Q2_0

The release ships three packings of the same ternary weights, so "support everything" is a question
about decoders, not about the runtime.

| file | type id | bytes per 128 weights | bits/weight | decoder in this tree |
|---|---:|---:|---:|---|
| `PTQ1_0` | 143 | 28 | 1.75 | **yes** - `prism_gguf.decode_ptq1_0` |
| `PQ2_0` | 142 | 34 | 2.13 | **yes** - `prism_gguf.decode_pq2_0` |
| `Q2_0` | **42** | 36 | 2.25 | **no** - stock llama.cpp's type, absent from both `GGUF_TYPES` and `_BLOCK_LAYOUT` |

The 36-byte block is measured, not assumed: `output.weight` in the Q2_0 file spans 357,580,800 bytes
for 1,271,398,400 elements, which is exactly 0.28125 B/elem. What sits in those 36 bytes is **not**
something to infer from the size - it has to be taken from llama.cpp's `ggml-common.h` for type 42, the
same upstream source section 1.4 of the design doc already cites for types 142 and 143.

Two consequences worth stating before anyone budgets the work:

**The runtime does not change at all.** The converter's job is to produce `t5_g128_fp16`; an extra GGUF
type is one more block-layout row plus one more decoder. The `.ninfer` artifact for a Q2_0 input is the
same artifact as for PTQ1_0 - same formats, same shapes, same kernels.

**Whether Q2_0 is a lossless repack is a question, not an assumption.** Prism states all three are the
same ternary model and that Q2_0 exists "for stock llama.cpp". If llama.cpp's type 42 stores a per-block
scale in the same g128 grouping, decoding PTQ1_0 and Q2_0 must yield **bit-identical** ternary codes and
scales. If it uses a different grouping or an added offset term, it does not - and then Q2_0 is a
requantisation, PTQ1_0 stays the quality-preferred source, and "support" should say so.

That gives a free oracle: convert both, compare the produced tensors, and the answer falls out with no
reference model and no GPU time. It is the same shape of check as the existing independent scalar oracle
in `tests/convert/test_prism_gguf.py`.

Cost: about a day for the Q2_0 decoder, its block-layout row, a synthetic fixture in
`tests/convert/bonsai_fixtures.py`, and the cross-packing equality test.

## 4c. Phase 0 result: it converts and decodes, with no code change

Ran it. `Ternary-Bonsai-2.5-27B-PTQ1_0.gguf` (5.99 GB), the recipe and the two commands exactly as
`bonsai-ternary-conversion.md` documents them, with the 2.5 file substituted:

    python3 -m tools.convert.bonsai_base --gguf <2.5>.gguf \
        --reference qwen3_8_27b_a8.ninfer --out <model dir>
    python3 -m tools.convert --model <model dir> --recipe bonsai2_27b --components text \
        --source gguf=<2.5>.gguf --name bonsai25 --out bonsai25.ninfer --device cuda

Both succeeded. The conversion wrote 714 objects into a 5.94 GB artifact in 4,067 s, every ternary
projection landing as `t5_g128_fp16` and every norm as `bf16` - the same shape as a Bonsai 2 build. No
source file was modified.

The artifact then loaded and decoded. On the Hopper MIG slice this work runs on:

| measurement | value |
|---|---|
| "The capital of France is" | `**Paris**.` |
| decode, 128 tokens generated | **143.8 tok/s** |
| text prefill, short prompt | 38-54 ms |
| GPU weights | 5.52 GiB |
| planned device total | 6.34 GiB |
| reasoning mode (the default) | emits a thinking block, then the answer; correct Chinese reply to a Chinese question |

Two caveats on the numbers: this card is **not** the RTX 4080 SUPER, so 143.8 tok/s does not transfer -
the 4080 SUPER figure in `NINFER_4080S_BASELINE.md` for Bonsai 2 is 91.6 tok/s on a 736 GB/s card. And
there is no Bonsai 2 artifact on this machine, so this is a first measurement of 2.5, not a comparison
against 2.

### The template warning was a false alarm, and a useful one

`bonsai_base` prints `chat template: reference DIFFERS FROM the GGUF's` and saves the GGUF's copy as
`gguf_chat_template.jinja`. Diffed against `tools/chat_templates/qwen3_8.jinja`, all sixty lines of
difference are NInfer's own documented fork changes - positional system/developer turns,
`continue_final_message`, and tool-result histories whose user turn was removed - not anything new from
Prism. So 2.5's template is the upstream Qwen3.8 template, and this engine's fork already covers it.
The warning exists to make sure nobody assumes that, which is the right behaviour.

Which closes the open question from section 5: **the chat template did not change in 2.5.**

## 5. What is not verified

- **No conversion has been run.** Everything above is inventory- and source-level. The first
  deliverable should be a real conversion of `PTQ1_0` (5.99 GB) with the recipe unchanged.
- **The chat template was not compared.** My range-request parser tripped over the bulk tokenizer
  arrays before reaching `tokenizer.chat_template`, so whether 2.5's template differs (reasoning
  efforts, control tokens) is unmeasured. The card says `xhigh` is the default and `low` is not
  supported, and this engine already exposes `--reasoning-effort xhigh|medium|...`.
- **Numerical parity of the draft head** between 2.5's Q4_0 block and Qwen3.8's head is untested.
- **Quality and speed on 2.5** are unmeasured on any card here. The card's 2.3-3.0x decode claim is
  about *their* kernels against Bonsai 2's release-day build, not about this engine.

## 6. Proposed plan

**Phase 0 - prove the claim (no code).** Download `PTQ1_0`, `PTQ1_0-mtp` and `mmproj-Q8_0`; run
`bonsai_base` + `tools.convert --recipe bonsai2_27b --components text,vision,mtp --source gguf=...
--source mmproj=... --source mtp=qwen3_8_27b.ninfer --proposal` exactly as
`bonsai-ternary-conversion.md` documents, with the 2.5 files substituted. Then decode a prompt and
compare against the Bonsai 2 artifact on the same prompt. Deliverable: either "supported today, here
are the numbers" or the precise failure.

**Phase 1 - guard the dtype change (small).** Extend the synthetic GGUF fixture
(`tests/convert/bonsai_fixtures.py`) so one configuration writes `ssm_alpha`/`ssm_beta` as F32, and
add the assertion to `test_bonsai_recipe.py`. Half a day, and it stops a future reader from
"fixing" `_raw_rows` back to BF16-only.

**Phase 2 - the template and the reasoning defaults.** Read 2.5's `tokenizer.chat_template` from a
full GGUF and diff it against `tools/chat_templates/qwen3_8.jinja`. If it differs, add
`tools/chat_templates/bonsai2_5.jinja` and point the launcher's preset at it. Also record that the
card asks for `min_p=0` while llama.cpp defaults to 0.05.

**Phase 3 - only if wanted: the native MTP head.** Implement the `blk.64` mapping so
`--source gguf=<mtp>.gguf` can supply the draft head without a Qwen3.8 reference. Justified only if
the shipped head measurably beats the copied one, which Phase 0 can answer by comparing acceptance
lengths.

**Phase 4 - presets and docs.** A 4080 SUPER / 4090 launcher preset for 2.5 (mirroring
`start-qwen38-*`), and a short note in `docs/README.md`.

## 7. Recommendation

Run Phase 0 before writing anything. On the evidence above the text and vision paths should convert
and load unchanged, and the MTP path should work by pointing at the existing Qwen3.8 reference - which
would make "Bonsai 2.5 support" a verification exercise plus a preset, not a port.
