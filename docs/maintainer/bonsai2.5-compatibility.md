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
