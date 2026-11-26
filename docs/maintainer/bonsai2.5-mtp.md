# Bonsai 2.5 MTP: two routes, and what they cost

Status: research in progress, branch `feat/bonsai2.5-research`. Covers Ternary Bonsai 2.5 27B's
multi-token-prediction draft head, which Bonsai 2 did not ship.

The release publishes the draft head inside the `-mtp` GGUFs: one `blk.64` nextn block, 15 tensors -
`attn_q/k/v/output` and `ffn_gate/up/down` in **Q4_0**, the two attention norms at width 256, the FFN
norms, and `nextn.eh_proj` [10240, 5120] plus `nextn.{enorm,hnorm,shared_head_norm}` in F32 - and
`qwen35.nextn_predict_layers = 1`.

There are two ways to get a draft head into the artifact:

- **Route A** - copy Qwen3.8-27B's head from an existing `.ninfer` (what Bonsai 2's recipe does). The
  base model is the same, the residual stream is never rotated, so the unrotated head applies.
- **Route B** - read `blk.64` out of the `-mtp` GGUF, so no Qwen3.8 reference is needed and the shipped
  head is what runs.

## Route A, measured

Converted with the existing recipe; no code change was needed:

    tools.convert --model model --recipe bonsai2_27b --components text,mtp \
        --source gguf=<PTQ1_0>.gguf --source mtp=qwen3_8_27b_a8.ninfer --proposal --device cuda

728 objects, 6.75 GB, 674 s. `--proposal` adds the indexed proposal head (a 131072-entry token table).

| draft tokens | decode | acceptance |
|---:|---:|---:|
| 0 (no MTP) | 73.5-78.3 tok/s | - |
| 1 | 91.0 tok/s | 89.6 % |
| **2** | **157.1 tok/s** | **81.2 %** |
| 3 | 155.9 tok/s | 74.4 % |

**Quality: the output is byte-identical with and without MTP** - 472 bytes either way on the
"why is the sky blue" prompt. That is the expected result rather than a lucky one: speculative decoding
verifies every accepted token against the target model, so enabling it cannot change what is emitted.
The "intelligence unaffected" criterion is therefore satisfied by construction, and the measurement
confirms the implementation does not break it.

Two caveats on the numbers. The card was shared with another workload throughout (424 W of a 500 W cap,
load average ~30), so the absolute rates are about half of what the same artifact does when idle - the
same artifact measured 143.8 tok/s with MTP off on a quiet machine. The **ratio** is the transferable
number: 157.1 / 73.5 is **2.1x**. And draft=3 was sampled once at 155.9 after an earlier run of the same
configuration measured 75.1, which is contention, not a property of the configuration.

## Route B, what it would take

Nothing in the tree reads `blk.64` from a GGUF, so all three pieces are new:

1. `bonsai_base` currently takes the MTP config from the reference artifact. It would have to derive it
   from `qwen35.nextn_predict_layers` when the source GGUF has one.
2. `PrismCheckpoint` maps GGUF tensor names to the HF names the builder expects. It has no mapping for
   `blk.N.nextn.*`, and the artifact's own `mtp/` names are what `_bonsai_mtp` walks, so the mapping has
   to be written against those.
3. Q4_0 is llama.cpp's legacy 4-bit block, not one of this converter's formats, so those tensors have to
   be decoded and re-quantized rather than copied as stored words.

Estimated at 1-2 days including a synthetic fixture.

## What Route B would buy

Route A already satisfies both acceptance criteria, so Route B has to be justified on something else:

- **No Qwen3.8 artifact at conversion time.** Real - a user with only the Bonsai GGUFs can build a
  speculative artifact today only with Route A.
- **A smaller draft head.** Route A copies Qwen3.8's Q8/BF16 head; the shipped one is Q4_0. A smaller
  draft head is read every draft step, so it should be faster, though the draft is a small share of the
  round.
- **Not a quality argument.** The draft head only proposes; rejected proposals cost a verify pass and
  nothing else. A better head raises acceptance, which is a speed argument.

So Route B is worth building if the shipped head's acceptance is meaningfully higher than the borrowed
one's 81.2 %, or if conversion without a Qwen3.8 reference matters. Both are cheap to check once the
mapping exists.

## Route B: the "missing gate" was my misreading

Correction to an earlier note in this file. I compared the artifact's 16 `mtp/` parameters against the
shipped head's 15 tensors, found no GGUF counterpart for `layers/0/attention/gate`, and concluded route B
needed a schema change. That was wrong, and the artifact says so.

For the **main** full-attention layers the converter already splits one fused tensor:

    params : attention/query  attention/key  attention/gate  attention/value
    sources: blk.3.attn_q (rows 0..6143)   blk.3.attn_k   blk.3.attn_q (rows 6144..12287)   blk.3.attn_v

`attention/query` and `attention/gate` are two halves of `blk.3.attn_q` [5120, 12288]. The shipped
`blk.64.attn_q` is **[5120, 12288]** - the same fused layout - so the MTP layer's gate comes from the same
split, and route B needs no schema change at all.

What route B needs is what the first estimate said:

1. `bonsai_base` emitting the MTP config when the source GGUF carries `blk.N.nextn.*`.
2. A `PrismCheckpoint` mapping from the `blk.64` names to the `mtp.` names the builder walks:
   `nextn.eh_proj` -> `mtp.fc.weight`, `nextn.enorm`/`hnorm`/`shared_head_norm` ->
   `mtp.pre_fc_norm_embedding`/`pre_fc_norm_hidden`/`norm`, `blk.64.attn_*` ->
   `mtp.layers.0.self_attn.*`, `attn_norm`/`post_attention_norm` -> `mtp.layers.0.*_layernorm`,
   `ffn_*` -> `mtp.layers.0.mlp.*`.
3. Q4_0 decode: llama.cpp's legacy 4-bit block, not one of this converter's formats, so those tensors are
   decoded and re-quantized rather than copied as stored words.

So the route is viable and the comparison is still open. The cost is real - the mapping plus a Q4_0
decoder plus a fixture - but it is a reader, not a schema change.
