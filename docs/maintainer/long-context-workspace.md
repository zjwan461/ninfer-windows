# Long-context workspaces beyond the native window

Planning reference for context *capacity* work, as opposed to the speed levers in
[performance-levers.md](performance-levers.md). Written after evaluating
[KVMem](https://arxiv.org/html/2609.04852v1) (arXiv 2609.04852) against this engine. Nothing here is
committed work; the point is to separate the three questions that "bigger context" hides, rank them
by risk, and name the one experiment that decides whether the expensive option is available at all.

## 1. Three different things get called "bigger context"

| Quantity | Meaning | Bounded by |
|---|---|---|
| **Native window** | what one forward pass may attend | the model's trained/rope-configured limit (256K for Qwen3.8-27B here) |
| **GPU-resident KV budget** | how much history can stay in VRAM | free VRAM after weights, workspace, graphs |
| **Addressable workspace** | how much history a session can *recall* without reprocessing it | backing store (host RAM, NVMe) |

The first is a model property, the second is what today's engine already maximises, and the third is
what KVMem builds. Asking for a 1M-token *native* window is a different (and worse) request: it needs
rope scaling, degrades quality, and the KV alone is 64 GiB. Asking for a 1M-token *addressable
workspace* with a 256K native view is sound, and it is the only version of this that is worth
building.

## 2. Where the engine stands

- **Measured (section 9), and it revises the baseline.** With the 15.9 GiB Qwen3.8-27B int8 artifact
  in *this* tree, the native ceiling is the *KV format*, not a plan:
  `--kv-dtype bf16 --max-context 208512` starts with 1.30 GB of slack, `--kv-dtype bf16
  --max-context 262144` refuses ("requires 17,672,843,008 bytes, but only 15,813,574,656 bytes are
  available"), and `--kv-dtype int8 --max-context 262144` starts comfortably (9.35 GB runtime) and
  serves 228,404 tokens at 4/4 needle recall. `--max-context 393216` is refused on a different ground
  — "max_context exceeds the configured position capacity" (the artifact's 262,144 positions). So the
  native path on this card ends at **262,144 tokens, and int8 KV is what reaches it**. The baseline's
  24.5 GB single-lane figure must be a different configuration (KV dtype or artifact).
- Qwen3.8-27B is hybrid: 48 GDN layers with O(1) recurrent state and **16 full-attention layers**
  (`full_attention_interval 4`). At 4 KV heads × head_dim 256 × K+V, one token costs 16 × 4 × 256 ×
  2 × 2 B = **64 KiB at fp16, ≈33 KiB at 4-bit**; the 262144 measurement implies 32.8 KiB/token,
  which corroborates the geometry. So GPU-only ceiling ≈ (32 GB − 15.9 GiB weights − ~2 GB runtime) /
  33 KiB ≈ **350-400k tokens**.
- Host tier: 33 KiB/token × 1M ≈ 33 GiB, which fits the machine's 64 GiB RAM with room to spare, so a
  1M workspace needs no NVMe at all — NVMe is overflow, not the design centre.
- The attention contract already separates the three tables KVMem needs:
  `ordinary_decode_batch(ids, cache_positions, rope_positions, kv_table_rows, linear_state_slots)`.
  What that contract does **not** allow is an arbitrary logical view: `causal_softmax_attention`
  documents "query head h attends cache rows `[0,p]` through table row `kv_table_rows[b]`"
  ([softmax_attention.h](../../include/ninfer/ops/softmax_attention.h), lines 118-121). The logical
  prefix is monotonic; only the page mapping is free.
- Existing assets that carry most of the distance: paged KV with typed pools, address spaces and
  consumer views ([paged-kv-cache.md](paged-kv-cache.md)); context cache with candidate selection,
  retention and Device/Host checkpoints
  ([resource-scheduling-and-context-cache.md](resource-scheduling-and-context-cache.md)); raw
  transition records for GDN state commitment ([replayssm-gdn.md](replayssm-gdn.md)).
- Product constraints that matter: startup-fixed concurrency 1-8, bounded FIFO, **no active-request
  preemption**, one compact decode batch per round. A recall point must therefore coincide with a
  request boundary. It happens to: one agent step is one request. That is also where KVMem schedules
  its recall (after prefill, before decode).

## 3. Three levels, ranked by risk not by ambition

### Level 1 — spend the VRAM: reach the position capacity (262,144 measured)

Measured, this level is smaller than planned and cheaper than planned: the model's own position
capacity is 262,144 tokens, and `--kv-dtype int8` reaches it on this card with 9.35 GB of runtime
(section 9). So Level 1 is a **KV-format and admission decision**, not new code: pick int8 KV for
long lanes, and keep bf16 for lanes that need numeric fidelity. Beyond 262,144 the position capacity
itself has to be raised (RoPE scaling / long-context fine-tune), which is a model-side question, not
a memory one — and that is precisely what makes Level 3 the interesting route: it raises reach
*without* touching the position encoding.

- **Gain:** 208,512 → 262,144 tokens (+26 %) at equal or better speed (prefill 1.37k → 1.68k tok/s,
  decode 22.6 → 26.8 tok/s at 228k tokens), zero quality change, no new state model.
- **Cost:** days. **Risk:** low.
- **Gate:** 4/4 needle recall at >=228k and no decode regression, both measured in section 9.

This level is also the *budget* that Level 3 must fit its execution view into, so doing it first is
not a detour.

### Level 2 — host-tier prefixes with text recall: ~1M addressable workspace

Keep overflowed history in host memory as checkpointed KV (the context cache can already hold it),
and on recall take the *text* back and prefill it again. No KV is re-positioned, no attention changes,
no new state semantics.

- **Gain:** 1M workspace with today's math. Cost of a recall is prefill: 8k tokens ≈ 2.4 s at the
  measured 3.41k tok/s.
- **Cost:** 1-2 weeks (turn "checkpointed prefix" into "fetch this span back into the view").
- **Risk:** low, because nothing new is assumed about the model.
- **Gate:** 30-step agent run with ~10 recalls stays interactive; the 2.4 s per recall must be
  acceptable for the workload, otherwise Level 3 is justified.

### Level 3 — KVMem-style: recall KV instead of text

Replace Level 2's re-prefill with: select historical blocks, restore their KV from host, re-rotate K
into the new logical positions, assemble a fresh prefix, run the query over it. Restoring 8.6 GB over
PCIe costs ~0.4 s against 2.4 s of re-prefill, and the cost does not grow with how much history the
block selection had to sift through.

What this engine already has, and what it would need:

| Piece | Status here | Work |
|---|---|---|
| Paged KV, page tables, address spaces | exists | — |
| Host/Device checkpoints, retention policy | exists | reuse as the backing repository |
| Separate cache vs rope positions | exists | — |
| Non-contiguous prefix assembler | **missing** | new: build a view's page table + position tables, then replay the query |
| Position-independent K capture (Mean-K index) | **missing** | new op in prefill: inverse-rotate K, store per-block mean per layer/KV head |
| Re-RoPE on restore | **missing** | new: rotate restored K from original position to view position, with an explicit numerical tolerance |
| Block selection | **missing** | host-side scorer (softmax over candidate blocks, sink+recent always kept, chronological reorder) |
| Tiering / prefetch overlap | partially (transfer path exists) | stage-out of completed blocks, overlap with chunked prefill |
| Workspace ownership across requests | **missing** | product change: sessions, not just requests |

KVMem's own reference points, for calibration only (their engine, their card): 24 GB RTX 5090 Laptop,
Qwen3.6/3.8-27B NVFP4 + MTP, 1M-token workspace, ~50 tok/s single session, DeepSWE long-context success
43.8 % → 48.4 % against compaction-only context management. They implement two carriers: a
purpose-built C++/CUDA engine (QW3) and a patched llama.cpp, which is the useful existence proof —
the change is portable to a third-party engine of this scale, not something that requires owning the
whole stack.

## 4. The one thing that decides Level 3: GDN semantics

The 16 full-attention layers can be virtualised. The 48 GDN layers cannot: their state is a
recurrence over the *actual* chronology, so a recalled block has no separate GDN state to restore.
Level 3 therefore rests on one hypothesis:

> The GDN state may stay at the frontier, responsible for the true timeline, while recalled blocks
> serve only the full-attention layers.

This is plausible precisely because of the hybrid split — the recurrent state is a compressed memory
of everything, and the recalled full-attention blocks supply the details it lacks — and it is what
KVMem's design implies. But it is an approximation, and if it fails it fails silently: no error, just
a model that gets quietly worse on long-horizon tasks.

**Falsification experiment (about one day, no new code):** run a 256K history through a full prefill
to obtain the reference state and the reference next-token distribution for a query placed at the
end; then run the same query against a *truncated* view (last 128K only) but with the GDN state
carried over from the full run, and compare the next ~20 tokens' distributions against the full-view
reference. If the carried state keeps the distribution close, the hypothesis holds and Level 3 is
available. If it collapses to the truncated-only behaviour, the state does not carry the timeline,
and Level 2 is the ceiling.

Note what this experiment does *not* need: no assembler, no re-RoPE, no index. It is a prefill-prefix
control that the engine can already express.

**Result (measured 2026-11-28, section 10): the hypothesis holds at the observable level.** At a
183,936-token prompt, the reference engine with a 114,688-token device window and a 98,304-token
recall budget produces a **byte-identical** 47-token greedy answer and a byte-identical verbatim
quote of the document's first lines, compared with the same engine in native full-attention mode and
with this fork's native mode. The same engine at a 36,864-token budget silently drops one of four
planted facts, which is the control that shows the probe can see failure. So carrying GDN state at
the frontier while windowing only the 16 full-attention layers does not change behaviour when the
budget covers the needed blocks.

## 5. Feasibility summary

| Component | Confidence | Basis |
|---|---:|---|
| Page-table / transport / assembler engineering | 90 % | the substrate exists; this is scheduling code |
| Mean-K index and block selection | 85 % | additive op plus a host-side scorer |
| Full-attention view ≈ native window within tolerance | **90 %** | section 10: byte-identical output through an 81,920-token view |
| ~~GDN "state at the frontier" hypothesis~~ — now **measured, holds** | **85 %** | section 10: identical output with carried state; the 50-60 % in earlier drafts was the absence of this experiment |
| Per-block original positions in the attention path (replaces re-RoPE) | 75 % | section 10: the reference keeps each block's original RoPE frame and just packs the addresses, so `baked_pos` metadata is needed, not a re-RoPE kernel |
| 1M workspace on this 32 GB card (host tier) | 85 % | 342k measured at 26 GiB pinned host KV and 637k at 30 GiB int8; 1M ≈ 33 GiB at bf16 fits the 64 GiB machine |
| 4-8 week single-session MVP | 70 % | Levels 1-2 land first and become its foundation |
| Productised (multi-session, regression-tested, restart-safe) | 40 % | needs session ownership on top of the FIFO/no-preemption model |

Overall: **~75 %** for Level 3 as a capability, **90 %+** for "a bigger addressable context by some
means". The remaining risk is not the GDN approximation any more; it is engineering volume — the
reference integration is 397 files and +32,257/-1,952 lines against a lineage this fork diverged from
in 983 shared files.

## 6. Acceptance criteria for whichever level is chosen

1. **Quality:** next-token distribution from the assembled view against a full-prefill reference on
   the same history, reported as top-5 overlap and KL, over a handful of queries; and the GDN drift
   test of section 4. Level 2 needs only the first half (it is exact by construction: it re-prefills
   the same tokens).
2. **Restore latency:** measured time from recall decision to first decoded token, at 32k / 128k /
   256k of recalled context, host tier and NVMe tier.
3. **Capacity:** peak VRAM at the maximum workspace, and the largest workspace that still admits the
   configured lane count.
4. **Utility:** an end-task comparison on the workloads this exists for (long agent runs with repeated
   recall), against today's "compact and re-prefill" behaviour — not against a synthetic needle.
5. **No silent degeneration:** the same short-context suite that guards today's decode must stay
   bit-identical when no recall has happened yet.

## 7. What not to do

- **Do not chase a 1M native window.** Rope extension plus 64 GiB of KV is both slower and worse than
  a 256K view over a virtualised workspace; the KVMem paper's whole premise is that the *view* stays
  inside the trained window.
- **Do not build the NVMe tier first.** 1M tokens fits in host RAM here; NVMe only pays off past that,
  and it doubles the transport work while the selection policy is still unvalidated.
- **Do not port the reference implementation's storage format.** The reusable part is the mechanism
  (Mean-K selection, re-RoPE, step-level scheduling), not their block encoding.
- **Do not start Level 3 before the section 4 experiment.** If the GDN hypothesis fails, Levels 1 and
  2 are still worth their few weeks; the eight-week path is not.

## 8. Sequence

| Order | Step | Cost | Decides |
|---:|---|---|---|
| 1 | ~~Level 1 envelope and memory plan~~ **done: int8 KV reaches the 262,144 position capacity** | done | the execution-view budget for Level 3 |
| 2 | ~~GDN carried-state experiment (section 4)~~ **done: output is byte-identical (section 10)** | done | Level 3 exists |
| 3 | Level 3 increments: original-position view, Mean-K index, host tier, then block selection | 3-4 weeks | the workspace size |
| 4 | Level 2 only if the index is deliberately deferred: host-tier prefix with text recall | 1-2 weeks | a fallback that needs the same host tiering |

Steps 1 and 2 are now measured rather than estimated, and both came out better than planned: the
native ceiling is a KV-format choice, and the GDN approximation is observationally invisible. What is
left is engineering volume, and the ordering follows from that — the view and the host tier are
required by any of the options, the Mean-K index is what makes the view cheap, and block selection is
the policy on top.

## 9. Measured: the reference KVMem integration on a 32 GB card

Everything above is planning. This section is measurement, taken 2026-11-28 on
`nvidia-geforce-rtx-4080-super-sm89` (32 GB, driver 621.14, CUDA 13.4, Windows) with the Qwen3.8-27B
int8 artifact (`E:\models\qwen3_8_27b_a8.ninfer`, 15.9 GiB weights, bf16 KV, thinking off).

**The instrument.** `kvmem/kvmem` publishes a maintained NInfer integration
(`backends/versions.json`: upstream `iamwavecut/ninfer-all` at `5fe3107`, seven ordered patches). The
patched tree was reproduced bit-exactly — all seven patches apply clean to `5fe3107`, and
`git write-tree` equals the declared `patched_tree c56b0a27…` — and built here for sm_89
(`-DNINFER_CUDA_ARCHITECTURES=89 -DNINFER_KVMEM_SOURCE_DIR=<kvmem core>`, 1301 units, MSVC 14.44 +
CUDA 13.4). That tree reads this repository's v3 artifact framing unchanged, so the numbers below are
produced by the same artifact this fork ships, not by a re-conversion.

**The workload.** A needle retainer: local corpus text sliced to a fixed size, four unique codes
planted at 5 %, 35 %, 65 % and 99.5 % of the haystack, one user turn asking for the codes. Prompts are
code-heavy (~3.4 chars/token), so sizes are reported in tokens.

| Mode | `--max-context` | Device KV window | Host KV | Prompt | Recalled | TTFT | Prefill | Decode |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| **this fork, bf16 KV** | 208,512 | 208,512 tok (12.7 GiB) | 8 GiB | 186,934 | 4/4 | 136.6 s | 1.37k tok/s | 22.6 tok/s |
| this fork, bf16 KV, second turn | 208,512 | 208,512 tok | 8 GiB | 186,934 | 4/4 | **0.58 s** | cached | 22.8 tok/s |
| **this fork, int8 KV** | 262,144 | 262,144 tok (9.35 GB) | 8 GiB | 228,404 | 4/4 | 136.2 s | 1.68k tok/s | 26.8 tok/s |
| this fork, int8 KV | 393,216 | — | — | — | **refused: position capacity** | — | — | — |
| reference, bf16 KV | 208,512 | 208,512 tok (12.7 GiB) | 8 GiB | 186,934 | 4/4 | 165.0 s | 1.13k tok/s | 22.2 tok/s |
| reference, bf16 KV | 393,216 | — | — | — | **refused: needs 26.3 GB** | — | — | — |
| KVMem, bf16, B=36,864 | 208,512 | 53,248 | 20,480 MiB | 186,934 | 3/4 | 140.8 s | 1.33k tok/s | 35.3 tok/s |
| KVMem, bf16, B=65,536 | 208,512 | 81,920 | 20,480 MiB | 186,934 | 4/4 | 165.3 s | 1.13k tok/s | 31.2 tok/s |
| KVMem, bf16, B=65,536 | 393,216 | 81,920 | 26,624 MiB | 342,354 | 3/4 | 317.7 s | 1.08k tok/s | 31.2 tok/s |
| KVMem, bf16, B=98,304 | 393,216 | 114,688 | 26,624 MiB | 342,354 | **4/4** | 373.3 s | 918 tok/s | 28.2 tok/s |
| KVMem, bf16, layerwise + meanq-cosine | 393,216 | 114,688 | 26,624 MiB | 342,354 | **4/4** | 325.7 s | 1.05k tok/s | 28.0 tok/s |
| KVMem, bf16, second turn | 393,216 | 114,688 | 26,624 MiB | 342,354 | **4/4** | **1.2 s** | cached | 28.2 tok/s |
| **KVMem, int8, B=98,304** | **1,048,576** | 114,688 | 30,720 MiB | 637,768 | **4/4** | 523.7 s | 1.22k tok/s | 34.1 tok/s |
| KVMem, int8, second turn | 1,048,576 | 114,688 | 30,720 MiB | 637,768 | **4/4** | **1.6 s** | cached | 34.1 tok/s |

What the table says:

1. **The cheapest capacity lever is the KV format, and it needs no code.** Switching this fork to
   `--kv-dtype int8` moves the served context from 208,512 to **262,144 tokens** (the model's own
   position limit) and *also* makes it faster: prefill 1.37k → 1.68k tok/s, decode 22.6 → 26.8 tok/s
   at 228k tokens, with 4/4 recall. Halving KV bytes cuts the attention read, so int8 is a
   simultaneous capacity and speed win at long depth.
2. **Two different walls sit just past that.** This fork refuses `--max-context 393216` with
   "max_context exceeds the configured position capacity" (the artifact's 262,144 positions), so the
   wall is configuration, not memory. The reference has no such limit but refuses the same request on
   VRAM: "minimum Engine runtime reservation requires 26,274,490,368 bytes … only 15,796,797,440 bytes
   are available after weights". Either way, **the native path on this card stops at 208-262k**.
3. **KVMem is what crosses that wall.** 342,354 tokens at bf16 (114,688-token device window, 26 GiB
   host) and **637,768 tokens at int8** (same window, 30 GiB host) both reach 4/4, at 918-1.22k tok/s
   prefill and 28-34 tok/s decode. The cold TTFT at 637k is 8m44s; nothing else measured here serves
   more than 262k on this card.
4. **Fidelity is budget-dependent, not free.** At 186,934 tokens, B=36,864 loses the middle needle and
   B=65,536 keeps all four; at 342,354 tokens the same shift (65,536 → 98,304) turns 3/4 into 4/4. A
   recall budget that is too small fails quietly — it returns a plausible answer with a missing fact,
   which is exactly what section 6's acceptance criteria have to catch.
5. **Bounded attention is a decode win.** At the same 186,934-token prompt, decode goes 22.2 → 31.2
   tok/s because the visible prefix is 81,920 tokens instead of 186,934; at 637,768 tokens the
   114,688-token window still decodes at 34.1 tok/s. Prefill is untouched: the history is still read
   once.
6. **Second-turn latency is already ours, and KVMem keeps it at 637k.** An identical repeat costs
   584 ms TTFT (2.2 s total) here at 186,934 tokens via `cache 186,927 (100.0%, turn closure)`; the
   reference at 637,768 tokens costs **1.6 s TTFT / 2.9 s total** via `cache 637,761 (100.0%, shared
   prefix)`. So the agent loop pays the prefill once per history, not once per step — and that path
   exists in this fork already; KVMem's contribution is the *size* it can hold, not the reuse.
7. **The newer strategy is worth taking.** `--kvmem-recall-strategy layerwise --kvmem-query-scorer
   meanq-cosine` reaches the same 4/4 at 342k with a 325.7 s TTFT instead of 373.3 s (-13 %), so a
   port should target the layerwise Query/Pressure path, not the legacy cumulative-attention one.
8. **This fork's prefill is faster than the reference's.** The same 186,934-token prompt costs 136.6 s
   TTFT (1.37k tok/s) here versus 165.0 s (1.13k tok/s) there, with equal decode and equal recall.
   A port must not trade this away for KVMem bookkeeping.

**What follows for the work.** Two of the three levels in section 3 are now measured, and the ordering
changes with them. Level 1 ("spend the VRAM") is really "choose the KV dtype": int8 already reaches
the model's position limit, and the remaining VRAM buys concurrency, not reach. Level 3 is the only
route past 262k, and it is proven on this card at 637,768 tokens. Level 2 (host-tier prefixes plus
text recall) is now the weakest option of the three: it needs the same host tiering that Level 3 needs
and leaves the prefill in the path, so the cheap version of "bigger workspace" is int8 KV plus KVMem,
not a text-recall layer.

**Operational pitfalls found while measuring** (all of them cost a run):

- `--kvmem-query-max-tokens` defaults to 0, meaning *capture the whole prompt*. At 187k tokens that is
  a multi-TB Query capture and the request dies with `bad allocation` after prefill. The reference
  recipe's 512 is not a tuning knob, it is a precondition.
- `--kvmem-host-mib` must cover the *whole* history, not the device window: at 187k tokens bf16 that
  is 11.4 GiB, so the recipe's 12,288 MiB is marginal and 20,480 MiB is not. The failure is again
  `bad allocation` at generation start, after the prefill has already been paid.
- `--kvmem` is a switch; the budget is `--kvmem-budget N` (`--kvmem 36864` is an "unknown argument").
- Native capacity is decided at startup from free VRAM; KVMem replaces it with an explicit
  device window (`B + R`), which is why the free-VRAM line drops from 931 MiB to 7-11 GiB.

**Reproducing the harness.** The patched reference tree lives at `E:\workspaces\ai\_kvmem-ref`
(branch `kvmem-patched`, `git write-tree` = `c56b0a27…`), the KVMem core at `E:\workspaces\kvmem`;
build with
`cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DNINFER_KVMEM_SOURCE_DIR=E:/workspaces/kvmem/kvmem -DBUILD_TESTING=OFF`.
`-DBUILD_TESTING=ON` fails on this machine at `tests/CMakeLists.txt:1 find_package(Python3)`.
The serving line used for the 342k rows is
`ninfer-serve.exe <artifact> --max-context 393216 --no-thinking --kvmem --kvmem-budget 98304 --kvmem-host-mib 26624 --kvmem-gen-reserve 16384 --kvmem-sink-tokens 12288 --kvmem-recent-tokens 1024 --kvmem-query-max-tokens 512 --port 18200`
and the line used for the 637k rows differs only in the capacity and KV format:
`--max-context 1048576 --kv-dtype int8 --kvmem-host-mib 30720` (the device window stays 114,688).
The needle driver is `E:\workspaces\ai\_tmp\needle_test.py` (stdlib only, run with
`E:\software\miniconda3\python.exe`), driven through `E:\workspaces\ai\_tmp\run-needle.ps1`; the
prompts are built from `eval/corpora/perplexity-1m/data/`, so a run reproduces with the same corpora
and no network.

## 10. Measured: the windowed view reproduces the full view

Section 4 said Level 3 rests on one hypothesis — that GDN state may stay at the frontier while only
the 16 full-attention layers are windowed. This section tests that hypothesis on the one observable
that matters: does the answer change?

**Design.** One fixed prompt of 183,936 tokens (the same 640,000-character haystack, four planted
fact codes at 5/35/65/99.5 % depth, plus a question). Two probes: one that needs all four distant
facts, one that needs a verbatim quote from the very beginning of the document. Greedy decoding
(`temperature 0`), so the comparison is exact rather than statistical. Four configurations, all
against the same 19.03 GiB artifact:

| Configuration | Attention view | Budget | codes probe | quote probe |
|---|---|---:|---|---|
| this fork, native bf16 | all 183,936 tokens | — | 4/4, 47 tokens | exact |
| reference, native bf16 | all 183,936 tokens | — | 4/4, 47 tokens, **byte-identical to the row above** | — |
| reference, KVMem | **81,920 tokens** (114,688 window, 98,304 budget) | 98,304 | 4/4, 47 tokens, **byte-identical** | **exact** |
| reference, KVMem, negative control | 43,008 tokens | 36,864 | **3/4 — BETA dropped, no error** | — |

Exact texts (identical in rows 1-3):

```
ALPHA=7134-MAGENTA
BETA=2291-ORANGE
GAMMA=5077-TURQUOISE
DELTA=8642-VIOLET
```

```
#pragma once

// Implements: include/ninfer/ops/residual_add.h
```

**What this establishes.**

1. **The GDN approximation is invisible here.** The truncated run attends 81,920 of 183,936 tokens,
   carries GDN state forward from the full timeline, and still emits the same 47 tokens the
   full-attention run emits. If carried state were semantically wrong, the divergence would show up
   as a different first token at the answer, not as a cosmetic difference later.
2. **Two independent lineages agree.** This fork's native path and the reference's native path — 983
   shared files different, different kernels, different prefill speed (1.37k vs 1.13k tok/s) — produce
   the same greedy output on the same artifact. The output comparison is therefore sensitive enough
   to be worth trusting, and the two builds are numerically compatible at this depth.
3. **The recall budget is the only fidelity knob, and failure is silent.** At 36,864 tokens the answer
   is a clean three-item list with no error, no truncation marker, no warning: one fact is simply
   gone. Any acceptance criterion has to be a retainer like this one, never "the answer looked fine".
4. **Re-RoPE can be dropped from the port.** The reference keeps every block in its original RoPE
   frame and only packs the *addresses* (`compact_start` is an attention address, and the payload
   carries `baked_pos` per block). The verbatim quote from position 0 is exact through the windowed
   path, so the port needs per-block original positions in the attention path, not a re-RoPE kernel —
   which removes the numerically touchiest item from the earlier plan.

**What it does not establish.** No native reference exists above 262,144 tokens on this card, so
beyond that length the needle retainer (section 9) is the only check; and this experiment exercises
the block *view*, not the block *selection* from a host tier, since the whole history fit in host RAM
and was selected once. Those remain to be validated in the port itself.

**Reproducing.** `E:\workspaces\ai\_tmp\fidelity_test.py` takes `--base-url`, `--probe codes|first_lines`,
`--chars 640000` and prints JSON with the decoded text; both probes run against any of the serving
lines above.

## 11. Where the port stands

The port is planned in [kvmem-port-plan.md](kvmem-port-plan.md), which is the authority for the
work; this section only records what already exists in the tree so the plan is not read as a wish
list.

- **The executed view is a plan, not a kernel.** src/core/kv_view.{h,cpp} turns a request's page
  order plus a selected block set into the compacted page order (build_kv_view). Pages keep their
  physical identity, so the keys already written keep the position phase they were written with.
- **The selection is a pure function.** select_kv_blocks in src/core/kv_workspace.{h,cpp}
  chooses whole blocks under a page budget, always keeping the sink prefix and the recent tail,
  and throws rather than silently shrinking a request. plan_workspace in the same header turns a
  selection into the residency move (stage_out before stage_in, a compacted remaps list, and
  the visible token count).
- **The index primitives are on the device.** include/ninfer/ops/kv_block_pool.h provides
  kv_block_mean_keys (per 32-token-scale block, per layer, per KV head, the mean of the
  position-free key) and kv_block_scores (cosine of the GQA-meaned query against those means),
  both qualified against an independent FP64 oracle.
- **The three compose, and the accounting is tested.** tests/test_kv_recall_plan.cpp runs
  scoring, selection, assembly and residency against each other over a 637k-token workspace and
  asserts the invariants the execution path will depend on: the budget is never exceeded, the
  needle and the compulsory blocks are always present, the executed order has no holes, a cold turn
  stages in exactly its selection, and a warm turn stages out only what the new selection dropped.
- **The acceptance gate exists.** tools/bench/run_kv_recall_fidelity.py compares a baseline and a
  candidate endpoint on byte-identical greedy output for two probes, and requires four markers to be
  recovered; it is the check that needs to pass before the view is allowed to change an answer.

The mainline step is now in the tree as well, behind `NINFER_KV_RECALL_APPLY` (off by default):
at the prompt boundary the sequence is moved onto an address published over the compacted page
order, the source address is released, and `rope_delta` carries the tokens that left the cache.
kvmem-port-plan.md section 7.11 records the three places that had to learn that a cache frontier
and a timeline frontier are two numbers.

### 11.1 Measured: capture and the boundary decision on this card

RTX 4080 SUPER, Qwen3.8-27B int8 artifact, `--max-context 262144 --kv-dtype int8`, capture enabled
with `NINFER_KV_RECALL=1`.

- **Capture keeps up.** `kv recall: captured 9 columns over 16 layers -> 0 complete blocks of 128`
  at startup, then `captured 200832 columns over 16 layers -> 1569 complete blocks of 128` after a
  200,897-token prompt. The 16 full-attention layers are dense; GDN layers never appear.
- **A wide budget is the identity.** With `NINFER_KV_RECALL_BUDGET=262144` (wider than the prompt)
  the boundary reports `view 200832 tokens (dropped 0), 1569 blocks kept, rope delta 0`, i.e. the
  decision is exactly the current view, so the answer must not move — and it does not: the same
  200,897-token needle prompt recovers the same four markers as the run with capture off
  (112.8 s vs 128.4 s; the difference is capture cost and run-to-run variance, not a different
  computation).
- **A narrow budget drops and says so.** With `NINFER_KV_RECALL_BUDGET=8192` on a 16,896-token
  history the boundary reports `view 8192 tokens (dropped 8704), 64 blocks kept, rope delta 8704`.
  Nothing is applied yet, so this is a decision, not an effect.
- **Cost.** Capture adds about 15 % to a 200k-token prefill on this card (112.8 s -> 128.4 s, one
  run each) and is inert when the environment variable is unset, which is the default.

### 11.2 Measured: the address swap, and what the view is worth

RTX 4080 SUPER, Qwen3.8-27B int8 artifact, `NINFER_KV_RECALL=1 NINFER_KV_RECALL_APPLY=1`, greedy,
`--kv-dtype int8`.

- **The swap commits and the next tokens decode from it.** A 14,382-token prompt with budget 4096:
  `boundary ... view 4014 tokens (dropped 10368), 32 blocks kept, rope delta 10368` then
  `applied view 4014 tokens, rope delta 10368, next sample 14382, address row 0`. The answer to the
  four-marker probe is the same string with the flag off and on, so keeping 4,014 of 14,382 tokens
  left the needed content in view.
- **Fidelity follows the retention ratio, not the mechanism.** The same 200,135-token prompt under
  three budgets:

  | budget | view | dropped | third marker |
  |---|---|---|---|
  | full (flag off) | 200,135 | 0 | `5077-TURQUOISE` |
  | 131,072 | 131,015 | 69,120 | `5077-TURQUOISE`, identical to the full view |
  | 32,768 | 32,711 | 167,424 | `5083-BL`, invented |

  In the 32k view the block holding that marker is inside the window (depth 0.95 of 200k, the window
  covers everything after 167k), so the failure is not a dropped block: at 16 % retention the model
  answers from a diluted context. Raising the budget restores the exact full-view answer, which is
  what rules out a structural error in the swap (positions, phases, GDN alignment).
- **The safe retention ratio is not one number.** `tools/bench/run_kv_recall_gate.py` serves the
  baseline and the candidate one after the other (a 262k context fills most of the card, so they
  cannot run together), compares greedy output byte for byte and prints the ratio. On the 200k
  prompt above: budget 196,608 is the identity (no drop, 4/4, identical); budget 131,072 keeps
  67.2 % and answers `GAMMA=5803-CYAN` where the full view says `GAMMA=5077-TURQUOISE` — correct
  shape, invented value, the other three markers right. So fidelity is prompt- and depth-dependent,
  and a gate has to report the ratio it ran at instead of claiming a global threshold. A depth-
  stratified recall curve is the missing evidence.
- **67 % is a genuine pass, and the earlier "passes" were not.** Three defects made the swap refuse
  instead of apply (a kept page short by construction, a frontier that was not page aligned, a
  protection wider than the columns the page held); the gate now reads `refused` out of the log, so a
  refusal can no longer look like a pass. After those fixes, budget 131,072 keeps 67.3 % and answers
  identically to the full view on two different marker layouts (5/35/65/95 % and 10/30/50/70/90 %),
  with `applied view 131072 ... rope delta 63744` in the log.
- **25 % fails by selection, not by dilution.** At budget 49,152 the marker at 65 % depth is answered
  with an invented code. `NINFER_KV_RECALL_TRACE=1` prints the kept block ids: 384 blocks in range
  0..1521, and the marker's block is not among them — it is outside sink and recent, and Mean-K
  scoring did not rank it high enough. The model never had the evidence. So the remaining work at low
  retention is selection quality, not the compaction mechanism.
- **A view needs room to grow.** The compacted address is entitled to exactly the pages it keeps;
  growth is restored after the source is released. If the card's capacity is already spent by the
  history (for example `--max-context 16384` on a 14k prompt), that restoration fails and the next
  append reports `KV coverage exceeds active entitlement`. Acceptance therefore runs with headroom
  (32768 or 262144 in these runs).
- **What this is worth.** 200k of history at 65 % retention answered identically to the full view
  while fitting in half the cache, which is the point of the level: the same card can hold a
  request it could not hold natively, at a retention ratio the operator chooses.

## 12. Research record: what was built, measured, and left open

This section closes the KVMem line of work at commit `25659b28` on
`docs/performance-levers`. Everything below was measured on an RTX 4080 SUPER (32 GB, sm_89) with
the Qwen3.8-27B int8 artifact (`E:\models\qwen3_8_27b_a8.ninfer`), `--kv-dtype int8
--no-thinking`, greedy decoding.

### 12.1 What is in the tree

| Layer | Module | State |
|---|---|---|
| Planning | `src/core/kv_workspace.{h,cpp}` — `select_kv_blocks`, `plan_workspace` | tested |
| View | `src/core/kv_view.{h,cpp}` — `build_kv_view`, `plan_view_boundary`, `plan_sequence_view` | tested |
| Decision | `src/core/kv_recall_decision.{h,cpp}` — `plan_recall` | tested |
| Index | `src/core/kv_recall_index.{h,cpp}`, `kv_recall_set.{h,cpp}` | tested |
| Ops | `include/ninfer/ops/kv_block_pool.h` — mean keys, scores, accumulate, finalize | FP64-oracle tested |
| Capture | `text.cpp attn_mix` before `text_rope`; `kv_recall_cursor` device accumulator | live, ~15 % prefill cost |
| Publish | `prepare_compacted_fork`, `commit_prefix_fork`, `adopt_writer`, partial tail page | store test |
| Swap | `ProgramImpl::apply_recall_view` behind `NINFER_KV_RECALL_APPLY` | measured |

Two environment switches gate everything: `NINFER_KV_RECALL` (capture + boundary decision) and
`NINFER_KV_RECALL_APPLY` (the address swap). Both default to off, so the shipped path is unchanged.
`NINFER_KV_RECALL_BUDGET` sets the token budget; `NINFER_KV_RECALL_TRACE` and
`NINFER_KV_RECALL_TRACE_BLOCKS` print the kept block ids and a block's score/rank.

### 12.2 Measured results

- **Capture keeps up.** `captured 200832 columns over 16 layers -> 1569 complete blocks of 128`
  after a 200,897-token prompt. The 16 full-attention layers are dense; GDN layers never appear.
- **A wide budget is the identity.** With a budget wider than the history the boundary reports
  `dropped 0` and the answer is unchanged; the swap is skipped in that case.
- **The swap works and is faithful at high retention.** 14,382-token prompt with budget 4096:
  `applied view 4014 tokens, rope delta 10368, next sample 14382`, answer identical to the full
  view. 200k-token prompt with budget 131,072 (67.3 % retention): identical answer on two different
  marker layouts (5/35/65/95 % and 10/30/50/70/90 %).
- **Low retention fails by selection, not by dilution.** Budget 98,304 (50.5 %) and 49,152 (25.2 %)
  both answer a mid-history marker with an invented value of the same shape. The trace shows why:
  at 49,152 the kept set is 384 blocks spread over 0..1521 and the marker's block is not among them.

  | block | role | score | rank of 1522 | kept |
  |---|---|---:|---:|---|
  | 990 | marker at 65 % depth | −0.056 | 482 | no |
  | 0 | sink | 0.330 | 0 | yes |
  | 1521 | recent tail | 0.078 | 40 | yes |
  | 700 | arbitrary mid-history | −0.068 | 544 | no |

- **The ranking is flat in the middle of the history.** Sink and recent score positive, everything
  between them is a near-zero negative, so an arbitrarily placed fact does not rise above its
  neighbours.
- **That is not a sampling or geometry artefact.** Halving the block to 64 tokens moved the failure
  from GAMMA to ALPHA (one block entered the kept set, another left it) without sharpening the
  signal; narrowing the query pool from 512 to 8 columns left the marker's score unchanged to six
  decimals.

### 12.3 Why it fails, and what would fix it

Mean-K cosine, averaged over 16 layers and 4 KV heads, has no discriminative power for an
arbitrary single-shot fact lookup. The operator is correct — it matches an FP64 oracle — but the
signal assumes what the KVMem paper assumes: an agent workspace where the query and the text to be
recalled share vocabulary and structure (a follow-up question about code that is already in the
transcript). A needle planted in unrelated text does not satisfy that assumption.

The measured consequence is that a compacted view is trustworthy when it is a **contiguous tail**
(drop the head, keep the rest), and untrustworthy when it has to punch holes through the middle of
the history. The 67 % pass is the contiguous-tail case; the 50 % and 25 % failures are hole-punching.

Better interior selection needs a different retrieval signal, not a scoring tweak. The two
candidates the evidence points at are a multi-query union (score each block against every turn's
question and keep the max, so a fact survives when any turn pointed at it) and a
retention-by-attention signal (rank blocks by the attention mass they actually received rather than
by a query-to-key cosine). The multi-query union was drafted and reverted unverified when this line
of work stopped; it is the cheaper of the two and is the recommended next step.

### 12.4 What this does not deliver

- **It does not make the context longer.** The position table stops at 262,144. Neither the KV
  window nor the swap extends it; both keep original positions by construction. Reaching 1M needs
  YaRN (rope_scaling with factor 4, `original_max_position_embeddings` 262144), which this engine
  does not implement at all.
- **It does not make a 16 GB card hold Qwen3.8-27B.** The int8 weights alone are 15.9 GiB and 256k
  of int8 KV is another 9.35 GB. A 16 GB card would need the 10.9 GiB IQ3_S weights, which this
  tree refuses to load (`unknown tensor layout: gguf_blocks_v1`), plus window-only KV residency.
- **It does not implement Level 2.** Host-tier prefixes with text recall was judged the weakest of
  the three levels and was not built; the host-tier primitives (`HostKVArena`, `HostKVExtentStore`)
  exist but carry no scheduler.
- **The low-retention path is not usable as shipped.** `NINFER_KV_RECALL_APPLY` stays off by
  default precisely because interior selection is not yet trustworthy.

### 12.5 What is worth having

Two things, in this order:

1. **`--kv-dtype int8` is the inexpensive capacity win.** It moves the card from 208,512 (bf16,
   VRAM-bound) to the full 262,144 position capacity, uses 9.35 GB instead of 12.7 GiB, and is
   faster at the same time (228k prompt: prefill 1.37k → 1.68k tok/s, decode 22.6 → 26.8 tok/s).
   No code change; it is an existing engine switch whose number this work measured.
2. **The compacted view is a residency technique, not a length technique.** Its payoff lands on a
   card too small for full residency, or on a model whose position table goes beyond 262,144. On
   this 32 GB card, with int8 KV that already fits, it saves VRAM and bounds decode cost (34 tok/s
   at 637k in the reference engine), but it does not unlock anything the card could not already do.

### 12.6 Reproducing the measurements

```powershell
# build (RTX 4080 SUPER, sm_89, 80 SM wave geometry)
cmake -S . -B build-4080s -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=89 -DNINFER_TARGET_SM_COUNT=80 -DBUILD_TESTING=ON
cmake --build build-4080s --target ninfer-serve

# the acceptance gate: baseline and candidate in sequence, greedy, byte comparison,
# prints the retention ratio and fails when the swap was refused
python tools\bench\run_kv_recall_gate.py ^
  --exe build-4080s\apps\ninfer-serve.exe ^
  --artifact E:\models\qwen3_8_27b_a8.ninfer\qwen3_8_27b_a8.ninfer ^
  --corpus eval\corpora\perplexity-1m\data\infer --chars 700000 ^
  --max-context 262144 --kv-dtype int8 --budget 131072
```

The gate sets `NINFER_KV_RECALL=1 NINFER_KV_RECALL_APPLY=1 NINFER_KV_RECALL_BUDGET=<budget>` for the
candidate run and leaves them unset for the baseline, kills the first server before starting the
second (a 262k context fills most of a 32 GB card), and reads `applied` or `refused` out of the
candidate's log so a refused swap cannot look like a pass.

### 12.7 Traps worth remembering

- **Editing a header alone does not rebuild its users here.** `kv_store.h` is not in the target's
  depfile, so ninja reports `no work to do` and a stale binary answers. Touch the participating
  `.cpp` files before rebuilding.
- **Probes must be inserted as their own line with the file's own line ending.** A script that split
  on `\r\n` and inserted `\n`-terminated lines glued code onto probe lines, and deleting "lines
  containing kvdbg" deleted real statements.
- **A refused swap looks exactly like a pass.** The store refuses to publish a view that would drop
  a live page; the answer then matches because the full view never moved. Three such defects were
  found only after the gate started reading `refused` out of the log.
- **A crashed test with no output is usually a locked exe or a stale object,** not a logic error.

