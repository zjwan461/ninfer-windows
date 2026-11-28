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

## 5. Feasibility summary

| Component | Confidence | Basis |
|---|---:|---|
| Page-table / transport / assembler engineering | 90 % | the substrate exists; this is scheduling code |
| Mean-K index and block selection | 85 % | additive op plus a host-side scorer |
| Full-attention view ≈ native window within tolerance | 80 % | measured in section 9: 4/4 needles at 342k tokens through a 114,688-token window |
| **GDN "state at the frontier" hypothesis** | **60 %** | the reference keeps recurrent state untouched and is coherent at 342k (section 9), but the carried-state truncation experiment is still unrun |
| 1M workspace on this 32 GB card (host tier) | 85 % | 342k measured at 26 GiB pinned host KV; 1M ≈ 33 GiB at bf16 fits the 64 GiB machine |
| 4-8 week single-session MVP | 70 % | Levels 1-2 land first and become its foundation |
| Productised (multi-session, regression-tested, restart-safe) | 40 % | needs session ownership on top of the FIFO/no-preemption model |

Overall: **~65 %** for Level 3 as a capability, **85 %+** for "a bigger addressable context by some
means" once Levels 1 and 2 are counted.

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
| 1 | Level 1 envelope and memory plan, measure the real GPU-only ceiling | days | the execution-view budget for Level 3 |
| 2 | GDN carried-state experiment (section 4) | 1 day | whether Level 3 exists |
| 3 | Level 2: host-tier prefix with text recall, 1M workspace | 1-2 weeks | the product value of a large workspace |
| 4 | Level 3 only if step 2 passes: assembler, Mean-K index, re-RoPE | 3-4 further weeks | the recall latency improvement |

The first two steps are cheap enough to run before any commitment, and together they turn every
number in this document into a measured one.

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