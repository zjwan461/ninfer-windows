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

- One 256K lane already fits the 4080 SUPER: peak 24.5 GB per
  [NINFER_4080S_BASELINE.md](../../NINFER_4080S_BASELINE.md) section 4, alongside 131072/4 (30.0 GB)
  and 262144/2 (29.5 GB). Nothing needs to change for the native window.
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

### Level 1 — spend the VRAM: ~350-400k tokens in one view

Today's 256K lane leaves roughly 5-7 GB idle. Raising the envelope to the measured ceiling is memory
planning plus long-context plumbing (envelope limits, prefill chunking at depth, vision off), not a
new mechanism.

- **Gain:** 1.4-1.6× native window with zero quality change and no new state model.
- **Cost:** days. **Risk:** low.
- **Gate:** peak VRAM and prefill rate at 350k, measured; decode must not regress because attention
  now walks a longer visible prefix.

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
| Full-attention view ≈ native window within tolerance | 70 % | measured on the reference implementation, not here |
| **GDN "state at the frontier" hypothesis** | **50-60 %** | untested; decides the whole level |
| 1M workspace on this 32 GB card (host tier) | 80 % | 33 GiB fits in 64 GiB RAM; NVMe only for overflow |
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