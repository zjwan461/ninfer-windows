# Performance levers: what is left on the two mainline models

Planning reference for decode and prefill work on the Windows profiles, opened on branch
`docs/performance-levers`. Nothing here is a commitment to implement; the purpose is to rank what is
left, put a measured number next to every claim, and name the gate that decides whether the next step
is worth taking. Implementation work orders from this catalogue rather than from intuition.

Scope: `Qwen3.8-27B` (groupwise-int, `qwen3_8_27b_a8.ninfer`) and `Ternary Bonsai 2 27B`, on the
RTX 4080 SUPER 32 GB (`build-4080s`) and RTX 4090 profiles. Every 4080 SUPER measurement below is an
existing result from [`NINFER_4080S_BASELINE.md`](../../NINFER_4080S_BASELINE.md) or the profile
section of [`README.md`](../../README.md); nothing new was measured for this page. Estimates are
marked **est.** and carry their falsifier.

## 1. Where the time actually is

| | Qwen3.8-27B int8 | Ternary Bonsai 2 27B |
|---|---:|---:|
| Weights resident | 15.9 GiB | 5.52 GiB |
| Decode, no speculation | 37.8 tok/s | 91.6 tok/s |
| Bandwidth roofline at 736 GB/s | 43.1 tok/s | 124.2 tok/s |
| Roofline reached | **88 %** | **73 %** |
| Decode, MTP | 74.9 prose / 86.6 code (k=3) | 143.4 (k=2) |
| MTP acceptance length | 2.89 tok/round | 2.10 tok/round |
| Prefill, chunk 1024 | 3.41k tok/s | 3.61k tok/s |

Two fitted cost models carry most of this page. Qwen's round is
`26.2 ms + k × 3.3 ms` (baseline section 2), fitted from k=1..5 and accurate to about 2 %. Bonsai's
round is directly measured at 14.6 ms for 2.10 committed tokens (143.4 tok/s), i.e. roughly a fixed
9 ms plus 3.2 ms per committed token if the same two-term shape holds; that decomposition is
inferred from one operating point, so treat it as a shape, not a fit.

**Decode is a bandwidth purchase with a token multiplier.** One verify forward reads the whole
artifact (23.2 ms floor, 26.2 ms measured — 86 %), so the cost of columns 5..16 in the same round is
almost entirely compute, while the 15.9 GiB is already spent. That asymmetry, not kernel efficiency,
is where the remaining headroom lives.

**Prefill is the GEMM, measured to the decimal.** 3,410 tok/s over 24.78 B matmul parameters
(27.32 B minus the `token_embd` gather and the `output_head`, which prefill only evaluates for the
last column) is 169 TFLOP/s of useful work. The Q4 6144×5120 report measures **168.54 TFLOP/s at
T=1024, 80.45 %** of the card's 209.5 TFLOP/s dense BF16 peak
([q4-linear](examples/q4-linear.md)). End-to-end prefill is therefore indistinguishable from the
GEMM's own efficiency; there is no scene-setting, no scheduling loss to recover. The remaining 20 %
of the peak and the 2× that INT8/FP8 tensor cores would add are the only prefill levers that exist.

**The 4090 ratios confirm the model.** 0.62× decode and 0.59× prefill against 736/1008 GB/s and
80/128 SMs — the 4080 SUPER is not mistuned, it is a narrower card.

## 2. The one asymmetry worth exploiting

At k=3 the verify reads 15.9 GiB to commit 2.89 tokens. At k=7 the same 15.9 GiB could commit six or
eight tokens if they were acceptable, and the extra columns cost compute that the round was not
paying anyway. Measured support for that claim already exists: with `--ngram chain` the same round
commits **6.8-11.8 tokens** on restating prompts at 2K context (4090 CLI, `WINDOWS_PORT.md`), and
MTP 3 + n-gram is 288.7 tok/s against MTP 3's 152.3 and DFlash2 d6's 214.6 on the same four prompts.

So the ranking below puts "accept more per round" ahead of "make the round cheaper". The reverse
prioritisation is what section 7 of the baseline document already refuted.

## 3. Lever catalogue

Ordered by (verified or strongly-supported gain × confidence) ÷ cost. Each entry names its gate.

### L1. Widen the verify window beyond 16 columns (n-gram rounds first)

Today every route stops at 16: `kMtpVerifyMaximumDrafts = 15`, `kMaximumVerifyTokens = 16`,
`gdn_input_proj_conv_record` `2..16`, `gated_delta_net_replay_record` `2..16`,
`kSamplerMaxColumns = 16` ([ngram plan](ngram-speculation-plan.md) lists all of them with files).
The n-gram policy already reaches 11.8 tokens/round on some prompts, so the cap is being leaned on.

- **Gain:** **est.** a round at T=16 costs the same 15.9 GiB of traffic; if verify grows from 26.2 ms
  to 30-35 ms and commits 8 tokens instead of 2.89, round throughput goes from 79 to roughly
  230-270 tok/s on n-gram-friendly workloads. The estimate is deliberately wide; the falsifier is
  cheap (below).
- **First step, one day:** measure the verify cost curve at T=4/8/16/32 before writing any Op work —
  `bench/ninfer_linear_bench --suite qwen3_6_27b --sweep 4:32:4` covers every projection of the model
  at once, and one session-level timing at a wider capture confirms it end to end. If the curve is
  already compute-steep by T=16, L1 loses most of its value and the ranking changes.
- **Cost:** Option A (record/fold to 65, one lane) is 6-10 days with oracle requalification at the new
  widths; Option B (prefill-route wide verify with state snapshot/replay) is 4-6 days but costs a
  second prefill of the committed prefix on partial acceptance — n-gram rounds are partially accepted
  at the end of every copied span, so B is likely worse. Option C (multi-lane) only with recorded
  multi-agent sessions.
- **Gate:** the simulator, not intuition. `tools/spec_sim` with
  `--ngram-n 8,12,24 --caps 15,32,64` for `mtp`, `chain:ngram-mod` and `select:ngram-mod`, both
  `--wide-verify` models, on recorded sessions. Phase 2 was written to proceed only if cap 32/64 beat
  cap 15 with a substantial `>15 tok` share.

### L2. Get the n-gram gain that already exists through the server

Measured, not estimated: the CLI shows +90 % on restating prompts (288.7 against 152.3), and the
server shows +86 % on long-context edits (258.3 against 138.6). But for short agent prompts the
server shows 165.4 against DFlash2's 169.1 — a tie — while the same four prompts are 273-374 tok/s in
the CLI. The pool, the graph family selection, three lanes and thinking-on are the candidates, and
none of them is a new algorithm.

- **Gain:** recovering even half of the CLI/server gap is 30-60 % on agent workloads, at zero new
  math and no quality risk.
- **Cost:** diagnosis only. `--request-log-jsonl` and `/metrics` already expose per-round widths and
  acceptance; the round-width histogram will show whether the pool is drafting at all in server mode.
- **Gate:** none. This is the cheapest item on the page and should be done before anything else.

### L3. Bonsai's ternary small-T kernels (73 % of roofline)

The README already names this: decode without speculation reaches 88 % of the roofline on Qwen3.8 and
73 % on Bonsai, "the ternary kernels are the ones with room left". A 73 → 90 % move is +23 % on
91.6 tok/s and moves MTP2 from 143 to roughly 175.

- **Gain:** **est.** +20-25 % on both Bonsai numbers, single-digit risk, no accuracy question.
- **Cost:** T=1 GEMV work on `t5_g128_fp16` (unpack and vectorize); shape coverage is the model's
  projections, already enumerated in `src/ops/linear/`.
- **Gate:** `ninfer_linear_bench --suite` at T=1 per shape first; if the individual GEMVs already sit
  at 85-90 %, the loss is elsewhere (launch count, fusion) and the lever is smaller than it looks.

### L4. INT8/FP8 tensor cores for the GEMM (prefill 2×, and it enables wide verify)

Prefill runs at the BF16 peak's 80 %, and Ada's INT8/FP8 MMA is 2× that peak. Nothing else can move
prefill by more than a few percent. This is the same mechanism that makes the upstream NVFP4 numbers
look good on the 5090 (8.3k prefill against 3.27k for groupwise-int on the same engine): FP4 MMA
versus dequantize-then-BF16.

- **Gain:** prefill **est.** 1.6-2× (5.5-6.8k tok/s) if w4a8 with in-register dequant behaves like
  llama.cpp's MMQ on Ada; verify compute halves, which is what wide verification and DFlash2/DSpine
  need to pay for themselves on this card.
- **Cost:** the largest item here — a new arithmetic path under `src/ops/linear/`, per-shape routes,
  and qualification against the same FP32 oracle. Not a tuning exercise.
- **Gate:** one shape, one format, measured against the BF16 route at T=1024 and T=1. If INT8 MMA
  does not beat BF16 by ≥1.5× on `q4_g64_fp16` 5120×17408 at T=1024, stop.

### L5. A native sub-4-bit storage format (density, not speed of a kernel)

15.9 GiB at ~5.0 bits/weight. A native 3.5-4 bit groupwise format would put the artifact near
12-13 GiB, which is a **1.2-1.3×** decode gain by bytes alone, before any spec change. This is the
only lever that improves *every* workload uniformly.

- **Constraint:** it must be a NInfer `QType` with its own kernels, not ggml block storage — the GGUF
  family was evaluated and rejected (see section 5).
- **Cost:** new format + per-shape kernels + conversion; the same surface L4 touches.
- **Gate:** perplexity and task scores against the current artifact, and the byte count actually
  landing below 13 GiB for the real parameter mix.

### L6. The unexplained 1.8 ms inside a draft hop

The hop is 3.3 ms. Its listed ops sum to ~1.5 ms with `--lm-head-draft`, and every one of them
already runs at 67-90 % of bandwidth (baseline section 7.1). K-split schedules C8..C40 measured
identical, and the MTP draft path is already graph captured (`capture_mtp_decode_batch`,
`mtp.cpp:215`), so the naive "one projection" and "launch overhead" hypotheses are both dead. The
residual is small ops the per-op suite does not cover plus graph replay.

- **Gain:** if the residual is real work at low efficiency, 3.3 → 2.4 ms moves the optimum k from 3 to
  4-5 and Qwen from 74.9 to **est.** 85-90 tok/s.
- **Cost:** one nsys timeline of a single hop. Reopen only with that picture in hand.
- **Gate:** the timeline must name a specific op with a floor gap, as section 7.1 did for the five
  large ones. Without it, treat the hop as closed and leave it alone.

### L7. Concurrency and memory layout

Measured: 262144/2 lanes peaks at 29.5 GB with 73.0 tok/s single and 102.9 aggregate; 131072/4 lanes
peaks at 30.0 GB with 127.0 aggregate; 262144/3 does not fit (≈33 GB). Bandwidth-bound decode means
aggregate throughput is nearly free until the KV traffic and the extra lanes' verify columns start to
cost.

- **Gain:** **est.** 10-25 % aggregate if a third 262K lane fits, or if 131K/4 can take a fifth.
- **Cost:** memory planning, not kernels. Smaller KV (dtype already `rk4v4-e8`) or fewer resident
  full-attention layers is the only way to buy the room.
- **Gate:** peak VRAM at the candidate configuration, measured, before any code change.

## 4. What is closed, and why

Do not reopen these without new evidence; each has a measurement behind it.

| Closed item | Evidence |
|---|---|
| Faster draft hop by K-split scheduling | C8/C16/C24/C32/C40 all measured 74.5-74.8 tok/s |
| Draft hop as "one projection" | Five ops measured at 67-90 % of bandwidth; hop already graph captured |
| 80 vs 128 SM build | 37.8 against 37.7 tok/s, indistinguishable |
| `--prefill-chunk` above 1024 | Flat: 3.41k / 3.40k / 3.39k / 3.39k at 1024-4096 |
| DFlash2 as the better backend for int8 | Own 5090 corpus: 147.0 against MTP3's 161.7 (−9.1 %); story 84.0 against 130.4 |
| The upstream NVFP4 artifact on this card | 22.09 GiB, larger than the int8 file, therefore slower |
| Porting `gguf_*` block layouts | ≈27k vendored lines for 1.17-1.30× decode and 2.0-2.4× worse prefill; rejected |

Two of these are worth restating because they are counter-intuitive. DFlash2 wins on NVFP4
(+19.5 % corpus, structured 356.8 against 219.8) and loses on int8 — the backend is not
format-neutral, which is a hint that its wide verify is compute-sensitive, and therefore that L4 is a
precondition for it, not an alternative. And the n-gram pool drafts nothing on free-form prose
(97.1 against MTP's 97.0), so its gain is workload-shaped and must be quoted per workload, never as
an average over a mixed corpus.

## 5. Sequencing

| Order | Item | Effort | Decides |
|---:|---|---|---|
| 1 | L2, n-gram server parity | days | whether 30-60 % is sitting in the serving path unused |
| 2 | L1 first step, verify cost curve T=4/8/16/32 | 1 day | whether wide verify is bandwidth-free or compute-steep on this card |
| 3 | L3, Bonsai T=1 GEMV audit | days | whether the 73 % is a kernel gap or a launch-count gap |
| 4 | L1 proper (Option A) if the simulator and the curve agree | 6-10 days | the single largest decode gain available |
| 5 | L4, INT8/FP8 GEMM proof on one shape | weeks | whether prefill gets 2× and wide verify becomes affordable |
| 6 | L5 density or L7 memory, when a uniform gain matters more than a workload-shaped one | weeks | — |

L6 stays parked until something else forces a profile of the draft path.

## 6. Measurement protocol

Every number above is a 3-run median at stock clocks on the profile named in its source; keep that
discipline, because single runs on this card disagree with themselves by several percent, which is
the same order as most of the differences quoted here.

- CLI sweep (decode): `infer --spec mtp --draft-tokens 3 --lm-head-draft --kv-dtype rk4v4-e8
  --prefill-chunk 1024`, one prompt per content type, greedy, 1024 new tokens.
- Per-op floors: `bench/ninfer_linear_bench --suite qwen3_6_27b` (or `--qtype q4 --n 4096 --k 5120
  --sweep 1:32:1` for one shape), built with `-DNINFER_BUILD_BENCHMARKS=ON`.
- Serving: `tools/bench/run_serve_concurrency.py --suite corpus-makespan`, with
  `--request-log-jsonl` so round widths and acceptance are recoverable.
- Policy projection: `tools/spec_sim/` (see its README for `--wide-verify`, `--cost-points` and the
  `--ngram-*` / `--caps` arguments). Rerun it with measured round costs before trusting a projection.
- Prefill: the 1972-token prompt at `--prefill-chunk 1024`; report the chunk sweep only when the
  chunk changes.