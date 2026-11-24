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
  (48 controllers, 144 tensors). The merge is computed in FP32 end to end: both RMS norms, the
  rank-64 down projection's consumers, and the two `tanh` gates.

  `ninfer::ops::message_merge` performs the blend in FP32; it is the only path by which one pass
  reaches the next.

  The message itself is read by attention, not by a dot product: `middle_cycle_cross_pass_queries`
  and `middle_cycle_cross_pass_kv_queries` are the learned queries of the four sublayer inputs
  (12 core layers x 2 each, stacked as `[24, 2304]`), attending non-causally with scale
  `1/sqrt(hidden)` over the previous pass's block states plus the running partial state. The
  per-layer `attn_res.{attn,mlp}_queries.{0..23}` are the block-residual reads, which is why four
  of them carry three rows instead of one.
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

`build_schedule` materialises this table and every message link. The first core pass reads nothing;
the 12 core layers read one message each on passes two and three, so a forward pass has 24 message
reads, not 48 — the other 24 are the first-pass visits that start from the prelude output.

DSpark taps the hidden state at visits **5, 23, 41, 48** — i.e. end of prelude, end of pass 2, end
of pass 3, and the final output. `mtp_config.target_layer_ids = [5, 23, 41, 48]` confirms the
visit numbering used here.

## 3c. Message queries: the checkpoint's own layout

The checkpoint stores the cross-pass queries as two `(2 * loop_layers, hidden)` tensors, and the
reference merges them into one `(loop_layers, 4, hidden)`: per loop layer the four rows are
**query, key, value, mlp**, taken as

```
queries    = cross_pass_queries.unflatten(0, (loop_layers, 2))     # [query, mlp]
kv_queries = cross_pass_kv_queries.unflatten(0, (loop_layers, 2))  # [key, value]
merged     = cat([queries[:, :1], kv_queries, queries[:, 1:]], dim=1)
```

That merge lives in `transformers.core_model_loading.Inkling2LoopMergeMessageQueries`, not in the
modelling file, which is why reading only the model source leaves the message queries unaccounted
for. The same file renames the controller weights: `controllers.{q,k,v,mlp}.down_weight` becomes
`*_message_gate.down_proj.weight`, `gain_weight` becomes `message_gate_proj.weight` and
`forget_weight` becomes `forget_gate_proj.weight`.

## 3d. The two short-convolution tap orders

The model has two short convolutions with **opposite tap conventions**, and reading one as the
other shifts the whole key/value path:

- `ShortConvolution` is an `nn.Conv1d` with `padding = kernel - 1`, so for a single token only the
  **last** kernel index (`kernel - 1`) sees the current token; the earlier taps see padding.
- `DynamicShortConvolution` indexes by **backward distance**: `static_weight[0]` weights the
  current token and `static_weight[3]` the token three steps back. Its normaliser is
  `sqrt(2 * kernel + 1)` = 3 for a kernel of four.

Using tap 3 for the dynamic convolution made every full-attention layer's keys and values a
three-step-back mix, which showed up as 3-5% error per full layer and 32% at the last visit. With
tap 0 each full layer lands at 0.2-0.3%.

## 3e. Reference dumps have to use the oracle's token

`dump_stages.py` hooks the model's own submodules, which is the only comparison that does not
share the engine's reading of the model source. It has to run with the token the visit oracle
used, or every comparison is meaningless: the block states, the attention inputs and the logits
all belong to a different prompt. `--token-id 1234` is the token the current oracle was recorded
with; `layer0.in_00` matching `visit_in_0` exactly is the check that it is the right one.

## 9e. The tokenizer is not a drop-in

The engine already carries a BPE encoder, and Inkling 2's `tokenizer.json` looks like the same
family: GPT-2 style byte-level BPE, 199998 vocabulary entries, 424413 merges, 60 added tokens. It
is not the same tokenizer, though. The existing encoder hard-codes Qwen's pre-tokenisation regex
and rejects anything else, while Inkling 2 splits with the Llama-3 style pattern
`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+...`.

So producing text from this engine needs a second pre-tokeniser implementation and its own tests,
not a namespace move. The reference is close at hand - `transformers` encodes and decodes the same
strings, so the encoder can be checked token for token - but it is a few hundred lines of Unicode
classification work, and worth knowing before anyone plans around it.

## 9f. The pre-tokeniser's exact split

Seven alternatives, in order, and the order is what decides the result:

```
[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
| [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
| \p{N}{1,3}
|  ?[^\s\p{L}\p{N}]+[\r\n/]*
| \s*[\r\n]+
| \s+(?!\S)
| \s+
```

`tests/text/llama3_pretokenizer_reference.py` evaluates exactly that pattern with a Unicode-aware
regex engine and writes `tests/text/fixtures/llama3_split.json`: 25 strings and the chunks they must
produce. The behaviours worth pinning down are visible in it - a single leading space attaches to
the following word, digits cap at three per chunk, contractions stay with their word except when
the alternation splits them (`I'll` then `'ve`), and trailing whitespace becomes its own chunk.

The C++ scanner is written against that file, so the pattern is the authority rather than anyone's
reading of it.

## 9g. What the encoder still needs, exactly

The pre-tokeniser split is implemented and checked. The rest of the encoder is pinned down by the
file itself, so nothing here needs guessing:

- `model.merges` is an array of **42,4413 two-element arrays** (`[["i","n"], ["e","r"], ...]`), the
  index being the rank - not the "a b" strings some tokenizers use.
- `model.vocab` maps the byte-mapped token string to its id, with 188 single-byte symbols present, so
  the GPT-2 bytes-to-unicode bijection is the right alphabet.
- `added_tokens` is 60 entries of `{content, id, lstrip, rstrip, single_word, normalized, special}`
  and all of them so far are `special: true`, which means they are extracted before the split rather
  than competing with it.
- `end_of_word_suffix` and `continuing_subword_prefix` are null, so there is no `</w>` bookkeeping.

So the remaining work is the byte bijection, a ranked merge table over it, added-token extraction,
and the inverse for decoding - testable end to end against `transformers`, which encodes the same
strings with the same file.

## 9h. Text in, text out

The session test takes text now, so the whole chain is one command:

```
build/tests/ninfer_inkling2_session_test --artifact inkling2-mini.ninfer \
    --prompt "The capital of France is" --generate 12 --tokenizer tokenizer.json
```

which prints the token ids it made from the prompt, the ids it generated, the decoded text, and the
warm-up and steady-state figures. With the artifact and tokenizer used here it answers

    text: The capital of France is Paris.

That the answer is also *right*, and not merely close to the oracle's logits, is worth a sentence: it
says the chain is correct at the level a reader would judge it, not only at the level a numerical
comparison would.

## 9d. Runbook: measuring on an RTX 4080 SUPER

The 32 GB card holds the pass-through artifact as it stands, so nothing has to be re-converted.
Build with the card's architecture and run the session test:

```
cmake -S . -B build -DNINFER_CUDA_ARCHITECTURES=89 -DNINFER_TARGET_SM_COUNT=76 \
      -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ninfer_inkling2_session_test -j

# steady-state decode rate, warm-up reported separately
build/tests/ninfer_inkling2_session_test \
    --artifact inkling2-mini.ninfer --prompt-ids 1234 --generate 32

# pipelined rate, 48 visits per step
build/tests/ninfer_inkling2_session_test --artifact inkling2-mini.ninfer --bench 128
```

Two numbers come back and both are wanted: the steady-state ms/token and the pipelined tok/s. The
first is what a user feels, the second is what the kernels do. On the 78-SM slice used during
development they are 41-65 ms/token and 16-18 tok/s, with about twenty percent run-to-run spread on
that shared machine; a dedicated card should be both faster and steadier.

The accuracy check needs the recorded oracle and is worth running on the same binary:
`--artifact ... --oracle <dir>` prints the worst visit against the 48-visit recording and the
logits error, and fails if either leaves the BF16 noise floor.

## 9c. The compact recipe, measured

`--compact` requantises the coda's four layers of experts from q8 to q4, which is the one lever
that exists without writing a new format:

| | pass-through | compact |
|---|---|---|
| artifact | 18.33 GiB | 16.31 GiB |
| worst visit against the 48-visit oracle | 0.84% | 4.6% |
| logits max abs error | 0.14 | 1.11 |
| top-5 shared with the reference | 5/5 | 3/5 |

So it buys 2.0 GiB for about five times the error, and 16.31 GiB still does not fit a 16 GB card.
Getting under 14 GiB also means quantising the attention projections and the unembedding, which is
another 1.4 GiB and more of the same error. That is a trade the model's owner should make, not one
to make quietly: the 32 GB card runs the pass-through artifact with no compromise at all.

## 9b. What fits on which card

The pass-through artifact is 18.33 GiB, and the routed experts are 14.87 GiB of it (81%): 9.49 GiB
of NVFP4 for the loop core's twelve layers and 5.38 GiB of q8 for the coda's four.

- **RTX 4080 SUPER 32 GB** holds it as it stands, with a 4096-token context worth about 400 MB of
  caches.
- **RTX 5060 Ti 16 GB** does not. It needs the experts under about 10.5 GiB, and the cheapest route
  there needs no new format: the coda's experts are q8 only because the converter had a values path
  for those layers, and requantising them to NVFP4 takes them from 5.38 to 2.85 GiB, for 15.8 GiB
  total. Quantising the attention projections and the unembedding as well, which are still BF16,
  is worth another 1.5 GiB and lands near 14.3.

A Q3 format would go further, but it means a codec, a kernel and their tests; the numbers above
say it is not needed for the 16 GB target, so it stays unbuilt.

## 8j. Device-side expert selection, concretely

What has to change for capture to be possible, in the order it has to be built:

1. **A device-side table per layer.** `Weight` is a flat POD of about a hundred bytes, so the whole
   set - 128 experts times two matrices - can be uploaded once at session init and indexed on the
   device. The session keeps the table pointer instead of 256 host-side handles.
2. **A gather route beside the existing one.** The 2304x2304 NVFP4 decode kernel already takes the
   unpacked pointers; the gather variant takes the table and a slot index and reads them there. One
   kernel per stored format this path uses (NVFP4 and q4), because the table entry carries the type.
3. **A scaled accumulate that reads its scale from the device.** The router weights already live in
   `s.router_weights`, so the eight accumulating adds take a slot index rather than a float.
4. **Then capture.** With 1-3 in place nothing inside a step depends on the host, the readbacks stop
   being needed at all - not made faster, made unnecessary - and `--graph` loses its probe
   dependency.

The measurement that justifies the order is in §8i: capture is worth forty percent, and it cannot be
taken until these exist.

## 8i. The answer is in the Qwen 3.8 path

The Qwen 3.8 decode does not call operations one after another the way this port does. It captures
them: `program/decode.cpp` has `capture_graph`, `DecodeGraphExecutable` and `install_graph_profile`,
and a token is one replay of an installed graph. That is the whole difference, and it explains every
number in §8g at once - a graph node costs 0.89 us against 4.49 for a launch, and the host stops
re-executing a thousand calls per token.

Mirroring it here needs one thing first. A graph can only be captured when nothing inside it depends
on a host decision, and this path's per-visit expert selection is exactly that decision: the router
writes eight indices on the device, the host reads them, and the eight weight handles follow. So the
order of work is:

1. resolve the selection on the device - a pointer table over the expert weights plus a gather route,
   so the choice never reaches the host;
2. capture a token (or a visit) into a graph and replay it;
3. only then look at arithmetic, which the measurements say is not the constraint.

## 8h. What the host cost is not

The 25 us per operation that §8g leaves unexplained was chased one step further, and three more
candidates are out:

- `ops::linear` validates shape, dtype, contiguity and alignment with a dozen integer comparisons and
  no allocation, so the wrapper is not it;
- no op wrapper allocates or synchronises - the only `cudaMemcpy` in the whole wrapper layer is the
  pitched scatter, which this path does not call;
- `Weight` is a flat struct returned by value, so `matrix_weight` copies about a hundred bytes.

That leaves the per-operation accounting itself, which cannot be settled by inspection: it needs a
profiler that counts calls by name and accumulates host time around them. Until that exists, the
honest statement is that the token is host-bound and that the cost is spread across the operations
rather than concentrated in one, which is a different fix - fewer operations per visit, and fewer
visits repeated - from making one function faster.

## 8g2. What nvidia-smi says that timing cannot

During a benchmark this slice reports GPU-Util 0 percent, 119 W of a 500 W budget, and 18.4 GiB held
by the process. The GPU is not busy - it is being woken up sixteen hundred times a token.

That reframes the measurements above. The token is not 34 ms of arithmetic; it is about 1,600 kernels
of a few microseconds each, with roughly 20 us of device-side latency between them, and the host
waiting on the same rhythm. Bandwidth and flops are irrelevant at this size: the sequence length is
what costs.

So the only optimisation that moves this number is **fewer operations per token**, which is what the
counters in §8g are for, and why the last three commits took 2,017 to 1,585. The obvious next step is
the one that remains: fuse the eight routed experts of a visit into three launches (gate/up for all
eight, swiglu for all eight, down-and-accumulate for all eight) instead of twenty-four, which takes
the expert part of the budget from 864 to about 150.

Measured with the GPU free: 25.3 / 21.8 / 24.5 tok/s launched, 28.9 / 29.0 captured.

## 8g6. Stacking the experts, and the obstacle that has to be decided first

The plan from §8g4 is one stacked bank per layer, expert `e` a row span. Checking it against the
container turns up one real obstacle before any code is written: **the NVFP4 container carries one
amortisation divisor per object**, and this checkpoint's divisor is per expert - 22192.8 for one and
18888 for the next in the same layer. A single bank has one divisor field, so stacking as-is would
scale 127 of 128 experts wrongly.

Two ways out, and they differ in what they cost:

1. **Fold the divisor into the block scales.** Exact in principle - rescale every E4M3 block scale by
   the ratio and the bank has one divisor - but it re-quantises the scales, and the container stores
   scales as E4M3, so the ratio has to survive a 4-bit-mantissa rounding.
2. **A divisor vector beside the bank.** Keep the bank's own divisor at 1.0 and emit a 128-element
   fp32 tensor per layer that the kernel reads by expert index. No re-quantisation at all, one extra
   tiny tensor, and the kernel already has the expert index in hand.

The second is the one to build: it costs a 512-byte tensor per layer and changes no arithmetic. That
matters because the point of this change is the memory layout, and it should not also move the numbers.

## 8g4. What the A3B MoE path teaches, even though it cannot be reused

The engine already has a fused sparse-MoE op (`ops::sparse_moe`), and reading it is worth more than
the op itself: it is **closed to one exact geometry** - 2048 hidden, 256 experts, top-8, plus a shared
expert - so it cannot be called for this model, which is 2304 hidden, 128 experts and no shared
expert.

What it does teach is the shape the data has to be in. Its routed bank is a single contiguous weight,
`[256*1024, 2048]`, and "expert e directly selects its stored row spans; no selected-weight gather or
repack occurs". That is why it can be one launch and coalesced: there is nothing to gather.

This port stores each expert as a separate artifact object, so it gathers 128 pointers per layer and
pays a scattered read for every one of them. Matching the A3B path means changing the container, not
the kernel: one stacked bank per layer for gate/up and one for down, expert e a row span. The kernels
then lose the indirection entirely, and the per-visit launch count falls to the three the fusion was
going to cost anyway.

## 8g5. The coalescing experiment, which lost

The bandwidth arithmetic above says to widen the reads: four consecutive 16-value groups per lane
gives 32 contiguous bytes of codes and four adjacent bytes of scales, where one group at a time
fetches a 32-byte sector for one useful byte.

Measured on the same free GPU, back to back:

| reads | tok/s |
|---|---|
| four groups per lane | 22.0 / 22.2 |
| one group per lane | 28.9 / 28.9 |

It lost by a quarter. So either the scale plane is a smaller share of the traffic than the arithmetic
assumed, or the wider read costs more in index work and in a loop that has to handle a partial block
than it saves in sectors. Either way the hypothesis is not supported, and the code was reverted.

The honest position is that 66 GB/s against a slice's 571 is still unexplained rather than explained
away. What to do next is measure where the kernel's time actually goes - with a clock around the
kernel or a profiler - instead of continuing to reason about where it should go.

## 8g3. The bandwidth arithmetic, and what it changes

The expert weights read per token are 2.14 GiB - 48 visits times 8 experts times both matrices - and a
token takes 34.5 ms, so the decode moves 66 GB/s. A 7g slice of this GPU has about 571 GB/s.

Roughly a ninth of the available bandwidth, which is what an uncoalesced read looks like: the gather
route reads its code plane eight bytes at a time with a lane stride, so a warp touches a 256-byte span
per group rather than pulling a contiguous run.

That reorders the next steps. Fewer launches would save the 6 ms a token of launch cost, but the
wasted bandwidth is the larger number and it comes from the access pattern rather than the count, so
coalescing the code-plane read comes first.

## 8g8. Which copies are actually removable

The rule from §8g7 needs a second half, and one experiment supplied it. Aliasing `attn_in`, `q_in`,
`k_in` and `v_in` onto whichever buffer the read wrote into removed sixty-eight copies and broke the
oracle: those four are fed from one source and then **written** by their own controllers, so the copies
were isolation, not decoration. Reverted.

So the rule is: remove a copy only when the destination is written once and read thereafter. Checking
that for the hidden-sized sites leaves few - `h` is written three times a visit, so its snapshot into
`h_in` is honest, and the rest are the same shape. The 895 that remain are near the floor for this
structure.

Which means the next lever is the operations rather than the copies: 681 `linear` calls a token, three
of them per visit being the query, key and value projections from one input, and two more being the
sink and gate projections from another.

## 8g12. The conv states want two buffers, not a copy

Each visit advances three convolution states through `state_scratch` and copies the result back into the
state array: 144 copies a token, about 4.6 KB each. The operation cannot write in place because it reads
the old state as its input.

The fix needs no new kernel. Allocate each per-visit state twice over and let the token position's parity
decide which half this step reads and which it writes; step p reads buffer p&1 and writes its complement,
nothing is copied, and reset zeroes both halves.

Not started, because the edit touches four allocations, three call sites and the reset, which is more than
belongs at the end of a session. The next round begins here.

## 8g11. The wattage says the GPU is idle

nvidia-smi during a benchmark: **120 W of a 500 W cap**, GPU-Util reading zero for most samples. The same
card on llama.cpp reaches 400 W.

Power draw is the cleanest signal this port has had, because it does not depend on clocks, load average
or what the host is doing: 120 W means the device is idle between bursts. Four hundred means it is
working. So the token is not limited by arithmetic, by bandwidth, or by the number of operations in the
way the counters measure them - it is limited by **how much of the time the GPU has something to do**.

That is consistent with everything measured before: 673 calls a token, each a few kilobytes, each so
short that the device finishes it and waits. The expert kernels are the only ones long enough to matter,
and they are about half the time.

The fix is therefore not "fewer operations" as an end in itself, but **keeping the device fed**: fewer
and larger launches, which is what the last few rounds did, and beyond that either a fused per-visit
kernel of the kind Qwen3.8 has, or more than one token in flight. The second is not available to a single
sequence; the first is the remaining work.

## 8g10. Where the remaining 30 ms goes

The budget adds up, for the first time, and it says the next lever is not the call count:

| part | estimate |
|---|---:|
| expert GPU work (2.5 GiB at the measured 155 GB/s) | 16.7 ms |
| 622 kernel launches at 4.5 us | 2.8 ms |
| 845 copies at about 2 us of driver time | 1.7 ms |
| **unaccounted, i.e. small-kernel GPU time** | **9.3 ms** |

Nine milliseconds in about six hundred kernels that each move a few kilobytes is fifteen microseconds
apiece - far more than the arithmetic in them. A hidden-sized kernel is nine blocks of 256 threads, so
it uses nine of seventy-eight SMs and then waits to be scheduled again.

That is the same shape of problem that the Qwen3.8 path solved by fusing several of these into one
launch, and it is where the next round has to go: the four controller projections, the three
convolution-state updates, and the norms around them are the candidates.

## 8g13. The closed operation budget

Every kind of call a step makes now has a counter, so this list is the whole of it:

| per token | count |
|---|---:|
| linear | 426 |
| routed expert calls | 162 |
| swiglu | 34 |
| accumulate | 0 |
| KV store | 51 |
| device copies | 564 |
| norms | 155 |
| **total** | **1392** |

Down from about 2,900 when this round of work started, with the oracle at 5 of 5 after every commit. The
three largest single reductions were the eight-expert gate/up and down fusions, the KV store's pitched
copy, and the message gate's norms.

Getting here took five rounds of discovering that the metric did not contain what a change removed - the
last one made a fusion *look* like a regression. The counter set is closed over the step now, which is
worth more than any single optimisation in it: without it the next round cannot tell a win from noise.

Decode rate at the last unobstructed measurement was 33.5 tok/s against llama.cpp's 51.4 on the same
slice. The card is now shared with another workload (289 W of a 500 W cap when idle for us), so further
rate claims have to wait for a quiet machine or for the 4080 SUPER numbers.

## 8g9. Where the operation budget ended up

Measured through this round of work, with the oracle at 5 of 5 after every step:

| quantity | start | now |
|---|---:|---:|
| kernels per token | 1531 | **622** |
| linear | 681 | 426 |
| routed expert calls | 816 | 162 |
| device copies | 997 | 845 |
| KV store calls | 771 | 51 |
| decode rate (graph, same load) | 26.5 | 32.3 - 32.8 tok/s |

The largest single reductions were the eight-expert fusions - gate/up with its activation in one launch,
then the down stage summed by one warp per output row walking the slots in order, which is bit-identical
to the eight launches it replaced - and the KV store, which was sixteen 256-byte copies a visit.

What remains is 426 line projections and 845 copies. The copies are now more numerous than the kernels,
and most of them are genuine: four attention streams that are fed from one buffer and then written by
their own controllers, and convolution states that are swapped rather than aliased. Removing those means
changing the data flow, not deleting a call.

## 8g7. The copies, which were never counted

A step issues 1531 kernels and **997 device copies**. Each copy is a driver call of a few hundred
bytes to a few kilobytes, so at roughly five microseconds apiece the copies are about 5 ms of a 34 ms
token - and they were absent from every account until a counter was put around them.

Thirty-six copy sites, eighteen of them moving a full hidden vector. Most are the same pattern: a
buffer is copied so that a later operation can read a familiar name. Three of them have now been
removed by pointing the later consumer at the producer's buffer instead (the KV store, the relative
states, the concatenated message gates), and each removal showed up as nothing in the counters until
the copies themselves were counted.

That is the pattern to keep applying: find a copy whose destination exists only to be read by the next
line, and delete both the copy and the name. It is the cheapest kind of change in this path - no
arithmetic moves, so the oracle either stays at 5 of 5 or the change was wrong - and it is now
measurable.

## 8g. Where the decode time actually goes

Measured, in this order, each one killing the previous theory:

| Probe | Result |
|---|---|
| skip the 48 routing readbacks | +2% |
| skip all 768 expert GEMVs | +21% |
| launch overhead, measured directly | 4.49 us per launch, 0.89 per graph node |
| host time vs wall time, with the readbacks | 42.14 vs 42.17 ms/token |
| host time vs wall time, readbacks removed | 40.89 vs 41.16 ms/token |

The last two are the answer. With every synchronisation removed the host still spends its time
queueing, and the wall time is the same number, so the token is **host-bound**: roughly 1,400
operations per token at 4.49 us of launch each is 6 ms, which leaves about 25 us per operation of
C++ work. That is the thing to fix, and it is not arithmetic, not bandwidth, and not synchronisation.

## 8f3. Where this port stands against the engine's architecture

Asked directly, so answered directly: the kernels are aligned and the driver is not.

**Aligned**: everything this port added below the model layer - the gather routes, the fused gate/up and
down kernels, the dual projection, the counting - lives in `src/ops` beside the operations every model
uses. There is no second implementation of a kernel and no fork of the op contract.

**Not aligned**: `ninfer::Engine` is bound to one model. Its translation unit names
`models::qwen3_5::PreparedPrompt`, `RetainedSessionSnapshot` and the rest directly, and nothing in
`src/runtime` defines a model interface to implement. So this port's `Session` - the thing that runs
forty-eight visits and decodes - is a **parallel driver** beside the engine rather than a model inside
it. It is why the model runs from a test binary and not from `ninfer` or `ninfer-serve`.

Two honest options:

1. **Give the engine a model interface** and let Inkling 2 implement it, the way the Qwen path does.
   That is the architecturally right answer and it is the one the repository's own rule asks for - no
   parallel authorities - but it means extracting an interface out of a six-thousand-line Qwen program
   and porting this model's driver onto it.
2. **Keep the standalone session** as the runtime for this model, documented as such, and revisit when a
   second loop model needs the same thing.

The first is the direction; the second is what exists today, and the difference should not be lost in a
summary that says "the model runs".


The obvious explanation for 29 tok/s is that this slice is slow. The same slice says otherwise: with
llama.cpp's own build and its own Q4_K_M of this model, from §8e, it decodes at 51.4 tok/s. Same
machine, same model, same hour.

So the hardware can do this model at 51 tok/s and this port does it at 29. That is a port that is
1.8 times behind, not a machine that cannot run it.

Worth also recording what this box *cannot* do, because it is a property of the environment rather
than of the code: the product engine (`ninfer::Engine`, and therefore the CLI and the server) needs
ffmpeg and libcurl development headers, and this container has neither - so the Qwen3.8 artifact cannot
be benchmarked here through the engine path at all. The model runtime that this port uses builds
without them.

## 8f. How far ahead this engine is supposed to be

The engine has a recorded head-to-head on a card of this class (`docs/llamacpp-comparison.md`,
Qwen3.8-27B, RTX 4090, same day, both engines): decode without speculation is 50.5 tok/s against
llama.cpp's 45.9 at depth 0, and 39.6 against 33.1 at 128K. With MTP it is 148.6 against 118.8.

So the honest expectation for a model of this shape is **ten to twenty-five percent ahead of
llama.cpp, not two to three times**. That is the number to hold this port to, and it makes the
current Inkling 2 result - 15 to 24 tok/s against 51.4 on the same slice - a three-fold regression
from where the engine normally sits rather than "a new model that needs tuning".

That gap is large enough that the cause is structural, and the three candidates are the ones §8e
lists: 48 routing readbacks per token where the Qwen path has none, roughly a thousand kernel
launches per token, and an expert GEMV on a runtime-extent route. The first is the only one that is
a different *shape* of execution rather than a slower version of the same one.

## 8e. The baseline to beat

`/workspaces/llama.cpp` carries a built `llama-bench` and an `Inkling-2-Mini-mainline-Q4_K_M.gguf`
(15.46 GiB, 26.07 B parameters), so the reference performance for this model on this machine is
measurable rather than guessed:

```
llama-bench -m Inkling-2-Mini-mainline-Q4_K_M.gguf -p 0 -n 32
tg32    51.42 +- 5.66 t/s      (CUDA, this 78-SM slice)
```

This engine decodes the same model at 15 to 24 tok/s, so it is two to three times behind, and that
is the number to close. The differences worth attacking, largest first:

1. **48 device-to-host routing readbacks per token.** llama.cpp keeps the selection on the device
   and gathers expert weights by index. Here the experts are separate artifact objects and the
   selection decides host-side weight handles, so every visit drains the queue. Removing this needs
   a device-side pointer table and a gather route, which is the same shape llama.cpp uses.
2. **Roughly a thousand kernel launches per token**, because the chain is a sequence of small ops
   rather than one graph.
3. The expert GEMV itself, which is a runtime-extent route rather than a shape-tuned one.

The GGUF is Q4_K_M at 15.46 GiB against this artifact's 18.33 GiB of NVFP4, so the comparison is
between two four-bit-ish recipes rather than like for like; it is still the right target, because
the user cares about tokens per second on the same question.

## 8b2. When this machine cannot measure anything

The shared slice used for development is not always usable for timing, and it is worth writing down
how bad it gets so nobody reads a number from it without checking first. Under a load average of
about 40, two runs of the *same* binary, back to back, gave 9.1 and 3.8 tok/s - a factor of 2.4 - and
a host time of 109 ms against 262 ms for identical work.

The rule that follows: check `uptime` before timing, take the host/wall ratio as the load-independent
signal, and never conclude anything from a single run on this machine. Several rounds of this port
were spent on changes that were then reverted for exactly this reason, which is cheaper than keeping
a change nobody can justify.

## 8d. Decode rate, and how it moves

`--generate` reports the two numbers separately, because averaging them is misleading:

- **warm-up**, the first step: 1 s to 8 s here, for the CUDA context, the module load and the first
  touch of every weight. It is a one-off and does not belong in a rate.
- **steady state**: 41 ms to 65 ms per token on the 78-SM slice used here, i.e. 15 to 24 tok/s, with
  the spread being the shared machine rather than the model. Argmax is 0.04 ms of that.

The benchmark's figure (16-18 tok/s) is a *pipelined* number - many steps queued and synchronised
once - while generation synchronises per token. They agree here, and neither is the wrong one, but
they measure different things and should not be quoted interchangeably.

## 8c. The session reproduces the reference

`execution/session.cpp` runs the 48 visits with a cache per visit and reproduces the recorded
reference logits to 0.14 absolute with the same five top candidates. On one 78-SM slice of an H20
it decodes at 21.4 tok/s with a 4096-token cache.

Getting there cost one bug worth recording: the message gate concatenates two normalised hidden
states before its 64-wide projection, and that scratch was a **hidden-sized** buffer. Writing
2 * hidden into it overran the allocation and poisoned whatever came after it in the arena. Pass 0
(no messages) was perfect while every later pass drifted, which read like a cross-pass logic error
and was really a one-line size mistake.

## 8b. What a decode step costs, measured

Single token, NVFP4 pass-through artifact, one 78-SM slice of an H20 (a weak card, so these are
ratios more than absolutes):

| | |
|---|---|
| whole 48-visit chain | 19.5 tok/s (51 ms) |
| attention half only | 105 tok/s (9.4 ms) |
| 768 expert GEMVs (2304x2304) | ~42 ms |

The expert GEMVs move 2 GB of codes and scales per token, which at this card's bandwidth would be
about 2 ms, so there is roughly a factor of twenty left in that kernel. The rest of the chain is
already within a factor of a few of what the op count suggests.

Two measurement traps cost a day of guessing here, both worth remembering:

- **`cudaMalloc` in the layer loop.** A per-visit allocation and free synchronise the device, 96
  times a token. Removing it took the chain from 5.5 to 19.5 tok/s.
- **Ablations taken while a constant dominates are noise.** Before that fix, ablating the whole
  mixture changed the total by nothing, and the numbers were read as "the experts are free". They
  were not; every ablation carried the same allocation cost, and the differences were noise around
  a constant.

## 3b. The layer pipeline

Read from the reference, because it is not a plain pre-norm block and the executor has to match it
visit by visit. There are four layer classes, chosen by `(sliding?, inside loop core?)`:

| Layers | Class | Inputs to attention |
|---|---|---|
| 0-5, 18-23 sliding | SlidingDecoderLayer | one stream, projected to q, k and v |
| 5, 11, 17, 23 | FullDecoderLayer | one stream |
| 6-17 sliding | SlidingLoopDecoderLayer | three streams, each fed by its own message gate |
| 6-17 global | FullLoopDecoderLayer | three streams |

Per visit, with `h` the incoming hidden state and `B` the block-state stack:

```
boundary = (layer % 6 == 0)              # is_block_boundary
attn_in  = layer > 0 ? atend(learned_query, [B, h]) : h     # block attention residual
attn_out = attn(ln_attn(attn_in))
attn_out = sconv_attn(attn_out)                              # conv_idx 2
residual = boundary ? attn_out : h + attn_out                # a boundary starts a new block
mlp_in   = atend(learned_query_mlp, [B, residual])
mlp_out  = moe(ln_mlp(mlp_in))
mlp_out  = sconv_mlp(mlp_out)                                # conv_idx 3
h_out    = residual + mlp_out
if boundary: B = cat(B, h)                                   # the pre-layer state joins the stack
```

`atend` is the block attention residual: one learned query per layer (fp32 `[hidden]`, or `[3,
hidden]` for the loop-layer classes) attending non-causally over the normalized stack
`[B, partial]` with scale `1/sqrt(hidden)`. That is what `attn_res.attn_queries.*` are, and why
four of them carry three rows.

Consequences the executor cannot abstract away: the residual chain **restarts** at every sixth
layer instead of accumulating; `block_states` grows by one entry per block; and inside the core,
q, k, v and the MLP input are four separate streams, each read from the previous pass by
`message_input` and blended by its own `message_merge` before the norm.

## 4. Attention

Per layer, from the checkpoint's tensor names:

| Tensor | Meaning |
|---|---|
| `wq_du`, `wk_dv`, `wv_dv`, `wo_ud`, `wr_du` | attention projections factorised through shared "u"/"d" and per-head "d"/"v" spaces, plus a separate relative-bias path |
| `rel_logits_proj.proj`, `rel_l2_clip.scale`, `d_rel = 16`, `rel_extent = 1024` | learned relative-position bias of rank 16 over a 1024-token extent |
| `q_l2_clip.scale` | L2 clip on q (threshold 22.627…), K clipped separately |
| `attn_sink_proj.weight`, `gate_attn.weight` | attention sink logit and output gate |
| `k_sconv`, `v_sconv`, `attn_sconv`, `mlp_sconv` | short convolutions, kernel 4. A sliding-window layer filters k and v directly (`[8x128, 1, 4]`); a global layer runs the same convolution in a learned 8x4 space (`weight_proj [32, 2304]`) over a fixed 4-tap filter (`static_w [4, 1024]`), so its layer has no `attn_norm` but three input norms instead |
| `use_value_norm_no_gamma = true` | value normalisation without a learned gain |
| `log_scaling_floor = 8192`, `alpha = 0.1` | log-scaled attention on the global layers |
| `final_logit_softcapping = 30.0` | logit soft cap at the head |

The reference implementation settles the mathematics (read from `transformers` 5.24,
`modeling_inkling2_loop.py`), so the kernels do not have to guess:

```
q = Wq x                                                # [heads, 128]
q *= scale_q[heads, 128]                                # fp32 learned per-element scale
q *= clamp(qk_clip / (||q|| + 1e-6), max = 1)           # L2 clip, per head
q *= tau(pos)   (global layers only)                    # tau = 1 + alpha * ln(max((pos+1)/floor, 1))

k = Wk x -> short conv (kernel 4) -> per head -> same L2 clip (no learned scale)
v = Wv x -> short conv (kernel 4) -> per head -> unweighted RMS norm (eps, no gamma)

rel = Wrel x -> [heads, 16] -> * scale_rel[heads, 16] -> L2 clip(rel_clip) -> * tau(pos)
bias[head, i, j] = dot(rel[head, i], proj[:, pos_i - pos_j])   for 0 <= distance < 1024, else 0
sink[head] = attn_sink_proj x                                  # one extra logit, value zero
out = softmax([q k^T / sqrt(128) + bias + mask, sink])[..., :-1] @ V
out = (out * sigmoid(Wgate x)) @ Wo                            # gate computed in fp32
```

`wr_du` therefore produces the per-head relative state (`heads * d_rel` rows), `rel_logits_proj`
is the `[16, 1024]` bank of bias-versus-distance profiles, and the two clip scales are the learned
per-head multipliers applied before the clip.

`ninfer::ops::block_residual_read` is the block attention residual: one or three learned FP32
queries attending non-causally over the normalized `[block_states, partial]` stack with scale
`1/sqrt(hidden)`. It is the same three lines of math as the causal attention but over a stack of
one to a few vectors, so it is its own Op.

`ninfer::ops::head_rmsnorm` is the value path's normalisation: the same reduction as `rmsnorm`
without a gain vector, which the checkpoint declares with `use_value_norm_no_gamma`.

`ninfer::ops::dynamic_short_conv` covers the global-attention variant: the four taps are a static
per-channel filter plus a per-token filter projected from the layer input and shared inside a
key/value head, and the residual is divided by `sqrt(2 * kernel + 1) = 3` rather than `sqrt(5)`.

`ninfer::ops::causal_dconv_residual` implements the static short convolution the sliding-window
layers use for k, v and both residual streams: a depthwise causal width-4 filter, the input added
back, and the sum divided by `sqrt(5)`. The three-value window lives in a per-visit state buffer,
which is why the loop's 48 visits need their own conv states rather than 24.

`ninfer::ops::scaled_l2_clip` implements the projection-side normalisation for both the query
path (head_dim 128, threshold 22.627) and the relative-state path (head_dim 16, threshold 8), with
the learned per-head scale applied before the clip and the log-scaled `tau` applied after it on
global layers. Its oracle errors sit exactly on the BF16 store quantum (2^-7 at magnitude one), so
the arithmetic is FP32-clean.

The decode half of this is implemented as `ninfer::ops::relbias_attention`: one warp per query
head, online softmax over the visible cache, the sink entering the normaliser as one extra logit
with no value, and the gate folded into the output. It matches an FP64 oracle to 1.6e-3 on full,
sliding-window, single-token and 2048-token caches whose far distances fall outside the bias
horizon.

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

Routing is sigmoid-scored, not softmax-scored (confirmed from the reference):

```
logits  = x . gate_weight                     # FP32, no bias on the logits
scores  = sigmoid(logits)
select  = top-8 of (scores + gate_bias)       # the bias only steers selection
weights = scores[select] / (sum + 1e-20) * 8  # renormalised, then scaled by route_scale
```

`mlp.gate.bias` is that selection bias, not a logit bias, which is why it is stored in FP32 while
the gate matrix is BF16. Expert outputs accumulate in FP32.

The expert activation is a softcapped SwiGLU where the sigmoid sees the **raw** gate:

```
ideal = tanh(gate/4)*4 * sigmoid(gate) * tanh(up/25)*25
```

`ninfer::ops::softcapped_swiglu` implements it over the fused `[2 * inter, tokens]` projection and
matches an FP64 oracle to 2.9e-3 on the real 1152 width.

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
after its source path (`text/layers/6/attn/wq_du`, `text/layers/6/mlp/experts/7/w13`). The C++
model module of stage 2 binds those slots to the mathematics; nothing here presumes how the u/d
factorisation composes.

Two naming facts the loader depends on: the trailing file suffix is dropped (`wq_du.weight` becomes
`text/layers/6/attn/wq_du`), and layers are not zero-padded. The checkpoint also stores four
`attn_res/attn_queries` rows as `[3, 2304]` instead of `[2304]` — layers 4, 10, 16 and 22, i.e. the
layer before each global-attention layer. The loader encodes that rule; its meaning is still open
and belongs with the block-residual mathematics in stage 2.

## 8b. Reference oracle (available)

The checkpoint is only published quantised, and `transformers` has no quantiser for its `modelopt`
layout, so the reference cannot load it directly. `tests/inkling2/materialize_bf16.py` dequantises
it once through the same `expert_matrix` the converter uses and writes a plain BF16 checkpoint;
`tests/inkling2/dump_visits.py` then runs the reference and captures all 48 layer visits.

Measured on one H200-class card (2026-11-23):

| Check | Result |
|---|---|
| Load | `Inkling2LoopForConditionalGeneration`, 49.2 GiB VRAM, text stack at `model.language_model` |
| Loss on a 38-token English passage | **1.969** |
| `The capital of France is` (5 tokens) | top-1 ` Paris` at 27.29 |
| Visits | 48, in the order prelude 0-5, core 6-17 three times, coda 18-23 |

The BF16 materialisation is 52.8 GB and takes ~32 minutes; it exists to give stage 2 a target, not
as a product. Anyone re-running it must expect the `causal_conv1d` fallback warning: the reference
convolution is then pure PyTorch and slower, which is fine for a forward pass.

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
| 2 | text-only, greedy, 8K: all 48 visit hidden states within the tolerance the vLLM write-up measured (≈ FlashAttention-vs-PyTorch divergence), short prompts token-identical to the reference. **Schedule**: `schedule.cpp` builds the 48 visits and the 24 message links, checked against the
visit order the reference implementation actually produced.

**Layer 0 parity**: the whole layer reproduces the reference for one token, 1.6e-2 of BF16 storage
on a hidden state of magnitude 1.5, with every stage checked on the way:

```
rmsnorm(attn_norm)                       max|err| 0
q after scale+clip                       max|err| 3.9e-3
k after conv+clip                        max|err| 3.9e-3
v after conv+rmsnorm                     max|err| 4.9e-4
sink                                     max|err| 0
self_attn output (after Wo)              max|err| 2.0e-3
block read + post_attention_layernorm    max|err| 1.0e-2
whole layer                              max|err| 1.6e-2
```

That exercises the block-boundary path: layer 0 does not add a residual, it pushes the pre-layer
state into `block_states` and reads the MLP input back out with a learned query.

**MLP sub-block**: matches: the router selects the same eight experts
(its order differs, which the reference allows with `sorted=false`) with weights agreeing to 6e-7,
and the accumulated expert output lands within 1.8e-2 of the reference, which is BF16 storage at
that magnitude over 8 x 2304-term sums.

Layer 0's attention sub-block reproduces the reference for one token, every
stage checked separately: `attn_norm` exact, q 3.9e-3, k 3.9e-3, v 4.9e-4, sink exact, the attention
output 2.0e-3 and the post-convolution output 3.9e-3 — all at BF16 storage rounding.

**Loader**: `load.cpp` derives all 6790 text slots from the config and validates the built artifact end to end — 7212 bindings, no missing slot, no unexpected one, every shape and stored format checked.

**Started**: `config.cpp` parses and validates the text component (`visits = 48`, `kv_caches = 48`, `global_layers = 4`, core `[6,18)` × 3, `q_width = 3072`, `kv_width = 1024`, 128/8 experts, DSpark taps 5/23/41/48, vision depth 23); `load.cpp` derives all 6396 text slots from that config and checks every binding's shape and stored format against the artifact, rejecting missing and unexpected parameters. Registry recognises `Inkling2LoopForConditionalGeneration` / `inkling2_loop_text`. No materialisation, no kernels yet |
| 3 | 48 KV + conv-state caches, 128K context on the 4080 SUPER, prefix reuse |
| 4 | vision + DSpark; GSM8K and 32K perplexity within noise of the reference |
| 5 | serving parity: chat template, reasoning effort, tool calls, `/v1/messages` |

## 10c. The gate/up row interleave (artifact bug, fixed)

The checkpoint stores the fused gate/up expert matrix **row-interleaved**: stored row `s` holds
logical row `s/2` for even `s` and `half + (s-1)/2` for odd `s`. Passing the stored order straight
through produced an artifact whose expert `w13` matrices were a row permutation of the reference's,
while the attention weights, the router and `w2` were all correct — so layer 0's MLP looked right
in a fixture test that fed the reference's own dequantised experts, and would have been wrong the
moment the artifact's experts were used.

Found by comparing the decoded payload against the reference-dequantised matrix row by row:

```
stored row 0    -> logical row 0      maxdiff 2.2e-4
stored row 1    -> logical row 1152   maxdiff 1.6e-4
stored row 2    -> logical row 1      maxdiff 2.2e-4
stored row 2303 -> logical row 2303   maxdiff 1.6e-4
```

`w2` is stored in logical order, so the mapping applies to `w13` only. The converter now remaps
the rows for that tensor alone.

## 10b. NVFP4 storage, verified

The NVFP4 container this repository reads (and this converter writes) is:

```
code plane   n*k/2 bytes            two E2M1 nibbles per byte, row-major along k
scale plane  at align_up(n*k/2,256) n*k/16 bytes, one E4M3 scale per 16 values
payload     both planes, then one float  (n*k/2 + n*k/16 + 4 for 2304x2304 = 2_985_988)
```

The scale plane is tiled `BlockScaleK16M128x4`, so its address for (row, group) is not
`row*(k/16)+group`:

```
m_tile=row/128; inner=row%128; tile=group/4; lane=group&3
offset = (m_tile*(k/64) + tile)*512 + (inner&31)*16 + (inner>>5)*4 + lane
```

`Weight::weight_scale_divisor` carries `denominator / amax` for that matrix, i.e. the dequantisation
**divides** by it: `w = e2m1(code) * e4m3(scale) / divisor`. `Weight::input_scale_divisor` must be
1 for this checkpoint, and nothing set it: every NVFP4 route rejected the converted weights with
"invalid NVFP4 weight" until `materialize` passed 1.0. That is fixed and unrelated to which route
runs.

There is no tuned NVFP4 route for 2304x2304 or 2304x1152, so `launch_nvfp4_generic_gemv` serves
them: one warp per output row, any extents, single token. Against the reference-dequantised
experts of layer 0 it lands within 4e-3 (BF16 storage of a 2304-term sum):

```
expert 22 (slot 0)  max|err| 2.9e-3
expert 24 (slot 1)  max|err| 4.0e-3
expert 63 (slot 2)  max|err| 3.9e-3
expert 93 (slot 3)  max|err| 3.0e-3
```

That check only passed after the row-interleave fix in 10c: the first version of this route read
the artifact's experts in the order the converter had written them, which was a row permutation
of the truth, and it disagreed by 1e26 rather than by a rounding error. The inflated error was
the signal that the *data* was wrong, not the kernel's arithmetic.

## 10a. The shape table and the generic route

Every existing linear route is templated on a compile-time geometry and covers exactly one
`(n, k)`: bf16 has three, q4 has twelve, and a shape outside the table throws. Inkling's widths
(3072x2304, 1024x2304, 2304x3072, 2304x1152, 384x2304, 2304x2304) are in none of them.

`launch_bf16_generic` is the runtime-extent fallback: one output element per thread, the activation
tile staged in shared memory. It is several times slower than a tuned route and exists so a model
whose shapes have not earned a kernel can still be checked for correctness. `bf16_dispatch` uses it
when the table misses, so the layer can be assembled and compared against the oracle before any
shape tuning starts.

## 11. Where the code goes

Additive only; the shared files are `src/models/registry.cpp`, `src/models/CMakeLists.txt`,
`src/ops/CMakeLists.txt` and the root `CMakeLists.txt`, and each gains lines, not restructuring.

```
src/models/inkling2/          config (done), load, program, execution, state, frontend
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
