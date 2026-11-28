# KVMem 式长上下文：移植计划

本文把 [long-context-workspace.md](long-context-workspace.md) 第 10 节已经证实的方向落成可执行的工程计划：在本仓库自己的引擎里，把"设备窗口 + 主机层 KV + 块级召回"做出来，从而把单会话可寻址上下文从实测的 262,144 tokens 推到 60 万以上。

实测前提（均在本机 4080S 32G 上取得，见 long-context-workspace.md 第 9、10 节）：

- 原生路径的墙是**位置容量 262,144**（int8 KV 可以达到，bf16 KV 只到 208,512），与显存无关。
- 参考实现（kvmem/kvmem 的 NInfer 集成）在 int8 KV 下实测 **637,768 tokens、4/4 召回、prefill 1.22k tok/s、decode 34.1 tok/s**，冷启 TTFT 8m44s，第二轮 1.6 s。
- 窗口化视图与全量视图在 183,936 tokens 上**逐字节一致**（含逐字引用文档开头），说明 GDN 状态留在前沿的近似在观测层面不可见。
- 召回精度只由**预算**决定，且失败是静默的：预算 36,864 时 4 个事实掉 1 个，没有任何报错。

所以剩下的不是可行性问题，而是工程量问题。

## 1. 目标与非目标

**目标（MVP）**

1. 单会话、单请求边界内的召回：历史超过设备窗口时，把未驻留的 KV 块放在主机内存，按块选回。
2. 原生能力零回退：上下文 ≤ 262,144 时不启用召回，输出与现在**逐字节一致**；短上下文套件保持 bit-identical。
3. 召回质量可验收：needle retainer 在 30 万 / 50 万 / 80 万 tokens 上 4/4，且与全量视图在可对照长度上逐字节一致。
4. 保持本 fork 的 prefill 优势：同长度 prefill 不低于当前水平（1.37k tok/s @183,936 bf16）。

**非目标（本计划不做）**

- NVMe 层。1M 以内主机内存够用（bf16 约 33 GiB），NVMe 只增加搬运而不改变正确性。
- 多会话/多 lane 的工作区共享与抢占。当前产品模型是无抢占 FIFO，召回点固定在请求边界。
- 学习式或跨层的复杂选择策略。先用与参考一致的 Mean-K + 页 softmax + 一跳图传播。
- 原生窗口的 RoPE 扩展。位置容量是模型属性，不靠改位置编码解决。

## 2. 与现有代码的对应关系

| 需要的部件 | 本仓库现状 | 结论 |
|---|---|---|
| 物理 KV 池、页组、block table、consumer view | `src/core/paged_kv_cache.h`：`PagedKVLayerView{k_pages,v_pages,block_table,...}`，页映射自由 | 直接复用 |
| **主机层 KV 存储（pinned）** | **已存在**：`src/core/host_kv_arena.{h,cpp}`（页布局、传输工作量规划）+ `src/models/qwen3_5/program/storage/host_kv_store.h` 的 `HostKVExtentStore`（按 extent 持有有序逻辑页，明确"无 checkpoint/retention/scheduling 策略"） | 零新增，只接策略 |
| 上下文缓存与候选/保留、StateImage 迁移 | `docs/maintainer/resource-scheduling-and-context-cache.md` | 复用，新增"块级"候选类型 |
| GDN 状态提交 | `docs/maintainer/replayssm-gdn.md` | 复用；召回不触碰 GDN |
| 注意力入口 | `ordinary_decode_batch(ids, cache_positions, rope_positions, kv_table_rows, linear_state_slots)`，两个位置张量**已经分开** | 见 3.3：这是最大的省力点 |
| 注意力算子 | `causal_softmax_attention(q, k, v, positions, valid_columns, kv_table_rows, ...)`：`positions` 只决定"读 cache 行 [0,p]"；RoPE 由 `text_rope(rope_positions, …)` 单独完成 | **不需要新 kernel** |
| 主机搬运 | 已有 pinned staging + `TransferWork` 规划 | 复用 |

两个关键结论，都是从现有接口读出来的：

1. **逻辑前缀单调，但页映射自由**（`softmax_attention.h` 第 118-121 行）。所以召回视图必须把选中的块**压实成新的连续前缀**，而不是在中间留洞。
2. **RoPE 位置与 cache 位置在本仓库已经是两个张量**（`text_rope(rope_positions, …)` 与 `causal_softmax_attention(..., positions, ...)` 分别调用，见 `src/models/qwen3_5/execution/text.cpp:353,376`）。因此可以按原始位置旋转 q/k、按压实后的位置限定读取范围，相对相位天然正确——参考实现需要 `baked_pos` 元数据与 re-RoPE，是因为它把 RoPE 烘进了 cache 写入；我们没有这个问题。

## 3. 模块拆分

### 3.1 规划层（纯函数，无设备依赖）

新增 `src/core/kv_workspace.h` / `.cpp`：

```cpp
struct KvBlock { std::int32_t id; std::int32_t orig_pos_start; std::int32_t tokens;
                 Tier tier; std::int32_t slot; };
struct WorkspaceBudget { std::int32_t selected_tokens, reserve_tokens, device_tokens; };
struct KvRemap { std::int32_t block; std::int32_t from_base; std::int32_t to_base; std::int32_t tokens; bool skip; };
struct WorkspacePlan { std::vector<KvBlock> stage_in, stage_out; std::vector<KvRemap> remaps; std::int32_t visible_tokens; };
WorkspacePlan plan_workspace(const WorkspaceBudget&, span<const KvBlock> resident, span<const KvBlock> scored);
```

- 块大小固定为页组的整数倍（页组 64 tokens，块取 128）。
- 不变量：先 stage-out 后 stage-in（有界池）；选中块按 id 升序压实，从 0 开始；sink 与 recent 永不被逐出。
- 纯逻辑，单元测试覆盖，先写它。

### 3.2 主机层块存储

无需新增：`HostKVArena` 提供 pinned 页布局与传输规划，`HostKVExtentStore` 提供"按 extent 持有有序逻辑页"。本计划只在其上增加块级簿记（哪个块在哪个 extent），并把 `TransferWork` 交给现有调度。验收：spill+restore 往返后 K/V 逐字节不变（fp16 精确，不需要容差）。

### 3.3 视图组装（不新增算子）

现有路径已经能把 q/k 按 `rope_positions` 旋转，并把 `positions` 单独传给注意力作为读取范围。因此视图只需要三件事：

1. **压实的 block table**：选中块按原始顺序写入设备窗口 0..visible-1，`kv_table_rows` 指向这些页；
2. **两份位置张量**：`rope_positions` = 原始位置（保证相位正确），`positions` = 压实位置（保证读取范围正确）；
3. **把新 token 的 K/V 追加在压实后的前沿**，其 RoPE 仍用原始位置。

掩码不需要新机制：可见集合就是被选中的块，压实后天然是单调前缀；sink 与 recent 通过"永远保留"实现。

**已落地（第 1 步的逻辑半边）**：`src/core/kv_view.{h,cpp}` 的 `build_kv_view(pages, {block_pages, sink_pages, recent_pages}, selected_blocks)`。它只做页序决策：块按 `block_pages` 切分（尾块可短），选中块 ∪ sink 前缀 ∪ recent 尾 按时间序保留，未命中者丢弃，返回执行页序与可见页数。页保持**物理身份**不变，所以写进 cache 的 K 仍然带着写入时的相位——这正是 §10 里"不需要 re-RoPE"的代码化表述。测试 `tests/test_kv_view.cpp` 覆盖身份映射、中段淘汰补洞、sink/recent 覆盖其所在块、短尾块、选择乱序与重复、以及六类非法几何。

剩下的真接线（写 `kv_table_rows`、两份位置张量、prefill 增量索引、请求边界上的工作区归属）都在 `qwen3_5/program` 的 decode/prefill 路径上，属于同一处改动。

工作量因此从"一个新 kernel + 逐字节一致性验证"降为"组装器 + 表构造 + 一致性验证"。

### 3.4 Mean-K 索引与打分

新增 `include/ninfer/ops/kv_block_pool.h` 与 `src/ops/{launcher,wrapper}/kv_block_pool.*`（已落地，纯数学、无引擎类型）：

- `kv_block_mean_keys(keys[D,H,T] bf16, block_tokens, means[D,H,B] fp32)`：块内均值，FP32 累加、除以精确块长。入参是**旋转前**的 K，所以天然与位置无关，不需要去 RoPE 算子。
- `kv_block_scores(query[D,Q] bf16, means[D,H,B] fp32, scores[B] fp32)`：GQA 分组内把 query head 平均成 KV head 的查询向量，与块均值做余弦相似度、再按 KV head 平均。softmax 与 top-k 留给 3.5。
- `kv_block_accumulate(keys[D,H,T] bf16, block_tokens, first_position, sums[D,H,B] fp32, counts[B] i32)` 与 `kv_block_finalize(sums, counts, means)`：prefill 是一段一段进来的，块与段的对齐关系不固定（256 的 chunk 对 128 的块正好两段，但末段与任意 chunk 大小都不保证），所以设备侧只累加"和 + 每一块的真实列数"，均值在需要时再除。`counts` 由独立内核按列数累加（不是按线程），因此一块跨两段也不会重复计数。
- 对应的 CPU 版索引是 `src/core/kv_recall_index.{h,cpp}`：按列喂 pre-RoPE 的 K，只对**完整块**产出均值——末段不完整时就该少一块，而不是拿半块的平均去参与打分。它的发布布局 `[D,H,blocks]` 的 stride 依赖块数，所以均值整体物化、块完成时只失效（`cc172215`，测试当场抓到了按块填充导致的错位）。

测试 `tests/ops/test_kv_block_pool.cpp` 用独立 FP64 oracle，覆盖 D=8/16/64/256、GQA 2/4/6 倍、含尾部不完整块、全零块（相似度定义为 0）与真实几何（head_dim 256、4 KV heads × 24 query heads）；增量路径另跑 1024 列 / 128 块 / 256 列 chunk 与"3 列 chunk 对 4 列块"的跨界组合。"块均值"的存储量：16 层 × 4 heads × 256 dim × 2 B × 块数，637k tokens / 128 = 4,976 块 → 约 65 MiB。

索引仍需在 prefill 过程中**增量**构建（不能等 prefill 结束再扫，否则块已被搬走），这部分属于 3.3 的接入点。

内存：16 层 × 4 heads × 256 dim × 2 B × 块数；637k tokens / 128 = 4,976 块 → 约 65 MiB，可忽略。

### 3.5 选择策略与调度接入

- 打分聚合：层与 head 取平均，查询取最后一条 user 消息（上限 512 tokens，与参考一致）。
- 先实现"只选 top-k、不做图传播"的最小版本，用验收数据决定是否要传播（参考实现用 alpha=0.2 的一跳传播，收益未在本卡验证）。
- 接入点：请求边界（prefill 之后、decode 之前），与 FIFO/无抢占兼容；不改 `Engine` 的并发语义。

**已落地（策略半边）**：`src/core/kv_workspace.h` 的 `select_kv_blocks(scores, page_count, {block_pages, sink_pages, recent_pages, budget_pages})`。先锁定覆盖 sink 前缀与 recent 尾的块（它们的真实页数计入预算），再按分数降序装填剩余块，同分取较小 id；只在整块页数放得下时装入，短尾块按自身页数计费；mandatory 已超预算或一块都放不下时抛 `std::runtime_error`，不静默缩小请求。测试覆盖预算三档、同分、短尾块计费与四类非法几何。

## 4. 实施顺序与工作量

| 顺序 | 工作 | 估时 | 结束时能验证什么 |
|---:|---|---:|---|
| 1 | 3.1 规划层 + 单元测试 | 2-3 天 | 预算/压实/不变量的纯逻辑正确 |
| 2 | 3.3 视图组装（复用 3.2 的存储）+ 与全量路径逐字节一致性 | 4-5 天 | 窗口化不改变 ≤262k 的行为 |
| 3 | 3.4 索引 + 3.5 最小选择（top-k，无传播） | 4-6 天 | 30 万 / 50 万 / 80 万 needle 4/4 |
| 4 | 端到端验收与调参（预算、块大小、sink/recent） | 3-5 天 | prefill/decode 不回退，失败模式被 retainer 捕获 |

合计 2-3 周。前三步都可以在开关后面推进、不改变默认行为；第 3 步起才需要真实长上下文数据。

## 5. 验收

1. **零回退**：短上下文套件在开关关闭时 bit-identical；开关打开但无逐出时也与今天一致。
2. **等价性**：≤262,144 tokens 时，窗口化路径与全量路径输出逐字节一致（用第 10 节的 fidelity 探针，`codes` 与 `first_lines` 两个）。该探针已固化为 `tools/bench/run_kv_recall_fidelity.py`：同一 prompt、`temperature 0`、两个端点，逐字节比较生成文本，并对 `codes` 探针额外要求 4 个标记全部命中；退出码即判定，报告写 JSON。
3. **容量**：30 万 / 50 万 / 80 万 tokens 上 needle 4/4；预算是记录项，不是调参项——预算不足必须被 retainer 判失败，而不是被"答案看起来合理"放过。
4. **性能**：同长度 prefill 不低于今天；decode 不因视图间接寻址下降超过 5 %。
5. **内存**：主机 pinned 用量随历史线性增长且可预测，1M tokens 在 64 GiB 机器上留有余量。

## 6. 风险

| 风险 | 现状 | 处理 |
|---|---|---|
| GDN 前沿状态近似 | 已实测不可见（第 10 节） | 保留为常态验收项，不额外设计 |
| 注意力算子的间接寻址开销 | 未测 | 第 2 步先做一致性，再测 T=1 与长上下文两条路径 |
| 预算策略选不中需要的块 | 已实测会发生（36,864 掉一个事实） | 用 retainer 定基线预算，而不是猜 |
| 与现有上下文缓存/StateImage 的所有权冲突 | 未验证 | 第 2 步先做纯存储视图，不接管调度；第 4 步再接 |
| 工程量 | 参考实现 397 文件 / +32k 行；本计划只取机制 | 主机层与页表已存在，视图不需要新 kernel，剩余主要是索引与策略 |

## 7. 接线规格（读码得出，尚未实施）

这一节记录把视图接进主线的精确位置，全部来自读码，未改一行代码。三处关键事实：

**7.1 视图在引擎里已经存在，只是成员表固定为 0..frontier-1。**
注意力读的是 `KVAddressSpaceStore` 的 `membership(address, page)`，`page < address.page_count`，由 `publish_membership`（`src/models/qwen3_5/program/storage/kv_store.h:1846`）按顺序发布成执行表。所谓"压实视图"就是**换一份 membership、把 page_count 变小**——不需要新 kernel，也不需要新表格式。地址结构体在 `kv_store.h:1739`（`page_count / committed_frontier / checkpoint_frontier / reservation / row`）。

**7.2 两个位置已经在 decode 入口分开了，而且只有一个标量偏移。**
`src/models/qwen3_5/program/decode.cpp:325-328`：

```cpp
ordinary_host_ingress->cache_positions[row] = checked_i32(frontier, ...);          // 执行空间里的下标
ordinary_host_ingress->rope_positions[row]  = checked_i32(frontier, ...) + sequence.rope_delta;  // 旋转相位
```

**这是整条路上最省力的一点**：被丢掉的历史全部在 frontier 之后不会回来，所以新 token 的原始位置恒等于 `压实后 frontier + 丢掉的总 token 数`——一个标量。也就是说现有的 `rope_delta` 足够表达压实视图，**不需要逐 token 位置张量，也不需要 re-RoPE**；cache 里的 K 当初写入时的相位原样保留，读回来照旧正确。这与第 10 节实测"按原始位置旋转 + 按压实位置限范围"的结论是同一条。

**7.3 用"广义 fork"而不是"原地重发布"。**

最初的写法是在活地址上做一次原地 `republish_compacted`。读透了 `KVPrefixForkReservation`（`kv_store.h:131-182`，持有 `frontier_`、`full_pages_`、`page_reservation_`、`row_`）与 `commit_prefix_fork`（`:1135-1195`）之后，更好的形状是**把前缀 fork 的"前缀"换成"显式页序"**：

- 前缀 fork 已经做完了我们要的全部动作：逐个 `retain_reference(page,false)` + `protect_coverage`、`tables_->publish(row, 0, scratch)`、给源地址的对应页记 active reference、必要时搬尾页 —— 只是它读的页序是 `membership(source, page)` 的前 `full_pages_` 个。
- 所以新增的是一个 `KvCompactedForkRequest{source, ordered_pages, entitlement, frontier}` 与一个 `commit_compacted_fork`，提交时按 `ordered_pages` 走同一套 retain/publish/active-reference 代码，源地址照旧走 `deactivate` / `release` 的既有生命周期。
- 好处是**没有"活地址被就地改了一半"的窗口**：旧地址在新地址发布前一直自洽，失败就整体回滚，正是这条路径已经被测试过的性质。原地重发布则要自己保证成员表、引用计数、`reservation`、`checkpoint_frontier` 四者同时一致，而 `reservation` 事实上无法在活地址上重排（见 7.5）。
- 代价是多一次地址激活/停用，但 compaction 每个请求只发生在**请求边界**，不在 decode 内循环里，这点开销看不见。

调用点不变：请求边界、continuation 发布之后（`src/models/qwen3_5/program/prefill.cpp:646` 的 `publish_continuation` 门），页序来自 `build_kv_view`、块集合来自 `select_kv_blocks`。记账同一处：`sequence.rope_delta += dropped_tokens`，`execution_frontier = visible_tokens`（两者已落为 `core::plan_view_boundary`：dropped 累加进 `rope_delta`，执行前沿取视图长度，提交前沿保留原始时间线；视图比历史长、或出现负数一律抛异常，避免相位被静默偏移）。

*（2026-11-28 补记：`prepare_compacted_fork` / 显式页序已经落地，`commit_prefix_fork` 现在按页序走同一套 retain/protect/publish/active-reference 代码；见 `tests/models/qwen3_5/test_context_store.cpp` 的 compacted-view 一节，覆盖顺序、pin 集合、引用计数与"丢掉的页随源地址释放回池"。还差的不是存储层，而是采样侧：prefill 期间按块捕获 Mean-K、用最后一条用户消息打分、把选择结果接到这个调用点上。）*

**7.6 捕获 Mean-K 的落点（读代码确认）。** 主路径里**没有**独立的 KV 追加点：KV 是在 `ops::causal_softmax_attention(qn, kn, v, cache_positions, {}, kv_table_rows, …)` 内部写进分页 cache 的（`src/models/qwen3_5/execution/text.cpp:916-923`，batch 形在 `:907-914`）。所以捕获要挂在**旋转之前**：`text_rope(rope_for_op, …, qn, kn, s)`（`text.cpp:882`）是原地旋转，它之前一刻的 `kn` 就是 pre-RoPE 的 K，形状 `[head_dim, kv_head_count, T]`——正好是 `kv_block_mean_keys` 要的 `[D,H,T]`，不需要反旋转，也不需要从分页布局里读回来。MTP 的对应位置是 `text.cpp:353`（旋转）与 `:486`（同一处旋转后再 `kv_cache_append`）。

由此，这一步的实现顺序是：一个 per-request 索引对象（按 128-token 块累计 FP32 和与计数）→ 在 `text.cpp:882` 之前把本 chunk 的 `kn` 交给它（块边界跨 chunk，所以要带 `positions` 判断哪些列完成了整块）→ 请求边界用最后一条用户消息的 `qn` 调 `kv_block_scores` → `select_kv_blocks` → `prepare_compacted_fork`。索引对象是增量建的，这一点与参考实现一致：prefill 结束后再扫分页 cache 会漏掉 prefill 中途被换到主机的块。

**7.4 动之前必须确认的三件事。**

1. `committed_frontier` / `checkpoint_frontier` 是否必须以原始 token 计数表达（checkpoint 复用与 slot 文件都依赖它）；
2. 若该模型启用了 log-scaling（位置相关的长上下文缩放），`rope_positions` 必须取原始位置，否则缩放因子错档；
3. 页的引用/覆盖顺序错了会变成 GPU 上的 use-after-free——发布必须在 retain 之前还是之后，按 fork 路径的既有顺序，不要自创。

**7.5 引用计数在两条路线上分别意味着什么。**

成员表是每个地址定长的单元格数组（`Address` 在 `kv_store.h:1739`，含 `page_count / committed_frontier / checkpoint_frontier / reservation / row / occupied / active`），引用计数是**每个物理页**的（`retain_reference(handle, writer)`、`release_reference(handle)`）。

**走 7.3 的广义 fork（推荐）**：新地址的成员表在提交时一次性写好，源地址一行都不改，被丢掉的页只失去一次引用；源地址按既有生命周期 `deactivate` / `release` 之后，这些页自然回到空闲池，**容量是真的回收的**。这里唯一要守的顺序，是 fork 路径本来就守着的那个：先 `retain_reference(page,false)` + `protect_coverage`，再 `tables_->publish(row, 0, scratch)`，再记 active reference（`kv_store.h:1135-1195`）。

**走原地重发布（仅作后备，7.3 已不推荐）**：那才需要"先全量 retain 新进入、再全量 release 离开、留下的一个都不碰"的置换顺序，而且 `Address::reservation` 描述的是地址最初的页分配、在活地址上无法重排——被丢掉的页会继续占着驻留，直到单独一步重排。也就是说那条路第一版买到的是**正确的视图**而不是**省下的显存**。这正是选 7.3 的原因之一。

**两条路线都要一并成立的检查**：

- 压实后 `missing_device_pages(address)` 必须为 0，否则 attention 会读到没驻留的行；
- 若某个被丢掉的页仍是某个前缀 fork 的源（`source_pins(page) != 0`），释放必须被拒绝而不是把它回收掉；
- 压实改变了 `checkpoint_frontier` 的覆盖，`rebuild_checkpoint_protection()` 必须跟着重跑；
- 只重排、不丢弃的压实（选择集等于全体）必须等价于空操作，这是最容易被写错又最容易测的边界。

### 7.7 应用时机：在采样之后、下一个请求之前（本轮读码得出）

要把决策真正应用上去，顺序不是"边界一算完就换地址"，而是：

1. **prompt 的最后一个 token 采样完成之后**。`advance_prefill` 在
   `sample_from_hidden(schedule_state, prefill_hidden, staged.prompt_tokens, ...)` 里用**原始**前沿做
   RoPE 相位；如果在此之前把地址换成压实视图，cache 里只剩可见页，而这一步仍按原始前沿去读，
   attention 的范围与相位就会对不上，而且不会报错——只会给出一组看起来正常的错误 logits。
2. **在请求边界**（下一轮 prefill 的入口，或事务里发布 continuation 的位置）。KVMem 的召回点本来就是
   请求边界；把压实放进 `transactions/materialization.cpp` 的前缀 fork 处，才能复用已有的 entitlement
   计算、retained-tail 备份与计时器。
3. **换地址之后**：`sequence.text_kv_valid = visible`、`sequence.rope_delta += dropped`，并且必须
   `recall_cursor->reset()`——块编号与内容在新地址上不再对应（§9 第 1 步已记这条约束）。
4. **下一轮的采样位置**由新前沿 + delta 推出（decode 本来就是
   `rope_pos = frontier + sequence.rope_delta`），不能再传原始 token 数。

第 1、2 条正是这次没有直接动手的原因：第 1 条要求先找到"prompt 已采样、请求尚未进入 decode"的那个点，
第 2 条要求在事务里加一条与 checkpoint 发布并列的分支。两者都必须与 `sequence.ledger`、prefix digests、
checkpoint 覆盖一起改，属于会改变产品行为的集成，应当单独提交、单独验收。

### 7.8 尾页：压实的最后一块拼图（本轮读码得出，尚未动手）

`prepare_compacted_fork` 要求目标地址覆盖 `frontier` 个 token、且 `pages_for_tokens(frontier) ==
保留页数`、每一页都是**整页**（`committed_columns >= kPagedKVPageSize`）。而 prompt 结束时的
`sequence.text_kv_valid` 一般不是 64 的倍数，最后那个**不完整页**（1..63 列）不属于任何保留页。

于是"把视图换成地址"这一步现在有两种写法：

- **只压到整页边界**：`frontier = floor(text_kv_valid / 64) * 64`，剩下不足一页的 token 不保留。
  这会**丢掉尾部几列**，下一次 decode 的 attention 就少看几个 token——静默的精度损失，不能接受。
- **让 fork 支持部分尾页**：`prepare_prefix_fork` 本来就有 `tail_columns` 与
  `materialize_transfer_destination` 的尾页拷贝路径（`kv_store.h:1054-1077`），`commit_prefix_fork`
  也照着 `tail_destination_` 搬列。把它接进压实版本（保留页 + 一个尾页拷贝）是把这条路走通的正解，
  代价是要给 store 测试补"整页 + 尾页"的组合，并想清楚尾页在新地址里是重拷还是共享。

在它落地之前，`NINFER_KV_RECALL_APPLY` 只会在 `dropped_tokens == 0` 之外什么都不做：决策照旧打印，
地址一个字都不改。这也是本轮没有把 `apply` 接进主线的原因——不是时机没想清楚，而是**最后一个字节还搬不过去**。

具体的改法（已实现，见 `531d35a6`）：保留尾页不需要拷贝——尾页就在保留页序列的最后，物理身份不变，
`committed_columns` 已经覆盖它需要的列。改动落在两处：

- `prepare_compacted_fork`：把 `frontier` 拆成 `full_pages = frontier / 64` 与 `tail_columns = frontier % 64`，
  前 `full_pages` 页要求完整、最后的保留页要求 `committed_columns >= tail_columns` 且必须是源地址最新的
  那一页；构造时传真实的 `tail_columns`、`tail_destination_` 仍为 `nullopt`（不拷贝）。
- `commit_prefix_fork`：压实分支按 `required_pages` 逐页 publish / retain / `protect_coverage`（尾页只覆盖
  `tail_columns` 列）/ membership；`needs_tail_copy()` 在压实分支恒为 false。

store 测试里加的就是"整页 + 尾页"的组合：源 232 token（最后一页 40 列），保留 `{0,2,3}`、frontier 168，
断言 `mapped_pages == 3`、`committed_frontier == 168`、尾页物理身份不变、被丢的那页引用归零、释放后池子清空。

## 7.9 换地址前还剩一个口径问题（下一轮从这里开始）

`report_recall_boundary` 现在用 **索引口径**的前沿：`committed = complete_blocks() * block_tokens`，即 128 的整数倍。
索引只对整块负责（半块的平均值没有意义），但序列真正持有的 token 数是 `sequence.text_kv_valid`，一般不是 128 的倍数，
两者之间还有 1..127 个 token（它们属于最后一个未完成的块，落在 recent 里，永远被保留）。

`apply_recall_view` 必须用**序列口径**：`pages = pages_for_tokens(sequence.text_kv_valid)`，
`boundary.visible_tokens` 也以 `text_kv_valid` 为准。因此要先把末尾那个半块的 `KvBlock{id = complete_blocks, tokens = text_kv_valid - committed, pinned = false}`
追加进 `history`，并给它一个分数——参考实现的做法是不让它参与打分、直接当作 recent 的成员（它总在 recent 里）。
两种可行写法：

1. 给 `select_kv_blocks` 增加一个"尾部块免于打分"的入参（只在 recent/sink 覆盖时保留），
2. 或者在 `plan_recall` 内部把尾部块拼进 `selected` 之后再调用 `select_kv_blocks`（选择器已经接受"被选中的块必须存在"，
   而 `build_kv_view` 本来就支持短尾块）。

写完这两处之后，`apply_recall_view` 本身只是 store 调用序列（§7.8 末尾），不再有语义上的未知。**下一步就只剩 `apply_recall_view` 本身**（开关 `NINFER_KV_RECALL_APPLY`）：deactivate 源 →
`create_inactive` → `prepare_compacted_fork` → `commit_prefix_fork` → `sequence.kv->text = *destination` →
`release(source)` → `plan_sequence_view` 写回 `rope_delta` / `text_kv_valid` / `execution_frontier` →
`recall_cursor->reset()`。失败路径必须把序列留在原地址上（源已 deactivate，异常时重新 activate）。

### 7.10 缩小缓存会打破“缓存长度 == ledger 长度”的不变量

实测：`28438a12` 的换装成功（`applied view 4014 tokens, rope delta 10368`），但下一步就报
`KV destructive truncate extends the frontier`。探针打印出真实参数：`truncate frontier 14382 committed 4014 pages 63`，
调用点是 `src/models/qwen3_5/program/prefill.cpp:587` 的 `trim_sequence_kv(sequence, base, …)`（`base = staged.base`，
即常驻前缀的**时间线**长度）；同一段代码紧接着执行 `sequence.text_kv_valid = base; sequence.ledger.resize(base);`。

这说明引擎在这条路径上假设两件事同时成立：缓存里的 token 数与 ledger 长度相等，且提交前沿等于序列前沿。压缩视图只能改前者（缓存），不能改后者（时间线）；而 `ledger.resize(base)` 会把时间线也砍到缓存长度，与 `rope_delta` 的设计直接矛盾。

结论：压缩不能在生成过程中的任意点发生。它只能发生在
1. 当前回合结束、序列被停放（turn closure / checkpoint）时，或
2. 下一个请求开始、在新的 ledger 建立之前。

两者都要求把“缓存前沿”从 ledger 长度里分离出来存储（现在只有 `text_kv_valid` 一个字段，它在这条路径上会被覆盖）。
在那之前，`NINFER_KV_RECALL_APPLY` 必须保持默认关闭。

**已解决（`6264a8e5`）**：不再就地截断，而是把两个坐标分开——缓存槽取 `sequence.text_kv_valid`，查询相位再加 `sequence.rope_delta`；`trim_sequence_kv` 在视图存在时只把缓存收到视图自身长度，ledger 照常按时间线裁剪；决策钩子挪到 prompt 的 ledger 与缓存提交之后，避免 `text_kv_valid = prompt_tokens` 把视图长度覆盖回去。详见 §7.11。

### 7.11 换装落地：三个“同一个数字被当成两个坐标”的地方（`6264a8e5`）

换装本身在 `28438a12` 就提交成功（源地址 deactivate → 压实 fork → commit → 指向新地址 → 释放源），但下一步解码连撞三处，都源于同一件事：压缩之后“缓存里的 token 数”不再等于“序列走到的位置”。

1. **写权限**：保留页在源地址释放前与源共享，任何页都不能持有 writer 引用（只有 `activate` 会在独占时授予）。视图是 `commit_prefix_fork` 而不是 `activate` 发布的，所以换装后必须 `adopt_writer`：对新地址里“只剩自己一个引用”的页补上写权限。漏了这一步，下一个 token 的 commit coverage 就报 `logical KV committed coverage is not monotonic`（探针实测 `writers=0`）。
2. **解码入口**：`cache_positions` 必须是缓存槽（`text_kv_valid`），`rope_positions` 必须是“缓存槽 + `rope_delta`”。没有视图时两者相等，主线逐位不变；有视图时用时间线位置当缓存槽会写到视图之外。
3. **决策时机与截断语义**：`report_recall_boundary` 现在在 prompt 的 ledger 与缓存提交之后调用；`trim_sequence_kv` 在视图存在时把缓存收到视图长度（时间线 frontier 无法命名视图内部的非连续前缀），ledger 仍按时间线裁剪。

**实测**（4080S，Qwen3.8-27B int8，14,382 token prompt，预算 4096，`--max-context 32768`）：

```
kv recall: boundary 14336 tokens, 112 blocks -> view 4014 tokens (dropped 10368), 32 blocks kept, rope delta 10368
kv recall: applied view 4014 tokens, rope delta 10368, next sample 14382, address row 0
```

换装后解码继续；同一 prompt 在开关关/开两种情况下都回答 `7134-MAGENTA, 2291-ORANGE, 5077-TUR…`——丢掉 10,368 个 token 之后，保留区（4,014 = 31 个整块 + 46 token 尾块）的回答与全量视图逐字相同。

**保真度随留存率变化（200,135 token 同一 prompt）**：

| 预算 | 视图 | 丢弃 | 回答第三个标记 |
|---|---|---|---|
| 全量（开关关） | 200,135 | 0 | `5077-TURQUOISE` |
| 131,072 | 131,015 | 69,120 | `5077-TURQUOISE`（与全量逐字相同） |
| 32,768 | 32,711 | 167,424 | `5083-BL`（编造，非任何埋点） |

32k 视图下第三个标记就落在保留窗口内（0.95 深度≈190k，窗口覆盖 167k 之后），但模型仍读错，说明问题不是“块被挤掉”而是
上下文稀释；把预算提到 131k 回答即恢复与全量一致。因此换装机制本身没有结构性偏差，`NINFER_KV_RECALL_APPLY` 的
可信区间由留存率决定，验收必须按留存率给阈值，而不是只说“4/4”。

**闸口脚本（`tools/bench/run_kv_recall_gate.py`）与它的第一次结论**：脚本按顺序起两个服务
（开关关 → 开关开，因为 262k context 的单个服务就吃掉大半显存，不能并行），同一 prompt 取贪心输出逐字比对，
并打印留存率。实测（200k 级 prompt，四码埋点在 5/35/65/95 % 深度）：

| 预算 | 视图 | 留存率 | 结果 |
|---|---|---|---|
| 196,608 | 无（不触发换装） | 100 % | 逐字相同，4/4 |
| 131,072 | 130,956 | 67.2 % | **不同**：`GAMMA` 被写成 `5803-CYAN`，其余三码正确 |

**197k 历史、67.3 % 留存首次真正换装并通过**（三处修复之后）：

| 预算 | 视图 | 留存率 | 结果 |
|---|---|---|---|
| 131,072 | 131,072 | 67.3 % | 逐字相同，标记 4/4（`applied view 131072 ..., rope delta 63744`） |

此前在同一预算上出现的失败，是视图**没有真正落地**造成的：先是 `KV compacted-view source page is not
stable`（活尾页列数不足被判不稳定），后是 `logical KV protection exceeds committed coverage`
（保留页按满页 64 列保护，而它只提交了更少）。三处修复：
1. `prepare_compacted_fork` 的列数要求按“该页在视图里贡献多少列”计算，活尾页取 `min(needed, 已写列数)`；
2. 压缩前沿对齐到页边界（`history_tokens -= history_tokens % 64`），正在写的活尾页不进入保留集合；
3. `commit_prefix_fork` 的 `protect_coverage` 对压缩视图取 `min(columns, committed_columns)`。
**教训**：这类失败会被闸口误判成“保真度不足”，因为它只看到答案不同；所以闸口必须同时检查换装是否真的发生
（`applied view` 行），否则拒绝换装的运行会被当成一次成功。

闸口现在支持 `--depths`（按深度埋 1-8 个标记），因此同一保留率可以换一个埋点布局再问一次。
用 `--depths 0.1,0.3,0.5,0.7,0.9` 在同一个 131,072 预算上重跑，结果**逐字相同（PASS）**：

| 预算 | 视图 | 留存率 | 布局 | 结果 |
|---|---|---|---|---|
| 131,072 | 130,956 | 67.2 % | 四码 5/35/65/95 % | 不同（`GAMMA` 编造） |
| 131,072 | 130,964 | 67.2 % | 五码 10/30/50/70/90 % | 逐字相同 |

同一保留率、同一 prompt 语料，只换埋点深度就一次失败一次通过，说明差异来自**具体证据落在窗口的哪个位置**
（以及它与 recent/block 边界的相对关系），不是“67 % 这条线”。所以结论保持：没有全局安全留存率，
闸口必须按工作负载跑；分层曲线需要更多样本，单点通过不能推广。

所以“换装在 65 % 留存下与全量一致”这句话**只对特定 prompt 成立**，不能当结论：同一个第 65 % 深度的埋点，
在 67 % 留存下就被读错，而错法是“形状正确、数值编造”。闸口因此不设全局阈值，它报告本次的留存率与
逐字结果，按工作负载判断；要给出安全留存率，需要一份按深度分层的召回曲线，而不是单个 prompt 的一次通过。

**选择器的信号是平的（`NINFER_KV_RECALL_TRACE_BLOCKS` 实测）**：25 % 留存时，被保留的 384 块范围
0..1521，目标块 990（GAMMA@65%）不在其中；把它的分数与排名打出来看：

```
block 990  score -0.056013 rank 482/1522 kept 0
block 0    score  0.329866 rank   0     kept 1   (sink)
block 1521 score  0.078475 rank  40     kept 1   (recent)
block 700  score -0.067552 rank 544     kept 0
```

sink 与 recent 为正（0.08..0.33），历史中段全部是接近零的负值（−0.06..−0.14），一个被埋进任意位置的事实块
并不比邻居高。换成 64-token 块也没有让信号变尖，只是把失败从 GAMMA 挪到 ALPHA（另一个块进了一个块出）；
把查询池从 512 列收到 8/32 列，block 990 的分数小数点后六位都不动，说明问题不在查询采样宽度。

结论：Mean-K 余弦按 16 层 × 4 KV head 平均后，对“针在干草堆”的任意事实检索没有区分度——查询本身就
不指向那条事实，分数在噪声里。这不是算错（算子已对 FP64 oracle 校准），而是 KVMem 原论文里 Mean-K
的有效前提是 agent 工作区里查询与被召回文本**共享词汇/结构**；单轮 needle 不在这个前提下。要在低留存
下把内部块选对，需要一条不同的检索信号（多路查询并集、或按注意力质量而非按查询余弦），这不是
“调一下打分函数”能补的。

**前置条件**：视图只按保留页数拿 entitlement，增长额度要等源释放后再补；如果整卡容量刚好被历史占满（例如 `--max-context 16384` 跑 14k prompt），补额度会失败并报 `KV coverage exceeds active entitlement`。验收要在有余量的容量下跑（本轮用 32768）。

## 8. 不在本计划内的做法

- 不移植参考实现的存储格式与 397 文件补丁树本体（那是另一条血缘的引擎）。
- 不引入 re-RoPE 路径：本仓库的 RoPE 位置与 cache 位置本来就是两个张量，压实视图只需要正确的相位与正确的范围，不需要重算 K。
- 不新增注意力 kernel：`causal_softmax_attention` 的 `positions` 只承担读取范围，相位由 `text_rope` 独立负责。
- 不在召回层做学习式策略；先用与参考一致的静态打分，把变量留给预算。
## 9. 下一步（接线，按此顺序执行）

模块层已经全部落地并有测试：捕获接口、设备累加器、块打分、选择、视图组装、驻留计划、位置记账，以及把它们串起来的
`core::plan_recall`。剩下的是把这条链接到主线的三个点上，顺序不能反：

1. **捕获**（`src/models/qwen3_5/execution/text.cpp:882`，已接好、默认关闭）：`TextContext::set_recall_sink` 为 null 时热路径
   没有额外工作；打开后每个标准注意力层在 `text_rope` 之前把 pre-RoPE 的 kn（`[head_dim, kv_head_count, T]`）交给
   `KvRecallSink`，由 `KvRecallCursor` 按层累加。程序层（`5c870d6e`）在 `ProgramImpl` 构造时按 `NINFER_KV_RECALL` 建游标，
   经 `ExecutionCore::recall` 只交给规范 prefill；投机校验、评分前向、图内 decode 都拿不到它。
   **实测**：一个 7,440 token 的 prompt 打印 `kv recall: captured 7440 columns over 16 layers -> 58 complete blocks of 128`，
   之后每个续写前向（host 驱动的续 prefill）继续累加（+2/+3/+4），说明索引跟得住增长的历史。
   **未做**：`captured_columns_` 跨轮次、跨复用一直累加；一旦压缩或换会话，块编号与内容不再对应，必须在序列被压实/复用时重建。
2. **边界打分**（`src/models/qwen3_5/program/prefill.cpp` 的 `publish_continuation` 闸口之后，`report_recall_boundary`）：
    `KvRecallCursor::finish` → `block_scores()`（每层每 KV head 的 cosine，查询取最后 ≤512 列的 GQA 分组平均）→
    `core::plan_recall`。查询侧已接好：`KvRecallSink::capture_query` 与 `capture_layer` 同点调用，`text_rope` 之前；
    游标按 `[head_dim, kv_heads]` 保留每层一个池化查询，`block_scores()` 是设备算子 `kv_block_scores` 的主机等价物。
    **实测**（262,144 窗口、int8 KV、`NINFER_KV_RECALL=1`、`NINFER_KV_RECALL_BUDGET=8192`）：
    `kv recall: boundary 16896 tokens, 132 blocks -> view 8192 tokens (dropped 8704), 64 blocks kept, rope delta 8704`。
    这就是一次真实召回点的完整决策，且没有改动任何地址。
 3. **发布压实视图**（已完成，见 §7.11）：`KvRecallDecision.view.pages` → `KVStore::prepare_compacted_fork` + `commit_prefix_fork`（已落地，
    `c75d9d48`）。**待做（本轮工作的收尾）**：为目标 continuation 分配新的 KV 地址空间并在提交后替换 `sequence.kv->text`；
    这一步必须走 `transactions/materialization.cpp` 的事务（`root_text_address`、`activation_frontier`、 entitlement、
    retained-tail 备份、计时器），不能就地改写活动地址——就地改写会在中途留下半个被改写的地址，且 `Address::reservation`
    无法重规划。
 4. **位置记账**（已完成，见 §7.11）：`KvRecallDecision.boundary` 写回序列（`rope_delta`、执行前沿、提交前沿）。decode 已经按行读两个位置，
    所以这里不需要改 kernel。**注意顺序**：先提交新地址，再改 `sequence.rope_delta`/`text_kv_valid`，否则中间一步的
    位置与地址不一致。
    这一步的纯函数已经落地：`core::plan_sequence_view(committed_tokens, boundary)`（`a532cfc2`）把边界换成三个数，
    并守住"视图只缩短 cache、不移动时间线"这条不变量——`execution_frontier + rope_delta == 原位置`，否则抛
    `std::invalid_argument`。多次压实会累加 `rope_delta`，`next_sample_position` 始终等于第一次调用时给出的原位置，
    测试覆盖了空位置、丢失 delta、视图越界、二次压实四种情况。
 5. **验收**（任一不满足即回退）：≤262,144 时窗口化与全量的 `tools/bench/run_kv_recall_fidelity.py` 逐字节一致；
    30/50/80 万 needle 4/4；预算不足必须判失败（负对照：B=36,864 丢中段针）；关开关时短上下文 bit-identical；
    prefill 不回退、decode 不降超 5%。
    **本卡的限制**：产物位置容量是 262,144，所以 300k/500k/800k 的 needle 必须先做位置扩展（RoPE 缩放或长上下文微调），
    这与参考实现能跑 637,768 的差别只在这里——它的位置表更大。在这一步之前，本卡能验收的是"不丢块时逐字节一致"和
    "强制丢块时仍 4/4"两件事。

第 1、3 步是唯一会碰热路径与生命周期的改动，建议各自单独提交、单独验收，并先跑一遍不带开关的全量基线留档。

### 已完成（便于核对进度）

| 项 | 位置 | 提交 |
|---|---|---|
| 规划器：紧凑工作集与驻留 delta | `src/core/kv_workspace.{h,cpp}` | `80459af6` |
| 块打分与块均值（设备） | `src/ops/{launcher,wrapper}/kv_block_pool.*` | `9557432a`、`bbe89d24` |
| 视图组装、选择器 | `src/core/kv_view.{h,cpp}`、`select_kv_blocks` | `bf5b6ab1`、`ce4c786a` |
| 压实 fork 的发布路径 | `src/models/qwen3_5/program/storage/kv_store.*` | `c75d9d48` |
| 位置记账 | `core::plan_view_boundary` | `1eb742fc` |
| 每层召回索引 / 层容器 | `src/core/kv_recall_index.*`、`kv_recall_set.*` | `5a2189ed`、`80cb4a6a` |
| 设备累加器（每层游标） | `src/models/qwen3_5/program/storage/kv_recall_cursor.*` | `96ff8fb1` |
| 捕获接口与热路径钩子 | `src/core/kv_recall_sink.h`、`TextContext::set_recall_sink` | `bdd0d9f0` |
| 一次召回点的合成决策 | `src/core/kv_recall_decision.{h,cpp}` | `9d08f766` |
| 验收闸口 | `tools/bench/run_kv_recall_fidelity.py` | `012f2280` |
| 主线捕获接线（默认关闭） | `program_impl`, `prefill`, `context.h` | `5c870d6e` |
| 边界查询捕获（GQA 池化、512 列上限） | `kv_recall_sink.h`、`kv_recall_cursor.*`、`text.cpp` | `19a796e9` |
| 块打分的主机形态与实测边界 | `KvRecallCursor::block_scores`、`report_recall_boundary` | `5e380bdd`、`8718f236` |
| 换装落地：写权限、缓存槽、截断语义 | `kv_store.h`、`decode.cpp`、`prefill.cpp`、`context.cpp` | `6264a8e5` |

### 增量构建的坑（`5c870d6e` 期间）

改了被广泛包含的头（`program/context.h` 给 `ExecutionCore` 加了一个成员）之后，ninja 只重编了 8-9 个目标就链接，
生成的 serve 在 `initializing runtime` 处稳定崩溃（`0xC0000374` 堆损坏 / `0xC0000005`），而同一份源码 `git stash`
后重编就正常——是旧目标文件与新结构体布局混用。把 `src/models/qwen3_5` 与 `src/core` 下所有源文件的时间戳刷新后整层重编，
崩溃消失。**规则：改这类头之后要么整层重编，要么先 `ninja -t clean` 相关目录，不要相信增量结果。**