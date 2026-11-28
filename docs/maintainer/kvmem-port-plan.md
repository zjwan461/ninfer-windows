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

## 8. 不在本计划内的做法

- 不移植参考实现的存储格式与 397 文件补丁树本体（那是另一条血缘的引擎）。
- 不引入 re-RoPE 路径：本仓库的 RoPE 位置与 cache 位置本来就是两个张量，压实视图只需要正确的相位与正确的范围，不需要重算 K。
- 不新增注意力 kernel：`causal_softmax_attention` 的 `positions` 只承担读取范围，相位由 `text_rope` 独立负责。
- 不在召回层做学习式策略；先用与参考一致的静态打分，把变量留给预算。