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

工作量因此从"一个新 kernel + 逐字节一致性验证"降为"组装器 + 表构造 + 一致性验证"。

### 3.4 Mean-K 索引与打分

新增 `include/ninfer/ops/kv_block_pool.h` 与 `src/ops/{launcher,wrapper}/kv_block_pool.*`（已落地，纯数学、无引擎类型）：

- `kv_block_mean_keys(keys[D,H,T] bf16, block_tokens, means[D,H,B] fp32)`：块内均值，FP32 累加、除以精确块长。入参是**旋转前**的 K，所以天然与位置无关，不需要去 RoPE 算子。
- `kv_block_scores(query[D,Q] bf16, means[D,H,B] fp32, scores[B] fp32)`：GQA 分组内把 query head 平均成 KV head 的查询向量，与块均值做余弦相似度、再按 KV head 平均。softmax 与 top-k 留给 3.5。

测试 `tests/ops/test_kv_block_pool.cpp` 用独立 FP64 oracle，覆盖 D=8/16/64/256、GQA 2/4/6 倍、含尾部不完整块、全零块（相似度定义为 0）与真实几何（head_dim 256、4 KV heads × 24 query heads）。"块均值"的存储量：16 层 × 4 heads × 256 dim × 2 B × 块数，637k tokens / 128 = 4,976 块 → 约 65 MiB。

索引仍需在 prefill 过程中**增量**构建（不能等 prefill 结束再扫，否则块已被搬走），这部分属于 3.3 的接入点。

内存：16 层 × 4 heads × 256 dim × 2 B × 块数；637k tokens / 128 = 4,976 块 → 约 65 MiB，可忽略。

### 3.5 选择策略与调度接入

- 打分聚合：层与 head 取平均，查询取最后一条 user 消息（上限 512 tokens，与参考一致）。
- 先实现"只选 top-k、不做图传播"的最小版本，用验收数据决定是否要传播（参考实现用 alpha=0.2 的一跳传播，收益未在本卡验证）。
- 接入点：请求边界（prefill 之后、decode 之前），与 FIFO/无抢占兼容；不改 `Engine` 的并发语义。

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
2. **等价性**：≤262,144 tokens 时，窗口化路径与全量路径输出逐字节一致（用第 10 节的 fidelity 探针，`codes` 与 `first_lines` 两个）。
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

## 7. 不在本计划内的做法

- 不移植参考实现的存储格式与 397 文件补丁树本体（那是另一条血缘的引擎）。
- 不引入 re-RoPE 路径：本仓库的 RoPE 位置与 cache 位置本来就是两个张量，压实视图只需要正确的相位与正确的范围，不需要重算 K。
- 不新增注意力 kernel：`causal_softmax_attention` 的 `positions` 只承担读取范围，相位由 `text_rope` 独立负责。
- 不在召回层做学习式策略；先用与参考一致的静态打分，把变量留给预算。