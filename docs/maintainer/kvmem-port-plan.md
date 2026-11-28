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
| 物理 KV 池、页组、block table | `docs/maintainer/paged-kv-cache.md`：typed pools、logical pages、Device/Host replica、consumer view | 直接复用，不改布局 |
| 上下文缓存与候选选择、保留策略 | `docs/maintainer/resource-scheduling-and-context-cache.md`：candidate/retention、Device/Host checkpoint、StateImage 迁移 | 复用调度框架，新增"块级"候选类型 |
| GDN 状态提交 | `docs/maintainer/replayssm-gdn.md`：raw transition records | 复用；GDN 状态只在请求边界提交，召回不触碰 |
| 注意力入口 | `ordinary_decode_batch(ids, cache_positions, rope_positions, kv_table_rows, linear_state_slots)` | 需要新增"块视图"参数，见 3.3 |
| 注意力算子 | `ops::causal_softmax_attention`（行 118-121 注释：逻辑前缀单调，只有页映射自由）；`ops::relbias_attention`（`args.q_pos` 标量） | 全注意力 16 层需要按块的原始位置，见 3.3 |
| 主机内存搬运 | 已有 Host checkpoint 与 pinned staging | 复用 |

关键约束：`causal_softmax_attention` 的逻辑前缀是**单调**的，因此召回视图必须**组装成新的连续前缀**（把选中的块按原始顺序压实到设备窗口），而不是在中间留洞。这也是参考实现的做法（`compact_start` 只是 attention 地址）。

## 3. 模块拆分

### 3.1 规划层（纯函数，无设备依赖）

新增 `src/core/kv_workspace.h` / `.cpp`：

```cpp
struct KvBlockId { std::int32_t index; };
struct KvBlock { KvBlockId id; std::int32_t orig_pos_start; std::int32_t tokens;
                 Tier tier; std::int32_t slot; std::int32_t baked_pos; };
struct WorkspaceBudget { std::int32_t selected_tokens, reserve_tokens, device_tokens; std::size_t host_bytes; };
struct WorkspacePlan { std::vector<KvBlock> stage_in, stage_out; std::vector<KvRemap> remaps; std::int32_t visible_tokens; };
WorkspacePlan plan_workspace(const WorkspaceBudget&, span<const KvBlock> resident, span<const KvBlock> scored);
```

- 块大小固定为页组的整数倍（当前页组 64 tokens，块建议 128）。
- 不变量：先 stage-out 后 stage-in（有界池），选中块按 id 升序压实到 0 开始，sink 与 recent 永不被逐出。
- 这一层可以完全用单元测试覆盖，不需要 GPU，先写它。

### 3.2 主机层块存储

新增 `src/core/kv_block_store.{h,cpp}`：按块持有 K/V 的 pinned host 副本 + 设备槽位映射，提供 `spill(block)` / `restore(block, slot)` / `drop(block)`。复用现有 staging 路径与 pinned 分配器，不引入新传输机制。验收：一次 spill+restore 往返后 K/V 逐字节不变（对 fp16 是精确的，不需要容差）。

### 3.3 注意力视图（唯一需要动算子的一处）

现状：解码注意力读 `kv_table_rows[b]` 给出的页；`q_pos` 是标量；RoPE 已在写入时烘进 K。

需要：让 16 个全注意力层在一次前向里，把"压实后的连续窗口"当作前缀读，同时知道**每个槽位的原始位置**（参考实现称 `baked_pos`）。

- 方案 A（推荐）：新增 `ops::causal_softmax_attention_view(..., const std::int32_t* slot_positions, ...)`，与现有算子的差别仅在相对位置偏置的来源：由 `slot_positions[k]` 取代 `q_pos - k`。K 本身不重算、不重写 RoPE。
- 方案 B（不推荐）：恢复位置一致的 re-RoPE。数值上引入 fp16 误差（约 1e-3），且第 10 节的逐字节一致告诉我们没有必要。

工作量为方案 A：一个 kernel + 一个 launcher + 形状校验，复用现有 softmax/pv 主体。

### 3.4 Mean-K 索引与打分

新增 `src/ops/launcher/kv_block_score.{h,cu}`：每块、每层、每 KV head 的 K 均值（去 RoPE 后求和再平均，FP16 存、FP32 累加），以及"查询向量 × 块均值"的 softmax 打分。索引在 prefill 过程中**增量**构建（不能等 prefill 结束再扫，否则块已被搬走）。

内存：16 层 × 4 heads × 256 dim × 2 B × 块数；637k tokens / 128 = 4,976 块 → 约 65 MiB，可忽略。

### 3.5 选择策略与调度接入

- 打分聚合：层与 head 取平均，查询取最后一条 user 消息（上限 512 tokens，与参考一致）。
- 一跳图传播（alpha=0.2），先实现"只选 top-k、不传播"的最小版本，用验收数据决定是否要传播。
- 接入点：请求边界（prefill 之后、decode 之前），与 FIFO/无抢占兼容；不改 `Engine` 的并发语义。

## 4. 实施顺序与工作量

| 顺序 | 工作 | 估时 | 结束时能验证什么 |
|---:|---|---:|---|
| 1 | 3.1 规划层 + 单元测试 | 2-3 天 | 预算/压实/不变量的纯逻辑正确 |
| 2 | 3.2 主机块存储 + 往返测试 | 3-4 天 | 块搬运无损 |
| 3 | 3.3 视图算子 + 与现有算子逐字节一致性测试 | 4-6 天 | 窗口化不改变 ≤262k 的行为 |
| 4 | 3.4 索引 + 3.5 最小选择（top-k，无传播） | 4-6 天 | 30 万 / 50 万 / 80 万 needle 4/4 |
| 5 | 端到端验收与调参（预算、块大小、sink/recent） | 3-5 天 | prefill/decode 不回退，失败模式被 retainer 捕获 |

合计 3-5 周，与 long-context-workspace.md 第 5 节的估计一致。前三步可以并行推进且都不改变默认行为（全部走开关），第 4 步开始才需要真实长上下文数据。

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
| 视图算子的间接寻址开销 | 未测 | 第 3 步先做一致性，再测 T=1 与长上下文两条路径 |
| 预算策略选不中需要的块 | 已实测会发生（36,864 掉一个事实） | 用 retainer 定基线预算，而不是猜 |
| 与现有上下文缓存/StateImage 的所有权冲突 | 未验证 | 第 2 步先做纯存储，不接管调度；第 5 步再接 |
| 工程量（参考实现 397 文件 / +32k 行） | 本计划只取机制不取实现 | 五行模块自研，复用本仓库已有池/搬运/调度 |

## 7. 不在本计划内的做法

- 不移植参考实现的存储格式与 397 文件补丁树本体（那是另一条血缘的引擎）。
- 不引入 re-RoPE 路径（第 10 节证明不需要）。
- 不在召回层做学习式策略；先用与参考一致的静态打分，把变量留给预算。