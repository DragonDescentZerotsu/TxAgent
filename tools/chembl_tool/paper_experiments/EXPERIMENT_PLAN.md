# 实验计划

本文档记录当前第一轮冻结矩阵的研究问题和运行协议。面向 ICLR 2027 的后续数据补齐、same-parent
retrieval 消融、第二模型、重复运行、source-quality annotation、baseline 和投稿时间表见
[ICLR_2027_EXECUTION_PLAN.md](ICLR_2027_EXECUTION_PLAN.md)。Visibility 升降原因的集中分析见
[VISIBILITY_ANALYSIS.md](VISIBILITY_ANALYSIS.md)。

## 研究问题

### RQ1：相似分子检索能否提高分子性质分类性能？

主分析在 deployment-visible agentic setting 中比较 `none` 与 `direct`。Operational retrieval 允许检索
同一 RDKit molecular parent 的盐型、溶剂化物和 formulation-linked record，代表真实部署；parent-disjoint
消融排除同一 parent 后用后续真正 analog 补齐 top-k，用于检验增益是否仍来自相似分子迁移。

### RQ2：Starling 原文抽取 evidence 与 ChEMBL curated structured assay records 有何差异？

- BBB：比较 ChEMBL direct 与 Starling direct。
- Bioavailability：比较 ChEMBL direct 与 Starling direct full。
- Bioavailability mechanism：比较 ChEMBL full mechanism 与 Starling full mechanism。

ChEMBL 本身也包含从科学文献人工抽取、整理和标准化的 bioactivity data，因此这里不是“literature vs
non-literature”，而是保留 source text/context 的 literature-extracted records 与 curated structured
records 的比较。Primary
source comparison 必须匹配 task、query、endpoint scope 和 retrieval view；性能必须与 coverage、evidence
volume、各 group availability 和独立 source-quality annotation 一起报告。只有跨任务方向稳定时才使用
“Starling generally better”这一结论，否则报告 task/family-specific advantage。

### RQ3：按任务机制拆分证据能否进一步提高 reasoning 性能？

每个任务在同一数据源内比较 `full_flat` 与 `full_mechanism`。两者包含相同的 evidence row，只改变分组方式和并行 mechanism reasoning。Bioavailability 同时提供 ChEMBL 和 Starling 的该项消融。

### RQ4：非数值文献证据是否带来额外价值？

对 Bioavailability direct Starling evidence 比较：

- 只使用数值 direct-F 的 scalar KNN
- 只接收数值 direct-F evidence 的 GLM
- 接收数值、非数值 evidence 和实验条件的 GLM

Starling content 只分 `numeric_only` 和 `full` 两种，不在主实验中引入更细的分类。

### RQ5：结构和 retrieved molecule 身份可见时，系统在真实部署 setting 下表现如何？

论文主实验使用 `deployment_visible`：只隐藏 query name，保留 query structure，并保留 retrieved molecule
的 structure、source ID 和数据源已有名称；模型自主决定是否调用工具。`identity_blind` 与
`deployment_visible_prefetched` 不进入主结果表，只作为补充控制，用于诊断 identity/structure visibility
和固定工具证据对模型行为的影响。它们不能替代真实部署的 agentic 结果，也不能与 agentic 条件构成纯
visibility 因果比较。

### RQ7：Retrieval coverage 与 macro-F1 增幅有什么关系？

在 deployment-visible agentic 主制度中，以每个 retrieval condition 为观察单位，报告 overall、正类和
负类 coverage，并计算相对同任务 `none` 的 paired macro-F1 差值。主性能量使用 macro-F1，以避免标签
不平衡使 accuracy 掩盖少数类行为；class-conditional coverage 是解释 retrieval availability 是否偏向
某一类别的诊断量。

该分析是描述性关联，不将 coverage 解释为独立操纵的 evidence quality，也不做额外的
`none/hybrid/retrieval` counterfactual。不同 task、source 和 retrieval view 分层保留，最终结果需同时
报告样本分母与 paired bootstrap 区间。

### RQ8：超出当前 curated mechanism envelope 多远后，增加更多 assay evidence 不再帮助 LLM？

> 2026-07-23 update：四任务 strict-hop census 得到 `0/4` publishable strict H2，因此下文
> `D/C/H1/H2` 四级主曲线已判定 no-go，只保留为 strict causal secondary/historical design。
> RQ8 主实验改为 `RELEVANCE_DILUTION_EXPERIMENT_PLAN.md` 定义的
> `D -> D+C -> D+C+R1 -> D+C+R1+R2 -> D+C+R1+R2+R3`
> controlled relevance-dilution curve。不得按下文旧 contract 启动 strict H2 LLM run。

这一实验从 base 到 extension 全程只使用 ChEMBL：`D/C` 来自当前 ChEMBL direct/full evidence，`H1/H2`
来自同一冻结 ChEMBL release 的扩展 assay。不得为任何一层运行 Starling extraction，也不得构造
`Starling D/C + ChEMBL H1/H2` 或其它跨 source 混合累计曲线。四层使用相同的 ChEMBL release、标准化流程、
measurement-quality gate、molecule aggregation、provenance contract 和 source manifest。它与 RQ2 正交：
RQ2 在匹配 scope 下比较 source；RQ8 固定 source 为 ChEMBL，研究 mechanistic distance 与 evidence quantity
的联合变化。

Distance 由每个 task 预先冻结的 mechanism-family graph 定义，而不是 assay description 的文本相似度。
`D` 是层级树的 root，当前 paper-facing C mechanism families 是第一圈子节点；H1/H2 是每条 C branch 向外展开的
tree depth，不是与 C 平行的全局 source groups：

```text
D root
  C.family_k
    H1.family_k: 该 C family 有且只有一个聚合 child；内部可含多个一跳 measurement families
      H2.family_k: zero or one 聚合 child；内部可含多个到 H1 一跳的 measurement families
```

`D` 与 `C` 必须互斥，且 `D+C` 必须逐 source group 精确等于当前 ChEMBL full evidence union。C 内已有
families 不要求相对 D 具有统一 hop 数；E12 不重新分层它们。每个 C family 必须声明唯一 H1 child spec，H1
至多声明一个 H2 child；一个 child node 可以聚合多个 targets/measurement families，但每条 assay 保留具体 path。
extension measurement family 必须唯一归属一个 parent C family，不能跨 branches 重复 evidence。H1 measured node
到 parent C 的最短合格路径必须为 1；H2 到其 H1 parent 有一条边、到整个 `B=D+C` 的最短路径必须为 2，发现
任何到 D/C 的一跳捷径即 validation failure。在声明 extension 前必须对所有 D/C assays 生成 measured-state
census；新增 assay ID 和 measured node 都必须与 base 不相交。
主累计曲线为 `none`、`D`、`D+C`、`D+C+H1`、`D+C+H1+H2`。

可扩展 `C mechanism families` 不包括 direct-exposure organization：direct family 属于 D root。为保持旧
`D+C` evidence union，direct tier 内 endpoint normalization 产生的 `context_dependent` 弱证据仍原样保留，但不
为它制造 H1。BBB 显式冻结可扩展 parents 为 passive/barrier、efflux 和 influx 三个 families。

一条合格推理桥必须连接两个命名明确、可测量、粒度相近的 biological state/process，并记录有方向的
biological relation、允许的 inference direction、适用条件和支持文献。默认只允许顺 causal direction 推断；
只有经过验证的 mechanistic readout 才允许反向 inference。共同器官/疾病/pathway/target class、文本相似度、统计相关、
分子结构相似或相同 assay format 都不构成 hop。物种、组织和实验系统差异单独记录为 `scope_match`，assay
可靠性单独记录为 `quality_status`；它们不计入 hop。映射或最短路径不确定的 assay 进入 `unresolved` 并从
主曲线排除。graph、edge rationale 和 assay mapping 在查看 test performance 前冻结。每个 C family 冻结前必须
找到可信 H1，否则 graph freeze 失败；某个 H1 没有可信 H2 时，只把该 family 的 H2 记为 unavailable，不为形成
完整层级强行纳入弱相关或 cross-task assay。

完整受控曲线使用 `deployment_visible_prefetched` matched-prefetch、固定 flat assembly、相同 retrieval
policy 和 frozen single prior。Flat 主曲线只在 `none`、`D`、`D+C` 与最大可辩护层级做 deployment-visible agentic
confirmation。主要指标为 macro-F1；coverage、unique molecule/assay/family counts、evidence rows 和 tokens
只作 quantity 解释。不增加 fixed-token replacement、rescue/harm rate 或 cross-task source 条件。

E12 同时要求 mechanism supplementary，且不能只跑 maximal point：

```text
D+C mechanism
D+C+H1 mechanism
D+C+H1+H2 mechanism
```

先完成 `D+C+H1`：若 D/C family 的 LLM-visible input hash 与当前 `D+C` mechanism condition 相同，则复用
D/C group outputs，只运行每个 C family 的聚合 H1 branch，并重跑 final。随后运行 `D+C+H1+H2`：复用 D/C/H1 branches，
只运行新增 H2 branches，并再次重跑 final。加入 H2 后，H1 的 group ID、neighbor selection、evidence rows、
serialization 和 branch hash 必须保持不变；否则 H1 branch 必须重跑且复用审计不能通过。

Retrieval budget 按 tree node 计算：每个 C、H1 或 H2 node 最多 3 个 unique molecular neighbors，并使用相同
similarity threshold/identity policy。聚合 child 内的不同 target/measurement families 共享这 3 个位置，不能按
target 各取 3 个。

完整 matched-prefetch supplementary 在 H1、H2 两点都运行；deployment-visible agentic 也确认这两个 mechanism
points。每个 prefix 的 flat 与 mechanism 必须使用相同 evidence-row multiset，只允许 grouping 和并行 reasoning
不同。`D+C+H1+H2` 只加入 available H2 children；若所有 C-family H2 均 unavailable，task 只运行到
`D+C+H1`，并逐 family 记录 H2 unavailable。

RQ8 的 ChEMBL-only 结果不能用于证明 Starling 优于 ChEMBL；该 source claim 只能来自 RQ2 的 matched-scope
比较和独立 source-quality annotation。E12 审计必须确认每个 retrieval prefix 的全部 evidence rows 都指向冻结的
ChEMBL source manifest，出现 Starling 或其它 source row 时该 condition 不进入曲线。

可执行的 distance ontology、配置 schema、独立代码路径、资源上限与回归验收标准见
`DISTANCE_EXPANSION_DESIGN.md`。

## 冻结实验矩阵

| Task | None | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism | Numeric Starling | Scalar KNN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BBB_Martins | yes | yes | yes | yes | yes | yes | yes | no | no |
| Skin_Reaction | yes | yes | yes | yes | yes | yes | yes | no | no |
| ClinTox | yes | yes | yes | yes | no | no | no | no | no |
| Bioavailability_Ma | yes | yes | yes | yes | yes | yes | yes | yes | yes |

当前已有 26 个 deployment-visible agentic 条件作为主实验候选；BBB/Skin 新增的 5 个 Starling
operational 条件、对应 test/valid identity-blind、全部 valid matched-prefetch，以及现有 22 个 retrieval
条件的 test/valid parent-disjoint 均已完成。Test matched-prefetch 保留原 21 条件，另有一个与 LLM
可见性无关的 scalar KNN 对照。ClinTox 三个 Starling 条件补齐后，最终 agentic operational 主矩阵为
29 个条件；新增 ClinTox retrieval 也必须同轮运行 parent-disjoint。当前结果仍属于 exploratory。

RQ8 的 ChEMBL-only cumulative distance curve 是与这套 29-condition source/grouping matrix 正交的新增
实验，不计入上述条件数；完整 curve 先作为 matched-prefetch 受控实验运行，再做少量 agentic confirmation。

## 固定模型和检索设置

```text
requested model: zai-org/GLM-5.2-FP8
temperature: 0
maximum output tokens: 20480
primary deployment mode: deployment_visible with agentic tool use
supplementary controls: identity_blind and deployment_visible_prefetched
operational retrieval policy: canonical-record exact exclusion; same-parent records allowed and annotated
analog-only ablation: parent-disjoint exclusion with top-k backfill
Morgan radius/bits: 2 / 2048
top k per group: 3
minimum similarity: 0.30
single prior: frozen from each task's none condition
SDK transport retry limit: 2; structured JSON validation: at most 4 total attempts; the last two attempts deterministically reserialize the same input to recover provider-side degeneration; remaining incomplete samples are excluded and rerun visibly at batch level

Provider request-size guard: cleaned group payloads above 750 KB use deterministic even-spacing with at most 100 evidence rows per oversized neighbor; truncation counts and byte metadata remain visible in the prompt trace.
```

The endpoint-returned model identifier is recorded from every response rather than inferred from the requested alias.

## 主要与次要指标

主要指标：test macro-F1。

次要指标：

- accuracy and class-specific confusion matrix
- retrieval coverage overall and by mechanism group
- retrieval coverage by true class, and condition-level association with paired macro-F1 gain versus same-task `none`
- prompt, completion, and total token count
- structured-output retries and failed runs
- number of LLM calls and frozen-prior reuse count
- query-SMILES trace leak count
- prompt-boundary structure, molecule-identifier, and source-name leak counts

## 统计分析

- 每个条件 macro-F1 的 95% test-set bootstrap 区间
- 每项预注册 macro-F1 差异的配对 bootstrap 区间
- 基于配对正确性的精确双侧 McNemar 检验
- 对预注册比较族的 McNemar p-value 做 Holm 校正
- 不根据 test set 选择阈值，也不做事后 deterministic label correction

配对比较预先声明在 `summarize_results.py` 中。存在失败样本或不满足相应可见性 contract 的结果，在修复并重新审计前不能用于论文。Prompt audit 检查保存的 system/user/tool request input；assistant response 不视为上游身份披露。

## 补充的非 Agent baseline

现有 MiniMol 结果可作为 learned baseline 背景，但不属于 retrieval ablation，也不能用来选择 agent setting。历史 DeepSeek 和 Bioavailability expert-policy run 仅作为 provenance 参考；它们与冻结的 GLM 矩阵不可直接比较，不进入论文主表。

当前 Starling random/scaffold 已完成 MiniMol `--train-all` head、full-test Morgan KNN `k=3` 和
复用冻结 MiniMol embedding 的 cosine KNN `k=3`。三者进入 Starling benchmark performance overview，
但都不是 agent retrieval condition；训练/选择口径、结果与入口见
`STARLING_BENCHMARK_RESULTS.md`。本节其余 29-condition 计数仍只描述旧 TDC-lineage agentic matrix。
