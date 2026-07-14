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

### RQ2：文献抽取的 Starling evidence 是否优于人工整理的 ChEMBL evidence？

- BBB：比较 ChEMBL direct 与 Starling direct。
- Bioavailability：比较 ChEMBL direct 与 Starling direct full。
- Bioavailability mechanism：比较 ChEMBL full mechanism 与 Starling full mechanism。

性能必须与 coverage、evidence volume 和各 group availability 一起报告，因为数据源价值由质量和可检索性共同决定。

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

## 冻结实验矩阵

| Task | None | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism | Numeric Starling | Scalar KNN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BBB_Martins | yes | yes | yes | yes | yes | no | no | no | no |
| Skin_Reaction | yes | yes | yes | yes | no | no | no | no | no |
| ClinTox | yes | yes | yes | yes | no | no | no | no | no |
| Bioavailability_Ma | yes | yes | yes | yes | yes | yes | yes | yes | yes |

当前第一轮已有 21 个 deployment-visible agentic 条件，作为主实验候选；另有 21 个 identity-blind 和
21 个 matched-prefetch 条件作为补充诊断，以及一个与 LLM 可见性无关的 scalar KNN 对照。Starling
补齐后，最终 agentic operational 主矩阵为 29 个条件；所有 retrieval 条件还要运行 parent-disjoint
消融。当前结果在该消融完成前均属于 exploratory。

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
