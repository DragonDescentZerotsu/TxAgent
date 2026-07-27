# 论文 Pipeline 与数据流

## 职责划分

系统只有一套共享的分子证据 agent pipeline，各任务仅提供轻量配置；不存在独立的 Bioavailability 执行引擎。

共享模块负责数据源接入、Minimal Evidence Contract、分子级索引、实验视图、可见性控制、经过验证的 LLM 调用、batch 执行、指标计算和统计分析。每个任务只负责 endpoint 到 mechanism 的映射、任务 prompt 和最终预测 schema。

Bioavailability 的特殊之处仅在于可用数据及其声明的证据类别：Direct oral bioavailability (F%)、Oral exposure proxies (AUC/Cmax)、Fa、Fg 和 Fh。BBB、Skin_Reaction 和 ClinTox 使用完全相同的执行路径，只是采用各自的机制类别。

## 离线证据构建

```text
ChEMBL assay/activity row 或 Starling parquet/HF row
    -> 数据源 profile parser
    -> 包含 endpoint、value、unit、condition、text、quality、provenance 的数据源 row
    -> 附加 minimal_evidence.v1
    -> 按源分子和 endpoint group 聚合重复 row
    -> 标准化源分子结构
    -> 计算 Morgan fingerprint
    -> 构建分子级 neighbor index
```

Minimal Evidence Contract 只描述证据，不进行预测。它保留：

- 数据源及记录 provenance
- 供 harness 侧索引使用的源分子身份
- 可用时的 endpoint 和标量测量值
- evidence text 和 condition text
- evidence role、scope、uncertainty 和 source quality
- 代表性数据源样例

它不会赋予 benchmark label，也不会判断证据能否迁移到 query。

## Query-time 检索

```text
query SMILES（始终供 harness 检索使用；是否发送给 LLM 由可见性制度决定）
    -> 标准化结构并计算 Morgan fingerprint
    -> 选择任务/数据源 experiment view
    -> 在每个选定 group 内独立排序源分子
    -> 应用冻结 retrieval policy
    -> 保留 similarity >= 0.30 的前 3 个 neighbor
```

论文区分两套 retrieval policy。Operational policy 代表真实部署：排除完全相同的 source record，但允许
同一 RDKit molecular parent 的盐型、溶剂化物、重复组分和 formulation-linked evidence，并标记其 entity relation。
Parent-disjoint policy 是 analog-only 消融：对 query 和 source molecule 使用同一 parent normalizer，排除
相同 parent 后继续向后取候选，直至补足 top-k 或耗尽候选。不能通过删除受影响 query 来代替重检索。

实验视图包括：

- `none`：不提供 analog evidence
- `direct`：只提供任务结果的直接证据 group
- `full_flat`：将所有选定的机制证据合并为一个 group
- `full_mechanism`：将同一证据集合按机制类别拆分

flat 视图由已经选定的 mechanism 视图生成，因此 flat 与 mechanism 接收完全相同的 evidence row。这样可以将“证据组织/机制拆分”的影响与“证据可用性”的影响分离开来。

## 可见性与工具执行制度

论文主结果表使用 deployment-visible agentic tool-use；identity-blind 和 matched-prefetch 作为补充控制。
三者复用同一套 retrieval、task config、reasoning、validation 和 batch 代码，但 agentic 与 prefetched 的
工具执行路径和计算量不同，不能把二者差异解释为纯 visibility 因果效应。

### 身份盲化补充控制

在报告所用实验中，LLM 从不接收 query 或 neighbor 的结构和标识符。

```text
query 与检索得到的 neighbor 结构
    -> harness 为 query 调用 molecule_properties
    -> harness 为每个分子对调用 mmp_structure_compare 和 properties_compare
    -> harness 删除结构、InChIKey、molecule ID、已知分子名称，
       以及 evidence text 中嵌入的身份信息
    -> LLM 只接收别名、similarity、数据源证据和预取的工具文本
```

原始 `retrieval.json` 会保留用于 provenance 和调试，但构造 LLM message 时只使用脱敏副本。报告会审计实际保存的请求历史，检查 query/neighbor 结构、分子标识符和已知源分子名称是否出现在 system、user 或 tool 输入中。该上游泄漏测试不检查模型生成的 assistant 文本。

### 部署可见：matched-prefetch 补充控制

该制度更接近实际使用场景：

```text
query:
  structure visible
  name hidden

retrieved molecule:
  structure visible
  source ID visible
  source name visible when the source record provides one
```

输入只要求 query SMILES，因此代码不会主动解析或补充 query 名称。group prompt 保留 neighbor 的 canonical SMILES、source ID，以及 `minimal_evidence.v1` 中实际存在的 source molecule name。若 ChEMBL row 本身没有名称，prompt 只显示其 ID 和结构；Starling row 有文献名称时则保留该名称。

在 `deployment_visible_prefetched` 中，每个 query 直接 replay 对应 identity-blind run 的冻结
`retrieval.json` 和 prefetched tool outputs；不会按当前 task config 重新检索，也不会重新执行可能受
MCS timeout 影响的工具。visible prompt 保留相同 evidence 和工具文本，同时恢复 query/neighbor 结构、
source ID 和数据源已有名称。因此该补充配对唯一改变的是 LLM 可见的身份与结构信息。

### 部署可见：Agentic 主实验

`deployment_visible` 保留 query structure、neighbor structure/source ID/source name；query properties 由
single branch 请求，group comparison 工具由模型自主选择。该制度回答真实部署中的端到端 agent 性能，
用于论文主表；不得与 `identity_blind` 构成严格 visibility 对照。

## 推理与汇总

```text
预取的 query properties
    -> single-molecule 理化性质先验（每个 task/query 冻结一个结果）

每个 evidence group + analog comparison
    -> 并行执行 group transferability reasoning
    -> 保留原始 group 输出用于审计
    -> 删除 group 模型推断出的任何数据源身份

冻结的 single prior + 脱敏后的 group 输出 + retrieval coverage
    -> task-specific 二分类最终预测
    -> JSON schema 验证
    -> 结构化输出无效时最多尝试 4 次
       （最后两次只对相同 JSON evidence 重新序列化）
    -> 如果清理后的 group payload 超过 750 KB，则对每个超限 neighbor
       采用确定性的等间距采样，最多保留 100 条 evidence row，并在 prompt 中记录
       原始行数、保留行数、字节数和采样方法
```

同一 task/query 的所有检索条件都会复用冻结的 single prior。因此，配对消融实验改变的只有检索证据及其组织方式。

在身份盲化制度中，身份控制会在两个 LLM 边界上执行。preflight gate 会拒绝 group prompt 中残留的任何已知结构、标识符或源分子同义词。原始 group response 会单独保存，因为即使 prompt 已脱敏，模型仍可能根据独特证据推断出 analog 名称；传递给 final synthesis 的版本会再次依据整个 retrieval 范围内的同义词集合进行清理。部署可见制度不运行该脱敏步骤，而是审计其正向 contract 是否满足。

## 任务配置

### Bioavailability_Ma

| 论文与 viewer 展示名称 | 内部 `group_id` | 证据语义 |
|---|---|---|
| Direct oral bioavailability (F%) | `Observed.direct_oral_bioavailability` | 绝对口服生物利用度 F；直接结果证据 |
| Oral exposure proxies (AUC/Cmax) | `Observed.oral_auc_cmax_exposure` | 口服 AUC/Cmax、剂量和制剂等系统暴露 proxy |
| Fa — Absorption, solubility & permeability | `Fa.absorption_solubility_permeability` | 吸收、肠道通透性、溶解度、溶出和胃肠道稳定性 |
| Fg — Gut-wall transport & intestinal metabolism | `Fg.gut_wall_efflux_intestinal_metabolism` | 肠壁转运、外排和肠道代谢 |
| Fh — Hepatic clearance & metabolic stability | `Fh.hepatic_clearance_metabolic_stability` | 首过提取、肝清除/内在清除和代谢稳定性 |

展示层使用左列名称；retrieval、trace、input hash 和 branch reuse 继续使用中间列的稳定内部 ID。

Starling 包含全部五类证据。ChEMBL endpoint group 会投影到同一组类别中。

### BBB_Martins

- 直接脑暴露/BBB 结果
- 被动通透性
- 外排转运
- 内流转运

ChEMBL 支持全部类别；Starling 只评估 direct evidence。

### Skin_Reaction

- 直接皮肤反应
- 致敏 AOP
- 光毒性、刺激性和局部损伤
- 皮肤暴露

实验条件和背景信息不作为独立 reasoning family。Dose、vehicle、formulation、duration、light exposure、
skin model 等信息继续随对应 evidence row 的 `text.context` / `scope` 进入 prompt。ChEMBL Tier 5 的
generic cytotoxicity、dermatology efficacy、target binding 等 weak background rows 保留在 source library
中用于审计，但不进入论文 `full_flat` 或 `full_mechanism` retrieval view。

### ClinTox

- 临床人体安全性
- 体内毒理学
- 器官特异性毒性
- 遗传毒性/致癌性
- 细胞应激通路
- 一般细胞毒性
- 脱靶、DDI 和暴露条件

## 评估输出

每个 batch 会写入每个 query 的 retrieval、single/group/final 输出、trace message、prediction、metric 和 manifest。论文汇总程序还会生成：

- accuracy、macro-F1、confusion matrix 和 95% bootstrap 区间
- 整体及各 group 的 retrieval coverage
- prompt/completion token、实际提供服务的模型、retry 次数和 prior 复用次数
- 身份盲化的泄漏审计，或部署可见制度的正向可见性 contract 审计
- 配对 macro-F1 差值和 95% 配对 bootstrap 区间
- 制度内和跨制度的精确双侧 McNemar 检验

Bioavailability 的标量 KNN 对照只使用数值型 direct-F 值，采用相同的 top-3/minimum-similarity 检索规则，对每个 neighbor 使用 20% 阈值，然后进行多数投票。它无法访问实验条件或定性文献文本。
