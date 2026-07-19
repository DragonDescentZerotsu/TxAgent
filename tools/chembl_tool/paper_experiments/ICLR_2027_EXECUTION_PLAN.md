# ICLR 2027 论文实验执行计划

## 目标与使用方式

本文档是从 2026-07-12 开始、面向 ICLR 2027 main conference 投稿的项目级执行计划。它回答：

- 论文要证明什么；
- 每个贡献由哪些实验支撑；
- 每个实验需要哪些数据、代码、算力、API 和人工资源；
- 当前已经有什么、还缺什么；
- 什么结果可以进入主表，什么只能作为补充或失败分析；
- 在预计 9 至 10 周的时间窗口内按什么顺序执行。

当前已完成的 63 个 GLM 条件是第一轮完整 exploratory matrix：21 个 identity-blind、21 个
matched-prefetch 和 21 个 deployment-visible agentic。它们用于确定研究问题、估算成本和
发现 failure modes，但不是自动成为最终论文主表。最终主表必须使用本文档冻结后的数据、排除规则、
模型版本、统计协议和完整性 gate。

本文档不是当前命令协议的替代品：

- 当前冻结矩阵和运行参数见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)；
- pipeline 数据流见 [PIPELINE.md](PIPELINE.md)；
- 已测结果见 [RESULTS.md](RESULTS.md)；
- visibility 升降原因见 [VISIBILITY_ANALYSIS.md](VISIBILITY_ANALYSIS.md)。

## 论文中心问题

论文不预设 retrieval、Starling 或 mechanism decomposition 必然提高所有任务。中心问题是：

> 对分子性质分类，结构化实验 evidence retrieval 在什么条件下能帮助 LLM agent；literature-derived
> evidence 与 curated database evidence 有何差异；mechanism decomposition 和 structure visibility
> 如何改变 evidence transfer、参数知识使用及错误模式？

如果最终结果仍与当前一样 mixed，论文应报告条件性结论和 failure boundary，而不是把点估计包装成
一致性增益。

## 核心贡献

| 编号 | 计划贡献 | 必须由什么证据支撑 |
|---|---|---|
| C1 | 通用、可审计的 molecular evidence agent harness | 4 个 task、2 个 source、统一 Evidence Contract、统一 runner、trace、validation、batch 和 audit 均实际运行 |
| C2 | ChEMBL curated evidence 与 Starling literature evidence 的受控比较 | 同 task/query/retrieval view 的 source 配对实验，加独立 source-quality annotation |
| C3 | Direct、flat、mechanism retrieval 的系统消融 | 相同 evidence row、只改变 grouping 的 flat-vs-mechanism 配对；none-vs-direct 测 retrieval |
| C4 | 真实 deployment-visible agentic workflow 及其可审计行为 | Agentic 主矩阵、工具调用/成本/trace；identity-blind 与 matched-prefetch 只作补充 attribution control |
| C5 | Parametric prior 与 retrieved evidence 的交互和失败分类 | 上升/下降 trace audit、身份核对、ontology conflict、重复运行与第二模型验证 |
| C6 | 完整 provenance、统计和可复现评估协议 | source manifest、entity relation/parent policy、coverage、tokens、失败数、bootstrap、McNemar、Holm 和 release checklist |

## 最终研究问题

| 编号 | 研究问题 | Primary comparison |
|---|---|---|
| RQ1 | Retrieval 是否帮助分类，且增益是否超出 same-entity lookup？ | Agentic `none` vs operational `direct`；再比较 operational vs parent-disjoint direct |
| RQ2 | Starling 是否优于 ChEMBL？ | `chembl_direct` vs `starling_direct_full`；`chembl_mechanism` vs `starling_mechanism` |
| RQ3 | Mechanism decomposition 是否优于 flat evidence？ | 同 source、同 evidence rows 的 `full_flat` vs `full_mechanism` |
| RQ4 | 非数值 Starling evidence 是否增加价值？ | `starling_direct_numeric` vs `starling_direct_full`，KNN 为独立对照 |
| RQ5 | Deployment-visible agent 如何调用和使用结构工具？ | Agentic tool-use、tokens、retries、trace；精选条件的 identity/matched attribution control |
| RQ6 | 观察到的增益和下降是否跨模型、跨重复稳定？ | 第二模型 confirmation matrix；GLM 关键条件重复运行 |

## 实验与贡献索引

| 实验 | 核心内容 | 对应贡献 | 优先级 | 当前状态 |
|---|---|---|---|---|
| E0 | 数据、same-parent relation、visibility 和运行完整性审计 | C1、C6 | P0 | 通用 identity/policy 和产物审计已实现；2026-07-17 完成同设置 valid 全矩阵诊断重跑：3 x 21 条件和 17 个 parent-disjoint 条件均 0 失败，prefetch 2,203 / 2,203 matched |
| E1 | None vs direct retrieval | C1、C3 | P0 | 第一轮完成；最终矩阵待重跑 |
| E2 | ChEMBL vs Starling | C2 | P0 | Bio/BBB direct 部分完成；数据不全 |
| E3 | Source-quality 人工 annotation | C2、C6 | P0 | 未开始 |
| E4 | Flat vs mechanism | C3 | P0 | 第一轮完成；全 Starling 未完成 |
| E5 | Deployment-visible agentic 主矩阵与工具行为分析 | C4、C5 | P0 | 当前 21 条件完成；缺 8 个 Starling 条件及最终数据冻结重跑 |
| E6 | Identity-blind/matched 与细粒度 visibility attribution | C4、C5 | P1 精选条件 | 21 对 matched 已完成；细粒度 policy 未实现 |
| E7 | Numeric vs non-numeric | C2 补充 | P1 | Bio 第一轮完成 |
| E8 | KNN、vote、ECFP learned、MiniMol、pretrained baselines | C1、C3 | P0 | MiniMol/KNN 部分完成 |
| E9 | 第二模型 confirmation | C1、C5 | P0 | 模型未冻结 |
| E10 | 关键条件重复运行 | C5、C6 | P0 | 未开始 |

## 当前资产与缺口

| 项目 | 当前状态 | 缺口或风险 | 截止 gate |
|---|---|---|---|
| 通用 runner、Evidence Contract、validation、trace、batch | 已实现并完成 63 个 GLM 条件 | 新实验不得复制 task pipeline | 持续 gate |
| ChEMBL 四任务 evidence | 已有 direct/flat/mechanism | 需要冻结最终 source manifest | 数据冻结前 |
| Bioavailability Starling | direct numeric/full、flat、mechanism 已有 | 检查最终 parquet 版本和 provenance | 2026-08-02 |
| BBB Starling | 当前只有 direct | 缺 passive permeability、efflux、influx families | 2026-08-02 |
| Skin Starling | direct 正在生成 | 缺 sensitization、phototoxicity/irritation/local damage、skin exposure 3 个 families | 2026-08-02 |
| ClinTox Starling | 当前没有论文矩阵数据 | 缺 7 个 mechanism families；ontology 风险高 | 2026-08-02 |
| Retrieval entity relation | `operational` / `parent_disjoint` 公共 policy 已实现；首轮 17 条 retrieval conditions 审计为 0 conflict | 最终数据冻结后重建 index 并复跑新增条件 | 2026-08-02 |
| Agentic deployment matrix | 21 条件完成 | 缺 8 个 Starling 条件；final parent policy/annotation 后 retrieval 条件需重跑 | 2026-08-23 |
| Source quality gold | 未建立 | 需要双人 annotation 和原始文献/assay 核验 | 2026-08-30 |
| Learned baselines | MiniMol 已有 | 缺 ECFP classifier、retrieval-only vote 和现代 pretrained baseline | 2026-08-16 |
| Cross-model evidence | 只有 GLM-5.2 | 缺第二个可复现模型 | 2026-08-23 |
| Run-to-run variance | 当前每个条件一次 | 关键比较需至少 3 次独立生成 | 2026-08-30 |
| Visibility failure audit | 第一轮已完成 | 需要第二 annotator 和 targeted causal ablation | 2026-08-30 |
| 论文主文与图表 | 未开始正式写作 | 不能等所有实验结束后才写 | 第一版 2026-09-06 |

## 数据冻结前的强制决策

### D1：Operational 与 parent-disjoint retrieval policy

真实部署中，同一 active moiety 的盐型、溶剂化物和 formulation-linked evidence 本身有价值，不应从
operational 主结果中静默删除；但它们也不能被称为普通 analog。数据冻结前实现并记录以下层级：

```text
exact_record:
  whole-record canonical SMILES 或 standard InChIKey 相同；两套 policy 都排除。

same_connectivity_variant:
  whole-record InChIKey connectivity block 相同，但立体、质子化等层不同；两套 policy 都排除。

same_parent:
  RDKit Cleanup + FragmentParent + Uncharger 后 parent InChIKey 相同，或一方主 parent 明确出现在另一方
  mixture components 中；operational 保留，parent_disjoint 排除。

structural_analog / unresolved:
  保留为候选并记录 relation。当前 normalizer 不做 tautomer 枚举，也不把共价 prodrug、代谢物或仅有语义关联的
  active moiety 自动折叠为 same parent，避免把真正的结构转化误当作同一实体。
```

最终论文报告两套 policy：

```text
operational:
  允许 same-parent evidence，但显式报告 relation、coverage 和结果分层；代表真实 deployment。

parent_disjoint:
  排除相同 parent 后继续向后检索，补足 top-k；用于证明 structural-analog retrieval 的独立价值。
```

Agentic operational 是主性能表；agentic parent-disjoint 是必须完成的主消融。Identity-blind 和
matched-prefetch 不需要为 parent 消融重跑完整矩阵。

验收标准：

- 每个 run manifest 写入 exclusion policy 和标准化版本；
- operational run 报告每个 relation 的 query/neighbor 数量和 performance stratum；
- parent-disjoint run 中 query 与 retained neighbor 的 parent key 审计为 0 冲突；
- 对 nelfinavir salt、propranolol salt、ChEMBL protonation counterpart 等已知案例建立测试；
- parent-disjoint 必须 backfill top-k，不能删除受影响 query 或只做 post-hoc 子集评分；
- relation annotation 或检索结果改变后，旧 retrieval 条件不得冒充最终冻结结果。

### D2：Starling 数据冻结

Starling 的生成单位必须是 mechanism family，不为每个 source-local endpoint group 单独运行昂贵 prompt。
最终需要：

```text
BBB:
  direct BBB outcome
  passive permeability
  efflux transport
  influx transport

Skin:
  direct skin reaction
  sensitization AOP
  phototoxicity / irritation / local damage
  skin exposure

ClinTox:
  clinical human safety
  in vivo toxicology
  organ-specific toxicity
  genotoxicity / carcinogenicity
  cellular stress pathways
  general cytotoxicity
  off-target / DDI / exposure context

Bioavailability:
  Direct oral bioavailability (F%)
  Oral exposure proxies (AUC/Cmax)
  Fa — absorption, solubility, and permeability
  Fg — gut-wall transport and intestinal metabolism
  Fh — hepatic clearance and metabolic stability
```

Skin 的实验条件和 background 不单独运行 Starling prompt。Concentration、vehicle、formulation、duration、
light exposure、skin integrity 和 skin model 等作为对应 evidence row 的 context/scope 字段保存。ChEMBL
Tier 5 weak-background rows也不进入 paper-facing flat/mechanism reasoning view。

每个 parquet/profile 必须记录 source version、生成 prompt/version、文献标识符、row counts、有效 SMILES
比例、去重后 molecule counts、endpoint/condition 字段覆盖率和无法解析原因。

### D3：模型和统计冻结

主模型继续使用当前 GLM-5.2 endpoint，但需要记录 endpoint 返回的实际 model identifier。第二模型必须在
2026-07-18 前按以下条件选择：可获得、允许保存输出、支持长 JSON context、可以完成结构化输出，且不能
仅因为 preliminary test 表现最好而选择。

Primary comparisons、bootstrap seed、Holm comparison family 和失败处理规则必须在新 full run 前冻结。

## 最终主结果与实验矩阵

论文主结果全部使用 query structure visible、query name hidden、neighbor structure/source identity visible
的 `deployment_visible` agentic workflow。主文预计包含：

```text
主表 1：Agentic operational retrieval
  四任务的 none、ChEMBL direct/flat/mechanism、Starling direct/flat/mechanism；
  Bioavailability 另含 Starling numeric direct。

主表或主图 2：Parent-disjoint analog ablation
  在同一 agentic workflow 中比较 operational 与 parent-disjoint；
  direct 是最低必做范围，flat/mechanism/source claim 涉及的条件也必须使用 parent-disjoint 候选池。

主表 3：Baselines 与成本
  MiniMol/modern encoder、ECFP model、retrieval-only vote、KNN，以及 agent tokens/tool calls。
```

Identity-blind 与 matched-prefetch 不进入主结果表。既有 21 对 parity-controlled 结果保留在 appendix 或
artifact 中，用于说明 visibility/tool-execution control，而不是决定 deployment 主结论。

Starling 补齐后，agentic operational 主矩阵有 29 个 GLM 条件：

| Task | None | ChEMBL direct | ChEMBL flat | ChEMBL mech | Starling direct | Starling flat | Starling mech | Starling numeric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BBB_Martins | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| Skin_Reaction | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| ClinTox | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| Bioavailability_Ma | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

总计：

```text
29 deployment-visible agentic operational conditions
25 deployment-visible agentic parent-disjoint retrieval conditions
1 visibility-independent scalar KNN
4 none conditions are shared because retrieval policy does not apply
```

当前已有 21 个 agentic operational 条件，缺 8 个 Starling 条件。若新增 entity-relation annotation 会进入
prompt，则最终冻结后需重跑 25 个 operational retrieval 条件；parent-disjoint 还需运行对应的 25 个
retrieval 条件。4 个 `none` 条件只有在 prompt、model 和 endpoint-returned model identifier 完全不变时
才能复用。既有 identity-blind/matched 42 条件不计入最终 agentic 主矩阵预算。

## 实验清单

### E0：数据、泄漏与运行完整性审计

**支撑贡献：C1、C6。优先级：P0，所有实验的前置依赖。**

执行内容：

1. 冻结 benchmark split hash、source files hash、profile version 和 evidence index manifest。
2. 运行 exact-record、whole-record connectivity variant、same RDKit molecular parent 三层 overlap audit；
   active-moiety/prodrug/metabolite relation 如需分析，必须作为独立人工或数据源 annotation，不能从 parent key 推断。
3. 检查 prompt visibility contract、operational relation annotation、parent-disjoint exclusion、failed branches、retry 和 truncation。
4. 检查 train/valid/test label distribution、duplicate connectivity 和冲突 label。
5. 生成一份 machine-readable release manifest。

资源：1 名工程人员，CPU/RDKit；预计 2 至 4 天。通用 parent normalizer 与首轮 overlap audit 已完成；
最终 source freeze 后仍需重跑审计，并补 active-moiety 独立 annotation（如论文需要）和 release manifest。

完成标准：所有进入主表的条件 `n_failed_runs=0`，agentic visibility contract 通过；operational relation
计数可审计，parent-disjoint conflict 为 0，source/split hash 可复现。

### E1：Similar-molecule retrieval 主实验

**支撑贡献：C1、C3。回答 RQ1。优先级：P0。**

Primary comparison：在 agentic setting 中比较每个 task/source 的 `none` vs operational `direct`，并比较
operational `direct` vs parent-disjoint `direct`。同时报告 coverage、neighbor similarity、entity relation、
排除后候选数量、top-k backfill 成功率和 paired flips。

资源：完整 test sets；GLM API；预计属于 agentic operational + parent-disjoint 主矩阵。当前 ChEMBL 和 Bio/BBB 部分 Starling 已有，
Skin/Clin/BBB full Starling 未完成。

完成标准：不只报告平均提升；必须给每个任务的 effect size、95% CI、McNemar/Holm、coverage 和
failure taxonomy。若任务间不一致，结论写成 task-conditional。

### E2：ChEMBL 与 Starling source comparison

**支撑贡献：C2。回答 RQ2。优先级：P0。**

对每个 task 配对比较：

```text
chembl_direct vs starling_direct_full
chembl_full_flat vs starling_full_flat
chembl_full_mechanism vs starling_full_mechanism
```

所有 primary source comparison 使用同一个 agentic protocol。Operational 结果回答真实数据源效用；
parent-disjoint 结果检查 source 差异是否主要来自 same-parent coverage。若二者结论不同，必须同时报告，
不能只选择更支持某一 source 的 policy。

必须并列报告 source coverage、每个 molecule 的 evidence count、condition completeness、provenance
availability、retrieved-neighbor overlap 和 token cost。不能把 coverage 差异静默解释成 row quality。

资源：补齐 Starling；agentic operational 当前缺 8 个条件，最终与 parent-disjoint 消融一起运行。Starling 生成成本单独
记账，不与 GLM reasoning tokens 合并。

完成标准：将“source quality”和“downstream performance”分开下结论；只有跨任务方向稳定时才声称
Starling generally better，否则报告在哪些 evidence families 上更好。

### E3：Source-quality 人工 annotation

**支撑贡献：C2、C6。回答 RQ2 的数据质量部分。优先级：P0。**

抽样单位是最终进入 retrieval library 的 evidence record。目标抽样 400 条：每 task × source 各 50 条；
最低可接受规模 200 条。按 direct/mechanism、numeric/non-numeric、species/route/formulation 分层。

Annotation 字段：

```text
molecule attribution correct
endpoint/relation/value/unit correct
species/route/dose/formulation captured
evidence text supports structured fields
mechanism family assignment correct
provenance resolves to original assay/paper
critical / major / minor error
```

资源：2 名具备药化/PK/tox 背景的 annotator；目标 400 条全量双标，最低要求至少 100 条重叠双标；
预计 60 至 120 person-hours。需要原始文献访问和 annotation guideline。

完成标准：报告每项正确率和 95% CI、Cohen's kappa 或适用的一致率、分歧仲裁流程。发现 critical
attribution/endpoint error 后先修数据再运行主矩阵，不能只在论文中备注。

### E4：Flat 与 mechanism decomposition

**支撑贡献：C3。回答 RQ3。优先级：P0。**

每个 source/task 的 flat 与 mechanism 必须使用相同 evidence row 集合；只允许 grouping、并行 branch
和 final synthesis 结构不同。主分析：paired Macro-F1；辅助分析：group coverage、group disagreement、
evidence duplication、token cost、failed branch 和 final evidence utilization。

资源：agentic 主矩阵中的 flat/mechanism 条件；GLM API。Operational 与 parent-disjoint 都必须保证 flat
和 mechanism 使用相同 evidence row 集合。当前结果已证明 gain 可能很小，因此不允许在
test 上继续改 mechanism prompt。

完成标准：若 gain 不稳定，改写为“mechanism decomposition 的收益依赖 family coverage/quality”，并
检验 gain 与 coverage、duplicate burden、group disagreement 的关系。

### E5：Deployment-visible agentic 主矩阵

**支撑贡献：C4、C5。回答 RQ5。优先级：P0。**

对最终 29 个条件运行 deployment-visible agentic workflow，报告 Macro-F1、accuracy、per-class F1、
confusion matrix、paired flips、token cost、tool-call/retry、身份声明、source-ID 使用和 evidence utilization。
Same-parent relation 必须进入 trace/audit；parent-disjoint 使用相同 agent protocol 和冻结 prompt。

资源：当前 21 个 agentic 条件已完成，缺 8 个 Starling 条件；数据和 relation contract 冻结后按需要重跑
25 个 operational retrieval 条件，并运行对应 parent-disjoint 条件。既有 21 对 identity-blind/
matched-prefetch、4,456 个样本 parity audit 保留为 appendix control，不扩成新的完整 29 对矩阵。

完成标准：主表只比较同一 agentic regime 内的 retrieval/source/grouping policy；不得把 agentic 与
identity-blind 的差异解释为纯 structure visibility。ClinTox 等不平衡任务必须同时检查 accuracy、
Macro-F1 和 flip direction。

### E6：Visibility attribution 小规模因果消融

**支撑贡献：C4、C5。回答 RQ5。优先级：P1，只运行精选条件。**

在 BBB、Bioavailability 各选一个上升条件，在 Skin、ClinTox 各选一个下降条件，运行：

```text
A: identity-blind
B: query structure visible; neighbor identity/structure hidden
C: query hidden; neighbor structure visible; source ID hidden
D: query hidden; neighbor structure and source ID visible
E: full deployment-visible
```

条件必须在 valid 或预先冻结的 discordant-analysis subset 上选择，不能根据 test 单样本反复调 prompt。
该实验区分 query parametric identity、neighbor structural transfer 和 source-ID memory。

资源：4 个代表条件 × 5 visibility variants；该实验属于 appendix/diagnostic，可先只跑原条件中发生预测翻转的样本，再对方向明确的
variant 跑完整 test set。需要新增通用 visibility policy，当前未实现。

完成标准：至少能回答 observed delta 主要来自 query structure、neighbor structure 还是 source ID；
如果差异不稳定，明确报告 attribution unresolved。

### E7：Numeric 与 non-numeric Starling evidence

**支撑贡献：C2 的补充分析。回答 RQ4。优先级：P1。**

保留简单的二分：`numeric_only` 与 `full`。主实验只在 Bioavailability direct-F 上进行；不增加更多
内容等级。比较 scalar KNN、numeric GLM 和 full GLM。

资源：当前结果已存在；operational relation contract 和 parent-disjoint policy 冻结后重跑。若最终 CI 仍跨 0，结论是没有证据
证明 non-numeric evidence 必然提升，不应作为主标题卖点。

### E8：非 LLM 与 learned baselines

**支撑贡献：C1、C3。回答 agent 是否超过简单替代方法。优先级：P0。**

必须包含：

```text
ECFP Tanimoto KNN classification
retrieval-only weighted vote using identical neighbors
RF or XGBoost on frozen ECFP features
MiniMol existing baseline
one modern pretrained molecular encoder selected without test tuning
```

所有 supervised baseline 只用 train 训练、valid 选配置、test 一次评估。Retrieval-only baseline 使用与
agent 对应 policy 下的相同 neighbors，防止比较不同候选池。Operational 与 parent-disjoint baseline 分开报告。

资源：1 至 4 张 GPU 足够完成 learned baseline；CPU 可跑 KNN/RF/XGBoost。当前只有 MiniMol 和
Bioavailability scalar KNN，其他入口未实现。

完成标准：报告 Macro-F1、accuracy、AUROC（有 score 时）、训练/推理成本。Agent claim 必须说明它
在哪些任务优于 learned model，以及 retrieval/trace 提供了什么额外能力。

### E9：第二模型 confirmation matrix

**支撑贡献：C1、C5。回答 RQ6。优先级：P0。**

不重跑第二个完整 agentic 矩阵。冻结 8 个 deployment-visible agentic 条件：

```text
BBB and Bioavailability:
  none, starling_direct, starling_mechanism
  6 conditions

Skin and ClinTox:
  starling_mechanism
  2 conditions
```

如 Starling 某任务最终不可用，必须在数据冻结前用对应 ChEMBL 条件替换，不能按结果替换。

资源：第二个长上下文模型 endpoint 或本地多 GPU；当前模型未选择。预计成本取决于模型，必须单独记录。

完成标准：比较 effect direction 和 paired flips，而不是要求绝对分数相同。若主要结论只在 GLM 成立，
论文明确标为 model-specific。

### E10：关键条件重复运行

**支撑贡献：C5、C6。回答 RQ6。优先级：P0。**

从预注册 primary comparisons 中选择 6 个 agentic 条件对：BBB retrieval gain、BBB parent ablation、
Bio Starling source、Bio mechanism、Skin failure、ClinTox failure。每个 condition 总计至少 3 次独立生成；
当前 run 可作为第 1 次，前提是数据、retrieval policy、prompt 和 model identifier 未改变，否则从头计数。

资源：约 24 个额外 condition runs（6 对 × 2 conditions × 2 additional repeats）；GLM API。预计是除主矩阵
外最大的 token 成本。

完成标准：报告 mean、SD、每次 paired delta、prediction disagreement 和结论方向稳定率。不能用表现最好
的一次作为主结果。

## 资源预算

### 计算与 API

当前三套 exploratory regime 共使用约 427M tokens。最终只扩展 deployment-visible agentic 主矩阵，并
增加 parent-disjoint 消融；初步按相同 prompt 规模规划：

```text
final agentic operational + parent-disjoint matrix: approximately 250M-450M tokens
key-condition repeats: approximately 100M-200M tokens
second-model confirmation: model-dependent, budget separately
Starling extraction: separate literature-processing budget
```

这些是容量规划区间，不是承诺用量。数据冻结后先用每个新增 family 的 5-sample smoke 实测 token，再更新
预算。禁止为了省 token 在正式 run 中临时改变 top-k、evidence rows 或 prompt。

### 人员

```text
1 engineering owner:
  ingestion, parent relation/exclusion, runner, manifests, baselines, reruns

2 domain annotators:
  source-quality gold and visibility identity audit

1 statistics/paper owner:
  frozen comparisons, figures, tables, writing and internal review
```

同一人可以承担多个角色，但 source-quality gold 至少需要第二位独立 annotator。

### 存储与发布

需要保存 evidence manifest、indices、predictions、metrics、trace、annotation 和 figure source data。正式发布
前检查 ChEMBL、Starling 原始文献文本和模型输出的许可；不能默认完整文献 text 可以公开。

## 实际执行顺序与命令边界

严格按以下依赖顺序执行：

```text
D1 operational/parent-disjoint policy + D3 protocol freeze
  -> D2 Starling data freeze
  -> E0 index/overlap/contract smoke
  -> E3 annotation sampling and E8 baselines in parallel
  -> E1/E2/E4/E5 primary matrix
  -> E9 second model and E10 repeats
  -> E6 targeted visibility attribution
  -> final statistics, figures and writing
```

当前已经存在并可以使用的统一入口：

```bash
# 查看当前 runner 已注册的实验
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent --list

# 运行论文主实验的 deployment-visible agentic 条件
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent \
  --visibility-mode deployment_visible \
  --experiments <experiment_id> [<experiment_id> ...]

# 生成统一统计和报告
python -m tools.chembl_tool.paper_experiments.summarize_results
```

Starling 当前可用 builder 命令见 [README.md](README.md)。以下内容尚无最终可运行入口，必须先实现并
通过测试，不能把本文档中的名称直接当作 CLI 参数：

- operational entity-relation annotation、parent-disjoint exclusion/backfill 与 overlap audit；
- BBB full、Skin、ClinTox 的最终 Starling family builders/profiles；
- E6 的细粒度 query/neighbor/source-ID visibility policies；
- E8 尚缺的 retrieval vote、ECFP learned 和 pretrained baseline runners；
- E3 annotation sampler、schema、agreement 和 adjudication summarizer；
- E9/E10 的模型/重复 run manifest 与聚合报告。

每实现一个入口，必须先补测试和 `--help`/README 命令，再把本列表对应项删除并更新 E 编号状态。

## 执行时间表

ICLR 2027 官方 deadline 尚未发布。本时间表按往年 9 月中下旬 abstract/full-paper deadline 倒排，并保留
至少 7 天 buffer；官方日期发布后立即调整。

| 时间 | 必须完成 | Go/No-Go gate |
|---|---|---|
| 07-12 至 07-18 | 冻结 thesis、primary comparisons、两套 parent policy、第二模型和 annotation guideline | 未冻结则不生成新 Starling full data |
| 07-19 至 08-02 | 补齐 BBB/Skin/Clin Starling；source manifest；开始人工 annotation | 数据不完整或 critical attribution error 未修则不跑主矩阵 |
| 08-03 至 08-09 | 构建 index、5-sample smoke、parent overlap audit；完成 baseline 入口 | contract/overlap/失败 gate 未通过则不 full run |
| 08-10 至 08-23 | Agentic operational/parent-disjoint 主矩阵、8 条件第二模型；持续完整性检查 | 08-23 必须得到可汇总主结果 |
| 08-24 至 08-30 | 关键重复、visibility attribution、人工 annotation 仲裁、最终统计 | 结论与 claim 不符时改 claim，不改 test prompt |
| 08-31 至 09-06 | 完成完整初稿、主图、主表、appendix 和 limitations | 内部读者必须能只靠文稿复现实验逻辑 |
| 09-07 至 09-13 | 内部 review、匿名化、artifact/license 检查、只修 blocker | 不再增加新核心实验 |
| 官方截稿前 | 处理排版、引用、OpenReview profile/abstract 和提交检查 | 保留至少 7 天 buffer |

## 每周状态模板

每周只更新以下内容，避免用零散聊天替代项目状态：

```text
本周完成:
  E 编号、run IDs、数据版本、关键输出

当前指标:
  primary effect sizes、CI、coverage、failures、tokens

阻塞项:
  owner、需要的资源、最晚解决日期

下周计划:
  按依赖顺序列出，不超过 5 项

论文 claim 变化:
  哪个 claim 被支持、削弱或需要重写
```

## 投稿前硬性清单

- [ ] 29 个 agentic operational 条件和预注册 parent-disjoint 条件完成，或对任何缺失条件给出数据不可得的预注册说明。
- [ ] 所有主表 run 使用冻结的 operational/parent-disjoint contract 和 source manifest。
- [ ] 主表 `n_failed_runs=0`，agentic visibility contract 通过；parent-disjoint conflict 为 0。
- [ ] Source-quality annotation 完成双人一致率和分歧仲裁。
- [ ] KNN、retrieval vote、ECFP learned、MiniMol、pretrained encoder baseline 完成。
- [ ] 第二模型 confirmation 和 6 个关键 comparison repeats 完成。
- [ ] Visibility gain/failure taxonomy 由第二 annotator 复核。
- [ ] 每个主 claim 都能指向一张主表或一幅主图，不依赖 anecdotal trace。
- [ ] Mixed/negative result 被如实写入 abstract、limitations 和 conclusion。
- [ ] 文稿、仓库、artifact、日志和引用全部匿名化并完成 license 检查。

## 最低可投稿标准

如果时间不足，不能通过删除负结果来缩小范围。最低可投稿版本必须保留：

1. 四任务完整 ChEMBL/Starling direct comparison；
2. 至少 BBB 和 Bioavailability 的完整 flat/mechanism comparison；
3. Agentic operational 主结果和 parent-disjoint analog 消融；
4. source-quality annotation；
5. 强 baseline、第二模型最小 confirmation、关键重复；
6. Identity-blind/matched 的最小 appendix control，以及完整 provenance、paired statistics 和 failure analysis。

若 Starling 无法按时补齐 Skin/Clin mechanism，应在数据冻结 gate 做一次明确 scope reduction，并将论文
主张收缩为“跨任务 direct source comparison + 两个任务的 mechanism case study”，而不是在截稿前留下
不完整矩阵。
