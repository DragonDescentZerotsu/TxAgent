# ICLR 2027 论文实验执行计划

## 目标与使用方式

本文档是从 2026-07-12 开始、面向 ICLR 2027 main conference 投稿的项目级执行计划。它回答：

- 论文要证明什么；
- 每个贡献由哪些实验支撑；
- 每个实验需要哪些数据、代码、算力、API 和人工资源；
- 当前已经有什么、还缺什么；
- 什么结果可以进入主表，什么只能作为补充或失败分析；
- 在预计 9 至 10 周的时间窗口内按什么顺序执行。

### 2026-08-01 v4 主矩阵决策

`record_agreement70_split811_v1` 及后续新数据集的正式默认改为
`identity_blind + parent_disjoint` fresh-run。Operational、deployment-visible 和 matched-prefetch 不再是
主矩阵前置依赖，只作为历史结果或显式 ablation。默认 endpoint 为本机 tunnel
`http://127.0.0.1:50000/v1` 上的 `nvidia/GLM-5.2-NVFP4`，保持历史 GLM reasoning 设置；全局 endpoint
并发预算上限为 512；首次压力运行出现 1/500 transport timeout 后，单 launcher 默认形状
先调整为 384；由于 BBB full-flat 仍出现大量长 group-request timeout，当前默认进一步调整为
单一 `parallelism=128` global prompt pool，禁止外层 fan-out 乘法超额。

这一决策覆盖下文基于 2026-07 historical operational/deployment-visible 矩阵的“主表”措辞，但不删除历史
结果或 relation taxonomy。新 v4 先跑 valid 并通过完整性、identity leak、parent conflict 和 held-out overlap
gate，冻结后再跑 test。blind+parent-disjoint fresh-run、独立 root 和全局并发实现 gate 已完成；
scaffold-valid 的 GLM、GPT-OSS-20B/120B blind 矩阵、GPT 两套 visible 矩阵与 matched baselines 已完成，
GLM visible 也已达到 6887/6887 严格成功并进入同一总图。正式 test 仍须等 valid 合同和 failure contingency
冻结后才可启动。

当前 test 已完成 73 个 GLM 条件：26 个 identity-blind、21 个 matched-prefetch 和 26 个
deployment-visible agentic；valid 三套制度均为 26 个条件，共 78 个。它们用于确定研究问题、估算成本和
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
| C2 | ChEMBL curated structured assay records 与 Starling 从原文抽取、保留 source context 的 evidence records 的受控比较 | 同 task/query/endpoint scope/retrieval view 的 source 配对实验，加独立 source-quality annotation |
| C3 | Direct、flat、mechanism retrieval 的系统消融 | 相同 evidence row、只改变 grouping 的 flat-vs-mechanism 配对；none-vs-direct 测 retrieval |
| C4 | 可审计的 identity-blind evidence-reasoning workflow | Blind parent-disjoint 主矩阵、harness-prefetched 工具证据、成本/trace；deployment-visible 只作补充 deployment control |
| C5 | Parametric prior 与 retrieved evidence 的交互和失败分类 | 上升/下降 trace audit、身份核对、ontology conflict、重复运行与第二模型验证 |
| C6 | 完整 provenance、统计和可复现评估协议 | source manifest、entity relation/parent policy、coverage、tokens、失败数、bootstrap、McNemar、Holm 和 release checklist |
| C7 | ChEMBL 内 evidence distance 与 quantity 的累计扩展曲线 | 冻结的当前 direct/mechanism envelope、task-specific graph、`D/C/H1/H2` flat 主曲线、H1/H2 mechanism supplementary、matched-prefetch 控制和 agentic confirmation |

## 最终研究问题

| 编号 | 研究问题 | Primary comparison |
|---|---|---|
| RQ1 | Retrieval 是否帮助分类，且增益是否超出 same-entity lookup？ | Identity-blind `none` vs parent-disjoint `direct` |
| RQ2 | 在 endpoint scope 匹配时，Starling 原文抽取 evidence 与 ChEMBL curated structured records 的效用和质量有何差异？ | `chembl_direct` vs `starling_direct_full`；可匹配的 current-mechanism families 内比较 `chembl_mechanism` vs `starling_mechanism` |
| RQ3 | Mechanism decomposition 是否优于 flat evidence？ | 同 source、同 evidence rows 的 `full_flat` vs `full_mechanism` |
| RQ4 | 非数值 Starling evidence 是否增加价值？ | `starling_direct_numeric` vs `starling_direct_full`，KNN 为独立对照 |
| RQ5 | Identity-blind 模型如何使用 harness-prefetched 结构工具证据？ | Blind tool evidence、tokens、retries、trace；精选条件的 deployment-visible ablation |
| RQ6 | 观察到的增益和下降是否跨模型、跨重复稳定？ | 第二模型 confirmation matrix；GLM 关键条件重复运行 |
| RQ7 | Retrieval coverage 与 macro-F1 增幅有什么关系？ | Blind parent-disjoint retrieval 条件的 overall/positive-class/negative-class coverage 与相对同任务 `none` 的 paired macro-F1 差值 |
| RQ8 | 超出当前 curated mechanism envelope 多远后，增加更多 assay evidence 不再帮助 LLM？ | ChEMBL-only 的 `none`、`D`、`D+C`、`D+C+H1`、`D+C+H1+H2` 累计扩展曲线 |
| RQ9 | Retrieval 差异是否在 group-to-final 汇总中被压缩？ | 固定 retrieval/single/group artifact 的 `summary_only`、`summary_plus_cards`、`cards_only` final-only 配对消融 |

## 实验与贡献索引

| 实验 | 核心内容 | 对应贡献 | 优先级 | 当前状态 |
|---|---|---|---|---|
| E0 | 数据、same-parent relation、visibility 和运行完整性审计 | C1、C6 | P0 | 通用 identity/policy 和产物审计已实现；2026-07-23 valid 扩展矩阵为 3 x 26 条件和 22 个 parent-disjoint 条件，均 0 失败，prefetch 2,713 / 2,713 matched |
| E1 | None vs direct retrieval | C1、C3 | P0 | v4 scaffold-valid 的 GLM、GPT-OSS-20B/120B blind 矩阵完成；GLM random-valid 余 2 个失败，正式 test 未启动 |
| E2 | ChEMBL vs Starling | C2 | P0 | BBB、Bio、Skin 的 v4 scaffold-valid 三模型 source comparison 已完成；ClinTox 因无同定义 Starling direct source 不进入当前 v4 |
| E3 | Source-quality 人工 annotation | C2、C6 | P0 | 未开始 |
| E4 | Flat vs mechanism | C3 | P0 | BBB/Bio/Skin v4 scaffold-valid 三模型 blind 配对已完成；coverage context pilots 为 mixed/no-go，正式 test 待跑 |
| E5 | Deployment-visible agentic 补充矩阵与工具行为分析 | C4、C5 | P0 | GPT-OSS-20B/120B 和 GLM scaffold-valid visible 各完成 22 条件；均为显式 ablation 而非主矩阵 |
| E6 | Blind/visible 合同差异与细粒度 visibility attribution | C4、C5 | P1 精选条件 | GPT-OSS 两模型已有完整 blind/visible valid 对照，但同时改变 identity 与 tool execution；细粒度单因素 policy 未实现 |
| E7 | Numeric vs non-numeric | C2 补充 | P1 | Bio 第一轮完成 |
| E8 | KNN、vote、ECFP learned、MiniMol、pretrained baselines | C1、C3 | P0 | 历史 strict-conflict full-test baselines 与 v4 scaffold-valid 的 MiniMol/Morgan/MiniMol-KNN 已完成；其余 baseline 待补 |
| E9 | 跨模型 confirmation | C1、C5 | P0 | 同一 v4 scaffold-valid blind contract 的 GLM、GPT-OSS-20B/120B 已完成；GPT 两模型 visible 已完成，独立模型家族与重复运行待补 |
| E10 | 关键条件重复运行 | C5、C6 | P0 | 未开始 |
| E11 | Coverage–performance 关联分析 | C3、C6 | P0 | 第一轮 test/valid 各 17 个 agentic retrieval 条件已完成；TSV/JSON/report/canonical SVG 和 class-conditional coverage 已生成，最终矩阵冻结后需重跑 |
| E12 | ChEMBL mechanistic-distance / evidence-quantity expansion | C7、C6 | P0 | C-family tree 协议已重冻；通用旧 graph/index/retrieval prototype 已实现，但 BBB v2 因 MMP-3 shortcut 只保留历史审计，四任务 tree mapping 与 batch/LLM runner 均待完成 |
| E13 | Final evidence surface / aggregation bottleneck | C3、C6 | P0 诊断 | record-supported v2 scaffold-valid 已完成 1,908/1,908、contract audit 0 failure；六个 paired CI 均跨 0，card surface 不升级默认 |

## 当前资产与缺口

| 项目 | 当前状态 | 缺口或风险 | 截止 gate |
|---|---|---|---|
| 通用 runner、Evidence Contract、validation、trace、batch | 已实现；v4 新增 model-specific roots、global prompt pool、selector/context profiles、strict watchdog 和 persistent tool cache | 新实验不得复制 task pipeline；v4 不得使用旧 TDC summarizer | 持续 gate |
| ChEMBL 四任务 evidence | 已有 direct/flat/mechanism | 需要冻结最终 source manifest | 数据冻结前 |
| ChEMBL distance expansion | C-family tree ontology 已冻结；通用旧 config/validator、assay-manifest、superset index 和 cumulative retrieval prototype 已实现 | 需先升级 tree-node schema/retrieval；BBB v2 需重建，另外三任务 graph/mapping 与独立 batch/replay、汇总、绘图 runner 均缺失 | 数据冻结前 |
| Bioavailability Starling | canonical direct v2、direct numeric/full、flat、mechanism 与 v4 held-out index 已冻结 | source-quality 人工核验和正式 test 尚缺 | 数据冻结 gate 已过 |
| BBB Starling | direct + passive permeability + efflux + influx、v4 held-out index 和 scaffold-valid blind/visible 已完成 | 正式 test 与 source-quality 尚缺 | 数据冻结 gate 已过 |
| Skin Starling | direct + sensitization + phototoxicity/irritation/local damage + skin exposure、v4 held-out index 和 scaffold-valid blind/visible 已完成 | 正式 test 与 source-quality 尚缺 | 数据冻结 gate 已过 |
| ClinTox Starling | 无与 toxicity-caused clinical-trial failure 同定义的 direct source，当前明确不构造 v4 split | 若未来新增 source 必须另做 ontology/source freeze，不能补几个 family 后混入当前 lineage | future lineage |
| Retrieval entity relation | `operational` / `parent_disjoint` 公共 policy 已实现；现有 22 条 retrieval conditions 在 test/valid 均完成且为 0 conflict | 最终数据冻结后若 index 改变需重建与复跑 | 2026-08-02 |
| V4 blind main matrix | GLM、GPT-OSS-20B/120B scaffold-valid 各 22 条件完成；GLM 0 failure，两个 GPT 各 1 个 context-limit failure 按预定 policy 计错 | random-valid/正式 test、repeats 和 release audit 待完成 | 2026-08-23 |
| Visible deployment ablation | GPT-OSS-20B/120B 与 GLM scaffold-valid 同合同各 22 条件均已完成 | 完整合同同时改变 identity 与 tool execution，不能当纯 visibility effect | 已完成；因果拆分仍待 E6 |
| Source quality gold | 未建立 | 需要双人 annotation 和原始文献/assay 核验 | 2026-08-30 |
| Learned baselines | 历史 strict-conflict random/scaffold full-test baselines 已完成；当前 v4 已完成 scaffold-valid MiniMol train-all、Morgan KNN 和 MiniMol embedding cosine KNN | 缺 v4 random/test、ECFP RF/XGBoost、matched-neighbor retrieval-only vote 和独立的第二种 pretrained encoder baseline | 2026-08-16 |
| Cross-model evidence | v4 scaffold-valid 已完成 GLM、GPT-OSS-20B 与 GPT-OSS-120B blind 同合同对照 | 缺独立模型家族、关键条件重复和正式 test | 2026-08-23 |
| Run-to-run variance | 当前每个条件一次 | 关键比较需至少 3 次独立生成 | 2026-08-30 |
| Visibility failure audit | 第一轮已完成 | 需要第二 annotator 和 targeted causal ablation | 2026-08-30 |
| 论文主文与图表 | 未开始正式写作 | 不能等所有实验结束后才写 | 第一版 2026-09-06 |

## 数据冻结前的强制决策

### D1：Parent-disjoint 主 policy 与 historical operational taxonomy

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

新 v4 主结果只要求 parent-disjoint；operational 可作为显式 sensitivity 另行报告：

```text
operational:
  允许 same-parent evidence，但显式报告 relation、coverage 和结果分层；代表真实 deployment。

parent_disjoint:
  排除相同 parent 后继续向后检索，补足 top-k；用于证明 structural-analog retrieval 的独立价值。
```

Identity-blind parent-disjoint 是 v4 主性能表。Operational/deployment-visible 若运行，必须写入独立 ablation
root，且不得成为主矩阵的 staging 或 reuse-plan 来源。

验收标准：

- 每个 run manifest 写入 exclusion policy 和标准化版本；
- 若显式运行 operational，则报告每个 relation 的 query/neighbor 数量和 performance stratum；
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

### D4：ChEMBL mechanistic-distance graph 与累计 source levels 冻结

E12 的 `D`、`C`、`H1`、`H2` 全部只使用同一冻结版本的 ChEMBL；不是只让新增的 one-hop/two-hop 层使用
ChEMBL。`D/C` 从当前 ChEMBL direct/full evidence 中冻结，`H1/H2` 从同版 ChEMBL 扩展，不为任何一层运行
Starling，也不得构造 `Starling D/C + ChEMBL H1/H2` 或其它跨 source 混合累计曲线。四层共享 ChEMBL release、
assay/activity 标准化、measurement-quality gate、molecule aggregation、provenance contract 和 source manifest，
以便在扩大 evidence quantity 时把 source/curation pipeline 固定不变。Starling 只参与 E2 的 matched-scope source
comparison；不得将 Starling direct 与 ChEMBL distant-expanded condition 比较后归因为 source quality。

Distance 不在单条 assay 文本上凭相似度定义，而在每个 task 的预注册 mechanism-family graph 上定义。
`D` 是层级树 root，当前 paper-facing C mechanism families 是第一圈子节点；H1/H2 是每条 C branch 向外展开的
depth，不是与 C 平行的全局 source groups：

```text
D root
  C.family_k
    H1.family_k: exactly one aggregate child; 可聚合多个到 C_k 一跳的 measurement families
      H2.family_k: zero or one aggregate child; 可聚合多个到 H1_k 一跳的 measurement families
```

必须满足 `D ∩ C = empty`，且 `D ∪ C` 逐 source group 精确等于当前 full evidence union。C 内现有 families
可能离 direct outcome 具有不同的绝对机制距离；E12 不重新给它们贴 one-hop/two-hop 标签。每个 task 必须先
对全部 D/C assay 做 measured-state census；extension assay ID 和 measured node 均不得与 base 重叠。若某个
upstream readout 已经被 coarse C family 收入，即使其 endpoint-group 名称没有显式写出该 node，也仍属于 B。
每个 C family 必须有唯一 H1 child spec，H1 至多有一个 H2 child；child node 可聚合不同 targets/assays，但每个
measurement family 必须唯一归属一个 C parent。H1 到 parent C 的最短路径为 1；H2 到其 H1 parent 有一条边且到
整个 B 的最短路径为 2。H2 若对任意 D/C node 存在一跳捷径，必须重分到相应 H1 或排除。

一条合格推理桥连接两个命名清楚、可测量且粒度相近的 biological state/process，并记录上游、下游、关系类型、
允许的 inference direction、适用条件和支持文献。默认只顺 causal direction 推断，只有经过验证的 mechanistic
readout 才允许反向 inference。只允许 causal/predictive process、functional component、exposure link 或已经验证的
mechanistic readout；仅仅同属一个器官、疾病、pathway、target class、assay format，或只在统计上相关，都不算
一条边。若 A 到 C 实际需要经过 B，除非存在独立的 A 到 C 直接证据，否则不能把两条边压成一条。

每条 ChEMBL assay 先根据 assay description、target、measurement、organism/system 和 task endpoint rules
映射到唯一 measurement family，再由该 family 所测 biological node 到冻结 envelope 的最短合格路径确定 level。
物种/组织/模型可迁移性记录为独立 `scope_match`，measurement/curation 可靠性记录为独立 `quality_status`；两者
都不增加 hop。`out_of_scope` 或未通过统一质量门槛的 assay 不进入任何 level。映射不确定、路径方向不清、
存在更短路径冲突或只能靠跨任务 label 猜测的 assay 不进入主实验。

当前没有外部领域专家，因此先采用 citation-backed 两遍独立审计：第一遍建立 node/edge/family mapping，第二遍
在不读取第一遍 level 结论的情况下重新推导最短路径；不一致项进入 `unresolved` 并排除。该流程不能冒充外部
expert annotation，论文中需如实披露，并把 graph、edge rationale、候选/纳入/排除 assay counts 全部发布。
所有 mapping 在查看 test performance 前冻结；允许用 train/valid 做 coverage、payload 和工程 smoke，不能用
valid/test macro-F1 改 graph、edge 或 distance level。

累计 evidence union 固定为：

```text
none
D
D + C
D + C + H1
D + C + H1 + H2
```

每个 C family 在 graph freeze 前必须找到可信 H1，否则 task graph 不能通过；某个 H1 没有可信 H2 时，将该
family 的 H2 显式记为 unavailable，不得为了形成完整曲线强行纳入弱相关 assay。
主曲线使用一致的 flat assembly、相同 similarity threshold/top-k policy 和冻结的 single prior，避免把
新增 mechanism branches 本身误当成 distance effect。

除 flat 主曲线外，每个通过 tree graph freeze 的 task 必须运行 mechanism supplementary：

```text
D + C                    current mechanism baseline
D + C + H1               reuse D/C branches; run H1 branches; rerun final
D + C + H1 + H2          reuse D/C/H1 branches; run H2 branches; rerun final
```

`D+C+H1+H2` 只加入各 family 中 available 的 H2 children；若所有 C families 的 H2 均 unavailable，task 停在
`D+C+H1`。同一 prefix 的 flat 与 mechanism 必须使用完全相同的 evidence union，只改变
reasoning organization。每个 C-family H1 tree node 的 group ID、retrieved neighbors、LLM-visible evidence 和 branch input hash
在加入 H2 后必须保持不变，否则不得声称复用 H1；任何 prefix 扩展都会改变 final 输入，因此 final 必须重跑。
完整 matched-prefetch supplementary 同时包含 H1 与 H2 两点；deployment-visible agentic 也运行这两个 mechanism
expansion points，并优先按 branch hash 复用当前 `D+C` 和前一 prefix 的 outputs。

每个 C/H1/H2 tree node 独立最多检索 3 个 unique molecular neighbors，并使用相同 similarity threshold 和 identity
policy。H1/H2 聚合 node 内的多个 target/measurement families 共享这 3 个位置，不得按 target 分别取 top-3。

完整定义、配置 schema、代码隔离边界、资源控制和回归 gate 见 `DISTANCE_EXPANSION_DESIGN.md`。实现顺序固定为：
先完成公共 schema/validator 和独立 runner，再建立四任务 graph 与新增 assay mapping；不能先凭关键词扩库、
再根据结果反推 one-hop/two-hop。

## 最终主结果与实验矩阵

论文主结果使用 `identity_blind + parent_disjoint` fresh-run：LLM 不看 query/neighbor identity，由 harness
预取冻结的分子性质和 pairwise comparison tool text。Deployment-visible 只作为补充 deployment contract，
不能决定主矩阵 setting，也不能把 blind/visible 差值解释成纯 identity visibility。主文预计包含：

```text
主表 1：Blind parent-disjoint evidence retrieval
  BBB、Bioavailability、Skin 的 none、ChEMBL direct/flat/mechanism、
  Starling direct/flat/mechanism；Bioavailability 另含 Starling numeric direct。

主表或主图 2：跨模型与部署合同补充
  GLM、GPT-OSS-20B、GPT-OSS-120B 在同一 blind contract 下比较；
  visible parent-disjoint、coverage context 和 MMP ledger 只作为 valid/appendix ablation。

主表 3：Baselines 与成本
  MiniMol/modern encoder、ECFP model、retrieval-only vote、KNN，以及 agent tokens/tool calls。
```

当前 v4 每个 split/model 的主矩阵是 22 个 condition：

```text
3 none conditions（identity policy not applicable）
19 identity-blind parent-disjoint retrieval conditions
6,887 sample-conditions on scaffold valid
```

ClinTox 不在这 22 条件中，因为当前没有与其 gold 定义匹配的 Starling direct source。若未来找到合格 source，
必须建立新 frozen lineage，不能把新条件追加到当前 6,887 分母。当前 scaffold-valid 已完成 GLM、
GPT-OSS-20B/120B blind 三套矩阵；两个 GPT 和 GLM visible 补充矩阵也已完成。正式 test
只能在 valid setting 冻结并通过完整性/泄漏/parent/held-out gates 后启动。

E12 是与上述 22 条件 source/grouping matrix 正交的 ChEMBL-only 扩展实验，不计入 22 条件。其完整累计
曲线先在 `deployment_visible_prefetched` matched-prefetch setting 中运行，用于控制工具证据和 single prior；
deployment-visible agentic confirmation 只复用/运行 `none`、`D`、`D+C` 和每个 task 最大科学可辩护层级
这些 flat 关键点。Mechanism supplementary 必跑 `D+C+H1`，并在至少一个 C-family H2 available 时再跑
`D+C+H1+H2`；按 `D+C -> D+C+H1 -> D+C+H1+H2` 顺序复用稳定 tree-node branches。Matched-prefetch 结果用于回答受控的
distance/quantity 与 flat-vs-mechanism 问题，不能替代 agentic deployment claim。

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

资源：完整 test sets；GLM API；预计属于 agentic operational + parent-disjoint 主矩阵。当前 ChEMBL 与
Bio/BBB/Skin Starling 条件已有且 parent-disjoint 已补齐；ClinTox Starling direct/full 尚未完成。

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

所有 primary source comparison 使用同一个 blind parent-disjoint protocol。Deployment-visible/operational
只回答补充部署敏感性问题；若与 blind 结论不同，必须并列报告，不能只选择更支持某一 source 的 policy。

必须并列报告 source coverage、每个 molecule 的 evidence count、condition completeness、provenance
availability、retrieved-neighbor overlap 和 token cost。不能把 coverage 差异静默解释成 row quality。

资源：当前 v4 只覆盖 BBB、Bioavailability、Skin；ClinTox 无同定义 direct source，明确不进入本 lineage。
三任务 scaffold-valid blind source comparisons 已在三模型完成，正式 test 尚缺。Starling 生成成本单独记账，
不与 LLM reasoning tokens 合并。

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

资源：blind 主矩阵中的 flat/mechanism 条件；GLM/GPT endpoints。主矩阵和任何 visible ablation 都必须保证
flat 和 mechanism 使用相同 evidence row 集合。当前结果已证明 gain 可能很小，因此不允许在
test 上继续改 mechanism prompt。

2026-07-27 对 Skin Starling parent-disjoint random/scaffold 完成一次 test-driven post-hoc failure
diagnostic：冻结并复用 full-mechanism 的 single、Tier 1 和 Tier 2 branch，只删除 Tier 3/4 后重跑 final。
该诊断不修改 prompt，也不注册为 primary condition。相对 full mechanism 的 macro-F1 delta 分别为
`+0.001013` 和 `+0.011211`，bootstrap 95% CI 均跨 0，且两套均未超过 direct；详细 provenance、
coverage 和命令记录在 `tools/chembl_tool/tasks/skin_reaction/AGENTS.md`。

完成标准：若 gain 不稳定，改写为“mechanism decomposition 的收益依赖 family coverage/quality”，并
检验 gain 与 coverage、duplicate burden、group disagreement 的关系。

### E5：Deployment-visible agentic 补充矩阵

**支撑贡献：C4、C5。回答 RQ5。优先级：P0。**

在与 blind 主矩阵相同的 scaffold-valid、held-out index 和 parent-disjoint retrieval 上运行显式
deployment-visible workflow，报告 Macro-F1、accuracy、per-class F1、paired flips、token/tool/retry、
identity/source-ID 使用和 evidence utilization。该合同同时恢复结构/身份并让模型自主 function-call，
所以 blind/visible 差异是完整部署合同差异，不是单一 visibility effect。

资源：GPT-OSS-20B/120B 各 22 条件已经完成；两者各有一个确定性的 Bioavailability context-limit failure，
按预先声明 policy 计错。GLM 同合同 22 条件也已达到 6887/6887 严格成功，并进入 blind+visible 总图。旧 TDC 26 条件 agentic、
21 条件 matched-prefetch 和 operational parent-disjoint 结果保留为 historical appendix，不扩成 v4 主表。

完成标准：visible 结果只能与同 model、同 split/subset、同 retrieval policy 的 blind 结果并列；不得用它
选择 test prompt 或覆盖 blind 主结果。完整性、parent/threshold/held-out 和 visible contract audit 必须通过。

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

资源：1 至 4 张 GPU 足够完成 learned baseline；CPU 可跑 KNN/RF/XGBoost。上一版 strict-conflict 的
三任务、两种 Starling split 已完成 MiniMol train-all、full-test Morgan KNN 和复用冻结 MiniMol embedding
的 cosine KNN；当前 v4 已完成对应的 scaffold-valid 三种 baseline。Bioavailability scalar KNN 仍作为
numeric direct-F 专项对照。尚缺 v4 random/test、ECFP RF/XGBoost、使用 agent
相同 neighbors 的 retrieval-only vote，以及用于确认表示选择稳健性的独立第二种 pretrained encoder。
当前数值与入口见 `STARLING_BENCHMARK_RESULTS.md`。

完成标准：报告 Macro-F1、accuracy、AUROC（有 score 时）、训练/推理成本。Agent claim 必须说明它
在哪些任务优于 learned model，以及 retrieval/trace 提供了什么额外能力。

### E9：跨模型 confirmation matrix

**支撑贡献：C1、C5。回答 RQ6。优先级：P0。**

当前已在完整 scaffold-valid 22-condition blind contract 上运行 GLM-5.2 NVFP4、GPT-OSS-20B 和
GPT-OSS-120B，并在 GPT 两个规模上完成 visible contract。后续不再按结果挑 8 个条件替代这一事实；正式
test 是否运行完整三模型矩阵，必须在 valid 后按算力和预注册 claim 冻结。

资源：现有三模型 blind/visible valid 结果和统一 `plot_starling_model_comparison.py` 总图已就绪；仍缺与
GPT-OSS 不同的独立模型家族、关键条件 repeats，以及 GLM visible 的 paired flips/合同分析。

完成标准：比较同 condition 的 effect direction、paired flips 和模型排序，而不是只比较各模型 best-of-7。
若 retrieval/source/mechanism 结论只在某一模型成立，论文明确标为 model-specific。

### E10：关键条件重复运行

**支撑贡献：C5、C6。回答 RQ6。优先级：P0。**

从预注册 primary comparisons 中选择 6 个 agentic 条件对：BBB retrieval gain、BBB parent ablation、
Bio Starling source、Bio mechanism、Skin failure、ClinTox failure。每个 condition 总计至少 3 次独立生成；
当前 run 可作为第 1 次，前提是数据、retrieval policy、prompt 和 model identifier 未改变，否则从头计数。

资源：约 24 个额外 condition runs（6 对 × 2 conditions × 2 additional repeats）；GLM API。预计是除主矩阵
外最大的 token 成本。

完成标准：报告 mean、SD、每次 paired delta、prediction disagreement 和结论方向稳定率。不能用表现最好
的一次作为主结果。

### E11：Coverage–performance 关联分析

**支撑贡献：C3、C6。回答 RQ7。优先级：P0。**

在 identity-blind parent-disjoint 主制度中，对每个 retrieval condition 计算：

```text
overall coverage = 有至少一个 retrieved neighbor 的 query 比例
positive-class coverage = 正类 query 中有至少一个 neighbor 的比例
negative-class coverage = 负类 query 中有至少一个 neighbor 的比例
delta macro-F1 = 该 retrieval condition 与同任务 none 条件在共同样本上的 paired macro-F1 差值
```

主图同时展示 condition-level overall coverage 与 `delta macro-F1`，并展示正负类别 coverage 的差距。
该分析使用 macro-F1 作为主性能量，因为四个任务的标签分布不平衡；class-conditional coverage 用于判断
retrieval 是否系统性偏向多数类或少数类。第一轮 operational test 结果用于建立分析和图表，不替代最终
数据冻结后的重跑。

该分析只报告描述性关联和 failure boundary，不把 coverage 当作随机化处理，也不声称更高 coverage
因果性地带来更高性能。不同 task/source/view 的 evidence quality 同时变化，因此不做复杂的
`none/hybrid/retrieval` counterfactual 拆分；后续 quality annotation 在 E3 中独立处理。

完成标准：统一生成 `coverage_performance.tsv`、主报告中的对应表和 canonical SVG；test/valid、不同
visibility regime 和新增 retrieval condition 不需要复制 task-specific 代码；每个点保留 task、source、
view、样本数、正负类分母和 paired bootstrap 区间。

### E12：ChEMBL mechanistic-distance / evidence-quantity expansion

**支撑贡献：C7、C6。回答 RQ8。优先级：P0。**

> 2026-07-23 update：四任务 strict-hop census 得到 `0/4` publishable strict H2。下文 H1/H2
> 执行细节只保留作历史，不得启动 strict 四级 LLM curve。E12 主线改为
> `RELEVANCE_DILUTION_EXPERIMENT_PLAN.md` 中的 controlled relevance-dilution prefixes；
> strict H1 只作为 secondary analysis。

2026-07-20 实现状态：BBB v2 先对 20,369 个 D/C assays 做 measured-state census，确认 tight-junction、
efflux/influx abundance 和 functional PXR/CAR 已属于 base；因此 v1 对这些 node 的 extension mapping 作废。
v2 historical prototype 曾映射 H1 `mmp9_activity` 与 H2 `mmp3_activity`，并完成 696 assays/2,747 molecules 的
retrieval-only audit。但后续 shortcut audit 找到 MMP-3 到 tight-junction integrity 的直接路径，故 MMP-3 的最短
距离也是 H1，原 H2 mapping 无效。v2 index/coverage 只保留作工程历史，不得送入 E12 LLM 或解释为新 ontology
的 coverage/performance。

下一版 BBB 必须按 C-family tree 重建：D 为 root，每个冻结 C mechanism family 恰有一个聚合 H1 child，每个 H1
至多有一个 optional H2 child；一个 child 可聚合多个 measurement families，但 node 共享 top-3 neighbor budget。
MMP-9/MMP-3 可同时进入 passive/barrier C family 的 H1；其它 C families 的 H1/H2 必须重新做 citation、shortcut、
D/C overlap 和 ChEMBL feasibility audit。当前尚未形成可运行的 BBB E12 graph，也未运行 E12 LLM macro-F1。

主实验是 D4 中冻结的累计 source expansion curve。主要指标只有 test macro-F1；每个相邻累计点和相对
`none` 的差异报告 paired bootstrap interval，并画每任务 response curve。Coverage、unique retrieved
molecule 数、assay/family 数、LLM-visible evidence rows 和 prompt tokens 作为解释 quantity 的辅助量，
不增加固定-token replacement、rescue rate/harm rate 或 cross-task assay 条件。

完整曲线使用 `deployment_visible_prefetched`，确保每个 query 的 prefetched tool outputs、single prior、
模型参数和 evidence serialization policy 一致。`D/C/H1/H2` 均来自同一冻结 ChEMBL source manifest，唯一系统
变化是该 ChEMBL evidence union 向更远 family 扩展；任何包含 Starling row 的 E12 condition 都应由审计直接判为
无效。
随后在每个 task 的 `none`、`D`、`D+C` 和最大可辩护层级做 deployment-visible agentic confirmation；已有
完全匹配的 agentic condition 可通过 input hash 审计后复用。

Mechanism supplementary 在同一 visibility regime 下运行 `D+C`、`D+C+H1`、`D+C+H1+H2`。执行必须先完成
`D+C+H1`：复用 hash 未变化的 D/C branches，只运行各 C family 的聚合 H1 branches 后重跑 final；随后以该 condition 为
`--group-analysis-source-batch` 运行 `D+C+H1+H2`，复用 D/C/H1，只运行 H2 branches 后再次重跑 final。
同一 prefix 的 flat/mechanism evidence-row multiset 必须一致；两者差异只能是 grouping 和并行 reasoning。

该实验回答的是“在同一 curated database 内，较远但更多的 evidence 是否仍然有用”，不能用于声称
Starling generally better than ChEMBL。E2 才回答 source provenance/representation；两条实验线必须分别
作图和下结论。若性能随扩展先升后降，报告 empirical frontier；若不同 task 的方向不一致，报告
task-specific boundary，不拟合一个跨任务通用距离阈值。

资源：ChEMBL assay mapping 和 citation-backed 两遍 graph/path audit；新 evidence library/index；四任务
matched-prefetch flat 累计曲线与 H1/H2 mechanism supplementary；flat 最多每任务一个新增 maximal-distance
agentic confirmation；mechanism 必含 H1 point，并在至少一个 family H2 available 时再含 H2 point。先在 valid 上做 coverage、payload
和完整性检查，但不按 valid/test 性能改变 graph ontology。

完成标准：graph、edge rationale、assay-to-family mapping 和 source manifest 可发布；每个累计点的实际
coverage/evidence volume/tokens 可审计；必需 H1 point 与任何 available H2 point 均完成且通过 branch-reuse
与 flat/mechanism evidence-parity audit；所有 paired comparison 使用共同 sample set；不存在用 Starling
缺少 distant coverage 来人为放大 ChEMBL quantity 的跨 source comparison。

### E13：Final evidence surface / aggregation bottleneck

**支撑贡献：C3、C6。回答 RQ9。优先级：P0 诊断。**

该实验固定 record-supported v2 scaffold-valid 的 identity-blind parent-disjoint
`starling_full_flat` retrieval、single 和 group artifacts，只重新运行 final：

```text
summary_only       = 当前冻结 control，final 只看 single/group summaries
summary_plus_cards = 同一 summaries + 确定性 compact evidence cards
cards_only         = 同一 raw evidence cards，不向 final 提供 group LLM summary
```

`summary_only` 必须保持历史 prompt 的严格 no-op。Card 只从已经 identity-redacted、tool-prefetched 的
reasoning retrieval 构建；固定最多 12 个 cards、每 card 最多 3 条 deterministic-even-spacing evidence rows，
保存 card contract version、SHA-256、原始/保留行数和字节数。它不得重新检索、改变 neighbor、暴露身份、
加入 label vote/threshold 或把 card 与对应 group summary 当作独立 evidence。

第一阶段只使用 scaffold-valid 诊断，不能根据结果修改 test prompt。三个 task 以共同样本做 paired
macro-F1 bootstrap、exact McNemar/Holm、prediction flips 和 rescue/harm。若 cards 不能稳定改善，停止把
aggregation bottleneck 当成主因；若有稳定改善，冻结 surface 后才能考虑一次 test confirmation。

Metadata census 只审计 endpoint、context、example-level measurement、multi-record/PMID support、scope 和
uncertainty 是否存在；它不产生 relevance score，也不自动进入 compatibility selector。内部 sidecar 不进入
`minimal_evidence.v1` 或 LLM prompt。完整协议、命令和输出结构见
`FINAL_EVIDENCE_SURFACE_EXPERIMENT.md`。

2026-08-07 实测完成六个 batch、`1,908/1,908` final、`n_failed_runs=0`，全量 artifact/card/identity
contract audit 为 0 failure。Summary+cards 相对 summary-only 的 macro-F1 delta 为 BBB `-0.0016`
（95% CI `[-0.0244,+0.0217]`）、Bioavailability `+0.0011`（`[-0.0376,+0.0392]`）、Skin `-0.0132`
（`[-0.0524,+0.0261]`）；cards-only 分别为 `-0.0285`、`+0.0073`、`+0.0018`，三个 interval 也均跨 0。
Final prompt token 均值增加到 control 的 `3.32x–4.72x`，仍无稳定增益。因此 E13 的结论是 no-go：
aggregation compression 不是当前主瓶颈，card surface 保留为可复现实验插件，不进入默认路径或 formal test。

### 近期小计划：Skin task alignment 与 final bottleneck audit

目标是先判断错误主要来自 task scope、上游 group reasoning，还是 final synthesis；这不是 group ablation，
也不搜索 router/selector。

1. **Skin scope 修复（待讨论后实现）**：保留历史 `legacy_skin_reaction_v1` 结果；新增版本化
   `sensitization_aligned_v2` prompt profile。Gold label 不变；phototoxicity、irritation/corrosion、generic local
   damage 和 exposure 只作为 out-of-scope/context，不得单独支持 `risk` 或 `no_risk`。Manifest 必须记录 profile。
2. **现有 trace audit（不新增模型调用）**：在修复后的 baseline 定义上，把错误分成
   `final_recoverable`（group outputs 已支持 gold、final 仍选反或引用越界证据）和 `upstream_failure`
   （相关 group 已遗漏、误读或错误 transfer）。保存机器可读逐样本分类和汇总，不查看 formal test。
3. **唯一 continuation gate**：只有 final-recoverable 是明确主导来源时，才允许一个 final-only ICL 候选：
   每个 query 总共 3 个 compact train examples，而不是每个 group 3 个；只用 train、排除同 parent/scaffold，
   不做 k、长度、retriever 或 hidden-CoT sweep。否则停止 demonstrations，直接处理 task/group reasoning。

明确不做：per-group train reasoning 注入、full-mechanism train trace bank、group drop/add ablation，以及把最终
label 正确的 trace 直接称为“正确 reasoning”。

## 资源预算

### 计算与 API

当前三套 historical exploratory regime 共使用约 427M tokens。V4 只扩展 blind parent-disjoint 主矩阵；
deployment-visible 只做冻结补充条件。初步按相同 prompt 规模规划：

```text
final blind parent-disjoint matrix: approximately 150M-300M tokens
selected deployment-visible ablations: budget separately after valid
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
D1 parent-disjoint policy + D3 protocol freeze + D4 ChEMBL distance graph freeze
  -> D2 Starling data freeze and E12 ChEMBL distant-library build in parallel
  -> E0 index/overlap/contract smoke
  -> E3 annotation sampling and E8 baselines in parallel
  -> E1/E2/E4/E5 primary matrix
  -> E12 matched-prefetch curve and selected agentic confirmation
  -> E11 coverage-performance analysis
  -> E9 second model and E10 repeats
  -> E6 targeted visibility attribution
  -> final statistics, figures and writing
```

统一入口已按 v4 合同完成代码迁移。主命令为：

```bash
# 查看当前 runner 已注册的实验
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent --list

# 运行 v4 identity-blind parent-disjoint 主条件；同一时间只启动一个 launcher
python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix \
  --benchmark-split <random|scaffold> \
  --visibility-mode identity_blind \
  --neighbor-identity-policy parent_disjoint \
  --parallelism 128 \
  --experiments <experiment_id> [<experiment_id> ...]

# 生成统一统计和报告
python -m tools.chembl_tool.paper_experiments.summarize_results

# 生成 coverage 与 macro-F1 增幅的 focused analysis 和 canonical SVG
python -m tools.chembl_tool.paper_experiments.analyze_coverage_performance
python -m tools.chembl_tool.paper_experiments.plot_coverage_performance
```

Starling 当前可用 builder 命令见 [README.md](README.md)。Parent-disjoint exclusion/backfill 和
BBB/Skin full Starling builders 已实现；blind+parent-disjoint fresh-run、独立 root、无需 operational
reuse plan 的完整 none/retrieval matrix，以及 512 全局并发约束已实现。以下入口必须继续
通过测试，不能把本文档中的名称直接当作 CLI 参数：

- ClinTox 的最终 Starling family builder/profile；
- E6 的细粒度 query/neighbor/source-ID visibility policies；
- E8 尚缺的 retrieval vote、ECFP learned 和独立第二种 pretrained encoder runner；
- E3 annotation sampler、schema、agreement 和 adjudication summarizer；
- E9/E10 的模型/重复 run manifest 与聚合报告；
- E12 的 publishable task configuration 与 LLM matrix integration。共享 graph/index/retrieval/audit 和 BBB
  v3 engineering artifacts 已实现，但当前 self-relevance/role gate 仍不允许把 H1/H2 当成论文结果。

每实现一个入口，必须先补测试和 `--help`/README 命令，再把本列表对应项删除并更新 E 编号状态。

## 执行时间表

ICLR 2027 官方 deadline 尚未发布。本时间表按往年 9 月中下旬 abstract/full-paper deadline 倒排，并保留
至少 7 天 buffer；官方日期发布后立即调整。

| 时间 | 必须完成 | Go/No-Go gate |
|---|---|---|
| 07-12 至 07-18 | 冻结 thesis、primary comparisons、两套 parent policy、第二模型和 annotation guideline | 未冻结则不生成新 Starling full data |
| 07-19 至 08-02 | 补齐 BBB/Skin/Clin Starling；冻结 ChEMBL distance graph/source manifest；开始人工 annotation | 数据不完整、distance mapping 未复核或 critical attribution error 未修则不跑对应主实验 |
| 08-03 至 08-09 | 构建 standard/distant indices、5-sample smoke、parent overlap audit；完成 baseline 入口 | contract/overlap/payload/失败 gate 未通过则不 full run |
| 08-10 至 08-23 | Agentic operational/parent-disjoint 主矩阵、E12 matched-prefetch 曲线与关键 agentic points、8 条件第二模型；持续完整性检查 | 08-23 必须得到可汇总主结果 |
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

- [ ] 三任务 22-condition blind parent-disjoint 正式 test 完成；ClinTox 不可得边界有冻结说明。
- [ ] 所有主表 run 使用冻结的 identity-blind/parent-disjoint contract、held-out index 和 source manifest。
- [ ] 主表 `n_failed_runs=0`，identity-blind contract 通过；parent-disjoint conflict、threshold violation 和 held-out overlap 均为 0。
- [ ] Source-quality annotation 完成双人一致率和分歧仲裁。
- [ ] KNN、retrieval vote、ECFP learned、MiniMol、pretrained encoder baseline 完成。
- [ ] 第二模型 confirmation 和 6 个关键 comparison repeats 完成。
- [ ] Visibility gain/failure taxonomy 由第二 annotator 复核。
- [ ] 每个主 claim 都能指向一张主表或一幅主图，不依赖 anecdotal trace。
- [ ] 最终 agentic retrieval 矩阵已生成 overall/class-conditional coverage、paired macro-F1 增幅和 coverage–performance SVG。
- [ ] E12 的 ChEMBL-only tree-prefix flat 曲线、`D+C/D+C+H1` 及任何 available `D+C+H1+H2` mechanism supplementary、branch-reuse/evidence-parity 审计和关键 agentic confirmation 完成；不可辩护的 H2 逐 C family 记为 unavailable。
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
7. ChEMBL-only mechanistic-distance 累计扩展曲线及其 task-specific boundary。

若 Starling 无法按时补齐 Skin/Clin mechanism，应在数据冻结 gate 做一次明确 scope reduction，并将论文
主张收缩为“跨任务 direct source comparison + 两个任务的 mechanism case study”，而不是在截稿前留下
不完整矩阵。
