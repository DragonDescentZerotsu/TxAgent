# Strict H1/H2 availability census across four molecular tasks

日期：2026-07-23
状态：ChEMBL-only feasibility decision；不改变现有 paper matrix、D/C index 或 reasoning results

## 审计目标

本轮回答一个 go/no-go 问题：在不降低已冻结 hop 定义、不重复 D/C assay、并要求 same-molecule causal
continuity 的前提下，四个 task 是否有足够 H1/H2 source 支撑：

```text
D
D+C
D+C+H1
D+C+H1+H2
```

三个新 task 使用同一个只读审计器：

```text
tools/chembl_tool/common/hop_availability_census.py
```

统一门槛：

- ChEMBL 36 only；
- 排除当前 paper-facing D/C assay-ID；
- 至少 50 molecular parents、3 independent documents；
- parent-disjoint、Morgan Tanimoto >=0.30、node-level top-3；
- 至少 10% test queries 有一个 neighbor；
- 最大单一 document 不超过 80% parents；
- mechanism path 与 same-molecule continuity 必须独立通过；
- organism/system/scope 与 evidence quality 单独记录，不计入 hop。

## 结果总览

| task | 严格可用 H1 | H1 parent-disjoint >=1 coverage | 严格可用 H2 | 结论 |
|---|---|---:|---|---|
| BBB_Martins | Passive: MMP-2 + ROCK2；与既有 MMP-9/MMP-3 聚合 | 新增两 family union 36.22% | none | 只能支持 one-hop |
| Skin_Reaction | GSK3B activity -> NRF2 state | 34.15% | none | 有 H1，但不是 sensitizer-specific |
| Bioavailability_Ma | pKa；LogD/LogP；PPB/Fu | union 98.44% | none | H1 丰富，但部分与 query property tools 语义重叠 |
| ClinTox | none | 0% | none | C envelope 过宽且 label 太异质 |

跨任务计数：

```text
tasks with at least one engineering-feasible strict H1: 3 / 4
tasks with at least one publishable strict H2:       0 / 4
```

因此，原定四级 strict-hop performance curve **no-go**。不得为了形成 `D+C+H1+H2` 条件而加入不合格 H2。

## Skin_Reaction

当前 C 已含 direct skin reaction、sensitisation AOP、local damage 和 skin exposure，共 1,998 个 assays。

| candidate | level | parents | parent-disjoint >=1 | top-3 | decision |
|---|---:|---:|---:|---:|---|
| KEAP1-NRF2 PPI | H1 | 355 | 2.44% | 0% | coverage fail |
| GSK3B functional activity | H1 | 7,950 | 34.15% | 21.95% | graph/data pass；scope caveat |
| CUL3 functional activity | H2 | 0 | 0% | 0% | unavailable |

GSK3B 可通过 phosphorylation/β-TrCP 改变 NRF2 stability，因此到 C 中 NRF2/ARE state 的一跳关系成立：

- [PMID 22964642](https://pubmed.ncbi.nlm.nih.gov/22964642/)
- [PMID 25937177](https://pubmed.ncbi.nlm.nih.gov/25937177/)

但它不是 sensitizer-specific source。KeratinoSens 的 NRF2 signal 是以 electrophile/sensitizer activation 为代理；
已有研究也观察到 NRF2-dependent false positive，说明直接药理性调控 NRF2 不等价于 skin sensitisation：

- [PMID 24055896](https://pubmed.ncbi.nlm.nih.gov/24055896/)

所以 Skin GSK3B 可以作为“更 distant、scope 更弱”的 H1 候选，却不应写成高质量 sensitisation evidence。
H2 的 CUL3 edge 有生物学支持，但 ChEMBL 没有可用 functional compound library：

- [PMID 15282312](https://pubmed.ncbi.nlm.nih.gov/15282312/)
- [PMID 15601839](https://pubmed.ncbi.nlm.nih.gov/15601839/)

产物：

```text
outputs/chembl_tool/tasks/skin_reaction/distance_expansion/analysis/hop_availability_census/
```

## Bioavailability_Ma

当前 C 已含 direct F、AUC/Cmax、Fa、Fg 和 Fh，共 70,391 个 assays。C 之外仍有大量 experimental molecular
properties：

| candidate | level | parents | parent-disjoint >=1 | top-3 | decision |
|---|---:|---:|---:|---:|---|
| measured pKa | H1 | 9,954 | 94.53% | 80.47% | available |
| measured LogD/LogP | H1 | 43,389 | 97.66% | 92.97% | available |
| measured PPB/Fu | H1 | 14,791 | 94.53% | 85.16% | available |
| accepted H1 union | H1 | 59,049 | 98.44% | 96.09% | available |
| defensible H2 | H2 | 0 | 0% | 0% | unavailable |

pKa 与 lipophilicity 对 passive permeability/solubility 的关系可形成到 Fa family 的 H1；Fu/PPB 是 hepatic
clearance model 的直接输入，可形成到 Fh family 的 H1：

- pKa/permeability：[PMID 14659483](https://pubmed.ncbi.nlm.nih.gov/14659483/)
- lipophilicity/permeability-solubility trade-off：[PMID 30395703](https://pubmed.ncbi.nlm.nih.gov/30395703/)
- protein binding/hepatic clearance：[PMID 7229915](https://pubmed.ncbi.nlm.nih.gov/7229915/)

关键 caveat：query 的 estimated logP、pKa 和 logD 已由 `molecule_properties` tool 提供。这里新增的是 **analog
experimental property measurements**，数据源与 grain 不同，但 prompt 语义高度相关。若纳入后性能提升，不能
简单归因于“更远机制”，还可能来自对已有 property prior 的重复强化。PPB/Fu 不在当前 tool 输出中，但只支持
clearance，不直接决定 absolute oral F。

产物：

```text
outputs/chembl_tool/tasks/bioavailability_ma/distance_expansion/analysis/hop_availability_census/
```

## ClinTox

当前 C 已覆盖 43 source groups、108,413 个 assays，包括 clinical/in-vivo/organ/genotoxicity/cellular
stress/cytotoxicity/off-target/DDI。代表性上游节点的结果：

| candidate | nominal level | post-D/C status | decision |
|---|---:|---|---|
| KEAP1-NRF2 PPI | H1 | 414 合格 rows 全部与 D/C assay 重叠 | unavailable |
| GSK3B activity | H1 | 数据覆盖 28.67%，但缺 toxic exposure/directional mapping | reject |
| complex-I activity | H1 | 无 confidence>=8 functional library | unavailable |
| CUL3 activity | H2 | 0 functional rows | unavailable |

GSK3B/NRF2 modulation 可保护也可加重不同毒性过程；单独 target activity 不能映射到 heterogeneous ClinTox
clinical-toxicity label。Complex-I inhibition 与毒性相关，但需要 dose、organ exposure 和 injury context，且低剂量
modulation 也可能产生适应性效应。因此这些不能仅因 ChEMBL 数据量大而成为正式 H1。

产物：

```text
outputs/chembl_tool/tasks/clintox/distance_expansion/analysis/hop_availability_census/
```

## Go/no-go decision

1. **停止构建 strict H2 performance condition。** 四个 task 均没有合格 H2；`D+C+H1+H2` 没有可比较的真实数据。
2. **严格 H1 只适合作为 secondary one-hop analysis。** BBB、Skin、Bioavailability 可以各自做
   `D+C -> D+C+H1`，但三个 H1 的语义不同，不能伪装成同质的四点 distance curve。
3. **公共 tree contract 最终应允许 sparse/unavailable child。** “每个 C 恰有一个 H1”不符合实测数据。
4. **原始 quality-quantity 问题需要新的 distance/relevance 操作化定义。** 下一步应在不叫 H1/H2 的独立设计中，
   明确区分：
   - strict causal evidence；
   - pathway-connected but scope-weaker context；
   - controlled relevance dilution。

新的 relevance-band contract 与分阶段执行 gate 已记录在
`RELEVANCE_DILUTION_EXPERIMENT_PLAN.md`。在其四任务 retrieval-only census 通过 go/no-go review 前，
不物化新的 LLM conditions，也不修改旧 paper matrix。
