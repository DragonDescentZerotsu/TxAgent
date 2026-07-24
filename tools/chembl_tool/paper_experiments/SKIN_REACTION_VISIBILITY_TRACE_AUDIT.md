# Skin Reaction validation 可见性差异 Trace 审计

> 状态：2026-07-23 的可复现诊断快照
> 数据 split：`valid`，40 个分子（18 个 label 0，22 个 label 1）
> 模型：`zai-org/GLM-5.2-FP8`，temperature 0
> 主要因果对照：`identity_blind` vs `deployment_visible_prefetched`
> 补充敏感性对照：agentic `deployment_visible` 与 `parent_disjoint`

## 技术摘要

Skin Reaction validation 上，identity-blind 的 Macro-F1 普遍高于 deployment-visible，主要不是
retrieval、工具执行或 parent-disjoint policy 的差异，而是完整结构可见后形成了更偏向 `risk`
的参数化结构先验。

最有解释力的对照是 `identity_blind` 与 `deployment_visible_prefetched`：两者使用冻结且完全匹配的
retrieval 和 prefetched tool outputs，只改变 query/neighbor 的结构与身份可见性。Prefetch contract
audit 覆盖 2,713/2,713 个 valid sample-condition，missing、extra 和 mismatch 均为 0。

逐样本、逐 stage 对齐得到以下结论：

1. 七个 Skin 条件共有 55 个 blind/visible prediction flips，其中 37 个是 blind 正确、visible 错误，
   18 个方向相反。
2. 37 个 blind-only correct condition-level 结果中，35 个来自同样 5 个 label-0 分子在七个条件中
   反复翻转。由于 single-molecule analysis 跨条件复用，这不是 35 次独立错误，而是 5 次
   single reasoning 的影响传播到多个 retrieval condition。
3. 这 35 个 visible 假阳性中，28 个没有任何 retrieved neighbor；其余 7 个的 group reasoning
   通常明确判断 evidence 为低 transferability、不能外推或不能覆盖 query 特有结构。
4. 35/35 个 visible final 都以 `sensitization_aop` 为 `main_evidence_type`，同时承认缺少
   direct/experimental confirmation；35/35 依赖 structural alert，34/35 同时引用 skin exposure，
   33/35 仍给出 `moderate` confidence。
5. 主要失败链条是：

   ```text
   raw structure visible
   -> topology-level structural alert
   -> single-molecule concern
   -> final 在没有实验锚点时仍映射为 benchmark risk
   ```

因此，当前问题不宜描述为“visible 看错了结构”。更准确的表述是：visible 能识别 blind 无法确认的
拓扑级化学警报，但 final policy 对“未经实验验证的 sensitization hazard plausibility”
到最终 `Skin_Reaction=1` 的映射过于 precautionary。

## Validation 性能对比

下表均为 40 个 valid 分子的 Macro-F1。`matched visible` 指
`deployment_visible_prefetched`，是判断 structure visibility 的主要 parity-controlled 对照。

| 条件 | Identity-blind | Matched visible | Agentic visible |
|---|---:|---:|---:|
| None | 0.7396 | 0.5943 | 0.5733 |
| ChEMBL direct | 0.6931 | 0.6577 | 0.5733 |
| ChEMBL full flat | 0.7163 | 0.6577 | 0.5733 |
| ChEMBL mechanism | 0.6647 | 0.6577 | 0.6050 |
| Starling direct | 0.7333 | 0.6011 | 0.6366 |
| Starling full flat | **0.7630** | 0.6366 | 0.6366 |
| Starling mechanism | 0.7333 | 0.6366 | **0.6703** |

Starling full-flat 的 confusion matrix 最能说明下降方向：

| 设置 | TN | FP | FN | TP | Specificity | Positive recall |
|---|---:|---:|---:|---:|---:|---:|
| Identity-blind | 11 | 7 | 2 | 20 | 61.1% | 90.9% |
| Matched visible | 7 | 11 | 2 | 20 | 38.9% | 90.9% |
| Agentic visible | 7 | 11 | 2 | 20 | 38.9% | 90.9% |

正类召回没有变化，差距全部来自 4 个额外 false positives。这说明 Macro-F1 下降的核心是负类
specificity，而不是 visible 丢失了阳性识别能力。

## Prediction flips 集中在一个小型分子簇

| 条件 | 总 flips | Blind-only correct | Matched-visible-only correct |
|---|---:|---:|---:|
| None | 7 | 6 | 1 |
| ChEMBL direct | 9 | 5 | 4 |
| ChEMBL full flat | 8 | 5 | 3 |
| ChEMBL mechanism | 10 | 5 | 5 |
| Starling direct | 8 | 6 | 2 |
| Starling full flat | 6 | 5 | 1 |
| Starling mechanism | 7 | 5 | 2 |
| **合计** | **55** | **37** | **18** |

五个系统性 label-0 false positives 占 35/37 个 blind-only correct condition-level 结果：

| Query index | 主要 visible alert | Blind single prior | Visible single prior | 首次关键分叉 |
|---:|---|---|---|---|
| 3 | primary sulfonamide + diarylamine oxidative pro-hapten | `no_risk`; reactive `not_apparent` | `mixed`; reactive `mixed` | Final 将两个较弱警报相加为 `risk` |
| 4 | 2-chloropyrimidine SNAr electrophile | `mixed`; reactive `not_apparent` | `risk`; reactive `concerning` | Single-molecule |
| 5 | 2-chloropyrimidine SNAr electrophile | `no_risk`; reactive `not_apparent` | `risk`; reactive `concerning` | Single-molecule |
| 14 | 环状 α,β-unsaturated ketone / Michael acceptor | `no_risk`; reactive `not_apparent` | `risk`; reactive `concerning` | Single-molecule |
| 24 | diarylamine/phenol oxidative hapten + quinolinone chromophore | `mixed`; reactive `mixed` | `mixed`; reactive/phototoxic `concerning` | Final 在低 permeation 下仍升级为 `risk` |

其中 query 3、4、5 位于相近的 kinase-inhibitor-like scaffold 区域，因此五个 case 并非五个完全独立的
化学空间失败。Valid 的小样本量和 scaffold composition 放大了这一结构簇对总体 Macro-F1 的影响。

## 分叉发生在哪里

### 2-chloropyrimidine：blind 缺少连接关系，visible 完成了更具体的化学解释

Blind trace 只看到 aryl chloride、ketone、alkene 等脱敏后的性质和官能团摘要。对 query 4/5，
blind 将 aryl chloride 视为通常较稳定，并判断缺少典型 electrophilic warhead。

Visible trace 从完整连接关系识别出 chlorine 位于 electron-poor pyrimidine 上，将其解释为潜在
SNAr electrophile，可与 protein thiol/amine 形成 covalent hapten。这个解释在化学上比 blind
的“普通 aryl chloride”更具体；问题在于模型随后把“合理的反应性假设”直接映射为 benchmark `risk`，
而没有 direct assay、DPRA、KeratinoSens、h-CLAT 或高 transferability analog 支持。

### 环状 enone：blind 明确表示无法确认共轭，visible 将其识别为 Michael acceptor

Query 14 的 blind trace 写明 ketone 和 alkene 作为独立 functional groups 出现，无法确认二者是否共轭，
因此不给 Michael-acceptor alert。Visible trace 能从 SMILES 确认环状 enone，并在 single stage
将 prior 改成 `risk`。

该样本的 retrieved analogs 都是只共享 benzyl fragment 的 distant analogs。Group reasoning 明确认为
它们的 mostly-negative 结果不能反驳 query 特有 enone。Final 因而完全退回 structural prior。

### 多个较弱 alert 被 final 合并成阳性

Query 3 和 24 的 visible single 输出并不是明确 `risk`，而是 `mixed_or_unclear`。但 final 将两个较弱的
pro-hapten/phototoxicity concern 组合后输出 `risk`，即使 trace 同时承认：

- 没有 direct experimental confirmation；
- alert 可能依赖 oxidative/metabolic activation；
- query 24 的 MW、TPSA、ionization 和 rotatable bonds 均不利于 dermal penetration。

这表明问题不仅位于 single-molecule alert detection，也位于 final evidence aggregation。

## Retrieval 分支没有制造这些阳性

五个核心分子在七个条件中形成 35 个 visible 假阳性：

| Trace 属性 | 数量 |
|---|---:|
| 没有任何 neighbor | 28 / 35 |
| 存在 group reasoning | 7 / 35 |
| Final 明确承认缺少 direct/experimental evidence | 35 / 35 |
| Final 使用 structural alert/prior | 35 / 35 |
| Final 引用 skin exposure/permeation | 34 / 35 |
| Final confidence 为 `moderate` | 33 / 35 |

有 group evidence 的 7 个结果也没有形成可靠阳性锚点：

- Query 4/5 的 pazopanib-like evidence 被判为低 transferability，因为临床 skin reaction 更可能由
  anti-VEGFR pharmacology 驱动；
- Query 14 的 mostly-negative analogs 被判为不包含 query 特有 enone，不能用于 negative read-across；
- 其它 evidence 主要是 distant fragment overlap 或 exposure context。

因此，group branch 的作用主要是正确地拒绝不可靠 read-across。Final 在拒绝 retrieval 后改用
single structural prior，才形成最终假阳性。

## Prompt 中的 evidence-sufficiency 缺口

当前 final prompt 已经包含三条正确约束：

1. isolated AOP key event 不等于 clinical skin reaction；
2. skin permeability/retention 只能作为 exposure context，不能单独证明 risk；
3. distant analog 必须降权，除非 shared scaffold 和 assay mechanism 构成强 read-across。

但 prompt 同时要求：

- 把 single-molecule analysis 作为 structural/physicochemical prior；
- evidence mixed 或 weak 时仍必须二选一；
- 通过 confidence 和 caveats 表达不确定性。

Prompt 没有定义最低 evidence threshold，例如“仅有未经实验验证的 structural alert 时，何时允许最终
输出 `risk`”。因此当前隐含决策规则接近：

```text
plausible structural alert
+ plausible exposure
+ no strong contradictory evidence
=> risk，通常仍给 moderate confidence
```

这与更适合该 benchmark 的候选规则可能不同：

```text
structural alert
=> hazard hypothesis
=> direct assay / reliable AOP key event / high-transferability analog anchor
=> 才映射为 Skin_Reaction label 1
```

需要先确认原始 label ontology，再决定第二条规则是否适合作为正式实验设置，避免在 40 个 valid
样本上直接进行 prompt tuning。

## Visible 并非总是更差

Query 22 是重要反例。Blind 仅根据多个 aromatic rings、quinoline 和 CF3 推断 phototoxicity，
将该 label-0 分子判为 `risk`。Visible 看清实际连接关系后，认为没有 classic phototoxic motif，
并结合 MW 593、logP 7.25 和低 dermal permeation，将结果正确改为 `no_risk`。

因此，不应把结论写成“blind 更懂化学”或“隐藏结构更好”。更准确的结论是：

> Blind 的信息限制在本 split 上形成了更保守的正则化。Visible 提供更准确的结构解释，但当前 final
> policy 对 positive structural alerts 的 label calibration 不足。

Agentic visible 还把 query 3 恢复成了 `no_risk`，说明这个 case 存在一定生成波动；另外四个核心误判
仍跨多个条件保留，所以运行波动是次要因素而不是主要根因。

## Parent-disjoint 与 same-parent 不是原因

Skin Reaction valid 的 operational/parent-disjoint paired summary 中，六个 retrieval conditions
均为 0 prediction flips，Macro-F1 不变。Starling 虽有少量 same-parent slots 被替换，但最终预测没有变化。

所以本次 blind-visible 差距不能由 same-parent neighbor leakage 或 parent policy 解释。

## 方法与定义

### 主要比较

主要因果对照使用：

```text
identity_blind:
  query/neighbor identity 和 raw structure 隐藏；
  使用 harness-prefetched molecule properties 和 comparison outputs。

deployment_visible_prefetched:
  query/neighbor 可见信息与 deployment-visible 相同；
  replay identity_blind 的冻结 retrieval 和 prefetched tools。
```

两侧仍是独立 LLM generation，因此差值包含 hosted endpoint 的生成波动，但不包含 retrieval 或
tool execution 的数据差异。

Agentic `deployment_visible` 只作为敏感性对照。它允许模型自主选择工具，因此不能代替 matched-prefetch
来归因纯 visibility effect。

### Trace 对齐方法

对每个 condition 和 query index 对齐：

1. `predictions.jsonl` 中的 gold label 与 final prediction；
2. `trace_messages.jsonl` 的 `single_molecule`、group branch 和 `final_summary`；
3. single prior、reactive prior、phototoxicity prior、permeation prior；
4. group transferability、evidence direction、confidence；
5. final `main_evidence_type`、confidence、main reasons、evidence gaps；
6. retrieval coverage 和 parent-disjoint paired outputs。

报告中的 condition-level counts 是用于描述错误传播范围的相关观测，不应当作独立重复。

## 局限与稳健性

- Valid 只有 40 个分子；每个条件的 paired bootstrap CI 大多跨 0，Holm 校正后没有单条件显著差异。
- 七个条件共享同一批分子和复用的 single outputs，不能用“七次均下降”做独立符号检验。
- Test split 上 blind 优势更弱且方向不完全一致，因此不能宣称 blind 普遍优于 visible。
- Trace 能定位模型如何形成决策，但不能证明 gold label 一定化学正确。
- 当前 TDC `Skin_Reaction` 原始标签究竟更接近临床 dermatologic reaction、sensitization hazard，
  还是其它聚合终点，仍需要 source-level provenance audit。
- 结构警报是否真实需要实验验证；本报告只说明模型如何使用这些警报，不把 reasoning 文本当作实验事实。

## 推荐的下一步验证

### 1. 先审计 endpoint provenance

优先核对五个核心 label-0 分子的原始标签来源、endpoint 定义和重复记录。若 benchmark label 偏临床反应，
则 sensitization structural alert 不能直接作为 label 1。

### 2. 只复用已有分支，重跑 discordant samples 的 final

无需重跑 retrieval、single 或 group。建议只对 unique discordant samples 做预声明的 final ablation：

```text
Control:
  当前 final prompt。

Evidence-threshold:
  未经实验验证的 structural alert 只能形成 hazard hypothesis；
  除非存在 direct evidence、可靠 AOP key-event evidence，
  或高相似度且机制匹配的 analog，否则不能单独支持 risk。

Exposure guard:
  favorable skin permeation 只能修改 exposure plausibility，
  不能为 structural alert 补足 hazard evidence。
```

每个版本至少运行 3 个 seed，区分 policy effect 与独立 generation variance。

### 3. 预先冻结判定标准

在运行 ablation 前写明：

- direct evidence、AOP key-event evidence 和 analog anchor 的最低定义；
- isolated structural alert 的允许作用；
- mixed evidence 的默认 class；
- 临床 endpoint 与 mechanistic hazard 的映射边界。

避免根据五个已观察样本逐条制作例外规则。

## 可复现来源

主要机器可读输入：

```text
outputs/paper/molecular_evidence_agent_valid/analysis/experiment_summary.tsv
outputs/paper/molecular_evidence_agent_valid/analysis/visibility_comparisons.tsv
outputs/paper/molecular_evidence_agent_valid/analysis/prefetch_contract_audit.json
outputs/paper/molecular_evidence_agent_valid/analysis/parent_disjoint_ablation/condition_results.tsv
```

主要 trace roots：

```text
outputs/paper/molecular_evidence_agent_valid/runs/skin_reaction/
outputs/paper/molecular_evidence_agent_valid/runs_deployment_visible_prefetched/skin_reaction/
outputs/paper/molecular_evidence_agent_valid/runs_deployment_visible/skin_reaction/
outputs/paper/molecular_evidence_agent_valid/runs_deployment_visible_parent_disjoint/skin_reaction/
```

Prompt 入口：

```text
tools/chembl_tool/tasks/skin_reaction/run_reasoning_pipeline.py
  _reason_single_molecule
  _reason_one_group
  _run_final_reasoning
```

## 维护说明

- 本文档是可人工编辑的正式分析记录；更新结论时应保留日期和 split。
- 新增 run 后先重新运行 valid summary/prefetch/parent-disjoint audit，再更新本文档的指标与案例。
- 对外 Site 是这一文档和机器可读 snapshot 的发布版本，不是 live connection。
- 修改底层数据或本文档后，需要重新导出 artifact 并发布新的 Site version。
