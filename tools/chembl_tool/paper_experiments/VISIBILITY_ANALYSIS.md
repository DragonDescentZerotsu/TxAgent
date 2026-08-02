# Identity-Blind 与 Deployment-Visible 结果分析

## 文档目的

本文档集中记录 structure visibility 为什么使性能上升或下降，供论文 Results、Discussion、Limitations
和 case study 直接引用。机器可读统计仍以
`outputs/paper/molecular_evidence_agent/analysis/visibility_comparisons.tsv` 为准；本文档不替代生成器输出。

> 2026-08-01 scope：本文主体分析的是旧 TDC/strict-conflict visibility experiments。
> `record_agreement70_split811_v1` 新主矩阵已改为 identity-blind + parent-disjoint；deployment-visible、
> matched-prefetch 和 operational 仅作显式补充，不再是新主矩阵前置依赖。

## 三套制度及当前结果口径

```text
identity_blind:
  query 和 neighbor 的结构、名称、source ID 不发送给 LLM；
  harness 预取 molecule properties、structure comparison 和 property comparison。

deployment_visible_prefetched:
  query/neighbor 可见信息同 deployment_visible；
  harness 预取与 identity_blind 完全相同的工具结果；
  这是 parity-controlled visibility 补充控制；test 为 21 个条件和 4,456 个配对样本，
  valid 已扩展到 26 个条件和 2,713 个配对样本。

deployment_visible:
  query structure visible，query name hidden；
  neighbor structure、source ID 和数据源已有名称 visible；
  LLM 自主选择分子工具；现有 26 个条件属于历史主制度，不是 v4 默认。
```

## Valid split 可见性诊断

2026-07-23 已用完全相同设置将 valid 的三套制度扩展到每套 26 个条件、2,713 个
sample-condition，全部 0 失败；matched-prefetch 与 identity-blind 的 contract audit 覆盖
2,713/2,713 samples，无 missing、extra 或 mismatch。因此 valid 制度对比在输入和工具
replay 边界上是完整的。

Valid 上 agentic deployment-visible 的最佳条件为 BBB Starling mechanism 0.7596、Skin Starling
mechanism 0.6703、ClinTox ChEMBL mechanism 0.6445 和 Bioavailability Starling mechanism 0.6952。
相应 identity-blind 最佳值为 0.7429、0.7630、0.6229 和 0.6938。这一 split 再次显示：
structure visibility 的效果不是单向提升，Skin 尤其容易因参数先验与运行波动而下降；
ClinTox 和 Bioavailability 则对更强的 mechanism/source context 更有利。Valid 样本数较小，
这些只是方向性 replication，不应用来替换 test 上的 paired inference。完整 valid 数值见
`RESULTS.md` 与 `outputs/paper/molecular_evidence_agent_valid/analysis/report.md`。

Skin Reaction 的逐样本、逐 stage reasoning trace 审计另见
`SKIN_REACTION_VISIBILITY_TRACE_AUDIT.md`。该审计用 matched-prefetch 排除 retrieval/tool
输入差异，定位了五个重复负类分子、single structural prior 与 final evidence-threshold 缺口，
并记录 parent-disjoint 无 prediction flip 的排除性证据。

供 teammate 直接查看的中英文静态案例报告位于
`reports/skin_reaction_visibility_trace_casebook/index.html`。报告默认显示英文，右上角按钮可切换中文；
其中“35 个 flips”明确表示同样 5 个分子在 7 个 retrieval/reasoning conditions 下的重复
sample-condition 输出，不是 35 个独立分子。

Parity-controlled 补充对照中，visible-prefetched 相对 identity-blind 的 26 个 Macro-F1 差值为
16 升、10 降，未加权平均为 -0.0010。BBB ChEMBL flat、Bio ChEMBL mechanism 和 Bio Starling flat
的 bootstrap 区间完全为正，BBB none 完全为负；经过 Holm 校正后仍没有稳定的跨条件单向提升。

| 任务 | none | direct | ChEMBL flat | ChEMBL mechanism | Starling numeric/full direct | Starling flat | Starling mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB | +0.0000 | +0.0114 | -0.0033 | +0.0326 | -0.0160 | 不适用 | 不适用 |
| Skin | -0.0131 | -0.0144 | +0.0208 | -0.0515 | 不适用 | 不适用 | 不适用 |
| ClinTox | +0.0518 | +0.0960 | -0.0074 | -0.0186 | 不适用 | 不适用 | 不适用 |
| Bioavailability | +0.0275 | +0.0355 | +0.0585 | +0.0244 | +0.0339 / +0.0543 | +0.0582 | +0.0220 |

表中为 `visible-prefetched - identity-blind` Macro-F1。该补充对照冻结了 retrieval 和工具输出，只改变
LLM 可见的结构与允许身份信息；但两侧仍是独立生成，因此 hosted endpoint 的运行波动仍包含在差值中。

下面旧的升降案例分析来自 agentic `deployment_visible`，不能称为严格 visibility ablation。它同时允许：

- 从 query structure 激活 parametric chemical/pharmacological knowledge；
- 更直接地解释 query-neighbor transformation；
- 从 source ID 或已知结构识别 molecule identity；
- 产生 identity misrecognition 和 benchmark ontology conflict。

因此论文必须把它描述为 `parametric prior × retrieved evidence` 的联合 setting。

## Agentic 总体统计

21 个一一配对条件中：

```text
Deployment-visible Macro-F1 higher: 14
Deployment-visible Macro-F1 lower: 7
unweighted mean delta: +0.0191
```

这 21 个条件不是 21 个独立任务：同一 test set 和同一批错误样本会在 direct、flat、mechanism 条件中
重复出现。不能用“14 胜 7 负”做简单符号检验，也不能把同一分子在多个条件中的错误当成多个独立案例。

## 性能下降条件

| 条件 | Macro-F1 差值 | Blind 对/Visible 错 | Blind 错/Visible 对 | 主要 confusion 变化 |
|---|---:|---:|---:|---|
| BBB Starling direct | -0.0321 | 17 | 9 | FP +5，FN +3 |
| Skin none | -0.0061 | 10 | 10 | accuracy 不变，类别分布改变 |
| Skin ChEMBL flat | -0.0172 | 12 | 11 | 更倾向预测 risk |
| Skin ChEMBL mechanism | -0.0671 | 13 | 8 | FP 9→15 |
| ClinTox ChEMBL flat | -0.0052 | 17 | 3 | FP 15→31 |
| ClinTox ChEMBL mechanism | -0.0246 | 19 | 2 | FP 13→32 |
| Bioavailability none | -0.0340 | 21 | 13 | FN 42→57 |

7 个 Macro-F1 下降的 paired-bootstrap 95% CI 全部跨 0。ClinTox mechanism 的 paired correctness
McNemar Holm `p=0.0046`，但 Macro-F1 差异区间仍跨 0；这是极端类别不平衡下 accuracy/correctness
与 Macro-F1 不等价的例子。

### 下降原因 A：正确身份知识与 benchmark ontology 冲突

GLM 能从 structure 识别 azacitidine、menadione、linezolid、bleomycin、barbiturates、phenytoin、
fosphenytoin、digitoxin 等分子或类别。它随后调用真实世界中的 cytotoxicity、narrow therapeutic index、
DILI、marrow suppression 或 CNS depression 知识，把它们判为 toxic；但当前 ClinTox gold 对其中很多是 0。

这不是普遍认错分子，而是现实毒性概念与 ClinTox 的窄 clinical-failure label 不一致。ClinTox mechanism
中 21 个 prediction flips 全部为 `non_toxic -> toxic`，只增加 2 个 TP，却增加 19 个 FP。

### 下降原因 B：参数先验或 structural alert 压过 retrieved evidence

Skin visible trace 更频繁使用 Michael acceptor、quinone/quinone-methide haptenation、phototoxic chromophore、
electrophilic sensitization 和 skin permeation alerts。它们是合理 risk signals，但不自动等于 benchmark
positive。三个 Skin 下降条件的 harmful cases 高度重叠，只涉及 14 个不同分子。

BBB Starling direct 中，foscarnet、cotinine、amphotericin B、penicillin V、cephalosporin 和 risperidone/
paliperidone 类结构使模型更重视 passive permeability 或 class prior，有时压过 direct Starling outcome。

### 下降原因 C：没有 retrieval 时做过度机制推断

Bioavailability `none` 没有 retrieved evidence。Structure visible 后，模型更积极推测 ester/carbamate
hydrolysis、CYP oxidation、N/O-dealkylation、glucuronidation、first-pass clearance 和 dissolution
limitation，产生 28 次 `high -> low` flip。FN 从 42 增至 57。

所以这部分不能描述为“参数知识压过 retrieval”，而应描述为 raw-structure prior 缺少实验约束。

### 下降原因 D：身份或家族识别错误

对 109 次 harmful flip 按 task/query 去重后得到 77 个分子。33 个 visible final trace 明确声称具体
名称、source identity、named analog 或药物/scaffold family：

```text
correct or correct-with-scope: 27
incorrect exact name: 2
incorrect scaffold family: 3
misleading mechanistic-family analogy: 1
```

明确错误包括：

- beta-propiolactone 被认成 GBL；
- thiopental 被认成 thiamylal；
- Rifogal 被归为 tetracycline-class；
- dezocine、pentazocine 被归为 morphinan；
- ambenonium 被类比为 hemicholinium-class neurotoxin。

严格名称错误占全部 77 个 unique harmful molecules 的 2.6%；加入 family/analogy 后为 7.8%。因此
misidentification 是真实原因，但不是总体下降的主要解释。

乐观反事实将这 6 个分子的 visible prediction 替换成 blind prediction 后：Skin none、Skin flat 和
ClinTox flat 的差值可变为正；BBB、Bioavailability、Skin mechanism 和 ClinTox mechanism 仍为负。

完整 annotation：

- `outputs/paper/molecular_evidence_agent/analysis/visibility_identity_claim_audit.tsv`
- `outputs/paper/molecular_evidence_agent/analysis/visibility_identity_error_report.md`

### 下降原因 E：运行波动和上下文成本

Hosted endpoint 即使 `temperature=0` 也没有固定 seed。两套 visibility 是独立生成。Skin 只有 82 个
test samples，区间尤其宽。复杂 visible 条件的 token 使用常为 blind 的 1.4 至 2.0 倍；更长的结构比较
可能增加 attention dilution，但当前只证明相关性，尚未完成因果消融。

## 性能上升条件

以下 5 个上升条件的 paired-bootstrap Macro-F1 区间完全高于 0：

| 条件 | Macro-F1 差值 | confusion matrix 改善 |
|---|---:|---|
| BBB none | +0.0281 | TN +4，TP +5 |
| BBB ChEMBL direct | +0.0396 | TN +6，TP +6 |
| BBB ChEMBL mechanism | +0.0429 | TN +6，TP +7 |
| Bioavailability Starling direct full | +0.0728 | TN +5，TP +2 |
| Bioavailability Starling mechanism | +0.0896 | TN +3，TP +7 |

它们同时改善正负类别，不只是 threshold shift。但经过 21 项 Holm correction 后，仍没有一项同时满足
Macro-F1 interval positive 和 McNemar significant，因此以下解释是 trace-supported、尚待重复和跨模型
验证的机制假设。

### 上升原因 A：识别 transporter-mediated exception

Identity-blind 容易根据高 TPSA、两性离子、永久电荷或低 logD 判 BBB fail。Structure visible 后，模型
识别到：

- adenosine 的 ENT/CNT transport；
- L-tryptophan、5-HTP 的 LAT1 transport；
- valproic acid 的 MCT1/MCT2 transport；
- choline 的 CHT1/SLC5A7 context；
- busulfan、hydroxyurea 的 clinical CSF exposure。

参数知识在这里补充了 descriptor-only analysis 缺少的 transport/exposure mechanism。

### 上升原因 B：更准确判断 analog transferability

Neighbor structure visible 后，模型可以把 evidence 与具体 transformation 对齐，例如 etoposide
methyl/thiophene replacement、artemether/beta-arteether OMe/OEt、sulindac sulfoxide/sulfone，以及
benzodiazepine、barbiturate、morphinan scaffold precedent。

它既能保留高度可转移 evidence，也能拒绝 Tanimoto 很高但 ionization、solubility 或 first-pass mechanism
已经改变的 analog。参数知识在这里充当 retrieval evidence 的解释器，而不是简单 override。

### 上升原因 C：解析 salt、charge、active-moiety 和 formulation scope

Bioavailability visible trace 更明确地区分：

- vincamine free base 与 HCl；
- nelfinavir free base 与 mesylate；
- propranolol free base、salt 和 enantiomer；
- ipratropium 的永久季铵 charge；
- estradiol 的高 absorption 与低 unchanged-parent F。

这可以修正 identity-blind 对同一 RDKit molecular-parent record 的普通 analog 处理。该结论解释旧
operational/deployment-visible lineage；新 v4 直接使用 parent-disjoint，不把 same-parent evidence 放入主矩阵。

### 上升原因 D：校准 borderline descriptor

对于 TPSA、logD、neutral fraction 等位于阈值附近的分子，identity-blind 容易累计多个小风险后直接
negative。Structure visible 提供 CNS-active、barbiturate、benzodiazepine、opioid 或 orally-active
scaffold precedent，帮助区分 moderate liability 与真正不可克服的障碍。

## Macro-F1 上升不一定是总体能力上升

| 条件 | Macro-F1 差值 | Blind-only correct | Visible-only correct | 变化 |
|---|---:|---:|---:|---|
| ClinTox none | +0.0518 | 12 | 3 | TP 0→2，FP 2→13 |
| ClinTox direct | +0.0577 | 10 | 3 | TP 0→2，FP 2→11 |

极少数 positive samples 使两个 TP 足以提高 positive-class F1 和 Macro-F1，但总体 accuracy 下降。因此
论文的 visibility 表必须同时给 Macro-F1、accuracy、per-class F1、confusion matrix 和 paired flips。

## 论文可直接使用的结论框架

Structure visibility 的作用分为五类：

```text
complement:
  补充 transporter、clinical exposure 和 scaffold precedent。

scope resolution:
  识别 salt、active moiety、formulation 和关键 transformation。

override:
  参数知识压过 retrieved evidence，可能正确也可能错误。

ontology conflict:
  真实世界知识与 benchmark label 定义不一致。

misidentification:
  具体名称、scaffold family 或 mechanism family 识别错误。
```

推荐论文表述：

> Structure visibility activates parametric chemical and pharmacological knowledge. This knowledge can complement
> retrieved evidence by resolving transport and transferability, but can also override source evidence, conflict with
> benchmark ontologies, or introduce molecular misidentification. Deployment-visible performance therefore measures
> a joint parametric-retrieval system rather than retrieval alone.

不应写成：

> Structure visibility generally improves molecular prediction.

也不应写成：

> Deployment-visible failures are all caused by memorization overriding retrieval.

## 尚需完成的验证

1. 由第二位 annotator 独立复核 33 个 identity-bearing harmful cases，并报告一致率。
2. 对预注册的上升/下降条件做至少 3 次独立生成，估计 run-to-run variance。
3. 运行 query-only、neighbor-only、source-ID-hidden visibility ablation。
4. 在第二模型上复现代表性的 BBB/Bio gain 和 Skin/Clin failure。
5. 最终 source freeze 后重跑 operational relation annotation 与 parent-disjoint backfill，再检查 salt、
   independently annotated active-moiety 案例和全部 effect size；不得用 RDKit parent key 代替 active-moiety annotation。

完成这些验证前，本文档中的原因分类是 exploratory mechanism analysis，而不是严格因果证明。
