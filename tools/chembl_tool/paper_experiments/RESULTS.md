# 冻结的全量实验结果

运行日期：2026-07-10 至 2026-07-13。

当前共有 63 个 GLM 条件和 1 个标量 KNN 基线：

- `identity_blind`：query 和 retrieved molecule 的结构、名称及源标识符均不发送给 LLM。
- `deployment_visible_prefetched`：结构与允许身份信息可见，但 harness 提供与 identity-blind 完全一致的工具结果；它是 parity-controlled visibility 补充控制。
- `deployment_visible`：query structure visible、query name hidden；retrieved molecule 的 structure、source ID 和数据源已有名称 visible；模型自主选择 group comparison tools。

论文使用口径已更新：`deployment_visible` agentic 结果作为真实部署主实验候选；identity-blind 和
matched-prefetch 只作为补充控制。当前三套结果都使用第一轮 operational retrieval，其中 same-parent
盐型、溶剂化物、重复组分或 formulation-linked record 仍可能进入 neighbor 集合。对应 17 个现有
retrieval 条件的 parent-disjoint 消融已经完成；由于 Starling 数据仍未补齐和最终冻结，本文件中的数值
仍是 exploratory result，不是最终论文主表。

三套制度各有 21 个条件，全部满足 `n_successful == n_total` 且 `n_failed_runs == 0`。Prefetch parity audit
逐 query 比较了 4,456 对工具 contract，结果为 4,456/4,456 完全一致。请求模型为
`zai-org/GLM-5.2-FP8`，端点实际返回 `hosted_vllm/nvidia/GLM-5.2-NVFP4`。

Identity-blind 使用 116,008,812 tokens；matched-prefetch visible 使用 121,482,001 tokens；agentic visible
使用 189,137,772 tokens。Agentic tool-use 比 matched-prefetch 多约 56% tokens。

机器可读指标、10,000 次 bootstrap、精确 McNemar 检验、Holm 校正、检索覆盖率、成本和可见性审计位于：

```text
outputs/paper/molecular_evidence_agent/analysis/
```

## Validation split 诊断重跑（2026-07-17）

Validation split 使用与 test 完全相同的冻结设置，输入只由各任务的 `test.jsonl`
替换为 `valid.jsonl`。这是用于检查结论方向和 pipeline 可复现性的诊断重跑，不代替
test 主结果。Identity-blind、matched-prefetch 和 deployment-visible 各完成 21 个条件、
2,203 个 sample-condition，三套失败数均为 0。Prefetch parity audit 为 21/21 conditions、
2,203/2,203 samples，missing、extra 和 mismatch 均为 0。

### Valid Identity-Blind

| 任务 | 无检索 | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB_Martins | 0.7098 | 0.7138 | 0.6954 | 0.7008 | **0.7429** | 不适用 | 不适用 |
| Skin_Reaction | **0.7396** | 0.6931 | 0.7163 | 0.6647 | 不适用 | 不适用 | 不适用 |
| ClinTox | 0.4774 | 0.4774 | **0.6229** | **0.6229** | 不适用 | 不适用 | 不适用 |
| Bioavailability_Ma | 0.5031 | 0.5594 | 0.5713 | 0.5194 | 0.6657 仅数值 / 0.6522 完整 | 0.6657 | **0.6938** |

### Valid Deployment-Visible（Matched Prefetch 补充控制）

| 任务 | 无检索 | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB_Martins | 0.6677 | 0.7187 | 0.7411 | 0.7050 | **0.7436** | 不适用 | 不适用 |
| Skin_Reaction | 0.5943 | **0.6577** | **0.6577** | **0.6577** | 不适用 | 不适用 | 不适用 |
| ClinTox | 0.5358 | 0.5420 | **0.6352** | 0.5971 | 不适用 | 不适用 | 不适用 |
| Bioavailability_Ma | 0.5333 | 0.6382 | 0.6382 | 0.6530 | 0.6395 仅数值 / 0.6938 完整 | **0.7388** | 0.7083 |

### Valid Deployment-Visible（Agentic 主制度）

| 任务 | 无检索 | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB_Martins | 0.6927 | 0.7138 | 0.7212 | 0.7138 | **0.7381** | 不适用 | 不适用 |
| Skin_Reaction | 0.5733 | 0.5733 | 0.5733 | **0.6050** | 不适用 | 不适用 | 不适用 |
| ClinTox | 0.5690 | 0.5842 | 0.6352 | **0.6445** | 不适用 | 不适用 | 不适用 |
| Bioavailability_Ma | 0.4421 | 0.6135 | 0.6135 | 0.6395 | 0.6264 仅数值 / 0.6395 完整 | 0.6264 | **0.6952** |

表中均为 valid macro-F1。Bioavailability 标量 KNN 的 valid macro-F1 为 0.6621，与 LLM
agent 条件分开报告。

### Valid Parent-disjoint 消融

| 任务 | 无检索 | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB_Martins | 0.6927 | 0.7138 | 0.7261 | 0.7138 | **0.7641** | 不适用 | 不适用 |
| Skin_Reaction | 0.5733 | 0.5733 | 0.5733 | **0.6050** | 不适用 | 不适用 | 不适用 |
| ClinTox | 0.5690 | 0.5842 | **0.6919** | 0.6137 | 不适用 | 不适用 | 不适用 |
| Bioavailability_Ma | 0.4421 | 0.6008 | 0.6000 | 0.6008 | 0.5749 仅数值 / 0.6008 完整 | 0.5995 | **0.6656** |

17 个 retrieval conditions 共 1,765 个 sample-condition，失败数为 0。其中 197 个
retrieval 输入变化，1,568 个整条 run 复用；35 个 prediction flips 中 14 个修正、
21 个破坏。最终 retained-neighbor identity conflict 和低于 similarity threshold 的补位均为 0。

Valid 上 parent-disjoint 后每个任务的最佳 retrieval 点估计仍高于 no-retrieval，但效果明显
依赖任务：BBB 最佳为 Starling direct，ClinTox 最佳为 ChEMBL flat，Skin 只在 mechanism
有小幅提升，Bioavailability 最佳为 Starling mechanism。Operational 到 parent-disjoint 的变化
方向并不一致，mechanism 也没有稳定超过 flat；17 项 Holm 校正后都不显著。
因此 valid 支持“retrieval 信号可复现但强烈依赖任务”，不支持将单个 valid 点估计
改写为新的最终论文 claim。

完整指标、paired tests、coverage、tokens 和 provenance 位于：

```text
outputs/paper/molecular_evidence_agent_valid/analysis/report.md
outputs/paper/molecular_evidence_agent_valid/analysis/prefetch_contract_audit.json
outputs/paper/molecular_evidence_agent_valid/analysis/parent_disjoint_ablation/result_report.md
outputs/paper/molecular_evidence_agent_valid/analysis/figures/retrieval_claims_overview.svg
outputs/paper/molecular_evidence_agent_valid/analysis/figures/retrieval_claims_overview_highres.png
```

## Identity-Blind

| 任务 | 无检索 | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB_Martins | 0.7148 | 0.7325 | 0.7399 | 0.7248 | **0.7972** | 不适用 | 不适用 |
| Skin_Reaction | 0.6683 | 0.6450 | 0.6459 | **0.6822** | 不适用 | 不适用 | 不适用 |
| ClinTox | 0.4819 | 0.4819 | 0.5553 | **0.5891** | 不适用 | 不适用 | 不适用 |
| Bioavailability_Ma | 0.5525 | 0.6866 | 0.6589 | 0.6664 | 0.6908 仅数值 / 0.6579 完整 | 0.6938 | **0.7101** |

## Deployment-Visible（Matched Prefetch 补充控制）

| 任务 | 无检索 | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB_Martins | 0.7149 | 0.7439 | 0.7366 | 0.7574 | **0.7812** | 不适用 | 不适用 |
| Skin_Reaction | 0.6553 | 0.6306 | **0.6667** | 0.6306 | 不适用 | 不适用 | 不适用 |
| ClinTox | 0.5337 | **0.5779** | 0.5480 | 0.5706 | 不适用 | 不适用 | 不适用 |
| Bioavailability_Ma | 0.5800 | 0.7221 | 0.7174 | 0.6908 | 0.7247 仅数值 / 0.7121 完整 | **0.7520** | 0.7322 |

## Deployment-Visible（Agentic 主实验候选）

| 任务 | 无检索 | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB_Martins | 0.7429 | **0.7721** | 0.7579 | 0.7677 | 0.7652 | 不适用 | 不适用 |
| Skin_Reaction | **0.6622** | 0.6511 | 0.6286 | 0.6150 | 不适用 | 不适用 | 不适用 |
| ClinTox | 0.5337 | 0.5396 | 0.5501 | **0.5645** | 不适用 | 不适用 | 不适用 |
| Bioavailability_Ma | 0.5185 | 0.7221 | 0.6908 | 0.7077 | 0.7174 仅数值 / 0.7307 完整 | 0.7520 | **0.7997** |

表中均为 test macro-F1。Bioavailability 标量 KNN 的 macro-F1 为 0.6952。

## Parent-disjoint 消融

| 任务 | 无检索 | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB_Martins | 0.7429 | **0.7721** | 0.7579 | 0.7677 | 0.7719 | 不适用 | 不适用 |
| Skin_Reaction | **0.6622** | 0.6511 | 0.6286 | 0.6150 | 不适用 | 不适用 | 不适用 |
| ClinTox | 0.5337 | 0.5396 | **0.5711** | 0.5424 | 不适用 | 不适用 | 不适用 |
| Bioavailability_Ma | 0.5185 | 0.7221 | 0.6908 | 0.7077 | 0.7006 仅数值 / 0.6770 完整 | 0.7339 | **0.7567** |

表中均为 test macro-F1。`无检索` 不包含 neighbor，因此不受 parent policy 影响，直接沿用同一
agentic Deployment-Visible 结果；其余单元格来自完整 test set 的 parent-disjoint condition。没有
same-parent overlap 或没有 prediction flip 的条件与上表数值相同。

2026-07-13 完成 agentic deployment-visible 的首轮 parent-disjoint 消融。公共 RDKit normalizer 对 whole
record、connectivity variant、fragment parent 和 mixture components 建立版本化 identity；消融排除
`exact_record`、`same_connectivity_variant` 和 `same_parent`，然后只从原 `min_similarity=0.30` 以上的
候选向后回填。17 个 retrieval 条件共 3,568 个 sample-condition，其中 424 个 LLM-visible retrieval
输入发生变化，3,144 个完整 run 直接复用；变化样本的 1,047 个 group branches 中又复用了 471 个。

全部 17 个条件均满足 `n_failed_runs=0`。最终产物重新标准化审计得到 retained-neighbor identity conflict
为 0，低于 similarity threshold 的补位为 0。60 个 prediction flips 中 26 个修正、34 个破坏。完整
condition 表、逐样本 flips、coverage 和 provenance 位于：

```text
outputs/paper/molecular_evidence_agent/analysis/parent_disjoint_ablation/
  result_report.md
  condition_results.tsv
  prediction_flips.tsv
  result_summary.json
```

主要点估计如下：

- BBB 的 ChEMBL direct/flat/mechanism 均无 prediction flip；Starling direct 从 0.7652 小幅升至 0.7719。
- Skin 三个条件没有 same-parent overlap，因此结果完全不变。
- ClinTox ChEMBL flat 从 0.5501 升至 0.5711，但 mechanism 从 0.5645 降至 0.5424；类别极不平衡，必须结合 confusion matrix 而非只看 accuracy。
- Bioavailability ChEMBL direct/flat/mechanism 均无 prediction flip。Starling numeric direct 从 0.7174 降至 0.7006，full direct 从 0.7307 降至 0.6770，full flat 从 0.7520 降至 0.7339，full mechanism 从 0.7997 降至 0.7567。

10,000 次 paired bootstrap 中，ClinTox flat 的 operational-to-parent 差值区间为 +0.0065 至 +0.0404，
Bioavailability Starling full direct 为 -0.1004 至 -0.0133；其余非零差值区间均跨 0。17 项 exact
McNemar 经 Holm 校正后没有一项显著，最小校正后 `p=0.2656`。因此同 parent 的影响有明确局部信号，
但当前不能宣称它在多重比较后普遍改变性能。

这说明 same-parent 文献证据确实解释了 Starling 部分 operational 增益，尤其影响 Bioavailability high 类
召回；但不支持“retrieval 提升只是同实体答案查找”。去除 same-parent 后，Bioavailability 的 ChEMBL
direct、Starling full direct 和 Starling full mechanism 仍分别比同一 agentic no-retrieval baseline 高
0.2037、0.1586 和 0.2383 macro-F1。BBB 的 ChEMBL/Starling direct 仍高约 0.029。相反，Skin direct
仍低 0.0112，说明 analog retrieval 效应明确依赖任务和证据质量。

在 parent-disjoint 下，Bioavailability Starling mechanism 为 0.7567，仍高于同源 flat 的 0.7339，差值
+0.0228；但该差值尚未做本轮 paired bootstrap/McNemar/Holm 更新，当前只能作为点估计，不能宣称显著。

## 制度内结论

### 相似分子检索

- Identity-blind Bioavailability 从无检索到 ChEMBL direct 提升 0.1342，95% CI 为 0.0471 至 0.2186，Holm 校正后的 McNemar `p=0.0041`。
- Matched-prefetch Bioavailability 的同项提升为 0.1421，95% CI 为 0.0552 至 0.2297，Holm 校正后 `p=0.0099`。
- Matched-prefetch BBB 的 ChEMBL direct 相对无检索提升 0.0290，bootstrap CI 为 0.0014 至 0.0566，但 Holm 校正后的 McNemar `p=0.6069`。ClinTox direct 点估计为 +0.0442，但 coverage 只有 3.15%；Skin direct 为 -0.0246。

因此，“retrieval 可以显著帮助分子性质分类”得到支持，但该效应明确依赖任务和 coverage；Bioavailability 是最稳定的正例。

### Starling 与 ChEMBL

- Identity-blind BBB 的 Starling direct 相对 ChEMBL direct 提升 0.0647，95% CI 为 0.0167 至 0.1130，Holm 校正后 `p=0.0218`，是最明确的数据源效应。
- Matched-prefetch BBB 的同项差异为 +0.0374，95% CI 为 -0.0128 至 0.0889，不显著。
- Matched-prefetch Bioavailability 的 Starling numeric direct 相对 ChEMBL direct 仅 +0.0026；Starling full mechanism 相对 ChEMBL mechanism 为 +0.0413，区间 -0.0448 至 0.1306，均不显著。
- Agentic Bioavailability 的 Starling full mechanism 达到 0.7997，但 matched-prefetch 对应值为 0.7322；该 agentic 高分同时改变了工具执行策略，不能作为纯数据源效应。

因此，Starling 有强正向结果，但“文献抽取数据普遍优于 ChEMBL”仍不能作为跨任务的普遍结论。

### 机制拆分

- Matched-prefetch 的 mechanism 相对同源 flat 点估计：BBB +0.0208、ClinTox +0.0226、ChEMBL Bioavailability -0.0266、Starling Bioavailability -0.0198、Skin -0.0361。
- Agentic 的对应点估计为 BBB +0.0098、ClinTox +0.0144、ChEMBL Bioavailability +0.0169、Starling Bioavailability +0.0478、Skin -0.0136。
- Identity-blind 的对应结果也不一致：Skin、ClinTox 和 Starling Bioavailability 为正，BBB 和 ChEMBL Bioavailability 为负。
- 所有 flat-vs-mechanism 比较的配对区间均跨零，Holm 校正后均不显著。

机制拆分目前只能表述为任务依赖的组织方式，尚未证明普遍、显著地提高性能；matched-prefetch 中 5 个点估计只有 2 个为正。它还显著增加 token 成本，必须作为性能与成本权衡报告。

### 非数值证据

- Matched-prefetch Bioavailability 中，Starling full direct 相对 numeric-only direct 为 -0.0126，95% CI 为 -0.0656 至 0.0425。
- Agentic 中该差异为 +0.0133，95% CI 为 -0.0393 至 0.0713。
- Identity-blind 中该差异为 -0.0330，95% CI 也跨零。
- 标量 KNN 与两套 GLM numeric direct 均无显著差异。

当前实验不支持“非数值 evidence 必然带来增益”；matched-prefetch 和 identity-blind 点估计均为负，只有 agentic setting 略为正向且区间跨零。

## Exploratory 跨制度比较

以下旧比较同时改变了结构可见性和工具执行策略，不是严格 visibility ablation。观察结果是：

- Deployment-visible 相对 identity-blind 的 macro-F1 在 BBB ChEMBL mechanism 为 +0.0429，在 Bioavailability Starling mechanism 为 +0.0896；两者 bootstrap CI 均为正。
- BBB Starling direct 为 -0.0321，Skin mechanism 为 -0.0671，ClinTox mechanism 为 -0.0246。
- 21 项跨制度比较经过 Holm 校正后，没有一项同时满足 macro-F1 配对区间为正且 McNemar 显著。
- ClinTox mechanism 的 McNemar Holm `p=0.0046`，但 deployment macro-F1 更低且其差异 CI 跨零；这反映类别不平衡下 accuracy/correctness 与 macro-F1 的目标并不等价。

因此这些数值只能用于提出假设。Matched-prefetch 补充控制已完成；当前 `deployment_visible`
结果只回答模型自主工具调用下的实际 agent 行为。

Matched-prefetch 与 agentic 的 21 个差值为 10 升、8 降、3 个相同，未加权平均仅 +0.0005；agentic
tokens 却高约 56%。只有 Bioavailability Starling mechanism 的 agentic 增益区间完全高于 0
（+0.0676，95% CI 0.0125 至 0.1308），但 Holm 校正后不显著。

### Deployment-visible 身份识别错误审计

对 7 个 deployment-visible Macro-F1 点估计下降条件中的 `blind correct -> visible wrong` 翻转
进行去重后，共有 109 次 harmful flip、77 个独立分子。其中 33 个 visible final trace 明确声称
了具体名称、source identity、named analog 或药物/scaffold family：27 个判断正确或只有盐型、
离子态和未指定立体化学等 scope 差异；2 个具体名称错误；3 个 scaffold family 错误；另有 1 个
具有误导性的 mechanistic-family 类比。

明确名称错误包括把 beta-propiolactone 识别为 GBL，以及把 thiopental 识别为 thiamylal。
较宽口径错误包括把 Rifogal 归为 tetracycline-class、把 dezocine/pentazocine 归为 morphinan，
以及把 ambenonium 类比为 hemicholinium-class neurotoxin。相反，ClinTox harmful flips 中
azacitidine、menadione、lindane、linezolid、barbiturates、phenytoin 等大量身份识别是正确的；
主要冲突来自模型正确识别真实药物毒性后，与 benchmark 的窄 label ontology 不一致。

把已确认错误样本乐观地替换回 blind prediction 后，Skin none、Skin flat 和 ClinTox flat 的差值
可以变为正，但 BBB Starling direct、Bioavailability none、Skin mechanism 和 ClinTox mechanism
仍为负。因此身份误识别解释了部分下降，但不是总体下降的主要原因。完整逐样本 annotation 和
反事实上界见 `outputs/paper/molecular_evidence_agent/analysis/visibility_identity_claim_audit.tsv`
及 `visibility_identity_error_report.md`。该 annotation 属于 exploratory manual audit，正式论文前
应由第二位 annotator 独立复核并报告一致率。

### Deployment-visible 性能上升机制

21 个跨 visibility 条件中有 14 个 Macro-F1 点估计上升，未加权平均差值为 +0.0191。其中只有
BBB none、BBB ChEMBL direct、BBB ChEMBL mechanism、Bioavailability Starling direct full 和
Bioavailability Starling mechanism 的 paired-bootstrap 区间完全高于 0。这 5 个条件同时改善了
正负类别：BBB 三个条件分别获得 `TN/TP +4/+5`、`+6/+6` 和 `+6/+7`；两个 Bioavailability
条件分别为 `+5/+2` 和 `+3/+7`。

Trace 支持四种主要增益机制：识别 LAT1、MCT、ENT/CNT 等 transporter-mediated BBB exception；
利用具体 scaffold/transformation 改善 analog transferability 判断；识别 salt、charge state、active
moiety 和 formulation scope；以及用 CNS-active/orally-active scaffold precedent 校准 borderline
descriptor。这里参数知识不是简单压过 retrieval，而是作为 retrieval evidence 的解释器。

ClinTox none/direct 虽然 Macro-F1 分别上升 +0.0518/+0.0577，但 accuracy 实际下降：模型各救回
2 个 TP，同时分别新增 11/9 个 FP。该现象来自极端类别不平衡，不能表述为总体能力提升。论文必须
同时报告 paired flips、confusion matrix、accuracy 和 per-class F1。完整分析见
`outputs/paper/molecular_evidence_agent/analysis/visibility_gain_analysis.md`。

因此 structure visibility 应描述为 parametric prior 与 retrieved evidence 的双向交互：它可以补充
mechanism、解决 evidence scope，也可以错误 override evidence、触发 ontology conflict 或发生身份误识别。
当前 trace 分析是机制解释而非因果证明；严格归因仍需在 discordant samples 上补充 query-only、
neighbor-only 和 source-ID-hidden ablation，并通过重复生成估计运行波动。

## 审计说明

- 21 个 identity-blind 条件在 prompt 边界上的 query/neighbor structure、identifier 和 name 泄漏均为 0。
- 21 个 matched-prefetch 和 21 个 agentic deployment-visible 条件全部通过正向 contract 审计：每个 query 的 input/canonical SMILES 可见；存在 neighbor 时，每个 neighbor 的 structure 和 source ID 可见；数据源提供名称时，至少一个对应 source name 可见。输入数据不包含 query name，因此不会主动向 LLM 提供 query 名称。
- Matched-prefetch 逐样本复用 blind 的冻结 `retrieval.json` 和 prefetched tool outputs。最终审计覆盖 21/21 conditions、4,456/4,456 样本，missing、extra、mismatch 均为 0；两侧所有 single/group/final branch 状态均为 `ok`。
- Deployment 正向审计同时支持 JSON 嵌套字符串中的反斜杠转义 SMILES，避免把实际可见的立体结构误报为缺失。
- Matched-prefetch 初次运行暴露了 retrieval mapping 漂移、MCS 输出非确定性及旧 branch 状态未进入 metrics 的问题。旧 attempt 均归档；修复后通过 retrieval/tool replay、严格 branch gate 和 `--skip-existing` 只重跑受影响 index。
- 一条 identity-blind BBB flat 的原始 assistant reasoning 根据允许使用的性质/MMP 证据重构出 `*CC(C)CO`。该字符串不存在于 LLM 请求输入，并在 final synthesis 前被清除，因此属于 assistant 侧重构诊断，不是上游泄漏。
- 结构化 JSON 最多尝试 4 次；batch 完整性 gate 会拒绝任一必需分支不完整的结果。报告只使用最终完整 test set。
