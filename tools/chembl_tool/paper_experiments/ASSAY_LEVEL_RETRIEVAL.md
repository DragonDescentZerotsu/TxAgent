# Starling assay-level retrieval

更新时间：2026-08-27。

本文档只维护当前合同、最终 valid 结果、可复现入口和历史边界。逐次 smoke、失败重试、endpoint
切换和被替代的 trace 不在这里重复记录；它们已集中归档，避免主输出目录继续膨胀。

## Canonical snapshot

当前 paper-facing gold 与 conditioned progressive 实验不是同一个 promotion 决策：

- BBB paper-facing gold 仍是 `experimental_meaningful_cns_access_v3`；本文的 BBB 曲线是隔离的
  gold-v4 / conditioned-v3 matched candidate。
- Bioavailability 和 Skin 仍使用各自当前 conditioned `record_supported_v2` lineage。
- ClinTox 没有合格 condition taxonomy，本轮不进入 progressive 曲线；既有 ClinTox pipeline 和
  historical controls 保持可复现，但不与三任务新图混表。

当前三任务 valid artifact：

| task | benchmark input | progressive protocol | source purity | n |
|---|---|---|---|---:|
| BBB | `processed_starling_context_conditioned_selected_v3` | `conditioned_assay_progressive_visible.v8` | `bbb_source_family_purity.v5` | 397 |
| Bioavailability | `processed_starling_context_conditioned_selected_v1` | `conditioned_assay_progressive_visible.v7` | `source_family_purity.v1` | 262 |
| Skin | `processed_starling_context_conditioned_selected_v1` | `conditioned_assay_progressive_visible.v7` | `source_family_purity.v1` | 246 |

所有完成点均为 scaffold-valid、`deployment_visible_prefetched`、DeepSeek-V4-Flash-0731、零失败。

## Retrieval and leakage contract

三个任务共享以下合同：

1. source preparation 只删除 valid/test parents 的 benchmark-defining direct rows；同一 heldout parent 的
   non-direct mechanism rows仍可作为 analog evidence。
2. query-time 对每个候选执行 `scaffold_disjoint`；相同 scaffold 不能进入 retrieval。
3. Morgan Tanimoto 最低阈值为 `0.3`。
4. candidate generation 以 cumulative record-family pool 中的 molecule 为单位，按全局 Morgan similarity
   排序；没有 per-assay neighbor cap。
5. assay identity 只作为 card provenance 和 card-selection diversity tie-break，不决定 family 可见性，
   也不再把一个 assay 强制映射到唯一 family。
6. family assignment 是 record-level；同一 physical assay 的不同 records 可以在不同 levels 出现，但一张
   record card 只能按自己的 family 解锁。
7. query SMILES 与 analog identity 对模型可见；prompt 保留
   `Do not identify the query by name even if its structure is recognizable.`，不禁止模型使用一般化学知识。

`direct_only_heldout_filtered + scaffold_disjoint` 同时保留了 mechanism analog 的覆盖能力和 scaffold-split
隔离，不能简写为“只从 train molecules retrieve”。

## Append-only progressive protocol

L1 从 direct pool 中选择最多 10 个不同 molecules；每个 molecule 最多 4 张 cards。后续每个 level：

- 最多新增 3 个此前从未 active 的 molecules，每个从该 level 新增最多 2 张 cards；
- 最多为 3 个已经 active 的 molecules 增补该 level 新解锁的 cards，每个最多 2 张；
- 两类名额不互借；已经 active 的 molecule 不占“新增 molecule”名额；
- active evidence 只增加、不删除，没有全程累计的 per-molecule card cap；
- 如果本 level 没有 evidence delta，直接 carry forward，不调用模型；
- 所有旧 cards 在下一轮仍可见，并标记上一轮是否进入 decision basis；模型先评估新增 cards 对 prior
  decision 的影响，但仍可回看旧 evidence 修正早期判断。

模型每个 level 只调用一次，不按 card 或 assay 分别调用。默认不启用 flip verifier；没有差异化新增信息时，
第二次同模型确认只会增加成本和保守偏差。

Prompt profile 为 `progressive_compact_tools_short_aliases.v2`。Card 使用短 alias，省略重复 ID 列表；原始
endpoint、measurement、species、conditions、support text 和 provenance 保留，不调用 GPT-OSS summary，
不做字段级截断。`reasoning_effort` 参数省略、provider thinking 保持默认，completion cap 为 20,480 tokens。

## Source-family purity

### BBB v5

`bbb_source_family_purity.v5` 审计全部 581,708 条 source rows，只修改 `group_id` 并附加审计字段，不改
measurement/support text。L1 仅允许当前 accepted non-prediction voters，或能够重放
`experimental_meaningful_cns_access_v4` gold contract 的实验 CNS-access outcomes。

五层依次为：

| level | family | cumulative assays |
|---:|---|---:|
| 1 | direct measured CNS access | 1,161 |
| 2 | central functional / prediction / generic near-direct proxy | 7,976 |
| 3 | passive permeability | 8,356 |
| 4 | efflux transport | 22,858 |
| 5 | influx transport | 22,958 |

PAMPA 不在 L1；MDCK records 按实际 permeability/efflux/influx readout 分配；QikProp、BOILED-Egg 和
missing/generic BBB outcomes 保留为 evidence，但不能冒充 gold-compatible direct measurement。明确 efflux
signal 优先于泛化 uptake/transporter 词。v5 的 purity gate 为：L1 residual prediction=0、L1 residual
non-gold-contract=0、cross-family efflux/influx precedence violation=0、被移动 gold-vote rows=0。

Gold-v4 本身来自 source-index 级 qualitative-direction review：批准 65 条新增 votes，得到 3,675 parents，
train/valid/test 为 2,945/365/365；shared-parent label/split changes 均为 0，identity/scaffold overlap 为 0。
Conditioned-v3 将其展开为 3,053/397/396 个 parent-condition rows，并保持 parent/scaffold disjoint。

### Bioavailability and Skin

Bioavailability 六层累计 assays 为 `462 / 478 / 595 / 1,467 / 1,890 / 2,140`：direct oral F、non-direct
bioavailability、oral AUC/Cmax exposure、absorption/solubility/permeability、gut-wall/efflux/metabolism、
hepatic clearance/metabolic stability。L2 仍记为 indirect information，不能与 L1 合并描述成 direct。

Skin 两层累计 assays 为 `530 / 1,219`：gold-compatible sensitization outcome 与 sensitization AOP evidence。
LLNA final outcome 只能在 direct；MIE/KE evidence 只进入 AOP。

## Completed valid results

Macro-F1：

| task | None | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB | 0.6207 | 0.7314 | 0.7475 | 0.7475 | 0.7563 | **0.7563** | — |
| Bioavailability | 0.5412 | 0.7517 | 0.7797 | 0.7797 | 0.7797 | 0.7797 | **0.7856** |
| Skin | **0.6119** | 0.6004 | 0.5939 | — | — | — | — |

最新 matched baselines：

| task | MiniMol head | MiniMol KNN condition | MiniMol KNN all | Morgan KNN condition | Morgan KNN all |
|---|---:|---:|---:|---:|---:|
| BBB gold-v4 | **0.6674** | 0.6071 | 0.5708 | 0.5958 | 0.6174 |
| Bioavailability | 0.5872 | 0.5984 | **0.6229** | 0.5891 | 0.5888 |
| Skin | **0.6030** | 0.5755 | 0.5755 | 0.5208 | 0.5180 |

最终 level 的平均 active evidence 与模型调用长度：

| task | molecules | cards | cards/molecule | prompt tokens/call | reasoning tokens/call |
|---|---:|---:|---:|---:|---:|
| BBB L5 | 10.61 | 25.43 | 2.32 | 16,257 | 4,898 |
| Bioavailability L6 | 12.21 | 39.85 | 3.29 | 17,889 | 9,348 |
| Skin L2 | 6.02 | 15.91 | 2.13 | 10,168 | 8,832 |

Prompt/reasoning 均只对实际模型调用求平均；carry-forward 和 reused-none checkpoints 不进入长度均值。

## Canonical artifacts

完整最终 traces：

```text
BBB:
outputs/paper/
  starling_conditioned_assay_progressive_visible_v8_global_molecule_source_purity_v5/
  scaffold_valid_deepseek_v4_flash_0731/

Bioavailability + Skin:
outputs/paper/
  starling_conditioned_assay_progressive_visible_v7_source_purity_v1/
  scaffold_valid_deepseek_v4_flash_0731/
```

当前可复现 inputs：

```text
outputs/paper/starling_conditioned_assay_family_curve_v1/
  source_overlays/bbb_source_family_purity_v5/
  source_overlays/source_family_purity_v1/{bioavailability_ma,skin_reaction}/
  family_catalogs_mechanism_tagged_v1/
    {bbb_martins_source_purity_v5,bioavailability_ma_source_purity_v1,
     skin_reaction_source_purity_v1}/
  indices/
    bbb_martins/mechanism_tagged_v4_source_purity_v5/
    bioavailability_ma/mechanism_tagged_v4_source_purity_v1/
    skin_reaction/mechanism_tagged_v4_source_purity_v1/
```

当前英文总图与数据表：

```text
outputs/paper/
  starling_conditioned_assay_progressive_visible_v8_global_molecule_source_purity_v5/
  analysis/three_task_progressive_overview_bbb_v5_matched_baselines/
    progressive_overview_metrics.tsv
    summary.json
    figures/three_task_progressive_overview_bbb_v5_matched_baselines.{png,svg}
```

Top-20 historical control 只在主目录保留 compact analysis 与 visible-standard `none` single-cache；其余完整
traces、旧 progressive versions、replay batches、旧 indices 和 prompt audits 可逆归档到：

```text
outputs/archive/conditioned_assay_superseded_20260827/
```

归档不参与当前 evidence retrieval；它保留原相对子目录，必要时可移回。历史 family-curve 与 Top-20
canonical paths 只保留 compatibility symlink，以继续解析 aggregate metrics 和 progressive single-cache。
Gold/source datasets、migration receipts、current metrics/predictions 和最终三任务 traces 未删除。
唯一移出 `data/` 主树的是从未进入正式实验的 unselected
`processed_starling_context_conditioned_reviewed_v1` 和已被 v2 取代的 BBB source-family review ledger；两者
仍完整保存在 archive。

## Maintained entrypoints

```text
progressive runner:
  tools/chembl_tool/paper_experiments/run_conditioned_assay_progressive_curve.py

shared retrieval/state:
  tools/chembl_tool/common/assay_retrieval.py
  tools/chembl_tool/common/progressive_assay_reasoning.py

source purity and audits:
  tools/chembl_tool/paper_experiments/build_bbb_source_family_purity.py
  tools/chembl_tool/paper_experiments/build_conditioned_source_family_purity.py
  tools/chembl_tool/paper_experiments/audit_bbb_source_family_purity.py
  tools/chembl_tool/paper_experiments/analyze_source_family_purity_retrieval_changes.py

catalog/index:
  tools/chembl_tool/paper_experiments/build_assay_family_catalog.py
  tools/chembl_tool/common/assay_retrieval.py

analysis/plot:
  tools/chembl_tool/paper_experiments/analyze_progressive_trace_adoption.py
  tools/chembl_tool/paper_experiments/plot_assay_retrieval_curve.py

parallel historical experiments:
  tools/chembl_tool/paper_experiments/run_conditioned_assay_family_curve.py
  tools/chembl_tool/paper_experiments/run_assay_retrieval_curve.py
```

公共逻辑只放在 `common/`；task directories 只维护 family/endpoint descriptions、benchmark adapter 和 prompt
schema。Progressive、cumulative-family 和 4x assay-prefix 三条 pipeline 共享检索组件，但 runner、manifest 和
artifact root 分离，不能互相覆盖。

## Reproduction

Resume/current runner（默认 output root 指向 v8/v5 successor；正式运行前应显式限制 tasks，避免把已冻结
Bio/Skin v7 意外覆盖到新 root）：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.run_conditioned_assay_progressive_curve \
  --tasks bbb_martins \
  --parallelism 128
```

重画当前完整图：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.plot_assay_retrieval_curve \
  --conditioned-progressive-overview \
  --conditioned-progressive-baseline-root \
    bbb_martins=outputs/baselines/starling_conditioned_bbb_gold_v4_valid_v1 \
  --analysis-dir \
    outputs/paper/starling_conditioned_assay_progressive_visible_v8_global_molecule_source_purity_v5/analysis/three_task_progressive_overview_bbb_v5_matched_baselines \
  --output-stem three_task_progressive_overview_bbb_v5_matched_baselines
```

绘图器会校验 task/sample count、完整 agent metrics 和 `n_failed_runs=0`；baseline cohort 不匹配时默认报错，
只有显式 `--omit-mismatched-progressive-baselines` 才允许诊断性省略，并在 summary 中记录原因。

## Historical boundaries

- 4x assay-count scaling 仍使用 `run_assay_retrieval_curve.py`，按 frozen assay relevance prefix 扩展到 all；
  它不是 progressive family curve。
- cumulative-family flat runner 仍使用每 assay Top-3 的历史合同；它不代表当前 global-molecule v8 candidate
  generation。它的 frozen catalog/index/replay defaults 显式指向 archive，不会污染当前 source-purity inputs。
- fixed Top-20 standard/causal-bridge 与 blind/visible matrix 是已完成且失败的 ablation；只保留 frozen
  artifacts/receipts，专用 launcher、causal-adoption wrapper 和绘图模式已从维护代码删除。
- indirect-only cap-4 是历史诊断 artifact；专用 schedule/plot mode 已删除，不是当前 append-only evidence
  budget。
- historical roots 在归档中保留 lineage；不得把其中指标与当前 BBB gold-v4 matched cohort 混表。
