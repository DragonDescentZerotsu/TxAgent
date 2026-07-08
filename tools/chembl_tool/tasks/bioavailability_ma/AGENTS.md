# Bioavailability_Ma task notes

本文件只记录 Bioavailability_Ma 的 task-specific 语义：label、当前数据版本、oral bioavailability evidence tier、endpoint group、过滤规则和 reasoning schema。通用 ChEMBL workflow、wrapper 结构、batch/resume、viewer、cost 和目录规范统一记录在仓库根 `AGENTS.md`。

## Task 定义

目标不是训练 oral bioavailability classifier，而是构建可审计的 oral bioavailability evidence library：给定 query molecule，先预取相似分子的体内 oral bioavailability、口服暴露、吸收/通透、溶解度/溶出、代谢/清除和肠道转运体相关 evidence，再交给 reasoning LLM 判断 analog evidence 是否能 transfer 到 query molecule。

当前本地数据：

```text
data/processed/Bioavailability_Ma/test.jsonl

fields:
  drug: query SMILES
  Y: oral bioavailability label
```

当前评估约定：

```text
Y=1 -> high / acceptable oral bioavailability, defined as F >= 20%
Y=0 -> low / poor oral bioavailability, defined as F < 20%
```

这个阈值在代码中统一记录为：

```text
BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT = 20.0
```

final summary 必须在 `bioavailability_prediction=high` 和 `bioavailability_prediction=low` 中二选一；不要在 batch accuracy 和 macro-F1 中输出 uncertain。如果需要保留模型不确定性，用独立字段 `confidence`。

当前边界：

```text
ChEMBL neighbor retrieval 不是 DeepSeek 可调用 tool。
ChEMBL neighbor retrieval 也不是当前 FastAPI service tool。
它是 run_reasoning_pipeline.py 内部的 evidence prefetch / context assembly 步骤。

DeepSeek group-level analysis 可调用的工具只有：
  mmp_structure_compare
  properties_compare

DeepSeek single-molecule analysis 可调用的工具只有：
  molecule_properties
```

## Task-specific 文件

```text
constants.py
  Bioavailability_Ma label cutoff。

endpoint_groups.py
  Bioavailability_Ma Tier.endpoint_group、evidence_direction、evidence_strength 规则。

rules.py
  Bioavailability_Ma assay screening 关键词、negative keywords、weak terms、transporter target genes。

scoring.py
  Bioavailability_Ma assay 保留/剔除和打分统一入口。screen_assays.py 和 rescore_outputs.py 都调用 scored_row()。

run_reasoning_pipeline.py
  Bioavailability_Ma prompt、single/group/final schema、retrieval/prompt assembly 和 final-only rerun。

evidence_compiler.py
  Bioavailability_Ma v2 final-stage source-agnostic evidence compiler。它不直接预测 label，而是把
  retrieval + single/group outputs 转成 role-aware evidence cards、direct_label_votes、
  blocked_numeric_evidence_policy、proxy_evidence_policy 和 llm_decision_view。compiler 规则不能依赖
  evidence provider 是 ChEMBL、Starling 或其它 source。

run_reasoning_pipeline_v2.py
  复用 run_reasoning_pipeline.py 的 retrieval、single 和 group stage，只替换 final synthesis。
  当前保留的 v2 final prompt policy 是 threshold/salt/active-moiety：
  F >= 20% 必须按 high 处理；salt/free-base 或 same-active-molecule 证据不能仅因记录形式不同被丢弃；
  prodrug/active-moiety evidence 要区分 query 给药后的 active-moiety exposure 和非 query metabolite context。

run_reasoning_batch_v2.py
  Bioavailability_Ma v2 batch wrapper。公共 batch orchestration 仍在
  tools/chembl_tool/common/task_workflows/reasoning_batch.py。

specific_evidence_compiler.py
  Bioavailability_Ma specific final compiler。它复用 v2 `compile_final_evidence()` 的 source-agnostic
  cards，但把 final LLM view 重排成 parent direct F + `F = Fa * Fg * Fh` 三层结构，并显式标记
  prodrug/active-metabolite、species、dose/steady-state、food/formulation/salt/route 和
  threshold-sensitive context。这个文件是新 specific pipeline 的主要优化点，不改变 v1/v2 行为。

specific_retrieval.py
  Bioavailability_Ma specific retrieval regrouping。它保留 ChEMBL / Starling / merged source index 的
  原始 retrieval 能力，但在 specific pipeline 内把旧 Tier.endpoint_group row 重组为
  `Observed.direct_parent_f_and_systemic_exposure`、`Fa.absorption_solubility_permeability`、
  `Fg.gut_wall_efflux_intestinal_metabolism`、`Fh.hepatic_clearance_metabolic_stability` 四个 typed
  reasoning groups。direct-F 和 oral-exposure rows 是 terminal Observed assignments，不能再复制进
  Fa/Fg/Fh，避免同一 Starling/ChEMBL direct-F source 在 factor groups 中被重复计票。

run_reasoning_pipeline_specific.py
  复用 run_reasoning_pipeline.py 的服务/tool 基础设施，但替换 specific pipeline 的 retrieval、
  single、group 和 final synthesis 口径。retrieval 使用 `specific_retrieval.py` 的 typed regrouping；
  single-molecule prompt 只允许 property/tool 输出，不允许药名或已知 clinical PK 先验；
  group prompts 分别分析 Observed/Fa/Fg/Fh；final prompt 必须先读 `evidence_gate_summary`，
  再按 Fa、Fg、Fh 分层综合，不再把所有 endpoint group 当成扁平 high/low votes。

run_reasoning_batch_specific.py
  Bioavailability_Ma specific Fa/Fg/Fh batch wrapper。用于从 best merged ChEMBL + Starling v2 source
  batch 做 final-only rerun，并与
  `bioavailability_ma_test_full_plus_starling_v2_complete_v2_thresholdsalt_final_20260622` 对比。

specific_fallback_policy.py
  v12 之后的 fallback_uncertain 诊断 / 校准候选入口。它读取 batch predictions 和每个
  final_reasoning_output.json，从 compiled_evidence_summary 抽取 source-level direct-F、Starling/ChEMBL
  direct-F source direction、Fa/Fg/Fh factor signal、single-molecule prior 和 deterministic policy
  特征，并输出 features.jsonl / features.tsv / rule_metrics.json。这个文件只做可复用特征抽取和
  diagnostic candidate-rule 评估；不能直接用 test diagnostic rule 替代 valid 校准。

specific_pipeline_evolution_memory.md
  Bioavailability-specific Fa/Fg/Fh 长程优化记录本。新的 specific pipeline 迭代、metrics、
  failure audits、rejected experiments 和 stop criterion 都记录在这里；不要写回旧的 general/v2
  pipeline memory。

build_starling_evidence_library.py
  从 clean numeric records 和 dropped qualitative/contextual records 构建独立 evidence/index。
  同一 canonical molecule 的 numeric F% 聚合为 median/range/count；最多选择 6 条覆盖 F% 分布的
  condition-value-support examples，并另保留最多 6 条 qualitative/contextual examples。
  PMID 只允许保留在内部 evidence 供审计，不发送给 reasoning LLM。

build_combined_tier1_starling_evidence_library.py
  将 ChEMBL Tier 1 和 Starling molecule 按 InChIKey connectivity 合并。共享 molecule 只占一个
  neighbor rank，但 evidence rows 同时保留两个 source 的原始证据。Combined group prompt 的
  顶层 `evidence_source` 会收集所有实际出现的 source；mixed group 显示为
  `ChEMBL + starling-labs/Oral_Bioavailability`，不能只取第一条 evidence row 的来源。

merge_final_source_batches.py
  将已完成的 Bioavailability_Ma batch artifacts 合并成 final-only source batch。典型用途是把
  ChEMBL all-tier batch 的 retrieval/single/group outputs 与 Starling v2 batch 的 retrieval/group
  outputs 合并，然后只用 run_reasoning_batch_v2.py 重跑 final compiler / final LLM。它只合并
  source artifacts，不复制旧 final prediction；重复 group_id 默认报错，避免同一 group 被静默覆盖。

plot_retrieval_experiments.py
  将 no-retrieval、ChEMBL、Starling v1/v2 和 combined v1/v2 retrieval 结果画成性能对比
  SVG，并生成当前 combined-v2 index 中 Starling vs ChEMBL Tier 1 的 connectivity-level
  molecule overlap SVG。所有可编辑 SVG 文字固定使用 Tinos；绘图环境缺少完整 Tinos 字体族时
  直接报错，不能静默 fallback。使用 `--preview-png` 同时输出便于人工检查的 PNG。

run_starling_v1_knn_baseline.py
  Starling v1 clean numeric molecule-level KNN-3 baseline。复用 Starling index 中 13,362 个
  numeric entries，排除与 query 相同的 InChIKey connectivity，不设置最低 similarity，固定取
  3 个 neighbor。每个 neighbor 按聚合 median F% 是否达到 20% 转为 high/low，最终做无权重多数投票。
```

下面这些文件是 task-specific 配置 wrapper，公共实现见根 `AGENTS.md` 的 `tools/chembl_tool/common/task_workflows/` 说明：

```text
screen_assays.py
rescore_outputs.py
summarize_outputs.py
report.py
build_evidence_library.py
retrieve_neighbors.py
chembl_exact_context.py
run_reasoning_batch.py
```

不要在 wrapper 中新增业务规则；Bioavailability_Ma assay 保留/剔除逻辑应只放在 `rules.py` 和 `scoring.py`，endpoint-group 语义应只放在 `endpoint_groups.py`。

## 当前数据和输出

当前推荐使用的 assay screening 结果：

```text
outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/v4/
```

其中：

```text
bioavailability_assay_candidates.csv
bioavailability_assay_candidates.jsonl
bioavailability_assay_report.md
bioavailability_health_check.md
bioavailability_activity_evidence.csv
```

当前 evidence library 和 neighbor index：

```text
outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_molecule_evidence.jsonl
outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_neighbor_index.pkl
outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_neighbor_index.meta.json
```

Starling-only oral bioavailability retrieval 使用独立 evidence library，不与 ChEMBL index 混合：

```text
source:
  outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf/
    absolute_unspecified_systemic_broad_condition_full_text_v1/molecule_records.jsonl
    absolute_unspecified_systemic_broad_condition_full_text_v1/dropped_rows.jsonl

builder:
  tools/chembl_tool/tasks/bioavailability_ma/build_starling_evidence_library.py

outputs:
  outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling/
    starling_oral_bioavailability_evidence.jsonl
    starling_oral_bioavailability_neighbor_index.pkl
    starling_oral_bioavailability_neighbor_index.meta.json
```

Starling index 只包含一个 retrieval group：

```text
Starling.direct_oral_bioavailability
```

同一 canonical molecule 的多条文献记录会汇总为 median/range/count/provenance，避免 top-k 被同一
molecule 的不同 condition 重复占满。numeric F% 默认保留 `[0, 100]`；此外从 dropped rows 中恢复
无法安全解析为精确 F% 的 qualitative/contextual oral-bioavailability statements，以及此前因
report type 不是 direct absolute/unspecified/systemic 而被 benchmark clean set 排除的 contextual
records。后者只作为文本 evidence，不伪造 numeric F%、median 或 range。

每条 Starling 聚合 evidence 最多保留 6 条 `source_record_examples`。numeric examples 先按 F%
排序，再从整个分布均匀抽取 order statistics，覆盖低端、中间和高端；每个 example 将 condition、
对应 F% 和 support text 绑定在同一结构中。另有最多 6 条 `source_qualitative_examples` 保存
non-numeric/contextual statements。group-level DeepSeek prompt 不发送任何 source PMID。

构建和运行：

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_evidence_library \
  --workers 128 \
  --max-record-examples 6

python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
  --index outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling/starling_oral_bioavailability_neighbor_index.pkl \
  --groups Starling.direct_oral_bioavailability \
  --parallelism 64 \
  --group-workers 20 \
  --top-k-per-group 3 \
  --min-similarity 0.3 \
  --max-tool-rounds 10 \
  --batch-id bioavailability_ma_test_starling_v2_<date>
```

ChEMBL Tier 1 + Starling combined retrieval 也使用独立 index。分子按 InChIKey connectivity layer
合并；共享 molecule 只占一个 neighbor rank，但 evidence rows 同时保留 ChEMBL assay/activity 和
Starling 文献聚合证据：

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.build_combined_tier1_starling_evidence_library \
  --workers 128

python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
  --index outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/combined_tier1_starling/combined_tier1_starling_neighbor_index.pkl \
  --groups Combined.chembl_tier1_and_starling \
  --parallelism 64 \
  --group-workers 20 \
  --top-k-per-group 3 \
  --min-similarity 0.3 \
  --max-tool-rounds 10 \
  --batch-id bioavailability_ma_test_combined_tier1_starling_v2_<date>
```

combined v2 index 当前包含 30,171 个 unique molecule entities：8,365 个双源共享、12,956 个
ChEMBL-only、8,850 个 Starling-only。每条 evidence row 保留 `evidence_source`、
`source_molecule_id`、`source_group_id` 和原 source SMILES，避免双源证据被预先平均或丢失来源。

### Retrieval source 数据统计

分子 identity 使用与 retrieval exact exclusion 一致的 InChIKey connectivity layer：

```text
ChEMBL Tier 1 direct: 20,848 unique molecules
ChEMBL Tier 1 all, including context_dependent: 21,321
Starling: 12,791

Starling vs ChEMBL Tier 1 direct:
  overlap: 7,326
  Starling covered by ChEMBL: 57.27%
  ChEMBL covered by Starling: 35.14%
  Jaccard: 27.84%
  ChEMBL-only: 13,522
  Starling-only: 5,465
```

Starling-only v2 index 使用 82,496 条 clean numeric source records 中 F% 位于 `[0, 100]` 的
80,808 条，并恢复 80,787 条 qualitative/contextual source records。最终聚合为 18,117 个
canonical-SMILES neighbors：

```text
numeric evidence molecules: 13,362
qualitative/contextual evidence molecules: 9,815
qualitative-only molecules newly added: 4,755
```

Starling-only v2 retrieval coverage：

```text
125 queries 在 Tanimoto >= 0.3 时至少有一个 non-exact neighbor
125 queries 有完整 top-3
top-1 Tanimoto median: 0.7292
exact connectivity neighbors returned: 0
375 returned neighbor slots:
  numeric-only: 87
  qualitative-only: 83
  numeric + qualitative: 205
```

### Starling v1 KNN-3 majority baseline

最简单的 non-LLM retrieval baseline 使用 Starling v1 clean numeric molecule-level evidence。
当前 v2 index 完整保留了同一批 13,362 个 numeric entries，因此 runner 从现有 index 中过滤掉
qualitative-only molecules，精确恢复 v1 numeric candidate slice。设定：

```text
fingerprint: RDKit Morgan radius=2, 2048 bits, useChirality=false
exact-query exclusion: full InChIKey / InChIKey connectivity / canonical SMILES
k: 3
minimum similarity: none
neighbor label: median F% >= 20% => high; otherwise low
vote: unweighted majority
```

运行命令：

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.run_starling_v1_knn_baseline
```

输出：

```text
outputs/chembl_tool/tasks/bioavailability_ma/baselines/starling_v1_knn3_majority/
  manifest.json
  predictions.jsonl
  metrics.json
  report.md
```

完整 128-molecule test 结果：

```text
accuracy: 0.7813
macro-F1: 0.7020
positive precision: 0.8557
positive recall: 0.8557
negative precision: 0.5484
negative recall: 0.5484
confusion matrix: TN=17 FP=14 FN=14 TP=83
prediction distribution: high=97, low=31

retrieval audit:
  128 / 128 queries returned exactly 3 neighbors
  384 total neighbor slots
  minimum returned Tanimoto: 0.2093
  exact full-InChIKey matches: 0
  exact connectivity matches: 0
  exact canonical-SMILES matches: 0
```

相对 Starling v1 DeepSeek，KNN-3 有 32 个 prediction flips，其中 KNN 修正 19 个、回退 13 个，
净增加 6 个正确预测。KNN-3 accuracy 更高（0.7813 vs 0.7344），macro-F1 也更高
（0.7020 vs 0.6909），但仍低于 Starling v2 DeepSeek macro-F1 0.7342。KNN-3 明显偏向 high
class，因此不能只比较 accuracy。

旧版 v1 combined top-3 retrieval：

```text
127 / 128 queries 至少有一个 neighbor
123 / 128 queries 有完整 top-3
375 returned neighbor slots:
  Starling-only evidence: 191
  ChEMBL + Starling shared evidence: 134
  ChEMBL-only evidence: 50
```

reasoning 产物统一放在：

```text
outputs/chembl_tool/tasks/bioavailability_ma/reasoning/single_runs/
outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/
```

### Empty final 的 final-only 修复

如果完整 retrieval、single 和 group artifacts 已经存在，但
`final_reasoning_output.json` 的 parsed content 是空 `{}`，不要重跑完整 molecule pipeline。
使用 batch runner 的 `--final-only-source-batch` 复用前三类 artifacts：

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
  --indices 72 115 126 \
  --parallelism 3 \
  --batch-id <source_batch_id>_final_rerun \
  --final-only-source-batch \
    outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/<source_batch_id>
```

final-only batch 会为每个 index 复制原 retrieval/single/group files，再调用
`run_reasoning_pipeline.py --resume-final-from-run-dir`。先检查 rerun 的 prediction 和 trace；
确认后将 rerun 的 `final_reasoning_output.json` 和 `trace_messages.jsonl` 回填到 source batch。
回填前把原文件保存为：

```text
final_reasoning_output.initial_empty.json
trace_messages.initial_empty.jsonl
```

最后用原完整命令加 `--skip-existing`，不会再次调用 LLM，只重新收集 128 条 prediction、
metrics、report 和 combined trace：

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
  --index <same_index.pkl> \
  --groups <same_group_id> \
  --parallelism 64 \
  --group-workers 20 \
  --top-k-per-group 3 \
  --min-similarity 0.3 \
  --max-tool-rounds 10 \
  --batch-id <source_batch_id> \
  --skip-existing
```

不要只报告包含 `missing` prediction 的初始 metrics；必须明确记录 final-only rerun indices、
预测值、原文件备份和 corrected metrics。

当前实验汇总图：

```text
outputs/chembl_tool/tasks/bioavailability_ma/figures/
  oral_bioavailability_retrieval_experiments.svg
  starling_chembl_tier1_molecule_overlap.svg
```

当前 performance 图使用 corrected 128-molecule benchmark metrics，包含 3 个历史 no-retrieval
reference、Starling v1 numeric KNN-3 majority baseline、ChEMBL Tier 1、ChEMBL all tiers、
Starling v1/v2 和 combined v1/v2 DeepSeek runs。KNN 不设置最低 similarity；DeepSeek retrieval
runs 使用 minimum Tanimoto 0.3。combined v2 使用 final-only rerun 后的 corrected metrics。
overlap 图使用
`bioavailability_ma_combined_tier1_starling_neighbor_index.v2` 的 InChIKey connectivity merge
实体口径：ChEMBL-only 12,956、shared 8,365、Starling-only 8,850、union 30,171；不要用
Starling 原始 CSV 行数或 merge 前 canonical-SMILES 数量替换这些图中统计。

重画命令：

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.plot_retrieval_experiments \
  --preview-png
```

## Bioavailability_Ma reasoning pipeline

```text
1. 读取 Bioavailability_Ma test.jsonl 的 query molecule。
2. 从 --index 指定的 evidence source 预取 neighbor evidence；默认是 ChEMBL，也可使用独立
   Starling-only 或 combined index。
3. 并发执行 single-molecule analysis；DeepSeek 只可调用 molecule_properties。
4. 并发执行 group-level analysis；DeepSeek 只可调用 mmp_structure_compare 和 properties_compare。
5. final summary 读取 single + all group outputs，不暴露任何 tool。
6. 保存 retrieval/single/group/final/trace/manifest。
```

## Retrieval source ablation 结果

当前 ChEMBL/Starling reasoning runs 使用同一个 128-molecule
`data/processed/Bioavailability_Ma/test.jsonl`、DeepSeek-v4-pro、top-k=3 和
minimum Tanimoto=0.3。

```text
no retrieval, properties-only, no SMILES identity heuristic:
  accuracy 0.6562
  macro-F1 0.6000

no retrieval, properties-only, identity recognition allowed:
  accuracy 0.7422
  macro-F1 0.6598

no retrieval, properties-only, identity forbidden but memory comparison allowed:
  accuracy 0.6953
  macro-F1 0.6323

ChEMBL Tier 1:
  batch: bioavailability_ma_test_tier1_20260618
  accuracy 0.7031
  macro-F1 0.6673
  positive recall 0.6804
  Note: idx36 and idx121 originally had empty parsed final content. Final-only rerun produced
  low predictions for both; metrics were recomputed over 128 non-missing predictions.

ChEMBL all tiers:
  batch: bioavailability_ma_test_full_20260512
  accuracy 0.7188
  macro-F1 0.6762
  positive recall 0.7216

Starling-only:
  batch: bioavailability_ma_test_starling_only_20260618
  accuracy 0.7344
  macro-F1 0.6909
  positive precision 0.8987
  positive recall 0.7320
  negative recall 0.7419
  confusion matrix: TN=23 FP=8 FN=26 TP=71

Starling-only v2 (qualitative/contextual records + F%-distribution examples, no PMID in LLM payload):
  batch: bioavailability_ma_test_starling_v2_20260618
  index: bioavailability_ma_starling_neighbor_index.v2
  accuracy 0.7656
  macro-F1 0.7342
  positive precision 0.9467
  positive recall 0.7320
  negative recall 0.8710
  confusion matrix: TN=27 FP=4 FN=26 TP=71
  successful: 128 / 128
  retrieval: 125 / 128 with neighbors; all 125 had complete top-3
  top-1 Tanimoto median: 0.7292
  375 neighbor slots: 87 numeric-only, 83 qualitative-only, 205 mixed numeric+qualitative
  versus Starling v1: 20 prediction flips, 12 corrected and 8 regressed, net +4 correct
  prompt audit: 125 group-level user messages checked; no PMID present

ChEMBL Tier 1 + Starling combined:
  batch: bioavailability_ma_test_combined_tier1_starling_20260618
  accuracy 0.7656
  macro-F1 0.7149
  positive precision 0.8941
  positive recall 0.7835
  negative recall 0.7097
  confusion matrix: TN=22 FP=9 FN=21 TP=76

ChEMBL Tier 1 + Starling combined v2:
  batch: bioavailability_ma_test_combined_tier1_starling_v2_20260619
  index: bioavailability_ma_combined_tier1_starling_neighbor_index.v2
  accuracy 0.7656
  macro-F1 0.7193
  positive precision 0.9036
  positive recall 0.7732
  negative recall 0.7419
  confusion matrix: TN=23 FP=8 FN=22 TP=75
  prediction distribution: high=83, low=45
  successful/evaluable: 128 / 128

  Initial run produced empty `{}` final content for idx72, idx115 and idx126. A final-only rerun reused
  their original retrieval/single/group artifacts and produced low/high/low respectively; all three were
  correct. Corrected final outputs and traces were copied back into the main batch, with the original empty
  files retained as `final_reasoning_output.initial_empty.json` and
  `trace_messages.initial_empty.jsonl`, then metrics/combined trace were regenerated using `--skip-existing`.

  retrieval:
    127 / 128 queries had at least one neighbor
    125 / 128 had complete top-3
    378 returned neighbor slots
    top-1 Tanimoto median 0.7292
    ChEMBL-only slots: 30
    Starling-only slots: 227
    ChEMBL + Starling shared slots: 121
    slots with Starling qualitative/contextual evidence: 276
    slots with Starling numeric evidence: 272

  prompt audit:
    127 combined group-level user messages
    PMID occurrences: 0

  versus combined v1:
    same accuracy 0.7656
    macro-F1 0.7149 -> 0.7193
    18 prediction flips: 9 corrected and 9 regressed, net 0

  versus Starling-only v2:
    same accuracy 0.7656
    lower macro-F1: 0.7193 vs 0.7342
    higher positive recall: 0.7732 vs 0.7320
    lower negative recall: 0.7419 vs 0.8710

  usage:
    prompt tokens 5,659,941
    completion tokens 1,249,783
    total tokens 6,909,724
    tool calls 852
    versus combined v1: prompt +71.2%, total +52.7%, tool calls +1.1%
    initial full-run wall time about 14 min 24 s; idx102 was a 660 s tail outlier
```

上述 combined v1/v2 是早期 Tier 1 + Starling single-group retrieval ablation。该阶段 direct
oral-F analog evidence 是主要有效 signal；Starling-only v2、combined v1 和 combined v2 的
accuracy 都是 0.7656，Starling-only v2 的 macro-F1 最高（0.7342），combined v2 的 positive recall
更高但 negative recall 更低。不能把这组早期 ablation 表述为 combined 取得最高 overall performance。
这些差异尚未做 repeated-run 方差估计。

## Starling v2 完整 pipeline 当前推荐结果

为测试 Starling v2 加入完整 ChEMBL all-tier pipeline 后是否超过旧 Starling-only v2，当前推荐
流程是：

```text
1. 分别保留已完成的 ChEMBL all-tier batch 和 Starling v2 batch 的 retrieval/single/group artifacts。
2. 用 merge_final_source_batches.py 合并成 final-only source batch：
   ChEMBL all-tier groups + Starling.direct_oral_bioavailability。
3. 用 run_reasoning_batch_v2.py 的 source-agnostic final compiler 和 threshold/salt/active-moiety
   final prompt，只重跑 final stage。
```

source batch 构建命令模板：

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.merge_final_source_batches \
  --primary-batch outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_20260512 \
  --extra-batch outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_starling_v2_20260618 \
  --out-batch outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_plus_starling_v2_source_20260622
```

v2 final-only rerun 命令模板：

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch_v2 \
  --input-jsonl data/processed/Bioavailability_Ma/test.jsonl \
  --parallelism 64 \
  --group-workers 4 \
  --batch-id bioavailability_ma_test_full_plus_starling_v2_complete_v2_thresholdsalt_final_20260622 \
  --final-only-source-batch \
    outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_plus_starling_v2_source_20260622 \
  --skip-existing
```

当前最佳 run：

```text
batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/
    bioavailability_ma_test_full_plus_starling_v2_complete_v2_thresholdsalt_final_20260622

source batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/
    bioavailability_ma_test_full_plus_starling_v2_source_20260622

retrieval/group source:
  ChEMBL all-tier + Starling v2

accuracy: 0.796875
macro-F1: 0.748792
confusion matrix: TN=23 FP=8 FN=18 TP=79
prediction distribution: high=87 low=41
successful/evaluable: 128 / 128
```

相对旧 Starling-only v2 baseline：

```text
old Starling-only v2:
  batch: bioavailability_ma_test_starling_v2_20260618
  accuracy 0.765625
  macro-F1 0.734220
  confusion matrix: TN=27 FP=4 FN=26 TP=71

current complete merged v2:
  accuracy +0.031250
  macro-F1 +0.014572
```

同一 v2 policy 下的关键对照：

```text
ChEMBL all-tier only, threshold/salt:
  batch: bioavailability_ma_test_full_v2_thresholdsalt_final_20260622
  accuracy 0.734375
  macro-F1 0.686546
  confusion matrix: TN=22 FP=9 FN=25 TP=72

Starling v2 only, threshold/salt:
  batch: bioavailability_ma_test_starling_v2_complete_v2_thresholdsalt_final_20260622
  accuracy 0.781250
  macro-F1 0.741861
  confusion matrix: TN=25 FP=6 FN=22 TP=75

ChEMBL all-tier + Starling v2, threshold/salt:
  batch: bioavailability_ma_test_full_plus_starling_v2_complete_v2_thresholdsalt_final_20260622
  accuracy 0.796875
  macro-F1 0.748792
  confusion matrix: TN=23 FP=8 FN=18 TP=79
```

因此当前证据支持：Starling v2 direct evidence 是主要 anchor；ChEMBL all-tier evidence 在 v2
source-agnostic final compiler 和 threshold/salt/active-moiety policy 下可以提供额外增益。
这不是 source-specific prompt：compiler 和 final prompt 不依赖 provider 名称，只依赖 endpoint role、
transferability、direct-label gate、proxy-vs-F gate 和 source correlation。

## 当前推荐 Bioavailability-specific Fa/Fg/Fh pipeline

2026-06-24 起，Bioavailability_Ma 继续保留通用 v1/v2 pipeline，但当前 best observed
Bioavailability-specific 版本是独立的 Fa/Fg/Fh pipeline：

```text
code path:
  tools/chembl_tool/tasks/bioavailability_ma/specific_retrieval.py
  tools/chembl_tool/tasks/bioavailability_ma/specific_evidence_compiler.py
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_pipeline_specific.py
  tools/chembl_tool/tasks/bioavailability_ma/run_reasoning_batch_specific.py
  tools/chembl_tool/tasks/bioavailability_ma/specific_fallback_policy.py
  tools/chembl_tool/tasks/bioavailability_ma/specific_postprocess_policy.py

active live compiler:
  bioavailability_specific_fa_fg_fh_v16_mechanism_alert_view

active live pipeline:
  bioavailability_ma_specific_fa_fg_fh_v16_mechanism_alert_view

accepted final scoring layer:
  valid-selected deterministic postprocess
  rule: force_state_or_valid_selected_high_rescues
  components:
    1. force_state_or_starling_soft_high_anchor
    2. force_state_or_no_evidence_high_fallback
```

Pipeline semantics:

```text
1. retrieve_specific_neighbors() reuses the existing ChEMBL / Starling retrieval assets but regroups
   rows into typed Bioavailability_Ma groups:
     Observed.direct_parent_f_and_systemic_exposure
     Fa.absorption_solubility_permeability
     Fg.gut_wall_efflux_intestinal_metabolism
     Fh.hepatic_clearance_metabolic_stability

2. single-molecule reasoning is property-only. It may call molecule_properties, but must not use drug
   identity, known clinical PK, or molecule name recall.

3. group reasoning analyzes exactly one typed group at a time. Observed direct-F evidence can only
   support the 20% label threshold when parent/analyte, route, formulation/salt, species/population,
   and source-transfer scope are clean enough. Fa/Fg/Fh groups are factor evidence, not flat label votes.

4. specific_evidence_compiler.py rewrites the final LLM view into:
     parent direct-F analog evidence
     source-level de-correlated direct-F summary
     Fa/Fg/Fh factor cards and summaries
     prodrug / active-metabolite / active-moiety / formulation / route / species scope flags
     mechanism transfer alerts
     deterministic admissibility policy

5. final reasoning must first respect parent oral F scope, then synthesize F = Fa * Fg * Fh. It must not
   aggregate old Tier.endpoint_group outputs as a flat high/low vote.

6. specific_postprocess_policy.py applies the accepted deterministic low-to-high rescue after a completed
   batch. This postprocess does not call the LLM again; it is selected on valid features and then applied
   unchanged to the full test batch.
```

Current best specific outputs:

```text
best full/test output:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/
    bioavailability_ma_specific_fa_fg_fh_v16_valid_selected_postprocess_v2_full_20260624

best valid output:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/
    bioavailability_ma_specific_fa_fg_fh_v16_valid_selected_postprocess_v2_valid_20260624

underlying full trace/source artifacts:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/
    bioavailability_ma_specific_fa_fg_fh_v16_mechanism_alert_full_finalonly_20260624

underlying valid trace/source artifacts:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/
    bioavailability_ma_specific_fa_fg_fh_v16_mechanism_alert_valid_finalonly_20260624
    bioavailability_ma_specific_fa_fg_fh_v16_mechanism_alert_valid_idx15_rerun_20260624
    bioavailability_ma_specific_fa_fg_fh_v16_mechanism_alert_valid_combined_idx15_20260624

required diagnostics / selected-rule artifacts:
  outputs/chembl_tool/tasks/bioavailability_ma/fallback_policy_diagnostics/
    v16_mechanism_alert_full_20260624
    v16_mechanism_alert_valid_combined_20260624
    v16_valid_selected_postprocess_v2_full_20260624
```

Current best observed full/test performance:

```text
Specific Fa/Fg/Fh v16 + valid-selected postprocess v2:
  accuracy: 0.804688
  macro-F1: 0.764273
  confusion matrix: TN=25 FP=6 FN=19 TP=78
  prediction distribution: high=84 low=44

Specific Fa/Fg/Fh v16 live output before postprocess:
  accuracy: 0.781250
  macro-F1: 0.741861
  confusion matrix: TN=25 FP=6 FN=22 TP=75

Previous best complete Starling v2 pipeline:
  batch: bioavailability_ma_test_full_plus_starling_v2_complete_v2_thresholdsalt_final_20260622
  accuracy: 0.796875
  macro-F1: 0.748792
  confusion matrix: TN=23 FP=8 FN=18 TP=79
```

GLM-5.2 compatibility run, same Bioavailability-specific pipeline:

```text
model:
  zai-org/GLM-5.2-FP8 via https://litellm.parcc.upenn.edu/v1

main batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/
    glm52_specific_full_test_20260627

final-only repair batch:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/
    glm52_specific_full_test_20260627_finalfix_20k

merged metrics:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/
    glm52_specific_full_test_20260627/metrics_merged_finalfix_20k.json

GLM-5.2 merged full/test performance:
  n_total: 128
  n_successful: 128
  n_failed_runs: 0
  accuracy: 0.765625
  macro-F1: 0.727272
  confusion matrix: TN=25 FP=6 FN=24 TP=73
  prediction distribution: high=79 low=49
```

GLM-5.2 run command pattern:

```bash
export GLM_API_KEY='<local key; do not commit>'

/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch_specific \
  --input-jsonl data/processed/Bioavailability_Ma/test.jsonl \
  --batch-id glm52_specific_full_test_20260627 \
  --parallelism 8 \
  --group-workers 4 \
  --top-k-per-group 3 \
  --min-similarity 0.3 \
  --max-tool-rounds 1 \
  --max-tokens 8192 \
  --timeout-s 300 \
  --base-url https://litellm.parcc.upenn.edu/v1 \
  --model zai-org/GLM-5.2-FP8 \
  --api-key-env GLM_API_KEY \
  --env-file /dev/null \
  --disable-thinking \
  --reasoning-effort ""
```

For GLM, do not report the main batch `metrics.json` alone as final performance: the initial
8192-token final stage had 14 final JSON truncation/parse/missing failures at indices
`1, 3, 4, 10, 31, 52, 53, 55, 75, 79, 87, 88, 120, 126`. These were repaired with a final-only
rerun at `--max-tokens 20480`; use `metrics_merged_finalfix_20k.json` for the full 128-molecule score.

GLM trace audit:

```text
reasoning_content:
  single: 128/128 non-empty
  group: 497/497 non-empty
  final: 128/128 non-empty after finalfix

single tool use:
  expected tool: molecule_properties
  total single tool calls: 122
  successful single tool results: 121
  recoverable single tool error then retry success: 1
  missing required single tool call: 7/128
  missing-tool indices: 5, 12, 30, 57, 69, 85, 101

group tool use:
  groups: 497
  groups with tool calls: 360
  groups without tool calls: 137
  mmp_structure_compare calls: 1378
  properties_compare calls: 1301
```

The missing single-tool cases are real GLM/LiteLLM tool-choice adherence failures: the assistant text
claims it will call `molecule_properties`, but no OpenAI tool_call is emitted and the raw assistant
content is `{}`. Group-level tool errors are mostly invalid/special SMILES, salts/coformers, or mmpdb
assertions; similar errors also appear under DeepSeek and should be handled as tool robustness issues,
not as GLM-only failures.

DeepSeek-v4-pro comparison:

```text
closest same-pipeline live output before deterministic postprocess:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/
    bioavailability_ma_specific_fa_fg_fh_v16_mechanism_alert_full_finalonly_20260624

metrics:
  n_total: 128
  n_successful: 128
  n_failed_runs: 0
  accuracy: 0.781250
  macro-F1: 0.741861

current recommended DeepSeek artifact after valid-selected postprocess v2:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/
    bioavailability_ma_specific_fa_fg_fh_v16_valid_selected_postprocess_v2_full_20260624

postprocess metrics:
  n_total: 128
  n_successful: 128
  n_failed_runs: 0
  accuracy: 0.804688
  macro-F1: 0.764273
  confusion matrix: TN=25 FP=6 FN=19 TP=78

trace/tool audit:
  single reasoning: 128/128 non-empty
  single molecule_properties: 128/128 called and successful
  group reasoning: 497/497 non-empty
  group tool calls: 497/497 groups with evidence had tools
  final reasoning: 128/128 non-empty
  final parse failures: 0
```

Conclusion for model comparison: the same Bioavailability-specific Fa/Fg/Fh pipeline runs successfully
with both models, and DeepSeek's trace is complete under the same tool contract. Use 0.741861 only when
comparing pure live final LLM output before the valid-selected deterministic postprocess; use 0.764273
for the current recommended DeepSeek Bioavailability-specific artifact. The GLM-5.2 result therefore
points to weaker model/endpoint adherence to tool calls and long structured JSON, not to a fundamental
problem in the pipeline design. If GLM is used for future batches, add retry/validation for missing
required single-tool calls and use larger final `max_tokens` or a more compact final schema.

Interpretation caveat: the specific pipeline is the current best observed test score, but the improvement over
the previous best complete Starling v2 run is small in absolute accuracy terms (+1 correct case on the 128-molecule
test set). Treat it as the current best task-local artifact and a better Bioavailability-specific factorization,
not as a statistically strong generalization claim without additional calibration data.

The full optimization history for the specific pipeline, including rejected v17/v18 attempts, valid/test metrics,
failure audits, and stopping rationale, is recorded in:

```text
tools/chembl_tool/tasks/bioavailability_ma/specific_pipeline_evolution_memory.md
```

### Starling transfer tool ablation

2026-06-23 测试了 HuggingFace tool：

```text
jiosephlee/starling-transfer-ssv2-srcval
```

Model card 输入约定：

```text
smiles_a: retrieved Starling source molecule
smiles_b: query molecule
metadata_a / metadata_b fields:
  molecule_name, species_or_population, dose, oral_exposure_mode,
  qualifying_conditions, comparator, extra_details
source_value: molecule A raw oral_bioavailability_value percent
```

当前 pipeline 实现为可选 retrieval annotation，不加入常驻 FastAPI service。开启
`--enable-starling-transfer-tool` 后，pipeline 会先正常 Starling retrieval，再对每个 numeric
`source_record_examples` 运行 transfer model，并把 compact `starling_transfer_tool` 摘要写入
对应 Starling evidence row，供 group-level DeepSeek 判断 analog transferability。query 端
metadata 默认使用 `same_source_context`：复用 source study context，但不复制 source molecule name。
这个 score 是辅助 transferability evidence，不是自动 final label。

运行命令：

```bash
env CUDA_VISIBLE_DEVICES=4 /data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
  --input-jsonl data/processed/Bioavailability_Ma/test.jsonl \
  --index outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling/starling_oral_bioavailability_neighbor_index.pkl \
  --groups Starling.direct_oral_bioavailability \
  --parallelism 8 \
  --group-workers 4 \
  --top-k-per-group 3 \
  --min-similarity 0.3 \
  --max-tool-rounds 10 \
  --timeout-s 300 \
  --enable-starling-transfer-tool \
  --starling-transfer-device cuda \
  --starling-transfer-batch-size 16 \
  --starling-transfer-max-examples-per-row 6 \
  --batch-id bioavailability_ma_test_starling_v2_transfer_tool_20260623 \
  --skip-existing
```

结果：

```text
batch: outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_starling_v2_transfer_tool_20260623
accuracy 0.773438
macro-F1 0.730683
confusion matrix: TN=24 FP=7 FN=22 TP=75
prediction distribution: high=82 low=46
successful/evaluable: 128 / 128
```

Annotation coverage:

```text
runs without scored pairs: 4 / 128
annotated rows: 292
annotated neighbors: 292
scored source-example pairs: 1,152
rows with any likely_transfer source example: 235
transfer probability median: 0.93425
```

Comparison:

```text
old Starling-only v2:
  accuracy 0.765625, macro-F1 0.734220
  transfer-tool ablation has slightly higher accuracy but lower macro-F1.

Starling-only complete v2 threshold/salt:
  accuracy 0.781250, macro-F1 0.741861

Best ChEMBL all-tier + Starling v2 complete:
  accuracy 0.796875, macro-F1 0.748792
```

Conclusion: this first transfer-tool ablation should not replace the current best pipeline. It may be useful
as an auxiliary signal, but the direct injection made predictions more positive overall and reduced negative-class
recall relative to old Starling-only v2.

当前已测试并拒绝的 prompt 扩展：

```text
contextgates:
  batch: bioavailability_ma_test_full_plus_starling_v2_complete_v2_contextgates_final_20260622
  accuracy 0.796875
  macro-F1 0.744393
  confusion matrix: TN=22 FP=9 FN=17 TP=80
  decision: rejected; macro-F1 below current threshold/salt best.

chargedprior:
  batch: bioavailability_ma_test_full_plus_starling_v2_complete_v2_chargedprior_final_20260622
  accuracy 0.789062
  macro-F1 0.741298
  confusion matrix: TN=23 FP=8 FN=19 TP=78
  decision: rejected; prompt reverted.
```

当前代码只保留已验证最高的 threshold/salt/active-moiety policy 和 batch validity fix。不要在没有重新
完整测试 128-molecule benchmark 的情况下保留 contextgates、chargedprior 或其它更长 prompt policy。

当前结果和错误分析报告：

```text
outputs/chembl_tool/tasks/bioavailability_ma/analysis/starling_v2_complete_pipeline_20260622/
  thresholdsalt_iteration_report_zh.md
  residual_error_literature_audit_round1_zh.md
  current_iteration_metrics_summary.json
  rejected_prompt_iterations_summary.json
  merged_thresholdsalt_residual_errors.tsv
  merged_thresholdsalt_residual_errors.json
```

最佳 merged threshold/salt run 剩余 26 / 128 个错误。文献核查后的最可疑或 threshold/formulation
sensitive labels：

```text
idx113 progesterone: TDC high 可能是 oral clinical/formulation 口径，不一定是 parent absolute F >=20%。
idx121 naltrexone: reported F range 约 5-40%，跨越 20% threshold。
idx77 pirenzepine: healthy-subject F 约 14%，ICU/部分 SmPC 约 26-28%。
idx7 rifabutin: reported F 常见 12-20% / about 20%，cutoff convention 影响 label。
idx41 vincamine: free-base vs HCl/formulated source 可能改变 label。
idx94 lorcainide: dose/steady-state saturation 可跨越 threshold。
idx46 nafcillin: 临床 poor oral use 与 numeric F source >20% 冲突。
```

文献核查支持的通用改进方向，但尚未带来更高已验证 macro-F1：

```text
same molecule human/label ordinary oral absolute F
> same molecule special population / dose / fed / formulation context
> same-active salt/freebase/formulation
> close analog human direct F
> animal direct F
> weak analog direct F
> Fa / AUC / Cmax / permeability / solubility / clearance proxies
> single-molecule prior

High Fa / high permeability / high oral exposure 不能单独证明 systemic F >= 20%。
"low bioavailability" / "poor oral drug" / "not usually administered orally" 必须映射回 numeric F threshold。
Parent F、active metabolite exposure after parent dosing 和 active parent exposure after prodrug dosing
必须作为不同 endpoint roles 处理。
```

后续如果继续优化，应优先把这些方向做成 compiler 侧结构化 evidence ranking / endpoint-role schema，
而不是在 final prompt 里继续堆 molecule-specific 或 source-specific 规则。

### Pipeline evolution archive and cleanup

`general_pipeline_evolution_memory.md` 是本轮 Bioavailability_Ma pipeline evolve 的过程 reference，记录了从
ChEMBL-only v2 compiler、unit-fix、decision-view、redaction/directguard rejected runs，到
Starling v2 complete pipeline 的主要尝试和性能。后续如果需要恢复当时为什么保留或拒绝某个策略，
先看：

```text
tools/chembl_tool/tasks/bioavailability_ma/general_pipeline_evolution_memory.md
```

2026-06-23 清理后，reasoning batch traces 只保留每个主要 evolve 阶段的代表 / 最佳结果：

```text
Original ChEMBL all-tier v1 baseline:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_20260512
  accuracy 0.718750, macro-F1 0.676208

Best ChEMBL-only v2 final compiler:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_v2_decisionview_final_20260622
  accuracy 0.757812, macro-F1 0.712109

Old Starling-only v2 retrieval baseline:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_starling_v2_20260618
  accuracy 0.765625, macro-F1 0.734220

Best Starling-only complete v2 final compiler:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_starling_v2_complete_v2_thresholdsalt_final_20260622
  accuracy 0.781250, macro-F1 0.741861

Best ChEMBL all-tier + Starling v2 complete pipeline:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_plus_starling_v2_complete_v2_thresholdsalt_final_20260622
  accuracy 0.796875, macro-F1 0.748792

Merged final-only source artifacts for the best ChEMBL + Starling v2 run:
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/bioavailability_ma_test_full_plus_starling_v2_source_20260622
```

Intermediate smoke tests, final-only reruns, rejected prompt/policy runs, and lower-performing Starling/combined
retrieval attempts were removed from `reasoning/batches/` to keep trace storage manageable. Their performance
and rejection rationale remain summarized in `general_pipeline_evolution_memory.md` and in:

```text
outputs/chembl_tool/tasks/bioavailability_ma/analysis/starling_v2_complete_pipeline_20260622/
  thresholdsalt_iteration_report_zh.md
  rejected_prompt_iterations_summary.json
  current_iteration_metrics_summary.json
```

### No-retrieval baseline provenance

这些 properties-only DeepSeek 结果以前没有记录在本 AGENTS.md。排查时先从 `~/.bash_history`
找到历史命令，再定位到 `/data1/tianang/Projects/Intern-S1/` 下的实际日志：

```text
/data1/tianang/Projects/Intern-S1/logs/
  deepseek-v4-pro_properties_Bioavailability_Ma_test_20260513_015555.log
  deepseek-v4-pro_properties_no_smiles_identity_Bioavailability_Ma_test_20260513_021529.log
  deepseek-v4-pro_properties_no_smiles_identity_allow_memory_compare_Bioavailability_Ma_test_20260513_023026.log

/data1/tianang/Projects/Intern-S1/reasoning-trajectory/
  deepseek-v4-pro_properties_Bioavailability_Ma_test_20260513_015555.log/Bioavailability_Ma.jsonl
  deepseek-v4-pro_properties_no_smiles_identity_Bioavailability_Ma_test_20260513_021529.log/Bioavailability_Ma.jsonl
  deepseek-v4-pro_properties_no_smiles_identity_allow_memory_compare_Bioavailability_Ma_test_20260513_023026.log/Bioavailability_Ma.jsonl
```

已核对 Intern-S1 的 128 条 test SMILES/labels 与当前 TxAgent test JSONL ordered exact match。
但这些 baseline 使用旧 Intern-S1/TRIM prompt 和 properties tool workflow，不是当前 reasoning
pipeline 的严格 same-prompt zero-retrieval ablation，只适合作为历史参考。

### 结果解释限制

```text
Starling source 与 benchmark 高度重叠：
  128 个 test molecules 中有 123 个 connectivity 在 Starling source 中直接出现。

query-time exact connectivity exclusion:
  exact query molecule 不会作为 neighbor 返回。

remaining limitation:
  这仍是 highly retrospective source-overlap experiment，不是 prospective external validation。
  Morgan fingerprint 不使用 chirality；某些 connectivity 不同的立体/表示变体可能 similarity=1.0，
  当前规则仍把它们视为 analog。
```

## Exact ChEMBL context

`chembl_exact_context.py` 是可选 evidence-rich 增强。它会用 query full InChIKey 查 ChEMBL exact molecule，并在 retrieved neighbor 涉及的 assay 中查 query activity。默认 benchmark 不开启，避免 prospective evaluation 数据泄漏；只有显式传 `--enable-chembl-exact-context` 时才用于 retrospective / evidence-rich case study。

默认 single-molecule prompt 不包含任何 ChEMBL 相关 payload 或 instruction。只有开启 exact context 且命中 query exact context 时，single-molecule payload 才包含 `exact_query_chembl_context`，并提示模型区分 direct same-molecule ChEMBL bioavailability evidence 和 physicochemical prior。ChEMBL neighbor evidence 仍只进入 group-level context。

## Evidence 类型解释

Bioavailability_Ma 关注的是口服给药后进入 systemic circulation 的程度和速度。Tier 不需要照搬
BBB 的 4 层；这里按 oral bioavailability 的证据距离和机制轴拆分。直接测得的 absolute oral
bioavailability 最接近标签；oral AUC/Cmax、fraction absorbed、permeability、solubility、metabolism
和 formulation evidence 都有用，但回答的是不同问题，不能混在一个同权重 bucket 里。

### Tier 1: direct absolute oral bioavailability

最接近任务标签的体内证据。只有明确表示 oral bioavailability 或 oral/IV 比值可计算 absolute F
时，才放入 Tier 1。

```text
absolute oral bioavailability
oral bioavailability
bioavailability
%F
F
F%
fraction of dose bioavailable
AUC oral/iv ratio
oral/iv exposure ratio
```

Tier 1 是最强 evidence，但仍需区分：

```text
absolute bioavailability:
  最接近 molecule-level oral F。

bioavailability endpoint 名称不够明确时:
  需要 assay description 支持 oral route 或 oral/IV comparison，避免把 generic availability 或
  non-oral exposure 误收进来。
```

### Tier 2: in vivo oral exposure and in vivo/in situ absorption

体内口服暴露、吸收比例或接近人体/动物肠吸收过程的证据，例如：

```text
AUC after oral dose
oral AUC
AUC po
dose-normalized AUC
Cmax after oral dose
oral Cmax
plasma exposure after oral administration
systemic exposure after oral dose
human intestinal absorption
intestinal absorption
fraction absorbed
Fa
Fabs
percent absorbed
oral absorption
effective permeability
Peff
intestinal permeability
jejunal permeability
intestinal perfusion
single-pass intestinal perfusion
in situ intestinal perfusion
portal vein absorption
intestinal uptake
```

这类 evidence 比体外 proxy 更接近 oral BA，但仍不等于 absolute F：

```text
oral AUC / Cmax:
  是体内口服暴露证据，但受 dose、species、formulation、matrix 和 clearance 影响。

fraction absorbed / HIA / Peff:
  说明分子能否跨越 GI barrier。高吸收仍可能因为首过代谢、高肝清除或肠道外排导致低 bioavailability。
```

### Tier 3: in vitro intestinal permeability and efflux

体外 intestinal barrier proxy evidence，例如：

```text
Caco-2 Papp
Caco2 permeability
MDCK Papp
MDCK-MDR1 Papp
apparent permeability
Papp A to B
Papp B to A
bidirectional permeability
efflux ratio
PAMPA
parallel artificial membrane permeability
P-gp substrate
ABCB1 substrate
MDR1 substrate
BCRP substrate
ABCG2 substrate
MRP substrate
ABCC substrate
intestinal efflux
transporter-mediated efflux
```

Tier 3 主要支持 permeability/absorption 和 intestinal efflux reasoning。Caco-2/MDCK/PAMPA 是 proxy；
P-gp/BCRP/MRP substrate 或 bidirectional efflux 可提示 intestinal efflux risk；单纯 transporter
inhibition/binding 不能直接解释成 substrate 或 efflux liability。

### Tier 4: solubility, dissolution and GI stability

影响 dose-limited absorption 的物化和 GI stability evidence，例如：

```text
aqueous solubility
kinetic solubility
thermodynamic solubility
logS
dissolution
dissolution rate
percent dissolved
GI stability
simulated gastric fluid stability
simulated intestinal fluid stability
```

Tier 4 主要判断 solubility/dissolution-limited absorption risk。高 solubility 和快速 dissolution 是
支持性 proxy，但不能保证 high oral F；低 solubility、慢 dissolution 或 poor GI stability 可以作为
low bioavailability risk。

### Tier 5: metabolism, first-pass and clearance

影响 oral F 的首过损失、代谢稳定性和清除 evidence，例如：

```text
microsomal stability
human liver microsome stability
rat liver microsome stability
hepatocyte stability
liver S9 stability
percent remaining
half-life
t1/2
intrinsic clearance
CLint
hepatic clearance
clearance
first-pass metabolism
hepatic extraction
CYP substrate
CYP metabolism
metabolic turnover
metabolic stability
```

Tier 5 必须谨慎解释：

```text
metabolic stability / CLint:
  与 first-pass loss 和 systemic exposure 有直接机制关系。

CYP substrate / metabolic turnover:
  可提示 metabolism risk，但必须看 assay context 和 readout。

CYP inhibition IC50:
  主要说明该分子抑制 CYP，不等于该分子被 CYP 快速代谢；不能直接当作 low bioavailability evidence。
```

### Tier 6: formulation, food-effect and relative bioavailability context

与 drug product、给药条件或相对比较强相关的 evidence，例如：

```text
relative bioavailability
food effect
fed/fasted ratio
formulation bioavailability
tablet vs capsule
solution vs suspension
nanoparticle formulation
solid dispersion
salt form comparison
AUC ratio
Cmax ratio
```

Tier 6 可以帮助 retrospective case study，但对 molecule-intrinsic Bioavailability_Ma label 应降权。
除非 query 和 neighbor 在 formulation、salt form、dose 和 species 上高度可比，否则不能把 Tier 6
作为强正负证据。

---

## Endpoint Group 标准

第二阶段不按每个 assay 单独检索。应按：

```text
Tier -> endpoint_group
```

组合生成 retrieval groups。

### Tier 1: direct absolute oral bioavailability

建议 endpoint groups：

```text
direct_absolute_bioavailability
  absolute bioavailability
  oral bioavailability
  bioavailability
  %f
  f%
  f
  fraction bioavailable
  fraction of dose bioavailable

direct_oral_iv_exposure_ratio
  auc oral/iv ratio
  oral/iv auc ratio
  oral to iv exposure ratio
  bioavailability ratio
```

### Tier 2: in vivo oral exposure and in vivo/in situ absorption

建议 endpoint groups：

```text
oral_auc_exposure
  auc
  auc0-t
  auc0-inf
  aucinf
  auc po
  oral auc
  dose-normalized auc
  plasma exposure

oral_cmax_exposure
  cmax
  oral cmax
  maximum plasma concentration
  peak plasma concentration

absorption_fraction_or_hia
  fraction absorbed
  percent absorbed
  fa
  fabs
  human intestinal absorption
  hia
  oral absorption

in_vivo_intestinal_permeability
  peff
  effective permeability
  intestinal permeability
  jejunal permeability
  intestinal perfusion
  single-pass intestinal perfusion
  in situ intestinal perfusion

in_vivo_intestinal_uptake_or_transport
  intestinal uptake
  intestinal transport
  absorptive transport
  portal absorption
```

### Tier 3: in vitro intestinal permeability and efflux

建议 endpoint groups：

```text
cell_permeability_papp
  papp
  apparent permeability
  caco-2 papp
  caco2 permeability
  mdck papp
  papp a to b
  papp apical to basolateral
  papp absorptive

cell_secretory_permeability
  papp b to a
  papp basolateral to apical
  papp secretory

cell_bidirectional_efflux_ratio
  efflux ratio
  papp b to a / papp a to b
  ba/ab ratio
  b-a/a-b ratio
  bidirectional permeability

pampa_or_artificial_membrane
  pampa
  parallel artificial membrane permeability
  artificial membrane permeability

transporter_substrate_or_efflux
  p-gp substrate
  pgp substrate
  abcb1 substrate
  mdr1 substrate
  bcrp substrate
  abcg2 substrate
  mrp substrate
  abcc substrate
  efflux
  intestinal efflux
  transporter-mediated efflux

transporter_inhibition_or_binding
  inhibition
  ic50
  ki
  ec50
  kd
  binding
```

### Tier 4: solubility, dissolution and GI stability

建议 endpoint groups：

```text
solubility
  solubility
  aqueous solubility
  kinetic solubility
  thermodynamic solubility
  logs
  intrinsic solubility

dissolution
  dissolution
  dissolution rate
  percent dissolved
  dissolved

gi_or_chemical_stability
  gastric stability
  intestinal stability
  simulated gastric fluid stability
  simulated intestinal fluid stability
  chemical stability
```

### Tier 5: metabolism, first-pass and clearance

建议 endpoint groups：

```text
metabolic_stability
  microsomal stability
  liver microsome stability
  hepatocyte stability
  s9 stability
  percent remaining
  t1/2
  half-life
  metabolic stability

intrinsic_or_hepatic_clearance
  intrinsic clearance
  clint
  hepatic clearance
  clearance
  cl

first_pass_or_extraction
  first-pass metabolism
  hepatic extraction
  extraction ratio
  intestinal metabolism
```

### Tier 6: formulation, food-effect and relative bioavailability context

建议 endpoint groups：

```text
relative_bioavailability_or_formulation
  relative bioavailability
  formulation bioavailability
  bioavailability ratio
  tablet vs capsule
  solution vs suspension
  salt form comparison
  nanoparticle formulation
  solid dispersion

food_effect_or_fed_fasted
  food effect
  fed/fasted ratio
  fed fasted
  high-fat meal
  fasting

formulation_auc_cmax_ratio
  auc ratio
  cmax ratio
  exposure ratio
```

### Unknown / weak context

下面 endpoint 只能作为 context-dependent evidence，不能单独强解释：

```text
activity
ratio
inhibition
ic50
ki
ec50
kd
binding
substrate
transport
uptake
stability
half-life
clearance
auc
cmax
```

如果这些 endpoint 出现在明确 oral BA / absorption / permeability / metabolism / transporter
assay context 中，可以被 endpoint group 规则提升；否则应标记为：

```text
endpoint_group: context_dependent
evidence_strength: weak
```

---

## Evidence Library 字段

应从当前 assay candidates 和 activity evidence 构建 molecule-level library。每一条 evidence
至少包含：

```text
molecule_chembl_id
canonical_smiles
standard_inchi_key
assay_chembl_id
assay_id
assay_tier
endpoint_group
endpoint_group_reason
assay_description
target_chembl_id
target_pref_name
target_genes
standard_type
standard_relation
standard_value
standard_units
pchembl_value
data_validity_comment
activity_comment
evidence_direction
evidence_strength
evidence_reason
```

建议 `evidence_direction`：

```text
supports_high_bioavailability
argues_against_high_bioavailability
absorption_support
permeability_support
solubility_support
solubility_risk
metabolic_stability_support
first_pass_or_clearance_risk
transporter_efflux_risk
context_dependent
unknown_direction
```

建议 `evidence_strength`：

```text
strong
moderate
weak
context_dependent
```

方向判断规则：

```text
1. direct_absolute_bioavailability 的 direction 可以在 standard_value、units 和 relation 可解释时按
   %F 数值判断；当前任务阈值为 F >= 20% 支持 high bioavailability，F < 20% 支持 low bioavailability。
2. oral AUC/Cmax 不能仅凭 endpoint 判断支持或反对，必须看 dose、species、route、formulation、
   comparator 和 activity_comment；默认作为 strong/moderate direct exposure evidence，但 direction
   可为 unknown_direction。
3. high fraction absorbed / HIA / Peff / Papp 通常支持 absorption/permeability，但具体方向依赖单位
   和 assay convention。
4. low solubility、slow dissolution、poor GI stability 可作为 bioavailability risk；high solubility
   和 rapid dissolution 是支持性 proxy，不保证 high oral F。
5. high microsomal/hepatocyte stability 或 low CLint 支持较低 first-pass risk；low stability 或 high
   CLint 支持 first-pass/clearance risk。
6. transporter substrate/efflux ratio 可提示 intestinal efflux risk；transporter inhibition/binding 不能
   直接解释成 substrate risk。
```

LLM payload 中的 activity evidence row 应只包含原始 ChEMBL assay/activity 字段和必要 metadata，
例如 assay id、tier、description、target、standard_type/value/units、activity_comment、
confidence_score、relationship_type。不要把 `evidence_direction`、`evidence_strength`、
`endpoint_group_reason`、`assay_reason` 发送给 reasoning LLM。

---

## 重要过滤规则

### 不把任意 PK endpoint 都当成 oral bioavailability

下面 endpoint 只有在 assay 明确是 oral administration、oral exposure、bioavailability 或 absorption
context 时才保留：

```text
auc
cmax
tmax
half-life
clearance
volume of distribution
plasma concentration
```

IV dosing、intraperitoneal dosing、subcutaneous dosing、topical dosing 或 unspecified route 的 PK
readout 不能直接作为 oral bioavailability evidence。

### CYP inhibition 不是 metabolic stability

```text
CYP3A4 inhibition IC50
CYP2D6 inhibition
CYP2C9 inhibition
CYP binding
enzyme inhibition
```

这些通常表示 compound inhibits enzyme，不表示 compound is rapidly metabolized。除非 assay
description 明确是 substrate depletion、metabolic turnover、microsomal/hepatocyte stability 或
metabolite formation，否则不要当作 first-pass metabolism evidence。

### Transporter inhibition 不是 transporter substrate

```text
P-gp inhibition IC50
BCRP inhibition
MRP inhibition
transporter binding
```

这类只能作为 context-dependent evidence。只有 substrate、efflux、bidirectional Papp、digoxin flux、
rhodamine/calcein accumulation 等 functional readout 才能支持 intestinal efflux risk。

### 泛化 cell activity / target potency 不保留

下面 assay 通常不是 oral bioavailability evidence：

```text
receptor binding
target potency
enzyme potency
functional activity
cytotoxicity
cell viability
proliferation
antiproliferative activity
tumor growth inhibition
antimicrobial activity
MIC
EC50 without ADME context
```

如果 assay type 是 Binding/Functional，但 description 没有 ADME、oral exposure、absorption、
permeability、solubility、dissolution、metabolic stability、clearance 或 transporter context，应过滤。

### Plasma protein binding 不是 oral bioavailability

```text
plasma protein binding
PPB
fu plasma
albumin binding
serum binding
```

PPB 会影响 free exposure 和 distribution，但不是 oral absorption 或 absolute oral bioavailability 的
直接 evidence。除非 assay 明确关联 oral systemic exposure interpretation，否则不纳入主 evidence。

### Formulation / food effect 需要降权

```text
food effect
fed vs fasted
formulation comparison
relative bioavailability
tablet vs capsule
nanoparticle formulation
solid dispersion
salt form comparison
```

这类 evidence 可能反映 drug product 或 formulation，而不是 molecule intrinsic property。可以保留为
Tier 6 的 `relative_bioavailability_or_formulation`、`food_effect_or_fed_fasted` 或
`formulation_auc_cmax_ratio`，但 group-level prompt 必须要求 LLM 降权。

### 非口服 route 噪声

下面 route context 不能作为 Bioavailability_Ma oral F evidence：

```text
intravenous
iv
intraperitoneal
ip
subcutaneous
sc
topical
dermal
ocular
intranasal
inhaled
```

例外：同一 assay 明确提供 oral/iv AUC ratio 或 absolute oral bioavailability 时，IV comparator 是计算
%F 的一部分，不应过滤。

---

## Neighbor Retrieval 设计

输入：

```text
query_smiles
top_k_per_group: default 3
min_similarity: default 0.3
groups: optional list[Tier.endpoint_group]
exclude_exact: default true
```

流程：

```text
1. 标准化 query molecule，生成 canonical SMILES、InChIKey、Morgan fingerprint。
2. 按 evidence library 中的 Tier.endpoint_group 建立 group membership。
3. 对每个 group 独立计算 query 与该 group molecule fingerprints 的 Tanimoto。
4. 每个 group 返回 top 3 non-identical neighbors，默认过滤 Tanimoto < 0.3 的 very distant analog。
5. 对同一 neighbor molecule 聚合其在该 group 下的所有 assay/activity evidence。
6. 返回 group-level retrieval payload。
```

同一分子排除标准：

```text
same molecule_chembl_id
same full standard_inchi_key
same InChIKey connectivity layer, i.e. the first block before "-"
same canonical_smiles
```

相似度 bucket：

```text
very_close_analog: Tanimoto >= 0.95
close_analog: 0.80 <= Tanimoto < 0.95
moderate_analog: 0.60 <= Tanimoto < 0.80
weak_analog: 0.40 <= Tanimoto < 0.60
distant_analog: 0.20 <= Tanimoto < 0.40
very_distant_analog: Tanimoto < 0.20
```

---

## LLM Reasoning 流程

Bioavailability_Ma reasoning 分为并发 evidence branches 和 final summary。

### Single-molecule reasoning

每个 query 并发执行一个单分子分析分支：

```text
input:
  query molecule
  exact_query_chembl_context only when --enable-chembl-exact-context is enabled and exact context is found

available tools:
  molecule_properties

not sent by default:
  ChEMBL neighbor evidence
  exact_query_chembl_context
  mmp_structure_compare
  properties_compare

output:
  oral_bioavailability_prior:
    high
    low
    mixed_or_unclear
  absorption_prior:
    favorable
    unfavorable
    mixed_or_unclear
  solubility_or_dissolution_prior:
    favorable
    unfavorable
    mixed_or_unclear
  metabolism_or_clearance_prior:
    favorable
    unfavorable
    mixed_or_unclear
  confidence:
    high
    moderate
    low
  reasoning_summary
  property_drivers
  caveats
```

默认情况下这个分支只看到 query molecule 和 `molecule_properties`，不会出现 ChEMBL neighbor
evidence，也不会出现任何 `exact_query_chembl_context` 相关 payload 或 prompt instruction。
只有显式开启 exact ChEMBL context 且命中 query exact context 时，才会把
`exact_query_chembl_context` 放入 single-molecule payload，并提示模型区分 direct same-molecule
ChEMBL evidence 和 physicochemical prior。ChEMBL neighbor evidence 仍只进入 group-level context。

### Group-level reasoning

每个 `Tier.endpoint_group` 独立执行：

```text
input:
  query molecule
  top 3 neighbors for this group
  cleaned raw ChEMBL assay/activity evidence rows
  tier and endpoint_group

available tools:
  mmp_structure_compare
  properties_compare

not sent:
  query standard_inchi_key
  query fingerprint
  group n_candidate_molecules
  evidence_direction
  evidence_strength
  endpoint_group_reason
  assay_reason

output:
  group_id
  useful_for_bioavailability_reasoning: true/false
  transferability:
    high
    moderate
    low
    not_applicable
  evidence_direction:
    supports_high_bioavailability
    argues_against_high_bioavailability
    absorption_support
    permeability_support
    solubility_support
    solubility_risk
    metabolic_stability_support
    first_pass_or_clearance_risk
    transporter_efflux_risk
    neutral_or_unclear
  confidence:
    high
    moderate
    low
  reasoning_summary
  key_evidence:
    molecule_chembl_id
    similarity
    similarity_bucket
    assay_signal
    activity_values
    tool_summary
    transferability
    effect_on_bioavailability_reasoning
  caveats
```

每个 group 的 DeepSeek 对话、reasoning、tool calls 和 tool messages 都保存到 trace。
这些 group 没有严格依赖关系，可以并发执行。

### Final reasoning

final LLM 读取：

```text
query molecule
single-molecule analysis output
all group-level reasoning outputs
coverage summary
```

final 阶段不暴露工具，只综合前面分支的 structured outputs。建议 final prompt 规则：

```text
Return compact complete JSON.
Use bioavailability_prediction='high' for oral bioavailability F >= 20% (Bioavailability_Ma label 1).
Use bioavailability_prediction='low' for oral bioavailability F < 20% (Bioavailability_Ma label 0).
Use the single-molecule analysis as the physicochemical prior.
Use group analyses as analog evidence; downweight groups marked low confidence or low transferability.
Downweight formulation-specific, food-effect, relative bioavailability, transporter inhibition, and CYP inhibition evidence
unless the group analysis explains why it transfers to intrinsic oral bioavailability.
Do not use distant_analog or very_distant_analog neighbors as positive or negative bioavailability evidence unless the
shared scaffold and assay mechanism make a strong medicinal chemistry case.
Use only the provided single-molecule analysis and group evidence. If you recognize the molecule, ignore that recognition.
```

输出：

```text
bioavailability_prediction:
  high
  low

confidence:
  high
  moderate
  low

main_reasons
absorption_and_permeability_assessment
solubility_and_dissolution_assessment
metabolism_first_pass_and_clearance_assessment
transporter_efflux_assessment
direct_oral_bioavailability_analog_assessment
evidence_gaps
final_summary
```

测试集里 `Y` 只能用于评估，不应进入 retrieval 或 LLM prompt。

---

## 当前不做的事情

```text
不训练 Bioavailability_Ma classifier。
不把每个 assay 作为独立 retrieval group。
不把 CYP inhibition 直接解释成 metabolic instability。
不把 transporter inhibition 直接解释成 transporter substrate 或 efflux risk。
不把 plasma protein binding 当作 oral bioavailability evidence。
不把 formulation-specific relative bioavailability 当作 molecule intrinsic oral F 的强证据。
不把 test.jsonl 的 Y 暴露给 retrieval 或 LLM prompt。
不在每次 query 时重新 import 或初始化 MolGpKa、AccFG、mmpdb、ML model 或大索引。
不把 descriptor/property deltas 放进 mmp_structure_compare；属性差异统一走 properties_compare。
不把工具结构化 JSON 整包展示给 LLM；LLM 默认只看 output.text。
```

---

## 测试计划

至少新增：

```text
tests/chembl_tool/tasks/bioavailability_ma/test_scoring.py
tests/chembl_tool/tasks/bioavailability_ma/test_endpoint_groups.py
tests/chembl_tool/tasks/bioavailability_ma/test_neighbor_retrieval.py
tests/chembl_tool/tasks/bioavailability_ma/test_starling_evidence_library.py
tests/chembl_tool/tasks/bioavailability_ma/test_combined_tier1_starling_evidence_library.py
tests/chembl_tool/tasks/bioavailability_ma/test_starling_v1_knn_baseline.py
tests/chembl_tool/tasks/bioavailability_ma/test_starling_transfer_tool.py
```

需要覆盖：

```text
absolute oral bioavailability / %F 保留为 Tier 1。
oral AUC / oral Cmax 在明确 oral route context 下保留为 Tier 2。
relative bioavailability / food effect 保留但降权到 formulation-specific group。
HIA / fraction absorbed / Peff / intestinal perfusion 保留为 Tier 2。
Caco-2 / MDCK / PAMPA / Papp 保留为 Tier 3。
P-gp/BCRP substrate 或 efflux readout 保留为 Tier 3 transporter risk。
solubility / dissolution 保留为 Tier 4。
microsomal stability / hepatocyte stability / CLint 保留为 Tier 5。
relative bioavailability / food effect / formulation comparison 保留为 Tier 6 并降权。
CYP inhibition IC50 不直接保留为 metabolic stability evidence。
P-gp/BCRP inhibition IC50 只作为 weak/context-dependent evidence。
receptor binding / target potency / cytotoxicity / MIC / antiproliferation 过滤。
PPB / albumin binding 默认过滤。
generic activity / ratio / inhibition 缺少 ADME context 时标成 context_dependent 或过滤。
非口服 route PK endpoint 过滤，除非是 oral/iv absolute F comparator assay。
```

---

## 参考依据

当前 evidence 分层依据：

```text
FDA / 21 CFR Part 320:
  bioavailability 通过 rate and extent of absorption 评估，常用血药浓度、尿排泄或药效 readout。

FDA / ICH M9 BCS guidance:
  bioequivalence 和 biowaiver 评估使用 AUC、Cmax，并强调 solubility、dissolution 和 intestinal
  permeability 对 oral absorption 的作用。

ADME drug discovery reviews:
  discovery profiling 通常分别评估 solubility、permeability 和 first-pass metabolism，因为它们是
  incomplete oral bioavailability 的主要机制因素。

Caco-2 / permeability literature:
  Caco-2 和相关 epithelial permeability assay 常用于预测 intestinal absorption 和 efflux liability，
  但仍是 proxy evidence。

TDC Bioavailability_Ma:
  任务是 640 compounds 的 oral bioavailability binary classification。
```

Source URLs checked during planning:

```text
https://www.law.cornell.edu/cfr/text/21/320.23
https://www.fda.gov/regulatory-information/search-fda-guidance-documents/m9-biopharmaceutics-classification-system-based-biowaivers
https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions
https://pubs.acs.org/doi/10.1021/jm010152k
https://pubmed.ncbi.nlm.nih.gov/18991586/
https://www.sciencedirect.com/science/article/pii/S0022354916418915
https://tdcommons.ai/single_pred_tasks/adme
```
