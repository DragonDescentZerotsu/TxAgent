# Starling 二分类 benchmark 构建协议

本协议用于把 Starling direct literature records 转成与当前 TDC task 语义兼容的二分类
`train.jsonl` / `test.jsonl`。它与 Starling evidence retrieval index 分离：benchmark builder
负责 gold-label 构建，`minimal_evidence.v1` 继续只负责 inference-time evidence。

## 为什么不能直接多数票

Starling 原始论文没有发布一个统一的二值化脚本。论文将每条 extraction 按“如果放进相应 TDC
会得到什么标签”进行分析，并明确指出同一分子会因 species、dose、formulation、disease state、
transporter mechanism 等上下文产生不同结果。论文 Appendix H 展示的是 molecule 内 positive
fraction，而不是把所有 contextual records 无条件压成一个 gold label。

因此本项目采用保守的 molecule-level 规则：

1. source row 先按 task-specific TDC 语义独立转成 `0/1/ambiguous`；
2. SMILES 用 `rdkit_fragment_parent.v1` 归一化，盐型不成为独立 benchmark molecule；
3. 一个 parent 的所有可用 source rows 必须一致；
4. 同时出现 0 和 1 的 parent 不做多数票，写入 `conflicting_molecules.jsonl`；
5. range/inequality 跨过分类 threshold 时写入 ambiguous，不取 midpoint；
   `mean ± error` 按完整 `[mean-error, mean+error]` 区间处理；
6. 相对比较、单位不兼容、非目标 endpoint、inconclusive 和无法确认 population 的记录不进入 gold label；
7. Starling 明确标为 `qualifying_conditions` 的 dose、formulation、disease state、co-treatment 等
   条件性 record 不进入当前 molecule-only gold；
8. test target 为 `min(500, floor(0.2 * n_binary_molecules))`；
9. 同一批 accepted parents 同时生成两个版本：
   - `random`：固定 seed 的 label-stratified stable-hash random split；
   - `scaffold`：以 canonical Bemis–Murcko scaffold 为不可拆分 group，按固定 seed 排序后
     用 subset-sum 从下方逼近 test target；
10. scaffold group 无法精确凑到 target 时只能从下方逼近，并在 summary 中记录 shortfall，
    不允许拆散同一 scaffold 来凑整数。

这套规则优先保证 label precision 和可审计性，而不是最大化保留率。

## Held-out 隔离要求

构造 gold split 和构造 retrieval evidence 是两个独立步骤。现有 Starling evidence index 是从
full direct source 建立的，其中包含新 test molecules 的 source rows，**不能**直接用于这个 benchmark。

- `random/test_molecule_labels.jsonl` 和 `scaffold/test_molecule_labels.jsonl`
  是两套必须分别排除的 parent-identity 清单；
- 后续 evidence library 必须只从 train-source records 重建，并排除 test parent 的全部盐型、片段和重复记录；
- 不允许只依赖 query-time exact-SMILES exclusion；identity-equivalent source rows 也必须排除；
- train-only index 完成覆盖率和 zero-overlap audit 之前，不能启动正式 paper rerun。

## 当前 task policy

### BBB_Martins

TDC 目标为 BBB pass/fail；常用数值表述是 `logBB >= -1` 为 positive。

- Starling 的明确 `bbb_permeability_label` 映射到 pass/fail；
- `bbb_transport_label` 只描述机制/上下文，不参与 gold label；
- 只有明确 `logBB` 数值使用 `-1` threshold；
- Papp、Kp、Kp,uu、brain concentration、CSF concentration 等异构量不互相换算；
- `qualifying_conditions` 非空的 context-dependent record 不进入 molecule-only gold；
- qualitative permeability 与 logBB 信号在同一 record 内冲突时，该 record 为 ambiguous；
- molecule parent 跨文献/条件冲突时整个 molecule 不进入 binary benchmark。

### Bioavailability_Ma

TDC 目标为 human oral bioavailability，`F >= 20%` 为 positive。

- 只保留可确认是 human subjects/patients/volunteers 的 direct oral bioavailability；
- `%`、`percent`、`per cent` 统一为 percent；
- 0–1.5 的明确 fraction/unitless F 转成百分数；
- absolute、systemic availability、extent F 和明确 direct unspecified reports 可用；
- relative comparison、fold change、AUC/Cmax/Tmax proxy 不作为 direct F label；
- `qualifying_conditions` 非空的 formulation/food/disease/co-treatment record 不进入 gold label；
- 数值 range 跨 20% 时为 ambiguous；
- 无数值时只映射明确 high/good/excellent/complete 或 low/poor/negligible 描述；
- “orally bioavailable”、moderate、variable、increased/decreased 等不提供 20% threshold 信息，
  不强行映射。

### Skin_Reaction

TDC 的 404-molecule Skin_Reaction task 来自 skin-sensitization 数据；正类是 sensitizer，
负类是 non-sensitizer。Starling direct source 中 irritation、generic local damage 和
skin exposure 不等于这个 label。

- 只保留 `sensitization` 或 allergic contact dermatitis/contact allergy scope；
- 使用 Starling 明确的 `outcome_label=positive/negative`；
- `inconclusive` 不进入 binary benchmark；
- `positive_count` / `total_tested` 保留为 provenance，不另造 incidence threshold；
- irritation、urticaria、generic skin reaction 和 photo-irritation 默认不进入该 TDC-compatible label。

### ClinTox

TDC ClinTox positive 表示因 toxicity 导致 clinical-trial failure；不是任意 in vitro liability。
当前仓库还没有 ClinTox Starling direct acquisition，因此本轮不构造 ClinTox Starling split。

## 代码与输出

公共逻辑：

```text
tools/chembl_tool/common/starling/benchmark_dataset.py
tools/chembl_tool/common/starling/build_benchmark_datasets.py
```

Task-specific adapters：

```text
tools/chembl_tool/tasks/bbb_martins/starling_benchmark.py
tools/chembl_tool/tasks/bioavailability_ma/starling_benchmark.py
tools/chembl_tool/tasks/skin_reaction/starling_benchmark.py
```

运行：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.common.starling.build_benchmark_datasets
```

只重建聚合 summary、不重读原始数据：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.common.starling.build_benchmark_datasets \
  --summarize-existing
```

默认输出：

```text
data/processed_starling/<Task>/
  molecule_labels.jsonl
  conflicting_molecules.jsonl
  source_rejection_examples.jsonl
  summary.json
  report_zh.md
  random/
    train.jsonl
    test.jsonl
    train_molecule_labels.jsonl
    test_molecule_labels.jsonl
    summary.json
  scaffold/
    train.jsonl
    test.jsonl
    train_molecule_labels.jsonl
    test_molecule_labels.jsonl
    summary.json
```

两个 split 目录中的 `train.jsonl` / `test.jsonl` 只含 runner 需要的 `drug` 和 `Y`。
详细 label method、source IDs、
PMIDs、raw value examples、identity metadata 和冲突信息放在 audit artifacts，不能整包进入 LLM prompt。
根目录 `molecule_labels.jsonl` 带有 `split_assignments.random/scaffold`；每个 split-specific audit
文件用于后续按对应 parent identity 构建 train-only retrieval library。

## 版本与来源

- Starling paper v3: https://arxiv.org/abs/2605.07022
- Starling code and extraction guidance: https://github.com/starling-labs/starling
- Starling BBB: https://huggingface.co/datasets/starling-labs/BBB
- Starling Oral Bioavailability: https://huggingface.co/datasets/starling-labs/Oral_Bioavailability
- TDC ADME tasks: https://tdcommons.ai/single_pred_tasks/adme/
- TDC task supplement: https://zitniklab.hms.harvard.edu/publications/papers/TDC-neurips21-supp.pdf

HF source revision 固定在 task adapter 常量中；local parquet 的 SHA-256 写入每次生成的
`summary.json`。源数据更新后必须使用新 output root 或明确审计 diff，不能静默覆盖并沿用旧结果。

## 冲突定义

冲突发生在 split 之前。每条通过 task scope、单位、population、context 和 ambiguity gate 的 source
record 先独立获得 `Y=0` 或 `Y=1`，再按 `rdkit_fragment_parent.v1` 聚合。如果同一个 normalized
parent 的 accepted records 同时出现 0 和 1，则该 parent 是
`conflicting_parent_level_labels`，整个 parent 从 random/scaffold 两套候选池中排除，不做多数票。

这与 `within_record_label_conflict` 不同：后者指同一条 source record 内的两个可用信号已经互相矛盾，
例如 BBB qualitative 写 `permeable`，但同条明确 logBB 小于 -1；这种 row 在 parent 聚合前就被拒绝。

## 2026-07-24 frozen build

默认 `max_test_size=500`、`test_fraction=0.2`、`seed=20260723`：

| task | binary parents | target | random test Y=0 / Y=1 | scaffold test Y=0 / Y=1 | conflicting parents |
|---|---:|---:|---:|---:|---:|
| BBB_Martins | 17,893 | 500 | 139 / 361 | 122 / 378 | 2,623 |
| Bioavailability_Ma | 1,862 | 372 | 99 / 273 | 113 / 259 | 385 |
| Skin_Reaction | 1,900 | 380 | 129 / 251 | 117 / 263 | 1,021 |

两种 split 均精确命中 target 且各自 train/test parent identity 为零重叠。三个 scaffold split 的
train/test Bemis–Murcko scaffold overlap 均为 0。random 与 scaffold 是两个独立版本，所以它们的
test 集之间允许重叠；实际交集依次为 BBB 9、Bioavailability 80、Skin_Reaction 79 个 parents。
