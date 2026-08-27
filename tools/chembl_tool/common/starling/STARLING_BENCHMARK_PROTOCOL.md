# Starling 二分类 benchmark 构建协议

本协议用于把 Starling direct literature records 转成冻结 task contract 下的二分类
`train.jsonl` / `valid.jsonl` / `test.jsonl`。它与 Starling evidence retrieval index 分离：benchmark builder
负责 gold-label 构建，`minimal_evidence.v1` 继续只负责 inference-time evidence。

在冻结 molecule-only gold 之上增加 exact external-condition groups 的当前规则、allowlist、
60% vote 和逐条 group/lineage artifact 合同，统一见
[`REVIEWED_CONDITIONED_BENCHMARK.md`](REVIEWED_CONDITIONED_BENCHMARK.md)。两套 lineage 不得混称。

## 当前 BBB experimental meaningful-CNS-access lineage（2026-08-26）

BBB 当前 paper-facing gold 是 `experimental_meaningful_cns_access_v3`，不再把所有 Starling
`bbb_permeability_label` 无条件混成一个 TDC-compatible endpoint。目标定义为：**系统给药后是否有实验支持的
meaningful/adequate CNS access，或相反的 restricted/poor access**。Positive 不要求 passive diffusion，也不等于
任何微量 signal 可检出；低但非零 exposure 可以保留为 negative。

允许进入 gold 的 endpoint family：

```text
brain_unbound
brain_systemic_ratio
brain_tissue
csf                     # proxy outcome，不等同于脑实质
pet_or_autoradiography
experimental_bbb_outcome
```

必须同时有明确 binary `bbb_permeability_label` 和可审计的实验/测量依据；若 source 只给 quantitative direct
metric 而没有显式实验语境，则必须同时有非空 value claim、PMID 且无 prediction cue。计算或 in-silico prediction、
PAMPA/MDCK/其它体外 passive-permeability assay、mechanism-only transporter proxy、非系统 CNS 给药、人工或
疾病改变屏障、间接疗效推断、`CSF half-life` 等消除 readout，以及明显的同记录方向冲突均拒绝。各种 brain/CSF metrics 不使用一个共享
numeric threshold；通过 scope gate 后只使用 source 的明确 qualitative label，再按公共 70% record-majority
聚合 parent。Gold 不按 passive/efflux/influx 或 endpoint family 设置配额，也不为了让某个 agent group 有用而
重采样或改标签；这些 group 在 held-out-filtered train-only evidence index 上另做 coverage audit。

v3 另外明确排除 source-native ADME/T/TCMSP computational prediction。它从 v2 的全部 source rows 独立
重新判定和重投票，并为所有 surviving parents 保留 v2 scaffold assignment。当前 build 为 3,666 parents，
Y=0/Y=1 为 `966/2,700`，train/valid/test 为 `2,934/366/366`；valid/test 分别有 21/22 singleton，且两者 label 均为
`97/269`，identity/scaffold overlap 为 0。构建、审计和产物：

```text
tools/chembl_tool/tasks/bbb_martins/experimental_meaningful_cns_access_benchmark_v3.py
tools/chembl_tool/common/starling/build_bbb_experimental_meaningful_cns_access_v3.py
tools/chembl_tool/common/starling/audit_bbb_v3_migration.py
tools/chembl_tool/common/starling/audit_bbb_experimental_meaningful_cns_access.py
data/processed_starling_experimental_meaningful_cns_access_v3/BBB_Martins/
```

迁移 receipt 记录 accepted source rows `8,273 → 8,268`，只移除旧 train 的 Digoxin parent；3,666 个
shared parents 的 label 和 split changes 均为 0。v2 的三名 `gpt-5.6-sol` reviewer 在多轮 replacement audit
中检查了 366 条 unique source records；294 条
family×label 分层 deterministic sample 全部人工通过。主要排除项包括 prediction/in-vitro、altered barrier、
间接 pharmacodynamic inference、query/analyte/PMID mismatch、parent/metabolite ambiguity、total-radioactivity
attribution 和不受当前 small-molecule identity/tool contract 支持的 metal complex。该抽样不能替代未来双人原文
annotation，因此当前状态是 reproducible high-precision build，不是最终 paper source-quality gold certification。
旧 `experimental_meaningful_cns_access_v2`、`experimental_direct_cns_v1`、BBB
`record_agreement70_split811_v1` 和 `record_supported_v2` 保留为 historical lineage。

唯一重建和审计命令：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.common.starling.build_bbb_experimental_meaningful_cns_access_v3

/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.common.starling.audit_bbb_v3_migration
```

第二条命令逐 parent 比较 v2/v3 的 membership、label 和 split，并同时比较 conditioned v1/v2；它不能被
retrieval-family audit 替代。若需要重新生成 source QA sample，仍显式运行 historical source audit 入口，
不得据此覆盖 v3 migration receipt。

## BBB experimental metric-direction review v4 candidate（2026-08-26）

`experimental_meaningful_cns_access_v4` 是独立 candidate lineage，尚未替换 paper-facing v3。它只重审 v3
中通过 direct-outcome 和 experimental-basis gates、但仅因缺少 `bbb_permeability_label` 被拒绝的 7,226 条
source rows。不同 brain/CSF/PET measurements 不使用共享 numeric threshold；只有 source row 自身提供无歧义
qualitative direction 才成为规则候选。规则在 1,418 条已有 explicit-label records 上逐规则回放，最终所有启用
规则 disagreement 均为 0。

规则提出 81 条 candidates，全部进行 source-index 级人工复核：16 条因 ex-vivo、metabolite/prodrug/analyte
attribution、mechanism-only statement、pharmacodynamic proxy、disease context、late-timepoint 或 PET confounding
被拒绝；65 条新增为 votes；其余 7,145 条继续是 non-voting retrieval evidence。审查绑定 frozen HF revision 和
local Arrow SHA-256 `faa63a4e2ecc234691c86bc63c2cd3a69db82387cfe0687cc930d9a5e458709c`，candidate set
发生漂移时构建直接失败。

v4 重投票得到 3,675 binary parents，Y=0/Y=1 为 `975/2,700`，train/valid/test 为
`2,945/365/365`。相对 v3 新增/移除 15/6 parents；3,660 个 shared binary parents 的 label flips 和 split
changes 均为 0。所有 surviving v3 parents 保留原 split；新 parent 若 scaffold 已存在则继承该 split，全新
scaffold 只进入 train。identity/scaffold overlap 均为 0。当前 v4 仅是 gold candidate；在相同 baseline 和
agent matrix 重跑并完成 promotion gate 前，不与 v3 performance 混表。

```text
tools/chembl_tool/tasks/bbb_martins/experimental_metric_direction_review.py
tools/chembl_tool/tasks/bbb_martins/experimental_metric_direction_review_adjudications.json
tools/chembl_tool/tasks/bbb_martins/experimental_meaningful_cns_access_benchmark_v4.py
tools/chembl_tool/common/starling/audit_bbb_experimental_metric_direction.py
tools/chembl_tool/common/starling/build_bbb_experimental_meaningful_cns_access_v4.py
tools/chembl_tool/common/starling/audit_bbb_v4_migration.py
data/starling_data/bbb_martins/experimental_metric_direction_review_v1/
data/processed_starling_experimental_meaningful_cns_access_v4/BBB_Martins/
```

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.common.starling.audit_bbb_experimental_metric_direction \
  --source-arrow /path/to/pinned/bbb-train.arrow

/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.common.starling.build_bbb_experimental_meaningful_cns_access_v4 \
  --source-arrow /path/to/pinned/bbb-train.arrow

/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.common.starling.audit_bbb_v4_migration
```

## Parent record-majority policy

Starling 原始论文没有发布一个统一的二值化脚本。论文将每条 extraction 按“如果放进相应 TDC
会得到什么标签”进行分析，并明确指出同一分子会因 species、dose、formulation、disease state、
transporter mechanism 等上下文产生不同结果。论文 Appendix H 展示的是 molecule 内 positive
fraction，而不是把所有 contextual records 无条件压成一个 gold label。

所有当前 lineage 在保留 source-row ambiguity gate 的前提下，采用用户冻结的 70% record-weighted majority：

1. source row 先按 task-specific frozen scope/label adapter 独立转成 `0/1/ambiguous`；BBB 当前使用上述
   experimental meaningful-CNS-access contract，不能再描述成 TDC-compatible conversion；
2. SMILES 用 `rdkit_fragment_parent.v1` 归一化，盐型不成为独立 benchmark molecule；
3. 同一个 parent 内以每条 accepted source record 为一票，计算
   `agreement=max(n0,n1)/(n0+n1)`；同一 PMID 的多条 record 仍分别计票；
4. agreement `>= 0.70` 且不是精确 50/50 tie 时，接受多数 label；否则写入
   `rejected_parent_molecules.jsonl`；`conflicting_molecules.jsonl` 保留所有同时出现 0/1 的 parent，
   并标记它最终是 majority-accepted 还是 rejected；
5. range/inequality 跨过分类 threshold 时写入 ambiguous，不取 midpoint；
   `mean ± error` 按完整 `[mean-error, mean+error]` 区间处理；
6. 相对比较、单位不兼容、非目标 endpoint、inconclusive 和无法确认 population 的记录不进入 gold label；
7. Starling 明确标为 `qualifying_conditions` 的 dose、formulation、disease state、co-treatment 等
   条件性 record 不进入当前 molecule-only gold；
8. 第一版 `record_agreement70_split811_v1` 的 valid 与 test target 分别为
   `min(500, floor(0.1 * n_binary_molecules))`，train 使用剩余 parents；
9. 第一版从同一批 accepted parents 同时生成两个版本：
   - `random`：固定 seed 的 label-stratified stable-hash random split；
   - `scaffold`：以 canonical Bemis–Murcko scaffold 为不可拆分 group，按固定 seed 排序后
     先选完整 test scaffold groups，再从剩余 groups 选择 valid，以保证 train/valid/test scaffold 两两不重叠；
10. 第一版 scaffold group 无法精确凑到 valid/test target 时只能从下方逼近，并在 summary 中记录 shortfall，
    不允许拆散同一 scaffold 来凑整数。

这套规则优先保证 label precision 和可审计性，而不是最大化保留率。

## Bioavailability/Skin 当前与 BBB historical record-supported v2 split policy

Bioavailability/Skin 当前 paper-facing lineage 是 scaffold-only `record_supported_v2`；BBB 同 lineage 自
2026-08-09 起只作 historical comparison。它复用第一版冻结的全部 binary parent
labels，只重新分配 scaffold groups；因此 label 变化和 split 变化不会混在一起。分配使用 lexicographic
MILP，并严格按以下优先级冻结前一层最优值后再优化下一层：

1. train/valid/test scaffold 和 parent identity 两两零重叠；
2. valid/test 必须精确达到目标大小：BBB 各 500，Bioavailability/Skin 各为总 parent 的 10%；
3. 最小化 valid+test 中的 singleton parents；
4. 最小化 valid/test singleton 数量差；
5. 最小化两个 held-out split 相对全数据正类比例的总偏差；
6. 仅在上述数据质量目标全部固定后，最大化第一版 valid molecule 复用；
7. 最后用固定 seed 的 scaffold hash rank 消除 solver tie，使构建可重放。

冻结结果为 BBB `18,425/500/500`、Bioavailability `1,674/209/209`、Skin
`1,966/245/245`。BBB 与 Bioavailability 的 valid/test 全部是 multi-record parents；Skin 每个 held-out
split 是 240 multi-record + 5 singleton，这 10 个 singleton 是精确 split 大小和 scaffold-disjoint 硬约束下
的全局最小值。三个 task 的 identity/scaffold overlap 都是 0。

## Held-out 隔离要求

构造 gold split 和构造 retrieval evidence 是两个独立步骤。现有 Starling evidence index 是从
full direct source 建立的，其中包含新 valid/test molecules 的 source rows，**不能**直接用于这个 benchmark。

- `random/heldout_molecule_labels.jsonl` 和 `scaffold/heldout_molecule_labels.jsonl`
  分别是各构造方法 valid+test parent 的 union 排除清单；
- 后续 evidence library 必须排除 valid/test parent 的全部盐型、片段和重复记录；
- 不允许只依赖 query-time exact-SMILES exclusion；identity-equivalent source rows 也必须排除；
- train-only index 对 valid+test union 完成覆盖率和 zero-overlap audit 之前，不能启动正式 paper rerun。
- inference-time direct evidence 还必须通过与 gold adapter 相同的 task scope gate；held-out parent exclusion
  不能修复 endpoint-scope mismatch。Skin current index 因此复用
  `is_tdc_skin_sensitization_scope()`，历史 broad-skin index 只作 v1 reproduction。

## 当前 task policy

### BBB_Martins

当前目标为系统给药后实验支持的 meaningful/adequate CNS access vs restricted/poor access；旧
TDC-compatible mapping 只作 historical lineage。

- brain/CSF access outcome 先通过 experimental scope gate，再读取明确 `bbb_permeability_label`；
- CSF 是 direct CNS proxy，不等同于 brain parenchyma；
- passive permeability、efflux/influx、transporter binding/inhibition 是 agent mechanism evidence，不产生 gold；
- low-but-nonzero exposure 可为 negative；mere detectability 不是 positive rule；
- 不把 Papp、Kp、Kp,uu、brain concentration、CSF concentration 等异构量用一个阈值强行换算；
- prediction、in-vitro-only、altered-barrier/non-systemic、indirect efficacy inference 和明显方向冲突拒绝；
- molecule parent 继续按统一 70% record-agreement policy 决定接受或拒绝。

### Bioavailability_Ma

TDC 目标为 human oral bioavailability，`F >= 20%` 为 positive。

- gold 与 agent direct evidence 共同读取
  `data/starling_data/bioavailability_ma/canonical_direct_v2/direct_claims.parquet`；
- canonical source 由固定 revision 的 HF `starling-labs/Oral_Bioavailability` 与 local exposure
  extraction 中有明确 absolute/oral-IV 锚点的记录合并；同 parent、PMID、threshold side 且数值/区间
  相差不超过 1 percentage point 的跨来源记录只形成一个 claim；
- 原始 HF snapshot 和 local parquet 保持 immutable；local 中确认 absolute 的行转入 canonical，active
  residual source 不再包含该分区。明确 relative 或没有 absolute 锚点的 bioavailability 行只保留为
  inference-time exposure/context，不进入 gold；
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
- current Tier-1 direct evidence library 使用同一 scope predicate；scope 外 rows 不得先聚合进 molecule card
  再要求 LLM 从 summary/examples 中自行忽略。

### ClinTox

TDC ClinTox positive 表示冻结 AACT toxicity-failure-associated source class；不是任意 in vitro liability，
也不是对每条试验重新完成的 molecule-causality adjudication。不从 Starling free text 或 heterogeneous
liability records 构造 ClinTox label。独立的 `clinical_trial_failure_v1` lineage 从冻结 AACT
toxicity-failure source 和 SWEETLEAD/FDA-approved comparator 重建原任务语义，路径为
`data/processed_clintox_clinical_trial_failure_v1/ClinTox/scaffold/`；它不是 Starling gold lineage，
不得与本协议的三个 Starling task 混称为同源数据。Starling clinical/mechanistic rows 只进入独立
retrieval library，不参与 gold 投票。

## Free-text 到 binary label 的精确规则

这里记录 task adapters 当前实际执行的 mapping，避免把解释性自然语言误当成隐式多数票或模型判断。
任何修改都必须同时修改 adapter tests 和本节。

### BBB_Martins

`bbb_permeability_label` 先转小写，并把空格/连字符归一化为下划线。只接受以下显式枚举：

```text
Y=1:
  permeable
  good_penetration
  increased_permeability
  high_permeability

Y=0:
  poor_penetration
  impermeable
  low_permeability
  restricted
```

`bbb_transport_label` 即使写了 influx/efflux，也只描述机制，不产生 gold label。任意其它 qualitative
文本只有在同一 row 还存在可解释的明确 logBB 时才可能由 logBB 得到 label；qualitative 与 logBB
冲突则整条 row 为 `within_record_label_conflict`。

### Bioavailability_Ma

没有可解析数字时，只接受能明确落在 20% threshold 两侧的描述：

```text
Y=1:
  high
  good
  excellent
  complete / completely
  near complete / nearly complete
  almost complete

Y=0:
  very low
  low
  poor
  negligible
  minimal
```

以下描述不提供 20% threshold 信息，单独出现时拒绝：

```text
moderate
variable
unpredictable
intermediate
orally bioavailable
orally available
```

`fold`、`times`、`relative`、`increase/decrease by`、`higher/lower than` 等相对比较也拒绝。
如果文本同时含数字，则优先按可审计的 numeric interval/range 和 unit normalization 判断；
跨 20% 的 range 不取 midpoint，仍为 ambiguous。

### Skin_Reaction

先要求 `reaction_type` 归一化后属于：

```text
sensitization
allergic contact dermatitis
contact allergy
```

然后只按明确 `outcome_label` 映射：

```text
positive -> Y=1
negative -> Y=0
inconclusive / missing / unknown -> reject
```

`positive_count` 和 `total_tested` 只保留为 provenance，不另造 incidence threshold。只写
`skin reaction`、irritation、urticaria、generic local injury、phototoxicity/photo-irritation 或
permeability/exposure 的记录不映射为 sensitization gold。

## 代码与输出

公共逻辑：

```text
tools/chembl_tool/common/starling/benchmark_dataset.py
tools/chembl_tool/common/starling/build_benchmark_datasets.py
tools/chembl_tool/common/starling/build_record_supported_benchmark.py
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
  rejected_parent_molecules.jsonl
  source_rejection_examples.jsonl
  summary.json
  report_zh.md
  random/
    train.jsonl
    valid.jsonl
    test.jsonl
    train_molecule_labels.jsonl
    valid_molecule_labels.jsonl
    test_molecule_labels.jsonl
    heldout_molecule_labels.jsonl
    summary.json
  scaffold/
    train.jsonl
    valid.jsonl
    test.jsonl
    train_molecule_labels.jsonl
    valid_molecule_labels.jsonl
    test_molecule_labels.jsonl
    heldout_molecule_labels.jsonl
    summary.json
```

两个 split 目录中的 `train.jsonl` / `valid.jsonl` / `test.jsonl` 只含 runner 需要的 `drug` 和 `Y`。
详细 label method、source IDs、
PMIDs、raw value examples、identity metadata 和冲突信息放在 audit artifacts，不能整包进入 LLM prompt。
根目录 `molecule_labels.jsonl` 带有 `split_assignments.random/scaffold`；每个 split-specific audit
文件用于后续按对应 parent identity 构建 train-only retrieval library。

Bioavailability/Skin current、BBB historical v2 构建命令与输出：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.common.starling.build_record_supported_benchmark
```

```text
data/processed_starling_record_supported_v2/<Task>/scaffold/
  train.jsonl
  valid.jsonl
  test.jsonl
  train_molecule_labels.jsonl
  valid_molecule_labels.jsonl
  test_molecule_labels.jsonl
  heldout_molecule_labels.jsonl
  summary.json
```

v2 不复制根级 source-row rejection/conflict artifacts；这些 immutable label provenance 仍由
`data/processed_starling/<Task>/` 提供。v2 目录只保存发生变化的 split 与 split-specific audit，避免重复
存放同一批 source records。

## 版本与来源

- Starling paper v3: https://arxiv.org/abs/2605.07022
- Starling code and extraction guidance: https://github.com/starling-labs/starling
- Starling BBB: https://huggingface.co/datasets/starling-labs/BBB
- Starling Oral Bioavailability: https://huggingface.co/datasets/starling-labs/Oral_Bioavailability
- TDC ADME tasks: https://tdcommons.ai/single_pred_tasks/adme/
- TDC task supplement: https://zitniklab.hms.harvard.edu/publications/papers/TDC-neurips21-supp.pdf

HF source revision、local parquet SHA-256、分区守恒和 claim-level dedup policy 固定在
`canonical_direct_v2/merge_manifest.json`；gold summary 与 agent index meta 必须记录同一份
`direct_claims.parquet` SHA-256。源数据更新后必须使用新 output root 或明确审计 diff，不能静默覆盖并沿用旧结果。

## 冲突与 agreement 定义

冲突发生在 split 之前。每条通过 task scope、单位、population、context 和 ambiguity gate 的 source
record 先独立获得 `Y=0` 或 `Y=1`，再按 `rdkit_fragment_parent.v1` 聚合。如果同一个 normalized
parent 的 accepted records 同时出现 0 和 1，则该 parent 是 label-conflict parent。v4 不再自动丢弃：
多数一侧占全部 accepted records 的比例达到 70% 就接受该多数 label；低于 70% 或精确 tie 才拒绝。
这里的 vote unit 是 record，不是 unique PMID。

这与 `within_record_label_conflict` 不同：后者指同一条 source record 内的两个可用信号已经互相矛盾，
例如 BBB qualitative 写 `permeable`，但同条明确 logBB 小于 -1；这种 row 在 parent 聚合前就被拒绝。

## 2026-08-02 record-agreement70 split811 v1 build

默认 `agreement_threshold=0.70`、`valid_fraction=test_fraction=0.1`、
`max_eval_size=500`、`seed=20260723`：

| task | binary parents | rejected | valid/test target | random valid Y=0 / Y=1 | random test Y=0 / Y=1 | scaffold valid Y=0 / Y=1 | scaffold test Y=0 / Y=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB_Martins | 19,425 | 1,091 | 500 / 500 | 139 / 361 | 139 / 361 | 150 / 350 | 148 / 352 |
| Bioavailability_Ma | 2,092 | 106 | 209 / 209 | 58 / 151 | 58 / 151 | 61 / 148 | 65 / 144 |
| Skin_Reaction | 2,456 | 465 | 245 / 245 | 75 / 170 | 75 / 170 | 74 / 171 | 71 / 174 |

两种构造均精确命中 valid/test target，且 train/valid/test parent identity 两两零重叠。三个 scaffold
版本的 train/valid/test Bemis–Murcko scaffold 也两两零重叠。random 与 scaffold 是两个独立版本，
因此同名 valid 或 test subset 跨构造方法允许出现 parent 交集。
