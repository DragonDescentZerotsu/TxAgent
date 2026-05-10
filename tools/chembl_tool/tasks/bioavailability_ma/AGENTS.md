# Bioavailability_Ma ChEMBL 工具说明

本目录用于实现 Bioavailability_Ma 任务的 ChEMBL evidence 筛选、molecule-level evidence
library、neighbor retrieval 和 reasoning pipeline。

目标不是训练 oral bioavailability classifier，而是构建一个可审计的 oral bioavailability
evidence library：给定 query molecule，先预取相似分子的体内 oral bioavailability、口服暴露、
吸收/通透、溶解度/溶出、代谢/清除和肠道转运体相关 evidence，再交给 reasoning LLM 判断这些
analog evidence 是否能 transfer 到 query molecule。

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

Bioavailability_Ma 当前本地数据：

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

这个阈值在代码中统一记录为 `BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT = 20.0`。

---

## 目录结构

```text
tools/chembl_tool/tasks/bioavailability_ma/
  __init__.py
  AGENTS.md
  endpoint_groups.py
  build_evidence_library.py
  retrieve_neighbors.py
  run_reasoning_pipeline.py
  run_reasoning_batch.py
  rules.py
  scoring.py
  screen_assays.py
  rescore_outputs.py
  summarize_outputs.py
  report.py
```

依赖的通用工具在：

```text
tools/chembl_tool/common/
  assay_loader.py
  export.py
  schema.py
  sqlite.py
  text.py
```

通用层负责 SQLite 连接、ChEMBL 表读取、assay/activity/target 聚合、文本标准化和 CSV/JSONL
导出。任务层只放 Bioavailability_Ma 专属规则和 CLI。

---

## 核心脚本

### `screen_assays.py`

全量筛选入口。读取 ChEMBL SQLite，按统一 Bioavailability_Ma scoring 逻辑筛出候选 assay，
并生成候选表、报告和可选 activity evidence。

推荐命令：

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.screen_assays \
  --chembl-sqlite tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db \
  --out-dir outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/raw \
  --min-score 40 \
  --progress-every 10000 \
  --export-activities
```

主要输出：

```text
outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/raw/bioavailability_assay_candidates.csv
outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/raw/bioavailability_assay_candidates.jsonl
outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/raw/bioavailability_assay_report.md
outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/raw/bioavailability_activity_evidence.csv
```

说明：

```text
bioavailability_assay_candidates.*: assay-level 候选结果
bioavailability_assay_report.md: Top examples 和 Tier 统计
bioavailability_activity_evidence.csv: 候选 assay 下的 molecule-level activity evidence
```

`--export-activities` 会额外导出每个候选 assay 对应的分子实验记录，包括 molecule ChEMBL ID、
canonical SMILES、standard_type、standard_value、units 等。这个文件是后续相似分子 evidence
retrieval 的基础。

### `rescore_outputs.py`

旧结果重打分工具。它不包含独立业务规则，只读取已有
`bioavailability_assay_candidates.csv`，调用统一的 `scoring.scored_row()` 重新打分并过滤。

用途：

```text
1. scoring/rules 更新后，不想重新扫描全量 ChEMBL assay 时，快速重算旧候选。
2. 尝试不同 --min-score。
3. 同步过滤已有 bioavailability_activity_evidence.csv。
```

注意：

```text
rescore_outputs.py 只能处理已有候选。
如果新规则变宽，可能发现原候选表之外的新 assay，这种情况必须重新跑 screen_assays.py。
如果新规则变严、重排 Tier 或调整分数，rescore_outputs.py 通常足够。
```

### `summarize_outputs.py`

健康检查报告生成工具。读取候选 CSV，统计 Tier 分布、matched keywords/endpoints、negative
flags，并列出每个 Tier 的 top examples。

### `endpoint_groups.py`

定义第二阶段的 `Tier.endpoint_group` 规则，并为 evidence library 内部审计生成：

```text
evidence_direction
evidence_strength
endpoint_group_reason
```

这些派生字段可以保留在 library/debug 文件中，但不要发送给 reasoning LLM。Bioavailability
evidence 的方向经常依赖数值、单位和实验上下文；如果无法可靠解释 `standard_value`，不要仅凭
endpoint 名称强行标成支持或反对 high bioavailability。

### `build_evidence_library.py`

把 `bioavailability_assay_candidates.csv` 和 `bioavailability_activity_evidence.csv` 构建成
molecule-level evidence library 和 query-time neighbor index。应复用预计算 ChEMBL fingerprints：

```text
tools/chembl_tool/chembl_data/chembl_36_fps/chembl_36.fps.gz
```

推荐输出：

```text
outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_molecule_evidence.jsonl
outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_neighbor_index.pkl
outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_neighbor_index.meta.json
```

### `retrieve_neighbors.py`

给定 query SMILES，按 `Tier.endpoint_group` 从 Bioavailability_Ma evidence molecule subset 中
检索 top-k non-identical neighbors。默认建议：

```text
top_k_per_group: 3
min_similarity: 0.3
exclude_exact: true
```

保留的低相似度 analog 会带上 `similarity_bucket`，交给 LLM 判断 transferability。`distant_analog`
和 `very_distant_analog` 不能作为正负 bioavailability evidence，除非共享 scaffold、同类 ADME
机制和 assay context 有很强的药化理由。

### `run_reasoning_pipeline.py`

Bioavailability_Ma 端到端 reasoning 入口。建议流程：

```text
1. 读取 Bioavailability_Ma test.jsonl 的 query molecule。
2. 调用 retrieve_neighbors.py 预取每个 group 的 ChEMBL neighbor evidence。
3. 并发执行 single-molecule analysis；DeepSeek 只可调用 molecule_properties。
4. 并发执行 group-level analysis；DeepSeek 只可调用 mmp_structure_compare 和 properties_compare。
5. final summary 读取 single + all group outputs，不暴露任何 tool。
6. 保存 retrieval/single/group/final/trace/manifest。
```

### `run_reasoning_batch.py`

批量 reasoning 入口。按 query index 调用 `run_reasoning_pipeline.py`，负责 molecule 级并行、
日志、trace 合并和评估报告。

当前建议 label 约定：

```text
Y=1 -> bioavailability_prediction=high
Y=0 -> bioavailability_prediction=low
final summary 必须在 high/low 中二选一；不要在 batch accuracy 和 macro-F1 中输出 uncertain。
其中 high 对应 oral bioavailability F >= 20%，low 对应 F < 20%。
```

如果需要保留模型不确定性，用独立字段：

```text
confidence:
  high
  moderate
  low
```

### 当前输出目录

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

reasoning 产物统一放在：

```text
outputs/chembl_tool/tasks/bioavailability_ma/reasoning/runs/
outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches/
```

查看 trace 使用通用 viewer：

```bash
bash tools/chembl_tool/trace_viewer/start_viewer.sh \
  outputs/chembl_tool/tasks/bioavailability_ma/reasoning/runs \
  8776
```

### `rules.py`

定义 Bioavailability_Ma 专属 assay screening 规则：

```text
Tier 1: direct absolute oral bioavailability
Tier 2: in vivo oral exposure and in vivo/in situ absorption
Tier 3: in vitro intestinal permeability and efflux
Tier 4: solubility, dissolution and GI stability
Tier 5: metabolism, first-pass and clearance
Tier 6: formulation, food-effect and relative bioavailability context
negative keywords
weak terms
transporter target genes
```

不要在脚本各处散落硬编码规则；新增或调整关键词优先改这里和 `scoring.py`。

### `scoring.py`

统一业务逻辑入口。全量筛选和旧结果重打分都调用：

```python
scored_row(row, min_score=40)
```

当前所有 Bioavailability_Ma assay 保留/剔除逻辑都应在 `rules.py` 和 `scoring.py` 中维护。
不要在 `rescore_outputs.py` 里新增另一套 postprocess 规则。

---

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

available tools:
  molecule_properties

not used:
  ChEMBL neighbor evidence
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

这个分支只能看到 `molecule_properties`，不能看到 ChEMBL neighbor evidence 或分子比较工具。

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
