# ClinTox task notes

本文件只记录 ClinTox 的 task-specific 语义：label、当前数据、临床毒性 evidence tier、
assay/endpoint 解释、过滤规则和 reasoning schema。通用 ChEMBL workflow、wrapper 结构、
batch/resume、viewer、cost 和目录规范统一记录在仓库根 `AGENTS.md`。

## Task 定义

目标不是训练一个黑盒 ClinTox classifier，而是构建可审计的 clinical toxicity evidence library：
给定 query molecule，先预取相似分子的临床/非临床毒性、器官毒性、safety pharmacology、
cellular stress、general cytotoxicity 和 safety-relevant off-target evidence，再交给 reasoning LLM
判断 analog evidence 是否能 transfer 到 query molecule。

ClinTox 的旧 task-specific optimization workflow 已进入归档状态：旧 native runner、历史 batch、trace 和
audit 工具继续保留用于复现与案例分析，不再按 test errors 调整 task-specific policy。ClinTox 本身仍是当前
21-condition 论文矩阵中的正式任务；paper-facing `none/direct/full_flat/full_mechanism` 由通用 runner 和
`experiment_config.py` 生成，不能用下面的历史 native metrics 代替当前论文结果。

当前 paper config 将 source-local groups 合并为 7 个 mechanism families：clinical human safety、in vivo
toxicology、organ-specific toxicity、genotoxicity/carcinogenicity、cellular stress、general cytotoxicity 和
off-target/DDI/exposure。并行 reasoning 以这 7 个 family 为上限，而不是以 43 个 endpoint groups 为单位。

旧 native workflow 的归档结论：

```text
ClinTox 不太适合当前 setting 直接作为主 benchmark 分类任务。

当前 workflow 擅长做 toxicity evidence retrieval 和 mechanistic risk explanation；
但 ClinTox label 是高层 clinical toxicity / clinical failure 二分类，而 ChEMBL 中检索到的
evidence 多数是 heterogeneous safety liability：
  hERG / QT / 5-HT2B / receptor binding
  CYP / transporter / DDI / exposure liability
  generic cell viability / cytotoxicity
  DILI / BSEP / mitochondrial hepatocyte stress
  LD50 / MTD / NOAEL / repeat-dose toxicology

这些 evidence 与 clinical toxicity 有关系，但很多不是 ClinTox-positive 的充分条件。
因此系统容易把机制性 liability 或 broad medicinal-chemistry toxicity risk 上升为 toxic，
导致 FP 偏多；同时部分 positive examples 需要很窄的 clinical-label ontology 才能稳定判中。
```

已归档的主要评估结果：

```text
v7 full final-only + missing rerun 合并估计:
  batch: outputs/chembl_tool/tasks/clintox/reasoning/batches/clintox_full_prompt_v7_final_only_from_v2
  fill:  outputs/chembl_tool/tasks/clintox/reasoning/batches/clintox_full_prompt_v7_missing_rerun_from_v2
  TN=220 FP=48 FN=12 TP=6
  macro-F1 ~0.523
  positive F1 ~0.167

v8 keygroups smoke:
  batch: outputs/chembl_tool/tasks/clintox/reasoning/batches/clintox_group_prompt_v8_keygroups_smoke
  indices: 22 40 56 65 73 84 124 170 250
  TN=1 FP=3 FN=1 TP=4
  macro-F1=0.50
  positive recall=0.80
```

`keygroups` 是一次受控 smoke test，不是正式默认流程。它手动只选择更接近 ClinTox label 的
high-value endpoint groups，例如：

```text
Tier 1 clinical_toxicity_or_trial_failure
Tier 1 human_maximum_tolerated_dose_or_safety_margin
Tier 2 in_vivo_acute_or_repeat_dose_toxicity
Tier 2 in_vivo_ld50_lc50_mtd_noael_loael
Tier 3 hepatic_dili_or_liver_injury
Tier 3 hepatic_cell_injury_or_steatosis
Tier 3 hepatic_mitochondrial_or_oxidative_stress
Tier 3 hepatic_bile_acid_transport_or_bsep
Tier 3 cardiac_herg_or_ikr_block
Tier 4 dna_damage_response
Tier 6 general_cytotoxicity_or_viability
Tier 7 cardiac_or_cns_offtarget_binding
Tier 7 cytochrome_p450_inhibition_or_induction
Tier 7 transporter_ddi_or_exposure_liability
```

keygroups 实验说明 final 输入选择/压缩有价值：它救回了 `idx73` 和 `idx250`，保住了
`idx84` 和 `idx170`，并把 `idx22` 从 FP 修成 TN。但它没有稳定解决 `idx40`、`idx56`、
`idx65` 的 FP，也没有救回 `idx124`。后续如果重启 ClinTox，优先做自动 final-context
compression/filter，而不是继续堆 final prompt：

```text
direct severe clinical anchor
in vivo dose-limiting anchor
mechanistic liability
weak/background context
```

其中 hERG、5-HT2B、CYP、transporter、DDI、generic cytotoxicity 等机制性 liability
应主要作为 risk explanation 或 uncertainty，不应单独决定 `clintox_prediction=toxic`。

当前本地数据：

```text
data/processed/ClinTox/train.jsonl
data/processed/ClinTox/test.jsonl

fields:
  drug: query SMILES
  Y: ClinTox label
```

当前数据统计：

```text
train: 1007 rows
test:  286 rows

Y=0: 1219
Y=1:   74
```

## Label 约定

MoleculeNet ClinTox 原始任务通常包含两个 label：

```text
FDA_APPROVED
CT_TOX
```

当前本地 processed JSONL 只保留单个 `Y`。本流程按已确认的本地 label 映射执行：

```text
Y=1 -> clintox_prediction=toxic
Y=0 -> clintox_prediction=non_toxic
```

如果后续数据准备脚本改变了本地 `Y` 含义，必须先更新本文件、`constants.py`、final prompt 和 batch
metrics，再运行正式评估。

final summary 建议二选一：

```text
clintox_prediction:
  toxic
  non_toxic

confidence:
  high
  moderate
  low
```

不要在 batch accuracy 和 macro-F1 中输出 `uncertain`。如果模型认为证据不足，用 `confidence=low`
和 `evidence_gaps` 表达。

由于 positive class 极少，batch report 除 overall accuracy / macro-F1 外，必须报告：

```text
positive_class_precision
positive_class_recall
positive_class_f1
confusion_matrix
prediction_distribution
```

## Legacy native runner 边界

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

旧 native ClinTox final 阶段不暴露工具，只综合 single-molecule analysis 和 endpoint-group outputs。当前
paper runner 使用 GLM-5.2 和通用 direct/mechanism-family orchestration，模型与可见性设置以
`tools/chembl_tool/paper_experiments/AGENTS.md` 为准。

## Task-specific 文件

当前实现：

```text
constants.py
  ClinTox label、prediction names、class mapping。

endpoint_groups.py
  ClinTox Tier.endpoint_group、evidence_direction、evidence_strength 规则。

experiment_config.py
  Paper-facing direct、full_flat 和 7-family full_mechanism retrieval view；不包含历史 native label policy。

rules.py
  ClinTox assay screening 关键词、negative keywords、weak/context-dependent terms、
  toxicology target genes 和 assay family 配置。

scoring.py
  ClinTox assay 保留/剔除和打分统一入口。screen_assays.py 和 rescore_outputs.py 都调用
  scored_row()。

run_reasoning_pipeline.py
  ClinTox prompt、single/group/final schema、retrieval/prompt assembly 和 final-only rerun。

run_reasoning_batch.py
  ClinTox 批量 reasoning wrapper。复用 common reasoning_batch.py，配置 input、batch root、
  neighbor index、pipeline module、prediction field 和 toxic/non_toxic label mapping。

audit_reasoning_batch.py
  ClinTox batch error audit。读取已有 batch 输出并生成 `audit/report.md`、`audit/audit_summary.json`、
  `audit/error_cases.csv` 和 `audit/group_direction_by_confusion.csv`；支持用 `--fill-missing-from`
  把 missing/error final-only 补跑 batch 按 query_index 合并进诊断，不修改原 batch 结果。
```

下面这些文件应是 task-specific 配置 wrapper，公共实现见根 `AGENTS.md` 的
`tools/chembl_tool/common/task_workflows/` 说明：

```text
screen_assays.py
rescore_outputs.py
summarize_outputs.py
report.py
build_evidence_library.py
retrieve_neighbors.py
chembl_exact_context.py
```

不要在 wrapper 中新增业务规则；ClinTox assay 保留/剔除逻辑只放在 `rules.py` 和 `scoring.py`，
endpoint-group 语义只放在 `endpoint_groups.py`。

## 资料依据

本文件的 evidence 类型设计参考了下面资料。后续如果补充或修改 ClinTox ontology，应优先查
FDA/ICH/OECD/NCATS/EPA/ChEMBL 等一手资料。

```text
ClinTox / MoleculeNet:
  https://huggingface.co/datasets/chao1224/MoleculeSTM/blob/main/MoleculeNet_data/clintox/CLINTOX_README
  https://scikit-fingerprints.readthedocs.io/v1.17.0/modules/datasets/generated/skfp.datasets.moleculenet.load_clintox.html

ChEMBL assay/activity semantics:
  https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions
  https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/target-questions
  https://chembl.gitbook.io/chembl-data-deposition-guide/file-structure/field-names-and-data-types-minimal-data-submission/activity.tsv/when-to-use-value-and-text_value

Tox21 / ToxCast:
  https://ncats.nih.gov/research/research-activities/Tox21/assays
  https://ncats.nih.gov/research/research-activities/Tox21/operational-model
  https://www.epa.gov/comptox-tools/exploring-toxcast-data

Clinical/nonclinical safety guidance:
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/m3r2-nonclinical-safety-studies-conduct-human-clinical-trials-and-marketing-authorization
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/drug-induced-liver-injury-premarketing-clinical-evaluation
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/e14-and-s7b-clinical-and-nonclinical-evaluation-qtqtc-interval-prolongation-and-proarrhythmic
  https://www.ema.europa.eu/en/ich-s7b-non-clinical-evaluation-potential-delayed-ventricular-repolarization-qt-interval-prolongation-human-pharmaceuticals-scientific-guideline
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/questions-and-answers-m7r2-assessment-and-control-dna-reactive-mutagenic-impurities-pharmaceuticals
  https://www.oecd.org/en/publications/test-no-471-bacterial-reverse-mutation-test_9789264071247-en.html

Organ/mechanistic toxicity examples:
  https://pubmed.ncbi.nlm.nih.gov/22120137/
  https://www.fda.gov/science-research/bioinformatics-tools/drug-induced-renal-injury-list-diril-dataset
  https://www.oecd.org/en/topics/sub-issues/testing-of-chemicals/in-vitro-assays-for-developmental-neurotoxicity.html
```

## ClinTox evidence 总原则

ClinTox 的 label 是临床开发/药品安全层面的结果，不是单一机制。ChEMBL 中的 assay evidence
应被视为不同距离的毒性证据：

```text
clinical or human safety evidence:
  最接近 label，但 ChEMBL 内可能稀疏，且可能来自药物标签、临床 observation 或文献摘要。

in vivo animal toxicology:
  比体外 proxy 更接近临床风险，但种属、暴露、给药途径、剂量和疗程会强烈影响解释。

organ-specific or safety-pharmacology assays:
  对特定 failure mode 有较强解释力，例如 hERG/QT、DILI/BSEP、renal tubular injury。

mechanistic stress pathway assays:
  能提示 risk mechanism，例如 p53/ATAD5/DNA damage、MMP/mitochondria、Nrf2/oxidative stress、
  ER stress、NF-kB/inflammation，但不能单独等价于 clinical toxicity。

general cytotoxicity / viability:
  是广义毒性 proxy，通常只能作为弱到中等证据。抗肿瘤、抗感染或靶向细胞杀伤 assay
  不能直接解释成临床毒性。

safety-relevant off-target pharmacology:
  例如 hERG、5-HT2B、AChE、monoamine transporters、CYP inhibition。只有当 target 和 assay
  endpoint 与安全风险有明确关系时才保留。
```

所有 group-level prompt 必须提醒 LLM：

```text
1. 相似分子上的 toxicity evidence 不是 query molecule 的直接标签。
2. dose/exposure/route/species/cell type/assay duration 决定 transferability。
3. distant_analog 和 very_distant_analog 不能作为强正负证据，除非共享 scaffold、toxicophore
   和 assay mechanism 都有很强药化理由。
4. "inhibition"、"activity"、"viability"、"growth"、"ratio" 等 endpoint 必须结合 assay
   context 解释，不能单独决定 evidence_direction。
5. hERG/QT、5-HT2B、AChE、GABA/NMDA、sodium/calcium channel、CYP 和 transporter evidence
   应先解释为 pharmacology / DDI / exposure / monitoring liability；只有同一 group 内有明确的严重
   clinical 或 in vivo toxicity endpoint 且 transferability 足够强时，才可上升为 clinical toxicity。
6. animal LD50/TD50/MTD/NOAEL 必须报告 species、route、dose、duration 和 endpoint severity；
   class pharmacology 或 therapeutic mechanism 本身不能替代 ClinTox 正类证据。
7. generic cytotoxicity / cell viability 必须区分 intended antiproliferative/anti-infective efficacy、
   nonspecific cell stress 和 safety cytotoxicity。缺少正常细胞/安全 counterscreen 或强 toxicophore
   解释时，不应直接作为 clinical toxicity。
```

## Evidence library 字段

ClinTox evidence library 每一条 activity evidence 至少保留：

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
assay_type
target_chembl_id
target_pref_name
target_type
target_genes
standard_type
standard_relation
standard_value
standard_units
pchembl_value
data_validity_comment
activity_comment
confidence_score
relationship_type
evidence_direction
evidence_strength
evidence_reason
```

内部可以保留 `evidence_direction`、`evidence_strength`、`endpoint_group_reason`、`assay_reason`
等规则派生字段，方便 debug 和审计；但这些字段不要发送给 reasoning LLM。LLM payload 中的
activity evidence row 只包含原始 ChEMBL assay/activity 字段和必要 metadata。

## 建议 evidence directions

```text
supports_clinical_toxicity
argues_against_clinical_toxicity
cardiotoxicity_risk
hepatotoxicity_risk
nephrotoxicity_risk
neurotoxicity_risk
genotoxicity_or_carcinogenicity_risk
mitochondrial_or_cell_stress_risk
general_cytotoxicity_risk
drug_interaction_or_exposure_risk
mechanistic_context
context_dependent
unknown_direction
```

## 建议 evidence strengths

```text
strong
moderate
weak
context_dependent
```

强弱不是 assay score 的同义词。它表示该 evidence type 对 ClinTox label 的解释距离：

```text
clinical toxicity / withdrawal / serious adverse event:
  strong, if molecule-level context is clear.

in vivo repeat-dose toxicity / NOAEL / MTD / DLT:
  strong to moderate, depending on dose/exposure/species.

hERG/QT, DILI/BSEP/mitochondrial hepatocyte toxicity, Ames/micronucleus:
  moderate to strong for the corresponding failure mode.

Tox21/ToxCast stress pathway, nuclear receptor, CYP inhibition:
  weak to moderate unless supported by organ-specific or clinical evidence.

generic cell viability/cytotoxicity:
  weak to moderate, context-dependent.
```

## Evidence 类型解释

ClinTox 的 evidence tier 不应按 BBB / bioavailability 的结构照搬。这里按与临床 toxicity
label 的距离、机制可解释性和 ChEMBL 可检索性来分层。

### Tier 1: clinical or human safety evidence

最接近 ClinTox label。优先保留明确发生在人类、临床研究、上市后或药品标签语境中的 toxicity
evidence。

建议 endpoint groups：

```text
clinical_toxicity_or_trial_failure
  clinical toxicity
  clinical trial toxicity
  toxicity in clinical trial
  dose limiting toxicity
  dlt
  treatment emergent adverse event
  serious adverse event
  severe adverse event
  adverse drug reaction
  discontinuation due to adverse event
  withdrawal due to toxicity
  black box warning
  boxed warning
  safety concern
  tolerability

human_maximum_tolerated_dose_or_safety_margin
  maximum tolerated dose
  mtd
  no observed adverse effect level
  noael
  lowest observed adverse effect level
  loael
  therapeutic index
  safety margin
  exposure margin
  tolerated dose
  tolerability limit

human_lab_toxicity_signal
  alt
  ast
  bilirubin
  alkaline phosphatase
  alp
  liver enzyme elevation
  creatinine
  blood urea nitrogen
  bun
  troponin
  qtc
  qt prolongation
  neutropenia
  thrombocytopenia
  myelosuppression
```

强解释条件：

```text
1. assay description 或 source 明确是 human / clinical / patient / volunteer / trial / postmarketing。
2. endpoint 是 adverse event、toxicity、tolerability、DLT、MTD、安全实验室指标或标签 warning。
3. activity row 能看出方向，例如 "toxic", "adverse", "elevated", "prolonged", "positive",
   "discontinued", "not tolerated"。
```

弱解释或剔除条件：

```text
1. 只有 efficacy adverse event reporting，不是 molecule-level toxicity assay。
2. oncology trial 中的 DLT/MTD 可作为 toxicity evidence，但要明确 oncology context；高毒性不一定
   泛化到非肿瘤适应症剂量。
3. "tolerated"、"safe"、"no adverse effect" 只有在剂量/暴露足够明确时才可作为反向证据。
```

### Tier 2: in vivo animal toxicology and toxicokinetic safety

动物 in vivo toxicology 比多数体外 assay 更接近 clinical toxicity，但 transferability 依赖种属、
route、dose、duration 和 exposure。

建议 endpoint groups：

```text
in_vivo_acute_or_repeat_dose_toxicity
  acute toxicity
  repeated dose toxicity
  repeat dose toxicity
  subacute toxicity
  subchronic toxicity
  chronic toxicity
  systemic toxicity
  oral toxicity
  intravenous toxicity
  dermal toxicity
  inhalation toxicity
  mortality
  morbidity
  clinical signs
  body weight loss

in_vivo_ld50_lc50_mtd_noael_loael
  ld50
  lc50
  td50
  mtd
  maximum tolerated dose
  noael
  loael
  no observed adverse effect
  lowest observed adverse effect
  minimum toxic dose
  toxic dose

in_vivo_histopathology_or_organ_weight
  histopathology
  pathology
  necrosis
  degeneration
  inflammation
  organ weight
  liver weight
  kidney weight
  spleen weight
  heart weight
  bone marrow

toxicokinetic_exposure_margin
  toxicokinetic
  tk
  auc at noael
  cmax at noael
  exposure margin
  safety margin
  dose proportional toxicity
```

解释规则：

```text
LD50 / LC50 / mortality:
  高剂量急性毒性 endpoint。对 clinical toxicity 有用，但不要过度外推到治疗剂量。

NOAEL / LOAEL / MTD:
  比单纯 LD50 更有药物开发意义。需要 dose、route、duration、species 和 toxicokinetic exposure。

histopathology / organ weight:
  可以支持器官 toxicity group，但 organ-specific 解释优先放到 Tier 3。

body weight loss / clinical signs:
  是广义 toxicity signal，通常 moderate 或 weak，除非与 dose-limiting toxicity 一起出现。
```

### Tier 3: organ-specific toxicity and safety pharmacology

这一级是 ClinTox 的核心。它们不等于 clinical toxicity label，但与常见 clinical hold / failure
mode 有明确机制联系。

#### Tier 3A: cardiotoxicity and electrophysiology

建议 endpoint groups：

```text
cardiac_herg_or_ikr_block
  herg
  hERG
  KCNH2
  IKr
  potassium channel
  potassium current
  patch clamp
  thallium influx
  tail current
  channel inhibition
  channel block

cardiac_qt_or_repolarization
  qt prolongation
  qtc prolongation
  delayed ventricular repolarization
  torsade
  torsades de pointes
  action potential duration
  apd
  field potential duration
  fpd
  cardiomyocyte electrophysiology

cardiac_contractility_or_cardiomyocyte_toxicity
  cardiomyocyte toxicity
  ipsc cardiomyocyte
  hiPSC-CM
  cardiac contractility
  beat rate
  beating
  calcium transient
  troponin
  creatine kinase mb
  cpk-mb
```

目标基因提示：

```text
KCNH2
SCN5A
CACNA1C
KCNQ1
KCNE1
RYR2
```

解释规则：

```text
hERG/KCNH2 inhibition:
  对 QT/proarrhythmia risk 是 strong mechanistic evidence。IC50/Ki 越低风险越高，但需要与
  exposure margin 结合；低 potency hERG inhibition 不应直接判为 clinical toxicity。

QT/QTc prolongation / APD / FPD:
  比单纯 hERG 更接近 phenotype。human或in vivo QT evidence 可强解释。

cardiomyocyte viability/contractility:
  支持 cardiotoxicity_risk，但需区分泛 cytotoxicity 和 cardiac-specific phenotype。
```

#### Tier 3B: hepatotoxicity, DILI and cholestasis

建议 endpoint groups：

```text
hepatic_dili_or_liver_injury
  drug induced liver injury
  dili
  hepatotoxicity
  liver toxicity
  hepatic toxicity
  hepatocellular injury
  cholestatic injury
  cholestasis
  hy's law
  liver enzyme elevation
  transaminase elevation
  alt
  ast
  bilirubin
  alkaline phosphatase
  alp

hepatic_bile_acid_transport_or_bsep
  bsep
  bile salt export pump
  ABCB11
  bile acid transport
  bile acid efflux
  taurocholate transport
  ntcp
  SLC10A1
  MRP2
  ABCC2
  MDR3
  ABCB4

hepatic_mitochondrial_or_oxidative_stress
  mitochondrial toxicity
  mitochondrial membrane potential
  mmp
  oxygen consumption rate
  ocr
  respiratory chain
  atp depletion
  ros
  reactive oxygen species
  oxidative stress
  glutathione depletion
  gsh depletion

hepatic_cell_injury_or_steatosis
  hepatocyte viability
  hepg2 cytotoxicity
  heparg cytotoxicity
  primary hepatocyte
  ldh release
  apoptosis
  necrosis
  steatosis
  lipid accumulation
  phospholipidosis
```

目标基因提示：

```text
ABCB11
SLC10A1
ABCC2
ABCB4
CYP3A4
CYP2C9
CYP2D6
CYP2C19
PXR/NR1I2
CAR/NR1I3
FXR/NR1H4
NFE2L2
```

解释规则：

```text
clinical ALT/AST/bilirubin/Hy's law:
  human clinical lab toxicity evidence，优先 Tier 1 clinical/human group。

BSEP/ABCB11 inhibition:
  可提示 cholestatic DILI risk，但单独 BSEP inhibition 不是 DILI 标签；需要 potency、exposure、
  hepatocyte toxicity、bile acid accumulation 或 clinical liver signal 支持。

mitochondrial toxicity / MMP loss / OCR decrease / ATP depletion:
  支持 hepatotoxicity 或 broader cell stress risk，尤其在 hepatocyte/HepG2/HepaRG/primary liver
  context 中。

HepG2/primary hepatocyte cytotoxicity:
  比 generic cell viability 更相关，但仍需区分 high-concentration nonspecific cytotoxicity。

phospholipidosis / steatosis:
  对 cationic amphiphilic drug liability 有用。通常是 mechanistic_context 或 moderate risk，
  不能单独等价于 clinical toxicity。
```

#### Tier 3C: nephrotoxicity and renal transporter injury

建议 endpoint groups：

```text
renal_tubular_injury_or_nephrotoxicity
  nephrotoxicity
  renal toxicity
  kidney toxicity
  renal tubular injury
  proximal tubule toxicity
  rptec
  hk-2
  kidney injury molecule
  kim-1
  havcr1
  ngal
  lcn2
  clusterin
  cystatin c
  nag
  bun
  creatinine
  albuminuria
  proteinuria

renal_transporter_accumulation_or_inhibition
  oat1
  oat3
  oct2
  mate1
  mate2-k
  organic anion transporter
  organic cation transporter
  renal uptake
  renal secretion
  transporter inhibition
```

目标基因提示：

```text
SLC22A6
SLC22A8
SLC22A2
SLC47A1
SLC47A2
HAVCR1
LCN2
```

解释规则：

```text
KIM-1/NGAL/clusterin/cystatin C/NAG:
  renal tubular injury biomarkers。human clinical biomarker evidence 强于 cell-line expression。

RPTEC/HK-2 viability or high-content injury:
  organ-relevant nephrotoxicity proxy，通常 moderate。

Renal transporter inhibition:
  更常说明 drug-drug interaction 或 exposure/creatinine handling 风险。只有和 renal accumulation
  或 tubular injury context 一起出现时才解释为 nephrotoxicity_risk。
```

#### Tier 3D: neurotoxicity and seizure/CNS safety

建议 endpoint groups：

```text
neurotoxicity_or_neuronal_viability
  neurotoxicity
  neuronal toxicity
  neural toxicity
  neuron viability
  sh-sy5y toxicity
  pc12 toxicity
  primary neuron
  ipsc neuron
  apoptosis
  neurite outgrowth
  synaptogenesis
  neural progenitor

neurofunctional_or_seizure_liability
  seizure
  convulsion
  epileptiform
  microelectrode array
  mea
  neuronal firing
  burst rate
  calcium oscillation
  gaba
  sodium channel
  nmda
  ampa
  acetylcholinesterase
  ache

developmental_neurotoxicity
  developmental neurotoxicity
  dnt
  neural differentiation
  neural migration
  oligodendrocyte
  myelination
  neurite outgrowth
```

目标基因提示：

```text
ACHE
GABRA1
GABRA2
GRIN1
GRIN2A
SCN1A
SCN2A
CACNA1A
SLC6A3
SLC6A4
DRD2
HTR2A
```

解释规则：

```text
neurite outgrowth / neural differentiation:
  对 developmental neurotoxicity 有机制意义，但不等同于 adult clinical toxicity。

MEA / neuronal firing / seizure phenotype:
  比单一 receptor binding 更接近 functional neurotoxicity。

receptor/channel binding:
  只有当 target 与 seizure、sedation、CNS adverse event 或 neurodevelopment 机制明确相关时保留。
```

#### Tier 3E: hematologic and immunotoxicity

建议 endpoint groups：

```text
hematologic_toxicity_or_myelosuppression
  myelosuppression
  bone marrow toxicity
  neutropenia
  agranulocytosis
  thrombocytopenia
  anemia
  leukopenia
  colony forming unit
  cfu-gm
  hematopoietic progenitor

immunotoxicity_or_cytokine_release
  immunotoxicity
  cytokine release
  il-6
  tnf alpha
  il-8
  immune activation
  complement activation
  hypersensitivity
  hla
```

解释规则：

```text
hematopoietic progenitor / CFU-GM inhibition:
  对 myelosuppression 有较强机制相关性。

generic immune cell viability:
  通常 weak/context-dependent。需要 cytokine release、immune activation 或 clinical hematologic
  signal 支持。
```

#### Tier 3F: reproductive, developmental and endocrine toxicity

建议 endpoint groups：

```text
reproductive_or_developmental_toxicity
  reproductive toxicity
  developmental toxicity
  embryotoxicity
  teratogenicity
  fetal toxicity
  embryo
  zebrafish embryo
  embryonic stem cell test
  est
  blastocyst
  differentiation inhibition

endocrine_nuclear_receptor_disruption
  estrogen receptor
  er alpha
  androgen receptor
  ar
  aromatase
  thyroid receptor
  tr beta
  progesterone receptor
  glucocorticoid receptor
  mineralocorticoid receptor
  ppar gamma
  ppar alpha
  ah receptor
  ahr
```

目标基因提示：

```text
ESR1
ESR2
AR
CYP19A1
THRB
PGR
NR3C1
PPARG
PPARA
AHR
```

解释规则：

```text
in vivo developmental/reproductive toxicity:
  strong to moderate, depending on species/dose/exposure.

ER/AR/aromatase/thyroid reporter assays:
  endocrine disruption evidence。对 ClinTox label 是 mechanistic_context，除非有 reproductive,
  developmental, clinical endocrine or hormonal adverse evidence。

zebrafish embryo toxicity:
  useful broad developmental toxicity proxy, but species and exposure limitations must be stated.
```

### Tier 4: genotoxicity, mutagenicity and carcinogenicity

Genotoxicity 对药物开发风险很重要，尤其 DNA-reactive mutagenicity。它不一定直接对应 ClinTox
clinical trial toxicity，但对 developability 和 regulatory safety 是强风险信号。

建议 endpoint groups：

```text
ames_or_bacterial_mutagenicity
  ames
  bacterial reverse mutation
  salmonella
  typhimurium
  e coli wp2
  s9
  mutagenicity
  revertant
  frameshift mutation
  base pair substitution

in_vitro_mammalian_genotoxicity
  micronucleus
  chromosomal aberration
  chromosome aberration
  mouse lymphoma
  tk assay
  hprt
  comet assay
  dna strand break
  sister chromatid exchange

dna_damage_response
  p53
  p53 response
  gamma h2ax
  h2ax
  atad5
  dna damage
  dna repair
  topoisomerase poison

carcinogenicity_or_tumorigenicity
  carcinogenicity
  tumorigenicity
  carcinogen
  td50
  tumor incidence
  ras transformation
  cell transformation
```

解释规则：

```text
Ames positive:
  strong genotoxicity_or_carcinogenicity_risk，尤其有 dose-response、S9 +/- 信息和明确 positive
  activity_comment。

Micronucleus/chromosomal aberration/mouse lymphoma:
  moderate to strong。需区分 cytotoxic concentrations 下的 secondary DNA damage。

p53/ATAD5/gamma-H2AX:
  mechanistic DNA damage response。通常 moderate，除非与 Ames/micronucleus/clinical evidence 一致。

Carcinogenicity:
  in vivo carcinogenicity strong，但 species, dose and chronic exposure matter.
```

### Tier 5: Tox21/ToxCast stress pathways and mechanistic cellular toxicity

这类 assay 适合覆盖 ClinTox 的机制空间，但大多数不能单独作为 clinical toxicity 结论。

建议 endpoint groups：

```text
mitochondrial_toxicity_or_energy_stress
  mitochondrial membrane potential
  mmp
  mitochondria
  mitochondrial toxicity
  oxygen consumption
  ocr
  spare respiratory capacity
  atp
  cellular atp
  uncoupler

oxidative_stress_or_nrf2_are
  oxidative stress
  reactive oxygen species
  ros
  nrf2
  nfe2l2
  antioxidant response element
  are
  glutathione
  gsh

er_stress_heat_shock_unfolded_protein
  er stress
  endoplasmic reticulum stress
  unfolded protein response
  upr
  heat shock response
  hse
  hsp70
  hsp90
  hsf1

inflammatory_or_stress_transcription
  nf-kb
  nfkb
  ap-1
  hif-1
  hif1a
  creb
  stat
  il-8
  tnf alpha
  cytokine

lysosomal_or_phospholipidosis
  phospholipidosis
  lipidtox
  nbd-pe
  lysotracker
  lysosomal accumulation
  lysosomal toxicity
  cationic amphiphilic

general_cell_stress_or_morphology
  high content screening
  cellular morphology
  nuclear size
  cell count
  membrane integrity
  ldh
  apoptosis
  caspase
  necrosis
```

解释规则：

```text
MMP loss / ATP depletion:
  broad mitochondrial liability. More meaningful if observed below nonspecific cytotoxic concentrations.

ARE/Nrf2, ROS, GSH depletion:
  oxidative stress evidence. Supports mechanistic_context or mitochondrial_or_cell_stress_risk.

ER stress / HSE / UPR:
  cellular stress evidence. Usually weak to moderate unless organ-specific.

NF-kB/AP-1/cytokines:
  inflammation/immunomodulation context. Avoid over-interpreting as direct clinical toxicity.

Phospholipidosis:
  screen for cationic amphiphilic drug liability; interpret with physicochemical properties and organ context.
```

### Tier 6: general cytotoxicity, viability and proliferation

广义 cytotoxicity 是 ChEMBL 中最容易误收的证据。它可以帮助判断 structural analog 的 general
toxicity liability，但对 ClinTox label 的距离较远。

建议 endpoint groups：

```text
general_cytotoxicity_or_viability
  cytotoxicity
  cell viability
  cell survival
  cell death
  viability
  survival
  toxic concentration
  cc50
  tc50
  lc50
  ic50
  gi50
  ld50
  ec50
  mtt
  mts
  xtt
  wst-1
  wst-8
  resazurin
  alamar blue
  celltiter-glo
  atp content
  ldh release
  neutral red uptake
  propidium iodide
  caspase

antiproliferative_or_growth_inhibition
  growth inhibition
  proliferation
  antiproliferative
  cell growth
  gi50
  tgi
  colony formation
```

解释规则：

```text
CC50/TC50/LC50:
  stronger than generic "activity" if assay is normal cell toxicity or broad cell panel.

IC50/GI50:
  often efficacy or antiproliferative pharmacology, especially oncology cell lines. Do not treat as clinical toxicity
  unless the assay context is toxicity/safety or normal-cell selectivity.

MTT/MTS/XTT/WST/resazurin/CellTiter-Glo:
  assay readout technologies, not endpoint semantics by themselves.

LDH/caspase/annexin/PI:
  stronger evidence of cell injury/death than metabolic viability alone, but still usually proxy evidence.
```

必须降权或剔除：

```text
1. antitumor / anticancer efficacy assays in cancer cell lines without normal-cell toxicity context.
2. antimicrobial MIC / parasite growth inhibition / antiviral cytopathic-effect assays unless toxicity context is explicit.
3. target engagement assays where "inhibition" means desired pharmacology, not host toxicity.
```

### Tier 7: safety-relevant off-target pharmacology and DDI/exposure risk

这一级包含可能导致 clinical safety issue 的 off-target 或 ADME liability。它们通常是机制 evidence，
不直接等价于 ClinTox positive。

建议 endpoint groups：

```text
cardiac_or_cns_offtarget_binding
  5-ht2b
  htr2b
  dopamine transporter
  dat
  slc6a3
  serotonin transporter
  sert
  slc6a4
  norepinephrine transporter
  net
  slc6a2
  gaba receptor
  nmda receptor
  sodium channel
  calcium channel

cytochrome_p450_inhibition_or_induction
  cyp inhibition
  cyp induction
  cyp3a4
  cyp2d6
  cyp2c9
  cyp2c19
  cyp1a2
  time dependent inhibition
  tdi
  mechanism based inhibition
  mbi

transporter_ddi_or_exposure_liability
  p-gp
  abcb1
  bcrp
  abcg2
  oatp1b1
  slco1b1
  oatp1b3
  slco1b3
  oat1
  oat3
  oct2
  mate1
  bsep

reactive_metabolite_or_covalent_liability
  reactive metabolite
  glutathione adduct
  gsh adduct
  covalent binding
  protein adduct
  bioactivation
  quinone imine
  acyl glucuronide
```

解释规则：

```text
5-HT2B agonism:
  safety-relevant because of valvulopathy concern. Binding alone is weaker than functional agonism.

CYP inhibition/induction:
  DDI/exposure risk, not direct toxicity. It can amplify toxicity if paired with narrow therapeutic index,
  high exposure, or toxic metabolite evidence.

Transporter inhibition:
  DDI/exposure or organ accumulation risk. BSEP is also hepatotoxicity-relevant and should be routed to Tier 3B
  when bile acid/cholestasis context is present.

Reactive metabolites/GSH adducts:
  support idiosyncratic toxicity or DILI risk, but need metabolic context and concentration.
```

### Tier 8: context-dependent or weak evidence

下面 endpoint 或 assay words 不能单独强解释：

```text
activity
inhibition
activation
ratio
response
signal
fluorescence
rfu
luminescence
absorbance
binding
displacement
substrate
uptake
transport
fold change
fc
percent effect
efficacy
potency
```

如果这些词出现在明确 toxicity context 中，可以被 endpoint group 规则提升；否则标记为：

```text
endpoint_group: context_dependent
evidence_strength: weak
evidence_direction: context_dependent
```

## Assay screening 关键词建议

`rules.py` 可以先按 assay families 组织关键词，不必一次性完美。第一版建议覆盖：

```text
clinical_safety:
  toxicity, adverse event, serious adverse event, dose limiting toxicity, maximum tolerated dose,
  tolerability, clinical safety, withdrawal, black box, boxed warning

in_vivo_toxicology:
  acute toxicity, repeated dose toxicity, subchronic toxicity, chronic toxicity, LD50, LC50,
  NOAEL, LOAEL, MTD, histopathology, organ weight, mortality

cardiotoxicity:
  hERG, KCNH2, IKr, QT, QTc, repolarization, torsade, action potential duration,
  cardiomyocyte, contractility, troponin

hepatotoxicity:
  DILI, hepatotoxicity, liver injury, cholestasis, BSEP, ABCB11, bile acid, ALT, AST,
  bilirubin, HepG2, HepaRG, primary hepatocyte, mitochondrial toxicity, glutathione

nephrotoxicity:
  nephrotoxicity, renal toxicity, kidney toxicity, proximal tubule, RPTEC, HK-2, KIM-1,
  NGAL, creatinine, BUN

neurotoxicity:
  neurotoxicity, neurite outgrowth, neural differentiation, neuronal viability, MEA,
  seizure, acetylcholinesterase, GABA, sodium channel

genotoxicity:
  Ames, bacterial reverse mutation, micronucleus, chromosomal aberration, mouse lymphoma,
  comet, gamma-H2AX, ATAD5, p53, DNA damage

stress_pathway:
  Tox21, ToxCast, Nrf2, ARE, oxidative stress, ROS, mitochondrial membrane potential,
  ER stress, heat shock, NF-kB, AP-1, HIF-1

general_cytotoxicity:
  cytotoxicity, cell viability, cell survival, CC50, TC50, LC50, LDH, apoptosis,
  necrosis, caspase, CellTiter-Glo, MTT, resazurin

offtarget_ddi:
  CYP inhibition, CYP induction, time-dependent inhibition, mechanism-based inhibition,
  5-HT2B, HTR2B, P-gp, BCRP, OATP, OCT2, MATE, reactive metabolite, GSH adduct
```

## Negative keywords and false-positive traps

ClinTox screening 必须有 aggressive negative rules，否则 ChEMBL 会收进大量 efficacy assay。

常见剔除或降权词：

```text
antitumor
anticancer
anti-proliferative against cancer
xenograft efficacy
tumor growth inhibition
kinase inhibition
enzyme inhibition
receptor binding
antimicrobial
antibacterial
antifungal
antiviral
antimalarial
antiparasitic
MIC
minimum inhibitory concentration
parasite growth
viral replication
cytopathic effect
plaque reduction
insecticidal
herbicidal
plant growth
```

注意：

```text
1. "cytopathic effect" 在 antiviral assay 中经常是 viral efficacy readout，不一定是 host toxicity。
2. "growth inhibition" 在 cancer cell line 中通常是 efficacy，不是 clinical toxicity。
3. "inhibition" 对 CYP/hERG/BSEP/AChE 等 safety target 有意义；对 therapeutic target 通常不是毒性。
4. "activity_comment=active" 只能说明 assay positive，不能说明 toxic positive。
```

## Endpoint group assignment 原则

`endpoint_group` 不能只看 `standard_type`。必须联合：

```text
assay_tier
standard_type
assay_description
assay_type
target_pref_name
target_genes
activity_comment
confidence_score
relationship_type
```

优先级建议：

```text
1. 明确 clinical/human safety context -> Tier 1
2. 明确 in vivo animal toxicology -> Tier 2
3. 明确 organ-specific toxicity or safety pharmacology -> Tier 3
4. 明确 genotoxicity/carcinogenicity -> Tier 4
5. Tox21/ToxCast/mechanistic stress pathway -> Tier 5
6. generic cytotoxicity/viability/proliferation -> Tier 6
7. safety-relevant off-target / DDI / exposure -> Tier 7
8. otherwise context_dependent
```

如果一个 assay 同时命中多个 group，选择更接近 clinical toxicity label 的 group。例如：

```text
BSEP inhibition in cholestasis assay:
  hepatic_bile_acid_transport_or_bsep, not generic transporter inhibition.

hERG thallium influx:
  cardiac_herg_or_ikr_block, not generic ion channel activity.

HepG2 CellTiter-Glo:
  hepatic_cell_injury_or_steatosis if hepatocyte toxicity context is explicit;
  otherwise general_cytotoxicity_or_viability.

ATAD5 luciferase:
  dna_damage_response, not generic reporter activity.
```

## Legacy native reasoning pipeline

旧 ClinTox native reasoning 与旧 BBB / Bioavailability_Ma runner 保持相同三段式：

```text
1. 读取 ClinTox test.jsonl 的 query molecule。
2. 调用 retrieve_neighbors.py 预取每个 source-local Tier.endpoint_group 的 ChEMBL neighbor evidence。
3. 并发执行 single-molecule analysis；该历史 runner 中 DeepSeek 只可调用 molecule_properties。
4. 并发执行 endpoint-group analysis；该历史 runner 中 DeepSeek 只可调用 mmp_structure_compare 和 properties_compare。
5. final summary 读取 single + all group outputs，不暴露任何 tool。
6. 保存 retrieval/single/group/final/trace/manifest。
```

single-molecule analysis 应关注：

```text
structural alert prior:
  electrophilic/reactive groups, Michael acceptors, anilines, nitro aromatics, quinones,
  hydrazines, acyl halides, epoxides, aldehydes, polyhalogenated lipophilic motifs.

physicochemical toxicity prior:
  high lipophilicity, high basicity/cationic amphiphilicity, high aromatic ring count,
  poor solubility, high molecular weight, high polar surface area, permanent charge,
  likely lysosomal trapping.

ADME/exposure prior:
  properties that may increase systemic exposure, tissue accumulation or low clearance risk.
```

这些 single-molecule priors 只是先验，不应替代 ChEMBL evidence。

默认 single-molecule prompt 不包含任何 ChEMBL 相关 payload 或 instruction。只有开启 exact context 且命中 query exact context 时，single-molecule payload 才包含 `exact_query_chembl_context`，并提示模型区分 direct same-molecule ChEMBL clinical-toxicity evidence 和 physicochemical prior。ChEMBL neighbor evidence 仍只进入 group-level context。

## Group-level reasoning output schema

建议 group output：

```text
group_id
useful_for_clintox_reasoning: true/false
transferability:
  high
  moderate
  low
  not_applicable
evidence_direction:
  supports_clinical_toxicity
  argues_against_clinical_toxicity
  cardiotoxicity_risk
  hepatotoxicity_risk
  nephrotoxicity_risk
  neurotoxicity_risk
  genotoxicity_or_carcinogenicity_risk
  mitochondrial_or_cell_stress_risk
  general_cytotoxicity_risk
  drug_interaction_or_exposure_risk
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
  effect_on_clintox_reasoning
caveats
```

`key_evidence` 是当前格式。不要使用旧的 `key_neighbors`。

## Final reasoning output schema

建议 final output：

```text
clintox_prediction:
  toxic
  non_toxic

confidence:
  high
  moderate
  low

main_reasons
clinical_or_human_safety_assessment
in_vivo_toxicology_assessment
cardiotoxicity_assessment
hepatotoxicity_assessment
nephrotoxicity_assessment
neurotoxicity_assessment
genotoxicity_or_carcinogenicity_assessment
mechanistic_cell_stress_assessment
general_cytotoxicity_assessment
offtarget_ddi_or_exposure_assessment
evidence_gaps
final_summary
```

final prompt 规则：

```text
Return compact complete JSON.
Use clintox_prediction='toxic' for ClinTox-positive molecules corresponding to evaluation label 1,
and clintox_prediction='non_toxic' for ClinTox-negative molecules corresponding to evaluation label 0.
Use the single-molecule analysis only as a physicochemical plausibility prior; it cannot by itself determine
clintox_prediction.
Use group analyses as analog evidence; downweight groups marked low confidence or low transferability.
Interpret ClinTox-positive as severe or development-relevant clinical toxicity, not broad
medicinal-chemistry toxicity risk.
Distinguish severe clinical toxicity evidence from ordinary adverse-effect, safety-liability, or mechanistic
concern. Safety-liability signals can support toxic only when they are strong, transferable, clinically
consequential, and not contradicted by more direct evidence.
Severe human toxicity, clinical trial toxicity or termination, withdrawal/black-box-level toxicity, fatal or
severe in vivo toxicity, strong genotoxic/carcinogenic liability, or close-analog evidence of severe organ
injury can support clintox_prediction='toxic'.
Do not treat ordinary adverse effects, routine liver enzyme elevation, weak or moderate DILI concern, mild
animal toxicity, generic in vitro cytotoxicity, or broad safety warnings as sufficient for
clintox_prediction='toxic' by themselves.
hERG block, transporter inhibition, CYP inhibition, receptor binding, nuclear receptor activity, and indirect
mechanistic assays are safety-liability concerns unless the group analysis explains why potency,
transferability, and clinical relevance are strong enough to support clinical toxicity.
Do not convert drug_interaction_or_exposure_risk or mechanistic_context into clintox_prediction='toxic' by itself.
When evidence is mixed, weigh severity, directness, transferability, and assay relevance together. Do not let
many low-severity liability signals outvote more direct negative or non-severe evidence.
Do not use distant_analog or very_distant_analog neighbors as positive or negative clinical toxicity evidence
unless the shared scaffold and assay mechanism make a strong medicinal chemistry case.
Use only the provided single-molecule analysis and group evidence. If you recognize the molecule or therapeutic
class, ignore that recognition.
```

## Exact ChEMBL context

`chembl_exact_context.py` 是可选 evidence-rich 增强。它会用 query full InChIKey 查 ChEMBL exact
molecule，并在 retrieved neighbor 涉及的 assay 中查 query activity。

默认 benchmark 不开启，避免 prospective evaluation 数据泄漏。只有显式传：

```bash
--enable-chembl-exact-context
```

才允许用于 retrospective / evidence-rich case study。默认 batch accuracy 和 macro-F1 报告都应
保持该开关关闭。

## 输出目录

ClinTox 输出目录：

```text
outputs/chembl_tool/tasks/clintox/
  assay_screening/
  evidence_library/
  reasoning/
    single_runs/
    batches/
```

当前文件命名：

```text
outputs/chembl_tool/tasks/clintox/assay_screening/<version>/
  clintox_assay_candidates.csv
  clintox_assay_candidates.jsonl
  clintox_assay_report.md
  clintox_health_check.md
  clintox_activity_evidence.csv

outputs/chembl_tool/tasks/clintox/evidence_library/
  clintox_molecule_evidence.jsonl
  clintox_neighbor_index.pkl
  clintox_neighbor_index.meta.json
```

## 当前实现状态

当前主版本是 assay screening `v6`：

```text
outputs/chembl_tool/tasks/clintox/assay_screening/v6/
  clintox_assay_candidates.csv
  clintox_assay_candidates.jsonl
  clintox_assay_report.md
  clintox_health_check.md
  clintox_activity_evidence.csv

candidate assays: 108,623
activity evidence rows used for library: 2,248,401
Tier distribution:
  Tier 1:    118
  Tier 2: 11,201
  Tier 3: 57,602
  Tier 4:  6,338
  Tier 5:  2,457
  Tier 6:    753
  Tier 7: 30,154
```

当前 evidence library / neighbor index 已构建：

```text
outputs/chembl_tool/tasks/clintox/evidence_library/clintox_molecule_evidence.jsonl
outputs/chembl_tool/tasks/clintox/evidence_library/clintox_neighbor_index.pkl
outputs/chembl_tool/tasks/clintox/evidence_library/clintox_neighbor_index.meta.json

n_evidence_rows:   2,248,401
n_index_molecules:   553,919
n_groups:                  43
index_version: clintox_neighbor_index.v6
workers used for build: 128
elapsed_s: about 1106
```

已做过的 smoke / pilot：

```text
outputs/chembl_tool/tasks/clintox/retrieval_smoke.jsonl
  3 ClinTox test molecules, all status=ok, each with nonempty group retrieval.

outputs/chembl_tool/tasks/clintox/reasoning/single_runs/clintox_smoke_v1/
  query_index=0, 2 groups, label 0, prediction non_toxic.

outputs/chembl_tool/tasks/clintox/reasoning/single_runs/clintox_smoke_positive_targeted_v1/
  query_index=36, targeted DILI/DNA damage/cytotoxicity groups, label 1, prediction toxic.

outputs/chembl_tool/tasks/clintox/reasoning/batches/clintox_pilot_v1/
  pilot indices include 0, 1, 36, 43, 73.
```

## 常用命令

只重新过滤 activity evidence：

```bash
python -m tools.chembl_tool.tasks.clintox.rescore_outputs \
  --in-dir outputs/chembl_tool/tasks/clintox/assay_screening/raw \
  --out-dir outputs/chembl_tool/tasks/clintox/assay_screening/v6 \
  --only-filter-activities
```

构建 evidence library / neighbor index：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.clintox.build_evidence_library \
  --workers 128 \
  --progress-every 50000
```

检索 smoke：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.clintox.retrieve_neighbors \
  --query-jsonl data/processed/ClinTox/test.jsonl \
  --limit 3 \
  --top-k-per-group 3 \
  --min-similarity 0.3 \
  --out outputs/chembl_tool/tasks/clintox/retrieval_smoke.jsonl
```

单分子 reasoning：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.clintox.run_reasoning_pipeline \
  --query-index 0 \
  --top-k-per-group 3 \
  --min-similarity 0.3 \
  --max-workers 4 \
  --max-tool-rounds 8 \
  --timeout-s 300 \
  --max-tokens 8192 \
  --model deepseek-v4-pro \
  --run-id <run_id>
```

针对少数组做 positive smoke：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.clintox.run_reasoning_pipeline \
  --query-index 36 \
  --groups "Tier 3.hepatic_dili_or_liver_injury" "Tier 4.dna_damage_response" "Tier 6.general_cytotoxicity_or_viability" \
  --top-k-per-group 3 \
  --min-similarity 0.3 \
  --max-workers 3 \
  --max-tool-rounds 8 \
  --timeout-s 300 \
  --max-tokens 8192 \
  --model deepseek-v4-pro \
  --run-id <run_id>
```

小 batch / pilot：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.clintox.run_reasoning_batch \
  --input-jsonl data/processed/ClinTox/test.jsonl \
  --indices 0 1 36 43 73 \
  --parallelism 1 \
  --group-workers 4 \
  --top-k-per-group 3 \
  --min-similarity 0.3 \
  --max-tool-rounds 8 \
  --timeout-s 300 \
  --max-tokens 8192 \
  --model deepseek-v4-pro \
  --batch-id clintox_pilot_v1
```

查看 ClinTox trace：

```bash
bash tools/trace_viewer/start_viewer.sh \
  outputs/chembl_tool/tasks/clintox/reasoning/single_runs \
  8776

bash tools/trace_viewer/start_viewer.sh \
  outputs/chembl_tool/tasks/clintox/reasoning/batches \
  8776
```

ClinTox group output 的 key evidence 使用 `effect_on_clintox_reasoning`。如果后续新增 task
或新增 task-specific effect 字段，必须同步检查 `tools/trace_viewer/viewer.html`，否则 trace
JSON 中字段存在但 viewer 的 Key Evidence 列可能显示为空。
