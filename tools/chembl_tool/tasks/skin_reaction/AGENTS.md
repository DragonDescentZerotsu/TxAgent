# Skin_Reaction task notes

本文件记录 Skin_Reaction 的 task-specific 语义、ChEMBL evidence ontology、assay screening 规则方向、
endpoint group 设计和 reasoning 约束。通用 ChEMBL workflow、batch/resume、viewer 和输出目录规范仍以仓库根
`AGENTS.md` 为准。

## Task 定义

目标不是训练一个单纯的 QSAR skin-reaction classifier，而是构建可审计的 skin-reaction evidence retrieval
和 reasoning workflow：给定 query molecule，从 ChEMBL 中检索与皮肤不良反应判断相关的相似分子实验读数，
再由 reasoning LLM 判断这些 analog evidence 是否能 transfer 到 query molecule。

当前本地数据：

```text
data/processed/Skin_Reaction/train.jsonl
data/processed/Skin_Reaction/test.jsonl

fields:
  drug: query SMILES
  Y: Skin_Reaction label
```

当前评估约定先按二分类处理：

```text
Y=1 -> skin reaction risk / positive
Y=0 -> no skin reaction risk / negative
```

当前 ChEMBL evidence 版本：

```text
assay screening raw:
  outputs/chembl_tool/tasks/skin_reaction/assay_screening/raw/

assay screening v1:
  outputs/chembl_tool/tasks/skin_reaction/assay_screening/v1/skin_reaction_assay_candidates.csv
  outputs/chembl_tool/tasks/skin_reaction/assay_screening/v1/skin_reaction_activity_evidence.csv
  outputs/chembl_tool/tasks/skin_reaction/assay_screening/v1/skin_reaction_health_check.md

evidence library:
  outputs/chembl_tool/tasks/skin_reaction/evidence_library/skin_reaction_molecule_evidence.jsonl
  outputs/chembl_tool/tasks/skin_reaction/evidence_library/skin_reaction_neighbor_index.pkl
  outputs/chembl_tool/tasks/skin_reaction/evidence_library/skin_reaction_neighbor_index.meta.json
```

当前 v1 screening 状态：

```text
retained assays: 3,648
activity evidence rows: 18,299
index molecules: 8,428
retrieval groups: 21

Tier 1 direct anchors: 326 assays
Tier 2 sensitisation AOP key-event: 303 assays
Tier 3 phototoxicity / irritation / local skin damage: 1,308 assays
Tier 4 skin exposure modifiers: 62 assays
Tier 5 weak/context background: 1,649 assays
```

## 当前代码入口和运行状态

主要入口：

```text
constants.py
  label / prediction mapping。当前二分类约定：Y=1 -> risk，Y=0 -> no_risk。

rules.py
  Skin_Reaction assay keyword、negative keyword、context/weak evidence family 配置。

scoring.py
  assay screening / rescore 的保留、剔除和打分入口。

endpoint_groups.py
  Tier.endpoint_group、evidence_direction、evidence_strength 和 endpoint assignment 规则。

screen_assays.py / rescore_outputs.py / summarize_outputs.py / report.py
  thin wrappers，复用 common task workflow 生成候选 assay、activity evidence、health check 和 report。

build_evidence_library.py
  从 v1 assay candidates + activity evidence 构建 molecule-level evidence library 和 neighbor index。

retrieve_neighbors.py
  对每个 Tier.endpoint_group 做 analog retrieval。当前 benchmark 使用 top-k-per-group=3、
  min-similarity=0.35；min-similarity=0 的排查显示 index/retrieval 正常，低覆盖主要来自
  chemical-space similarity threshold。

chembl_exact_context.py
  exact-query ChEMBL context wrapper。benchmark 默认不开启，避免 retrospective leakage。

run_reasoning_pipeline.py
  单分子 reasoning pipeline：retrieval prefetch、single-molecule branch、group-level 并发 reasoning、
  final summary、trace 保存，以及 --resume-final-from-run-dir final-only rerun。

run_reasoning_batch.py
  批量 reasoning wrapper。复用 common reasoning_batch.py，输出 predictions、metrics、report、logs、
  runs 和 combined trace；支持 --skip-existing 断点续跑。
```

当前 v1 全量 test 结果：

```text
batch:
  outputs/chembl_tool/tasks/skin_reaction/reasoning/batches/skin_reaction_calib_50_v1

run settings:
  input=data/processed/Skin_Reaction/test.jsonl
  indices=0-81
  parallelism=3
  group-workers=20
  top-k-per-group=3
  min-similarity=0.35

metrics after idx00074 final-only rerun:
  n_total=82
  n_evaluable=82
  n_successful=82
  n_failed_runs=0
  accuracy=0.682927
  macro-F1=0.678140
  positive precision=0.733333
  positive recall=0.702128
  positive F1=0.717391
  confusion matrix: TN=23 FP=12 FN=14 TP=33
  prediction distribution: no_risk=37 risk=45
```

当前 v1 规则已主动排除：

```text
generic PubChem/Tox21 Nrf2 assays without HaCaT/keratinocyte/ARE-luc/sensitisation context
EGFR/FGFR/other kinase biochemical assays triggered only by target names such as epidermal/fibroblast
PAF-induced vascular permeability assays misread as skin permeability
anti-inflammatory / dermatology efficacy in reconstructed human epidermis models
permeation-enhancer assays where the tested molecule promotes another compound's transdermal permeation
```

后续必须确认数据集原始定义。如果原始 label 指的是 clinical dermatologic adverse reaction，而不是
chemical skin sensitisation，则 ChEMBL 的 sensitisation / irritation / phototoxicity evidence 只能作为
mechanistic hazard evidence，不能被等同于 clinical label。

## Skin_Reaction evidence 原则

Skin reaction 不是一个单一 assay endpoint。ChEMBL 里有价值的 evidence 大致分成三条轴：

```text
hazard axis:
  compound 是否能引发皮肤致敏、刺激、腐蚀、光毒性或局部细胞损伤。

mechanism axis:
  是否命中皮肤致敏 AOP 中的关键事件，例如蛋白共价结合、角质细胞激活、树突状细胞激活。

exposure axis:
  compound 是否能进入、滞留或穿透皮肤，从而让 hazard 有机会表现出来。
```

重要约束：

```text
1. skin permeability / dermal absorption 不是 skin reaction hazard。
   它只能增强或削弱 exposure plausibility，不能单独支持 Y=1。

2. general cytotoxicity 不是 skin reaction hazard。
   非皮肤细胞系的 CC50 / GI50 / viability 只能作为 weak background，不应单独决定 positive。

3. dermatology efficacy 不是 skin reaction hazard。
   抗炎、抗菌、抗银屑病、抗痤疮、melanoma efficacy、skin whitening、wound healing 等治疗性 assay
   不能当作 adverse skin reaction evidence。

4. target binding / enzyme inhibition 通常不是 skin reaction evidence。
   除非 assay description 明确指向 skin sensitisation、irritation、phototoxicity、keratinocyte /
   dendritic-cell activation、haptenation 或 dermal toxicity。

5. validated skin sensitisation AOP assays 的一致阳性 evidence 比单个弱 cytotoxicity assay 更重要。
   DPRA/ADRA/kDPRA、KeratinoSens/LuSens/EpiSensA、h-CLAT/U-SENS/IL-8 Luc/GARDskin 对应不同
   key event；多 key-event 一致时 evidence strength 应上调。
```

## Evidence families and endpoint groups

Skin_Reaction 不应照搬 BBB / Bioavailability 的 tier 语义。这里的 `assay_tier` 应表示 skin-reaction
reasoning 价值，而不是体内/体外距离的简单排序。

建议初版分层：

```text
Tier 1: direct skin reaction anchors
Tier 2: validated skin sensitisation AOP key-event assays
Tier 3: phototoxicity and local skin irritation / corrosion / dermal toxicity
Tier 4: skin exposure, barrier penetration, and local context modifiers
Tier 5: weak or context-dependent background
```

### Tier 1: direct skin reaction anchors

这些是最接近 Skin_Reaction label 的 evidence。命中时应优先保留。

Endpoint groups:

```text
human_patch_or_clinical_skin_reaction
  human patch test
  human maximization test
  repeated insult patch test
  HRIPT / RIPT
  allergic contact dermatitis
  contact allergy
  skin rash / erythema / edema / pruritus when explicitly adverse
  clinical skin reaction / dermatologic adverse event

llna_or_lymph_node_proliferation
  local lymph node assay
  LLNA
  BrdU-ELISA / BrdU-FCM LLNA
  stimulation index / SI
  EC3
  lymph node proliferation after dermal application

guinea_pig_sensitization
  guinea pig maximization test
  GPMT
  Buehler test
  OECD TG 406-like skin sensitization

in_vivo_dermal_irritation_or_toxicity
  dermal irritation
  dermal toxicity
  skin edema
  skin erythema
  Draize skin irritation
  repeat-dose dermal toxicity
```

Direction rules:

```text
supports_skin_reaction_risk:
  positive human patch / HRIPT / clinical skin adverse reaction
  LLNA SI >= 3, positive LLNA call, low EC3 indicating sensitizer potency
  positive GPMT / Buehler / dermal sensitization call
  clear dermal irritation, erythema, edema, necrosis, ulceration, or local toxicity

argues_against_skin_reaction_risk:
  non-sensitizer / non-irritant calls from adequately described validated assays
  LLNA SI < 3 at adequate tested concentrations
  negative human patch / HRIPT only if exposure and concentration are meaningful
```

Evidence strength:

```text
strong:
  human patch / clinical adverse skin reaction with clear positive or negative call
  LLNA with SI / EC3 or explicit positive/negative interpretation

moderate:
  GPMT, Buehler, in vivo dermal irritation/toxicity with clear assay context

weak:
  ambiguous clinical skin terms without dose, route, or adverse-event context
```

### Tier 2: validated skin sensitisation AOP key-event assays

These assays are highly relevant even when they are not direct clinical reactions. They map to the accepted skin
sensitisation adverse outcome pathway: covalent protein binding, keratinocyte activation, and dendritic-cell activation.

Endpoint groups:

```text
protein_binding_or_haptenation
  DPRA
  ADRA
  kDPRA
  direct peptide reactivity assay
  cysteine depletion
  lysine depletion
  peptide depletion
  amino acid derivative reactivity
  haptenation
  protein binding
  covalent binding to protein

keratinocyte_activation_are_nrf2
  KeratinoSens
  LuSens
  EpiSensA
  ARE-Nrf2 luciferase
  antioxidant response element
  Nrf2 activation
  Keap1 / Nrf2
  keratinocyte activation
  HaCaT when assay is explicitly sensitisation-related

dendritic_cell_activation
  h-CLAT
  U-SENS
  IL-8 Luc
  GARDskin
  dendritic cell activation
  THP-1 activation
  U937 activation
  CD54
  CD86
  IL-8 reporter
  genomic allergen rapid detection

glutathione_or_thiol_reactivity
  glutathione depletion
  GSH reactivity
  thiol reactivity
  cysteine adduct formation
```

Direction rules:

```text
sensitization_risk:
  positive DPRA/ADRA/kDPRA, high cysteine/lysine depletion, high peptide reactivity
  positive KeratinoSens/LuSens/EpiSensA, ARE-Nrf2 induction, EC1.5-like activation
  positive h-CLAT/U-SENS/IL-8 Luc/GARDskin, CD54/CD86 upregulation, IL-8 reporter induction
  strong GSH/thiol reactivity when assay explicitly links to sensitisation or electrophilic reactivity

argues_against_skin_reaction_risk:
  negative call in a validated AOP assay, especially if multiple key events are negative
  no peptide depletion / no ARE-Nrf2 activation / no dendritic-cell activation at adequate non-cytotoxic concentrations

context_dependent:
  raw GSH depletion or generic protein binding without skin-sensitisation assay context
```

Evidence strength:

```text
strong:
  validated method with explicit positive/negative call and interpretable concentration-response
  two or more AOP key events agree for the same molecule or close analog

moderate:
  one validated AOP key-event assay with clear positive/negative signal

weak:
  partial mechanistic readout, missing concentration context, or assay only weakly linked to skin sensitisation
```

### Tier 3: phototoxicity, irritation, corrosion, and local skin damage

These assays can support skin-reaction risk, but they answer different questions from allergic sensitisation.

Endpoint groups:

```text
phototoxicity_3t3_nru_or_rhe
  3T3 NRU phototoxicity
  photoirritation factor / PIF
  mean photo effect / MPE
  UVA / UVB photocytotoxicity
  reconstructed human epidermis phototoxicity
  RhE phototoxicity
  phototoxicity / photoallergy / photosafety when assay is adverse

skin_irritation_rhe
  reconstructed human epidermis irritation
  RhE skin irritation
  EpiSkin / EpiDerm / SkinEthic / LabCyte irritation
  MTT viability in skin irritation assay
  OECD TG 439-like assay

skin_corrosion_rhe
  reconstructed human epidermis corrosion
  RhE skin corrosion
  irreversible tissue damage
  necrosis
  OECD TG 431-like assay

keratinocyte_or_skin_cell_cytotoxicity
  keratinocyte viability
  HaCaT viability
  epidermal cell cytotoxicity
  dermal fibroblast cytotoxicity
  skin cell MTT / NRU / LDH release

skin_inflammation_or_barrier_stress
  IL-1 alpha
  IL-6
  IL-8
  TNF alpha
  PGE2
  COX-2
  barrier disruption
  oxidative stress in skin cells
```

Direction rules:

```text
phototoxicity_risk:
  positive 3T3 NRU phototoxicity, high PIF/MPE, cytotoxicity only or much stronger under irradiation
  positive RhE phototoxicity or explicit photosafety concern

irritation_or_corrosion_risk:
  RhE viability below validated irritation/corrosion thresholds
  explicit irritant/corrosive call
  strong local erythema/edema/necrosis in dermal models

local_skin_damage_risk:
  potent keratinocyte or dermal-fibroblast cytotoxicity in a skin-relevant assay
  inflammatory cytokine induction in skin cells with adverse context

context_dependent:
  generic cell viability loss without skin cell type or skin assay context
  anti-inflammatory activity, cytokine inhibition, wound-healing efficacy
```

Evidence strength:

```text
strong:
  validated phototoxicity, RhE irritation, or RhE corrosion assay with explicit positive/negative call

moderate:
  skin-cell cytotoxicity or inflammatory stress with clear skin-relevant cell model and concentration-response

weak:
  cytokine, oxidative stress, or viability readout without clear adverse skin-reaction framing
```

### Tier 4: skin exposure and barrier penetration modifiers

These assays are useful because skin reaction requires local exposure, but they do not define hazard by themselves.

Endpoint groups:

```text
skin_permeability_or_absorption
  skin absorption
  dermal absorption
  percutaneous absorption
  Franz diffusion cell
  diffusion cell
  skin permeation
  skin permeability
  transdermal permeation
  flux
  Jmax
  Kp / logKp
  permeability coefficient
  OECD TG 428-like assay

skin_retention_or_distribution
  skin retention
  epidermis retention
  dermis retention
  stratum corneum retention
  tape stripping
  skin deposition

skin_pampa_or_artificial_membrane
  skin PAMPA
  artificial membrane skin permeability
  silicone / isopropyl myristate skin PAMPA
```

Direction rules:

```text
skin_exposure_support:
  high dermal absorption, high flux, high permeability, strong skin retention
  exposure support can strengthen a hazard signal from Tier 1-3

reduced_skin_exposure:
  low or absent dermal absorption/permeability can weaken but not eliminate hazard concern

context_dependent:
  permeability evidence without any hazard evidence
```

Evidence strength:

```text
moderate:
  validated or well-described skin absorption/permeation assay with quantitative flux/Kp/retention

weak:
  artificial membrane or qualitative permeability call without formulation, dose, or skin model details
```

### Tier 5: weak or context-dependent background

These rows may help the LLM understand analogs, but they should not dominate final prediction.

Endpoint groups:

```text
general_cytotoxicity_context
  CC50
  GI50
  IC50 viability
  LDH release
  apoptosis
  cell proliferation
  non-skin cell viability

immune_or_inflammation_context
  cytokine modulation
  immune-cell activation
  COX / LOX / NF-kB activity
  anti-inflammatory or pro-inflammatory assay without skin context

dermatology_efficacy_context
  anti-acne
  anti-psoriasis
  anti-atopic dermatitis efficacy
  wound healing
  skin whitening
  melanogenesis
  melanoma efficacy
  antimicrobial activity for skin pathogens

target_binding_context
  receptor binding
  enzyme inhibition
  kinase activity
  transporter activity
  target-based pharmacology not explicitly framed as adverse skin reaction
```

Default direction:

```text
context_dependent
```

These rows should usually receive `evidence_strength=weak` or `context_dependent`. They can be retained only when they help
explain a close analog, but they should not be used as primary positive evidence for Skin_Reaction.

## Screening keywords

Initial positive keyword families for `rules.py`:

```text
skin sensitization / sensitisation
skin reaction
contact dermatitis
contact allergy
allergic contact dermatitis
skin allergy
dermal allergy
human patch
patch test
HRIPT
repeated insult patch
maximization test
LLNA
local lymph node
stimulation index
EC3
BrdU-ELISA
BrdU-FCM
guinea pig maximization
Buehler
GPMT

DPRA
ADRA
kDPRA
peptide reactivity
peptide depletion
cysteine depletion
lysine depletion
hapten
haptenation
protein binding
covalent binding
glutathione
GSH
thiol reactivity

KeratinoSens
LuSens
EpiSensA
ARE-Nrf2
Nrf2
Keap1
keratinocyte activation
HaCaT

h-CLAT
U-SENS
IL-8 Luc
GARDskin
dendritic cell activation
THP-1
U937
CD54
CD86
IL-8 reporter

phototoxicity
photoallergy
photoirritation
photosafety
3T3 NRU
PIF
MPE
UVA
UVB

skin irritation
dermal irritation
skin corrosion
dermal corrosion
reconstructed human epidermis
RhE
EpiSkin
EpiDerm
SkinEthic
LabCyte
MTT skin
erythema
edema
necrosis

skin absorption
dermal absorption
percutaneous absorption
skin permeation
skin permeability
transdermal
Franz diffusion
diffusion cell
skin retention
stratum corneum
tape stripping
logKp
Kp
flux
skin PAMPA
```

Negative / exclusion keywords:

```text
melanoma
melanogenesis
tyrosinase inhibition
skin whitening
anti-aging
wrinkle
collagenase
elastase
hair growth
alopecia
sebocyte
acne efficacy
psoriasis efficacy
atopic dermatitis efficacy
eczema treatment
wound healing
antimicrobial
antifungal
antiviral
anti-inflammatory
COX inhibition
LOX inhibition
NF-kB inhibition
cytokine inhibition
topical formulation release only
permeation enhancer assay where the tested molecule is the enhancer vehicle, not the query compound
```

Do not hard-exclude every row containing these terms. If the same description also contains clear adverse skin-reaction terms
such as sensitisation, LLNA, irritation, phototoxicity, or dermal toxicity, keep the row and let `endpoint_groups.py` assign the
more specific group.

## Endpoint assignment requirements

`endpoint_group` must be assigned from the combination of:

```text
assay_tier
standard_type
assay_description
target_pref_name
target_genes
activity_comment
standard_units
```

Do not assign from `standard_type` alone. Examples:

```text
viability
  RhE skin irritation/corrosion context -> skin_irritation_rhe or skin_corrosion_rhe
  HaCaT / keratinocyte context -> keratinocyte_or_skin_cell_cytotoxicity
  generic cancer cell context -> general_cytotoxicity_context

IC50
  phototoxicity +/- irradiation context -> phototoxicity_3t3_nru_or_rhe
  skin-cell viability context -> keratinocyte_or_skin_cell_cytotoxicity
  target inhibition context -> target_binding_context

activity
  h-CLAT / dendritic activation context -> dendritic_cell_activation
  anti-inflammatory efficacy context -> dermatology_efficacy_context
  unclear context -> context_dependent

permeability / flux / Kp
  skin / dermal / Franz / transdermal context -> skin_permeability_or_absorption
  PAMPA skin context -> skin_pampa_or_artificial_membrane
  generic Caco-2 or BBB context -> exclude or context_dependent, not skin evidence
```

## Evidence library row fields

Evidence rows should preserve the same raw ChEMBL fields used by other tasks, plus Skin_Reaction-specific derived fields.

Minimum fields:

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

Skin_Reaction-specific directions:

```text
supports_skin_reaction_risk
argues_against_skin_reaction_risk
sensitization_risk
irritation_or_corrosion_risk
phototoxicity_risk
local_skin_damage_risk
skin_exposure_support
reduced_skin_exposure
context_dependent
unknown_direction
```

Skin_Reaction-specific strengths:

```text
strong
moderate
weak
context_dependent
```

Derived fields such as `endpoint_group_reason`, `evidence_direction`, and `evidence_strength` are for debug and audit. They
should not be sent directly to the reasoning LLM as if they were raw evidence.

## LLM payload rules

The LLM payload should include:

```text
assay_chembl_id
assay_tier
endpoint_group
assay_description
target_pref_name
target_genes
standard_type
standard_relation
standard_value
standard_units
pchembl_value
activity_comment
data_validity_comment
confidence_score
relationship_type
similarity
similarity_bucket
```

Do not include internal rule fields in the LLM evidence rows:

```text
endpoint_group_reason
evidence_direction
evidence_strength
evidence_reason
assay_reason
```

The group-level prompt must make these distinctions explicit:

```text
1. Direct human/LLNA/validated skin reaction evidence can support or oppose final label.
2. AOP key-event evidence supports sensitisation hazard, but one isolated key event is not equal to clinical skin reaction.
3. Phototoxicity is a skin-reaction subtype; it should be separated from allergic sensitisation.
4. Skin irritation/corrosion is local damage evidence; it should not be conflated with immune sensitisation.
5. Skin permeability/retention modifies exposure only; it cannot by itself prove skin reaction risk.
6. Generic cytotoxicity, dermatology efficacy, target binding, and antimicrobial assays are weak context only.
```

## Reasoning schema expectations

Single-molecule branch should focus on:

```text
reactive_or_haptenation_prior
electrophile_or_thiol_reactivity_alerts
skin_permeation_prior
phototoxicity_structural_prior
irritation_or_corrosion_structural_prior
physicochemical_exposure_prior
```

Group-level output should include:

```text
useful_for_skin_reaction_reasoning
transferability
evidence_direction
confidence
reasoning_summary
key_evidence[].effect_on_skin_reaction_reasoning
caveats
```

Final output should include:

```text
skin_reaction_prediction: risk | no_risk
confidence: low | moderate | high
main_evidence_type:
  direct_skin_reaction_anchor
  sensitization_aop
  phototoxicity
  irritation_or_corrosion
  exposure_context_only
  weak_or_no_evidence
key_evidence
conflicting_evidence
caveats
```

Final predictions used for accuracy and macro-F1 must be binary: `risk` or `no_risk`. If the model is uncertain, keep that in
`confidence` and `caveats`, not in the prediction field.

## Exact ChEMBL context

As with the other ChEMBL reasoning tasks, exact-query ChEMBL context can cause retrospective evidence leakage. It must remain
off by default for benchmark runs. Only enable exact-query context for retrospective case studies with an explicit flag such as:

```bash
--enable-chembl-exact-context
```

By default the single-molecule prompt contains no ChEMBL-specific payload or instruction. Only when exact context is enabled and
query exact context is found should the single-molecule payload include `exact_query_chembl_context`, with an instruction to
distinguish direct same-molecule ChEMBL skin-reaction evidence from the physicochemical prior. ChEMBL neighbor evidence still
belongs only in group-level context.

## Initial references used for ontology design

The evidence ontology above follows the regulatory skin-safety assay landscape:

```text
OECD TG 429 / 442B:
  LLNA and non-radioactive LLNA variants for skin sensitisation.

OECD TG 442C:
  DPRA, ADRA, and kDPRA for covalent protein binding / peptide reactivity.

OECD TG 442D:
  KeratinoSens, LuSens, and EpiSensA for keratinocyte activation through ARE-Nrf2-related pathways.

OECD TG 442E:
  h-CLAT, U-SENS, IL-8 Luc, and GARDskin for dendritic-cell activation.

OECD TG 439:
  reconstructed human epidermis skin irritation.

OECD TG 431:
  reconstructed human epidermis skin corrosion.

OECD TG 432 / TG 498 and ICH S10:
  3T3 NRU and RhE phototoxicity / photosafety assessment.

OECD TG 428:
  in vitro skin absorption / dermal absorption.
```
