# DILI task notes

## Testing discipline

Do not add circular tests that merely assert newly written prompt prose or copy
implementation literals into the test. Prompt wording is validated with reviewed
input/output fixtures or a real pilot/evaluation. Automated tests should cover
executable behavior, failure modes, schemas, rendering validity, and provenance.

本文件记录 DILI 的 task-specific 语义、机制驱动的 evidence ontology、assay screening 方向、
endpoint group 设计和 reasoning 约束。通用 ChEMBL workflow、tool service、batch/resume、
trace viewer、输出目录和成本规范仍以仓库根 `AGENTS.md` 为准。

DILI 当前不在四任务、21-condition 的 paper matrix 中，也还没有 `experiment_config.py`。本文件的大部分运行
说明用于复现 2026-06 的 native endpoint-group runner。若将 DILI 纳入当前论文框架，必须先把细粒度 groups
映射到少量、数据源无关的 mechanism families，再由通用 paper runner 做 family-level reasoning；不得直接
把旧 endpoint groups 一一升级成论文分支。

## Task 定义

目标不是训练一个普通 hepatotoxicity QSAR classifier，而是构建可审计的 DILI evidence retrieval
和 reasoning workflow：给定 query molecule，从 ChEMBL 或后续 Starling evidence source 中检索
与 DILI 机制相关的相似分子实验读数，再由 reasoning LLM 判断这些 analog evidence 是否能 transfer
到 query molecule。

当前计划适配 TDC DILI 二分类数据：

```text
data/gold_labels/legacy/processed/DILI/train.jsonl
data/gold_labels/legacy/processed/DILI/valid.jsonl
data/gold_labels/legacy/processed/DILI/test.jsonl

fields:
  drug: query SMILES
  Y: DILI label
```

评估约定：

```text
Y=1 -> dili_prediction=dili_risk
Y=0 -> dili_prediction=no_dili_risk
final summary 必须在 dili_risk/no_dili_risk 中二选一；不要输出 uncertain prediction
```

TDC DILI / LTKB / DILIrank 类型标签是 human DILI concern 的 drug-level label，不等同于任意
体外 hepatocyte toxicity、任意 CYP/transporter inhibition 或 generic cytotoxicity。Reasoning 时应
把直接人类 DILI evidence、体内肝损伤 phenotype、关键机制 liability 和弱 proxy 严格分开。

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

DILI pipeline 应复用现有 general task workflow 的工程结构，但 DILI 的 evidence tier、endpoint
group、prompt 和 final decision rule 必须是 DILI-specific。不要从 ClinTox 的 broad safety ontology
继承 hERG、neurotoxicity、renal toxicity、genotoxicity 等非肝脏安全分支；这些最多作为排除或背景
context，不能进入 DILI 主 evidence tier。

## Task-specific 文件规划

后续新建代码时建议保持与 BBB_Martins / Skin_Reaction 类似的薄 wrapper 结构：

```text
constants.py
  label / prediction mapping。建议：Y=1 -> dili_risk，Y=0 -> no_dili_risk。

rules.py
  DILI assay keyword、negative keyword、mechanism family 和 weak/context 配置。

scoring.py
  assay screening / rescore 的保留、剔除和打分入口。

endpoint_groups.py
  DILI Tier.endpoint_group、evidence_direction、evidence_strength 和 endpoint assignment 规则。

screen_assays.py / rescore_outputs.py / summarize_outputs.py / report.py
  thin wrappers，复用 common task workflow 生成候选 assay、activity evidence、health check 和 report。

build_evidence_library.py
  从 DILI assay candidates + activity evidence 构建 molecule-level evidence library 和 neighbor index。

retrieve_neighbors.py
  旧 native runner 对每个 source-local Tier.endpoint_group 做 analog retrieval。

chembl_exact_context.py
  exact-query ChEMBL context wrapper。benchmark 默认不开启，避免 retrospective leakage。

run_reasoning_pipeline.py
  旧 DILI-specific retrieval prefetch、single-molecule branch、endpoint-group 并发 reasoning、final summary、
  trace 保存和 --resume-final-from-run-dir final-only rerun。

run_reasoning_batch.py
  批量 reasoning wrapper。复用 common reasoning_batch.py，输出 predictions、metrics、report、logs、
  runs 和 combined trace；支持 --skip-existing 断点续跑。当前已接入公共 global prompt stage pool，但旧
  DILI pipeline 尚未迁移到共享 experiment retrieval/identity/context contract，因此 batch 只允许默认
  `native + operational + similarity + standard`；公共 parser 会拒绝 parent-disjoint、coverage 或 branch-reuse
  参数，避免 manifest 声明超出实际 retrieval 能力。
```

不要在 wrapper 中新增业务规则；DILI assay 保留/剔除逻辑应只放在 `rules.py` 和 `scoring.py`，
endpoint-group 语义应只放在 `endpoint_groups.py`，prompt/schema 语义应只放在
`run_reasoning_pipeline.py`。

## 当前实现状态

2026-06-29 已创建 DILI task v0 的可执行 screening / retrieval skeleton；2026-06-30 已完成 full
screening、全量分布审核、evidence 校准和 DILI-specific reasoning pipeline/schema：

```text
tools/chembl_tool/tasks/dili/
  AGENTS.md
  __init__.py
  constants.py
  rules.py
  scoring.py
  endpoint_groups.py
  report.py
  screen_assays.py
  rescore_outputs.py
  summarize_outputs.py
  build_evidence_library.py
  retrieve_neighbors.py
  chembl_exact_context.py
  run_reasoning_batch.py
  run_reasoning_pipeline.py
```

当前实现边界：

```text
rules.py / scoring.py / endpoint_groups.py 已实现 DILI v0 ontology：
  Tier 1 direct human/clinical DILI
  Tier 2 in vivo liver injury
  Tier 3 cholestasis/hepatobiliary transporter
  Tier 4 mitochondrial/oxidative/organelle stress
  Tier 5 reactive metabolite/bioactivation/immune-idiosyncratic
  Tier 6 hepatic cell injury/exposure-property modifiers

screen_assays.py / rescore_outputs.py / summarize_outputs.py / report.py 已接入 common workflow。
build_evidence_library.py / retrieve_neighbors.py / chembl_exact_context.py 已接入 common retrieval/index workflow。
run_reasoning_batch.py 已配置 dili_prediction label mapping。
run_reasoning_pipeline.py 已实现 DILI-specific prompt/schema、single-molecule branch、group branch、
final branch、tool orchestration、trace 输出和 final-only resume。
tools/trace_viewer/viewer.html 已适配 dili_prediction、useful_for_dili_reasoning、
effect_on_dili_reasoning 和 DILI-specific assessment fields。
```

## Historical TRIM / DeepSeek properties-only baselines

2026-06-29 跑了 3 个 Intern-S1/TRIM no-retrieval properties-only DeepSeek-v4-pro baseline，
用于和当前 DILI-specific ChEMBL retrieval pipeline 做历史参考比较。它们不使用 TxAgent 当前
`run_reasoning_pipeline.py`，也不使用 ChEMBL/Starling retrieval；prompt 来自
`trim.reasoning.task_user_prompts.render_task_user_message`，tool mode 为 `properties`，
唯一可见工具是 `get_mol_properties_and_fg`。数据 split 使用
`/data1/tianang/Projects/Intern-S1/DataPrepare/TDC_no_conflict_labels_salt_removed/test/DILI.jsonl`
的 96 条样本。

```text
identity allowed:
  log: /data1/tianang/Projects/Intern-S1/logs/deepseek-v4-pro_properties_Skin_Reaction_DILI_test_20260629_194945.log
  trace: /data1/tianang/Projects/Intern-S1/reasoning-trajectory/deepseek-v4-pro_properties_Skin_Reaction_DILI_test_20260629_194945.log/DILI.jsonl
  n=96, failed parses=0
  accuracy=0.7604
  macro-F1=0.7506
  class 0 precision/recall/F1=0.8710/0.5870/0.7013
  class 1 precision/recall/F1=0.7077/0.9200/0.8000
  tool usage: 94/96 questions with tools, avg tools/sample=0.98

strict no identity / no memory comparison:
  log: /data1/tianang/Projects/Intern-S1/logs/deepseek-v4-pro_properties_no_smiles_identity_Skin_Reaction_DILI_test_20260629_195403.log
  trace: /data1/tianang/Projects/Intern-S1/reasoning-trajectory/deepseek-v4-pro_properties_no_smiles_identity_Skin_Reaction_DILI_test_20260629_195403.log/DILI.jsonl
  n=96, failed parses=0
  accuracy=0.6354
  macro-F1=0.5988
  class 0 precision/recall/F1=0.7619/0.3478/0.4776
  class 1 precision/recall/F1=0.6000/0.9000/0.7200
  tool usage: 96/96 questions with tools, avg tools/sample=1.00

identity forbidden but memory comparison allowed:
  log: /data1/tianang/Projects/Intern-S1/logs/deepseek-v4-pro_properties_no_smiles_identity_allow_memory_compare_Skin_Reaction_DILI_test_20260629_195652.log
  trace: /data1/tianang/Projects/Intern-S1/reasoning-trajectory/deepseek-v4-pro_properties_no_smiles_identity_allow_memory_compare_Skin_Reaction_DILI_test_20260629_195652.log/DILI.jsonl
  n=96, failed parses=0
  accuracy=0.6667
  macro-F1=0.6306
  class 0 precision/recall/F1=0.8500/0.3696/0.5152
  class 1 precision/recall/F1=0.6184/0.9400/0.7460
  tool usage: 96/96 questions with tools, avg tools/sample=1.01
```

Interpretation caveat:

```text
These are historical TRIM properties-only baselines, not same-prompt zero-retrieval ablations of the
current DILI-specific pipeline. Use them as a lower-context DeepSeek/tool reference point. The strong
identity-allowed score may include molecule/class recognition from SMILES and should be reported
separately from strict no-identity settings.
```

当前测试：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  /data1/tianang/anaconda3/envs/vllm/bin/pytest tests/chembl_tool/tasks/dili -q

result:
  56 passed
```

`vllm` 环境中的 pytest 插件自动扫描在当前机器上可能卡在 conda dist-info entry point 读取；跑 DILI
单元测试时使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。这不是 DILI 代码问题。

2026-06-30 additional verification:

```text
PYTHONDONTWRITEBYTECODE=1 /data1/tianang/anaconda3/envs/vllm/bin/python -m py_compile \
  $(find tools/chembl_tool/tasks/dili tests/chembl_tool/tasks/dili -name '*.py' | sort)

PYTHONDONTWRITEBYTECODE=1 /data1/tianang/anaconda3/envs/vllm/bin/python \
  -m tools.chembl_tool.tasks.dili.run_reasoning_pipeline --help

PYTHONDONTWRITEBYTECODE=1 /data1/tianang/anaconda3/envs/vllm/bin/python \
  -m tools.chembl_tool.tasks.dili.run_reasoning_batch --help

result:
  ok
```

2026-06-29 smoke screening / evidence QA 当前校准状态：

```text
clean smoke version:
  outputs/chembl_tool/tasks/dili/assay_screening/smoke_v8_200k/

screen command:
  PYTHONDONTWRITEBYTECODE=1 /data1/tianang/anaconda3/envs/vllm/bin/python \
    -m tools.chembl_tool.tasks.dili.screen_assays \
    --chembl-sqlite tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db \
    --out-dir outputs/chembl_tool/tasks/dili/assay_screening/smoke_v8_200k \
    --min-score 40 \
    --limit 200000 \
    --progress-every 50000 \
    --export-activities

smoke_v8_200k result:
  scanned=200000
  retained_assays=100
  activity_evidence_rows=241
  Tier 2=93
  Tier 4=4
  Tier 5=1
  Tier 6=2
  Tier 1=0
  Tier 3=0

current smoke evidence index:
  outputs/chembl_tool/tasks/dili/evidence_library/smoke_v8_200k/
  evidence_rows=241
  indexed_molecules=142
  groups=5

retrieval smoke files:
  retrieval_smoke_acetaminophen.json
  retrieval_smoke_diclofenac.json
```

Important screening calibrations already applied:

```text
1. Exclude jaundiced-animal feces/clearance models; these are disease-model PK/clearance assays, not human DILI.
2. Exclude bone-marrow ALP/ALPL; ALP is only liver-relevant with liver/bile/clinical context.
3. Exclude ASBT/SLC10A2, ileal taurocholate, ileal brush-border and intestinal bile-acid uptake; these are not
   hepatobiliary/cholestatic DILI transporter evidence.
4. Exclude P-gp/ABCB1/MDR1; do not treat generic P-gp transport as DILI cholestasis evidence.
5. Exclude APAP-induced hepatoprotective/protection assays; they measure protective efficacy, not compound-induced DILI.
6. Exclude receptor/target covalent binding unless there is reactive metabolite, bioactivation, microsome, NADPH,
   glutathione/GSH or hepatic metabolism context.
7. Exclude hypolipidemic/hyperlipidemic/cholesterol-diet liver-weight efficacy assays.
8. Exclude primary-hepatocyte genotoxicity/comet/DNA-break assays unless explicitly DILI/hepatotoxicity.
9. Exclude generic HepG2 cancer-cell cytotoxicity / MTT / Alamar blue / acid-phosphatase assays; keep HepG2/C3A ADMET,
   primary hepatocyte, HepaRG, LDH/apoptosis/caspase or explicit hepatotoxicity models.
10. Require liver/hepatic context for Tier 2 necrosis/pathology/degeneration; this removes tumor-necrosis efficacy
    assays such as EMT-6 tumor necrosis after photodynamic therapy.
11. Downweight liver-weight-only in vivo findings to evidence_strength=moderate, while hepatic necrosis/pathology and
    ALT/clinical chemistry remain strong.
12. Map GSH/glutathione content/elevation assays to Tier 4.energy_failure_and_oxidative_stress but set direction to
    neutral_or_unclear and strength=weak unless depletion/ROS/oxidative-stress language is present.
```

Additional QA notes:

```text
smoke_v6_200k retained 119 assays and exposed residual generic HepG2 cytotoxicity plus tumor-necrosis false positives.
smoke_v7_200k retained 113 assays after generic HepG2 filtering but still had tumor-necrosis false positives.
smoke_v8_200k retained 100 assays after both filters; generic HepG2 cytotoxicity and tumor-necrosis counts are 0.

A full 1.89M-assay screening attempt reached 200k/1.89M at about 425 assays/s, implying roughly 75 minutes for a
single full pass in the current common workflow. Use smoke windows for rule iteration, then run full screening once
rules are stable enough.

Post-200k audit window:
  offset=400000
  limit=10000
  retained=12
  tiers: Tier 3=2, Tier 6=2, Tier 2=2, Tier 4=6
  generic_hepg2_cytotox_against=0
  retained examples were MRP2 transporter, HepG2/C3A ADMET toxicity, liver weight, and rat liver mitochondrial swelling.
```

2026-06-30 full screening / full-distribution calibration:

```text
Initial full screening:
  out_dir: outputs/chembl_tool/tasks/dili/assay_screening/v1_full
  scanned=1,890,749
  retained_assays=7,709
  initial_activity_evidence_rows=514,437

Full-screen calibration versions:
  v1_full retained 7,709 assays
  v2_full retained 7,222 assays, activity rows 38,493
  v3_full retained 7,206 assays, activity rows 38,411
  v4_full retained 7,065 assays, activity rows 36,232
  v5_full retained 7,056 assays, activity rows 36,167
  v6_full retained 7,024 assays, activity rows 36,112
  v7_full retained 7,023 assays, activity rows 36,110

Current clean full artifact:
  outputs/chembl_tool/tasks/dili/assay_screening/v7_full/

Current clean evidence library:
  outputs/chembl_tool/tasks/dili/evidence_library/dili_molecule_evidence.jsonl
  outputs/chembl_tool/tasks/dili/evidence_library/dili_neighbor_index.pkl
  outputs/chembl_tool/tasks/dili/evidence_library/dili_neighbor_index.meta.json

Final index metadata:
  n_evidence_rows=36,110
  n_index_molecules=8,284
  n_groups=17
  workers=128
  elapsed_s≈16.3 for final index build

Final v7 assay tier distribution:
  Tier 6=2,753
  Tier 3=1,550
  Tier 2=1,281
  Tier 4=616
  Tier 5=517
  Tier 1=306

Final v7 molecule-level evidence group distribution:
  hepatobiliary_transporter_panel=8,967
  human_dili_or_hepatotoxicity=5,692
  human_liver_laboratory_signal=5,458
  hepatocyte_or_hepatic_cell_injury=4,543
  bsep_or_bile_acid_efflux=2,593
  in_vivo_liver_histopathology=2,322
  reactive_metabolite_or_covalent_binding=1,534
  er_lysosomal_lipid_stress=1,413
  severe_liver_outcome_or_regulatory_signal=1,263
  mitochondrial_function_or_respiration=736
  in_vivo_liver_clinical_chemistry=593
  energy_failure_and_oxidative_stress=477
  context_dependent=315
  cholestasis_or_bile_acid_accumulation=101
  hepatic_metabolism_bioactivation=97
  immune_or_idiosyncratic_context=5
  in_vivo_hepatotoxic_dose_or_margin=1
```

Full-screen false-positive clusters found and calibrated out:

```text
1. Generic PCSK9/LDLR HepG2 reporter and target-biology assays.
2. ABHD10/PME target assays and non-DILI activity-based protein profiling.
3. Kidney/renal microsome, transporter or necrosis contexts without hepatic DILI relevance.
4. LDHA/lactate-production target assays; do not confuse them with LDH release.
5. Broad hepatoprotective/protection/rescue assays, including APAP protection and rotenone ATP rescue.
6. H2O2-induced ROS antioxidant/protection assays; keep only direct ROS induction or reactive GSH/MPO/HRP chemistry.
7. HFD/CCl4, NASH/NAFLD, antidiabetic, hypolipidemic and liver-disease efficacy models.
8. Non-hepatobiliary bile-acid-adjacent enzymes/pathogens such as AKR1C/HSD, glucosidases, carboxylic ester
   hydrolase and Cryptosporidium target assays.
9. Generic receptor/kinase covalent binding without reactive metabolite, bioactivation or hepatic metabolism context.
```

Final retrieval smoke files:

```text
outputs/chembl_tool/tasks/dili/evidence_library/retrieval_smoke_acetaminophen_v7_full.json
outputs/chembl_tool/tasks/dili/evidence_library/retrieval_smoke_diclofenac_v7_full.json

settings:
  top_k_per_group=3
  min_similarity=0.2

result:
  both status=ok
  both n_groups=17
  both n_groups_with_neighbors=15
  both n_neighbors_total=40

QA conclusion:
  acetaminophen retrieves direct human DILI/LTKB-like rows, human liver lab/severe outcome rows, mouse ALT/necrosis
  rows, BSEP/OATP/MRP context, ROS induction, GSH adduct and hepatic cell injury rows.
  diclofenac retrieves close human DILI/hepatotoxicity rows, BSEP/MRP/OATP rows, cholestatic liver injury,
  mitochondrial dysfunction, GSH reactivity/acyl-glucuronide-related rows and HepG2 injury/apoptosis rows.
  Previously observed protection/rescue/HFD disease-model false positives are absent in v7 smoke.
```

2026-06-30 DeepSeek-v4-pro full-pipeline LLM smoke:

```text
input:
  /data1/tianang/Projects/Intern-S1/DataPrepare/TDC_no_conflict_labels_salt_removed/test/DILI.jsonl

batch:
  outputs/chembl_tool/tasks/dili/reasoning/batches/dili_smoke_full_v7_idx0_3_6_20260630/

command shape:
  python -m tools.chembl_tool.tasks.dili.run_reasoning_batch \
    --input-jsonl /data1/tianang/Projects/Intern-S1/DataPrepare/TDC_no_conflict_labels_salt_removed/test/DILI.jsonl \
    --indices 0 3 4 6 7 9 \
    --parallelism 2 \
    --batch-id dili_smoke_full_v7_idx0_3_6_20260630 \
    --max-tokens 8192 \
    --timeout-s 300 \
    --max-tool-rounds 4 \
    --top-k-per-group 3 \
    --min-similarity 0.3 \
    --skip-existing \
    --stream-logs

metrics:
  n_total=6
  n_successful=6
  n_failed_runs=0
  labels: 3 negative / 3 positive
  prediction_distribution: no_dili_risk=3 / dili_risk=3
  accuracy=1.0
  macro_f1=1.0
  confusion_matrix: tn=3, fp=0, fn=0, tp=3

per-index predictions:
  idx0 label=0 prediction=no_dili_risk confidence=moderate
  idx3 label=1 prediction=dili_risk confidence=moderate
  idx4 label=0 prediction=no_dili_risk confidence=low
  idx6 label=1 prediction=dili_risk confidence=moderate
  idx7 label=1 prediction=dili_risk confidence=moderate
  idx9 label=0 prediction=no_dili_risk confidence=high

QA:
  final JSON complete for all 6 samples.
  single-molecule branch called molecule_properties once for every sample.
  all group outputs had status=ok.
  no tool_result status=error in the 6-sample smoke.
  combined trace lines=66.
```

Smoke-discovered retry fix:

```text
The first 6-sample smoke surfaced one blank DeepSeek group response:
  idx9 Tier 4.er_lysosomal_lipid_stress raw_content was whitespace only.

run_reasoning_pipeline.py now retries once when assistant content is blank or completely unparseable JSON by appending:
  "Your previous response was empty or not valid JSON. Return only the required JSON object now."

Regression test:
  test_empty_or_unparsed_json_response_triggers_retry

patched retry validation:
  batch: outputs/chembl_tool/tasks/dili/reasoning/batches/dili_smoke_full_v7_retry_idx9_20260630/
  idx9 status=ok prediction=no_dili_risk correct=True
  groups=9 all ok
  unparsed=[]
  tool_errors=0
  tool calls: molecule_properties=1, mmp_structure_compare=23, properties_compare=22
```

Scheduled full DILI run:

```text
scheduled_at:
  2026-06-30 06:00:00 America/New_York

scheduler script:
  outputs/chembl_tool/tasks/dili/reasoning/scheduled/run_dili_full_20260630_0600.sh

scheduler log:
  outputs/chembl_tool/tasks/dili/reasoning/scheduled/run_dili_full_20260630_0600.scheduler.log

scheduler pid file:
  outputs/chembl_tool/tasks/dili/reasoning/scheduled/run_dili_full_20260630_0600.scheduler.pid

batch:
  outputs/chembl_tool/tasks/dili/reasoning/batches/dili_full_v7_deepseek_20260630_0600/

parameters:
  input_jsonl=/data1/tianang/Projects/Intern-S1/DataPrepare/TDC_no_conflict_labels_salt_removed/test/DILI.jsonl
  n=96
  parallelism=3
  group_workers=20
  max_tokens=8192
  timeout_s=300
  max_tool_rounds=4
  top_k_per_group=3
  min_similarity=0.3
  skip_existing=true
  chembl_exact_context=false

The scheduler checks tool service health at 127.0.0.1:8765 before starting; if it is down, it starts
uvicorn tools.service.app:app on that port and waits for health before running the batch.
```

下一步工作流：

```text
1. 若要做模型评估，直接用 v7_full evidence library 跑 run_reasoning_batch.py。
2. Benchmark 默认不要开启 --enable-chembl-exact-context，避免 same-molecule ChEMBL evidence 泄漏。
3. 下一轮建议从 6-sample smoke 扩到 20-sample balanced smoke，重点看 false positive/false negative traces。
4. 如果 GLM-5.2 跑 DILI，沿用根 AGENTS.md 的 --disable-thinking / reasoning_effort="" 兼容参数。
5. Starling acquisition 以冻结后的 DILI mechanism family 为 task/prompt 粒度；endpoint subtype、species、
   dose 和 assay context 作为 family 内 schema 字段。旧 Tier.endpoint_group 只做接入审计，不逐组运行 Starling。
```

## DILI evidence 总原则

DILI 是 clinical phenotype，不是单一机制。一个药物可能通过胆汁酸转运干扰、线粒体损伤、
反应性代谢物、氧化/ER stress、免疫介导反应、肝细胞死亡、肝暴露/剂量等多条链条导致 DILI。
因此 evidence tier 应按“与 DILI label 的距离 + 机制可解释性 + 后续 Starling 可单独构建数据”的
原则设计，而不是按 ChEMBL 关键词随意细分。

初版 DILI tier 应保持少量、可运行、可解释：

```text
Tier 1: direct human or clinical DILI anchors
Tier 2: in vivo liver injury phenotype and clinical pathology
Tier 3: cholestasis and hepatobiliary transporter liability
Tier 4: mitochondrial, oxidative and organelle stress
Tier 5: reactive metabolite, bioactivation and immune/idiosyncratic liability
Tier 6: hepatic cell injury models and exposure/property modifiers
```

这些 tier 可以先归并为未来 Starling acquisition 的 mechanism families。原则上每个 family 对应一个
Starling task；不要把 mechanism 切得过细，例如不要把
BSEP、MRP2、NTCP、MDR3 各自拆成单独 tier；它们属于同一条 cholestasis / hepatobiliary transporter
机制轴。也不要把 ROS、ATP、MMP、ER stress 全拆成单独 tier；它们属于 organelle stress 机制轴。

重要约束：

```text
1. Tier 1 始终是最直接的 DILI 测量或 human/clinical liver safety outcome。
2. Tier 2 是 organism-level liver injury phenotype，比体外 proxy 更强，但仍需 species、route、
   dose、duration 和 exposure context。
3. Tier 3-5 是关键机制 liability。它们可以强烈支持 DILI risk，但单一弱阳性通常不能替代直接 DILI
   phenotype。
4. Tier 6 是有用的 supporting evidence，不应单独把 query 判为 dili_risk，除非与更高 tier 或多个机制
   分支一致。
5. Exact-query ChEMBL context 默认关闭；benchmark 不应把 query molecule 的同分子已知 ChEMBL
   liver evidence 直接注入 prompt。
6. ChEMBL 当前只是 Starling 之前的替代 evidence source。Ontology 不应被 ChEMBL 现有 assay 丰度牵着走。
```

## Evidence library 字段

DILI evidence library 每一条 activity evidence 至少保留：

```text
molecule_chembl_id
canonical_smiles
assay_chembl_id
assay_tier
endpoint_group
standard_type
standard_relation
standard_value
standard_units
pchembl_value
activity_comment
data_validity_comment
assay_description
target_pref_name
target_genes
organism
confidence_score
relationship_type
source
evidence_direction
evidence_strength
evidence_reason
```

内部可以保留 `evidence_direction`、`evidence_strength`、`endpoint_group_reason`、`assay_reason`
方便 debug 和审计；LLM payload 中的 activity evidence row 只包含原始 ChEMBL assay/activity 字段和
必要 metadata。不要把内部规则派生字段整包送给 reasoning LLM。

建议 `evidence_direction`：

```text
supports_dili_risk
argues_against_dili_risk
clinical_dili_signal
in_vivo_liver_injury_signal
cholestasis_or_bile_acid_transport_risk
mitochondrial_or_organelle_stress_risk
reactive_metabolite_or_bioactivation_risk
immune_or_idiosyncratic_context
hepatic_cell_injury_risk
exposure_or_property_context
neutral_or_unclear
context_dependent
```

建议 `evidence_strength`：

```text
strong
moderate
weak
context
```

强弱不是 assay score 的同义词。它表示该 evidence type 对 DILI label 的解释距离：

```text
strong:
  direct human/clinical DILI, Hy's-law-like laboratory pattern, severe liver adverse event,
  liver failure/transplant/death, withdrawal/boxed-warning liver signal, hepatic necrosis/pathology, or in vivo
  ALT/AST/ALP/bilirubin/bile-acid clinical chemistry with liver context.

moderate:
  liver-weight-only in vivo findings, BSEP/bile acid transporter liability with potency/exposure context,
  mitochondrial/organelle stress in hepatic cells, reactive metabolite/bioactivation with liver context,
  or coherent multi-assay hepatic cell injury.

weak:
  HepG2/C3A, primary hepatocyte or HepaRG viability/LDH at high concentration, GSH content/elevation without depletion
  or oxidative-stress direction, structural alert, high logP/high dose prior, or single mechanistic proxy without
  liver injury phenotype.

context:
  assay text is liver-adjacent but endpoint direction, disease context, efficacy/toxicity distinction, dose,
  species or cell model is not clear enough.
```

## Evidence 类型解释

DILI 的 evidence tier 不应照搬 BBB / Bioavailability / ClinTox。这里按 DILI label 距离和机制轴拆分。
Tier 1 是最直接的测量结果；Tier 2-6 是从体内 phenotype 到机制 proxy 的逐步降权证据。

### Tier 1: direct human or clinical DILI anchors

最接近 TDC DILI label。优先保留明确发生在人类、临床研究、上市后、药品标签或 human case 语境中的
drug-induced liver injury evidence。

Endpoint groups:

```text
human_dili_or_hepatotoxicity
  drug-induced liver injury
  DILI
  hepatotoxicity
  liver injury
  hepatic injury
  hepatic adverse event
  liver adverse event
  drug-induced hepatitis
  toxic hepatitis
  liver toxicity in patient / volunteer / clinical trial / postmarketing

severe_liver_outcome_or_regulatory_signal
  acute liver failure
  fulminant hepatic failure
  liver transplant
  fatal liver injury
  liver-related death
  drug withdrawal due to hepatotoxicity
  boxed warning / black box warning for liver injury
  contraindication due to liver toxicity
  dose interruption/discontinuation due to liver enzyme elevation

human_liver_laboratory_signal
  ALT elevation
  AST elevation
  transaminase elevation
  bilirubin elevation
  total bilirubin
  alkaline phosphatase / ALP
  GGT
  jaundice
  Hy's law / Hy law
  hepatocellular pattern
  cholestatic pattern
  mixed liver injury pattern
```

强解释条件：

```text
1. assay description、source 或 metadata 明确是 human / clinical / patient / volunteer / trial /
   postmarketing / FDA label / LiverTox-like context。
2. endpoint 是 DILI、hepatotoxicity、liver failure、jaundice、Hy's-law-like lab pattern、
   liver enzyme elevation with bilirubin, or liver-related discontinuation/withdrawal/warning。
3. activity row 能看出方向，例如 "positive", "elevated", "injury", "hepatotoxic", "withdrawn",
   "not tolerated", "liver failure", "acute liver failure"。
```

弱解释或剔除条件：

```text
1. 只有 routine mild ALT/AST monitoring，不伴随 bilirubin、症状、剂量中断或严重 outcome 时，通常是
   monitoring evidence，不等于 positive DILI label。
2. Oncology / antiviral / anti-infective efficacy trial 中的 liver lab abnormality 要区分疾病背景、
   联合用药和高剂量治疗语境。
3. "no liver injury", "no ALT elevation", "well tolerated" 只有在 dose/exposure 和 duration 明确时
   才可作为反向证据。
```

### Tier 2: in vivo liver injury phenotype and clinical pathology

动物或非临床 in vivo liver injury phenotype。比多数体外 assay 更接近 human DILI，但 transferability
依赖 species、route、dose、duration、metabolite coverage 和 exposure margin。

Endpoint groups:

```text
in_vivo_liver_histopathology
  liver histopathology
  hepatic necrosis
  centrilobular necrosis
  hepatocellular degeneration
  hepatocyte hypertrophy
  liver inflammation
  bile duct injury
  bile duct hyperplasia
  portal inflammation
  liver fibrosis
  liver weight increase / decrease

in_vivo_liver_clinical_chemistry
  ALT
  AST
  ALP
  bilirubin
  GGT
  bile acid level
  serum bile acid
  hepatic enzyme elevation
  transaminase elevation in rat / mouse / dog / monkey

in_vivo_hepatotoxic_dose_or_margin
  NOAEL with liver finding
  LOAEL with liver finding
  MTD with liver toxicity
  liver toxic dose
  repeated-dose liver toxicity
  subacute / subchronic / chronic liver toxicity
  toxicokinetic exposure margin for liver finding
```

解释规则：

```text
histopathology:
  liver tissue injury 是 strong/moderate evidence，尤其是 necrosis、bile duct injury、inflammation
  或 repeated-dose finding。

clinical chemistry:
  ALT/AST/ALP/bilirubin/bile acids 在 in vivo context 中是 liver injury phenotype，但需要看 fold
  change、dose、duration 和 reversibility。

NOAEL/LOAEL/MTD:
  只有明确 liver finding 或 liver clinical chemistry 时才纳入 DILI reasoning；generic MTD/NOAEL
  没有 liver context 时不保留到 DILI 主 tier。
```

### Tier 3: cholestasis and hepatobiliary transporter liability

胆汁酸稳态和肝胆转运体是 DILI 中最清晰、最可 assay 化的机制轴之一。该 tier 覆盖 BSEP、MRP2、
MDR3、NTCP、OATP 等 hepatobiliary transporter 和 bile acid accumulation / cholestasis phenotype。

Endpoint groups:

```text
bsep_or_bile_acid_efflux
  BSEP
  ABCB11
  bile salt export pump
  bile acid efflux
  taurocholate efflux
  bile salt transport
  canalicular bile acid transport

hepatobiliary_transporter_panel
  MRP2 / ABCC2
  MRP3 / ABCC3
  MRP4 / ABCC4
  MDR3 / ABCB4
  NTCP / SLC10A1
  OATP1B1 / SLCO1B1
  OATP1B3 / SLCO1B3
  bile acid uptake
  hepatobiliary transporter inhibition

cholestasis_or_bile_acid_accumulation
  cholestasis
  cholestatic liver injury
  intrahepatic cholestasis
  bile acid accumulation
  serum bile acids
  impaired bile flow
  bile canalicular network
```

解释规则：

```text
BSEP/ABCB11 inhibition:
  是 cholestatic DILI risk 的核心机制 evidence。强度取决于 potency、assay system、free exposure
  margin、bile acid accumulation 和是否伴随 mitochondrial/hepatocyte injury。

MRP2/MDR3/NTCP/OATP:
  支持 hepatobiliary disposition / bile acid handling context。单一 transporter inhibition 通常
  moderate/weak，除非与 cholestasis phenotype 或多 transporter liability 一致。

cholestasis phenotype:
  如果是 human 或 in vivo cholestatic injury，应优先按 Tier 1 或 Tier 2 解释；Tier 3 保留机制维度。
```

### Tier 4: mitochondrial, oxidative and organelle stress

线粒体功能损伤、ATP depletion、氧化 stress、ER stress 和 lysosomal/phospholipidosis 等 organelle
stress 是 DILI 的重要机制轴。该 tier 合并这些相互交织的 stress pathways，避免为 Starling 过度拆分。

Endpoint groups:

```text
mitochondrial_function_or_respiration
  mitochondrial toxicity
  mitochondrial dysfunction
  mitochondrial membrane potential
  MMP loss
  oxygen consumption rate
  OCR
  respiratory chain
  electron transport chain
  complex I / II / III / IV inhibition
  mitochondrial respiration
  mitochondrial swelling

energy_failure_and_oxidative_stress
  ATP depletion
  cellular ATP
  oxidative stress
  ROS
  reactive oxygen species
  glutathione depletion
  GSH depletion
  Nrf2 / NFE2L2
  antioxidant response
  JNK activation

er_lysosomal_lipid_stress
  ER stress
  unfolded protein response
  UPR
  phospholipidosis
  lysosomal trapping
  lysosomal stress
  steatosis
  lipid accumulation
  fatty liver
```

解释规则：

```text
mitochondrial assays:
  Hepatic or metabolically competent cell context 中的 MMP/OCR/ATP readout 是 moderate/strong
  mechanistic evidence。Generic non-hepatic mitochondrial readout 通常降为 weak/context。

oxidative stress:
  ROS/GSH/Nrf2 是 DILI 相关 stress evidence，但单个 reporter assay 不等于 DILI。

phospholipidosis/steatosis:
  对 cationic amphiphilic or lipid-disposition liability 有用，通常 moderate/weak，除非与 liver
  phenotype 或 hepatocyte injury 一致。
```

### Tier 5: reactive metabolite, bioactivation and immune/idiosyncratic liability

许多 idiosyncratic DILI 与 bioactivation、反应性代谢物、covalent binding、GSH adduct、drug-protein
adduct、危险信号和 adaptive immune response 有关。该 tier 把 bioactivation 和 immune/idiosyncratic
context 合并，因为它们在 evidence retrieval 中常常共同出现，且单独拆分会显著增加 Starling 分支数。

Endpoint groups:

```text
reactive_metabolite_or_covalent_binding
  reactive metabolite
  bioactivation
  covalent binding
  protein adduct
  drug-protein adduct
  GSH adduct
  glutathione adduct
  cysteine trapping
  cyanide trapping
  quinone imine
  quinone methide
  acyl glucuronide
  iminium ion

hepatic_metabolism_bioactivation
  liver microsome bioactivation
  hepatocyte bioactivation
  CYP-mediated bioactivation
  CYP3A4 bioactivation
  CYP2C9 / CYP2C19 / CYP2D6 metabolism when linked to reactive intermediate
  metabolic activation
  metabolite-mediated toxicity

immune_or_idiosyncratic_context
  idiosyncratic DILI
  immune-mediated liver injury
  HLA association
  T cell response
  cytokine release with liver injury
  inflammasome / danger signal with hepatotoxicity
  adaptive immune response
```

解释规则：

```text
reactive metabolite:
  GSH/covalent-binding evidence suggests electrophilic intermediate formation. It is important mechanism evidence,
  but not a sufficient positive DILI label by itself; many reactive metabolites do not produce clinical DILI.

bioactivation:
  Stronger when observed in hepatocyte/liver microsome/S9 with relevant CYP or metabolite identity and paired with
  hepatic cell injury, mitochondrial stress, oxidative stress, or clinical/in vivo DILI.

immune/idiosyncratic:
  HLA/T-cell/cytokine evidence is highly DILI-relevant when drug-specific. Generic immune activation without liver
  injury context should be weak/context.
```

### Tier 6: hepatic cell injury models and exposure/property modifiers

该 tier 覆盖较接近 liver biology 但仍偏 proxy 的 evidence：hepatocyte/HepaRG/HepG2/3D spheroid injury、
high-content hepatotoxicity、transcriptomic liver stress，以及会改变 DILI plausibility 的 dose/exposure/
property context。它是 useful supporting evidence，不应单独替代 Tier 1-5。

Endpoint groups:

```text
hepatocyte_or_hepatic_cell_injury
  primary human hepatocyte
  hepatocyte viability
  HepaRG cytotoxicity
  HepG2 cytotoxicity
  liver spheroid
  hepatic organoid
  micropatterned hepatocyte co-culture
  LDH release
  apoptosis
  necrosis
  caspase activation
  high-content hepatotoxicity

liver_omics_or_stress_signature
  toxicogenomics
  transcriptomic DILI signature
  liver stress gene expression
  metabolomics liver toxicity
  proteomics liver toxicity
  high-content imaging liver toxicity

exposure_dose_or_property_context
  high daily dose
  high lipophilicity
  rule of two
  logP
  logD
  cationic amphiphilicity
  high hepatic extraction
  liver accumulation
  extensive hepatic metabolism
  CYP substrate with high exposure
```

解释规则：

```text
hepatocyte viability:
  Liver-cell context 比 generic cytotoxicity 更 relevant，但仍需 concentration、time、metabolic competence
  和 assay specificity。High-concentration nonspecific cytotoxicity 通常弱。

omics / HCS:
  机制覆盖更广，可作为 strong supporting evidence when liver-specific and replicated，但仍需避免把任何
  stress signature 自动等同于 clinical DILI。

exposure/property:
  高 dose + 高 lipophilicity、强肝代谢、cationic amphiphilicity 等可提高 DILI plausibility。它们是 prior
  或 modifier，不是 direct evidence。
```

### Weak / excluded context

下面 evidence 不进入 DILI 主 tier，除非 assay description 明确给出 liver/DILI context：

```text
generic cytotoxicity in non-hepatic cancer cell lines
generic anti-proliferative / GI50 / growth inhibition efficacy
anti-infective replication or cytopathic-effect assays
hERG / QT / cardiac electrophysiology
renal, neuro, reproductive, endocrine or hematologic toxicity without liver context
generic CYP inhibition not tied to bioactivation, exposure margin or hepatotoxicity
generic transporter inhibition not tied to hepatobiliary bile acid transport or liver exposure
target binding / enzyme inhibition / receptor activity without liver injury context
general oxidative-stress reporter without hepatic cell or DILI context
drug-drug interaction liability without liver injury or hepatic exposure link
```

这些行可以在 debug report 中保留为 excluded / context-dependent，但不应送入 DILI reasoning prompt 作为
positive evidence。

## Endpoint Group 标准

第二阶段不按每个 assay 单独检索。应按：

```text
Tier -> endpoint_group
```

组合生成 retrieval groups。初版 endpoint group 数量要克制，便于后续 Starling 每 tier 单独跑。

建议初版分组：

```text
Tier 1.human_dili_or_hepatotoxicity
Tier 1.severe_liver_outcome_or_regulatory_signal
Tier 1.human_liver_laboratory_signal

Tier 2.in_vivo_liver_histopathology
Tier 2.in_vivo_liver_clinical_chemistry
Tier 2.in_vivo_hepatotoxic_dose_or_margin

Tier 3.bsep_or_bile_acid_efflux
Tier 3.hepatobiliary_transporter_panel
Tier 3.cholestasis_or_bile_acid_accumulation

Tier 4.mitochondrial_function_or_respiration
Tier 4.energy_failure_and_oxidative_stress
Tier 4.er_lysosomal_lipid_stress

Tier 5.reactive_metabolite_or_covalent_binding
Tier 5.hepatic_metabolism_bioactivation
Tier 5.immune_or_idiosyncratic_context

Tier 6.hepatocyte_or_hepatic_cell_injury
Tier 6.liver_omics_or_stress_signature
Tier 6.exposure_dose_or_property_context
```

如果 Starling 运行成本需要进一步压缩，默认 Starling acquisition 可以按 tier 合并，不按 endpoint_group
拆开跑；endpoint_group 只作为 retrieval/ranking 和 prompt 内部结构。

## Assay screening 规则方向

优先保留：

```text
1. 明确 human/clinical/postmarketing/labeled DILI 或 liver adverse event。
2. 明确 in vivo liver pathology、liver clinical chemistry 或 liver toxic dose/margin。
3. BSEP/bile acid/hepatobiliary transporter functional readout。
4. Hepatic mitochondrial/OCR/MMP/ATP/ROS/GSH/ER stress assay。
5. Reactive metabolite/GSH/covalent binding/bioactivation with liver context。
6. Hepatocyte/HepaRG/HepG2/3D liver model/high-content liver toxicity。
7. Dose/lipophilicity/hepatic metabolism/exposure context only as modifier evidence。
```

主动剔除：

```text
1. 只有 liver cancer efficacy、hepatocellular carcinoma anti-proliferation 或 antiviral efficacy。
2. 只有 generic target inhibition/binding，没有 liver injury, bile acid, bioactivation or hepatic exposure context。
3. 只有 CYP inhibition IC50，没有 substrate/bioactivation/exposure 或 liver injury context。
4. 只有 generic cytotoxicity in non-hepatic cell lines。
5. 心毒、肾毒、神经毒、genotox、skin reaction 等非肝脏 safety evidence。
6. disease biology assay，例如 fibrosis/inflammation target activity，除非 readout 是 compound-induced
   liver injury/toxicity。
```

## Legacy native reasoning pipeline 约束

```text
1. 读取 DILI test JSONL 的 query molecule。
2. 调用 retrieve_neighbors.py 预取每个 source-local Tier.endpoint_group 的 ChEMBL neighbor evidence。
3. 并发执行 single-molecule analysis；该历史 runner 中 DeepSeek 只可调用 molecule_properties。
4. 并发执行 endpoint-group analysis；该历史 runner 中 DeepSeek 只可调用 mmp_structure_compare 和 properties_compare。
5. final summary 读取 single + all group outputs，不暴露任何 tool。
6. 保存 retrieval/single/group/final/trace/manifest。
```

Single-molecule analysis 只能作为 DILI plausibility prior，不应直接决定 label。应关注：

```text
logP/logD and lipophilicity
molecular weight and polarity
ionization / cationic amphiphilicity
reactive or electrophilic structural alerts
possible acyl glucuronide / quinone-imine / Michael acceptor liabilities
high dose / exposure proxy if available
functional groups associated with mitochondrial or phospholipidosis risk
```

Group-level prompt 必须要求模型：

```text
1. 只分析当前 Tier.endpoint_group。
2. 判断 analog evidence 到 query 的 structural/property transferability。
3. 区分 direct DILI phenotype、in vivo liver injury、mechanistic liability 和 weak proxy。
4. 不把 distant_analog 或 very_distant_analog 作为主要正负证据，除非 scaffold/mechanism 很清楚。
5. 对每条 key_evidence 输出 effect_on_dili_reasoning。
```

建议 group output schema：

```text
useful_for_dili_reasoning
transferability
evidence_direction
confidence
reasoning_summary
key_evidence
caveats
```

`key_evidence` 使用当前统一格式，不要使用旧的 `key_neighbors`：

```text
key_evidence:
  - molecule_chembl_id
    similarity
    similarity_bucket
    assay_signal
    activity_values
    tool_summary
    transferability
    effect_on_dili_reasoning
```

新增 `effect_on_dili_reasoning` 后，必须同步检查 `tools/trace_viewer/viewer.html` 的 key evidence
渲染逻辑；否则 trace 原始 JSON 有值但 viewer 可能显示为空。

Final prompt 必须围绕 DILI，而不是 broad clinical toxicity：

```text
dili_prediction:
  dili_risk | no_dili_risk

required fields:
  confidence
  main_reasons
  single_molecule_dili_prior
  direct_human_dili_assessment
  in_vivo_liver_injury_assessment
  cholestasis_transporter_assessment
  mitochondrial_organelle_stress_assessment
  reactive_metabolite_immune_assessment
  hepatic_cell_exposure_assessment
  conflicting_evidence
  evidence_gaps
  final_summary
```

Final decision rules:

```text
1. Tier 1 direct human/clinical DILI anchor is strongest, especially severe liver outcome, Hy's-law-like signal,
   liver-related discontinuation/withdrawal/warning, or direct DILI label.
2. Tier 2 in vivo liver phenotype can support dili_risk when liver-specific and transferable; generic systemic
   toxicity without liver finding is not DILI evidence.
3. Tier 3-5 mechanisms can support dili_risk when strong, close-transferable and coherent, especially when multiple
   mechanisms agree or pair with liver-cell injury/exposure context.
4. Tier 6 alone usually cannot determine positive label. It can raise or lower confidence and explain plausibility.
5. Negative evidence must be endpoint-specific. A negative BSEP assay does not exclude mitochondrial or reactive
   metabolite DILI; negative HepG2 viability does not exclude idiosyncratic immune-mediated DILI.
6. If evidence is weak, distant, generic, non-liver-specific or contradictory, prefer no_dili_risk and express
   uncertainty through confidence/evidence_gaps rather than inventing a positive mechanism.
```

## Exact ChEMBL context

`chembl_exact_context.py` 是可选 evidence-rich 增强。它会用 query full InChIKey 查 ChEMBL exact molecule，
并在 retrieved neighbor 涉及的 assay 中查 query activity，生成：

```text
same_endpoint_activity:
  same assay_chembl_id + same normalized standard_type

same_assay_different_endpoint_activity:
  same assay_chembl_id + different standard_type
```

这会使用 query molecule 的已知 ChEMBL 实验记录，可能造成 prospective benchmark 的数据泄漏。因此默认关闭；
只有显式传 `--enable-chembl-exact-context` 时才用于 retrospective / evidence-rich case study。默认批量评估
不要开启。

默认 single-molecule prompt 不包含任何 ChEMBL 相关 payload 或 instruction。只有开启 exact context 且命中
query exact context 时，single-molecule payload 才包含 `exact_query_chembl_context`，并提示模型区分
direct same-molecule ChEMBL DILI evidence 和 physicochemical prior。ChEMBL neighbor evidence 仍只进入
group-level context。

## 参考资料

本 ontology 参考了 DILI clinical guidance、FDA LTKB/DILIrank 和机制综述。后续如修改 tier，应优先复核
这些资料以及更新的 regulatory / hepatotoxicity review。

```text
TDC Toxicity / DILI task:
  https://tdcommons.ai/single_pred_tasks/tox/

FDA Liver Toxicity Knowledge Base:
  https://www.fda.gov/science-research/bioinformatics-tools/liver-toxicity-knowledge-base-ltkb

FDA DILIrank 2.0:
  https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-20-dataset

FDA DILI premarketing clinical evaluation guidance:
  https://www.fda.gov/media/116737/download

EASL Clinical Practice Guidelines: Drug-induced liver injury:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9186030/

Mechanisms of drug induced liver injury:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12106265/

Drug induced cholestasis:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3089004/

Mitochondria as the target of hepatotoxicity and DILI:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8951158/

Mitochondrial dysfunction as a mechanism of DILI:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6261533/

Idiosyncratic DILI: mechanistic and clinical challenges:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7998339/

Immune mechanisms of idiosyncratic DILI:
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6410666/

High lipophilicity and high daily dose rule-of-two:
  https://pubmed.ncbi.nlm.nih.gov/23258593/

High-content screening for hepatic oxidative stress:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7828515/
```
