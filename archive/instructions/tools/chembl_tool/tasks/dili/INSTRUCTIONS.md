# DILI task notes

## Active data ownership

Active data lives with its semantic owner: gold-bound data under
`data/gold_labels/<Task>/<version>/`, evidence data under its task/release, and
shared reusable caches under `data/caches/`. `data/artifacts/` is audit-only and
must not be a required build or runtime input; complete retired products belong
under `data/legacy/`. Do not add compatibility symlinks.

The evidence-library pipeline owns scientific level assignment. Preserved
gold-version mappings live under `data/gold_labels/<Task>/level_mappings/<version>/`.
BBB and Bioavailability runtime consumers use the active release-owned
`data/evidence_libraries/<task>/<release>/level_mapping/`; Ames, DILI,
Carcinogens, and Skin keep their gold-owned mappings until reviewed replacements.
Voter membership may validate L1 coverage but must never derive or rewrite levels.
Corrections and publication belong to the evidence-library pipeline and must use
reviewed UID decisions with pinned input hashes.

## Testing discipline

Do not add circular tests that merely assert newly written prompt prose or copy
implementation literals into the test. Prompt wording is validated with reviewed
input/output fixtures or a real pilot/evaluation. Automated tests should cover
executable behavior, failure modes, schemas, rendering validity, and provenance.

This document records DILI task-specific semantics, mechanism-driven evidence ontology, assay screening direction,
endpoint group design, and reasoning constraints. General ChEMBL workflow, tool service, batch/resume,
trace viewer, output directory, and cost specifications still follow the repository root `AGENTS.md`.

DILI is not currently in the four-task, 21-condition paper matrix, and does not yet have `experiment_config.py`. Most of the
operational notes in this document are for reproducing the 2026-06 native endpoint-group runner. If DILI is to be
included in the current paper framework, the fine-grained groups must first be mapped to a small number of
data-source-independent mechanism families, and then the general paper runner performs family-level reasoning; do not
directly upgrade the old endpoint groups one by one into paper branches.

## Task definition

The goal is not to train an ordinary hepatotoxicity QSAR classifier, but to build an auditable DILI evidence retrieval
and reasoning workflow: given a query molecule, retrieve similar-molecule experimental readings related to DILI
mechanisms from ChEMBL or a later Starling evidence source, and then have a reasoning LLM determine whether these
analog evidence can transfer to the query molecule.

The current plan adapts the TDC DILI binary classification data:

```text
data/gold_labels/legacy/processed/DILI/train.jsonl
data/gold_labels/legacy/processed/DILI/valid.jsonl
data/gold_labels/legacy/processed/DILI/test.jsonl

fields:
  drug: query SMILES
  Y: DILI label
```

Evaluation convention:

```text
Y=1 -> dili_prediction=dili_risk
Y=0 -> dili_prediction=no_dili_risk
final summary must choose one of dili_risk/no_dili_risk; do not output uncertain prediction
```

TDC DILI / LTKB / DILIrank type labels are drug-level labels of human DILI concern, not equivalent to any
in vitro hepatocyte toxicity, any CYP/transporter inhibition, or generic cytotoxicity. When reasoning, strictly
separate direct human DILI evidence, in vivo liver injury phenotypes, key mechanism liabilities, and weak proxies.

## Legacy native runner boundaries

```text
ChEMBL neighbor retrieval is not a DeepSeek-callable tool.
ChEMBL neighbor retrieval is also not a current FastAPI service tool.
It is an evidence prefetch / context assembly step inside run_reasoning_pipeline.py.

The only tools callable by DeepSeek group-level analysis are:
  mmp_structure_compare
  properties_compare

The only tools callable by DeepSeek single-molecule analysis are:
  molecule_properties
```

The DILI pipeline should reuse the engineering structure of the existing general task workflow, but DILI's evidence
tier, endpoint group, prompt, and final decision rule must be DILI-specific. Do not inherit non-liver safety branches
such as hERG, neurotoxicity, renal toxicity, or genotoxicity from ClinTox's broad safety ontology; these may at most
serve as exclusion or background context and must not enter the DILI main evidence tier.

## Task-specific file planning

When creating new code later, it is recommended to maintain a thin wrapper structure similar to BBB_Martins / Skin_Reaction:

```text
constants.py
  label / prediction mapping. Recommended: Y=1 -> dili_risk, Y=0 -> no_dili_risk.

rules.py
  DILI assay keyword, negative keyword, mechanism family, and weak/context configuration.

scoring.py
  Entry points for assay screening / rescoring retention, exclusion, and scoring.

endpoint_groups.py
  DILI Tier.endpoint_group, evidence_direction, evidence_strength, and endpoint assignment rules.

screen_assays.py / rescore_outputs.py / summarize_outputs.py / report.py
  Thin wrappers that reuse the common task workflow to generate candidate assays, activity evidence, health checks, and reports.

build_evidence_library.py
  Builds a molecule-level evidence library and neighbor index from DILI assay candidates + activity evidence.

retrieve_neighbors.py
  The old native runner performs analog retrieval for each source-local Tier.endpoint_group.

chembl_exact_context.py
  Exact-query ChEMBL context wrapper. Not enabled by default in benchmarks to avoid retrospective leakage.

run_reasoning_pipeline.py
  Old DILI-specific retrieval prefetch, single-molecule branch, endpoint-group concurrent reasoning, final summary,
  trace saving, and --resume-final-from-run-dir final-only rerun.

run_reasoning_batch.py
  Batch reasoning wrapper. Reuses common reasoning_batch.py, outputs predictions, metrics, report, logs,
  runs, and combined trace; supports --skip-existing checkpoint resume. Currently integrated with the public global
  prompt stage pool, but the old DILI pipeline has not yet migrated to the shared experiment retrieval/identity/context
  contract, so batch only allows the default `native + operational + similarity + standard`; the public parser rejects parent-disjoint, coverage, or
  branch-reuse parameters to avoid manifest declarations exceeding actual retrieval capability.
```

Do not add new business rules in the wrappers; DILI assay retention/exclusion logic should be placed only in
`rules.py` and `scoring.py`, endpoint-group semantics only in `endpoint_groups.py`, and prompt/schema semantics only in
`run_reasoning_pipeline.py`.

## Current implementation status

On 2026-06-29, an executable screening / retrieval skeleton for DILI task v0 was created; on 2026-06-30, full
screening, full distribution review, evidence calibration, and the DILI-specific reasoning pipeline/schema were completed:

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

Current implementation boundaries:

```text
rules.py / scoring.py / endpoint_groups.py already implement the DILI v0 ontology:
  Tier 1 direct human/clinical DILI
  Tier 2 in vivo liver injury
  Tier 3 cholestasis/hepatobiliary transporter
  Tier 4 mitochondrial/oxidative/organelle stress
  Tier 5 reactive metabolite/bioactivation/immune-idiosyncratic
  Tier 6 hepatic cell injury/exposure-property modifiers

screen_assays.py / rescore_outputs.py / summarize_outputs.py / report.py are integrated with the common workflow.
build_evidence_library.py / retrieve_neighbors.py / chembl_exact_context.py are integrated with the common retrieval/index workflow.
run_reasoning_batch.py is configured with the dili_prediction label mapping.
run_reasoning_pipeline.py implements the DILI-specific prompt/schema, single-molecule branch, group branch,
final branch, tool orchestration, trace output, and final-only resume.
tools/trace_viewer/viewer.html has been adapted for dili_prediction, useful_for_dili_reasoning,
effect_on_dili_reasoning, and DILI-specific assessment fields.
```

## Historical TRIM / DeepSeek properties-only baselines

On 2026-06-29, three Intern-S1/TRIM no-retrieval properties-only DeepSeek-v4-pro baselines were run
as historical reference comparisons against the current DILI-specific ChEMBL retrieval pipeline. They do not use the
current TxAgent `run_reasoning_pipeline.py`, nor ChEMBL/Starling retrieval; the prompt comes from
`trim.reasoning.task_user_prompts.render_task_user_message`, tool mode is `properties`,
the only visible tool is `get_mol_properties_and_fg`. The data split uses
`/data1/tianang/Projects/Intern-S1/DataPrepare/TDC_no_conflict_labels_salt_removed/test/DILI.jsonl`
with 96 samples.

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

Current tests:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  /data1/tianang/anaconda3/envs/vllm/bin/pytest tests/chembl_tool/tasks/dili -q

result:
  56 passed
```

In the `vllm` environment, pytest plugin auto-scanning may hang on conda dist-info entry point reading on the
current machine; use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` when running DILI unit tests. This is not a DILI code issue.

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

2026-06-29 smoke screening / evidence QA current calibration status:

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

Next workflow steps:

```text
1. For model evaluation, directly run run_reasoning_batch.py with the v7_full evidence library.
2. By default, do not enable --enable-chembl-exact-context in benchmarks to avoid same-molecule ChEMBL evidence leakage.
3. Next round, expand from a 6-sample smoke test to a 20-sample balanced smoke test, focusing on false positive/false negative traces.
4. If running DILI with GLM-5.2, follow the --disable-thinking / reasoning_effort="" compatibility parameters in the root AGENTS.md.
5. Starling acquisition uses the frozen DILI mechanism family as the task/prompt granularity; endpoint subtype, species,
   dose, and assay context are schema fields within the family. The old Tier.endpoint_group is only used for access audit, not for running Starling per group.
```

## General principles for DILI evidence

DILI is a clinical phenotype, not a single mechanism. A drug may cause DILI through multiple pathways such as bile acid transporter interference, mitochondrial damage,
reactive metabolites, oxidative/ER stress, immune-mediated reactions, hepatocyte death, and hepatic exposure/dose.
Therefore, evidence tiers should be designed based on "distance from the DILI label + mechanistic interpretability + ability to independently construct data for subsequent Starling,"
rather than arbitrarily subdivided by ChEMBL keywords.

The initial DILI tiers should be few, runnable, and interpretable:

```text
Tier 1: direct human or clinical DILI anchors
Tier 2: in vivo liver injury phenotype and clinical pathology
Tier 3: cholestasis and hepatobiliary transporter liability
Tier 4: mitochondrial, oxidative and organelle stress
Tier 5: reactive metabolite, bioactivation and immune/idiosyncratic liability
Tier 6: hepatic cell injury models and exposure/property modifiers
```

These tiers can be consolidated into mechanism families for future Starling acquisition. In principle, each family corresponds to one
Starling task; do not over-split mechanisms, for example, do not split
BSEP, MRP2, NTCP, MDR3 into separate tiers; they belong to the same cholestasis / hepatobiliary transporter
mechanism axis. Also, do not split ROS, ATP, MMP, ER stress into separate tiers; they belong to the organelle stress mechanism axis.

Important constraints:

```text
1. Tier 1 is always the most direct DILI measurement or human/clinical liver safety outcome.
2. Tier 2 is organism-level liver injury phenotype, stronger than in vitro proxies, but still requires species, route,
   dose, duration, and exposure context.
3. Tiers 3-5 are key mechanistic liabilities. They can strongly support DILI risk, but a single weak positive usually cannot replace a direct DILI
   phenotype.
4. Tier 6 is useful supporting evidence and should not alone classify a query as dili_risk unless consistent with higher tiers or multiple mechanistic
   branches.
5. Exact-query ChEMBL context is disabled by default; benchmarks should not directly inject known ChEMBL liver evidence for the query molecule
   into the prompt.
6. ChEMBL is currently only a substitute evidence source before Starling. The ontology should not be driven by the abundance of existing ChEMBL assays.
```

## Evidence library fields

Each activity evidence in the DILI evidence library must retain at least:

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

Internally, `evidence_direction`, `evidence_strength`, `endpoint_group_reason`, `assay_reason` can be retained
for debugging and auditing; the activity evidence row in the LLM payload contains only the original ChEMBL assay/activity fields and
necessary metadata. Do not send the entire package of internally derived rule fields to the reasoning LLM.

Suggested `evidence_direction`:

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

Suggested `evidence_strength`:

```text
strong
moderate
weak
context
```

Strength is not synonymous with assay score. It indicates the explanatory distance of that evidence type to the DILI label:

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

## Evidence type interpretation

DILI evidence tiers should not copy BBB / Bioavailability / ClinTox. Here, they are split by distance to the DILI label and mechanism axis.
Tier 1 is the most direct measurement; Tiers 2-6 are progressively down-weighted evidence from in vivo phenotype to mechanistic proxy.

### Tier 1: direct human or clinical DILI anchors

Closest to the TDC DILI label. Prioritize retaining drug-induced liver injury evidence that clearly occurs in humans, clinical studies, postmarketing, drug labels, or human case contexts.

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

Strong interpretation conditions:

```text
1. The assay description, source, or metadata clearly indicates human / clinical / patient / volunteer / trial /
   postmarketing / FDA label / LiverTox-like context.
2. The endpoint is DILI, hepatotoxicity, liver failure, jaundice, Hy's-law-like lab pattern,
   liver enzyme elevation with bilirubin, or liver-related discontinuation/withdrawal/warning.
3. The activity row shows direction, e.g., "positive", "elevated", "injury", "hepatotoxic", "withdrawn",
   "not tolerated", "liver failure", "acute liver failure".
```

Weak interpretation or exclusion conditions:

```text
1. Routine mild ALT/AST monitoring without bilirubin, symptoms, dose interruption, or severe outcome is usually
   monitoring evidence, not equivalent to a positive DILI label.
2. Liver lab abnormalities in oncology / antiviral / anti-infective efficacy trials must be distinguished by disease context,
   combination therapy, and high-dose treatment context.
3. "no liver injury", "no ALT elevation", "well tolerated" can only serve as counter-evidence when dose/exposure and duration are clear.
```

### Tier 2: in vivo liver injury phenotype and clinical pathology

Animal or non-clinical in vivo liver injury phenotype. Closer to human DILI than most in vitro assays, but transferability
depends on species, route, dose, duration, metabolite coverage, and exposure margin.

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

Interpretation rules:

```text
histopathology:
  Liver tissue injury is strong/moderate evidence, especially necrosis, bile duct injury, inflammation,
  or repeated-dose findings.

clinical chemistry:
  ALT/AST/ALP/bilirubin/bile acids in an in vivo context are liver injury phenotypes, but fold change,
  dose, duration, and reversibility must be considered.

NOAEL/LOAEL/MTD:
  Only include in DILI reasoning when there is a clear liver finding or liver clinical chemistry; generic MTD/NOAEL
  without liver context is not retained in the main DILI tiers.
```

### Tier 3: cholestasis and hepatobiliary transporter liability

Bile acid homeostasis and hepatobiliary transporters are among the clearest and most assayable mechanistic axes in DILI. This tier covers BSEP, MRP2,
MDR3, NTCP, OATP and other hepatobiliary transporters, as well as bile acid accumulation / cholestasis phenotypes.

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

Interpretation rules:

```text
BSEP/ABCB11 inhibition:
  Is core mechanistic evidence for cholestatic DILI risk. Strength depends on potency, assay system, free exposure
  margin, bile acid accumulation, and whether mitochondrial/hepatocyte injury is present.

MRP2/MDR3/NTCP/OATP:
  Support hepatobiliary disposition / bile acid handling context. Single transporter inhibition is usually
  moderate/weak unless consistent with a cholestasis phenotype or multi-transporter liability.

cholestasis phenotype:
  If it is human or in vivo cholestatic injury, prioritize interpretation as Tier 1 or Tier 2; Tier 3 retains the mechanistic dimension.
```

### Tier 4: mitochondrial, oxidative and organelle stress

Mitochondrial dysfunction, ATP depletion, oxidative stress, ER stress, and lysosomal/phospholipidosis and other organelle
stress are important mechanistic axes in DILI. This tier merges these intertwined stress pathways to avoid over-splitting for Starling.

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

Interpretation rules:

```text
mitochondrial assays:
  MMP/OCR/ATP readouts in hepatic or metabolically competent cell contexts are moderate/strong mechanistic evidence.
  Generic non-hepatic mitochondrial readouts are usually downgraded to weak/context.

oxidative stress:
  ROS/GSH/Nrf2 are DILI-related stress evidence, but a single reporter assay does not equal DILI.

phospholipidosis/steatosis:
  Useful for cationic amphiphilic or lipid-disposition liability, usually moderate/weak unless consistent with liver
  phenotype or hepatocyte injury.
```

### Tier 5: reactive metabolite, bioactivation and immune/idiosyncratic liability

Many idiosyncratic DILI cases are associated with bioactivation, reactive metabolites, covalent binding, GSH adducts, drug-protein
adducts, danger signals, and adaptive immune responses. This tier merges bioactivation and immune/idiosyncratic
contexts because they often co-occur in evidence retrieval, and splitting them would significantly increase the number of Starling branches.

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

Interpretation rules:

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

This tier covers evidence that is closer to liver biology but still proxy: hepatocyte/HepaRG/HepG2/3D spheroid injury,
high-content hepatotoxicity, transcriptomic liver stress, and dose/exposure/
property context that modifies DILI plausibility. It is useful supporting evidence and should not replace Tier 1-5 alone.

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

Interpretation rules:

```text
hepatocyte viability:
  Liver-cell context is more relevant than generic cytotoxicity, but still requires concentration, time, metabolic competence,
  and assay specificity. High-concentration nonspecific cytotoxicity is usually weak.

omics / HCS:
  Broader mechanistic coverage can serve as strong supporting evidence when liver-specific and replicated, but avoid automatically
  equating any stress signature with clinical DILI.

exposure/property:
  High dose + high lipophilicity, strong hepatic metabolism, cationic amphiphilicity, etc., can increase DILI plausibility. They are priors
  or modifiers, not direct evidence.
```

### Weak / excluded context

The following evidence does not enter the DILI main tier unless the assay description explicitly provides liver/DILI context:

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

These lines can be retained as excluded / context-dependent in the debug report, but should not be fed into the DILI reasoning prompt as
positive evidence.

## Endpoint Group Standards

In the second phase, do not retrieve per assay individually. Instead, use:

```text
Tier -> endpoint_group
```

Combinations generate retrieval groups. The initial number of endpoint groups should be restrained to facilitate subsequent Starling runs per tier.

Suggested initial grouping:

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

If Starling runtime costs need further reduction, the default Starling acquisition can be merged by tier, not split by endpoint_group;
endpoint_group serves only as retrieval/ranking and prompt internal structure.

## Assay screening rule directions

Prioritize retaining:

```text
1. Explicit human/clinical/postmarketing/labeled DILI or liver adverse event.
2. Explicit in vivo liver pathology, liver clinical chemistry, or liver toxic dose/margin.
3. BSEP/bile acid/hepatobiliary transporter functional readout.
4. Hepatic mitochondrial/OCR/MMP/ATP/ROS/GSH/ER stress assay.
5. Reactive metabolite/GSH/covalent binding/bioactivation with liver context.
6. Hepatocyte/HepaRG/HepG2/3D liver model/high-content liver toxicity.
7. Dose/lipophilicity/hepatic metabolism/exposure context only as modifier evidence.
```

Actively exclude:

```text
1. Only liver cancer efficacy, hepatocellular carcinoma anti-proliferation, or antiviral efficacy.
2. Only generic target inhibition/binding, without liver injury, bile acid, bioactivation, or hepatic exposure context.
3. Only CYP inhibition IC50, without substrate/bioactivation/exposure or liver injury context.
4. Only generic cytotoxicity in non-hepatic cell lines.
5. Cardiac, renal, neuro, genotox, skin reaction, or other non-liver safety evidence.
6. Disease biology assays, such as fibrosis/inflammation target activity, unless the readout is compound-induced
   liver injury/toxicity.
```

## Legacy native reasoning pipeline constraints

```text
1. Read the query molecule from the DILI test JSONL.
2. Call retrieve_neighbors.py to prefetch ChEMBL neighbor evidence for each source-local Tier.endpoint_group.
3. Concurrently execute single-molecule analysis; in this historical runner, DeepSeek can only call molecule_properties.
4. Concurrently execute endpoint-group analysis; in this historical runner, DeepSeek can only call mmp_structure_compare and properties_compare.
5. Final summary reads single + all group outputs, without exposing any tool.
6. Save retrieval/single/group/final/trace/manifest.
```

Single-molecule analysis should only serve as a DILI plausibility prior and should not directly determine the label. Focus on:

```text
logP/logD and lipophilicity
molecular weight and polarity
ionization / cationic amphiphilicity
reactive or electrophilic structural alerts
possible acyl glucuronide / quinone-imine / Michael acceptor liabilities
high dose / exposure proxy if available
functional groups associated with mitochondrial or phospholipidosis risk
```

Group-level prompts must require the model to:

```text
1. Only analyze the current Tier.endpoint_group.
2. Judge the structural/property transferability of analog evidence to the query.
3. Distinguish direct DILI phenotype, in vivo liver injury, mechanistic liability, and weak proxy.
4. Not use distant_analog or very_distant_analog as primary positive or negative evidence unless scaffold/mechanism is clear.
5. Output effect_on_dili_reasoning for each key_evidence.
```

Suggested group output schema:

```text
useful_for_dili_reasoning
transferability
evidence_direction
confidence
reasoning_summary
key_evidence
caveats
```

`key_evidence` Use the current unified format, not the old `key_neighbors`:

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

After adding `effect_on_dili_reasoning`, you must synchronously check the key evidence
rendering logic of `tools/trace_viewer/viewer.html`; otherwise, the trace raw JSON may have values but the viewer may display empty.

The final prompt must focus on DILI, not broad clinical toxicity:

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

`chembl_exact_context.py` is an optional evidence-rich enhancement. It uses the query full InChIKey to look up the ChEMBL exact molecule,
and queries query activity in assays involved in retrieved neighbors, generating:

```text
same_endpoint_activity:
  same assay_chembl_id + same normalized standard_type

same_assay_different_endpoint_activity:
  same assay_chembl_id + different standard_type
```

This uses the query molecule's known ChEMBL experimental records, which may cause data leakage in prospective benchmarks. Therefore, it is disabled by default;
only use it for retrospective / evidence-rich case studies when explicitly passed `--enable-chembl-exact-context`. Do not enable it in default batch evaluations.

The default single-molecule prompt does not include any ChEMBL-related payload or instruction. Only when exact context is enabled and query exact context is hit,
the single-molecule payload includes `exact_query_chembl_context`, and prompts the model to distinguish
direct same-molecule ChEMBL DILI evidence from physicochemical priors. ChEMBL neighbor evidence still only enters
group-level context.

## References

This ontology references DILI clinical guidance, FDA LTKB/DILIrank, and mechanistic reviews. If tiers are modified later, prioritize reviewing
those materials and updated regulatory / hepatotoxicity reviews.

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
