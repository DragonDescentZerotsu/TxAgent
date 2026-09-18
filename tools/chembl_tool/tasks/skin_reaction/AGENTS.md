# Skin_Reaction task notes

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

This file documents Skin_Reaction task-specific semantics, ChEMBL evidence ontology, assay screening rule directions,
endpoint group design, and reasoning constraints. General ChEMBL workflow, batch/resume, viewer, and output directory conventions are governed by the repository root
`AGENTS.md`.

The current paper-facing Starling path retains only two mechanism families aligned with binary endpoints: direct skin
sensitization outcome and sensitisation AOP key events. Phototoxicity/irritation/local damage and skin exposure
immutable raw acquisition remain for historical audit but do not enter the current Starling reasoning view. ChEMBL historical
ontology and native runner still retain old Tier descriptions for reproduction; do not launch parallel reasoning for each fine-grained group based on them.

2026-07-23 strict-hop availability census is in
`outputs/chembl_tool/tasks/skin_reaction/distance_expansion/analysis/hop_availability_census/`. The graph/data gate from GSK3B activity
to NRF2 state passed (7,950 parents; parent-disjoint >=1 coverage 34.15%), but direct pharmacological
NRF2 regulation is not sensitizer-specific and must retain a scope caveat. KEAP1-NRF2 PPI coverage is only 2.44%,
CUL3 H2 is 0; this task has no usable strict H2. Do not add old paper conditions based on this.

2026-07-22 imported Starling direct Skin_Reaction parquet, as well as three mechanism-family acquisitions: sensitization AOP, phototoxicity/irritation/local
damage, and skin exposure. `build_starling_evidence_library.py` uses a public profile
reader to build the four data types into a molecule-level index, shared by the three paper conditions Starling direct/full-flat/full-mechanism.

## Task definition

The goal is not to train a simple QSAR skin-reaction classifier, but to build an auditable skin-reaction evidence retrieval
and reasoning workflow: given a query molecule, retrieve similar molecule experimental readings relevant to skin adverse reaction judgment from ChEMBL,
then have a reasoning LLM determine whether these analog evidences can transfer to the query molecule.

The current only condition-aware Starling gold benchmark:

```text
data/gold_labels/Skin_Reaction/v1/scaffold/{train.jsonl,valid.jsonl,test.jsonl}

fields:
  drug: query SMILES
  Y: Skin_Reaction label
```

The current build has 2,491 molecule-condition rows, with train/valid/test at 1,997/246/248; the old molecule-only and
selected-vN paths are only migration provenance. Conflicting parents are calculated with 70% agreement based on accepted source records,
multiple records from the same PMID each count as votes, and exact ties are rejected. Before formal runs, the train-only retrieval index must be rebuilt from the heldout
detailed labels of the scaffold valid+test union. The unified contract is in
`data/processing/gold_labels/README.md`.

Current binary classification convention:

```text
Y=1 -> skin sensitizer / positive
Y=0 -> non-sensitizer / negative
```

The original 404-molecule task comes from binary LLNA skin-sensitization data, not arbitrary clinical dermatologic
reactions. The new Starling-held-out benchmark therefore only accepts explicit positive/negative records within the scope of sensitization or allergic contact
dermatitis/contact allergy; irritation, generic local damage,
skin exposure, and inconclusive results are not converted to gold labels.

Historical `run_reasoning_pipeline.py` final prompt/schema allowed phototoxicity and irritation/corrosion to become
`risk` main evidence types; this is inconsistent with the above sensitization-only gold scope and is a confirmed
`legacy_skin_reaction_v1` prompt bug. On 2026-08-08, a versioned
`sensitization_aligned_v2` was implemented in `prompt_profiles.py` and set as the default for new runs: single/group/final are all explicitly limited to sensitization/contact
allergy; phototoxicity, irritation/corrosion, generic local damage, and exposure can only be out-of-scope/context.
Manifest records `task_prompt_profile` and `label_scope`; when reusing single/group/final branches, profiles must be consistent.
When old manifests lack a profile field, they are fixed to be interpreted as legacy v1; historical results and final-evidence-surface replay continue to explicitly use
legacy v1. On 2026-08-09, the aligned-v2 GPT-OSS-120B scaffold-valid four conditions were completed: none/direct/full-flat/
full-mechanism macro-F1 is `0.5225/0.5725/0.5698/0.5423`, all 245/245 successful. Scope-contaminated errors dropped from 23 in
legacy to 0, proving the contract fix is effective; but full-mechanism performance did not improve, and the fresh-run paired delta also crosses 0,
so the scope fix must not be presented as a performance method.

On the same day, a model-free audit was performed on frozen GPT-OSS-120B scaffold-valid `starling_full_mechanism` legacy traces:
of 90 errors, only 6 had clean gold-aligned Tier 1/2 signals and the final still chose wrong; 84 were upstream conflicts, wrong
direction, or insufficient. 23 erroneous finals had main evidence types crossing label scope, but only 2 were strictly
final-recoverable. Machine-readable results and method limitations are in:

```text
tools/chembl_tool/paper_experiments/audit_skin_reasoning_bottleneck.py
outputs/paper/skin_reaction_task_alignment_audit_record_supported_v2_valid_gpt_oss_120b/
outputs/paper/skin_reaction_task_alignment_audit_record_supported_v2_valid_gpt_oss_120b_sensitization_aligned_v2/
```

### Direct evidence scope parity (2026-08-09)

The gold builder has always accepted only sensitization/contact-allergy records, but the historical Starling Tier-1 evidence profile aggregated
photoallergy, irritation, urticaria, and other broad-skin records into the same direct card. This is a source-to-agent
scope mismatch that cannot be stably reversed from aggregated counts and at most 6 examples by prompt alone.

`build_starling_evidence_library.py`'s historical v2 profile uses `sensitization_contact_allergy_v2`, directly reusing
`starling_benchmark.is_tdc_skin_sensitization_scope()`; historical behavior is preserved through explicit
`--source-profile broad_skin_reaction_v1`. The public `StarlingSourceProfile.record_filter` handles general row-scope
filtering and records filter name/count in metadata. The new full-source direct profile filters 5,049 out-of-scope records from 66,597 input rows,
retaining 44,752 loadable records and 3,275 direct molecules; the old v1 artifact is not covered.

Deterministic retrieval audit on all 245 scaffold-valid queries shows: 37 top-3 neighbor identity lists
changed, 41 historical retrieved neighbors were removed from the union, 20 scoped neighbors were backfilled; since card text/counts were rebuilt
synchronously, 203 LLM-visible direct contexts changed. The clean-index `sensitization_aligned_v2` direct fresh run is
245/245 successful, macro-F1 `0.5666`, with `0.5725` relative to historical broad-index aligned-v2 direct being `-0.0059`
(paired-bootstrap 95% CI `[-0.0724,+0.0603]`; 55 flips, 28/27 old/new-only correct). Therefore scope parity is a
data contract fix, not an observed performance improvement.

In clean-index traces, low/moderate-transferability negative direction still has 6 gold-aligned and 13
gold-opposed, so the only versioned candidate `sensitization_negative_transfer_v3` was run: analog-only negative can support no-risk only when
high transferability, sufficiently exposed validated assays, and consistent records. The rule indeed reduced negative
direction from 25 to 3, and the remaining 3 are all gold-aligned; but macro-F1 dropped to `0.5531`, Y=0 recall from `0.4658`
to `0.3425`. The delta relative to clean-index v2 is `-0.0135` (95% CI `[-0.0882,+0.0612]`). This profile is retained only as a
failed experimental lineage, not the default, and is not extended to full-flat/full-mechanism/test.

```text
tools/chembl_tool/paper_experiments/audit_skin_direct_scope_retrieval.py
outputs/paper/skin_direct_scope_retrieval_audit_record_supported_v2_valid/
outputs/paper/skin_negative_transfer_v3_audit_record_supported_v2_valid/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_gpt_oss_120b_skin_direct_scope_v2/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_gpt_oss_120b_skin_direct_scope_v2_negative_transfer_v3/
```

Reproduction entry points for scoped source/index and deterministic audit:

```bash
python -m tools.chembl_tool.tasks.skin_reaction.build_starling_evidence_library \
  --source-profile sensitization_contact_allergy_v2 --workers 32

python -m tools.chembl_tool.paper_experiments.build_starling_benchmark_indices \
  --splits scaffold --indices skin_reaction_starling_full \
  --source-evidence skin_reaction_starling_full=outputs/paper/molecular_evidence_agent/evidence/skin_reaction_starling_sensitization_v2/starling_skin_reaction_evidence.jsonl \
  --benchmark-data-root data/gold_labels/legacy/processed_starling_record_supported_v2 \
  --benchmark-lineage record_supported_v2_skin_direct_scope_v2 --workers 32

python -m tools.chembl_tool.paper_experiments.audit_skin_direct_scope_retrieval
```

### Canonical direct/AOP partition (2026-08-13)

Current inference evidence no longer directly reads the two acquisition parquets and aggregates separately; instead, a versioned canonical builder
performs mutually exclusive partitioning. Raw inputs remain immutable; each raw row is written to the partition audit:

```text
tools/chembl_tool/tasks/skin_reaction/canonical_starling_source.py
tools/chembl_tool/tasks/skin_reaction/build_canonical_starling_source.py
data/artifacts/starling/skin_reaction/canonical_sources/canonical_sensitization_v3/
```

The contract is: validated LLNA/GPMT/Buehler/human patch/contact-allergy and other final outcomes only enter direct; MIE, KE2,
KE3, KE4 experimental evidence only enter AOP; photo hazards, irritation/corrosion, prediction-only/in-silico,
integrated/unclassifiable endpoints are all rejected. Validated adverse-outcome rows in the AOP acquisition are transferred to canonical
direct, while raw sources and gold benchmarks are not rewritten. The canonical full-source has 54,596 direct records and 11,434
AOP records; AOP events are only `MIE/KE2/KE3/KE4`, and source-record direct/AOP overlap is 0. In the full-field scope audit,
photo, in-silico, integrated, and AOP irritation hits are all 0; records in direct that mention irritation are retained only when they also have an explicit
sensitization/contact-allergy outcome anchor; irritation itself is not used as label evidence.

Corresponding DeepSeek-v4-pro, MiniMol top-3, cosine >=0.3, scaffold-valid fresh runs are all 245/245, 0 failures: direct
macro-F1 `0.6410`, direct+AOP mechanism `0.6123`. Mechanism relative to direct delta `-0.0287`, paired-bootstrap
95% CI `[-0.0801,+0.0224]`; therefore canonical partition is a data contract fix, not promoting AOP mechanism as the default
performance condition.

On 2026-08-14 E22 also tested stricter outcome-calibrated multi-event causal cards: reference must have direct outcome,
MIE, at least one downstream KE, and consistent direction. The heldout-filtered pool has only 99 cards (89 positive, 10 negative).
After merging the frozen 64-query DeepSeek seed back into 245 entries, macro-F1 dropped from direct `0.6410` to `0.6151`; 10 flips are
2 beneficial/8 harmful. 55/64 branches were judged by the model as low-transferability, 48/64 neutral/unclear, proving that "causal completeness within reference"
is not equivalent to "transferable to query". E22 is not promoted, BBB seed is not started, Skin test is not read; code and
trace diagnosis are in `tools/chembl_tool/paper_experiments/skin_causal_panel_seed/` and
`outputs/paper/skin_causal_panel_seed_v1_scaffold_valid_deepseek_v4_pro/analysis/`.

### Distance semantics of paper-facing tiers

Skin's `Mechanism.tier_1` to `tier_4` are progressively expanding evidence scopes, not layer-by-layer causal decomposition like oral bioavailability
`F = Fa × Fg × Fh`:

```text
Tier 1:
  direct sensitization/contact-allergy anchors; closest to current gold.

Tier 2:
  sensitization AOP key events; aligned with gold, but a single key event is not equivalent to the final clinical outcome.

Tier 3:
  phototoxicity, irritation, corrosion, local skin damage; all are skin hazards,
  but not constituent mechanisms of the current sensitization label.

Tier 4:
  skin permeability, retention, and exposure context; only change the plausibility of hazard manifestation,
  cannot alone prove sensitization.
```

Therefore, expanding from direct to full is not adding increasingly complete parts of the same causal chain, but adding increasingly distant, possibly semantically
misaligned evidence. `experiment_config.py` is the code truth for paper-facing source/group mapping.

### 2026-07-27 Starling retrieval degradation trace audit

Observed macro-F1 for deployment-visible parent-disjoint Starling:

| split | direct | full flat | full mechanism |
|---|---:|---:|---:|
| random | 0.643141 | 0.635255 | 0.629991 |
| scaffold | 0.597332 | 0.592139 | 0.583574 |

Full-flat and full-mechanism LLM-visible evidence-row multisets are exact matches per query at 380/380 for both random and scaffold; mechanism does not get extra rows, it only splits the same union into multiple branches before final.
Prediction flips from direct to mechanism are:

```text
random:   38 flips, 17 corrected / 21 broken, net -4 correct
scaffold: 29 flips, 12 corrected / 17 broken, net -5 correct
```

Main failure modes in traces are: phototoxicity/irritation elevated to sensitization hazard, Tier 4 exposure
support mistaken as risk, weak/distant AOP narrative amplified by branch packaging, broad mixed negatives diluting
closer positive anchors, and prompt-boundary instability when no neighbors exist. Starling random average logical
tokens increase from direct 27.7k to flat 77.4k and mechanism 96.1k; scaffold is 26.9k, 73.4k, 91.6k.
More tokens mean more group calls/repeated synthesis, not more label-aligned information.

Full quantification and per-flip traces:

```text
outputs/paper/skin_reaction_retrieval_diagnostic/agent_quant_summary.json
outputs/paper/skin_reaction_retrieval_diagnostic/agent_random_flips.jsonl
outputs/paper/skin_reaction_retrieval_diagnostic/agent_scaffold_flips.jsonl
outputs/paper/skin_reaction_retrieval_diagnostic/report_trace_examples.csv
```

Current ChEMBL evidence version:

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

Current v1 screening status:

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

After the paper-view audit on 2026-07-12, Tier 5 remains in the ChEMBL source library for endpoint/data-quality
auditing, but no longer enters the paper-facing `full_flat` or `full_mechanism`, and no longer creates a separate LLM reasoning branch.
In the existing two 82-sample mechanism runs, Tier 5 covers 42 samples each and produces 84 group calls; identity-blind
never judged it useful, deployment-visible judged it useful only once and the direction was still `neutral_or_unclear`, consuming about
816k tokens in total. It mainly repeats that generic cytotoxicity, efficacy, or target binding cannot determine the Skin label,
with no strong positive effect observed.

This deletion only affects the reasoning view; original Tier 5 evidence is not deleted. Context such as concentration, vehicle,
formulation, duration, light condition, skin model, etc. for valid assays must continue to be retained with their respective Tier 1-4 evidence rows.
Skin full-mechanism exploratory runs generated before 2026-07-12 include Tier 5; runs after the configuration change must use
a new batch ID and cannot be mixed with old runs for checkpoint resume.

## Current code entry points and run status

Main entry points:

```text
constants.py
  label / prediction mapping. Current binary convention: Y=1 -> risk, Y=0 -> no_risk.

rules.py
  Skin_Reaction assay keyword, negative keyword, context/weak evidence family configuration.

scoring.py
  Entry point for assay screening / rescore retention, exclusion, and scoring.

endpoint_groups.py
  Tier.endpoint_group, evidence_direction, evidence_strength, and endpoint assignment rules.

experiment_config.py
  Paper-facing direct, full_flat, and 4-family full_mechanism retrieval views; Tier 5 does not enter the reasoning view.

screen_assays.py / rescore_outputs.py / summarize_outputs.py / report.py
  Thin wrappers that reuse the common task workflow to generate candidate assays, activity evidence, health checks, and reports.

build_evidence_library.py
  Builds molecule-level evidence library and neighbor index from v1 assay candidates + activity evidence.

build_starling_evidence_library.py
  By default builds the current two-family evidence/index from canonical direct/AOP parquet; historical broad/scoped-v2 sources
  can still be reproduced via explicit `--source-profile`. Default build first validates canonical manifest, partition reconciliation,
  direct/AOP zero overlap, and SHA-256 of the two parquet files. Current artifacts are written to
  `outputs/paper/molecular_evidence_agent/evidence/skin_reaction_starling_sensitization_canonical_v3/`.
  The default directory name for the heldout-parent filtered index is `skin_reaction_starling_sensitization_canonical_v3`; the old
  `skin_reaction_starling_full` belongs only to the historical source profile.

  Build command:
  `python -m tools.chembl_tool.tasks.skin_reaction.build_starling_evidence_library --workers 32 --progress-every 10000`

starling_benchmark.py
  Converts sensitization/contact-allergy records from the direct Skin_Reaction parquet into an auditable
  parent-level binary label; does not treat other skin mechanism families as gold outcomes.

retrieve_neighbors.py
  Old native runner performs analog retrieval for each source-local Tier.endpoint_group. Historical v1 benchmark uses top-k-per-group=3,
  min-similarity=0.35; investigation with min-similarity=0 shows index/retrieval is normal, low coverage mainly comes from
  chemical-space similarity threshold.

chembl_exact_context.py
  Exact-query ChEMBL context wrapper. Not enabled by default in benchmark to avoid retrospective leakage.

run_reasoning_pipeline.py
  Old native single-molecule reasoning pipeline: retrieval prefetch, single-molecule branch, endpoint-group concurrent reasoning,
  final summary, trace saving, and --resume-final-from-run-dir final-only rerun.

run_reasoning_batch.py
  Batch reasoning wrapper. Reuses common reasoning_batch.py, outputs predictions, metrics, report, logs,
  runs, and combined trace; supports --skip-existing checkpoint resume.
```

## Starling Tier 1+2 final-only diagnostics (2026-07-27)

To check whether distant Tier 3 (phototoxicity/irritation/local damage) and Tier 4 (skin exposure) evidence drags down the
sensitization gold label, a post-hoc scope ablation was performed on the random/scaffold deployment-visible parent-disjoint
`starling_full_mechanism`. The condition:

```text
Retain: frozen single-molecule output, Mechanism.tier_1, Mechanism.tier_2
Delete: Mechanism.tier_3, Mechanism.tier_4
Rerun: only final synthesis
Unchanged: query, retrieval policy, source artifacts, branch outputs, final prompt/schema, model/config
```

The public entry point is `reasoning_batch.py --final-only-source-batch ... --final-only-groups ...`. Must use
`--final-only-groups` for artifact-level filtering; ordinary `--groups` only controls group reasoning for fresh pipelines,
cannot replace resume-final filtering. Current batch:

```text
random:
  outputs/paper/molecular_evidence_agent_starling_random/runs_deployment_visible_parent_disjoint/
    skin_reaction/skin_reaction__starling_tier12_final_only/

scaffold:
  outputs/paper/molecular_evidence_agent_starling_scaffold/runs_deployment_visible_parent_disjoint/
    skin_reaction/skin_reaction__starling_tier12_final_only/
```

Both are 380/380 successful, 0 failures. Per-sample checks confirm single outputs are byte-identical to the source batch,
retained group outputs are exactly the Tier 1/2 subset from the source, and retrieval/group/trace have no Tier 3/4 group leakage.
Results:

| split | condition | macro-F1 | accuracy | TN / FP / FN / TP |
|---|---|---:|---:|---|
| random | direct | 0.643141 | 0.663158 | 81 / 48 / 80 / 171 |
| random | Tier 1+2 final-only | 0.631003 | 0.652632 | 78 / 51 / 81 / 170 |
| random | full mechanism | 0.629991 | 0.652632 | 77 / 52 / 80 / 171 |
| scaffold | direct | 0.597332 | 0.634211 | 63 / 54 / 85 / 178 |
| scaffold | Tier 1+2 final-only | 0.594785 | 0.626316 | 66 / 51 / 91 / 172 |
| scaffold | full mechanism | 0.583574 | 0.621053 | 61 / 56 / 88 / 175 |

Relative to full mechanism, the paired macro-F1 delta for Tier 1+2 is random `+0.001013`
(13 better / 13 worse; bootstrap 95% CI `[-0.026550, 0.028667]`) and scaffold `+0.011211`
(10 better / 8 worse; 95% CI `[-0.009740, 0.033671]`). This indicates that cutting Tier 3/4 gives a small
point-estimate recovery on scaffold, but both intervals cross 0 and neither exceeds direct; this test-driven post-hoc result can only serve as a
failure diagnostic, not as a new pre-registered primary condition.

Historical native v1 full test results (TDC lineage, not the current Starling split):

```text
batch:
  outputs/chembl_tool/tasks/skin_reaction/reasoning/batches/skin_reaction_calib_50_v1

run settings:
  input=data/gold_labels/legacy/processed/Skin_Reaction/test.jsonl
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

## Historical TRIM / DeepSeek properties-only baselines

On 2026-06-29, 3 Intern-S1/TRIM no-retrieval properties-only DeepSeek-v4-pro baselines were run
for historical reference comparison with the current ChEMBL retrieval pipeline. They do not use TxAgent's current
`run_reasoning_pipeline.py`, nor ChEMBL/Starling retrieval; the prompt comes from
`trim.reasoning.task_user_prompts.render_task_user_message`, tool mode is `properties`,
the only visible tool is `get_mol_properties_and_fg`. The data split uses
`/data1/tianang/Projects/Intern-S1/DataPrepare/TDC_no_conflict_labels_salt_removed/test/Skin_Reaction.jsonl`,
consistent with the current Skin_Reaction test split of 82 samples.

```text
identity allowed:
  log: /data1/tianang/Projects/Intern-S1/logs/deepseek-v4-pro_properties_Skin_Reaction_DILI_test_20260629_194945.log
  trace: /data1/tianang/Projects/Intern-S1/reasoning-trajectory/deepseek-v4-pro_properties_Skin_Reaction_DILI_test_20260629_194945.log/Skin_Reaction.jsonl
  n=82, failed parses=0
  accuracy=0.7683
  macro-F1=0.7640
  class 0 precision/recall/F1=0.7222/0.7429/0.7324
  class 1 precision/recall/F1=0.8043/0.7872/0.7957
  tool usage: 81/82 questions with tools, avg tools/sample=0.99

strict no identity / no memory comparison:
  log: /data1/tianang/Projects/Intern-S1/logs/deepseek-v4-pro_properties_no_smiles_identity_Skin_Reaction_DILI_test_20260629_195403.log
  trace: /data1/tianang/Projects/Intern-S1/reasoning-trajectory/deepseek-v4-pro_properties_no_smiles_identity_Skin_Reaction_DILI_test_20260629_195403.log/Skin_Reaction.jsonl
  n=82, failed parses=0
  accuracy=0.7439
  macro-F1=0.7408
  class 0 precision/recall/F1=0.6842/0.7429/0.7123
  class 1 precision/recall/F1=0.7955/0.7447/0.7692
  tool usage: 82/82 questions with tools, avg tools/sample=1.00

identity forbidden but memory comparison allowed:
  log: /data1/tianang/Projects/Intern-S1/logs/deepseek-v4-pro_properties_no_smiles_identity_allow_memory_compare_Skin_Reaction_DILI_test_20260629_195652.log
  trace: /data1/tianang/Projects/Intern-S1/reasoning-trajectory/deepseek-v4-pro_properties_no_smiles_identity_allow_memory_compare_Skin_Reaction_DILI_test_20260629_195652.log/Skin_Reaction.jsonl
  n=82, failed parses=0
  accuracy=0.6707
  macro-F1=0.6695
  class 0 precision/recall/F1=0.5952/0.7143/0.6494
  class 1 precision/recall/F1=0.7500/0.6383/0.6897
  tool usage: 82/82 questions with tools, avg tools/sample=1.00
```

Strict vs memory-allowed trace audit:

```text
The two no-identity runs differ on only 8/82 predictions.
Memory-allowed is worse on 7 flips and better on 1 flip.
Main degradation: positive recall drops from 35/47 to 30/47.
Failure mode: memory-allowed often falls back to a narrow "classic direct electrophile only" checklist
and misses pro-hapten / pre-hapten / autoxidation / less-canonical sensitizer mechanisms
such as azlactone/reactive lactone, isothiourea-like reactivity, ortho-quinone-methide formation,
terpene autoxidation, squaric-acid-like dicarbonyl chemistry, and pyrazolone/pro-hapten behavior.
The one useful memory-allowed flip was a nitroaromatic pro-hapten case.
```

The current v1 rules actively exclude:

```text
generic PubChem/Tox21 Nrf2 assays without HaCaT/keratinocyte/ARE-luc/sensitisation context
EGFR/FGFR/other kinase biochemical assays triggered only by target names such as epidermal/fibroblast
PAF-induced vascular permeability assays misread as skin permeability
anti-inflammatory / dermatology efficacy in reconstructed human epidermis models
permeation-enhancer assays where the tested molecule promotes another compound's transdermal permeation
```

The original task definition has been confirmed by source chain as binary LLNA skin sensitisation. Irritation, phototoxicity,
local damage, and skin exposure can only serve as mechanistic/context evidence and cannot be equated with the gold label.

## Skin_Reaction source evidence principles (legacy v1 ontology)

The following layering explains what skin-related evidence the source library collects, not the current sensitization gold. Valuable
source evidence in ChEMBL roughly falls into three axes:

```text
hazard axis:
  whether the compound can cause skin sensitization, irritation, corrosion, phototoxicity, or local cell damage.

mechanism axis:
  whether it hits key events in the skin sensitization AOP, such as protein covalent binding, keratinocyte activation, dendritic cell activation.

exposure axis:
  whether the compound can enter, retain, or penetrate the skin, allowing the hazard to manifest.
```

Important constraints:

```text
1. Skin permeability / dermal absorption is not a skin reaction hazard.
   It can only enhance or weaken exposure plausibility and cannot alone support Y=1.

2. General cytotoxicity is not a skin reaction hazard.
   CC50 / GI50 / viability in non-skin cell lines can only serve as weak background and should not alone determine positive.

3. Dermatology efficacy is not a skin reaction hazard.
   Therapeutic assays such as anti-inflammatory, antibacterial, anti-psoriasis, anti-acne, melanoma efficacy, skin whitening, wound healing
   cannot be treated as adverse skin reaction evidence.

4. Target binding / enzyme inhibition is usually not skin reaction evidence.
   Unless the assay description explicitly points to skin sensitisation, irritation, phototoxicity, keratinocyte /
   dendritic-cell activation, haptenation, or dermal toxicity.

5. Consistent positive evidence from validated skin sensitisation AOP assays is more important than a single weak cytotoxicity assay.
   DPRA/ADRA/kDPRA, KeratinoSens/LuSens/EpiSensA, h-CLAT/U-SENS/IL-8 Luc/GARDskin correspond to different
   key events; when multiple key events agree, evidence strength should be upgraded.
```

## Evidence families and endpoint groups

Skin_Reaction should not copy the tier semantics of BBB / Bioavailability. Here `assay_tier` should represent skin-reaction
reasoning value, not a simple ranking of in vivo/in vitro distance.

Suggested initial layering:

```text
Tier 1: direct skin reaction anchors
Tier 2: validated skin sensitisation AOP key-event assays
Tier 3: phototoxicity and local skin irritation / corrosion / dermal toxicity
Tier 4: skin exposure, barrier penetration, and local context modifiers
Tier 5: weak or context-dependent background
```

### Tier 1: direct skin reaction anchors

These are the evidence closest to the Skin_Reaction label. Prioritize retention when hit.

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

These assays describe other skin hazards, but they must not support or oppose the current sensitization label by themselves.

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

## LLM payload rules and known legacy mismatch

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

The current legacy group/final prompt retains the following broad skin-reaction distinctions to reproduce historical results; among them, categories 3/4 must not directly support the binary label in the current `sensitization_aligned_v2`:

```text
1. Direct human/LLNA/validated skin reaction evidence can support or oppose final label.
2. AOP key-event evidence supports sensitisation hazard, but one isolated key event is not equal to clinical skin reaction.
3. Phototoxicity is a skin-reaction subtype; it should be separated from allergic sensitisation.
4. Skin irritation/corrosion is local damage evidence; it should not be conflated with immune sensitisation.
5. Skin permeability/retention modifies exposure only; it cannot by itself prove skin reaction risk.
6. Generic cytotoxicity, dermatology efficacy, target binding, and antimicrobial assays are weak context only.
```

## Reasoning schema expectations

The following is the current `sensitization_aligned_v2` contract; do not use the legacy fields below to interpret new runs.

Single-molecule branch:

```text
label_scope: skin_sensitization_contact_allergy.v2
skin_sensitization_prior: risk | no_risk | mixed_or_unclear
activation_prior: direct_hapten | pre_hapten | pro_hapten | none_apparent | mixed_or_unclear
reactive_or_haptenation_prior
skin_exposure_context
confidence
reasoning_summary
```

Group-level branch:

```text
label_scope: skin_sensitization_contact_allergy.v2
useful_for_skin_sensitization_reasoning: boolean
endpoint_scope: direct_sensitization | sensitization_aop | out_of_scope_other_skin_hazard | exposure_context | weak_context
sensitization_evidence_direction: supports_sensitizer | argues_against_sensitizer | neutral_or_unclear | context_only
transferability
confidence
reasoning_summary
key_evidence[].effect_on_sensitization_reasoning
caveats
```

Final branch:

```text
label_scope: skin_sensitization_contact_allergy.v2
skin_reaction_prediction: risk | no_risk
confidence: low | moderate | high
main_evidence_type:
  direct_sensitization_anchor
  sensitization_aop
  structural_haptenation_prior
  weak_or_no_sensitization_evidence
main_reasons
conflicting_evidence
evidence_gaps
final_summary
```

Below only records the `legacy_skin_reaction_v1` reproduction schema.

Legacy single-molecule branch:

```text
reactive_or_haptenation_prior
electrophile_or_thiol_reactivity_alerts
skin_permeation_prior
phototoxicity_structural_prior
irritation_or_corrosion_structural_prior
physicochemical_exposure_prior
```

Legacy group-level output:

```text
useful_for_skin_reaction_reasoning
transferability
evidence_direction
confidence
reasoning_summary
key_evidence[].effect_on_skin_reaction_reasoning
caveats
```

Legacy final output:

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
