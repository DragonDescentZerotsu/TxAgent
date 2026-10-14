# BBB_Martins task notes

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

This file only records BBB_Martins task-specific semantics: label, current data version, BBB evidence tier, filtering rules, and reasoning boundaries. General ChEMBL workflow, wrapper structure, batch/resume, viewer, cost, and directory conventions are uniformly recorded in the repository root `AGENTS.md`.

The current paper path merges fine-grained ChEMBL endpoint groups into 4 mechanism families via `experiment_config.py`: direct brain exposure, passive permeability, efflux transport, and influx transport. `full_mechanism` only starts parallel reasoning for these 4 families; `full_flat` uses the same evidence union and synthesizes one branch. The description below, per the `Tier.endpoint_group` branch, only describes the old `run_reasoning_pipeline.py` native runner, not a template for new tasks.
On 2026-07-22, Starling passive permeability, efflux, and influx three mechanism-family acquisitions were imported; they are merged with existing direct BBB evidence into a separate full index, without changing the old direct-only index.

## Task definition

The goal is not to train a BBB classifier, but to build an auditable BBB evidence library: given a query molecule, first prefetch BBB / permeability / transporter evidence for similar molecules, then hand it to a reasoning LLM to judge whether analog evidence can transfer to the query molecule.

Current evaluation convention (Conditioned Benchmark):

```text
Y=1 -> bbb_prediction=pass
Y=0 -> bbb_prediction=fail
final summary must choose exactly one of pass/fail; uncertain prediction is no longer allowed
```

The current gold is experimentally supported meaningful/adequate CNS access vs restricted/poor access after systemic administration, not passive permeability, nor any CNS trace detection: brain tissue, unbound brain, brain/systemic ratio, CSF, PET/autoradiography, and explicit in vivo BBB outcomes are eligible; PAMPA/cell models, computational predictions, mechanism-only proxies, non-systemic administration, altered barrier, indirect efficacy inference, and clear direction conflicts are rejected. Low but non-zero exposure can be negative; CSF remains a proxy family. Parent-level conflicts continue to compute 70% agreement from accepted source records; multiple records from the same PMID each count as votes; exact ties are rejected.

The only active split is at:

```text
data/gold_labels/BBB_Martins/v1/scaffold/
```

train/valid/test are 3,053/397/393 molecule-condition rows, with identity/scaffold overlap all 0. The old molecule-only, gold-vN, and selected-vN paths are only source provenance for the current cohort, not parallel benchmarks; the exact migration relationship is in `data/artifacts/gold_labels/conditioned_benchmark/migration_receipt.json`.

### Normalized-v7 artifact map

The current normalized source through Stage 03 lives at
`outputs/chembl_tool/tasks/bbb_martins/evidence_library/starling_normalized_v7/`.
The complete conditioned-benchmark stages 06–09 and Morgan index live at
`outputs/paper/molecular_evidence_agent_starling_scaffold_conditioned_benchmark/evidence/bbb_starling_v7/`.
That paper view pins Stage-03 manifest SHA-256
`c7b219c7b13074b1d7e916ae427122a71f284541e8adc09ec1493faaa8da0c26`;
it is the model-matched complete artifact. The similarly named tracked
`artifacts/chembl_tool/tasks/bbb_martins/starling_normalized_v7/` archive is an older
normalization lineage and must not be selected merely because it contains later stages.
The detailed contract remains `tools/chembl_tool/paper_experiments/ASSAY_LEVEL_RETRIEVAL.md`.

Conditioned Stage 06 distinguishes `direct_vote` (inserted gold-training labels),
`direct_residual` (preserved `direct_bbb` records), and `indirect` (passive, efflux,
and influx records) with `retrieval_source_id`. Stage 08 places the first two in the
same direct group, so consumers must use record provenance when they need separate
strata. The V19.1 BBB numeric checkpoint and prompt projection cover only the three
indirect sources; direct residual scoring remains a separate model/cache contract.

The historical five-level cache view deliberately refines that boundary. Conditioned
benchmark `direct_vote` rows remain L1. Revised L2 combines historical
`Proxy.central_functional_access` with preserved non-voting rows that historical v5
placed in L1. Historical passive, efflux, and influx row assignments remain L3, L4,
and L5. Build this view from current normalized-v7 Stage 06 eligibility and use the v5
overlay only for row-family assignment; do not substitute the old archived index.

The complete progressive valid uses an audited BBB record-level family overlay. The matched 397-row valid has completed all five layers and five baselines with zero failures; the complete results are maintained only at
`paper_experiments/ASSAY_LEVEL_RETRIEVAL.md`. The five layers are fixed as
direct measured CNS access, central functional/prediction/generic proxy, passive permeability, efflux, influx;
explicit efflux signal takes precedence over uptake/influx; generalized `ratio` or `transporter_mediated` does not constitute family assignment.
The ledger for all `581,708` rows and the 0-violation gate are located in the source overlay's `purity_audit/`. Full build, index, stable-identity retrieval diff/reuse, and run entry points are maintained only in `paper_experiments/ASSAY_LEVEL_RETRIEVAL.md`.
External comparative rows involving 5-FU/5-FC, lacking influx assays or verifiable experimental measurements, are only visible in the near-direct layer and never participate in gold voting; the influx payload must not reuse that card.

The BBB reasoning contract is versioned independently via `prompt_profiles.py`. When old artifacts lack a profile field, it must be interpreted as
`meaningful_cns_access_v1`; cross-profile single/group branch reuse must be rejected. The two valid-only candidates from 2026-08-09 were both retained but not promoted: `meaningful_cns_adjudication_v2` only allows direct negative outcome/measured efflux to support fail, causing matched valid 322/366 to predict pass and macro-F1 dropping to 0.5652; v3 adds a strict `convergent_intrinsic_barriers` fail basis, raising matched macro-F1 to 0.6696, but full-Starling direct is 0.6416, not exceeding v1's 0.6452 (paired CI crosses 0). Therefore the current default remains v1; v2/v3 are only for explicit opt-in historical reproduction; formal tests were not run. Detailed paired results are in `tools/chembl_tool/paper_experiments/STARLING_BENCHMARK_RESULTS.md`.

The E16 no-LLM valid audit has shown that the matched train pool's Morgan top-20 generally contains candidates that better match query ionization/BBB properties: after freezing a 0.02 rank-3 similarity-cost budget, descriptive KNN vote macro-F1 improved from 0.5896 to 0.6335, but the paired CI still slightly crosses 0. The only selector-only matched-v3 valid candidate subsequently changed agent macro-F1 from 0.6696 to 0.6708 (paired CI `[-0.0410,+0.0450]`), with accuracy unchanged; 58 flips were 29/29.
Therefore E16 is terminated; do not treat the availability audit as an agent improvement, continue searching for wider budgets/selectors/prompts, or run formal tests. Entry points are `paper_experiments/matched_train_label_agent/bbb_property_compatibility_audit.py` and
`paper_experiments/matched_train_label_agent/bbb_property_compatible_experiment.py`.

The E16 paired trace diagnosis further split same-membership into 42 exact ordered group inputs and 107 reorders.
Among exact controls, 20 group cores and 23 final states changed; among the candidate's 166 unanimous-positive analog sets, 38 were judged fail by intrinsic barriers, causing 25 false negatives. The previously discussed compact structured analog ledger would increase BBB-specific state and compiler complexity; it has been rejected and archived, and is not the next step. The current BBB default remains
`meaningful_cns_access_v1`; do not continue adding final clauses, selectors, or transport candidates from the same valid trace. Historical diagnosis entry points are
`paper_experiments/matched_train_label_agent/bbb_property_compatible_trace_diagnosis.py`.

Legacy native runner boundaries:

```text
ChEMBL neighbor retrieval is not a DeepSeek-callable tool.
ChEMBL neighbor retrieval is also not a current FastAPI service tool.
It is an evidence prefetch / context assembly step inside run_reasoning_pipeline.py.

Tools callable by DeepSeek group-level analysis are only:
  mmp_structure_compare
  properties_compare

Tools callable by DeepSeek single-molecule analysis are only:
  molecule_properties
```

## Task-specific files

```text
experiment_config.py
  Paper-facing direct, full_flat, and 4-family full_mechanism retrieval views; no label policy.

endpoint_groups.py
  BBB Tier.endpoint_group, evidence_direction, evidence_strength rules.

rules.py
  BBB assay screening keywords, negative keywords, weak terms, transporter target genes.

scoring.py
  Unified entry point for BBB assay retention/exclusion and scoring. Both screen_assays.py and rescore_outputs.py call scored_row().

run_reasoning_pipeline.py
  BBB_Martins retrieval/prompt assembly, reasoning stages, and final-only rerun.

prompt_profiles.py
  Versioned single/group/final schemas, label scope, instructions, and cross-field validation; old profiles are not modified in place.

starling_benchmark.py
  Historical TDC-compatible mixed-permeability adapter; must not be used for new BBB primary results.

experimental_meaningful_cns_access_benchmark.py
  Current experimental meaningful-CNS-access gold adapter; isolates prediction/in-vitro/altered-context, mechanism-only proxies, and identity/provenance-unreliable records.

build_starling_evidence_library.py
  Builds Starling BBB molecule-level evidence and neighbor index from `starling-labs/BBB`.
  Supports two Tier 1 replacement sources: `--mode qualitative` (does not expose quantitative metric/value to the LLM) and `--mode all` (retains qualitative + quantitative fields).

build_starling_full_evidence_library.py
  Loads the direct family from existing direct BBB evidence and connects via the public profile reader to `data/raw/starling/bbb_martins/{passive_permeability,efflux_transport,influx_transport}`, building the four-family index shared by paper `starling_full_flat` / `starling_full_mechanism`.
```

The following files are task-specific configuration wrappers; the common implementation is described in the root `AGENTS.md`'s `tools/chembl_tool/common/task_workflows/` notes:

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

Do not add new business rules in wrappers; BBB assay retention/exclusion logic should only be in `rules.py` and `scoring.py`, and endpoint-group semantics should only be in `endpoint_groups.py`.

## Current data and outputs

Currently recommended:

```text
outputs/chembl_tool/tasks/bbb_martins/assay_screening/v6/
```

Where:

```text
bbb_assay_candidates.csv
bbb_assay_candidates.jsonl
bbb_assay_report.md
bbb_health_check.md
bbb_activity_evidence.csv
```

Current v6 statistics:

```text
candidate assays: 20,369
activity evidence rows: 113,929
Tier 1: 4,306
Tier 2: 4,609
Tier 3: 10,752
Tier 4: 702
```

Current evidence library and neighbor index:

```text
outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_molecule_evidence.jsonl
outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_neighbor_index.pkl
outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_neighbor_index.meta.json
```

Starling BBB Tier 1 replacement index defaults to:

```text
outputs/chembl_tool/tasks/bbb_martins/evidence_library/starling/qualitative/
outputs/chembl_tool/tasks/bbb_martins/evidence_library/starling/all/
```

Starling BBB paper-facing full index is at:

```text
outputs/paper/molecular_evidence_agent/evidence/bbb_starling_full/
```

Build command:

```bash
python -m tools.chembl_tool.tasks.bbb_martins.build_starling_full_evidence_library \
  --workers 32 \
  --progress-every 10000
```

## E12 BBB ChEMBL distance expansion (C-family tree; v3 current)

This experiment is isolated from the old paper matrix and uses only ChEMBL 36 throughout from base to extension. Old D/C indexes and existing results are unchanged; new artifacts are in `outputs/chembl_tool/tasks/bbb_martins/distance_expansion/`.

v2 previously performed a measured-state census on all 20,369 D/C assays.
Besides direct/passive/efflux/influx, the old D/C already actually included tight-junction, efflux/influx-transporter abundance, and functional PXR/CAR readouts; these nodes and their new assays must not re-enter H1/H2. The v1 prototype was replaced by v2 because it ignored this fine-grained base-state overlap and is not used for LLM experiments.

The new frozen structure is not global parallel H1/H2, but: D is the root, each current C mechanism family has exactly one aggregate H1 child, and each H1 has at most one optional H2 child. A child can contain multiple targets/measurement families/assays, but they share that tree node's top-3 unique-neighbor budget and the same similarity threshold; do not take 3 per target. Each assay must save its own measured node and admissible path, and uniquely belong to one parent C family.

BBB v3's extensible `c_family_ids` is only `Mechanism.tier_2/3/4`. `Mechanism.tier_1` organizes direct brain-exposure evidence, semantically belonging to the D root; among it, `Tier 1.context_dependent` is retained to preserve the old `D+C` union, but is not considered an independent mechanism branch that must forcibly find an H1.

The v2 historical prototype used:

```text
H1 mmp9_activity
  direct human MMP-9 activity/inhibition -> tight-junction integrity (already in D/C)
  binding-only, mRNA-only, migration/invasion, and other indirect phenotypes are not included

H2 mmp3_activity (judged invalid)
  direct human MMP-3 activity/inhibition -> MMP-9 activity -> tight-junction integrity (already in D/C)
  binding-only, non-MMP-3 targets, and indirect cellular phenotypes are not included
```

A subsequent shortcut audit found direct support from `MMP-3 activity -> tight-junction integrity`, so the shortest path from MMP-3 to the entire B is 1, and it cannot continue as H2. v3 has been implemented per the new tree contract:

```text
Mechanism.tier_2 passive/barrier
  Distance.h1.passive_permeability
    mmp9_activity -> tight_junction_integrity
    mmp3_activity -> tight_junction_integrity
  H2 unavailable (no frozen trusted child currently)

Mechanism.tier_3 efflux
  Distance.h1.efflux_transport
    nrf2_activation -> efflux_transporter_abundance
  Distance.h2.efflux_transport
    keap1_nrf2_interaction -> nrf2_activation -> efflux_transporter_abundance

Mechanism.tier_4 influx
  Distance.h1.influx_transport
    hif1_activation -> influx_transporter_abundance (limited to GLUT1-related parts)
  Distance.h2.influx_transport
    phd2_activity -> hif1_activation -> influx_transporter_abundance
```

Assay mapping is based on the actual measured state, not just target ID: cellular NRF2 translocation/ARE reporters under KEAP1/NRF2 targets belong to H1; only direct biochemical PPI inhibition belongs to H2. Under PHD2 targets, if the readout itself is a cellular HIF/HRE state, it belongs to H1; only direct hydroxylase activity/inhibition belongs to H2. HIF downstream VEGF/EPO-only, generic viability, KEAP1 thermal-shift/SPR binding-only, and all records lacking usable compound endpoints are excluded.

### v3 same-molecule causal continuity audit (2026-07-21)

A subsequent audit found that the correct node-to-node shortest path does not guarantee predicting the BBB label of the **assay molecule itself**. Full records are in
`DISTANCE_SELF_RELEVANCE_AUDIT.md`, machine-readable records in `distance_self_relevance.py`.

```text
pass_same_molecule:
  mmp9_activity
  mmp3_activity
  Reason: the compound, as a barrier perturbagen, alters the physical junction/barrier it itself must face; scope conditions such as injury/inflammation, target exposure, time scale, and route sensitivity are still retained.

requires_query_role (formal E12 does not pass):
  nrf2_activation
  keap1_nrf2_interaction
    Missing: evidence that the same query molecule is an induced ABCB1/ABCG2/ABCC2 substrate.

  hif1_activation
  phd2_activity
    Missing: evidence that the same query molecule is a GLUT1/SLC2A1 substrate.
```

NRF2 literature uses sulforaphane to induce transporters, but uses verapamil/other probe substrates to measure efflux; HIF-1 literature demonstrates GLUT1-dependent glucose uptake. Neither can automatically equate a regulator perturbagen with a downstream transporter substrate. The previously discussed AhR candidate has the same gap and has been downgraded to `context/exposure modifier` candidate, not entering H1.

Therefore, v3 manifest/index/retrieval replay and coverage remain as structural/retrieval engineering artifacts; old D/C and old paper matrix remain completely unchanged; but v3 must not directly start the official E12 LLM performance run. The next version must first pass:

```python
validate_self_relevance_audit(DISTANCE_CONFIG, SELF_RELEVANCE_AUDIT, require_publishable=True)
```

Prompt reminders must not be used to let the LLM guess substratehood to patch missing evidence. If validated transporter-substrate evidence for the same molecule is jointly retrieved in the future, the corresponding family may be re-audited.

2026-07-22 Completed same-parent role-gated feasibility audit, entry at `python -m tools.chembl_tool.tasks.bbb_martins.audit_role_gated_overlap`. Strictly only positive ABCB1/ABCG2/ABCC2 substrate/transport or GLUT1 substrate-uptake evidence for the same molecular parent is accepted; inhibitor, binding, ATPase, probe accumulation, single-direction Papp, low efflux ratio, and inactive rows are not qualified. Measured: NRF2 has 27 overlap parents, 392-query >=1-neighbor coverage 5.87%, top-3 coverage 0.51%; KEAP1 has 6 overlap parents but query coverage 0; HIF-1/PHD2 both have 0 overlap. Conclusion: role-gated join is insufficient to form stable Efflux/Influx H1/H2; they remain `requires_query_role`, and the official E12 LLM run must not start. Report located at `outputs/chembl_tool/tasks/bbb_martins/distance_expansion/analysis/v3/role_gated_overlap/`.

2026-07-22 bounded replacement search completed, see `PASSIVE_DISTANCE_CANDIDATE_AUDIT.md`. New acceptable families are only MMP-2 H1 and ROCK2 H1; both aggregate into one Passive H1 node and share top-3, without adding an independent reasoning branch. ROCK1 lacks BBB isoform-specific support, MYLK/RhoA coverage is insufficient, and MMP-14 cannot be declared H2 due to the risk of a direct barrier shortcut not passing through MMP-2. Efflux/Influx same-parent and same-document remediation remains insufficient, so the next version candidate structure is:

```text
Passive H1 = MMP-9 + MMP-3 + MMP-2 + ROCK2; Passive H2 unavailable
Efflux H1/H2 unavailable
Influx H1/H2 unavailable
```

Before formally building v4, the public tree contract must first turn `H1 unavailable` into an explicit auditable state; unqualified families must not be stuffed into nodes to satisfy the old "each C has exactly one H1" structural constraint. v3 and old D/C/paper results remain frozen.

ChEMBL 36 full-library scan v2 historical artifact:

```text
scanned assays: 1,890,749
included extension assays: 696
base/extension assay-ID intersection: 0
H1 MMP-9: 443 assays, 2,268 indexed unique molecules
old H2 MMP-3: 253 assays, 1,206 indexed unique molecules (needs re-stratification)
extension evidence rows: 4,280
extension unique molecules: 2,747
superset index: 55,260 molecules, 21 source groups
```

ChEMBL 36 v3 current tree artifact:

```text
scanned assays: 1,890,749
included extension assays: 1,514
base/extension assay-ID intersection: 0
passive H1: 696 assays (MMP-9 443 + MMP-3 253)
efflux H1: 389 assays (functional NRF2)
efflux H2: 121 assays (KEAP1-NRF2 PPI)
influx H1: 146 assays (functional HIF-1)
influx H2: 162 assays (direct PHD2 hydroxylase)
extension activity rows: 34,838
extension indexed molecules: 21,127
superset index: 73,175 molecules, 24 source groups
```

`base_measured_state_census.tsv` remains a mandatory input for the next version. The manifest builder must fail directly when an extension measured node already appears in the D/C census; the intersection of the inclusion set with old D/C assay IDs must also be 0. To fill coverage, tight-junction, PXR/CAR, or transporter-abundance assays must not be re-labeled as extension.

The following commands build isolated v3 tree artifacts; they do not overwrite old D/C index or old paper conditions:

```bash
python -m tools.chembl_tool.tasks.bbb_martins.build_distance_assay_manifest \
  --chembl-sqlite tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db \
  --export-activities \
  --progress-every 100000

python -m tools.chembl_tool.tasks.bbb_martins.build_distance_extension_library \
  --workers 128 \
  --progress-every 50000

python -m tools.chembl_tool.paper_experiments.build_distance_index \
  --base-index outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_neighbor_index.pkl \
  --extension-index outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/extension/bbb_distance_extension_index.pkl \
  --output-index outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/bbb_distance_superset_index.pkl \
  --output-meta outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/bbb_distance_superset_index.meta.json \
  --index-version bbb_distance_expansion.v3 \
  --source-release "ChEMBL 36" \
  --workers 128

python -m tools.chembl_tool.paper_experiments.audit_distance_expansion

python -m tools.chembl_tool.paper_experiments.materialize_distance_retrieval
```

v3 audit must maintain: D/D+C LLM-visible base parity 0 failures, tree-node source nestedness 0, D+C->H1->H2 evidence retention 0, H1-after-H2 stability 0, and add: each C has exactly one H1 spec, each H1 has at most one H2 spec, each measurement family has a unique parent, H2 global shortcut 0, each tree node top-3 with unified similarity threshold.

2026-07-20 v2 measured 392-query retrieval audit: all four old strict failure categories are 0. Cumulative coverage: D 0.673, D+C 0.878, D+C+H1 0.888, D+C+H1+H2 0.888; H1 MMP-9 family coverage 0.199, H2 MMP-3 family coverage 0.120. Since the ontology is invalid, these numbers are kept only as historical engineering records, not new tree coverage or LLM performance; E12 macro-F1 has not been run.

2026-07-20 v3 measured 392-query retrieval audit: all four strict failure categories are also 0. Cumulative coverage: D 0.673, D+C 0.878, D+C+H1 0.923, D+C+H1+H2 0.923; mean unique neighbors 1.39, 5.51, 8.28, 8.45. Tree-node coverage: passive H1 0.204, efflux H1 0.666, influx H1 0.528, efflux H2 0.043, influx H2 0.071. H2 coverage is very low and does not increase the total number of covered queries; the official report must state this clearly; macro-F1 has not been run.

Frozen retrieval replay located at:

```text
outputs/chembl_tool/tasks/bbb_martins/distance_expansion/retrieval_replay/v3/
  distance_d/
  distance_dc/
  distance_dc_h1/
  distance_dc_h1_h2/
  distance_mechanism_dc/
  distance_mechanism_dc_h1/
  distance_mechanism_dc_h1_h2/
```

Each condition contains 392 `runs/<condition>_idxNNNNN/retrieval.json`. The reasoning batch must use these frozen inputs via `--retrieval-replay-source-batch`; the task pipeline must not re-retrieve according to old experiment configs.

Historical TDC/E12 frozen query set (not the current Starling gold split):

```text
data/gold_labels/legacy/processed/BBB_Martins/B3DB_cleaned/test/test_efflux.jsonl

fields:
  drug: query SMILES
  Y: BBB label
```

Reasoning artifacts are uniformly placed in:

```text
outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs/
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/
```

## Legacy native BBB reasoning pipeline

```text
1. Read query molecules from test_efflux.jsonl.
2. Call retrieve_neighbors.py to prefetch ChEMBL neighbor evidence for each source-local Tier.endpoint_group.
3. Run single-molecule analysis concurrently; in this historical runner, DeepSeek can only call molecule_properties.
4. Run endpoint-group analysis concurrently; in this historical runner, DeepSeek can only call mmp_structure_compare and properties_compare.
5. Final summary reads single + all group outputs, exposing no tools.
6. Save retrieval/single/group/final/trace/manifest.
```

When `--tier1-replacement-index` is passed, the pipeline excludes all `Tier 1.*` groups from the original ChEMBL index and replaces them with `Tier 1.starling_direct_bbb_evidence` from the replacement index; Tier 2/3/4 still come from the original ChEMBL index. Typical commands for Starling BBB experiments:

```bash
python -m tools.chembl_tool.tasks.bbb_martins.build_starling_evidence_library \
  --mode qualitative \
  --workers 128 \
  --progress-every 10000

python -m tools.chembl_tool.tasks.bbb_martins.build_starling_evidence_library \
  --mode all \
  --workers 128 \
  --progress-every 10000

python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch \
  --tier1-replacement-index outputs/chembl_tool/tasks/bbb_martins/evidence_library/starling/qualitative/starling_bbb_neighbor_index.pkl \
  --batch-id <batch_id>
```

Final-only rerun:

```bash
python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline \
  --resume-final-from-run-dir outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs/<run_id>
```

## Exact ChEMBL context

`chembl_exact_context.py` is an optional evidence-rich enhancement. It uses the query full InChIKey to look up the ChEMBL exact molecule and query activity in assays involving retrieved neighbors, generating:

```text
same_endpoint_activity:
  same assay_chembl_id + same normalized standard_type

same_assay_different_endpoint_activity:
  same assay_chembl_id + different standard_type
```

This uses known ChEMBL experimental records of the query molecule and may cause data leakage in prospective benchmarks. Therefore, it is disabled by default; only when `--enable-chembl-exact-context` is explicitly passed is it used for retrospective / evidence-rich case studies. Do not enable it in default batch evaluation.

The default single-molecule prompt does not include any ChEMBL-related payload or instruction. Only when exact context is enabled and the query exact context is hit does the single-molecule payload include `exact_query_chembl_context`, prompting the model to distinguish direct same-molecule ChEMBL BBB evidence from physicochemical priors. ChEMBL neighbor evidence still only enters group-level context.

## Evidence type explanation

### Tier 1

Direct BBB / brain exposure evidence, e.g.:

```text
brain/plasma ratio
brain to plasma ratio
logBB
Kp,uu,brain
brain concentration
brain level
brain perfusion
CSF/plasma
```

This is the closest evidence to "whether the molecule enters the brain."

### Tier 2

Passive permeability or in vitro barrier model evidence, e.g.:

```text
PAMPA-BBB
Caco-2 Papp
MDCK Papp
brain endothelial cell model
hCMEC/D3
bEnd.3
BMEC / BBMEC / RBEC
```

These are proxy evidence and are not equivalent to in vivo brain exposure.

### Tier 3

Efflux transporter evidence, e.g.:

```text
ABCB1 / P-gp / MDR1
ABCG2 / BCRP
ABCC / MRP
efflux ratio
bidirectional Papp
rhodamine 123 efflux
calcein AM
Hoechst 33342 accumulation
```

P-gp/BCRP substrate, efflux ratio, and bidirectional Papp are closer to BBB reasoning than mere inhibitor IC50.

### Tier 4

Uptake transporter evidence, e.g.:

```text
LAT1 / SLC7A5
GLUT1 / SLC2A1
OATP1A2 / SLCO1A2
OAT3 / SLC22A8
MCT1 / SLC16A1
TFRC
```

Tier 4 must be functional uptake/transport/substrate-related assays. Mere gene expression, Western blot, phosphorylation, binding affinity, etc., are not retained.

---

## Important filtering rules

### `brain plasma membrane` is not brain/plasma

`brain/plasma` represents the ratio of brain tissue exposure to plasma exposure and is BBB evidence.

But:

```text
brain plasma membrane
brain plasma membranes
```

refers to cell membranes/membrane preparations from brain tissue, often appearing in receptor binding assays, and is not brain/plasma ratio. Current logic filters such false hits.

### Do not retain non-functional influx assays

For example:

```text
GLUT1 expression
SLC2A1 RNA stability
Western blot
phosphorylation
Kinobead pull down
binding affinity
```

Without functional readouts such as uptake/transport/substrate, these are not retained as BBB influx evidence.

### Resistant-cell-line phenotype noise for non-transporter targets

For example, when the target is MAP3K5, proteasome, etc., but the description contains:

```text
ABCB1-substrate-selected resistant cell line
ABCG2-substrate-selected resistant cell line
overexpressing ABCB1
overexpressing ABCG2
```

These are usually phenotypic/cytotoxicity settings and are not considered high-confidence BBB transporter assays. Current logic filters obvious noise.

But functional transporter readouts are retained, e.g.:

```text
P-gp-mediated rhodamine 123 efflux
calcein AM assay
Hoechst 33342 accumulation
mitoxantrone accumulation
doxorubicin accumulation
transepithelial transport
```

---

## Cases where canonical SMILES is empty

In `bbb_activity_evidence.csv`, a small number of rows have empty `canonical_smiles`, which is due to missing original structure data in ChEMBL, not a join error.

In v6:

```text
activity evidence rows: 113,929
missing canonical_smiles: 279
affected molecule_chembl_id: 147
```

Common causes:

```text
structure_type = NONE
structure_type = SEQ
metals/inorganics
radioactive technetium complexes
protein/peptide/sequence-type molecules
ChEMBL has no standard structure
```

When building similarity retrieval indexes later, molecules with empty `canonical_smiles` should be filtered. Original evidence rows can be retained for audit and reporting.

---

## Long task monitoring

One-off monitoring script:

```text
tools/chembl_tool/utils/BBB_Martins/monitor_chembl_assay.sh
```

Purpose: checks at fixed intervals whether a tmux session has completed, and automatically generates a health check report upon completion.

Example:

```bash
tmux new-session -d -s chembl_assay_monitor \
  'tools/chembl_tool/utils/BBB_Martins/monitor_chembl_assay.sh chembl_assay outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw 5400 /tmp/chembl_assay_monitor.log'
```

Parameters:

```text
chembl_assay: monitored tmux session
outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw: output directory
5400: check interval in seconds, i.e., 90 minutes
/tmp/chembl_assay_monitor.log: monitor log
```

This script is not core business logic and can be kept as a long-task operations tool.

---

## Testing

Run:

```bash
pytest -q tests/chembl_tool
```

Current coverage:

```text
text normalization
P-gp / Caco-2 / blood-brain barrier / Kp,uu brain matching
generic permeability / generic uptake filtering
ABCB1 target + substrate retention
brain plasma membrane false-hit filtering
Tier 4 non-functional GLUT1/SLC2A1 assay filtering
non-transporter target resistant-cell-line noise filtering
functional rhodamine efflux retention
```

---

## Current retrieval / reasoning status

Phase 2 has built a molecule-level evidence library based on `bbb_activity_evidence.csv`:

```text
1. Filter out molecules with empty canonical_smiles
2. Normalize molecule identity, excluding query exact same molecule, including records with identical full InChIKey, InChIKey connectivity layer, and canonical SMILES
3. Use Morgan fingerprint Tanimoto to retrieve similar analogs
4. Aggregate evidence by Tier, endpoint type, similarity bucket
5. Output analog evidence summary, not a hard BBB pass/fail decision
```

Current implementation entry points:

```text
build_evidence_library.py
retrieve_neighbors.py
run_reasoning_pipeline.py
```

ChEMBL neighbor retrieval is currently an internal evidence prefetch in the pipeline, not an LLM tool.
When adding other ChEMBL tasks later, continue to reuse the general tools of `tools/chembl_tool/common/`.

---

## Normalized Starling v6 BBB library

BBB now has a policy-decoupled normalized-v6 path alongside the historical
Starling and ChEMBL libraries. The historical files, runtime source names, and
defaults remain unchanged.

Architecture:

```text
data/processing/evidence_library/
  shared/v1/clustered_auxiliary_mapping.py  shared reconciliation mechanics
  stage_artifact_store.py                   deterministic stage packaging
  versions/v7/build_normalized_evidence_library.py
  versions/v7/pair_bucket_build.py

data/processing/evidence_library/versions/v7/tasks/bbb_martins/
  starling_*.py                           BBB source and normalization policies
  build_normalized_starling_evidence_library.py
```

The complete layout is:

```text
01_cleaned/
02_canonicalized/
03_records/                 complete unfiltered normalized source records
04_pair_buckets/            complete source-aware bucket membership
05_distance_calibration/    complete-data, label-free SD and empirical-CDF geometry
06_remove_heldout_overlap/  independent random/scaffold record views
07_molecule_evidence/       split-specific relational evidence
08_neighbor_index/          split-specific compact indices
09_audits/
```

The Stage-02/03 canonical measurement tuple is exclusively an assay-transfer numerical contract; retrieval
and LLM evidence continue to use the cleaned source measurement, unit, and support projection. The shared
parser, BBB endpoint/unit rules, source reconciliation, v7 projection, and frozen
`data_processing/assay_transfer_measurements_v2/policy.json` compose one final measurement/unit/scalar
tuple. Stage 04 retains every record and marks reviewed unit defects `assay_transfer_eligible=false`; it
does not delete them or change `retrieval_eligible`.

Stages 04 and 05 use the complete unfiltered Stage-03 records. This means held-out
`direct_bbb` measurements contribute only to aggregate bucket validation, first-class sample SD,
and continuous value-CDF geometry; the shared contract also supports ordinal category-rank CDFs, although
BBB currently declares no ordinal scale. Stage 05 does not use benchmark labels or expose
individual records to prediction. This is the sole permitted held-out-data exception,
and it follows the shared all-task contract in `tools/chembl_tool/tasks/AGENTS.md`.

Stage 06 is the mandatory leakage boundary. It removes `direct_bbb` rows for benchmark
held-out parents before molecule evidence and neighbor indices are built. Passive,
efflux, and influx mechanism records remain available even when their molecule identity
appears in the benchmark. Pair-bucket membership remains a complete source audit, and
calibration must not declare a held-out source or held-out key loader. The v3 artifact stores no raw-
distance CDF; its historical stage and filename remain unchanged for archived-v1 compatibility.

The four LLM-visible group IDs remain exactly:

```text
Tier 1.starling_direct_bbb_evidence
Mechanism.passive_permeability
Mechanism.efflux_transport
Mechanism.influx_transport
```

The direct snapshot is the complete `starling-labs/BBB` train split at revision
`f50c638621fcc2dedfc9e00f7074f867640efc1b`, with a stable `source_index` and no
row filtering or text rewriting. Assay context and species are reconciled by
embedding distinct raw values with `sentence-transformers/all-MiniLM-L6-v2`,
clustering with the frozen seed `20260801`, and mapping each cluster with
`gpt-5.4-mini` at medium reasoning effort. The private API key is loaded at
runtime and is never stored in mappings, manifests, caches, or traces.

Endpoint and unit normalization is a separate, human-gated workflow:

```text
data_processing/build_direct_endpoint_mapping.py
  Sends only distinct direct_bbb.quant_metric values to gpt-5.4-mini.
  Embedding-local requests contain at most 500 values and write an unpublished
  raw-to-provisional mapping plus response/cluster provenance.

data_processing/reconcile_direct_endpoints.py
  prepare          assigns every provisional label and all of its raw values to
                   one primary subagent review packet
  prepare-catalog  creates one full candidate-label catalog plus disjoint owner
                   packets, so aliases from different GPT clusters can be merged
  propose          requires independent checker and adjudicator records for every
                   changed raw-level or catalog-level decision

starling_endpoint_normalization.py
  Loads only the explicitly human-approved Direct map. Passive, efflux, and
  influx endpoints and their low-cardinality context fields use deterministic,
  source-specific reviewed rules.
```

There is no automatic publication command. The proposal remains
`human_approved=false` until a person copies an explicitly approved, signed-off
mapping into `data_processing/direct_endpoint_normalization_v1/approved/`.
Normalized stages 02-09 must not be rebuilt before that checkpoint. Physical
permeability rows without an explicit source unit or unit embedded in the raw
endpoint/measurement remain unit-unresolved and are excluded from pair buckets;
they remain available as sidecar evidence. Binary categorical values are encoded
as dimensionless scalar anchors without changing the normalized source endpoint.
Source-aware pair keys separate them using canonical unit, measurement scale,
and the relevant canonical context fields.

Measurement values and units now use the shared exact Stage-02 contract. The frozen BBB extraction is
endpoint-aware and every successful multi-quantity result is exploded. Positive source decimals with a unit
bypass the model unchanged. The single shared JSON maps or excludes exact
`(bbb_martins, canonical_endpoint, input_unit)` keys, including reviewed PAMPA header scales, logBB aliases,
and percent-to-ratio conversions. No runtime unit regex, support-text factor correction, or per-record numeric
override runs on this path. Original source fields and the parent cleaned ID remain attached to every child;
variation is null. The historical normalization modules remain available only for replay.

BBB additionally freezes `canonical_reference_scope` and `canonical_reference_basis` per scalar row.
Stage 04 accepts absolute values, endpoint-defined ratios with an explicit endpoint denominator,
standardized assay-control ratios, and declared categorical scales. Comparator-relative and unknown rows
remain evidence-only. Both scope and basis are pair-key dimensions, preventing plasma/brain, A-to-B/B-to-A,
or assay-control denominators from being pooled accidentally. BBB v2 first resolves only three frozen,
definition-level cases without an API call: approved physical scalars, explicit tissue/fluid denominator
ratios, and ineligible fold changes. It sends every remaining row at most once using 50 rows per
`gpt-5.4-mini` request; each visible row contains only a batch-local ID, raw measurement value, and support
text, and the response contains only scope and basis. Existing v1 submissions are never retried. A prior
GPT assignment that contradicts a v2 safe gate is preserved as provenance but published as `unknown`.
The shared durable ledger enforces nine million combined input/output tokens per key epoch.

The frozen 2026-08-06 v2 generation covers all 57,564 candidate rows. Definition-level gates resolved
29,097 previously unattempted rows without an API call, while 14,432 remaining rows were submitted once
in 289 requests. The new requests used 1,200,609 input and 460,163 output tokens; 14,172 rows passed strict
response validation and 260 (1.80%) failed closed as `unknown`, with no retry. Across the complete mapping,
399 prior assignments contradicted a safe gate and were also retained only as provenance while publishing
`unknown`. The key epoch finished at 8,617,163 of 9,000,000 accounted tokens. The frozen mapping is
`data_processing/reference_semantics_v2/reference_semantics.parquet` with SHA-256
`431a8a962c8595f2098474a82fb17fe8ee40964d42ec161d5825be80a3f95bef`. Generation does not rebuild
Stages 02-09; those stages must consume this mapping in one later rebuild.

Build and retrieve:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bbb_martins.build_starling_direct_source

/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m data.processing.evidence_library.versions.v7.tasks.bbb_martins.data_processing.build_embedding_bucket_mapping

/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m data.processing.evidence_library.versions.v7.tasks.bbb_martins.build_normalized_starling_evidence_library \
  --workers 128

/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bbb_martins.retrieve_normalized_starling_neighbors \
  --benchmark-split random --query-smiles '<SMILES>'
```

Reasoning uses the explicit `--retrieval-source starling_v6` plus a matching
split index passed through `--index`. This opt-in source does not change the
legacy `chembl` default or the historical `starling` configuration.
