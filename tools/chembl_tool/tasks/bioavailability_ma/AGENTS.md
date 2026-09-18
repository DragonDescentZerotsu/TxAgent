# Bioavailability_Ma paper-path notes

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

This directory only retains the concise pipeline that can enter the paper's main method: a universal evidence contract, molecule-level retrieval, single/group/final LLM reasoning, and task ontology. Historical Fa/Fg/Fh full expert policy, deterministic force/block/rescue, fallback calibration, postprocess, and test-error-driven evolution have been removed from `main`.

The complete snapshot of the old implementation is saved at:

```text
branch: archive/bioavailability-full-expert-policy-20260710
commit: 14803c2
```

Do not restore class-changing policy from that archive to the paper path.

## Task contract

```text
input:
  drug: query SMILES
  Y: 0/1 evaluation label

label:
  Y=1 -> high, oral bioavailability F >= 20%
  Y=0 -> low, oral bioavailability F < 20%
```

The new Starling-held-out benchmark is built by `starling_benchmark.py`: only direct oral F with confirmable human context is accepted; percentages and explicit fractions are unified to percent; ranges crossing 20%, relative comparisons, non-human, unclear population, or records with non-empty `qualifying_conditions` do not enter the gold label. Parent-level 0/1 conflicts are computed as 70% agreement based on accepted source records; multiple records from the same PMID are counted separately, and only exact ties or agreement below 70% are rejected. This benchmark label conversion is distinct from the inference-time Starling label policy prohibited below; it must not enter the LLM prompt or change valid predictions.

The only active condition-aware build is located at:

```text
data/gold_labels/Bioavailability_Ma/v1/scaffold/
```

There are 2,489 molecule-condition rows in total; train/valid/test are 1,958/262/269, of which 2,092 are null-condition rows and 397 are reviewed external-condition rows. The old molecule-only and selected-vN names are retained only in the migration receipt, not as a second set of gold. The public contract is in `data/processing/gold_labels/README.md`. Before formal runs, the retrieval index must be rebuilt using the heldout detailed labels from the scaffold valid+test union.

Exact-query evidence is disabled by default. Neighbor retrieval is evidence prefetch, not an LLM function tool.

## Versioned prompt contract

Bio single/group/final prompts use an independent task-local versioned profile:

```text
tools/chembl_tool/tasks/bioavailability_ma/prompt_profiles.py

legacy_bioavailability_v1
  Freezes historical prompts before 2026-08-09; used only for exact reproduction of old experiments.

f20_evidence_calibrated_v2
  Current default. All conclusions are calibrated against the absolute oral F=20% threshold; single does not directly interpret QED/Lipinski/individual physicochemical risks as F<20%; group separates observed evidence direction from query-specific transferability; final does not treat neutral/insufficient/low-transferability as low evidence.
```

New run manifests must save `task_prompt_profile` and `label_scope`. Historical manifests missing these fields are all mapped to `legacy_bioavailability_v1`; single/group/final-only artifact reuse must come from the same prompt profile, and cross-profile mixing is prohibited. Old settings can still be explicitly rerun:

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
  --bioavailability-prompt-profile legacy_bioavailability_v1 \
  <other frozen parameters>
```

`f20_evidence_calibrated_v2` only changes the LLM task contract, without adding deterministic overrides, batch quotas, train-ratio priors, routers, or postprocess. In the 2026-08-09 GPT-OSS-120B scaffold-valid run, full-flat/full-mechanism macro-F1 improved from `0.6117/0.5869` to `0.7004/0.6844`, with paired 95% CIs all above 0; all conditions were 209/209, 0 failed, and retrieval SHA mismatch=0. `none` macro-F1 dropped to `0.4129` and almost all predictions were high, so the conclusion for that profile is "fix evidence adjudication," not an independently usable high prior.

On 2026-08-10, after freezing the settings, the scaffold-test full-flat/full-mechanism and three train-derived baselines were run for the first and only time. The two agent macro-F1 values are `0.6663/0.6720`; Morgan KNN, MiniMol embedding KNN, and MiniMol trained head are `0.6801/0.6484/0.7027`. All five covered the same 209 test items, both agent batches were 209/209 successful with 0 failed, and all agent-minus-baseline paired-bootstrap macro-F1 95% CIs crossed 0. This test must not be used to retroactively select prompts, retrievers, or thresholds; full statistics and artifact paths are in `tools/chembl_tool/paper_experiments/STARLING_BENCHMARK_RESULTS.md`.

## Paper pipeline

```text
source rows
  -> minimal_evidence.v1
  -> molecule-level aggregation and fingerprint index
  -> operational or parent-disjoint identity filtering
  -> source-local groups mapped into direct/mechanism families
  -> parallel single-molecule and mechanism-family reasoning
  -> final LLM synthesis
  -> structured-output validation only
```

Allowed engineering safeguards:

- canonical SMILES and molecule-level aggregation;
- exact-query exclusion;
- source/provenance preservation;
- `molecule_properties` for the single branch;
- `mmp_structure_compare` / `properties_compare` for the group branch;
- JSON schema validation and bounded retry with trace (current default max 4 total attempts);
- trace, batch resume, and metrics.

Prohibited additions:

- `force_high` / `force_low`;
- final prediction override;
- blocker/rescue targeting a specific test molecule or failure pattern;
- Starling-specific label policy;
- valid/test-selected postprocess;
- sending internal `evidence_direction` / `evidence_strength` directly to the LLM.

## Minimal evidence contract

All ChEMBL, Starling, and future source rows uniformly use in the LLM prompt:

```text
tools/chembl_tool/common/evidence_contract.py
contract_version: minimal_evidence.v1
```

The contract includes:

```text
source
molecule
group
endpoint + measurement
evidence/context text
annotations: evidence_role, scope, transferability, uncertainty
quality
provenance
representative examples
```

Among these, `transferability=not_assessed` is the retrieval-time default; query-specific transferability must be judged by the group LLM based on structural comparison and evidence context. The contract does not include threshold votes or label recommendations.

## Data sources

The only configuration entry for paper-facing source/group mapping:

```text
tools/chembl_tool/tasks/bioavailability_ma/experiment_config.py
```

It declares ChEMBL/Starling direct groups and 5 mechanism families; this mapping must not be duplicated in runners or source adapters.

ChEMBL evidence library:

```text
outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/
  bioavailability_molecule_evidence.jsonl
  bioavailability_neighbor_index.pkl
```

Starling task data:

```text
data/raw/starling/bioavailability_ma/
  Oral_AUC-Cmax_Exposure/extractions.parquet        # immutable upstream
  canonical_direct_v2/
    hf_oral_bioavailability_snapshot.parquet
    direct_source_rows.parquet
    direct_claims.parquet
    cross_source_dedup_audit.parquet
    local_partition_audit.parquet
    merge_manifest.json
  oral_exposure_residual_v2/
    exposure_records.parquet
    partition_manifest.json
  Fa/extractions.parquet
  Fg/extractions.parquet
  Fh/extractions.parquet
```

Unified canonical source build entry:

```text
tools/chembl_tool/tasks/bioavailability_ma/build_canonical_starling_source.py
```
The original HF snapshot and local parquet files are not modified in place. Local `bioavailability` rows are transferred to canonical direct only when explicit absolute wording or oral/IV anchors are present; relative and ambiguous rows without absolute anchors remain in residual. Cross HF/local near-equivalent claims with the same parent+PMID are deduplicated one-to-one and retain dual-source provenance. The gold builder and agent direct evidence must read the same `direct_claims.parquet` SHA-256.

Starling factor builder:

```text
tools/chembl_tool/tasks/bioavailability_ma/build_starling_factor_evidence_library.py
```

Starling gold benchmark adapter:

```text
tools/chembl_tool/tasks/bioavailability_ma/starling_benchmark.py
```

The former builds inference-time evidence/index; the latter only implements the binary label adapter for direct human oral F. They cannot substitute for each other.

The Starling factor builder uses shared profile ingestion:

```text
data/processing/evidence_library/evidence_library.py
```

The task wrapper only declares column mapping and group/role:

| Source | Group | Evidence role |
|---|---|---|
| canonical direct v2 claims | `Observed.direct_oral_bioavailability` | `direct_outcome` |
| nondirect HF oral-bioavailability rows | `Observed.nondirect_oral_bioavailability` | `surrogate_proxy` |
| residual Oral_AUC-Cmax/relative/ambiguous rows | `Observed.oral_auc_cmax_exposure` | `surrogate_proxy` |
| Fa parquet | `Fa.absorption_solubility_permeability` | `mechanistic_factor` |
| Fg parquet | `Fg.gut_wall_efflux_intestinal_metabolism` | `mechanistic_factor` |
| Fh parquet | `Fh.hepatic_clearance_metabolic_stability` | `mechanistic_factor` |

Fa/Fg/Fh are task ontology, not deterministic classifiers. The final prediction is still synthesized by the LLM from group outputs.

Conditioned assay-family curves must not discard the nondirect HF group. This source has no native assay-system field; use `build_nondirect_assay_context.py` to build a versioned, bounded, interpretable assay-context overlay based on report type, coarse-grained population, and oral exposure mode; do not build assays by PMID, molecule, or individual record. The repaired family catalog/index maintains an independent lineage from historical missing-nondirect artifacts; full paths and run contracts are in `tools/chembl_tool/paper_experiments/ASSAY_LEVEL_RETRIEVAL.md`.

## Source ingestion rules

- Parquet column differences are resolved via `StarlingSourceProfile` configuration, not by writing a separate parser for each parquet.
- Rows missing SMILES or that RDKit cannot parse do not enter structural retrieval and are counted in source stats.
- SMILES are canonicalized before aggregation; the same canonical molecule uses the same source molecule id across profiles.
- Multiple source rows are aggregated into one molecule/group evidence row; representative examples retain the binding of endpoint, value, unit, context, and support text.
- PMID/DOI may be kept in raw internal rows but must not enter LLM-visible minimal evidence.
- Direct numeric outcomes generate aggregate measurements only when the endpoint is single and units are consistent; proxy/mechanism evidence retains examples and does not mix heterogeneous values into a synthetic value.

## Commands

Build Starling index:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_factor_evidence_library \
  --out-dir outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_full_v2 \
  --workers 32
```

Run Starling source with the generic pipeline:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
  --index outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_full_v2/starling_factor_neighbor_index.pkl \
  --api-key-env GLM_API_KEY \
  --base-url https://litellm.parcc.upenn.edu/v1 \
  --model zai-org/GLM-5.2-FP8 \
  --disable-thinking \
  --reasoning-effort "" \
  --batch-id bioavailability_ma_paper_starling_<date>
```

Formal paper runs use only the above `outputs/paper/` index. The task-level builder default directory under `outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/` is for temporary development only; do not copy or symlink historical indexes into the formal experiment path. Before running, check the meta for `index_version`, `canonical_contract_version`, canonical/residual SHA-256, `scope`, `evidence_content`, and the five stable group IDs.

The 2026-07-23 strict-hop availability census is in `outputs/chembl_tool/tasks/bioavailability_ma/distance_expansion/analysis/hop_availability_census/`. Experimental pKa, LogD/LogP, PPB/Fu outside C can form an H1 candidate union (59,049 parents; parent-disjoint >=1 coverage 98.44%), but there is no qualified H2. pKa/LogD/LogP overlap semantically with the query `molecule_properties` tool; PPB/Fu only supports hepatic clearance, not direct absolute F. They can only be independent distance/relevance design candidates and must not modify the existing paper matrix.

API keys must be provided only via environment variables or an uncommitted local env file; they must not be written into code, manifests, command examples, or git.

## Tests

```bash
/data1/tianang/anaconda3/envs/vllm/bin/python -m pytest \
  tests/chembl_tool/common \
  tests/chembl_tool/tasks/bioavailability_ma -q
```

At least cover:

- legacy ChEMBL row -> `minimal_evidence.v1`;
- profile-driven parquet column mapping;
- missing/invalid SMILES stats;
- molecule-level aggregation and exact-query exclusion;
- direct/proxy role split;
- JSON required-field/value/tool validation retry;
- API key not entering tracked files.

## Evaluation boundary

The historical Bioavailability test set has been repeatedly inspected during old expert-policy iterations and cannot serve as the untouched final test for the paper. Paper results should use a new holdout, a re-frozen split, or external evaluation. Historical metrics from the archive branch cannot be presented as results of the current simplified paper pipeline.
