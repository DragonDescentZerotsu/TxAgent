# DILI and Carcinogens: final reviewed retrieval sources

Both tasks use Starling-only gold_v4. The September 2026 cleaning cycle is closed;
new source reviews and inference are not part of routine restoration. The retained
retrieval release is `data/starling_data/<task>/retrieval_final/`.

## Current data and reconstruction

| Layer | Location |
|---|---|
| Original acquisition and extraction prompts | `data/starling_data/<task>/raw_v1/` |
| Frozen gold and publication-vote provenance | `data/starling_data/<task>/gold_v4/` |
| Active benchmark | `data/conditioned_benchmark/<Task>/{scaffold,random}/` |
| Final source and source-change ledger | `data/starling_data/<task>/retrieval_final/` |
| Portable source bundles | `artifacts/chembl_tool/tasks/<task>/retrieval_final/03_records/` |
| Source archive inventory | `artifacts/chembl_tool/starling/current_records/manifest.json` |
| Restored Stage-03 files | `outputs/chembl_tool/starling/current_records/<task>/03_records/` |
| Runtime catalog and indices | `outputs/paper/starling_conditioned_assay_family_curve_v1/{family_catalogs,indices}/<task>/` |
| Ordinary Parquet level/card tables | `artifacts/chembl_tool/starling/current_level_records/<task>/` |

The final DILI snapshot incorporates the completed v5 diagnostic source repairs.
Carcinogens materializes all 1,735 modified R18 card decisions into the corresponding
source rows. Corrections are now part of the source; new runs do not need the old
indexed review overlay. Both sources preserve original `raw_record_json`, permanent
UIDs and frozen voter flags, and store corrected retrieval fields separately.
Known irrelevant/unresolved bindings remain auditable but are retrieval-ineligible.
This freezes the completed bounded review; it does not certify the entire library
or resolve candidates explicitly left pending.

```bash
python -m tools.chembl_tool.paper_experiments.rebuild_current_starling_retrieval restore-records --tasks dili carcinogens
python -m tools.chembl_tool.paper_experiments.rebuild_current_starling_retrieval build --tasks dili carcinogens --workers 16
python -m tools.chembl_tool.paper_experiments.rebuild_current_starling_retrieval verify --tasks dili carcinogens
python -m tools.chembl_tool.paper_experiments.export_current_starling_level_records --tasks dili carcinogens --output-dir outputs/exports/dili_carcinogens
```

Run `restore-records` after cloning. The source bundle also restores large gold/audit
JSONL files at their original paths, with exact byte hashes; those files exceed the
ordinary Git per-file limit and are not stored twice. A changed local audit is never
overwritten automatically. `source_validation.json` and `publication.json` in each
final release record the source and index verification.

Reconstruction uses the frozen identity cache; it does not repeat PubChem or model
calls. Task-specific gold builders remain `tasks/<task>/build_reviewed_starling.py`.
Their frozen identity/direction evidence and compact older gold ledgers are provenance,
not alternative active datasets. The generic benchmark publisher refuses to replace
these cohorts; deliberate gold promotion uses its existing `--reviewed-release` entry.
The retired initial gold_v2, gold_v3 tautomer-migration, progressive_v1 overlay and
TDC-augmentation pipelines are available only in Git history. Their frozen ledgers
remain inputs where the current gold builder needs provenance; routine restoration
never runs those pipelines.

Current source maintenance uses `stage_new_task_retrieval.py` with explicit
`--source` and `--output`, and either `--refresh-votes` or `--record-review`.
Content/identity repairs remain available through its `apply_content_repairs()` API;
indexed-card repair and targeted replay remain in the shared review and runner code.
For a source/index semantic audit, `validate_new_task_retrieval_identity.py` requires
`--source-root` and `--index-dir`; it no longer guesses a retired staging layout.
Current gold labels, family policies, prompts and prediction-reuse contracts are
independent of these removed historical build entrypoints.

## Frozen benchmark and retrieval policy

| Task | Train / Valid / Test, each split scheme | Gold positives / negatives |
|---|---:|---:|
| DILI | 3,220 / 402 / 402 | 3,158 / 866 |
| Carcinogens | 3,754 / 469 / 469 | 3,578 / 1,114 |

Carcinogens has five organism conditions: rodent, human, dog, monkey and rabbit.
Sex, strain, route, dose and duration remain source qualifiers but are pooled in gold.
Missing organism does not create an organism-specific vote. DILI uses its approved
coarse conditions. Publication-parent-condition votes abstain on internal conflicts;
accepted labels require a strict majority and at least 60% agreement. These are
source-publication consensus labels, not exhaustive independent-experiment verification.

L1 contains actual voters; L2 contains nonvoter direct-related outcomes; L3–L7 are
configured by each task's `starling_levels.py` and `experiment_config.py`. Both indices
prefilter heldout L1 only. L2 remains eligible; scaffold/parent disjoint applies at
query time to every level. TDC labels are absent from both benchmarks.
The explicit `new_task_tautomer_identity.v2` leakage identity is separate from gold
stereochemical identity. Source repair never grants a new L1 vote.

## Results and retention

Current result locations and historical comparisons are recorded in
`paper_experiments/current_conditioned_results.json`. DILI v5 and Carcinogens R18
remain the latest completed diagnostics on already inspected Valid/Test cohorts.
Final source consolidation does not create new model results: original predictions,
prepared inputs, figures and reuse receipts remain pinned to their actual run inputs.
Any new source-to-result equivalence claim requires a selected-surface audit.

Temporary source/index copies and per-round executable scripts are removed after
final publication validation. Raw acquisition, frozen gold, compact review ledgers,
source-change histories and the dependencies of retained traces are preserved.
Inference and plotting continue through the shared family/progressive runners and
`plot_assay_retrieval_curve.py`; there are no task-local copies of these entrypoints.
