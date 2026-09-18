# Skin_Reaction auxiliary reconciliation v2

This directory is the human-gated cross-cluster reconciliation of the completed
Skin auxiliary extraction pass. It preserves the exact GPT cluster outputs and
the deterministic aggregated local mapping before proposing a smaller,
semantically reviewed vocabulary for the pair-bucket policy.

The artifacts have three layers:

```text
provenance/
  provisional_cluster_mapping.json  exact cluster and local-label provenance
  cluster_assignments.parquet       local and post-helper label per raw value
  pre_reconciliation_mapping.json   runtime-shaped aggregated local baseline
  provisional_manifest.json         frozen input/output hashes
label_catalog.parquet               review labels aggregated by namespace
review_packets/                     disjoint, cluster-atomic reviewer inputs
reviews/                            primary, checker, and adjudicator records
proposal/                           unpublished runtime-shaped candidate mapping
```

Eight open-vocabulary `source_id/output_field` namespaces are reviewed
independently. `direct_skin_reaction/global_severity_grade` and
`skin_exposure/global_context` are frozen passthrough sections.

Independent role rotation is fixed by namespace group:

| Namespace group | Primary | Checker | Adjudicator |
|---|---|---|---|
| direct reaction + skin exposure | `skin_direct_audit` | `skin_phototoxicity_audit` | `skin_sensitization_audit` |
| sensitization AOP | `skin_sensitization_audit` | `skin_direct_audit` | `skin_phototoxicity_audit` |
| phototoxicity/local damage | `skin_phototoxicity_audit` | `skin_sensitization_audit` | `skin_direct_audit` |

Commands:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m data.processing.evidence_library.versions.v10.tasks.skin_reaction.data_processing.reconcile_auxiliary_mapping \
  snapshot --overwrite

/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m data.processing.evidence_library.versions.v10.tasks.skin_reaction.data_processing.reconcile_auxiliary_mapping \
  prepare-review

# `propose` requires explicit primary, checker, adjudicator, and reviewer-summary files.

/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m data.processing.evidence_library.versions.v10.tasks.skin_reaction.data_processing.reconcile_auxiliary_mapping \
  audit
```

The CLI intentionally has no publish command. The proposed mapping is intended
for the Skin pair-bucket policy, but human approval is required before copying
it to `globally_reconciled_auxiliary_value_mapping.json` or rebuilding any
normalized, pair-bucket, transfer-policy, heldout, evidence, or index stage.

## Current audited proposal

The 2026-08-03 review covers all 17,368 non-null provisional labels in the
eight open namespaces, across 58 packets, 1,544 local clusters, and 218,198 raw
assignments. Three primary reviewers proposed 856 changes. Independent
checkers disagreed with 16, and the third-reviewer adjudication accepted 839
changes and rejected 17. The accepted changes affect 8,316 raw assignments.

`proposal/FINAL_INTEGRITY_AUDIT.json` passes with no issues. It replays the
proposal exactly from `provenance/pre_reconciliation_mapping.json`, validates
all ten runtime-shaped sections and null masks, preserves the two frozen
sections, verifies every review/provenance hash, and proves the existing
downstream artifact tree is unchanged. The unpublished proposal mapping has
SHA-256 `5cd011cbd536dd5e720e9e7e42c18619788b54e33ec384f56933ee24bb50c6b5`.

After explicit approval on 2026-08-03, the proposal was published to the
runtime mapping. The runtime attacher exposed six typography-equivalent tuple
conflicts in phototoxicity context; seven raw keys were normalized to existing
reviewed labels, with no new semantic concept introduced. The exact resolutions
and proposal/runtime hashes are frozen in `PUBLICATION_RECORD.json`. The final
runtime mapping SHA-256 is
`34db0efcd64769699e4b86f7cdcb16e9dc12e466e5f04eb2352eca43440ebd45`.
Stages 02-09 were rebuilt and atomically published; Stage 04 contains 27,081
pair buckets, of which 1,099 pass the complete assay-transfer eligibility
policy.
