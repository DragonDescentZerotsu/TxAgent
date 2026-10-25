# Evidence-library construction

The [V10 contract](versions/v10/README.md) documents the current construction
lineage. A saved experiment's manifests select its exact release; do not replace
those bindings with today's `CURRENT` or rebuild frozen artifacts during cleanup.

| Owner | Responsibility |
| --- | --- |
| `shared/v1/` | Construction kernel used by V7/V8 |
| `shared/v2/` | UID-aware construction kernel used by V9/V10 |
| `versions/vN/` | Release-wide builders and scientific policy integration |
| `versions/vN/tasks/` | Task-specific source cleaning, policies and reviewed mappings |
| `versions/vN/prompts/` | That release's construction prompts |
| `data/evidence_libraries/<task>/<release>/` | Built data and immutable source/validation manifests, separate from code |

Release contracts: [V7](versions/v7/README.md), [V8](versions/v8/README.md),
[V9](versions/v9/README.md), [V10](versions/v10/README.md). Import these canonical
packages directly; no compatibility construction namespace is provided.

`views.py`, `evidence_library.py`, `heldout_index.py`, `compact_artifacts.py` and
`assay_catalog.py` consume completed libraries across versions.
`stage_artifact_store.py` packages completed stages.
[Gold-label construction](../gold_labels/README.md) owns benchmark publishing;
[semantic buckets](../../../optimization/semantic_buckets/README.md) owns reviewed
semantic grouping and filtering. Exploratory analyses live in
`analysis/evidence_library/`, including `record_informativeness.py`.

Gold-v1 level mappings are published from hash-pinned originals, with reviewed L1
voter corrections, through the existing entry point:

```bash
python -m data.processing.evidence_library.publish_gold_level_mappings
```

This requires its actual source files. The command is not a replacement for
restoring a missing frozen mapping. For repository integration and unresolved
input availability, see the [release status](../../../docs/release_refactor.md).
