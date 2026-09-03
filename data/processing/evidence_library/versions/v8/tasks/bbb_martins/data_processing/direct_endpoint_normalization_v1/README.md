# BBB Direct endpoint normalization v1

This directory contains the reviewed proposal and approved runtime mapping for
normalizing `direct_bbb.quant_metric`.

Current local-generation inventory:

- 22,798 distinct non-null raw endpoint spellings
- 75 embedding-local `gpt-5.4-mini` requests at medium reasoning effort
- at most 500 values per request (observed maximum: 499)
- 8,012 provisional endpoint labels before subagent reconciliation

Reviewed proposal:

- 8,012 raw-level primary decisions; 150 proposed changes, 145 accepted after
  independent checking and adjudication
- 8,259 final catalog decisions; 351 proposed merges, 332 accepted after the
  same three-role review
- 22,743 cleaned runtime keys mapping to 7,927 final endpoint labels
- 53 cleaned raw-alias groups (55 extra spellings), all converged without conflict
- 38 conservative near-duplicate audit candidates retained for the human checkpoint
- proposal SHA-256:
  `dccace52917d2c01740988f4ed5bb0b50d75b11d90ffa1d00c306f43510e7b77`

Joseph approved this proposal on `2026-08-04T14:17:53Z`. The approved runtime
copy is `approved/endpoint_mapping.json`; the proposal remains unchanged so the
human-approval boundary is explicit in the provenance.

The local GPT step and the global reconciliation are deliberately separate.
The local step preserves every cluster response and its request provenance.
The global steps use subagent review, not a second GPT API pass.

## Workflow

```bash
# OPENAI_API_KEY must already be injected into the child environment.
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m data.processing.evidence_library.versions.v8.tasks.bbb_martins.data_processing.build_direct_endpoint_mapping

/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m data.processing.evidence_library.versions.v8.tasks.bbb_martins.data_processing.reconcile_direct_endpoints prepare

# Run after raw-level primary, checker, and adjudicator JSONL files are complete.
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m data.processing.evidence_library.versions.v8.tasks.bbb_martins.data_processing.reconcile_direct_endpoints prepare-catalog

# Run after the global catalog primary, checker, and adjudicator files are complete.
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m data.processing.evidence_library.versions.v8.tasks.bbb_martins.data_processing.reconcile_direct_endpoints propose
```

Raw-level primary rows use `retain`, `rename`, or `split`. A split must map every
raw value assigned to that provisional label exactly once. Catalog primary rows
use `retain` or `merge`; every merge target must be another candidate label whose
catalog decision is `retain`. Every non-retain decision requires a distinct
primary reviewer, checker, and adjudicator.

`propose` verifies the frozen source/inventory hashes and binds the proposal to
the local mapping, generation audit, cluster assignments, provisional manifest,
raw-level review packet manifest, global catalog manifest, and all six review
files. It writes `approval.human_approved=false` and has no publication mode.

After a human reviews the proposal, approval is a separate explicit filesystem
operation: create `approved/endpoint_mapping.json`, record `approved_by` and
`approved_at`, and retain the proposal provenance. Until that file exists, the
default normalized-v6 build fails closed. Do not rebuild production stages
02-09 from an unpublished proposal.

Passive-permeability, efflux, and influx endpoint normalization is deterministic
and lives in `starling_endpoint_normalization.py`; those source vocabularies are
not sent to GPT.
