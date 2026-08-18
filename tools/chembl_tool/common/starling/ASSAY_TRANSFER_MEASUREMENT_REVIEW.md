# Assay-transfer canonical measurement review

## Contract

`canonical_measurement_text`, `canonical_unit_text`, and `finite_scalar_value` are the numerical contract
for assay transfer. They are not retrieval-facing evidence. Retrieval and LLM presentation use the cleaned
source projection (`measurement_text`, `unit_text`, support text, and declared context fields), and a change
here must not change `retrieval_eligible` or retrieval membership.

Before a production rebuild, `audit_assay_transfer_retrieval_invariance.py` snapshots an order-independent
digest of cleaned-row membership, molecule/family identity, duplicate-group identity and size, retrieval
status, and the exact source projection. The
post-build comparison must match. Canonical measurement/unit/scalar fields and artifact file hashes are
intentionally outside that digest.

```bash
python -m tools.chembl_tool.common.starling.audit_assay_transfer_retrieval_invariance \
  --task <task_id> --records <old-03-records.parquet> --snapshot <snapshot.json>
python -m tools.chembl_tool.common.starling.audit_assay_transfer_retrieval_invariance \
  --task <task_id> --records <rebuilt-03-records.parquet> --snapshot <snapshot.json> --compare
```

The final canonical tuple is composed through one pipeline, in this order: shared parsing and unit arithmetic;
task endpoint standardization; source/context reconciliation; controlled encoding and task validity enrichment;
the frozen assay-transfer base-unit correction; v7 canonical projection; and the frozen raw/log10 tail transform.
The pre-base pair is retained only to freeze retrieval deduplication identity. Stage 04 retains every record and adds
`assay_transfer_eligible`/`assay_transfer_ineligibility_reason`; probable unit defects receive no bucket key
but are not removed from the sidecar or from retrieval.

Every finalized Stage-03 row declares
`assay_transfer_measurement_contract_version=assay_transfer_canonical_tuple.v1`, its raw/log10 transform ID,
and its task measurement-policy version. Consumers must fail closed when this provenance is absent or mixed;
they never infer or repeat the transformation. Stage 05 separately publishes
`assay_transfer_bucket_eligible` and `assay_transfer_bucket_ineligibility_reason`. These bucket fields are
distinct from the Stage-04 row fields. Residual-heterogeneity flags never contribute to either verdict.

Production assay-transfer-only rebuilds start from the frozen Stage-03 records so unrelated changes to endpoint,
category, molecule, or retrieval deduplication code cannot enter the artifact. The tuple-only entrypoint updates
the Stage-03 canonical measurement fields and provenance, after which the normal task wrapper rebuilds Stage 04/05:

```bash
python -m tools.chembl_tool.common.starling.rebuild_assay_transfer_stage03 \
  --task <task_id> --root <staged-v7-root>
python -m tools.chembl_tool.tasks.<task_id>.build_normalized_starling_evidence_library \
  --out-dir <staged-v7-root> --from-stage index --through-stage index \
  --validation-level full --cache-mode off
```

Newly scalar-valid rows that had no historical reference-semantics assignment remain explicitly `unknown` and
assay-transfer-ineligible unless a frozen deterministic rule resolves them. The build never guesses a reference
scope to make a repaired row eligible.

Scientific notation such as `10-6`, `10−6`, `10‐6`, or `10‑6` denotes the factor `10^-6 = 1e-6` when it is
unambiguously attached to the reported measurement. A support-text factor is applied once; already-scaled
measurements are not scaled again.

## Tail-candidate review protocol

A positive continuous bucket with at least 25 records is reviewed when either its absolute raw skew is at
least 2 or `log10(Q95 / median) >= 2`. Reviewers inspect the endpoint meaning, unit, raw and log summaries,
and representative support text. The statistical screen only nominates candidates: the frozen task policy
is the authority for `raw` versus `log10`.

## Dataset decisions

Independent reviews used the same screen, then froze exact post-canonicalization pair-bucket keys in each
task's `data_processing/assay_transfer_measurements_v1/policy.json`.

| Task | reviewed major-tail buckets | `log10` | raw | reviewed unit-defect rows marked assay-transfer-ineligible |
|---|---:|---:|---:|---:|
| BBB Martins | 97 | 96 | 1 already-log permeability bucket | 18 |
| Bioavailability_Ma | 224 | 222 | 2 bounded fraction buckets | 75 |
| Skin_Reaction | 120 | 113 | 7 bounded applied-dose fraction buckets | 216 |

BBB's ordinary `10^-6` forms were already correct: 2,500 reviewed Stage-03 rows had a persisted `1e-6`
notation factor. The BBB repairs instead address percent-to-ratio base normalization and 18 evidence-backed
lost-sign/omitted-factor defects. The existing `-log10(cm/s)` endpoint remains raw and is never double logged.

Bioavailability repairs include U+2010 `10‐6` parsing in the shared unit layer; distinct per-kg and
per-mg-protein intrinsic-clearance bases; inverted per-million-cell factors; and reviewed table-header
factors. Valid kg-based clearances move to `mL/min/kg` or `mL/min/kg liver`; they are not discarded.

Skin repairs include explicit support-heading factors, denominator-preserving percent units, and four
`cm/h` to `cm/s` corrections. Percent in a numerator uses factor `0.01`; percent in a denominator uses the
reciprocal factor. For example, `% applied dose` becomes `fraction of applied dose` with factor `0.01`, while
`mL/%·m^2·min` becomes `mL/fraction·m^2·min` with factor `100`.

The shared `review_assay_transfer_measurements.py` command reproduces the broad bucket screen and nominates
positive-factor log-tail rows. It never changes policy automatically. A row may be marked
assay-transfer-ineligible only after its measurement, unit, support text, and exact-decade landing evidence
are reviewed. Residual heterogeneity remains in Stage-05 audit metadata and never rejects an entire bucket.
