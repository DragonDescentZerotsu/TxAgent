# Normalized evidence library core

The canonical library has three task-independent stages. Paper-specific
retrieval views, held-out filtering, neighbor indices, and benchmark label
overlays are downstream products, not row-acceptance rules in this core.

## Stage 1: cleaning

Stage 1 applies source cleaning plus task-local repair/drop registries and
SMILES repairs. A row is retained only when it has a nonempty canonical SMILES
with `structure_status == "resolved"`; rejected rows remain in
`01_cleaned/structure_rejections.parquet` for lineage. Task-specific behavior
is supplied through each task's `StarlingTaskPolicy` hooks.

## Stage 2: canonicalization

Stage 2 canonicalizes measurements, units, endpoints, and task-declared context
columns while retaining raw source values. Unknown context values are kept as
unknowns rather than used to reject records.

Exact measurement/unit reconciliation is conservative:

1. Existing exact rules are applied first.
2. Whitespace-normalized aliases are accepted automatically only when they
   resolve to one existing rule.
3. Remaining units are grouped in packets of 50. Character 1-4 gram TF-IDF
   cosine similarity supplies the top 100 existing unit strings for each new
   unit; the packet contains the union of those candidates and their existing
   rules.
4. Review chooses an existing canonical unit plus a positive scale, or preserves
   the new unit at scale 1. Reviewed rules are persisted in each task's
   `exact_measurement_unit_map.v2.json`.

The fuzzy score only narrows the review context. It never performs an automatic
scientific merge.

Unit reconciliation never discards a finite resolved quantity. A scientifically
fixed conversion is mapped; otherwise the exact reported unit is preserved at
scale 1. Controlled categorical encoding is used only when the row has no scalar
measurement, and it never creates a sibling copy of a scalar record.

## Stage 3: pair buckets, deduplication, and calibration

Stage 3 writes one directory, `03_pair_buckets/`, containing deduplicated
records, the aligned pair-bucket sidecar, duplicate lineage, direct-vote
mapping, bucket metadata, and distance calibration.

- Pair identity is source-scoped: source, canonical endpoint, canonical unit,
  and task-declared assay/context dimensions.
- Reference scope is metadata, not identity. Missing dimensions use
  `__unknown__` instead of causing row deletion.
- Numeric calibration accepts only finite `absolute` and
  `endpoint_defined_ratio` records with resolved units. Comparator-relative,
  standardized-control, unknown-scope, and free-text records stay in the
  library but do not calibrate numeric transfer.
- Controlled binary/ordinal categories use their declared scale and are not
  subject to the numeric reference-scope gate.
- A bucket needs at least 20 deduplicated records and 16 distinct molecules.
- `direct_vote` rows are retained with their gold-train mapping but do not
  calibrate transfer. `direct_residual` and `indirect` remain supported.
- There is no record collapse, residual-heterogeneity gate, numeric-domain
  rejection, or automatic endpoint-pruning rejection in the core.

Skin partition decisions and direct/nondirect decisions are retained as
metadata for later task views; they do not delete canonical rows.
