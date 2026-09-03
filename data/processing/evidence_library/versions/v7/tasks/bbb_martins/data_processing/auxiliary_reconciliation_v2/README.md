# BBB auxiliary reconciliation v2

This directory is the human-gated second pass over the six BBB auxiliary
mapping namespaces. It preserves the exact cluster-local GPT output before
review and keeps the currently published v6 mapping as the rollback baseline.

Directory layout:

```text
provenance/
  provisional_cluster_mapping.json   exact cluster-local output
  cluster_assignments.parquet        one row per non-null raw assignment
  pre_reconciliation_mapping.json    byte-identical current published mapping
  provisional_manifest.json          input/output hashes and the two old deltas
label_catalog.parquet                 labels aggregated within namespace
review_packets/                       disjoint, cluster-atomic reviewer inputs
reviews/                              primary, checker, and adjudicator records
proposal/                             unpublished candidate mapping and change audit
```

The six `source_id/output_field` namespaces are independent. A decision may
merge labels across local embedding clusters, but never across namespaces.
Every non-null provisional label must receive exactly one primary disposition.
Every proposed change must then be checked by a different reviewer and accepted
or rejected by an adjudicator.

Commands:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m data.processing.evidence_library.versions.v7.tasks.bbb_martins.data_processing.reconcile_auxiliary_mapping \
  snapshot --overwrite

/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m data.processing.evidence_library.versions.v7.tasks.bbb_martins.data_processing.reconcile_auxiliary_mapping \
  prepare-review
```

`propose` requires explicit primary, checker, and adjudicator JSONL inputs. The
CLI intentionally has no publish command. Human approval is required before
replacing `globally_reconciled_auxiliary_value_mapping.json`, changing its
runtime mapping version, or rebuilding normalized stages 02–09.
