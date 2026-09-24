# Canonical TDC Skin Reaction test results

The user-selected upstream-v2 optimization pair is installed in the [final collection](../../../paper/assay_transfer_harness/joseph/final/collection.json). Both arms cover all 82 TDC-v1 Skin Reaction test queries. The results below reproduce the completed source metrics; no inference was launched for this copy.

| Selection | Optimization profile | Test macro-F1 | Accuracy | Complete queries |
|---|---|---:|---:|---:|
| Direct | `ga100_mc010_label025` | 0.634097 | 0.646341 | 82/82 |
| Direct + indirect | `ga075_mc010_sr050_sd010_ld025` | 0.666875 | 0.682927 | 82/82 |

The [TSV](results.tsv) and [provenance](provenance.json) pin each source batch, optimization manifest, and final arm. Each final arm includes the predictions, metrics, all per-query run files and traces, a combined trace, and a file-hash ledger. The source batches remain unchanged.
