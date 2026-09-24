# Canonical Gold Skin and TDC Oral direct test results

The selected upstream-v2 optimization arms are installed in the [final collection](../../../paper/assay_transfer_harness/joseph/final/collection.json). The completed source metrics were recomputed before copying; no new inference was launched.

| Benchmark and task | Selection | Test macro-F1 | Accuracy | Complete queries |
|---|---|---:|---:|---:|
| Gold-v1 Skin Reaction | Direct | 0.599320 | 0.705394 | 241/241 |
| Gold-v1 Skin Reaction | Direct + indirect | 0.615042 | 0.713693 | 241/241 |
| TDC-v1 Oral Bioavailability | Direct | 0.746709 | 0.820312 | 128/128 |

Gold Skin direct includes one locally served recovery response, and Gold Skin direct + indirect includes eleven. Each arm's `query_provenance.tsv` records the actual served model and provider. The [TSV](results.tsv) and [provenance](provenance.json) pin the source leaves, selection manifests, and final arm manifests. Final arms contain per-query run files and traces as well as combined traces; source batches remain unchanged.
