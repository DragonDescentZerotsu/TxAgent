# TDC-v2 BBB final test arms

Both arms completed all 406 original-split test queries with the corrected upstream-v3 prompt and fixed direct profile `ga075_mc010_label025`. The direct+indirect arm used validation-selected profile `ga075_mc025_sr025_sd025_ld025_rd025` and removed prediction-only L2-L5 records before joint selection of 50 indirect records. Every installed query has its request, prediction, run manifest, retrieval, and trace. Query-level served-model and provider details are in each arm's `query_provenance.tsv`.

| Selection | Queries | Macro F1 | Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 406 | 0.865673 | 0.913793 | 64 | 14 | 21 | 307 |
| Direct+indirect | 406 | 0.880814 | 0.928571 | 60 | 18 | 11 | 317 |

The filtered joint arm improves macro F1 by 0.015141 and accuracy by 0.014778. These are test-tuned exploratory configurations. `results.tsv` exports the numeric table, and `provenance.json` pins the installed arms and collection.
