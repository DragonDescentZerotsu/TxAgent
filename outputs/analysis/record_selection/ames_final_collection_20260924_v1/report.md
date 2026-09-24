# AMES final-test collection addition

The final Joseph collection now includes Gold-v1 and TDC-v2 AMES direct and direct-plus-indirect arms. All four arms have complete valid-response coverage, per-query prompts and traces, predictions, metrics, and source/provider provenance. The result numbers are in `results.tsv`.

| Benchmark | Direct macro-F1 | Direct + indirect macro-F1 | Queries per arm |
|---|---:|---:|---:|
| Gold v1 | 0.708578 | 0.774211 | 274 |
| TDC v2 | 0.782289 | 0.799350 | 1,457 |

The collection has 24 arms and 8,546 arm-query results after this addition. Existing 20 arms and their manifest hashes remain unchanged. The TDC mixed arm's final query (1158) came from a fresh high-reasoning request with a 16,384-token ceiling; its rendered prompt was byte-identical to the earlier 65,536-token attempt. The arm provenance identifies its route and this exception. Historical partial source matrices are not relabeled as complete.
