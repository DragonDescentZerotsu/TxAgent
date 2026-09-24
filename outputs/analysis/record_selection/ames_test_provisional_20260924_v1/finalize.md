# AMES test snapshot (provisional)

Snapshot: 2026-09-24 13:44:55 UTC. This is **not** a final test publication: both direct + indirect runs were still active. The numbers below are calculated from valid `final_reasoning_output.json` responses, using each benchmark's frozen test labels. The same-query direct rows make the provisional comparisons fair despite incomplete indirect coverage. Machine-readable numbers are in `results.tsv`; source paths and snapshot hashes are in `manifest.json`.

| Benchmark | Retrieval | Valid / test | Macro-F1 | Accuracy | Status |
|---|---|---:|---:|---:|---|
| Gold v1 AMES | Direct only | 274 / 274 | 0.708578 | 0.748175 | Complete |
| Gold v1 AMES | Direct only, on indirect-valid queries | 273 / 274 | 0.711416 | 0.750916 | Matched comparison |
| Gold v1 AMES | Direct + indirect | 273 / 274 | 0.773887 | 0.816850 | Provisional |
| TDC v2 AMES | Direct only | 1,457 / 1,457 | 0.782289 | 0.792725 | Complete response coverage across original and recovery runs |
| TDC v2 AMES | Direct only, on indirect-valid queries | 1,363 / 1,457 | 0.783357 | 0.792370 | Matched comparison |
| TDC v2 AMES | Direct + indirect | 1,363 / 1,457 | 0.800735 | 0.811445 | Provisional |

Gold direct + indirect is ahead by 0.062471 macro-F1 on the 273 matched queries. Query 151 remains unresolved at this snapshot: two attempts had invalid final responses and two requests were still active. TDC direct + indirect is ahead by 0.017378 macro-F1 on the 1,363 matched queries. Eleven TDC responses had invalid final outputs, and 83 had no final output yet. These provisional differences can change as the runs finish.

The Gold direct run uses `ga075_mc025_label025`; its indirect selection uses `ga100_mc010_sr050_sd010_ld010_rd010`. The TDC direct run uses `ga050_mc010_label025`; its indirect selection uses `ga075_mc010_sr035_sd010_ld010_rd010`. The TDC direct score combines valid responses from two runs; it is not a single completed run receipt. The inference prompt is `full_flat_context_v5_six_tasks_upstream_v3`, with DeepSeek V4 Flash 0731 and high reasoning effort.

Response artifacts remain in their raw Joseph run directories under `outputs/paper/assay_transfer_harness/joseph/`: `ames_direct_v3_full_test_20260924_v1`, `ames_gold_indirect_best_full_test_upstream_v3_20260924_v1`, `ames_gold_indirect_best_full_test_upstream_v3_openrouter_resume_20260924_v1`, `ames_gold_indirect_query151_parallel_a_20260924_v1`, `ames_gold_indirect_query151_parallel_b_20260924_v1`, `ames_direct_exactscore_full_test_upstream_v3_20260924_v1`, `ames_direct_exactscore_full_test_upstream_v3_openrouter_resume_20260924_v1`, and `tdc_v2_ames_indirect_best_full_test_upstream_v3_20260924_v1`. This report copies the response-derived results, not the large raw traces. No incomplete run is marked complete here.
