# Luna Gold Carcinogens and TDC-v2 AMES FINAL comparison

Both direct controls were rerun with the same `full_flat_context_v5_six_tasks_upstream_v3` prompt assets as their mixed counterparts. The earlier direct test runs used the `final_full_flat` default because `--mixed-harness-version` did not set the direct-grid prompt; those runs remain for audit. All four comparable test runs have complete receipts, zero failed queries, per-query traces, and recomputed metrics matching the stored metrics.

| Task | Test queries | Luna direct F1 | Luna mixed F1 | Existing DeepSeek direct / mixed F1 |
|---|---:|---:|---:|---:|
| Gold-v1 Carcinogens | 469 | 0.482328 | 0.623776 | 0.532669 / 0.647253 |
| TDC-v2 AMES | 1,457 | 0.787868 | 0.811343 | 0.782289 / 0.799350 |

The new Luna AMES cells exceed the existing DeepSeek cells. The new Luna Carcinogens cells fill empty Luna slots but trail the existing DeepSeek cells; this comparison does not isolate a model effect because routes and experiment configurations differ. The result table records each cell and its difference from the existing FINAL value.

All four runs are included in FINAL as model-specific Luna results. The existing DeepSeek arms remain the stronger Gold-v1 Carcinogens results in this comparison.

Gold Carcinogens screened direct and indirect profiles on a 100-query validation cohort; its selected direct and mixed macro F1 were 0.558638 and 0.634915. The partial Gold cache used per-query observed score means for selection during validation only, with imputed numbers hidden from the prompt; the test cache was fully scored. TDC-v2 AMES used the v3 parent-100 exact-score overlay. Its fixed direct profile scored 0.828472 on all 727 validation queries. Indirect profiles were screened on a 100-query cohort; the chosen mixed profile scored 0.880000 against the same direct panel's 0.848775. The full-validation mixed run was stopped and did not inform the choice, so AMES indirect remains a provisional 100-query validation selection.

`results.tsv` and `validation.tsv` contain every numerical result above. `provenance.json` pins the source matrices, batches, selections, metrics, predictions, and prompt hashes.
