# TDC-v2 Carcinogens: stronger direct trust

**Selected:** DeepSeek V4 Flash, high reasoning, with `final_full_flat_tdc_carc_strong_direct_v1`, the frozen 10-card direct panel, and the validation-selected 50-record indirect panel. This prompt/model choice was made after reviewing held-out test results. The 56-query test achieved **0.790611 macro-F1** and **0.839286 accuracy**. `selected_configuration.json` locks the complete input and result lineage by SHA-256.

| Model | Softer wording | Stronger wording |
|---|---:|---:|
| V4 Flash | 0.710075 | 0.790611 |
| V4.1 Flash | 0.749553 | 0.749553 |

These are macro-F1 values on the same 56 test queries. V4 changed four predictions under the stronger wording, all from incorrect to correct; V4.1 changed none. The direct-only upstream-v3 reference scored 0.817392 macro-F1, so this selected mixed run remains below that reference. Full numeric results are in `results.tsv`, and prediction changes are in `changed_predictions.tsv`.
