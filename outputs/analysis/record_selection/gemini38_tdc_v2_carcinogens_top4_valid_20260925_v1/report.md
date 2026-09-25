# TDC-v2 Carcinogens Gemini profile search

Four profiles chosen from prior DeepSeek and Luna validation screens were rerun with Gemini 3.8 Flash on the 28-query validation set. All four returned the same prediction vector and macro F1 (0.717172). The frozen tie rule selected `ga075_mc025_sr050_sd025_ld025_rd025`, with the direct panel fixed at `ga075_mc000_label010`.

The selected mixed profile completed all 56 test queries with zero failures. Its macro F1 was 0.787879 (TN 40, FP 5, FN 3, TP 8), recomputed from predictions. This exceeds the earlier Gemini mixed test arm (0.773737) but remains below Gemini direct (0.830303). The difference to direct is -0.042424 macro F1. See `validation_results.tsv`, `test_results.tsv`, `freeze.json`, and `test_report_manifest.json` for numeric results and pinned sources.

FINAL contains four complete validation arms and the selected test arm under `final/gemini38/`. The test arm uses `full_flat_context_v5_six_tasks_upstream_v3`, the frozen direct profile `ga075_mc000_label010`, and the TDC-v2 exact-score overlay. All 168 Gemini final decisions (112 validation, 56 test) were served by Google AI Studio through OpenRouter, split between Flex and standard routes. The cached single-molecule query prior came from an earlier DeepSeek run and is preserved in each request and trace. The model index, route ledgers, prompt snapshot, selection manifests, and per-query files are hash-pinned in `finalization.json` and the arm manifests.

This test cohort had already been used in prior model and prompt comparisons. The profile was selected on validation, but the test number is descriptive rather than an untouched holdout estimate. No Gemini direct or earlier mixed FINAL entry was replaced.
