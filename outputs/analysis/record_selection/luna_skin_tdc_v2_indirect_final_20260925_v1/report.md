# Luna TDC-v2 Skin indirect FINAL successor

Five new full-validation profiles were evaluated with frozen direct profile `ga100_mc000_label025`. `ga100_mc025_sr050_sd025_ld025_rd025` ranked first among the eight total mixed profiles at 0.702502 macro F1 on 40 validation queries. The frozen profile scored 0.611005 macro F1 on all 82 test queries. The two other test profiles scored 0.632051 and 0.579488; they remain exploratory raw runs because test scores did not choose the FINAL successor.

The new successor is `final/luna/test/tdc_v2/skin_reaction/direct_plus_indirect_valid_selected_v2/`. The earlier mixed FINAL arm scored 0.605770 and remains intact. The Luna direct equivalent-cohort replay scored 0.700912, but it used another prompt and retrieval lineage, so that cross-arm difference is descriptive. The selected mixed arm used `full_flat_context_v5_six_tasks_tdc_v2_mixed_v1`, exact-score TDC-v2 retrieval, OpenRouter Flex `openai/gpt-6-luna`, and high reasoning effort.

All five new validation arms and the selected test arm have complete per-query requests, responses, traces, route ledgers, frozen selection manifests, and source hashes. The validation launch required two missing-only recovery batches; the selected test launch recovered four missing requests across two retries. The selected test confusion matrix is TN 16, FP 19, FN 11, TP 36. The mixed model continues to overpredict positives. Every 282 copied response requested and served Luna through OpenAI.

`results.tsv` contains the numeric comparisons, and `run_audit.tsv` contains coverage and recovery counts. `provenance.json` pins the raw analysis, parameter snapshot, selected FINAL arm, indexes, and table. The compressed Git archive was not rebuilt for this local FINAL update.
