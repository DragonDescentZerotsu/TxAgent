# TDC-v2 DILI indirect validation

All 18 profiles completed on 47 validation queries. The direct panel was frozen at `ga075_mc025_label000`; indirect selection used the parent-disjoint top-100 universe, geometric DILI score, and tied diversity terms.

Winner: `ga050_mc025_sr025_sd025_ld025_rd025` (macro F1 0.825465, accuracy 0.851064). Three profiles tied on both metrics; the lexical profile name broke the tie. Full results are in `validation_metrics.tsv`; pinned inputs are in `validation_selection.json`.

## Test result

The frozen validation winner completed 96/96 TDC-v2 test queries with macro F1 0.760436 and accuracy 0.770833. The earlier direct-only upstream-v3 test completed 96/96 with macro F1 0.829711 and accuracy 0.833333. The indirect run was lower by 0.069275 macro F1 and 0.062500 accuracy. The two runs use different prompt versions and candidate policies, so this comparison is descriptive rather than a controlled ablation.

The semantic projection extension covers 1,487 successor UIDs from the selected central ledger; 142 use provisional task-level bucket means and its manifest remains `candidate_unreviewed`. The parent audit is in `parent_overlap_audit.tsv`. Complete test metrics are in `test_results.tsv`; hashes and raw run paths are in `test_result_manifest.json`.

## Paired direct versus indirect comparison

The same 96 query structures, labels, cached query priors, and ten ordered L1 context cards were used in both runs. The indirect selection added 50 L2–L7 records per query. Predictions changed on 16 queries: 11 losses and 5 gains. Nine negative queries flipped from correct `no_dili_risk` to false-positive `dili_risk`, while three false positives were recovered; two positives were lost and two recovered. Median prompt size rose from 7,298 to 76,904.5 characters. Ten of the 11 losses had high assay-score coverage.

For query index 0, the indirect answer cited liver-injury reports for related ergoline molecules and flipped a benchmark-negative query to `dili_risk`. Similar positive-analog transfer appears in other newly false-positive summaries. This supports a plausible overtransfer mechanism, but the test also changed system guidance and OpenRouter routing, so the effect of indirect data alone is not identified. Paired query outcomes and prompt lengths are in `paired_test_changes.tsv`; aggregate counts are in `paired_test_summary.tsv`; immutable input hashes are in `paired_comparison_manifest.json`.

## Validation trace audit

A subsequent [same-prompt direct-only control](../dili_tdc_v2_direct_same_prompt_valid_20260924_v1/report.md) completed these 47 queries. It had five false positives among 17 negatives and scored macro F1 0.831541, versus six false positives and 0.825465 for the indirect winner. Only two predictions changed, and their upstream providers differed. Thus most validation false positives were already present without indirect records. Across the 18 indirect profiles, all had 6–8 false positives; four negative queries were called `dili_risk` in every profile, with three more in at least 16 profiles. The three tied leaders differed on 2–4 query predictions.

The winner’s negative-error summaries repeatedly elevated related-molecule DILI labels, animal hepatocyte injury, reactive-metabolite mechanisms, or cholestatic-cell assays into a clinical-risk call. Cases include validation indices 6, 13, and 44; this is a qualitative reading of final claims, not an assay-validity adjudication. On test, negative specificity declined from 0.647059 to 0.586957, while positive sensitivity changed from 0.966667 to 0.940000. Validation was 30/47 positive versus 50/96 on test, making the existing false-positive tendency more costly on test. Per-negative profile error frequencies are in `validation_negative_errors.tsv`, subset rates in `validation_test_class_balance.tsv`, and source hashes in `validation_trace_audit_manifest.json`.

The top three indirect profiles selected materially different records (pairwise median overlap 30–38 of 50 per query), yet the four universal false positives persisted. See `top_profile_overlap.tsv`.
