# Final Joseph full-test collection

The selected Gold BBB, Oral Bioavailability, and Carcinogens direct and
direct+indirect optimized arms, plus TDC Oral Bioavailability direct+indirect
optimized, are installed under
[final](outputs/paper/assay_transfer_harness/joseph/final/collection.json).
The collection contains seven complete arms and 2,390 successful test queries.
Each arm includes predictions, metrics, per-query run artifacts and traces,
a combined trace, a query-level model/provider ledger, and hash-pinned source
and copied-file manifests. The original batches remain at their source paths.

The [results report](outputs/analysis/record_selection/joseph_final_gold_bbb_oral_carcinogens_tdc_oral_20260924_v1/report.md)
and [results table](outputs/analysis/record_selection/joseph_final_gold_bbb_oral_carcinogens_tdc_oral_20260924_v1/results.tsv)
summarize the selected arms. Gold Carcinogens direct uses the complete
\`ga075_mc000_label000\` profile. Gold arms include 46 local recovery responses
served as \`deepseek-v4-flash-0731\`; these are mixed-route results, and every
query's served model is recorded in its arm's \`query_provenance.tsv\`.
TDC Oral joins 87 recovery-leaf queries with 41 valid predecessor runs.

The final copy passed query-coverage, recomputed-metric, prediction-to-trace,
and copied-file hash checks after installation. The dated
[recovery snapshot](outputs/analysis/record_selection/full_test_upstream_v2_recovery_20260924_v1/report.md)
remains a historical selection aid; its Gold source hashes predate the
reconciled source leaves now pinned in the final arm manifests.

The complete collection, including all traces, is also preserved in Git as
[two compressed parts](outputs/paper/assay_transfer_harness/joseph/final/git_bundle/README.md).
The bundle restores all 16,781 collection files byte-for-byte. Rebuild it with
\`bash outputs/paper/assay_transfer_harness/joseph/final/git_bundle/build.sh\`
after adding a trace; Git sees the refreshed parts without forced staging.
