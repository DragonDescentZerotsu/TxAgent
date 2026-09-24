# Final Joseph full-test collection

The [final collection](outputs/paper/assay_transfer_harness/joseph/final/collection.json)
contains 14 complete arms and 3,968 successful test queries. It includes Gold
BBB, Oral Bioavailability, Carcinogens, DILI, and Skin Reaction direct and
direct+indirect arms; and TDC Oral Bioavailability and Skin Reaction direct and
direct+indirect arms. Each arm includes predictions, metrics, per-query run artifacts
and traces, a combined trace, a query-level model/provider ledger, and
hash-pinned source and copied-file manifests. The original batches remain at
their source paths.

The [original results report](outputs/analysis/record_selection/joseph_final_gold_bbb_oral_carcinogens_tdc_oral_20260924_v1/report.md)
covers the first seven arms. The [TDC Skin report](outputs/analysis/record_selection/joseph_final_tdc_skin_upstream_v2_test_20260924_v1/report.md)
and [TSV](outputs/analysis/record_selection/joseph_final_tdc_skin_upstream_v2_test_20260924_v1/results.tsv)
cover the selected upstream-v2 pair: direct macro-F1 0.634097 and
direct+indirect macro-F1 0.666875, each on all 82 test queries. Gold Carcinogens
direct uses `ga075_mc000_label000`. The [Gold Skin and TDC Oral direct report](outputs/analysis/record_selection/joseph_final_gold_skin_tdc_oral_direct_20260924_v1/report.md)
and [TSV](outputs/analysis/record_selection/joseph_final_gold_skin_tdc_oral_direct_20260924_v1/results.tsv)
cover three further arms. TDC Oral direct+indirect joins 87 recovery-leaf queries
with 41 valid predecessor runs. Each arm's `query_provenance.tsv` records its
actual served model and provider.

The [recovery snapshot](outputs/analysis/record_selection/full_test_upstream_v2_recovery_20260924_v1/report.md)
is a dated selection aid; final arm manifests pin their installed source leaves.
The full collection, including traces, is preserved in Git as
[compressed parts](outputs/paper/assay_transfer_harness/joseph/final/git_bundle/README.md).
Rebuild the bundle with
`bash outputs/paper/assay_transfer_harness/joseph/final/git_bundle/build.sh`
after adding an approved arm or trace.
