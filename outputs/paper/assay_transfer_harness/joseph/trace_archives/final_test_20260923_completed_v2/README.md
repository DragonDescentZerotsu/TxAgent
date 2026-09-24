# Joseph Gold and TDC final test traces: completed v2

This immutable successor combines the primary final-test matrices with their separately completed recovery requests. It was captured at 2026-09-24T06:40:09.900153+00:00 from source commit `5c3a6ae257ffe73ec2e57199ff494c9e11ed2082`. The previous `final_test_20260923_snapshot_v1` remains the original partial snapshot.

All **24 matrix branches** and **31 result leaves** in this archive are complete: **9,755 successful queries**, including **340 recovered queries**. `status.tsv` gives branch counts; `metrics.tsv` gives each result leaf's metrics; `manifest.tsv` gives every archived file's source path, size, SHA-256, and capture time. The archive has 79,825 files (7,471,776,139 uncompressed bytes) in 8 parts (380,959,454 compressed bytes).

The six archive prefixes are `bbb_gold_direct_ga125_mc000_label025_test_v2_20260923_v1`, `bbb_oral_direct_mixed_full_test_upstream_v2_20260923_v1`, `safety_direct_full_test_upstream_v2_relaunch_20260923_v2`, `safety_direct_ga100_mc010_label025_full_test_v2_20260923_v1`, `safety_direct_joint_full_test_upstream_v2_v27_20260923_v1`, `skin_direct_vs_joint_full_test_upstream_v2_20260923_v1`. Each matrix uses the `test` split. Its `matrix.json` identifies the benchmark, tasks, prompt, and selection. Each original result leaf now has combined `predictions.jsonl`, `metrics.json`, `report.md`, and `trace_messages.jsonl`. Its `reconciliation.json` pins the original and recovery source hashes and lists the recovered query indices. The original failed attempts remain under `runs/`; successful recovery attempts are under `recovery/runs/`. No inference was launched during reconciliation.

Restore one family from this directory with:

```bash
mkdir -p restored
cat bbb_oral_direct_mixed_full_test_upstream_v2_20260923_v1.tar.zst.part-* \
  | zstd -dc | tar -xf - -C restored
```

Check compressed parts first with `sha256sum -c SHA256SUMS`. Restored files appear under `restored/_batches/<family>/`. Run paths inside saved `predictions.jsonl` record their original live workspace locations; use the archive-relative paths in `manifest.tsv` to locate extracted copies.
