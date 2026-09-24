# Joseph Gold and TDC final test trace snapshot

This directory preserves the selected primary final-test traces from Joseph's Gold and TDC assay-transfer comparisons. It is a fixed snapshot of live `_batches` data captured between 2026-09-24T03:00:58+00:00 and 2026-09-24T03:05:52+00:00 at source commit `1f4946e2de44c29276ba0f91ee57e660a1e3ac91`. It contains 78,674 files (7,341,468,433 uncompressed bytes) across seven run families, stored in 9 compressed parts (418,663,227 bytes). These archives are raw inference evidence, not a claim that every run finished.

`status.tsv` records each selected branch's completion receipt at capture time: 7 complete, 17 incomplete, and 1 recovery branch without a completion receipt. Statuses may have changed in the live workspace after capture. See the original `completion.json` files in the archives for details.

## Find a task and benchmark

Every matrix in this snapshot uses the `test` evaluation split. The benchmark is either **Gold** (the Gold-v1 task labels) or **TDC** (the TDC task labels). Task IDs in the saved files are `bbb_martins` (BBB), `bioavailability_ma` (oral bioavailability), `skin_reaction` (Skin Reaction), `ames` (Ames), `dili` (DILI), and `carcinogens` (Carcinogens).

Each `.tar.zst.part-*` group is named for one family below. After extraction, look under `_batches/<family>/<branch>/`. The BBB-only family has its matrix directly under the family. Branch names identify the comparison method; `direct` uses direct context selection, while `mixed` combines direct and indirect selection. The suffixes such as `r2` and `ga100_mc010_label025` distinguish saved run variants and should be kept when comparing results.

| Family (archive filename prefix) | Branches | Benchmark and task |
| --- | --- | --- |
| `bbb_gold_direct_ga125_mc000_label025_test_v2_20260923_v1` | Family root | Gold: BBB |
| `bbb_oral_direct_mixed_full_test_upstream_v2_20260923_v1` | `gold_direct`, `gold_mixed_r2` | Gold: BBB, oral bioavailability |
| Same BBB/Oral family | `gold_direct_label025` | Gold: BBB |
| Same BBB/Oral family | `tdc_direct`, `tdc_mixed_r2` | TDC: BBB, oral bioavailability |
| `skin_direct_vs_joint_full_test_upstream_v2_20260923_v1` | `gold_direct`, `gold_mixed_r2`; `tdc_direct`, `tdc_mixed` | Gold and TDC respectively: Skin Reaction |
| `safety_direct_full_test_upstream_v2_relaunch_20260923_v2` | `gold_ames`, `gold_dili`, `gold_carcinogens`; matching `tdc_*` branches | Gold and TDC respectively: Ames, DILI, Carcinogens; one task per branch |
| `safety_direct_ga100_mc010_label025_full_test_v2_20260923_v1` | `gold_dili`, `gold_carcinogens`; matching `tdc_*` branches | Gold and TDC respectively: DILI or Carcinogens; one task per branch |
| `safety_direct_joint_full_test_upstream_v2_v27_20260923_v1` | `gold_ames_dili_mixed` | Gold: Ames, DILI |
| Same safety direct/joint family | `gold_carcinogens_direct`, `gold_carcinogens_mixed` | Gold: Carcinogens |
| Same safety direct/joint family | `tdc_dili_mixed` | TDC: DILI |
| `nonablation_full_test_recovery_20260923_v1` | `safety_direct_joint_full_test_upstream_v2_v27_20260923_v1/gold_carcinogens_direct/...` | Gold: Carcinogens direct recovery only |

For a precise check, read a branch's `matrix.json`: its `benchmark`, `tasks`, and `evaluation_subsets` fields define the saved matrix. The recovery tree has no top-level `matrix.json`; its path identifies the originating Gold Carcinogens direct branch, and its leaf `manifest.json` identifies the recovered experiment. `status.tsv` lists the branch paths and their status at snapshot time. A `complete` receipt applies only to that branch, not to every branch in its family.

## Restore and read the files

Each family is a deterministic tar/zstd stream split into parts of at most 95,000,000 bytes. From this directory, restore one family with:

```bash
mkdir -p restored
cat bbb_oral_direct_mixed_full_test_upstream_v2_20260923_v1.tar.zst.part-* \
  | zstd -dc | tar -xf - -C restored
```

This example reconstructs `restored/_batches/bbb_oral_direct_mixed_full_test_upstream_v2_20260923_v1/...`. Substitute another family prefix to restore a different archive. `sha256sum -c SHA256SUMS` checks the compressed parts before extraction.

Start with the branch's `matrix.json` for task, benchmark, split, prompt, model, and selection setup, then `completion.json` for the branch status. Under each task and experiment directory, `manifest.json` records the batch inputs and run configuration; `metrics.json` and `report.md` summarize available results; `predictions.jsonl` has one prediction record per line. For an individual query, open its `runs/<run-id>/request.json` to see the saved prompt and visible-reference index, and `final_reasoning_output.json` for the model result. The sibling `cache_matched_retrieval/runs/` tree holds `retrieval.json` evidence records; match its query index to the prediction or request run index. `trace_messages.jsonl` contains per-query trace records. Some incomplete runs have missing or empty outputs; use the receipt and the per-query `status` fields before computing aggregate results.

`manifest.tsv` lists every restored file's original source path, archive path, byte count, SHA-256, source mtime, and capture time. `snapshot.json` records the source commit and capture interval. The file selection keeps trace and result artifacts while omitting locks, raw progress logs, provider logs, cache payloads, unrelated optimization validation runs, and superseded comparison branches.

The original `_batches` trees remain in place. This branch is an archival snapshot; later live recoveries need a new snapshot identity.
