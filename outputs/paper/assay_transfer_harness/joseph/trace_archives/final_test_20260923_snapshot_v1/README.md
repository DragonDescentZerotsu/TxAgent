# Joseph Gold and TDC final test trace snapshot

This directory preserves the selected primary final-test traces from Joseph's Gold and TDC assay-transfer comparisons. It is a fixed snapshot of live `_batches` data captured between 2026-09-24T03:00:58+00:00 and 2026-09-24T03:05:52+00:00 at source commit `1f4946e2de44c29276ba0f91ee57e660a1e3ac91`. It contains 78,674 files (7,341,468,433 uncompressed bytes) across seven run families, stored in 9 compressed parts (418,663,227 bytes). These archives are raw inference evidence, not a claim that every run finished.

`status.tsv` records each selected branch's completion receipt at capture time: 7 complete, 17 incomplete, and 1 recovery branch without a completion receipt. Statuses may have changed in the live workspace after capture. See the original `completion.json` files in the archives for details.

Each family is a deterministic tar/zstd stream split into parts of at most 95,000,000 bytes. To restore one family from this directory:

```bash
cat FAMILY.tar.zst.part-* | zstd -dc | tar -xf - -C /path/to/restore
```

This reconstructs `_batches/FAMILY/...`. `SHA256SUMS` checks compressed parts. `manifest.tsv` gives the original source path, restored archive path, byte count, SHA-256, source mtime, and capture time for every file. `snapshot.json` records the source commit and capture interval. The file selection keeps trace and result artifacts (`trace_messages.jsonl`, reasoning outputs, request/retrieval/manifest JSON, predictions, metrics, reports, and status receipts). It omits locks, raw progress logs, provider logs, cache payloads, unrelated optimization validation runs, and superseded comparison branches.

The original `_batches` trees remain in place. This branch is an archival snapshot; later live recoveries need a new snapshot identity.
