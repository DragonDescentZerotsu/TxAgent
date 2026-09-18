# Semantic-bucket ownership

This directory owns all semantic-bucket code, prompts, policies, artifacts,
provenance, history, audits, and tests for BBB and Oral.

## Invariants

- Resolve active artifacts through `artifacts.py` and each release manifest.
- Resolve current BBB/Oral levels from the active evidence-library release;
  gold-owned mappings are retained only for explicit version-specific workflows.
- A semantic bucket combines pair buckets only within one evidence level.
- A readout bucket has exactly one semantic parent.
- Join records by `canonical_record_id` and `source_row_uid`, never display text.
- Keep reviewed generations immutable; publish a new generation for policy changes.
- Active request ledgers may use `/local`, but once the writer finishes, copy the
  closed run to `/vast`, validate and publish it under `provenance/`, then remove
  local staging. A run is not durable or published while only on `/local`.
- Newly launched LLM work uses `reasoning_effort=high`.
- V5 and V6 seed requests remain on `deepseek/deepseek-v4-pro`. Historical V5
  anchored rows use the mixed gold OpenRouter pool. After a recorded migration,
  remaining V5 anchored rows may use the benchmark-selected speculative
  DGX executor. The current post-round-127 V5 contract uses fixed fanout 16 and
  takes the first four schema-valid Flash replicas; earlier DGX rows retain their
  recorded first-eight contract. Average each weight with half-up hundredth
  rounding, retain per-bucket standard deviation and range, and use the full-vector
  closest replica for rationales. Preserve all
  completed pre-migration rows and record the benchmark hash and replica receipts.
  If migration enters one partially completed round, record its old/new request
  counts explicitly and never replace its completed historical responses. Advance
  each task-level chain independently once its own three-batch round has completed;
  never impose a cross-task or cross-level round barrier.
- V6 may start early only as a task-level shard whose V5 task-level is complete.
  Freeze the selected V5 rows and source manifest inside the V6 shard; never read
  mutable V5 partial weights at execution time. Reuse the shard's exact request
  rows when assembling the eventual full V6 run instead of repeating paid calls.
  The current V6 shard uses DGX008 Flash for seed and anchored requests with high
  reasoning, fixed fanout 16, and the first four schema-valid responses. It must
  not reuse earlier OpenRouter Pro seed rows.
- Do not add compatibility symlinks or duplicate canonical releases.
- Keep every newly written or materially refactored function at or below 60 lines.

Run `pytest -q semantic_buckets/tests` after changes.
