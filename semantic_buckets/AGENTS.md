# Semantic-bucket ownership

This directory owns all semantic-bucket code, prompts, policies, artifacts,
provenance, history, audits, and tests for BBB, Oral, and Skin.

## Invariants

- Resolve active artifacts through `artifacts.py` and each release manifest.
- The evidence-library `CURRENT` pointer owns the active semantic-bucket release;
  do not add a second semantic-release pointer.
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
  closest replica for rationales. Preserve all completed pre-migration rows and
  record the benchmark hash and replica receipts. If migration enters one partially
  completed round, record its old/new request counts explicitly and never replace
  its completed historical responses. Advance each task-level chain independently
  once its own three-batch round has completed; never impose a cross-task or
  cross-level round barrier.
- V6 may start early only as a task-level shard whose V5 task-level is complete.
  Freeze the selected V5 rows and source manifest inside the V6 shard; never read
  mutable V5 partial weights at execution time. Reuse the shard's exact request
  rows when assembling the eventual full V6 run instead of repeating paid calls.
  The current V6 shard uses DGX008 Flash for seed and anchored requests with high
  reasoning, fixed fanout 16, and the first four schema-valid responses. It must
  not reuse earlier OpenRouter Pro seed rows.
- BBB and Oral accept their completed V5/V6 rankings and must not be rerun merely
  to conform to a successor workflow. For future tasks, define the scoring world
  independently for each task-level as the L2+ union of physical UIDs under the
  Morgan-top-100 parents across valid and test queries. Never mix levels in one
  prompt, ordering, or anchor chain. Pass one uses deterministic independent
  batches of 12 with no prior weights or anchors; every batch may run concurrently,
  with at most 128 logical scoring calls per endpoint. Pass two sorts each
  task-level by its pass-one score, jointly rescores the first 12, carries items
  8-12 as five anchors, then scores batches of seven while carrying the previous
  batch's final two items. Task-level chains advance independently. Both passes
  use high reasoning, Flash-0731, fixed fanout 16, and the first four schema-valid
  responses; average with half-up hundredth rounding, retain standard deviation
  and range, and use the full-vector closest replica for rationales. Pass-two
  weights are final, while pass one remains immutable ordering and audit evidence.
  Record endpoint identities only in execution provenance.
- Skin uses its reviewed semantic families directly as the scoring unit. Do not
  insert a semantic-to-readout splitting stage into the Skin weight workflow.
  Rank only families present in the frozen Morgan-top-100 retrieval world and
  retain uncovered reviewed families as an explicit audit table.
- Do not add compatibility symlinks or duplicate canonical releases.
- Keep every newly written or materially refactored function at or below 60 lines.

## Official two-pass successor

Official successor runs use the V6 scientific bucket-card representation in both
passes. Pass 1 has no anchors and uses exactly one request and one valid response
per logical batch. After Pass 1, freeze its ranking, render the exact first Pass-2
prompt for every level, and stop at `awaiting_pass2_prompt_review`; Pass 2 must not
launch without explicit approval of that review manifest hash.

For the AMES V6-card run, alternate the complete sorted Pass-2 schedule between
local DGX execution and the shared OpenRouter qualified pool, yielding the closest
possible half split without splitting by level. Both backends use fanout 4 and
the first two schema-valid responses. Freeze and hash-pin the seven-route
DeepSeek V4 Flash 0731 OpenRouter profile at Pass-2 start, use its shared spend
credential loader without a spend ledger, and preserve actual routes in replica receipts.
Do not embed provider inventories or credentials in the task runner.

Preserve superseded responses for audit, but never reuse scores produced from a
different bucket representation in a successor run. Candidate weights from an
unreviewed semantic generation remain unreviewed and cannot update an active
release or `CURRENT` pointer.

Run `pytest -q semantic_buckets/tests` after changes.
