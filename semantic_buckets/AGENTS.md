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
The approved AMES endpoint-family child-only successor is an exception to the
repeat approval gate: retain and hash-verify its rendered review manifest, then
advance directly to Pass 2. Existing buckets are locked initial anchors only;
subsequent windows use scores from the preceding child-chain window.
When Pass 1 is pinned to one prepared endpoint, use that endpoint's frozen
`max_inflight` limit; the current AMES override therefore uses 128 on
`dgx027:50002`.

For the AMES V6-card successor, Pass 2 uses 12-candidate task-level wavefronts
with frozen widths L2=1, L3=1, L4=4, and L5=1. Every chain begins with an
unanchored seed. Each later batch receives the aggregated score of the final
ranked candidate from every batch in the preceding wave. Levels progress
independently and a new wave may begin as soon as its prior wave has aggregates.

For selected DILI and Carcinogens bucket-only releases, build the L2-L7 valid/test
Morgan-top-100 world from the Gold-v1 addon-v2 caches and the selected release
record map. Use V6 bucket cards with enriched canonical records in both passes.
Pass 1 has one high-reasoning Flash request per independent 12-bucket batch.
Pass-1 concurrency is launcher-local. A resumed run may set
`--max-inflight-per-endpoint` without changing its frozen prompts or prepared
profile, but never overlap two writers on the same request ledger; the runtime
override and its history must be recorded in the run manifest.
After its review gate, Pass 2 uses three OpenRouter GPT-6 Luna Flex calls at high
reasoning and averages all three valid responses. The two largest levels per task
use four 12-candidate chains; other levels use one. In a four-chain wave, persist
the first two completed chains and use their final two scored buckets each as
the next wave's four anchors. The remaining chains must finish and retain their
receipts before candidate rankings publish. Keep weights separate from the
selected semantic release and do not change CURRENT.

The DILI/Carcinogens Pass-2 successor reuses frozen Pass-1 artifacts under a new
run identity. Its user-approved exception uses OpenRouter GPT-6 Luna at medium
reasoning on the standard tier (no Flex), with three calls per scoring request.
Choose each level's chain width from 1, 2, 4, or 8 so its bucket count per chain
is closest to the largest level's count divided by eight. Preserve the existing
two-completed-chain anchor release, then require all chains before publication.
The approved V6 prompt representation does not require a repeat prompt-review
gate for these two tasks; retain rendered review manifests and hashes for audit.
High/Flex responses remain audit evidence in the predecessor runs.

Every Pass-2 request launches exactly six replicas: two OpenRouter
`openai/gpt-6-luna` Flex calls at medium reasoning, two OpenRouter
`deepseek/deepseek-v4-flash-0731` calls forced to Together at high reasoning,
and two local high-reasoning calls to `dgx027:50002`. Average the first three
schema-valid responses, but let all six finish and preserve all six receipts.
Use `OPEN_ROUTER_KEY_TWO` for both remote models, `DEEPSEEK_API_KEY=EMPTY`
locally, no fallbacks, no Alibaba, no direct OpenAI credential, and no spend
ledger. Hash-pin the fixed provider profile and preserve actual routes in every
replica receipt.

Preserve superseded responses for audit, but never reuse scores produced from a
different bucket representation in a successor run. Candidate weights from an
unreviewed semantic generation remain unreviewed and cannot update an active
release or `CURRENT` pointer.

Run `pytest -q semantic_buckets/tests` after changes.
