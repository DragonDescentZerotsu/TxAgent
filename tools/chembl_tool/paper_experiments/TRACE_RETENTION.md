# Paper trace and artifact retention

Updated: 2026-09-14.

Retain evidence needed to reproduce the paper result families, not every smoke,
retry, or abandoned method branch. The canonical allowlist of result roots is
`current_conditioned_results.json` together with its linked `cleanup_history` receipt;
do not duplicate those path lists here.
Current retrieval inputs are independently locked by
`artifacts/chembl_tool/starling/current_records/manifest.json` and
`current_starling_retrieval.json`.

## Retain

For `replicate_suites.l3_parent_disjoint_progressive_test_20260914`, retain the
retrieval diagnostic, mixed-policy prepared contexts, exact prefix-reuse receipts,
old/current tool and prompt runtime provenance, queue-handoff receipt, and actual
requests/results. Keep its strict reference trajectories, including Bioavailability/
Skin legacy replicate 03. Paused v5 test full-flat dependencies remain retained.

For the registered Skin identity-rebinding replay, retain the full-row source patch
ledger, original-document hashes, source snapshot publication, frozen legacy runtime,
480-row surface comparison, exact prefix-reuse receipts and actual requests/results.
The v5 hybrid and v6 quarantine runs remain its provenance dependencies; reused
outputs do not count as fresh replicates.

For registered OpenRouter Batch runs, retain `batch_jobs/` alongside actual
requests, structured results and model/usage provenance. Its remote IDs and
request custom IDs are required for recovery without duplicate paid submissions.
Preserve matched source tool/card hashes and fresh single/None dependencies;
completed upstream jobs alone do not establish validated query completion.
After a smoke batch is adopted by a completed formal run, remove its abandoned
local directory only after matching remote/custom IDs, exact requests and frozen
retrievals to the retained run. Keep the small handoff and cleanup receipts;
the smoke is not a separate replicate or a second billing item.

For the running L3+ parent-disjoint matched full-flat suite, retain its coordinator,
run plan, frozen runtimes, prepared inputs and all successful checkpoints. Keep the
admission/rebalance receipts and failed-attempt traces: local admission precedes the
HTTP deadline and retry races share the same limits. Read `execution_status.json`
for live progress; preparation completion is not inference completion.

For `replicate_suites.dili_v5_trace_source_replay_20260913`, retain the source
decision ledger, publication/restore checks, all-804-row selected-input audit,
complete valid/test trajectories and actual-request verification. Its reference
is `progressive_evidence_revision.v5`, not the historical v2-prompt DILI diagnostic.
Retain the original v5 suite's frozen pre-repair DILI catalog/index dependencies
for its queued matched full-flat stages. Reused outputs are not new replicates.

For `replicate_suites.ames_trace_source_replay_20260913`, retain the complete new
valid/test results, all-query selected-input audit, prefix-reuse receipts, code
equivalence receipt and changed-decision comparison. The original v5 progressive
roots remain provenance dependencies; reused levels do not count as fresh replicates.

Retain the registered Bioavailability/Skin legacy restoration package, including
its pinned runtime, prepared historical tool/prior inputs, source hashes,
preflight and commands, plus its linked input-comparison report. No predictions
are implied by configuration restoration. Retain the ongoing v5 suite's frozen
runtime and pre-repair Ames benchmark/index/catalog dependencies until all
matched stages and lineage checks finish. The Ames trace-source repair ledger,
raw/canonical sample, primary-paper verification and publication/validation
receipts are source provenance, separate from the preceding read-only diagnostic.

For the Ames scaffold valid/test rename, retain the registered
`starling_conditioned_ames_scaffold_test_swapped_4_2_v1` model roots, their
original reuse sources, and `outputs/paper/ames_scaffold_split_swap_20260908/`.
The latter contains the original split/index metadata snapshots referenced by
the canonical rename receipt. Renaming a cohort does not create a new replicate;
the figures must disclose that current Ames test was previously validation.

For the registered functional-group tree diagnostic, retain its paired prepared
inputs, tool receipts, requests/responses, manual structural review, auxiliary fresh
single-prior pair and service verification receipts, including the registered code-review and
43-molecule text-equivalence receipts. Retain the non-executable execution snapshots
in place of the retired diagnostic scripts. Its original trace roots remain
provenance dependencies. The diagnostic is not a paper benchmark replicate or an
alternate maintained launcher.

### Benchmark and source provenance

Retain the complete `data/conditioned_benchmark/` contract, including manifests,
migration/random receipts, detailed molecule-condition labels, split summaries,
condition distributions, and task-specific accepted/rejected/conflict/review
ledgers. Also retain the packaged current Stage-03 parts and the current
overlays/catalogs/indices enumerated by the retrieval manifests.

Superseded record and family versions are not active alternatives. Their useful
history belongs in compact receipts and Git history.
The 2026-09-14 maintenance removed the unreferenced Skin v2–v4 overlays,
catalogs and scaffold/random indices (12 directories, 1.325 GiB logical size).
Exact paths, file hashes and dependency checks are in
`receipts/source_version_cleanup_20260914.json`. Skin v5/v6 remain frozen run
dependencies; v7 is current. Incremental Ames/DILI review ledgers remain source
provenance and must not be treated as disposable duplicate datasets.

### Matched baselines

Retain the registered scaffold-test baseline predictions, metrics, embedding and
input audits, execution receipt, and train-only CV summaries. BBB and Skin test
results reuse the explicitly registered valid-run ensemble checkpoints; those
checkpoint/CV dependencies remain retained. Bioavailability has a fresh train-CV
and final ensemble after the train-row correction. Every KNN reference is train.

Retain the registered Ames scaffold-valid 4/2 progressive and matched full-flat
roots, fresh single/None traces, shared prepared evidence/tools, launch receipt,
and the current five-baseline bundle under `outputs/baselines/ames_scaffold_valid_v2`.
Its two condition-first KNN reruns fill missing slots from unrestricted train;
retain the fallback audit and all 274 predictions. Preserve the v1 dependency
root `outputs/baselines/ames_scaffold_valid_v1`: v2 links its unchanged head and
unrestricted-KNN methods, and its old condition-KNN exclusion ledger documents
the historical 252-row cohort. The bounded startup smoke is diagnostic only.

For the Pro valid nitrosamine structure repair, retain the registered replacement
root and its original Pro source root, including the shared single/None cache.
Preserve the seven-record correction receipt, all-query input comparison, prompt
equivalence check and prefix-reuse receipts. This replay is a correction to one
run per organization, not an additional independent replicate.

For the registered Ames test identity repair, retain both the replacement suite
and its original six-run source as a retained provenance dependency after promotion.
Preserve the source-exclusion receipt, all-query selected-surface comparison and
per-run prefix-reuse receipts; copied results depend on the corresponding original
repetition, not on a pooled or newly selected prediction.

For DILI v1–v5 targeted replays, retain original predictions and requests, frozen
single/None/tool caches, all-query selected-surface audits, prompt-equivalence
receipts and progressive checkpoint-reuse sidecars. Later runs depend on the
verified unchanged outputs of earlier runs; they are not independent replicates.
The source versions and changed-output counts belong to the linked historical
receipts. Only the final source/index materialization remains active.

DILI L1 condition/card-selection, hosting comparisons and query-identity pilots
are stopped diagnostics indexed by `cleanup_history`. Retain their frozen inputs,
raw requests/responses, metrics, code hashes, rejection/selection receipts and
any local NVMe targets still backing retained traces. Removed per-round controller
and launcher scripts are recoverable from Git where committed; they are not
required live dependencies or instructions to restart these pilots.

For registered Carcinogens source-cleanup diagnostics, retain the hash-bound
review overlay, source proposals and adjudication overrides, unresolved-concern
ledger, all-query input comparison, replacement-card audit and prefix-reuse
receipts. Round 3 also records opt-in endpoint diversity. Its completed round-2
Hosted source is a provenance dependency; partial or changed-input outputs cannot
be substituted for a matched full-flat result or an independent replicate.

### Paper result cells

For every applicable paper cell, retain the latest hash-compatible complete
artifact for:

- ChEMBL and Starling retrieval;
- one-shot cumulative full-flat reasoning;
- append-only progressive reasoning;
- full-mechanism reasoning;
- scaffold and random split;
- identity-blind and deployment-visible-prefetched visibility.

If a current cell is incomplete, retain one last-complete historical reference
and mark it as such in the registry. Different directory names do not justify
duplicate retries. Registered exact-contract replicates used to report run
variability (SD or observed min-max) are the exception.

Retain the completed matched full-flat test outputs, input-hash manifest,
preflight/launch receipts and comparison figure together with the source progressive root.
Retain all registered replicate 1/2/3 roots used for run-variability estimates,
plus the repeat suite plan, status and launch receipt. Prepared request files
alone do not count as completed outputs.
The latest registered Flash/Pro comparison has a compact Git-tracked export:
PNG/SVG, aggregate metric/reference TSVs, summary and figure receipt. Large raw
run/trace directories remain local. Completed startup snapshots, duplicate command
JSON files already covered by launch receipts, and routine successful launcher
stdout can be removed after final verification; keep source/endpoint preflight receipts.
For registered cross-model matched suites, also retain the refreshed single/None
cache, its completion hashes, the price-routing configuration, endpoint/input
preflights, and both source/result prepared hashes. Never pool different models
as statistical replicates.
The completed three-run aggregate is the primary test comparison. Its receipt
retains per-run scores, the exact plotting command and completeness/lineage
checks. Original single-run figures remain explicitly labeled replicate-1 views
for the associated trace audit; they are not additional replicates.

The current progressive status is:

- All three registered scaffold-test 4/2 runs per method are complete with zero failed outputs.
  Retain its query/pair tool audits, dependency and inference recovery receipts,
  and compressed pre-repair/first-pass failure traces. Prior valid/random runs have
  not yet been checked for the repaired shared MolGpKa graph-state race;
  source-lineage status below does not certify corrected tool equivalence.

- Retain the registered scaffold-test L2→L3 audit, its nine case reports,
  BBB valid comparison cases, paired statistics, integrity checks, and source
  hash receipt alongside the referenced original traces.

- BBB scaffold-valid strict-voter-L1 v6 is current; random needs replay.
- Bioavailability scaffold-valid is current via the split-scoped nitrendipine
  zero-change receipt; random remains unaudited. The receipt does not authorize
  baseline reuse.
- Skin scaffold-valid source-purity v5 2/1, 4/2, and 8/4 are last-complete references
  pending targeted replay of 1/2/3 queries after the final MDAM L2 rebuild;
  random needs full replay. Current MiniMol/Morgan baseline artifacts use the
  unchanged 239-row cohort.
- Historical broad-L1, connectivity-confounded, L1-only diagnostic, and
  superseded BBB roots are not current result cells.

For the record-card ablation, retain only roots registered under
`record_card_budget_ablation`: the registered current task cells, BBB 8/4, both
Bioavailability exact-contract 8/4 full-curve replicates, and the three Skin
last-complete roots needed as targeted-replay sources. Older broad-L1 Skin roots
remain excluded.

Until current-conditioned replacements exist, retain the single historical
blind and visible source/reasoning matrices and the last complete one-shot
family curve named in the registry. Their different lineages do not authorize
a matched visibility comparison.

Retain `execution_status.json` and bounded `failed_attempts/` histories for
unattended progressive/full-flat retries. A later valid output must not erase the
failed provider/validation evidence; `needs_attention` is not a complete result.
Retain per-level `retry_races/` receipts for six-way retries: completed candidate
responses, validation errors, winner identity and cancelled-attempt statuses.
Preserve the explicit distinction between client cancellation and verified
server abort. The suite's one bounded endpoint race smoke receipt is retained
as execution provenance, not counted as an experiment replicate.

## Minimum complete trace bundle

For each retained batch or progressive level, keep:

- experiment and batch manifests;
- input, retrieval, family, and index hashes plus visibility contract;
- per-query retrieval payload or prepared state;
- exact LLM request and raw response/reasoning;
- parsed output and validation/provider-attempt history;
- predictions and metrics;
- reuse/carry-forward receipt when no fresh call was made;
- registered analysis TSV/JSON and final figure.

Keep logs only when they explain an incomplete or failed canonical run. Routine
stdout/stderr is disposable after a replacement is complete and audited.

## Remove after verification

- smoke, prompt-debug, aborted, timeout-only, and one-query directories;
- superseded retries already represented by a retained batch;
- duplicate figures from one-off plotters;
- old source/index copies without unique provenance;
- intermediate prompt audits after retaining their aggregate ledger and small
  representative examples;
- no-go method code and artifacts once a compact receipt/Git history preserves
  the conclusion;
- a stale random trace after its current replay or change audit is complete.

Never delete current benchmark/source ledgers, migration receipts, packaged
current records, or the only complete trace for a retained paper cell. Router
and RL packages, dedicated tests and obsolete runbooks were removed at the user's
request on 2026-09-14. `receipts/retired_research_code_20260914.json` pins the
recoverable Git commit and file inventory. Their historical outputs/checkpoints
remain preserved; they are not current paper results or maintained entrypoints.

## Safe cleanup

1. Resolve exact retained paths from `current_conditioned_results.json`.
2. Verify predictions, metrics, zero-failure gates, and semantic hashes.
3. Produce an explicit deletion list; never use a broad wildcard on
   `outputs/paper`, `data`, or the repository root.
4. Obtain approval before permanent deletion of material artifacts not already
   explicitly rejected.
5. Remove only approved paths, scan for dangling references, run focused tests,
   and revalidate the registry.

## Viewer contract

The trace viewer supports only the current append-only progressive pipeline.
The launcher resolves scaffold task roots and current/stale status from
`current_conditioned_results.json`; it must not infer lineage from directory
names. The viewer aligns Prior and L1...LN predictions by query index, shows
correctness transitions and prediction flips, and loads each level's
`prepared.json`, `request.json`, and `output.json` on demand. It also shows
whether a level was called, reused, or carried forward. The right-side level
view is intentionally limited to vertically stacked, collapsible Prompt,
Reasoning, and Output sections; Prior is labeled separately because it is not
a model call. Structured fields use a title-above-value layout. Serialized JSON
in the user message is parsed into semantic prompt sections, analogs, and
evidence cards; raw message JSON remains available only as a nested audit view.

```bash
bash tools/trace_viewer/start_viewer.sh 8776
```
## DILI / Carcinogens dependencies

The bounded reviews are frozen after publication. Keep only the final retrieval source,
portable archive parts and current catalog/scaffold/random indices; source inventory
and hashes are in `current_starling_retrieval.json`. Per-task final release receipts
and `source_changes.jsonl` are under `data/starling_data/<task>/retrieval_final/`.
Raw acquisition and frozen gold/direction/identity ledgers remain provenance.

Remove obsolete source/index/table backups and completed per-round Python/shell
launchers after final publication checks. Do not remove prediction, request, prepared
input or tool files merely because their round is old: later exact-input replays and
trace views may depend on them. Retained experiment roots and compact historical
registrations are indexed by `current_conditioned_results.json` and its history receipt.

Historical DILI retrieval-review v5 and Carcinogens R18 remain retained diagnostics;
the newer DILI prompt-v5 replay has its own registry entry above. Preserve original
inputs, metrics, figures and reuse receipts, plus previous-round figure references.
Final source consolidation does not rerun the models or authorize automatic reuse
under a changed selected evidence surface. Both Valid/Test cohorts have been inspected.

The completed `l3_parent_disjoint_progressive_valid_20260914` suite retains its
experiment-local execution snapshots, valid source preflights, strict legacy
reference repair for Skin #193/#206, unchanged-prefix receipts and fresh actual
request traces. It completed after the registered test suite; both policy arms and
their baseline-verified comparison figures remain retained.
