# Paper trace and artifact retention

Updated: 2026-10-18.

Retain evidence needed to reproduce the paper result families, not every smoke,
retry, or abandoned method branch. The canonical allowlist of result roots is
`current_conditioned_results.json` together with its linked `cleanup_history` receipt;
do not duplicate those path lists here.
Current retrieval inputs are independently locked by
`artifacts/chembl_tool/starling/current_records/manifest.json` and
`current_starling_retrieval.json`.

## Offline archives after storage cleanup

The [storage completion record](RESULTS.md#storage-cleanup-2026-09-30) links the
runtime migration and build-cache cleanup receipts. Retain their shared archives,
manifests, checksums, provenance and removal journals. `reproduction/` contains
historical experiment dependencies; `recovery/` conservatively preserves source
material and unresolved versions. Recovery status does not establish current use
or permit deletion without resolving those preservation reasons.

The deleted local build cache was an acceleration layer. Future builds recreate
it from available published inputs; they cannot recover a missing historical
input merely from its old pathname. Restore exact historical hashes and companion
files from the archive using the [restoration guide](../../../outputs/paper/local_build_cache_audit_20260930/README.md#restore-retained-inputs).
Keep original paths/hashes in frozen run receipts. Prepared prompts, cached priors,
predictions and selected-model dependencies remain covered by the rules below.

## Retain

For `budgeted_evidence_protocol.indirect_scaffold_disjoint` and
`indirect_scaffold_disjoint_luna`, retain all prepared/selection/evaluation/output
records, retry receipts, frozen priors/references, run plans, provider configs,
scores and verification receipts. Luna's scaffold identity verification depends
on the exact DeepSeek selections and its complete verification receipt; the
DeepSeek candidate cache is also a documented preparation dependency.
Keep the single experiment-local `verify.py` and both source snapshots. The
DeepSeek snapshot contains the original auditor; Luna's snapshot pins the later
`--root`/audit-reuse extension. Do not mistake that deliberate change or finalized
run-plan metadata for inference source drift. Historical ZIP member paths may
include the original absolute checkout prefix; restore into a scratch checkout
with that prefix mapped to the checkout root, never over current files blindly.

The Luna figure variant is the latest combined scaffold comparison; the earlier
DeepSeek-only variant is historical. Both keep their inputs, per-panel receipts,
PDF/PNG/SVG files and hashes. Preserve original parent-disjoint paper figures.
Completed top-level console logs live under their original basenames inside each
experiment's `execution_history.zip`; structured launch/status/verification JSON
remains in place. Temporary controllers are archived, not maintained entrypoints.
The [maintenance receipt](receipts/indirect_scaffold_maintenance_20261018.json)
records member hashes and exact-path cleanup. No prediction or source dependency
is removed by this cleanup.

For `diagnostics.tdc_dili_prompt_20260924`, retain the pinned upstream v2 source
requests/retrievals, retained prompt and all 288 replay requests/responses, retry
and invocation receipts, scores, paired comparisons and maintenance receipt.
A/B variants are rejected diagnostics, not active prompt options; keep their
frozen system files and `reproduction_source.zip` for provenance. Do not remove
failed-attempt traces or overwrite the historical controls. The registry owns
local paths; the shared TDC handoff owns collaborator integration and scope.

For `budgeted_evidence_protocol.llm_vs_standard_baselines`, retain the complete
`comparison.json`, per-benchmark `all_scores.csv`, paired `ML_APPENDIX.md` and
full-precision `ml_appendix.csv`, together with `build.py` and both verification
receipts. Keep every referenced prediction and source-suite receipt. The appendix
has 198 ML cells across conditioned and two TDC training pools; its CSV is the
manuscript table source. Preserve raw scores for AUROC and validation-selected
thresholds for Macro-F1; rounded Markdown cells are presentation only. Existing
figures retain their separately documented scope. Retain each current figure's
`points.csv`, `metadata.json`, its referenced Ours result record, and
`baseline_bars_{mixed,horizontal}_receipt.json`: metadata owns group membership and the source
pointer; the shared Ours record owns supplied scores and task versions. Receipts preserve
source hashes, winners/ties and explicit missing points. The audited baseline
CSV does not contain Ours. Both final bar layouts share inputs and calculations;
retain their task/mean receipts and explicit backbone bindings. Keep the user-requested
`design_snapshots/20260925_backbone_pastel_mixed` frozen source and results.
Superseded designs and temporary preview code are retained in the verified
`design_snapshots/retired_designs_20260925.zip`, not as active entrypoints. See [plotting instructions](README.md#current-comparison-figures).

For `budgeted_evidence_protocol.retrieval_forest`, retain prepared features and donor
indices, source-cache bindings, all candidate forests/validation scores, selections
and final predictions. Keep `verify.py`, data/result audits, resource
samples, manifest and source ZIP; the execution-only matrix launcher is archived
inside that ZIP. Superseded RF and MiniMol summary scripts are retained in
`skin_indirect_level_mapped_20260925/superseded_reporting.zip`; the shared comparison
exporter owns current tables. The six TDC-only condition-policy symlinks reuse all-train models.

For `budgeted_evidence_protocol.minimol_indirect_assay`, retain both source suites,
the six-task comparison and all candidate checkpoints/validation histories,
including non-selected λ values needed to reproduce validation selection.
Keep source Parquet/import hashes, prepared targets/graphs, selected predictions,
source snapshots, verification and resource receipts. The four-task frozen-control
symlinks depend on `conditioned_minimol_val_selected`; preserve their targets.
Execution-only launch/preflight scripts may live in the source ZIP at their
original repository paths. Keep local result auditors/renderers and the maintenance
receipt; duplicate console dumps of structured JSON audits are disposable.
The `llm_vs_standard_baselines` table includes all six encoder comparisons;
its current figures group ML variants as declared in their metadata.

For `budgeted_evidence_protocol.tdc_minimol_val_selected`, retain the manifest,
per-cell commands, train/valid/test input bindings, embedding provenance and caches,
five-member checkpoints, validation histories, selected epochs/thresholds and
valid/test predictions. The current all-settings Markdown/CSV/JSON and its
experiment-local `export_comparison.py` share this root. Retain referenced older
KNN and LLM predictions plus their verification receipts: reused inputs are
dependencies, not disposable duplicates. Earlier CV heads remain explicitly
historical; current scores are never replaced by per-task test-score winners.

For `budgeted_evidence_protocol.tdc_minimol_indirect_assay`, retain both main
training pools, all 240 candidate checkpoints/validation histories and 36 ensemble
cells. Keep prepared targets/graphs, input hashes, auxiliary exclusion audit,
selection/threshold receipts and resource samples. The 12 frozen-control symlinks
depend on `tdc_minimol_val_selected`; preserve their targets. Launch/preflight
receipts live in `training_source_snapshot.zip`; local `verify.py [data]` and
`summarize.py` audit/render the completed suite. Its maintenance receipt binds
the source archive and confirms existing conditioned/TDC results are unchanged.

For `budgeted_evidence_protocol.tdc_knn3_baselines`, retain the referenced original
Morgan KNN neighbors and TDC train rows, prepared selections/messages, independent
Direct/Indirect equivalence checks, None reuse receipt, final outputs and command.
Preserve the original ML/None references; do not copy their artifacts into this run.

For `budgeted_evidence_protocol.tdc_scaffold_baselines`, retain the manifest-bound
TDC splits/indices and acquisition receipts, cached priors, selected records,
actual prepared prompts/predictions, ML outputs/checkpoints and full verification.
Keep the four-output Ames recovery receipt and its quarantined unreadable files.
For `tdc_starling_mixed_baselines`, retain the mixed input manifest, exclusion and
CV audits, frozen build/command receipts, embedding provenance, fitted models,
predictions, comparison CSV and verification. For `record_rendering_deduplication`,
retain the audit script/receipt and before/after examples; this is offline
verification, not a new prediction suite. These entries are all under
`budgeted_evidence_protocol`; do not maintain a second result-root list here.
Generated Python bytecode and a byte-identical completed builder's partial
manifest are disposable once the final manifest is verified and the builder
resumes from that final manifest.

For `budgeted_evidence_protocol.aligned_record_scaling` and
`budgeted_evidence_protocol.aligned_record_scaling_test` and the independent
`aligned_record_scaling_test_direct100_fill` follow-up, retain the Direct/Indirect
manifests, prepared prompts, selected records, actual outputs, evaluation rows,
per-cell summaries, the validated Direct-budget selection and final verification.
Keep cached-prior provenance/snapshots and referenced source/ML lineage. Keep all
four figure exports, portable CSV/metadata, receipts and the local audit replay
script. Audit scripts under an experiment root are provenance, not maintained
runner entrypoints. Retain both filesystem-outage recovery/quarantine receipts
and preparation-upgrade originals: these explain preserved checkpoints and code
hash boundaries. Do not move linked files merely to shorten a directory listing.
Temporary `/tmp` editing scripts and symlink test overlays may be removed after
verifying no live process depends on them; raw predictions and cached priors are
not disposable scratch files.

For `budgeted_evidence_protocol.baseline_pilot`, retain verified per-query
prepared prompts, selections, tools, outputs and evaluation rows; frozen runtime,
completion verification and paired direct-set checks; and the registered figures,
score CSV and source-hash/condition comparison receipt. Keep the five baseline
prediction files and lineage audit referenced by that receipt. Earlier blind
outputs are separate provenance and must not count as visible predictions.
Retain the same inputs/traces for the completed `baseline_no_threshold` valid/test
suites, including launch, preflight, final progress, full verification and verifier
provenance. Keep both current curve figures (PNG/PDF/SVG), CSV scores and figure
receipts; historical thresholded figures must stay in their own roots.
Retain registered `citations/` per-query role/parent mappings, summary TSV and
manifest alongside the completed suites. These are offline final-answer citation
statistics; they do not imply a upstream-prompt rerun or reasoning-mention analysis. The old `baseline_test` 0.3
run is stopped; retain its stop receipt and partial outputs separately. Neither
the stopped test nor the completed 0.3 validation supplies predictions to the
fresh no-threshold runs.
Retain the registered preparation-upgrade parity/test/migration receipts and
prior manifests, plus the pre-review code snapshots and artifact checks linked by
`receipts/baseline_maintenance_20260917.json`. Node-local candidate-cache entries are disposable acceleration
artifacts; prepared prompts and recorded request hashes remain authoritative.

For the paused `budgeted_evidence_protocol`, retain the current source input scope,
`candidate_annotations_current` SQLite checkpoint, manifest, progress and
`pause.json`, plus the frozen donor checkpoint and exact migration receipt.
Retain source/selected-context audits and their frozen input/example references,
baseline lineage audit, and the Skin 2026-09-17 identity/source/index/export
receipts. The earlier extraction rounds are provenance dependencies, not extra
current releases; do not delete them merely because the job is paused.
`receipts/joint_pause_review_20260917.json` pins the stopped state and checkpoint.
Completed preparation and valid JSON are not completed or scientifically validated
prediction results. No large artifact deletion is authorized by this cleanup.

For `replicate_suites.fixed25_no_threshold_progressive_full_flat_20260915`, retain
the predecessor completion barrier, retrieval census and native card-selection
parity checks, per-level molecule shortfalls, and explicit unavailable-tool
receipts, alongside the frozen contracts, prepared inputs and actual traces.

For `replicate_suites.expanded_budget_progressive_full_flat_20260915`, retain the
budget census, frozen run contract/plan/runtime hashes, all-query card-ID checks,
prepared inputs, actual requests/results and matched-input verification. Keep
HAProxy/admission configuration and endpoint preflight receipts. The coordinator
must finish progressive verification before starting matched full-flat; live
execution status is separate from completed results.

Both 2026-09-15 budget suites are paused by user as of 2026-09-16 01:11 UTC.
Retain successful partial full-flat checkpoints, fixed-25 progressive checkpoints,
prepared inputs, and the fixed-25 pause receipt/checkpoint hash inventory for resume. Also
retain the registered completed-pair comparison, its prediction/input verification
and pause receipt; do not classify incomplete full-flat runs as completed results.

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

For the completed L3+ parent-disjoint matched full-flat suite, retain its coordinator,
run plan, frozen runtimes, prepared inputs and all successful checkpoints. Keep the
admission/rebalance receipts and failed-attempt traces: local admission precedes the
HTTP deadline and retry races share the same limits. Keep all six valid/test
verification receipts and the registered figures with hashed TSVs and render
commands. Temporary rendering scripts are redundant once receipt-based reproduction
is verified; frozen runtime files and historical comparison dependencies are not.

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
PARCC source is a provenance dependency; partial or changed-input outputs cannot
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

For record-diversity comparisons, retain registered old/new selected lists,
source-bound annotation snapshots, feature manifests and reviewed overlays,
prepared/request/output traces, reuse receipts and the canonical comparison
figure. A newer selector with no inference is not a replacement for the completed
arm. The six-task suite's `molecule_local_replay` registry entry pins both stages.
Suite-local scripts and runtime snapshots are provenance, not maintained CLI
entrypoints. The unused standalone observation prototype was retired from active
code with a source/test archive; see
`receipts/record_diversity_maintenance_20260916.json`.

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

## BBB indirect test development (2026-09-24)

Retain all six local arms, prepared inputs, source selections, cached priors, actual outputs, retries and verification under the registered `tdc_bbb_indirect_test_iteration_20260924` root. Its original preparation scripts are hash-bound historical provenance, not the maintained entrypoint. Share only the compact frozen input bundle, checksum/source manifest, per-query final labels, comparison and output-hash receipts in `artifacts/chembl_tool/bbb_indirect_20260924/`; raw reasoning and API traces remain local. Maintained preparation and usage: [BBB guide](../tasks/bbb_martins/INDIRECT_ABLATION.md).

## Matched seed-42 Skin/Carcinogens naive baselines (2026-09-24)

Retain `outputs/paper/tdc_seed42_neighbor_fill_20260924/`: source-bound benchmark
manifest/indices, shared prior analyses, prepared requests/evaluation labels,
model outputs, predictions, provider snapshots and retry-race receipts. Four
OpenRouter suites use the existing family entrypoint. Registry
`budgeted_evidence_protocol.tdc_seed42_neighbor_fill` owns completion status.
Original seed-1 traces remain historical; they cannot fill missing seed-42 cells.
