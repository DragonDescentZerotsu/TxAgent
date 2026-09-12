# Paper trace and artifact retention

Updated: 2026-09-08.

Retain evidence needed to reproduce the paper result families, not every smoke,
retry, or abandoned method branch. The canonical allowlist of result roots is
`current_conditioned_results.json`; do not duplicate that path list here.
Current retrieval inputs are independently locked by
`artifacts/chembl_tool/starling/current_records/manifest.json` and
`current_starling_retrieval.json`.

## Retain

For the Ames scaffold valid/test rename, retain the registered
`starling_conditioned_ames_scaffold_test_swapped_4_2_v1` model roots, their
original reuse sources, and `outputs/paper/ames_scaffold_split_swap_20260908/`.
The latter contains the original split/index metadata snapshots referenced by
the canonical rename receipt. Renaming a cohort does not create a new replicate;
the figures must disclose that current Ames test was previously validation.

### Benchmark and source provenance

Retain the complete `data/conditioned_benchmark/` contract, including manifests,
migration/random receipts, detailed molecule-condition labels, split summaries,
condition distributions, and task-specific accepted/rejected/conflict/review
ledgers. Also retain the packaged current Stage-03 parts and the current
overlays/catalogs/indices enumerated by the retrieval manifests.

Superseded record and family versions are not active alternatives. Their useful
history belongs in compact receipts and Git history.

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

For the registered DILI retrieval_review_v1 targeted replay, retain its original
no_prior_grounded_sim0 source run, frozen single/None cache, all-query surface
audit, tool-freeze receipt, per-level checkpoint-reuse sidecars and paired metrics.
Unchanged outputs and requests may be immutable hardlinks to that source run.
The replacement combines fresh affected outputs with verified unchanged outputs;
it is not an additional independent replicate.

For the registered DILI retrieval_review_v2 targeted replay, retain the v1
source run, the v2 four-record source release, the 804-row selected-surface
audit, complete prompt/tool equivalence receipts, and checkpoint-reuse sidecars.
Only 94 valid level outputs are freshly inferred; the other 11,162 are reused.
The test executor explicitly forbids new model calls. Keep the shared-runner
controller, run plan and final paired metrics with this registered result root.

For `dili_l1_condition_priority_v1`, retain the pinned source-only card condition
map and its source-ID lineage, contract/command, all 402 request comparisons,
exact-input reuse receipt, L1 traces and paired metrics. Its reviewed-source
baseline and frozen single/None/tool inputs remain provenance dependencies.
Only L1 is executed; the full L1–L7 prompt plan is retained.
The authorized OpenRouter replacement runs both valid/test with all 804 L1
predictions fresh. Retain its version-pinned provider configuration, preflight,
per-split commands, complete raw responses and input/paired-performance audits.
Its PARCC comparison retains frozen priors/tools but changes hosting; it is not
an isolated retrieval-effect estimate or another complete L1–L7 replicate.
Retain the subsequent `dili_l1_condition_priority_parcc_endpoint_v1` comparison:
all 804 prepared files and complete nominal requests match the completed OpenRouter
run, with fresh PARCC L1 outputs for both splits. Preserve paired request hashes,
raw responses and endpoint-disagreement metrics. Both DILI endpoint roots may be
stored on local NVMe behind their original project paths; the completed OpenRouter
storage-migration receipt verifies every retained file. Keep these physical targets
while the registry or comparison depends on them.

Retain `dili_l1_morgan_neighbors_condition_cards_v1`: source-only condition map dependency,
frozen commands/contract, original-neighbor equality checks, frozen common tools, all
valid/test fresh L1 requests/responses and changed/unchanged-input paired metrics. It
changes only within-molecule card selection. Keep its local NVMe target behind
`outputs/paper/starling_conditioned_dili_l1_condition_priority_v1/morgan_neighbors_condition_cards_v1`.

Retain `dili_l1_morgan_neighbors_exact_condition_cards_v1` and its local NVMe target
behind `outputs/paper/starling_conditioned_dili_l1_condition_priority_v1/morgan_neighbors_exact_condition_cards_v1`:
frozen plan/code hashes, exact-only contract, no-match original-card equality audit,
fresh L1 requests/responses, and paired comparisons with both original and previous
card-policy runs. It shares the preceding suite's bounded executor; retain that dependency.

Retain `dili_l1_query_identity_pilot_v1` at
`outputs/paper/starling_conditioned_dili_l1_condition_priority_v1/query_identity_pilot_v1`
and its local NVMe target: frozen balanced cohort and selection contract, identity-cache
joins, input/code hashes, both control/identity arms and their two fresh repeats,
paired metrics and semantic trace reviews, rejection receipt and frozen implementation.
The true-name method is rejected and audit-only. Preserve the exact-condition source run
while this pilot depends on its frozen inputs.

Retain `dili_l1_query_identity_exclusion_pilot_v1` at
`outputs/paper/starling_conditioned_dili_l1_condition_priority_v1/query_identity_exclusion_pilot_v1`
and its NVMe target: the three pre-reviewed cases, exclusion-only request checks,
two fresh repeats per control/exclusion arm, code/input hashes, endpoint receipts
and manual identity/subject-attribution review. It depends on frozen exact-only L1 inputs.

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
and RL remain separately archived/stopped work and require their own deliberate
cleanup decision.

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

The September cleaning cycle is closed. Keep only the final retrieval source,
portable archive parts and current catalog/scaffold/random indices; source inventory
and hashes are in `current_starling_retrieval.json`. Per-task final release receipts
and `source_changes.jsonl` are under `data/starling_data/<task>/retrieval_final/`.
Raw acquisition and frozen gold/direction/identity ledgers remain provenance.

Remove obsolete source/index/table backups and completed per-round Python/shell
launchers after final publication checks. Do not remove prediction, request, prepared
input or tool files merely because their round is old: later exact-input replays and
trace views may depend on them. Retained experiment roots and compact historical
registrations are indexed by `current_conditioned_results.json` and its history receipt.

Latest completed diagnostics are DILI v5 and Carcinogens R18. Preserve their original
inputs, metrics, figures and reuse receipts, plus the previous-round figure references.
Final source consolidation does not rerun the models or authorize automatic reuse
under a changed selected evidence surface. Both Valid/Test cohorts have been inspected.
