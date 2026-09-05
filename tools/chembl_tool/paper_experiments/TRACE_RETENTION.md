# Paper trace and artifact retention

Updated: 2026-09-05.

Retain evidence needed to reproduce the paper result families, not every smoke,
retry, or abandoned method branch. The canonical allowlist of result roots is
`current_conditioned_results.json`; do not duplicate that path list here.
Current retrieval inputs are independently locked by
`artifacts/chembl_tool/starling/current_records/manifest.json` and
`current_starling_retrieval.json`.

## Retain

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
