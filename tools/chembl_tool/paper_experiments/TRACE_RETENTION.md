# Paper trace and artifact retention

Updated: 2026-09-02.

The repository retains traces needed to reproduce the paper result families,
not every smoke, retry, or abandoned method-development branch. The canonical
allowlist is generated from
`tools/chembl_tool/paper_experiments/current_conditioned_results.json`.

## What must be retained

### Benchmark and source provenance

Retain the full active benchmark, not only minimal train/valid/test rows:

```text
data/conditioned_benchmark/
data/starling_data/<task>/<current canonical source>/
```

Required benchmark artifacts are enumerated in
`tools/chembl_tool/common/starling/CONDITIONED_BENCHMARK.md`: manifests,
migration/random receipts, detailed molecule-condition labels, split summaries,
condition distributions, and task-specific accepted/rejected/conflict/review
ledgers.

### Paper-facing result families

Retain the latest hash-compatible complete artifact for each applicable cell of:

- ChEMBL and Starling retrieval;
- one-shot cumulative full-flat reasoning;
- append-only progressive reasoning;
- full-mechanism reasoning;
- scaffold and random split;
- identity-blind and deployment-visible-prefetched visibility.

If a current cell is incomplete, retain exactly one last-complete reference and
mark its historical lineage in the registry. Do not retain multiple equivalent
retries merely because they have different directory names; intentional
exact-contract replicates used for registered variation ranges are exempt.

### Registered progressive roots

The registry currently points to:

```text
outputs/paper/starling_conditioned_assay_progressive_visible_bbb_source_purity_v6_card_budget_4_2_v1/
outputs/paper/starling_conditioned_assay_progressive_visible_v10_bio_legacy_gold_vote_pure_v1/
outputs/paper/starling_conditioned_assay_progressive_visible_card_budget_4_2_current_control_v1/
outputs/paper/starling_conditioned_assay_progressive_random_quality_v2_parent_disjoint/
```

BBB scaffold-valid is current. BBB random is retained as a pre-v6 prediction
reference after the v6 random index rebuild and requires LLM replay. Skin
strict-voter-L1 v2 scaffold-valid 4/2 is current; its random predictions remain
a historical broad-L1 reference and require replay. Bioavailability scaffold-valid is current via
`receipts/bioavailability_scaffold_valid_nitrendipine_fix_zero_change.json`:
all 262 query-level selected retrieval surfaces were unchanged across L1-L6, so
the retained traces and agent predictions remain canonical without new model
calls. Bioavailability random remains unaudited and must be retained as a
pre-fix reference until a separate change audit or replay is complete.

### Current record-card budget ablation roots

The current scaffold-valid comparison retains only the 2/1, 4/2, and 8/4 task
roots registered under `record_card_budget_ablation` in
`current_conditioned_results.json`. Retain the shared three-task 2/1 root, the
BBB strict-voter-L1 v6 4/2 and 8/4 roots, the Skin strict-voter-L1 v2 4/2 root,
and both registered full-curve Bioavailability 8/4 exact-contract replicates.
Bioavailability 4/2 reuses the current scaffold root above. Skin has no current
8/4 cell; its historical broad-L1 8/4 trace does not fill that gap.

```text
outputs/paper/starling_conditioned_assay_progressive_visible_card_budget_2_1_v1/
outputs/paper/starling_conditioned_assay_progressive_visible_bbb_source_purity_v6_card_budget_8_4_v1/
outputs/paper/starling_conditioned_assay_progressive_visible_card_budget_8_4_tool_prefetch_fixed_v1/
outputs/paper/starling_conditioned_assay_progressive_visible_bio_card_budget_8_4_exact_replay_v1/
```

The rejected Bioavailability L1-only 0.7148 diagnostic changed the prompt
contract and is not a retained replicate. The connectivity-confounded
Bioavailability 8/4 task root and superseded BBB roots listed under
`excluded_invalid_root`, `bbb_excluded_invalid_roots`, and
`bbb_historical_v5_roots` are not retained result cells. After their receipts
and aggregate diagnostics are preserved, those task-level traces are cleanup
candidates rather than additional canonical lineages.

### Last-complete retained references

Until current-conditioned replacements exist, retain:

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_glm_5_2_nvfp4/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_glm_5_2_nvfp4_visible_parent_disjoint/
outputs/paper/starling_conditioned_assay_family_curve_v1/scaffold_valid_epyc_deepseek_v4_flash_0731/
```

The blind and visible matrix roots above use different historical benchmark
lineages. Retention does not authorize a matched comparison.

## Minimum complete trace bundle

For every retained batch or progressive level, keep:

- experiment and batch manifests;
- input/retrieval/family/index hashes and visibility contract;
- per-query retrieval payload or prepared state;
- exact LLM request messages and raw response/reasoning;
- parsed structured output and validation/provider-attempt history;
- predictions and metrics;
- reuse/carry-forward receipt when no fresh call was made;
- analysis TSV/JSON and final paper figure generated from the batch.

Logs are retained only when they are needed to explain an incomplete or failed
canonical run. Once a replacement is complete and audited, routine stdout/stderr
logs are disposable.

## What should be removed after verification

- smoke, prompt-debug, aborted, timeout-only, and one-query directories;
- superseded retries whose successful outputs are already represented in the
  retained batch;
- duplicate figures generated by one-off plotting scripts;
- old source/index copies with no unique provenance receipt;
- intermediate prompt audits after their aggregate ledger and representative
  examples are preserved;
- isolated router/RL research only after a separate deliberate removal decision;
- other no-go method-development branches outside the retained paper scope;
- stale Bioavailability random pre-fix traces after its separate change audit or
  replay is complete; retain the scaffold trace referenced by the zero-change
  receipt.

Never delete current benchmark gold/source ledgers, migration receipts, or the
only complete trace for a retained paper cell.

## Safe cleanup procedure

1. Resolve the exact paths from `current_conditioned_results.json`.
2. Verify each retained root has its expected predictions, metrics, zero-failure
   completeness gate, and matching semantic hashes.
3. Generate an explicit candidate-deletion list. Do not use a broad wildcard on
   `outputs/paper`, `data`, or the repository root.
4. Obtain approval for permanent deletion of tracked code or material artifacts.
5. Remove only approved paths.
6. Scan code/docs for dangling imports and references.
7. Run the focused test suite and revalidate the artifact registry.

## Viewer contract

The trace viewer reads the retained manifests and predictions; it must not infer
lineage from directory names. It displays whether a query/level was freshly
called, reused, or carried forward, plus the exact visibility and retrieval
policies.

```bash
bash tools/trace_viewer/start_viewer.sh 8776
```
