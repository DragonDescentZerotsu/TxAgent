# Current paper results and artifact status

Updated: 2026-09-03.

This is the human-readable companion to `current_conditioned_results.json`,
the machine-readable authority for result roots and freshness. Current
records-to-index reconstruction is documented separately in
`CURRENT_STARLING_RETRIEVAL.md`; the retrieval protocol is in
`ASSAY_LEVEL_RETRIEVAL.md`.

## Evaluation data and freshness

All active experiments use:

```text
data/conditioned_benchmark/<Task>/{scaffold,random}/
```

| Task | Scaffold train / valid / test | Random train / valid / test |
|---|---:|---:|
| BBB_Martins | 3,053 / 397 / 393 | 3,075 / 384 / 384 |
| Bioavailability_Ma | 1,956 / 262 / 269 | 1,989 / 249 / 249 |
| ClinTox | 1,144 / 142 / 142 | 1,142 / 143 / 143 |
| Skin_Reaction | 1,997 / 246 / 248 | 1,993 / 249 / 249 |

Scaffold and random contain the same molecule-condition labels. Random is
parent-grouped, not molecule-row independent. A result is current only when
benchmark/split, heldout index, family catalog, source/organization, model,
prompt, visibility, and identity policy all match. Directory names and sample
counts are insufficient.

## Retained paper families

| Paper result family | Current state |
|---|---|
| ChEMBL versus Starling | Complete historical identity-blind reference retained; current-conditioned rerun required |
| One-shot cumulative full-flat | Complete historical reference retained; current source-purity rerun required |
| Append-only progressive | Scaffold BBB v6 and Skin v2 current; Bioavailability scaffold current via receipt/fresh runs; all random rows require replay or audit |
| Full-flat versus full-mechanism | Implementation and historical reference retained; current-conditioned rerun required |
| Scaffold versus random | Current scaffold results exist; random agent results are stale after purity/source updates |
| Identity-blind versus visible | Both implementations retained; no complete current-conditioned matched pair exists |

The last complete blind and visible source matrices use different historical
benchmark lineages and cannot be reported as a matched visibility comparison.

## Current scaffold progressive results

Macro-F1 on valid, default 4/2 card budget:

| Task | None | L1 | L2 | L3 | L4 | L5 | L6 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BBB | 0.6207 | 0.6956 | 0.7240 | 0.7300 | 0.7291 | 0.7338 | — | current strict-voter-L1 v6 |
| Bioavailability | 0.6314 | 0.6900 | 0.7312 | 0.7378 | 0.7545 | 0.7545 | 0.7608 | current via zero-change receipt |
| Skin | 0.6151 | 0.6334 | 0.6328 | 0.6294 | — | — | — | current strict-voter-L1 v2 |

Bioavailability valid still has 262 rows and 212 parents. Removing six bad
nitrendipine source identities changed the index hash, but frozen-protocol
reconstruction found 0/262 changed model-visible selected surfaces across
L1-L6. The split-scoped receipt therefore authorizes retained agent predictions;
it does not authorize random results or baselines trained on old rows.

## Random progressive references requiring replay

These Macro-F1 values are preserved only as historical references:

| Task | None | L1 | L2 | L3 | L4 | L5 | L6 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BBB | 0.5165 | 0.6412 | 0.6624 | 0.6579 | 0.6792 | 0.6779 | — | pre-v6; replay required |
| Bioavailability | 0.5948 | 0.7695 | 0.7572 | 0.7612 | 0.7634 | 0.7615 | 0.7675 | pre-fix; replay/audit required |
| Skin | 0.6848 | 0.7204 | 0.7212 | — | — | — | — | pre-v2 broad-L1; replay required |

Random retrieval uses `parent_disjoint`. Applying scaffold exclusion would be
a different experiment and requires a distinct manifest.

## Scaffold baselines

Macro-F1 on valid:

| Task | MiniMol head | MiniMol KNN same→null | MiniMol KNN all | Morgan KNN same→null | Morgan KNN all | Status |
|---|---:|---:|---:|---:|---:|---|
| BBB | 0.6674 | 0.6071 | 0.5708 | 0.5958 | 0.6174 | current |
| Bioavailability | 0.5872 | 0.5984 | 0.6229 | 0.5891 | 0.5888 | pre-fix; retrain required |
| Skin | 0.6030 | 0.5755 | 0.5755 | 0.5208 | 0.5180 | current |

The Bioavailability correction removed two train rows, so its trained and KNN
baselines must be regenerated before matched publication.

## Record-card budget ablation

The current combined figure compares all available 2/1, 4/2, and 8/4 curves
under the same v8 molecule quotas. Only per-molecule card limits change. Skin
strict-voter-L1 has no 8/4 run, so that cell is intentionally empty.

Bioavailability 8/4 was replayed over all 262 scaffold-valid queries with the
same full L1-L6 plan and inference/retrieval/tool contract as the retained
historical run. The fresh replay completed with zero failures. No level differed
significantly: L1 was 0.6311 versus 0.6375 (McNemar p=0.804), and L6 was 0.6861
versus 0.6852 (p=0.839). The two exact-contract full curves are summarized by
their arithmetic mean and observed min-max range.

The 2/1 runs completed with zero failed queries: BBB 397/397,
Bioavailability 262/262, and Skin 246/246. Every selected molecule set matches
its comparison run and every 2/1 card set is a subset of the higher budget.

| Task | 2/1 Macro-F1, L1 → last | Comparison | Reference L1 → last | Last-level Δ | Paired result |
|---|---:|---|---:|---:|---|
| BBB | 0.6913 → 0.7127 | current 4/2 | 0.6956 → 0.7338 | -2.11 pp | p=0.401; 95% CI [-6.32, +1.96] pp |
| Bioavailability | 0.6474 → 0.6962 | exact 8/4 replay | 0.6311 → 0.6861 | +1.01 pp | p=0.860; 95% CI [-4.09, +6.14] pp |
| Skin | 0.6376 → 0.6515 | current 4/2 | 0.6334 → 0.6294 | +2.21 pp | p=0.804; 95% CI [-2.16, +6.68] pp |

No last-level accuracy difference is significant. The deterministic resource
reduction is substantial:

| Task | Mean last-level visible cards, reference → 2/1 | Mean prompt characters |
|---|---:|---:|
| BBB | 23.39 → 15.73 | -11.2% |
| Bioavailability | 55.39 → 20.63 | -31.7% |
| Skin | 19.61 → 12.56 | -12.1% |

Thus 2/1 is a clear context-cost reduction, not a demonstrated universal
accuracy improvement.

Current combined analysis:

```text
outputs/paper/analysis/progressive_record_card_budget_2_1_4_2_8_4/
  progressive_configuration_metrics.tsv
  progressive_configuration_references.tsv
  summary.json
  figures/progressive_record_card_budget_2_1_4_2_8_4.{png,svg}
```

## Historical source/organization reference

The retained identity-blind GLM matrix is complete but not current-conditioned:

| Task | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism |
|---|---:|---:|---:|---:|---:|---:|
| BBB | 0.6601 | 0.6800 | 0.6936 | 0.6954 | 0.7197 | 0.7092 |
| Bioavailability | 0.5917 | 0.6583 | 0.6190 | 0.6158–0.6099 | 0.6405 | 0.6182 |
| Skin | 0.6057 | 0.5965 | 0.5873 | 0.6164 | 0.6289 | 0.6216 |

Bioavailability has separate historical numeric and full direct conditions,
hence the range. Keep all values labeled historical until the matched current
matrix is complete.

## Removed no-go branches

The 2026-09-01 broad-L1 Skin 8/4 comparison is not a current result. A
connectivity-confounded Bioavailability 8/4 root is excluded. The L1-only
0.7148 Bioavailability diagnostic changed the visible `full_level_plan`, so it
is not an exact replay or a valid budget replicate. The generated record-summary
representation reduced accuracy and added an unnecessary model/cache stage.
Prefix-only level plans also showed no reliable gain. Their implementation and
active artifacts were removed; compact receipts and Git history preserve why
they were rejected.

## Publication boundary

- Read exact roots and status from `current_conditioned_results.json`; this
  summary intentionally avoids duplicating them.
- Bioavailability scaffold agent scores are current, but its baselines require
  retraining and its random split requires a separate audit or replay.
- Skin scaffold 2/1 and 4/2 are current. Do not substitute historical broad-L1
  8/4 or random outputs for missing strict-voter-L1 results.
- Do not present historical blind/visible roots as a matched comparison.
- Do not tune on formal test results or mix ClinTox source roles with assay votes.
- Smokes, retries, router/RL no-go work, and source probes do not enter the
  current result registry.
