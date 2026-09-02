# Current paper results and artifact status

Updated: 2026-09-02.

This file is the human-readable companion to
`current_conditioned_results.json`. It records only the retained paper result
families and their freshness. Historical material is limited to the compact
receipts needed to explain a retained reference or an excluded comparison.

## Evaluation data

All active experiments use the Conditioned Benchmark:

```text
data/conditioned_benchmark/<Task>/{scaffold,random}/
```

| Task | Scaffold train / valid / test | Random train / valid / test |
|---|---:|---:|
| BBB_Martins | 3,053 / 397 / 393 | 3,075 / 384 / 384 |
| Bioavailability_Ma | 1,956 / 262 / 269 | 1,989 / 249 / 249 |
| ClinTox | 1,144 / 142 / 142 | 1,142 / 143 / 143 |
| Skin_Reaction | 1,997 / 246 / 248 | 1,993 / 249 / 249 |

The construction and leakage contract is
`tools/chembl_tool/common/starling/CONDITIONED_BENCHMARK.md`. Scaffold and
random contain the same molecule-condition labels. Random is parent-grouped,
not molecule-row independent.

## Freshness rules

A result is current only when all relevant identities match:

- benchmark input and split;
- held-out-filtered retrieval index and source-family catalog;
- evidence source and organization;
- model and prompt profile;
- visibility and neighbor-identity policies.

Matching a directory name, molecule count, or valid-input hash is not enough.
`current_conditioned_results.json` is the machine-readable authority.

## Retained result families

| Paper result family | Current state |
|---|---|
| ChEMBL versus Starling retrieval | Last complete identity-blind reference exists on historical record-supported-v2; a complete current-conditioned matrix is still required |
| One-shot full-flat cumulative levels | Last complete reference exists; current source-purity rerun is required |
| Append-only progressive levels | BBB scaffold-valid strict-voter-L1 v6 and Skin scaffold-valid strict-voter-L1 v2 are current; their random results require replay; Bioavailability scaffold is current via receipt/fresh runs and random requires replay |
| Full-flat versus full-mechanism | Preserved in the source/reasoning matrix; current-conditioned rerun is required |
| Scaffold versus random | BBB and Skin scaffold are current while random remains a pre-voter-only reference; Bioavailability scaffold is current and random remains unaudited |
| Identity-blind versus visible | Both implementations and complete historical artifacts are retained, but there is no complete current-conditioned matched pair yet |

The last complete visible and blind source matrices use different historical
benchmark lineages. They must not be used as a matched visibility comparison.
The previously indexed record-supported-v2 visible directory was only a partial
reuse checkpoint and contains no batch-level predictions or metrics.

## Current and retained scaffold progressive curve

Macro-F1 on valid:

| Task | None | L1 | L2 | L3 | L4 | L5 | L6 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BBB | 0.6207 | 0.6956 | 0.7240 | 0.7300 | 0.7291 | 0.7338 | — | current strict-voter-L1 v6 (4/2) |
| Bioavailability | 0.6314 | 0.6900 | 0.7312 | 0.7378 | 0.7545 | 0.7545 | 0.7608 | current via zero-change retrieval receipt |
| Skin | 0.6151 | 0.6334 | 0.6328 | 0.6294 | — | — | — | current strict-voter-L1 v2 (4/2) |

The Bioavailability benchmark valid input is unchanged: 262 molecule-condition
rows, 212 unique parents, and the same ordered input hash. Removing six
erroneous nitrendipine-identity source records changed the held-out index hash,
but a frozen-protocol reconstruction found 0/262 changed selected retrieval
surfaces across L1-L6. The retained agent predictions are therefore current via
the split-scoped zero-change receipt; no LLM replay is required. Audit-only
global source-pool counts changed and remain excluded from the model prompt.

## Current and retained random progressive curve

Macro-F1 on valid:

| Task | None | L1 | L2 | L3 | L4 | L5 | L6 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BBB | 0.5165 | 0.6412 | 0.6624 | 0.6579 | 0.6792 | 0.6779 | — | pre-v6 reference; random replay required |
| Bioavailability | 0.5948 | 0.7695 | 0.7572 | 0.7612 | 0.7634 | 0.7615 | 0.7675 | pre-fix reference; replay required |
| Skin | 0.6848 | 0.7204 | 0.7212 | — | — | — | — | pre-v2 broad-L1 reference; replay required |

Random retrieval uses `parent_disjoint`. Applying scaffold exclusion to these
rows would define a different experiment and must use a distinct manifest.

## Latest scaffold baselines

Macro-F1 on valid:

| Task | MiniMol head | MiniMol KNN same→null | MiniMol KNN all train | Morgan KNN same→null | Morgan KNN all train | Status |
|---|---:|---:|---:|---:|---:|---|
| BBB | 0.6674 | 0.6071 | 0.5708 | 0.5958 | 0.6174 | current |
| Bioavailability | 0.5872 | 0.5984 | 0.6229 | 0.5891 | 0.5888 | pre-fix reference; retrain required |
| Skin | 0.6030 | 0.5755 | 0.5755 | 0.5208 | 0.5180 | current |

The Bioavailability correction removed two train rows. Trained and KNN
baselines must be regenerated before they are paired with the repaired agent
results. The zero-change agent receipt does not authorize baseline reuse.

## Last complete source and organization reference

The retained identity-blind GLM reference is:

```text
outputs/paper/
  molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_glm_5_2_nvfp4/
```

It is complete and useful for preserving the ChEMBL-versus-Starling and
full-flat-versus-full-mechanism implementations, but it is not a current
Conditioned Benchmark result. Macro-F1:

| Task | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism |
|---|---:|---:|---:|---:|---:|---:|
| BBB | 0.6601 | 0.6800 | 0.6936 | 0.6954 | 0.7197 | 0.7092 |
| Bioavailability | 0.5917 | 0.6583 | 0.6190 | 0.6158–0.6099 | 0.6405 | 0.6182 |
| Skin | 0.6057 | 0.5965 | 0.5873 | 0.6164 | 0.6289 | 0.6216 |

Bioavailability has separate historical numeric and full direct conditions,
hence the direct range. These values must remain labeled as a historical
reference until the matched current-conditioned matrix is complete.

## Historical 4/2 versus 8/4 record-card snapshot

This 2026-09-01 snapshot is retained to document the original 4/2-versus-8/4
analysis, but it is not the current cross-task figure: its Skin rows predate the
strict-voter-L1 v2 replay. The current combined result is recorded in the next
section. In the historical snapshot, scaffold-valid 8/4 completed for BBB (397/397),
Bioavailability (262/262), and Skin (246/246), with zero failed queries. The
comparison keeps the v8 molecule quotas fixed at L1 Top-10 and later Top-3 new
plus Top-3 active molecules; only the per-molecule card limits change from 4/2
to 8/4. The performance panels include the configuration-matched `None` point
and the five task-specific MiniMol/Morgan baseline references; resource panels
remain level-only because those references have no progressive evidence budget.
BBB and Skin baselines match the current benchmark cohort and training lineage.
Bioavailability baselines remain pre-fix references pending retraining after the
two erroneous train rows were removed.

| Task | 4/2 Macro-F1, L1 → last | 8/4 Macro-F1, L1 → last | Last-level Δ |
|---|---:|---:|---:|
| BBB | 0.6956 → 0.7338 | 0.6800 → 0.7444 | +1.1 pp |
| Bioavailability | 0.6900 → 0.7608 | 0.6375 → 0.6852 | -7.6 pp |
| Skin | 0.6273 → 0.6342 | 0.6410 → 0.6375 | +0.3 pp |

BBB now uses fresh, paired strict-voter-L1 v6 4/2 and 8/4 reruns. Both read the
same input, family manifest, heldout-filtered index, query prior, model identity,
and 1,493 model-called checkpoints; both finish 397/397 queries with zero failed
runs. The only configured difference is the per-molecule record-card budget,
so no BBB cross-lineage reuse receipt is required. Historical v5 BBB roots are
retained only as provenance. The Skin comparison used a transport-matched v1
broad-L1 source and is now historical after the strict-voter-L1 v2 rebuild; its
numbers remain an audit reference, not a current method result. Bioavailability reuses its completed
v8 4/2 reference with matched model identity and evaluation inputs, but its
historical per-query provider path is not fully matched; it additionally relies
on the scaffold-valid nitrendipine
zero-change receipt across its index/family lineage update. Its corrected 8/4
preparation has zero `127.0.0.1:8765` connection/permission failures and exactly
matches all 5,694 shared analog-tool surfaces in the 4/2 reference. Both retain
the same 72 deterministic `mmp_structure_compare` runtime errors, so those are
not an 8/4-specific retrieval difference. The earlier Bioavailability 8/4 root
under `...card_budget_8_4_v1.../bio` is connectivity-confounded and excluded.
The remaining cross-lineage audit receipts cover Bioavailability. The paired
figure and machine-readable summary are under:

```text
outputs/paper/analysis/progressive_record_card_budget_4_2_vs_8_4/
```

### Rejected Bioavailability L1 record-summary diagnostic

On 2026-09-02, an analysis-only scaffold-valid L1 diagnostic compared raw
record cards with one DeepSeek-generated summary per source molecule and
evidence level. The summary representation was rejected: at 4/2, Macro-F1 fell
from 0.6831 to 0.6547 while mean prompt characters decreased only 2.8%; at 8/4,
Macro-F1 fell from 0.7148 to 0.6433 while prompt characters decreased only 6.4%.
For the paired 8/4 decisions, the Macro-F1 difference was -7.14 pp (bootstrap
95% interval -12.54 to -1.89 pp; exact McNemar p=0.029), with 19
raw-correct-to-summary-wrong and 7 raw-wrong-to-summary-correct flips. This did
not justify the extra model stage or maintenance surface. The implementation,
cache, and run artifacts were removed; this result receipt is retained only to
prevent the discarded representation from being reintroduced as a current
method.

The fresh L1-only raw 8/4 score of 0.7148 is **not** an exact repeat of the
historical full-curve raw 8/4 score of 0.6375. A post-run payload audit found deterministic retrieval:
all 262/262 L1 `active_evidence` payloads, selected card IDs, query priors, tool
summaries, conditions, and selection audits were identical. All 183/183
model-called system messages were also identical. However, every user payload
differed in exactly two places: the historical request exposed the complete
L1-L6 `full_level_plan`, whereas the L1-only diagnostic exposed only L1, and
the diagnostic added `protocol.evidence_representation=raw_records`. Removing
those two fields made all 183/183 user payloads identical. The diagnostic user
messages were therefore 856 characters shorter per model call.

The 21 historical-to-fresh flips were directionally balanced (11 low-to-high,
10 high-to-low), so they do not indicate a simple label-threshold shift.
Provider-key routing was also balanced (historical Mark 1/Mark 4: 92/91; fresh:
94/89), and both traces report the same mutable OpenRouter model alias. At the
same time, mean completion tokens rose from 2,636 to 7,220, mean reasoning
tokens from 2,119 to 6,617, and calls requiring structured-output retry from
11/183 to 23/183; 19 fresh calls initially returned an empty or non-object
response. This indicates a materially different generation regime, but the
stored trace does not expose an immutable upstream model/backend revision.
Consequently the 0.6375-to-0.7148 difference must not be labeled pure run
variance, a retrieval change, or a record-budget effect. The exact full-curve
replay below supersedes that invalid comparison.

## Current exact Bioavailability 8/4 replay and 2/1 budget ablation

On 2026-09-02, Bioavailability 8/4 was replayed over all 262 scaffold-valid
queries using the complete L1-L6 plan and the same model, provider-pool,
prompt, generation, visibility, identity, retrieval, and prefetched-tool
contract as the historical run. It finished with zero failed queries. L1 was
0.6311 versus the historical 0.6375 (difference -0.65 pp; 16 prediction flips;
exact McNemar p=0.804; paired-bootstrap 95% interval -4.65 to +3.40 pp). L6 was
0.6861 versus 0.6852 (difference +0.09 pp; 24 flips; p=0.839; interval -4.48
to +4.52 pp). No level differed significantly. This restores the historical
performance scale and confirms that the rejected 0.7148 L1-only run was not a
valid same-contract replay.

The 2/1 ablation then completed for BBB (397/397), Bioavailability (262/262),
and Skin (246/246), all with zero failed queries. It keeps the same molecule
selection quotas and changes only the L1/later per-molecule card limits from
the comparison budget to 2/1. Across every query and level, query priors,
prefetched query-tool summaries, conditions, retrieval audits, and selected
molecule sets match their comparison run; every 2/1 card set is a subset of
the higher-budget card set.

| Task | 2/1 Macro-F1, L1 → last | Comparison | Reference L1 → last | Last-level 2/1 Δ |
|---|---:|---|---:|---:|
| BBB | 0.6913 → 0.7127 | current 4/2 | 0.6956 → 0.7338 | -2.11 pp |
| Bioavailability | 0.6474 → 0.6962 | exact-replay 8/4 | 0.6311 → 0.6861 | +1.01 pp |
| Skin | 0.6376 → 0.6515 | current 4/2 | 0.6334 → 0.6294 | +2.21 pp |

None of the last-level differences is statistically significant. BBB has 51
last-level flips (22 favor 2/1, 29 favor 4/2; McNemar p=0.401; bootstrap 95%
interval -6.32 to +1.96 pp). Bioavailability has 32 flips (17 favor 2/1, 15
favor 8/4; p=0.860; interval -4.09 to +6.14 pp). Skin has 16 flips (9 favor
2/1, 7 favor 4/2; p=0.804; interval -2.16 to +6.68 pp).

The lower budget materially reduces context. At the last level, mean visible
cards and mean prompt characters change from 23.39 to 15.73 and by -11.2% for
BBB, from 55.39 to 20.63 and by -31.7% for Bioavailability, and from 19.61 to
12.56 and by -12.1% for Skin. The task pattern is therefore not a universal
accuracy gain from fewer records: BBB trends lower at 2/1, Skin trends higher,
and Bioavailability is lower at L2-L4 but catches up by L5-L6. All of these
accuracy differences remain within paired uncertainty, while the context and
cost reduction is deterministic.

The retained run roots are:

```text
Bioavailability exact 8/4 replay:
outputs/paper/starling_conditioned_assay_progressive_visible_bio_card_budget_8_4_exact_replay_v1/scaffold_valid_deepseek_v4_flash_0731

BBB/Bioavailability/Skin 2/1:
outputs/paper/starling_conditioned_assay_progressive_visible_card_budget_2_1_v1/scaffold_valid_deepseek_v4_flash_0731

Skin current 4/2 control:
outputs/paper/starling_conditioned_assay_progressive_visible_card_budget_4_2_current_control_v1/scaffold_valid_deepseek_v4_flash_0731/skin
```

The combined current-lineage figure and machine-readable summary are under:

```text
outputs/paper/analysis/progressive_record_card_budget_2_1_4_2_8_4/
```

It places all available 2/1, 4/2, and 8/4 curves in one figure together with
the matched no-retrieval and baseline references. The two exact-contract
Bioavailability 8/4 full-curve runs are summarized by their arithmetic mean;
whiskers show their observed minimum and maximum at each level. All other
curves currently have one run and therefore no nonzero run-range whisker.
There is no current strict-voter-L1 Skin 8/4 run, so that cell is marked
unavailable rather than populated from the historical broad-L1 artifact.

An exploratory 2026-09-02 run that hid future level-plan entries produced no
significant task/level benefit and trended lower throughout Bioavailability.
The alternative prompt branch, tests, registry entry, and run/analysis
artifacts were removed; the progressive contract exposes the complete level
plan at every level.

## Canonical artifact roots

```text
BBB progressive scaffold:
outputs/paper/starling_conditioned_assay_progressive_visible_bbb_source_purity_v6_card_budget_4_2_v1/scaffold_valid_deepseek_v4_flash_0731

Bioavailability progressive scaffold, current agent curve via zero-change receipt:
outputs/paper/starling_conditioned_assay_progressive_visible_v10_bio_legacy_gold_vote_pure_v1/scaffold_valid_deepseek_v4_flash_0731

Skin progressive scaffold, current strict-voter-L1 v2 4/2 curve:
outputs/paper/starling_conditioned_assay_progressive_visible_card_budget_4_2_current_control_v1/scaffold_valid_deepseek_v4_flash_0731/skin

Progressive random combined root:
outputs/paper/starling_conditioned_assay_progressive_random_quality_v2_parent_disjoint/random_valid_deepseek_v4_flash_0731

One-shot family curve, last complete reference:
outputs/paper/starling_conditioned_assay_family_curve_v1/scaffold_valid_epyc_deepseek_v4_flash_0731
```

For the exact status, source/index hashes, and result roles, read
`current_conditioned_results.json`; this Markdown summary deliberately does not
duplicate every manifest field.

## Publication boundary

- Bioavailability scaffold progressive scores are current, but its pre-fix
  baselines still require retraining and its random split still requires a
  separate change audit or replay.
- Skin scaffold progressive 2/1 and 4/2 scores are current. Do not publish the
  retained broad-L1 v1 curves as current or treat them as a replacement for the
  missing strict-voter-L1 v2 scaffold 8/4 and random replays.
- Do not present the historical blind and visible roots as a matched comparison.
- Do not tune a method on formal test results.
- Do not mix ClinTox source-role labels with assay-vote counts.
- Do not promote transient smokes, timeout retries, router/RL no-go experiments,
  or source probes into the paper result registry.
