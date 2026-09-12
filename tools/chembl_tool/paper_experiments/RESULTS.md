# Current paper results and artifact status

Updated: 2026-09-12.

`current_conditioned_results.json` is the authority for result roots and freshness.
Source reconstruction is documented in `CURRENT_STARLING_RETRIEVAL.md`; the retrieval
protocol is in `ASSAY_LEVEL_RETRIEVAL.md`.

## DILI and Carcinogens: completed cleaning cycle

Both tasks retain Starling-only gold_v4 and the final reviewed retrieval snapshots in
`data/starling_data/<task>/retrieval_final/`. Scaffold and random use the same frozen
cohorts: DILI 3,220/402/402 and Carcinogens 3,754/469/469. L1-only heldout prefiltering,
L2 retention and query-time disjoint are unchanged. The final-source publication
receipt is `data/starling_data/retrieval_final_cleanup.json`.

Latest completed evaluations and figures remain in these registry entries:

- DILI: `replicate_suites.dili_retrieval_review_v5_no_prior_grounded_sim0`,
  valid/test × progressive/full-flat, 11,256 level outputs and zero failures.
- Carcinogens: `replicate_suites.carcinogens_gold_v4_scaffold_4_2.retrieval_cleanup_diagnostic.progressive_round18`,
  progressive valid/test, 6,566 level outputs and zero failures.
- Five matched baselines remain attached to the corresponding gold_v4 suites.

Source consolidation adds no model predictions and does not itself prove equivalence
to the completed diagnostic inputs. Figures and retained predictions keep their actual
prepared-input lineage. Both evaluation sets have been inspected during iterative
cleaning, so these are diagnostic results. Earlier rounds and stopped L1 pilots are
indexed once in `receipts/dili_carcinogens_cleanup_history.json`; their prediction and
reuse dependencies are retained. Pending source questions are not declared resolved.

## Historical DILI / Carcinogens gold_v3 results (2026-09-08)

This historical suite used the identity-repaired TDC-augmented benchmark. Scaffold
train/valid/test counts are DILI 828/103/103 and Carcinogens 840/312/312;
random counts are 828/103/103 and 1164/150/150. Gold, storage, seven-level
semantics and source-build entrypoints are documented in
[`NEW_TASK_SOURCE_DATA.md`](../common/starling/NEW_TASK_SOURCE_DATA.md).

DILI scaffold valid/test names were exchanged at the user's request **after
viewing both cohorts**. Current test is former valid; it is not an unseen test
or a model improvement. Train, random, labels and heldout union are unchanged.
Use `replicate_suites.dili_scaffold_swapped_4_2` and
`outputs/paper/starling_conditioned_dili_scaffold_swapped_4_2_v1/` for current
DILI names. Immutable trace directories retain their original names; the dataset
mapping is `data/conditioned_benchmark/DILI/provenance/scaffold_valid_test_swap_20260908.json`.

### Agent and baseline scope

`replicate_suites.dili_carcinogens_scaffold_4_2` is complete: eight
task/organization/subset cells, 11,620 level predictions and 830 shared fresh
single/None pairs, with zero remaining failures. Each setting has one Flash run
(no repeat SD), using PARCC DeepSeek-V4-Flash-0731, matched prepared evidence,
4/2 cards, at most 256 requests per organization and 1,024 total. Failed-response
six-way races remain within those budgets. Valid checks preceded test execution.
Recovery and frozen-surface receipts are linked in the registry; no scientific
setting changed during recovery. Random and Pro runs for these tasks are absent.

Current final-level macro-F1:

| Task / subset | Rows | None | Progressive L7 | Full-flat L7 |
|---|---:|---:|---:|---:|
| DILI / valid (former test) | 103 | 0.6710 | 0.5657 | 0.6411 |
| DILI / test (former valid) | 103 | 0.6921 | 0.7399 | 0.7299 |
| Carcinogens / valid | 312 | 0.7118 | 0.7296 | 0.7203 |
| Carcinogens / test | 312 | 0.7240 | 0.7509 | 0.7562 |

Five baselines were fitted on mixed training rows and separately on Starling-only
training rows. Each suite has 20 pooled evaluation cells and 4,150 predictions,
with frozen train-only CV/OOF choices, full coverage and zero failures. The
Starling training cohorts are DILI 463 (426+/37−) and Carcinogens 719 (690+/29−).
Unseen TDC null conditions explicitly use zero condition features in this ablation;
the default remains error. Both suites evaluate the same heldout rows. This
matches label source while also changing training size, balance and coverage;
it does not isolate one causal factor or equalize the entire evidence pool.

Current test macro-F1, mixed → Starling-only training:

| Baseline | DILI | Carcinogens |
|---|---:|---:|
| MiniMol head | 0.7456 → 0.5657 | 0.6204 → 0.4809 |
| MiniMol KNN condition | 0.8243 → 0.4148 | 0.7028 → 0.4782 |
| MiniMol KNN all | 0.7031 → 0.4494 | 0.4870 → 0.4593 |
| Morgan KNN condition | 0.8349 → 0.4246 | 0.7000 → 0.4564 |
| Morgan KNN all | 0.7148 → 0.4246 | 0.6134 → 0.4583 |

Canonical baseline entries are `matched_baselines.dili_carcinogens_scaffold`,
`matched_baselines.dili_starling_only_scaffold_swapped`, and the combined suite's
Starling-only ablation entry. Original pre-rename plots and baseline directories
are historical views. Carcinogens mixed-training head AUROC is 0.7061 pooled,
but 0.4974 within Starling and 0.5086 within TDC; pooled performance does not
establish strong discrimination inside either source.

### Starling-label test subset and figures

Filtering current metadata by **`label_source=starling`** requires no new inference,
training, retrieval or split. This is row-level label provenance, not removal of
all molecules that also occur in TDC.

| Current test subset | Rows (+ / −) | None | Progressive L7 | Full-flat L7 | Starling-trained head |
|---|---:|---:|---:|---:|---:|
| DILI | 56 (53 / 3) | 0.5771 | 0.6190 | 0.5962 | 0.8188 |
| Carcinogens | 280 (258 / 22) | 0.7382 | 0.7500 | 0.7607 | 0.4796 |

DILI head predicts all three negatives correctly, progressive only one; this
small denominator makes the comparison unstable. All levels, five baselines,
val/test source strata, row IDs and prediction hashes are in
`outputs/paper/analysis/dili_carcinogens_current_source_strata/`; its
`figures/starling_only_test.png` and `.svg` contain 40 verified points/references.

The historical gold_v3 combined test overview is registered under `test_performance_figures`
at `outputs/paper/analysis/conditioned_scaffold_test_all_tasks_dili_swapped/`:
performance/resource overviews, two readable task panels and six individual
panels, in PNG/SVG. All 104 curve points and 40 references were checked. The
older four tasks retain their existing Flash three-run mean/SD and Pro single
runs. DILI/Carcinogens show single Flash runs and Starling-trained baselines;
missing Pro and ClinTox cells remain explicit. Ames and DILI test were previously
inspected as validation. Rename receipts bind 1,030 baseline and 3,296 agent/None
predictions to current DILI rows without changing their values.

### Trace and source audit limits

`outputs/paper/analysis/dili_test_evidence_decline_v2/REPORT.md` describes **former
test/current valid**: 1,442 outputs, 721 matched input pairs, 10 harmful and four
beneficial None-to-L7 progressive flips, with only one flip after L2. Three
Bosentan conditions wrongly transfer clazosentan clinical observations at L2;
full-flat L7 distinguishes the compounds. Other cases show exposure/endpoint
transfer and an internally contradictory diphenhydramine card. These are
single-run descriptions, not causal ablations or repaired source claims.

Frozen-surface audits cover both heldout suites (415 queries × seven levels each).
The direct-alias guard includes exact reviewed hold `card_04953d07291cbd0c`;
replacement-card and real-request audits are preserved in the original suite.
These are bounded Codex reviews of supplied cards, not exhaustive human-expert
or original-paper verification. Raw records and all frozen experiment inputs
remain intact. Current records, indices and repaired gold now use the shared
restore/export layout; historical `.build` paths survive only as provenance.

## V4 Pro 0813 scaffold-test run (2026-09-05)

The registered `replicate_suites.scaffold_test_pro_0813_4_2` completed one fresh
progressive and one matched full-flat pass over BBB 393 × 5 levels,
Bioavailability 269 × 6, and Skin 241 × 3. It retains the Flash 4/2 cards, tools,
visibility, prompts, temperature 0, 20,480-token output limit and provider-default
reasoning. Fresh Pro single/None outputs are shared by both organizations;
Flash predictions do not count as Pro replicates. All 8,604 level predictions and
903 shared single/None query pairs are complete, with zero remaining failures.
The first rounds had 3 progressive and 8 full-flat failed queries; all recovered
in the first extra retry round. The suite finished at 2026-09-05 21:20:55 UTC.

| Task (final level) | Pro progressive macro-F1 | Pro full-flat macro-F1 |
|---|---:|---:|
| BBB (L5) | 0.7046 | 0.7000 |
| Bioavailability (L6) | 0.8065 | 0.7988 |
| Skin (L3) | 0.6589 | 0.6678 |

These are single-run scores; no repeat SD is available for Pro.

The combined [Flash versus Pro figure](../../../outputs/paper/analysis/conditioned_scaffold_test_flash_vs_pro/figures/flash_vs_pro_full_flat_progressive.png)
overlays both organizations at every level, retaining Flash's three-run mean ±1
sample SD and all five original baselines per task. Pro is a single-run curve
without an interval; None is model-specific. The 56 curve points are backed by
112 run-level metrics, with identical prepared evidence/tools across models and
unchanged Flash scores, SD and baseline values. The command and verification
receipt are registered under `replicate_suites.scaffold_test_pro_0813_4_2`.

One launcher runs the two organizations sequentially, with all three tasks sharing
768 request slots and a 256-per-task cap. Level retries use six-way first-valid
races. OpenRouter Mark1 routes `deepseek/deepseek-v4-pro-0813` by price with
input/output caps of $0.66/$1.98 per million tokens. The successful endpoint
preflight used StreamLake; request IDs, served model and returned usage remain in traces
(the client does not retain OpenRouter's provider-name field).
All 4,302 source prepared levels passed the tool-completeness preflight.
Read the registered `suite_status` and active root's `execution_status.json` on
demand; there is no continuous Codex monitor. The completed Flash three-run
mean/SD comparison below remains separate from this one-run model comparison.

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
| Skin_Reaction | 1,941 / 239 / 241 | 1,937 / 242 / 242 |
| Ames | 1,926 / 274 / 274 | 1,980 / 247 / 247 |
| DILI | 3,220 / 402 / 402 | 3,220 / 402 / 402 |
| Carcinogens | 3,754 / 469 / 469 | 3,754 / 469 / 469 |

Scaffold and random contain the same molecule-condition labels. Random is
parent-grouped, not molecule-row independent. A result is current only when
benchmark/split, heldout index, family catalog, source/organization, model,
prompt, visibility, and identity policy all match. Directory names and sample
counts are insufficient.

Ames scaffold-valid progressive 4/2 is complete for all 274 rows with zero
failures. None/L1/L2/L3/L4/L5 macro-F1 is
0.7002/0.7199/0.7542/0.7561/0.7561/0.7506. Matched full-flat is also complete
for all 274 rows with zero failures; its None/L1/L2/L3/L4/L5 macro-F1 is
0.7002/0.7165/0.7491/0.7510/0.7597/0.7633. All 1,370 paired prepared files
are byte-identical; both curves are single runs. The existing level plotter includes Ames as the fourth column in
`outputs/paper/analysis/progressive_record_card_budget_with_ames_valid/`.
That current-lineage figure leaves Skin curves empty pending replay and Ames
2/1 and 8/4 empty because they have not run. Stale Bioavailability valid baselines
are omitted. Ames v2 includes all five baselines on 274 rows: MiniMol head,
condition-first MiniMol KNN, unrestricted MiniMol KNN, condition-first Morgan KNN,
and unrestricted Morgan KNN have macro-F1 0.6142/0.6328/0.5452/0.5823/0.5255.
Condition-first KNN now fills remaining k=3 slots from unrestricted train
molecules; only the 22 previously unpredicted queries change. Original v1
252-row condition-KNN scores remain historical and are not mixed with v2.
The four-column figure now also overlays Ames matched full-flat 4/2. A compact
paired comparison with shared None and all five baselines is in
`outputs/paper/analysis/ames_scaffold_valid_full_flat_vs_progressive/`, registered
under `replicate_suites.ames_scaffold_valid_4_2.comparison_figure`. Both figure
receipts retain the exact existing-plotter commands and matched-input audit.

Ames trace diagnosis is recorded in
`outputs/paper/analysis/ames_scaffold_valid_trace_diagnosis/TRACE_DIAGNOSIS.md`.
The progressive L4-to-L5 drop is exactly one harmful flip (query 173, TA102
without activation); 53 of the 54 L2 errors remain wrong through L5 despite
792 actual L3-L5 calls. Inspected reasoning exposes cross-system/condition
transfer errors, retention of earlier analog judgments, and compound-specific
Ames-label recall in the one beneficial progressive L3 flip (query 119).
These are observational findings, not a causal prompt ablation. Full-flat wins
6 of 9 final discordances; the paired parent-bootstrap 95% interval for its
macro-F1 advantage is [-0.0179, 0.0444], with model run variability unmeasured.
The report links raw reasoning and source cards; no predictions or gold changed.
The follow-up `RECORD_RELEVANCE_AUDIT.md` audits all 3,015 selected unique cards
and identifies two name/structure-mismatched source cards used by query 57,
plus one method-only card whose empty result is filled from a group aggregate
for query 18. These findings are recorded for bounded source/renderer repair;
the frozen inputs and predictions have not yet been changed or replayed.

The revised Ames prompt scaffold-test suite is complete under
`replicate_suites.ames_scaffold_test_4_2`: 274 rows, progressive and matched
full-flat 4/2 each repeated three times, plus five train-only baselines once.
After shared fresh test single/None and frozen evidence/tool preparation, the
three repetitions run concurrently with 256 request slots each (768 total);
each repetition runs progressive then full-flat. First attempts use one request;
only failed calls/validation retries race up to six requests inside that same
256-slot budget. Epoch 16 and threshold 0.5281078815460205 remain frozen from
the previous train-only MiniMol CV; condition KNN uses the full fallback policy.
The new prompt explicitly permits transferable mechanistic evidence to change
an Ames prediction without new direct Ames measurements. The renderer and scientific settings remain frozen; the confirmed source identity
error below was subsequently excluded with a bounded replay. All 8,220 level predictions
(6 runs x 274 rows x 5 levels, including protocol-defined no-evidence reuse)
have valid outputs, with zero failed queries; all five baselines also completed.
The complete four-task test figure is registered under
`replicate_suites.scaffold_test_4_2.four_task_comparison`, with both full
performance/resource and compact performance-only PNG/SVG exports in
`outputs/paper/analysis/conditioned_scaffold_test_full_flat_vs_progressive_4tasks_3runs_3acaba_repair/`.
It includes all 20 baselines and four shared None references; each agent point
shows the mean of three run-level metrics ±1 sample SD. All 34,032 level
predictions, current input/index/family hashes, paired prepared inputs and
baseline evaluation rows were verified. The prior three-task performance and
resource values are preserved exactly. After the source repair below, Ames L5 Macro-F1 is
0.7574 ± 0.0036 progressive and 0.7459 ± 0.0112 full-flat; these SDs describe
reasoning-run variability over frozen inputs, not confidence intervals.

The pre-repair revised-prompt test [trace diagnosis](../../../outputs/paper/analysis/ames_scaffold_test_trace_diagnosis/report.html)
audits all 8,220 outputs and the eight progressive L2-to-L5 label flips:
3 corrections versus 5 harms, compared with 68 versus 64 adjacent changes for
full-flat. Progressive L5 never changes a label. The case ledger identifies a
misassigned 3-Ac-ABA structure (`ames_base:76701`, PMID 14981162), resistance
selection presented as induced mutation, and a model-invented frameshift meaning
for yeast his1-7 (checked against PMID 9560384). Same-parent TA102/TA97 examples
show how one surrogate negative corrects one condition and harms another.
These are post-hoc observations, not causal ablations or an exhaustive source
audit; its original artifacts remain preserved, and the remaining recorded
defects constrain scientific interpretation of the completed scores.

The separate L1-to-L2 audit in `ames_scaffold_test_l2_diagnosis` finds 12 corrections
versus 13 harms for progressive and 19 versus 36 for full-flat across three repeats.
Full-flat decreases in every repeat; 14 of its 36 harmful changes cite only L1
cards as final prediction basis. Repeated q140 extra-enzyme transfer and q196
strain-unspecified weak-analog negatives contrast with six consistent beneficial
q0 changes. These are descriptive trace findings, not causal effect estimates;
see the registered `l1_to_l2_diagnosis` for complete transitions and caveats.

The confirmed 3-Ac-ABA identity error (`ames_base:76701`) has now been excluded
through the payload-pinned source-review ledger. Both indices were rebuilt;
all gold votes and benchmark files remain byte-identical. The complete test
selected-surface comparison changes only q254/q255/q256 at L3-L5. Targeted replay
completed in `starling_conditioned_ames_scaffold_test_4_2_3acaba_repair_v1`:
54 fresh level outputs across six runs, with 8,166 unchanged-prefix outputs
verified equal to each corresponding original repetition, zero failures, and
all 274-row metrics independently recomputed. Progressive repeat 3 q256 is
corrected at L3-L5; full-flat repeat 2 q255 becomes wrong at L4, while repeat 3
q256 is corrected at L4 and becomes wrong at L5. L5 mean Macro-F1 changes by
+0.001761 progressive and -0.001695 full-flat; this small mixed effect does not
resolve the late-level plateau or isolate model sampling variation. Baselines
and single/None are unchanged. The registry's `source_identity_repair` points to
the completion validation, per-level metrics and exact six prediction changes.

## Retained paper families

| Paper result family | Current state |
|---|---|
| ChEMBL versus Starling | Complete historical identity-blind reference retained; current-conditioned rerun required |
| One-shot cumulative full-flat | Matched visible scaffold-test 4/2: three runs complete for all three tasks; historical identity-blind valid reference retained |
| Append-only progressive | Scaffold-test 4/2: three runs complete for all three tasks; scaffold-valid Skin still needs 2-query 4/2 replay; random requires replay or audit |
| Full-flat versus full-mechanism | Implementation and historical reference retained; current-conditioned rerun required |
| Scaffold versus random | Three-task scaffold-test complete; Skin scaffold-valid has a bounded targeted-replay gap and random agent results are stale |
| Identity-blind versus visible | Both implementations retained; no complete current-conditioned matched pair exists |

The last complete blind and visible source matrices use different historical
benchmark lineages and cannot be reported as a matched visibility comparison.

## Matched independent full-flat scaffold test: replicate 1

The current three-task scaffold-test control is registered under
`one_shot_full_flat_levels.matched_scaffold_test`: all 903 rows / 4,302 levels are
complete with zero failures. The 3,812 nonempty-evidence level calls ran with a
per-task cap of 256 and explicitly authorized global cap of 768. Two automatic
retry rounds resolved the nine first-pass failures. Every level is independent, with no previous prediction/state,
new-card priority, or cross-level prediction reuse. This visible matched control
is separate from the historical identity-blind, per-assay full-flat reference.

| Task | Level | Full-flat Macro-F1 | Progressive Macro-F1 |
|---|---|---:|---:|
| BBB | L5 | 0.7040 | 0.6923 |
| Bioavailability | L6 | 0.7991 | 0.7779 |
| Skin | L3 | 0.5760 | 0.5910 |

These are final-level, single-run scores, not test-selected best levels.
The per-level comparison with all five matched baselines is under
`outputs/paper/analysis/conditioned_scaffold_test_full_flat_vs_progressive/`.

## Current scaffold progressive results

### Three-run scaffold-test repeat suite

`replicate_suites.scaffold_test_4_2` registers the existing complete full-flat and
progressive runs as replicate 1, plus two fresh runs of each method in
`outputs/paper/starling_conditioned_test_replicates_4_2_v1/`. The new four jobs
completed on 2026-09-05 at 13:32 UTC, covering
3,612 query-runs / 17,208 level outputs with zero final failures. They ran sequentially, each with
three concurrent tasks, a per-task request cap of 256 and shared cap of 768.
Six-way first-valid races apply only to retries, with all copies inside those caps.
The original run used sequential retries; that execution-policy difference is recorded.
None/query priors, tools, evidence and model settings stay frozen. These repeats
measure conditional level-reasoning variability, not end-to-end variability of
new query priors. All six method-runs (including the originals) passed current
input/index/family hash checks, exact prepared-input matching, and per-level
metric recomputation from 25,812 predictions (84 task/level metrics).
All 15 baseline points and three shared None points match the original figure.
Read suite status on demand; no Codex monitor is scheduled.

The complete per-level comparison is in
`outputs/paper/analysis/conditioned_scaffold_test_full_flat_vs_progressive_3runs/`:
`figures/full_flat_vs_progressive_mean_sd.{png,svg}`, metric/reference TSVs,
`summary.json`, and `figure_receipt.json` (including the exact command and audit).
Curves show the arithmetic mean of three run-level Macro-F1 scores ±1 sample SD
(`ddof=1`); SD is descriptive run variability, not a confidence interval or
test-sample uncertainty. None and baselines are fixed references without SD bars.

| Task | Final level | Full-flat mean ± SD | Progressive mean ± SD |
|---|---:|---:|---:|
| BBB | L5 | 0.7178 ± 0.0129 | 0.6997 ± 0.0066 |
| Bioavailability | L6 | 0.7895 ± 0.0119 | 0.7824 ± 0.0112 |
| Skin | L3 | 0.5501 ± 0.0271 | 0.5870 ± 0.0035 |

These are the predefined final levels, not test-selected best levels. Skin
progressive declines slightly from L2 (0.5912 ± 0.0016) to L3; its full-flat L3
varies much more across the three runs. BBB progressive's three-run mean rises
from L2 (0.6880) to L3 (0.6941), so the original single-run decline is not a
consistent mean pattern. No significance claim is made from SD-bar overlap.

### Scaffold test, default 4/2: replicate 1 and trace provenance

All 903 scaffold-test rows are complete, with zero failed outputs at every level.
PARCC served `deepseek-ai/DeepSeek-V4-Flash-0731`; the requested per-task
concurrency cap was 256, with an explicitly authorized shared cap of 768.

| Task | Test rows | Valid final outputs | Final level | Macro-F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| BBB | 393 | 393 | L5 | 0.6923 | 0.7583 |
| Bioavailability | 269 | 269 | L6 | 0.7779 | 0.8290 |
| Skin | 241 | 241 | L3 | 0.5910 | 0.7137 |

The full None/L1–LN curve and exact input/index/family hashes are in the
registered `test_results.json` and `TEST_RESULTS.md`. Metrics were independently
recomputed from predictions and frozen test labels. The shared MolGpKa
graph-state race was repaired before inference resumed: all 8,867 unique
property calls and 4,302 prepared level surfaces were checked, 3 changed query
priors replayed, and 57 affected model outputs invalidated. One unsupported
molecule retains explicit pKa/logD unavailable values in 3 pair calls.
The first inference pass had 6 BBB validation failures; the registered recovery
receipt preserves their attempt history and verifies unchanged successful outputs.

The test performance/baseline and resource overview uses the same
`plot_conditioned_progressive_overview` function as valid. PNG/SVG, the
32-row source table, and a metric/input-hash verification receipt are registered
under `progressive_append_only.scaffold_test.analysis`. The subtitle reads
"Scaffold test" directly from the experiment manifest.

The registered `scaffold_test.l2_l3_trace_audit` reviews every BBB/Skin test
L2→L3 flip and all eight BBB valid flips. BBB test changes one prediction to
correct and five to incorrect (Macro-F1 0.695472→0.684232); Skin changes two
to correct and one to incorrect (0.592905→0.590993), with accuracy increasing.
The report separates molecule-identity misattribution, indirect-evidence transfer,
conflicting gold, and class-balanced metric effects. This is a post-hoc audit,
not a test-selected protocol change.

### Scaffold valid references

Earlier valid/random tool texts have not been audited against the serialized
MolGpKa correction. Their source-lineage status does not establish corrected
tool-output equivalence.

Macro-F1 on valid, default 4/2 card budget:

| Task | None | L1 | L2 | L3 | L4 | L5 | L6 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BBB | 0.6207 | 0.6956 | 0.7240 | 0.7300 | 0.7291 | 0.7338 | — | current strict-voter-L1 v6 |
| Bioavailability | 0.6314 | 0.6900 | 0.7312 | 0.7378 | 0.7545 | 0.7545 | 0.7608 | current via zero-change receipt |
| Skin | 0.6279 | 0.6375 | 0.6349 | 0.6429 | — | — | — | last complete pre-MDAM-repair reference; 2 queries need targeted replay |

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

### Test, frozen train-only selection

All five methods cover all 903 test rows with zero failures. Macro-F1:

| Method | BBB (393) | Bioavailability (269) | Skin (241) |
|---|---:|---:|---:|
| MiniMol head | 0.6437 | 0.7102 | 0.5451 |
| MiniMol KNN same→null | 0.6010 | 0.6184 | 0.5410 |
| MiniMol KNN all train | 0.5949 | 0.6153 | 0.5410 |
| Morgan KNN same→null | 0.5479 | 0.6542 | 0.5138 |
| Morgan KNN all train | 0.5495 | 0.6278 | 0.5108 |

MiniMol preserves the valid-baseline selection protocol: train-only scaffold CV
selects epoch by AUROC and an OOF macro-F1 threshold, followed by all-train fitting.
BBB and Skin reuse input-hash-matched checkpoints and reproduce all saved valid
predictions. Bioavailability repeats five-fold CV and training after the two-row
train correction (epoch 5, threshold 0.61925513). The existing per-task fold
counts remain BBB/Bioavailability/Skin = 5/5/3. All four KNN variants use k=3,
unique molecules, majority voting, and train-only references. Each recorded
neighbor and all metrics were independently audited. Full accuracy/AUROC, hashes,
checkpoint dependencies, and the comparison with fixed-final-layer progressive
4/2 are in the registered `matched_baselines.scaffold_test` report.

### Valid references

Macro-F1 on valid:

| Task | MiniMol head | MiniMol KNN same→null | MiniMol KNN all | Morgan KNN same→null | Morgan KNN all | Status |
|---|---:|---:|---:|---:|---:|---|
| BBB | 0.6674 | 0.6071 | 0.5708 | 0.5958 | 0.6174 | current |
| Bioavailability | 0.5872 | 0.5984 | 0.6229 | 0.5891 | 0.5888 | pre-fix; retrain required |
| Skin | 0.6020 | 0.5858 | 0.5858 | 0.5005 | 0.4976 | current v5, 239/239 |

The Bioavailability correction removed two train rows, so its trained and KNN
baselines must be regenerated before matched publication. Skin baselines were
regenerated on the current 1,941-row train split and evaluated on all 239
scaffold-valid rows. The MiniMol head selected epoch 5 and threshold 0.57725
using train-only three-fold scaffold CV; valid labels were not used for tuning.

## Record-card budget ablation

The last-complete Skin figure compares 2/1, 4/2, and 8/4 curves under the same
v8 molecule quotas. Only per-molecule card limits change. These three curves
completed on the pre-MDAM-rebuild source-purity-v5 index with zero failures and
identical within-snapshot input/index/family hashes. They are historical
references, not current results: the final MDAM L2 rebuild changed the selected
surface for 1/2/3 queries at 2/1, 4/2, and 8/4, respectively, so the affected
prefixes require targeted replay before this table or its figure can be
published as current.

| Skin budget | L1 | L2 | L3 | Mean L3 visible cards/query |
|---|---:|---:|---:|---:|
| 2/1 | 0.6118 | 0.6287 | 0.6287 | 10.75 |
| 4/2 | **0.6375** | **0.6349** | **0.6429** | 16.51 |
| 8/4 | 0.6152 | 0.6164 | 0.6209 | 23.19 |

The matched None Macro-F1 is 0.6279. In this last-complete snapshot, 4/2 was the
best Skin budget at every level; increasing to 8/4 added context without
improving the observed score. This comparison must be refreshed after the
bounded replay.

Bioavailability 8/4 was replayed over all 262 scaffold-valid queries with the
same full L1-L6 plan and inference/retrieval/tool contract as the retained
historical run. The fresh replay completed with zero failures. No level differed
significantly: L1 was 0.6311 versus 0.6375 (McNemar p=0.804), and L6 was 0.6861
versus 0.6852 (p=0.839). The two exact-contract full curves are summarized by
their arithmetic mean and observed min-max range.

The retained 2/1 runs completed with zero failed queries: BBB 397/397,
Bioavailability 262/262, and pre-MDAM-repair Skin v5 239/239. Within each lineage,
every selected molecule set matched its comparison run and every 2/1 card set
was a subset of the higher budget.

| Task | 2/1 Macro-F1, L1 → last | Comparison | Reference L1 → last | Last-level Δ | Paired result |
|---|---:|---|---:|---:|---|
| BBB | 0.6913 → 0.7127 | current 4/2 | 0.6956 → 0.7338 | -2.11 pp | p=0.401; 95% CI [-6.32, +1.96] pp |
| Bioavailability | 0.6474 → 0.6962 | exact 8/4 replay | 0.6311 → 0.6861 | +1.01 pp | p=0.860; 95% CI [-4.09, +6.14] pp |
| Skin | 0.6118 → 0.6287 | pre-MDAM-repair 4/2 reference | 0.6375 → 0.6429 | -1.42 pp | historical pending targeted replay |

The BBB and Bioavailability paired comparisons are not significant. The Skin
statistics were valid within the immediately preceding v5 lineage, but the
final reproducibility rebuild added 37 MDAM outcomes to L2 and changed selected
surfaces for 1/2/3 queries under 2/1, 4/2, and 8/4. They are therefore retained
only as last-complete references until targeted replay. The deterministic
resource reduction in that completed lineage was substantial:

| Task | Mean last-level visible cards, reference → 2/1 | Card reduction |
|---|---:|---:|
| BBB | 23.39 → 15.73 | 32.7% |
| Bioavailability | 55.39 → 20.63 | 62.8% |
| Skin | 16.51 → 10.75 | 34.9% |

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

This combined figure is not current for Skin until the affected prefixes in
`receipts/skin_scaffold_valid_mdam_family_rebuild.json` are replayed; BBB and
Bioavailability cells remain current.

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
- Bioavailability scaffold-test baselines were freshly completed after the train
  correction. Its older valid baseline table remains a pre-fix reference, and
  its random split requires a separate audit or replay.
- Skin matched MiniMol/Morgan baselines remain current because the split is
  byte-identical. The completed scaffold 2/1, 4/2, and 8/4 agent roots are
  last-complete references pending targeted replay after the MDAM L2 repair;
  historical broad-L1 and random outputs remain excluded from current cells.
- Do not present historical blind/visible roots as a matched comparison.
- Do not tune on formal test results or mix ClinTox source roles with assay votes.
- Smokes, retries, router/RL no-go work, and source probes do not enter the
  current result registry.
