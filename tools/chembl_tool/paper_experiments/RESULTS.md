# Current paper results and artifact status

Updated: 2026-09-14.

`current_conditioned_results.json` is the authority for result roots and freshness.
Source reconstruction is documented in `CURRENT_STARLING_RETRIEVAL.md`; the retrieval
protocol is in `ASSAY_LEVEL_RETRIEVAL.md`.

## Skin primary-source rebinding replay (2026-09-14)

`replicate_suites.skin_identity_rebinding_replay_20260914` uses source-purity v7:
seven records rebound from primary structures, one repaired INF duplicate represented
by the existing correct record, two unresolved conjugate structures still quarantined.
Gold/splits and the legacy prompt/tools/prior remain fixed. All 480 scaffold rows were
audited against actual original hybrid inputs. Relative to the completed quarantine
replay, only test #39 L3 and valid #222 L3 require new inference; 22 unchanged levels
are reused. Both new L3 calls are complete and actual-request/state verified;
test #39 and valid #222 keep their previous risk labels. Full-test final Macro-F1
remains 0.582343 (171/241 correct); valid remains 0.650761 (178/239 correct).
The source repair improves identity fidelity but has no observed label/score gain.

The preceding `skin_trace_source_replay_20260914` v6 quarantine experiment is complete
on both sets, with no label changes among its six affected queries. Its full-test
Macro-F1 is 0.582343 and valid is 0.650761; this is not the rebinding result.
Source evidence and reproducibility: `data/starling_data/skin_reaction/trace_review_20260914/README.md`.

## L3+ parent-disjoint test progressive priority run (2026-09-14)

The matched-full-flat follow-up is launched under
`replicate_suites.l3_parent_disjoint_matched_full_flat_20260914`: valid first
(2,043 rows / 11,741 levels), then test (2,048 rows / 11,769 levels). It consumes
the exact frozen prepared inputs of the progressive comparison below, including
legacy Bioavailability/Skin prompt/tools/prior; the later Skin v7 repair is separate.
All six tasks dynamically share three endpoint pools of 512 admission permits each
(1,536 total),
including width-6 retry races, through `common.request_admission`. The initial HAProxy-only limiter
allowed excess client-side queueing inside the 900-second deadline; it was replaced
with admission before HTTP submission, preserving successful checkpoints. Calls now
use direct SSH endpoints without a second proxy queue: dgx007:50002,
dgx011:50001 and dgx014:50002, all reporting DeepSeek-V4-Flash-0731 and the same default
thinking/max-reasoning settings. Model weight revision was not independently verified.
The contended dgx005 backend has been removed from this run. dgx007 passed a
512-request stress probe; dgx011 initially had 80 connection ReadErrors during
simultaneous burst probes, then passed two isolated 512-request repeats. These probes
use a repeated real prompt capped at 128 output tokens and do not establish
full-completion throughput. Rebalance and checkpoint preservation evidence is in
`endpoint_rebalance_three_hosts_20260915.json` under the run root, with a link to
the previous dgx014 admission check. Experiment calls bypass LiteLLM.
Both subsets passed input preparation checks. Valid inference has been launched;
test is gated on successful output, independent-request and metric verification.
Live phase, exact commands, frozen runtime hashes and the resumable coordinator are in
`outputs/paper/l3_parent_disjoint_matched_full_flat_20260914/`.

The complete valid/test six-task policy comparison figures, including None and five
matched baselines, are registered under `diagnostics.scaffold_vs_l3_parent_disjoint_figures_20260914`.
PNG/SVG and exact plotting tables are in `outputs/paper/analysis/scaffold_vs_l3_parent_disjoint_20260914/`.
All 60 baseline cells were checked against current molecule/label/condition row order
and their metrics recomputed. These are single-run paired policy comparisons, with
no estimated uncertainty interval; the later Skin source-rebinding intervention is separate.

The valid overlay `valid/performance_with_v5_full_flat.{png,svg}` adds all six completed
original v5 scaffold-disjoint matched-full-flat curves (11,741 verified level outputs),
with exact values in `valid/metrics_with_v5_full_flat.tsv` and provenance in
`full_flat_overlay_receipt.json` under the same figure root. It preserves the original
two-arm figures. The third curve is a historical configuration comparison: Bioavailability/Skin
also differ in prompt/tools/prior, and Ames/DILI use pre-repair evidence. The displayed
None references belong to progressive; this overlay is not a matched organization ablation.

L3+ parent-disjoint progressive is complete and verified on all six tasks: test 2,048 rows /
11,769 levels, followed by valid 2,043 rows / 11,741 levels.
The user paused both remaining v5 test matched-full-flat stages. This six-task
2,048-row experiment retains scaffold exclusion on source-family L1/L2 cards
and relaxes only L3+ cards to parent exclusion. Budgets and similarity thresholds
match the completed retrieval diagnostic. Bioavailability/Skin use their frozen
legacy v2 prompt, old query tools/prior, and commit 80f26e7 tool implementation
for newly selected neighbor pairs; the other four tasks use v5. Exact unchanged
prefixes are reused and downstream states are regenerated after the first changed
input. Three frozen runners dynamically share one 1,024-request upstream limit through
a local HAProxy gate, without task or runtime reservations,
including width-6 retry races. Source inputs, queue handoff and results are linked
by `replicate_suites.l3_parent_disjoint_progressive_test_20260914`.
The default retrieval protocol and paused v5 test queue are not promoted or resumed.

## L3+ parent-disjoint valid follow-up (2026-09-14)

`replicate_suites.l3_parent_disjoint_progressive_valid_20260914` is complete and
verified: all six scaffold-valid tasks, 2,043 rows / 11,741 levels. Runtime, prompts,
tools/prior, retrieval settings, shared 1,024 request cap and retry behavior match
test. Bioavailability/Skin retain legacy v2. Five source-repaired Skin valid levels
(#193/#206) received the missing legacy strict-reference replay before hybrid
inference. Exact unchanged prefixes were reused; paired results and baseline-verified
figures are linked by the registry. This is one policy comparison, not a new replicate.

## L3+ parent-disjoint retrieval diagnostic (2026-09-14)

`diagnostics.l3_parent_disjoint_retrieval_20260914` records a retrieval-only
comparison across all 4,091 current scaffold valid/test rows. L1/L2 retain
scaffold exclusion; only source-family L3+ cards admit parent-disjoint neighbors.
Current source indices, thresholds and 4/2 append-only budgets remain fixed.
All 23,510 strict-arm selected surfaces match current prepared inputs; L1/L2
are identical between arms. Final query-mean neighbor similarity increases in
all six tasks on both subsets. No model inference was run, so no predictive
performance improvement is claimed. See
`outputs/paper/analysis/l3_parent_disjoint_retrieval_20260914/REPORT.md` and `level_statistics.csv`.
Production retrieval defaults are unchanged.

The subsequent trace audit, `diagnostics.l3_parent_disjoint_trace_effects_20260914`,
compares all 4,091 rows and reads 32 purposively selected paired trajectories.
Of 2,403 changed final contexts, 136 correct earlier errors and 82 introduce errors;
2,295 cite new cards in claims. In 2,150 contexts the fixed budget also replaces
some strict-arm cards, so this is not a pure evidence-addition experiment.
Endpoint/condition transfer and unsupported prior judgments remain important observed
failure patterns; citation alone does not establish causal benefit. Full task metrics,
examples and limitations are in the registered report. This audit is separate from
the later Skin source rebinding and the running matched full-flat comparison.

## Six-task v5 rerun: progressive complete, valid full-flat complete, test full-flat paused

`replicate_suites.reasoning_prompt_v5_scaffold_20260913` tracks fresh scaffold
valid and test runs of BBB, Bioavailability, Skin, Ames, DILI and Carcinogens:
2,043 valid / 2,048 test rows, one progressive and one matched-full-flat run,
47,020 expected level outputs. Fresh single/None and tree tools; DILI retains
its no-prior/grounded similarity-zero setting. Sequential existing runners use
one global 1024-slot pool per stage (explicitly authorized); failed/invalid attempts race at width 6
within that pool. Settings are frozen before inference. Completion and scores
are recorded in the registry: all valid matched-full-flat tasks are now complete and
verified (11,741 level outputs), while test full-flat remains paused by user request.
This is not an isolated prompt ablation against historical tools.
Progressive is complete on both subsets: valid 11,741 and test 11,769 successful
level outputs. The suite retains provider/tunnel changes, DILI preparation acceleration
and checkpoint-preservation receipts as operational provenance; configured providers
must not be confused with providers observed in successful request traces.

The completed valid figure is registered as `valid_progressive_figure`, with
PNG/SVG, per-level metrics and a 30-cell baseline audit. Bioavailability valid
baselines were refreshed with frozen current-train settings; Ames historical test
baseline rows were verified against current valid after the split-name exchange.
The older `test_progressive_five_task_figure` omits DILI and remains a historical view;
the six-task figure below includes DILI and the user-selected legacy Bioavailability/
Skin runs. Figure versions must not be pooled as independent experiments.

## Six-task test figure with user-selected historical versions

`replicate_suites.scaffold_test_six_tasks_selected_versions_20260913` registers
[the six-task progressive figure](../../../outputs/paper/analysis/scaffold_test_six_tasks_selected_versions_20260913/figures/test_six_tasks_progressive_baselines.png),
its SVG, plot command and verification receipt. It adds completed DILI test
(402 queries, 2,814 successful level outputs) to the previous five-task overview.
All six tasks cover 2,048 test queries with 30 matched baseline cells.

At the user's request, Bioavailability and Skin use **all three historical v2
progressive runs**, with their frozen old tool context and priors, including the
corresponding old None references. Their final Macro-F1 means are 0.7824 ± 0.0112
and 0.5870 ± 0.0035 (sample SD). These are existing measured results, not results
from a new run of the restored configuration. BBB, Carcinogens and DILI use one
v5 run each; Ames uses the completed source-repair replay, whose test predictions
are unchanged. DILI final L7 Macro-F1 is 0.6053, versus None 0.5629.
This is a **mixed-version overview**, not a uniform v5 suite or an isolated prompt
ablation. Every included level prediction and None score was recomputed against
current ordered test labels; all baselines also passed molecule/condition/label
and training-lineage checks. The older five-task v5 figure remains historical.

## Current DILI prompt-v5 source replay

`replicate_suites.dili_v5_trace_source_replay_20260913` uses the current
`reasoning_prompt_v5_scaffold_20260913` progressive runs and
`progressive_evidence_revision.v5`, not the historical v2-prompt diagnostic below.
The source review corrected or quarantined 13 attribution/identity/scope records;
gold and both benchmark split schemes are unchanged. Both indices, portable
source restore and exported card tables passed validation. All 804 selected-input
trajectories were compared: valid has no changed input, while 15 test queries
require 75 fresh level outputs. Exact unchanged prefixes and frozen v5 tools/priors
are reused. The replay is complete and verified: 75 fresh outputs, zero failed queries. Test L7 Macro-F1 is 0.605307 (before 0.605307), with 0 corrected and 0 regressed final labels. Valid L7 remains 0.679991. Full per-level changes and actual-request checks are linked from the registry.

Before repair, current test L2→L7 has 18 adjacent flip events, 10 beneficial and
8 harmful: correct predictions rise from 281 to 283 of 402, and Macro-F1 from
0.595980 to 0.605307. Object misidentification also remains in current traces
(for example #284 and #395); source validity and model misuse are separate.
The detailed report is `outputs/paper/dili_trace_source_replay_20260913/REPORT.md`.
The verified before/after test Macro-F1 figure, including None and five matched
baselines, is linked from this suite's `test_performance_figure`; both curves use
prompt v5. L1/L2 improve; L3-L7 are identical.

## Historical DILI test plateau diagnostic

`diagnostics.dili_legacy_test_level_plateau_20260913` audits the old v2-prompt
progressive run on retrieval-review v5 (402 rows, 2,814 levels). L2→L7 changes
only six labels: four corrected, two regressed, with 116 errors persisting;
Macro-F1 moves from 0.5895 to 0.6019. All levels actually called the model.
The report separates weak analog transfer, inherited evidence-state judgments,
protective-outcome misuse, source-attribution conflicts, two verified APO
structure mismatches, a naloxone glucuronide annotation inconsistency whose
original numerical table remains unverified, and an MTT placement candidate.
Source findings and wrong/correct trace applications are separate; presence or
citation does not establish causal harm. Existing v4→v5 repairs left every test
progressive label unchanged, verified again. This diagnostic does not modify
source, gold, retrieval, prompts or run new inference; it is not a full-library
quality certificate.

## Ames valid level-decline diagnostic (2026-09-13)

`diagnostics.ames_valid_v5_level_decline_20260913` links the read-only review of
all 23 adjacent label flips in the current 274-row valid run. Macro-F1 falls
from 0.7546 at L1 to 0.7152 at L5; L2 accounts for 13 harmful and 6 beneficial
flips. Two source-level placement candidates coexist with endpoint/condition
transfer errors, unsupported mechanistic vetoes and incomplete SAR context.
The report retains same-card beneficial controls and an unresolved quinoline
TA98 cross-study/gold discrepancy. No source, prompt or gold change and no
model replay accompany this diagnostic; it does not establish removal effects.

The subsequent authorized source repair is separately registered as
`diagnostics.ames_trace_source_repair_20260913`: `ames_v2:425939` moves from L2
to L4 (SOS/umu), and `ames_base:186648` from L2 to L3 (plant chromosome damage).
The cinnoline card `ames_base:80868` now preserves the quinoline exception in
its series-level structural context. PMID7022455's original Table 2 and text
confirm the older quinoline TA98/+S9 positive result; the newer study's negative
result remains a cross-study disagreement. Study-unit labels and split membership
are unchanged. Rebuilt retrieval requires a selected-input audit or fresh replay;
the ongoing v5 suite retains its pre-repair Ames inputs in its frozen runtime.

`replicate_suites.ames_trace_source_replay_20260913` now records the completed
targeted progressive replay over current repaired retrieval: valid 19 affected
queries / 77 levels, test four queries / 20 levels. The other 2,643 levels pass
exact unchanged-prefix reuse checks. There are 93 newly inferred levels and four
carry-forward levels, with zero final failures; all 548 rows and 2,740 level
inputs/outputs are verified against frozen v5 priors, tools, prompt and gold.
Only valid #173 needs tools for one newly selected neighbor.

Final Macro-F1 and accuracy are unchanged: valid 0.7152 / 0.7555, test 0.7545 /
0.7993. Valid #211/#220 improve and #173/#229 regress; all test level labels stay
the same. Valid L3 rises from 0.7186 to 0.7268, but the gain disappears by L5.
In #230, moving SOS/umu to L4 delays the wrong negative flip from L2 to L4;
#173/#229 also use this explicitly non-Ames assay as decisive negative evidence.
The repair changes error distribution without improving final performance in
this single targeted replay. The linked report separates source correctness,
budgeted selection changes and model transfer behavior. This is not a new
independent replicate; the original v5 matched suite remains on its frozen source.

## Bioavailability / Skin legacy configuration restoration (2026-09-13)

`replicate_suites.bio_skin_legacy_configuration_20260913` records the requested
restoration of the v2 prompt/validator, cached historical query/neighbor tools,
query prior and None through the existing matched family runner. The isolated
package contains 1,011 rows and 4,626 level inputs; all pass the matched-input
preflight, and 32 rendered requests exactly match saved historical requests.
Skin valid #193/#206 retain the MDAM retrieval repair (five level inputs).
This is a verified runnable configuration, with no new model calls or scores.
The main v5 implementation and running DILI stage remain separate.

Final progressive Macro-F1 (old versus v5): Bioavailability valid 0.7608 versus
0.7716, test 0.7824 versus 0.7707; Skin valid 0.6429 versus 0.6114, test 0.5870
versus 0.5696. Old test values are means of three runs (SD 0.0112 and 0.0035);
v5 is one run. Old Skin valid is a pre-MDAM historical reference. This is not
a prompt-only comparison: functional-group context and priors also changed,
with additional saved numerical-tool differences in valid and one missing
Bioavailability test neighbor tool result. The registry links the complete
input comparison, limitations, frozen configuration and execution commands.

## Reasoning prompt revision (2026-09-13)

`diagnostics.reasoning_prompt_revision_20260913` records `progressive_evidence_revision.v5`:
shared instruction cleanup, empirical-claim source boundaries, decisive analog-transfer
explanations, and old-error correction without mandatory new-card citations. Existing
output fields and Bioavailability high/low rules are preserved. No model calls or new
benchmark scores accompany this implementation; historical results retain their original
prompt profiles. Validation and scope are recorded in the linked receipt.

## Functional-group tree diagnostic (2026-09-13)

`diagnostics.functional_group_tree_20260913` records the completed tool/context change
and its bounded Flash paired replay: three previously inspected error cases, two runs
per arm, 32 level outputs, 52 requests including validation retries, zero final failures.
At the reviewed levels, correct labels were unchanged: BBB #343 L3 2/2 versus 2/2,
DILI #345 L4 0/2 versus 0/2, Carcinogens #104 L1 0/2 versus 0/2. Both arms avoided
BBB's historical AZD3759 identity error; DILI's sultam error and Carcinogens' enone
error persisted. One additional fresh single-prior pair (two calls) reproduced the
enone error in both arms without inheriting the old prior.

The experiment used `progressive_functional_group_tree.v3`. The subsequent code
review fixed match-specific hierarchy reduction and cached-tool failure handling;
that review introduced v4 with tool cache namespace `tool-service-2026-09-13-v5`. All 167
related tests passed, and the review checked all 43 diagnostic molecules against
the saved inputs: numerical properties and complete tool text were unchanged.
The original experiment keeps its v3 identity; no new model run or benchmark gain
is claimed. The registry links the review receipt, preceding source trace audit,
and frozen execution snapshots; the three one-off scripts are no longer runnable
entrypoints. Prior text still embeds the query tree, while progressive/full-flat
expose independent query/neighbor tree fields through the same batch tools.

## DILI and Carcinogens: completed cleaning cycle

Both tasks retain Starling-only gold_v4 and the final reviewed retrieval snapshots in
`data/starling_data/<task>/retrieval_final/`. Scaffold and random use the same frozen
cohorts: DILI 3,220/402/402 and Carcinogens 3,754/469/469. L1-only heldout prefiltering,
L2 retention and query-time disjoint are unchanged. The final-source publication
receipt is `data/starling_data/retrieval_final_cleanup.json`.

The completed pre-prompt-v5 evaluations and figures remain historical registry entries:

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
random counts are 828/103/103 and 1164/150/150. The counts and commands in this section describe gold_v3 only. Current gold_v4
and restoration entrypoints are documented in
[`NEW_TASK_SOURCE_DATA.md`](../common/starling/NEW_TASK_SOURCE_DATA.md).

DILI scaffold valid/test names were exchanged at the user's request **after
viewing both cohorts**. Within that historical cohort, renamed test is former valid; it is not an unseen test
or a model improvement. Train, random, labels and heldout union are unchanged.
Use `replicate_suites.dili_scaffold_swapped_4_2` and
`outputs/paper/starling_conditioned_dili_scaffold_swapped_4_2_v1/` for the gold_v3
renamed results. Immutable trace directories retain their original names. The retained
`outputs/paper/starling_conditioned_dili_scaffold_swapped_4_2_v1/reuse_receipt.json`
and baseline reuse receipt bind predictions to those renamed rows. Their original
split-rename receipt path is historical and no longer exists in the active gold_v4 cohort.

### Agent and baseline scope

`replicate_suites.dili_carcinogens_scaffold_4_2` is complete: eight
task/organization/subset cells, 11,620 level predictions and 830 shared fresh
single/None pairs, with zero remaining failures. Each setting has one Flash run
(no repeat SD), using Hosted DeepSeek-V4-Flash-0731, matched prepared evidence,
4/2 cards, at most 256 requests per organization and 1,024 total. Failed-response
six-way races remain within those budgets. Valid checks preceded test execution.
Recovery and frozen-surface receipts are linked in the registry; no scientific
setting changed during recovery. Random and Pro runs for these tasks are absent.

Historical gold_v3 final-level macro-F1:

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

Historical gold_v3 test macro-F1, mixed → Starling-only training:

| Baseline | DILI | Carcinogens |
|---|---:|---:|
| MiniMol head | 0.7456 → 0.5657 | 0.6204 → 0.4809 |
| MiniMol KNN condition | 0.8243 → 0.4148 | 0.7028 → 0.4782 |
| MiniMol KNN all | 0.7031 → 0.4494 | 0.4870 → 0.4593 |
| Morgan KNN condition | 0.8349 → 0.4246 | 0.7000 → 0.4564 |
| Morgan KNN all | 0.7148 → 0.4246 | 0.6134 → 0.4583 |

Historical baseline entries are `matched_baselines.dili_carcinogens_scaffold`,
`matched_baselines.dili_starling_only_scaffold_swapped`, and the combined suite's
Starling-only ablation entry. Original pre-rename plots and baseline directories
are historical views. Carcinogens mixed-training head AUROC is 0.7061 pooled,
but 0.4974 within Starling and 0.5086 within TDC; pooled performance does not
establish strong discrimination inside either source.

### Starling-label test subset and figures

Filtering the historical gold_v3 metadata by **`label_source=starling`** requires no new inference,
training, retrieval or split. This is row-level label provenance, not removal of
all molecules that also occur in TDC.

| Historical gold_v3 test subset | Rows (+ / −) | None | Progressive L7 | Full-flat L7 | Starling-trained head |
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
predictions to the renamed gold_v3 DILI rows without changing their values.

### Trace and source audit limits

`outputs/paper/analysis/dili_test_evidence_decline_v2/REPORT.md` describes **former
test/renamed valid in gold_v3**: 1,442 outputs, 721 matched input pairs, 10 harmful and four
beneficial None-to-L7 progressive flips, with only one flip after L2. Three
Bosentan conditions wrongly transfer clazosentan clinical observations at L2;
full-flat L7 distinguishes the compounds. Other cases show exposure/endpoint
transfer and an internally contradictory diphenhydramine card. These are
single-run descriptions, not causal ablations or repaired source claims.

Frozen-surface audits cover both heldout suites (415 queries × seven levels each).
The historical direct-alias guard included exact reviewed hold `card_04953d07291cbd0c`;
replacement-card and real-request audits are preserved in the original suite.
The unused direct-alias filter was retired; its historical receipts remain evidence
of the original experiment, not a current filtering policy. These are bounded Codex reviews of supplied cards, not exhaustive human-expert
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

Ames scaffold valid/test names were formally exchanged on explicit user request
on 2026-09-08. Current test is the former 274-row validation cohort; it has already
been inspected and is not an untouched test set. Train, random, gold labels and
the heldout identity union are unchanged. The canonical rename receipt is
`data/conditioned_benchmark/Ames/provenance/scaffold_valid_test_swap_20260908.json`.
Current test Flash (three runs per organization) and Pro (one per organization)
are registered in `replicate_suites.ames_scaffold_test_swapped_4_2` and
`replicate_suites.ames_scaffold_test_swapped_pro_0813_4_2`. All 10,960 level outputs
passed cohort/lineage and recomputed-metric checks with zero failures: 24 Flash
outputs were rerun for q57 L2–L5, and 10,936 outputs were exactly reused. Flash
predictions and scores did not change after this source repair. Flash mean
L1–L5 macro-F1 is 0.7323/0.7458/0.7483/0.7532/0.7477 (progressive) and
0.7228/0.7556/0.7536/0.7464/0.7543 (full-flat). Five baselines were migrated by
exact ordered molecule-condition/label equality and retain their original scores.
The current four-task Flash/Pro performance and full resource figures are under
`outputs/paper/analysis/conditioned_scaffold_test_flash_vs_pro_4tasks_swapped/`.
All 76 curve points and 28 references passed comparison checks; the original
BBB/Bioavailability/Skin numeric fields are unchanged. Both images explicitly
disclose the Ames split rename. The existing plotter now uses up to 16 processes
for large trace collections while retaining its complete contract checks.

Before the rename, Ames scaffold-valid Flash completed three fresh 4/2 runs per organization under the
test prompt and retrieval preceding the nitroso structure repair: all 8,220 level outputs passed
completion checks with zero failed queries. Execution used at most three
simultaneous 256-slot runners, with width-six racing only after failure. The
None control was refreshed once under the current prompt; unchanged
structure-only priors and tools were reused. See `replicate_suites.ames_scaffold_valid_4_2_replicates` and its
launch/completion receipts for live status. The previous valid run cannot count
toward these repetitions because its task instructions and source lineage differ.
These historical performance and complete resource figures are under
`outputs/paper/analysis/ames_scaffold_valid_full_flat_vs_progressive_3runs/`;
all curves show three-run means ±1 sample SD and all five 274-row baselines.
The seven-record nitroso structure correction was replayed in the current test
suite above. Original Flash test is now current valid and still requires its own
selected-input audit before reuse. Frozen gold and matched baselines are unchanged.

Before the rename, Ames scaffold-valid Pro 0813 completed one matched full-flat and one progressive
run through OpenRouter. The seven-record nitroso structure correction was then
applied to both methods: all 1,370 selected inputs were compared, only q57 L2–L5
changed, and eight new level calls completed successfully. The other 2,732 outputs
were reused with exact equality checks; original Pro single/None priors and the
prompt remain unchanged. All 2,740 level outputs, recomputed cohort metrics and
current input/index hashes passed validation, with zero failures. The repair used
concurrent 256-slot pools (512 total) and failure-only width-six racing.
None macro-F1 is 0.6806. Progressive L1–L5 is
0.7174/0.6998/0.7032/0.7066/0.7002; full-flat is
0.7339/0.7303/0.7572/0.7292/0.7122. These are single-run results.
Progressive q57 L4/L5 changed from incorrect positive to correct negative;
full-flat labels are unchanged. Completion validation and corrected frozen inputs are registered under
`replicate_suites.ames_scaffold_valid_pro_0813_4_2`.

The historical Ames scaffold-valid progressive 4/2 run completed all 274 rows with zero
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
Hosted served `deepseek-ai/DeepSeek-V4-Flash-0731`; the requested per-task
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

The per-neighbor target-label pilot showed no gain on 16 Bioavailability validation
parents and increased invalid outputs. Its code, CLI and pilot artifacts were
removed; only the [compact retirement receipt](receipts/neighbor_task_assessment_removed_20260912.json)
remains. It is not a formal result or an active alternative.

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
