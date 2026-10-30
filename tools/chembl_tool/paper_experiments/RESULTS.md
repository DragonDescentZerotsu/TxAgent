# Current paper results and artifact status

Updated: 2026-10-18.

`current_conditioned_results.json` is the authority for result roots and freshness.
Source reconstruction is documented in `CURRENT_STARLING_RETRIEVAL.md`; the retrieval
protocol is in `ASSAY_LEVEL_RETRIEVAL.md`.

## Indirect scaffold-disjoint follow-up (2026-10-13)

GPT-6 Luna via OpenRouter completed and passed verification on 2026-10-14:
4,273 predictions / 12 cells, no missing predictions or record-budget shortfalls.
Six-task mean Macro-F1: Conditioned 0.6543 parent → 0.6297 scaffold;
TDC 0.6771 parent → 0.6645 scaffold (current seed-42 Skin/Carcinogens).
See the [protocol and verification](../../../outputs/paper/indirect_scaffold_disjoint_luna_20261014/README.md)
and [complete scores](../../../outputs/paper/indirect_scaffold_disjoint_luna_20261014/RESULTS.md),
registered as `indirect_scaffold_disjoint_luna`. The
[updated task and mean figures](../../../outputs/paper/indirect_scaffold_disjoint_luna_20261014/figures/README.md)
include DeepSeek and Luna scaffold results; Ours Direct+Indirect and Gemini naive
Direct+Indirect remain omitted. Historical transport/provider and inference
variation remain comparison limits.

User-requested Conditioned/TDC six-task test comparison, Direct 50 + Indirect 50
Neighbor-fill, OpenRouter `deepseek/deepseek-v4-flash-0731`: completed and verified
on 2026-10-14, with 4,273 predictions / 12 cells and no missing predictions or
record-budget shortfalls.
Only indirect query-relative exclusion changes; reference prompt adapters,
generation parameters, priors and Direct records are preserved. Six-task mean
Macro-F1: Conditioned 0.6784 parent → 0.6575 scaffold; TDC 0.6916 parent → 0.7011
scaffold. Historical serving providers may differ, so comparisons are descriptive.
See registry `budgeted_evidence_protocol.indirect_scaffold_disjoint` and the
[experiment protocol and run plan](../../../outputs/paper/indirect_scaffold_disjoint_20261013/README.md)
and [complete scores](../../../outputs/paper/indirect_scaffold_disjoint_20261013/RESULTS.md).

## Scaffold follow-up maintenance (2026-10-18)

The [scoped maintenance receipt](receipts/indirect_scaffold_maintenance_20261018.json)
records review of the DeepSeek/Luna scaffold exclusion, cache/manifest binding,
historical rendering option, saved-result auditor and figure exclusions.
All 8,546 predictions / 24 cells were rechecked against prepared request hashes,
models, labels and 50/50 budgets; recomputed Macro-F1/accuracy and six-task means
match the saved scores and registry. Frozen source archives and figure input/output
hashes passed verification. Scores and published figure files are unchanged.

One bounded plotter fix rejects ML groups in `excluded_bar_settings`, which could
otherwise leave empty ticks; this option is for LLM backbones, while ML groups use
`baseline_settings`. The four relevant test modules passed 136 tests; after the
fix, all 11 targeted bar tests passed, including the new invalid-ML check.
The [entrypoint guide](README.md#scaffold-disjoint-indirect-follow-up) documents
the shared family runner, original plotter, admission broker and single saved-result
auditor; no maintained entrypoint was added.

All 58 completed top-level console logs were byte-verified into the two experiment
`execution_history.zip` archives; two archived temporary controllers were removed
from `/tmp`. Structured receipts and all requests, predictions, source/prior/cache
dependencies remain. The earlier DeepSeek-only figure is explicitly historical;
the latest follow-up includes DeepSeek and Luna. Gemini scaffold inference has not
run; the original Gemini parent-disjoint baseline remains separate. This scoped
review does not complete the broader release/independent-environment work.

## Storage cleanup (2026-09-30)

The `runtime migration receipt` (local-only historical record)
records offline SQLite archives and the shared MiniMol runtime. The subsequent
`build-cache cleanup receipt` (local-only historical record)
records 203 verified archives: 112 reproduction dependencies (44.15 GiB compressed)
and 91 conservative recovery files (51.75 GiB compressed). Confirmed obsolete derived
files, shared duplicates and scratch were removed; the local build-cache directory
was cleared, freeing 646.02 GiB. These are completion-time measurements.

Archive locations, provenance and restoration guidance are in the
[cache audit](../../../outputs/paper/local_build_cache_audit_20260930/README.md).
Every retained file passed original and full-decompression SHA-256/size checks.
This verifies archived bytes, not a fresh model run or complete historical replay.
No paper scores or benchmark labels changed; the molecular tool service remains stopped.
Next work is [release preparation](README.md#release-preparation); the review boundaries below remain open.

## Maintenance review (2026-09-29)

[Review receipt](receipts/paper_maintenance_20260929.json) records the code, result,
registry, documentation and live manuscript checks. All 306 baseline cells were
recomputed from saved predictions; the 198-cell ML Appendix and all Ours/naive-LLM
appendix values match their canonical tables. Registry means now include Gemini.
The manuscript TDC panel had reversed the two MiniMol+indirect training-pool bars;
it now uses the registered panel, with its previous PDF archived. No scores changed.

Open boundaries: the manuscript's Direct=10 molecules versus the naive baseline's
Direct=50 records remains unchanged at the user's request, pending reconciliation.
Ours scores remain user-reported; local component-ablation run provenance is not
registered. The literature MiniMol matched control increases from 0.5897 to 0.5923;
the manuscript's decrease holds against the displayed Direct group upper bound, so its
matched-control wording still needs clarification. Seven historical registry
references share one missing gold-v3 DILI split-swap receipt; they are not current
gold-v4 inputs. The review does not certify missing historical evidence or reopen
the deferred Gemini cost/provenance audit.

The **Skin auxiliary evidence correction (2026-09-25)** is complete and verified:
[9 affected cells](../../../outputs/paper/skin_indirect_level_mapped_20260925/README.md)
across all three pools, 45 MiniMol seed fits and 150 RF seed fits, using upstream's
release-level-mapped source (`59c05729`; 3,287 rows / 67 buckets before heldout
exclusion). Current TDC seed-42 inputs are retained. All other baseline predictions
are unchanged; the RASAR six-feature cache matches exactly. Registry:
`skin_indirect_level_mapped`. The unified 198-cell Appendix, six current panels and three composite figures
and manuscript now resolve these corrected Skin auxiliary scores. The indirect
side of the original RASAR ablation report remains a historical broad-pool comparison.

The **RASAR direct-only control (2026-09-25)** is complete and verified: 36 report
cells across the six conditioned tasks and both TDC training pools, with 30 unique
model suites / 180 candidates / 900 seed fits. Inputs are exactly the first six
main-label/Morgan-similarity features of the matched current RASAR+indirect caches.
Skin/Carcinogens use the current seed-42 TDC test cohorts (82/56 rows). The original
validation protocol is retained. [Current direct-only versus indirect comparison](../../../outputs/paper/retrieval_forest_direct_20260925/RESULTS.md)
and [full-precision scores](../../../outputs/paper/retrieval_forest_direct_20260925/all_scores.csv);
registry `retrieval_forest_direct`. The new RASAR rows are also included in the unified baseline CSV/JSON, Appendix, and both final figure layouts.

The [six-task TDC split audit](../../../outputs/paper/skin_tdc_upstream_split_20260924/SPLIT_AUDIT.md)
identifies different latest upstream test cohorts for Skin and Carcinogens.
BBB/Oral/Ames/DILI match local test identities and labels after standardization;
train/validation alignment remains a separate check. The Skin ML rerun on
upstream's exact split is complete and verified: 18 ML report cells, including
Macro-F1 and AUROC, registered as `tdc_skin_upstream_split`. Its
[results](../../../outputs/paper/skin_tdc_upstream_split_20260924/RESULTS.md) must not
be mixed with seed-1 local LLM scores. Its six auxiliary report cells are now
superseded by `skin_indirect_level_mapped`; validation selects lambda
0.6 / 1 for the corrected TDC-only / mixed MiniMol models.
Frozen/main-only and KNN results remain in the seed-42 suite.

The Carcinogens rerun is also complete and verified: 18 cells across TDC-only
and TDC + Starling, with [Macro-F1/AUROC results](../../../outputs/paper/carcinogens_tdc_upstream_split_20260924/RESULTS.md).
Registry: `tdc_carcinogens_upstream_split`. Raw TDC scaffold seed 42 reproduces all
56 upstream v2 test identities and labels; train/valid (196/28) are reconstructed
and not independently verified against his unpublished v2 files. Both main and
auxiliary heldout-parent checks pass, as do added-Starling scaffold and RF
query-self exclusions. Validation selects lambda 1 / 1.

The current 198-cell ML Appendix and the manuscript Appendix now use these two
seed-42 successors, nine corrected Skin auxiliary cells, and the 36 added RASAR
cells (198 cells / 396 metric values). Export:
`outputs/paper/llm_vs_standard_baselines_20260921/build.py --seed42-appendix-output
outputs/paper/llm_vs_standard_baselines_20260924`. The four fresh OpenRouter
Neighbor-fill suites completed and verified all 828 predictions (12 cells).
They are registered as `tdc_seed42_neighbor_fill`; see [results](../../../outputs/paper/tdc_seed42_neighbor_fill_20260924/RESULTS.md) and their
[run plan](../../../outputs/paper/tdc_seed42_neighbor_fill_20260924/README.md).

TDC DILI now retains the conditional-transfer prompt with hidden transfer-score
lines **for TDC only**. On the same 96-query upstream v2 cohort, Macro-F1 is
0.784076 (76 correct), versus historical Mixed 0.760436 (74) and Direct 0.829712
(80). The mechanism/concise ablations scored 0.748376 (73) / 0.774168 (75) and
were rejected; the combined arm was not run. All 288 replay outputs are verified.
The identity repair is already present in the inspected v2 records. These are
post-test diagnostics with ten L1 plus fifty indirect records, not the fixed
50/50 paper baseline. See [the current TDC handoff](shared/TDC_DILI_HANDOFF.md)
and registry `diagnostics.tdc_dili_prompt_20260924` for assets, exact metrics,
local entrypoints and retention. Gold-v1 and other tasks are unchanged.

Historical TDC BBB test development with the retired v1 filter completed
2,436 DeepSeek/OpenRouter predictions
with shared concurrency 512 and retry race width 6. The one-pass balanced20 arm
improves matched Direct Macro-F1 0.762083 → 0.792696 (accuracy 0.830049 → 0.849754;
24 corrected / 16 damaged). The two-stage indirect increment is much smaller:
0.798533 → 0.802925, one net correct answer. These test-tuned results do not replace
the frozen baseline or paper figure; original Direct accuracy remains higher.
The requested matched budget-50 follow-up is complete: Macro-F1 0.795046,
accuracy 0.854680, 20 corrected / 18 damaged versus balanced20. Actual records
are 24–50 (median 48) under the unchanged donor cap and frozen candidate pool;
the observed gain over budget 20 is small (+0.002350 Macro-F1).
Registry: `tdc_bbb_indirect_test_iteration_20260924`;
[portable results and current filter status](../tasks/bbb_martins/INDIRECT_ABLATION.md).
The current assay-aware v2 filter has offline verification only; no current-filter
LLM score is available.

Gemini 3.8 Flash is recorded as the completed naive LLM paper baseline at the
user's direction on 2026-09-25: 12,819 predictions / 36 cells, conditioned and
TDC six-task test, None / Direct 50 / Direct 50 + Indirect 50 (Neighbor-fill).
TDC Skin/Carcinogens use current seed-42 inputs. Medium reasoning, 16,384 tokens,
upstream adapter v2 and cached priors are unchanged. Ordinary API execution reused
629 compatible Batch outputs; source and reuse receipts remain preserved.
Counts, input hashes/labels and scores are checked. Further provenance and cost
reconciliation are explicitly deferred by the user, not marked complete.
Registry: `gemini38_flash_api_b50`;
[baseline tables, full-precision CSV and protocol](../../../outputs/paper/gemini38_flash_api_b50_20260925/README.md).
`llm_vs_standard_baselines.additional_baseline_sources` pins this result; both
current task layouts and the separate mean figures include Gemini.
Old Batch cancellation attempts returned 404; the stopped local Batch suite is
historical (`gemini38_flash_batch_b50`), not the current baseline score source.

The [current paper baseline comparison](../../../outputs/paper/llm_vs_standard_baselines_20260924/RESULTS.md)
now maintains two final pastel-B **bar** layouts: vertical task panels (`mixed`) and aligned horizontal task panels (`horizontal`).
Both now show tasks only. A separate `mean_results.pdf` places literature-derived
means on the left and TDC means on the right, using vertical bars. This is the
accepted mean design, extended to Gemini: titles above one shared color-legend row
and a separate centered abbreviation row, with no bar values or y-axis titles.
The left-line/right-point layout and early stage-grouped bars are retired.
Both final layouts are generated by the shared renderer and exporter; commands,
inputs, backbone bindings and current artifacts are documented once in the
[plotting guide](README.md#current-comparison-figures).

Task means weight the six tasks equally. ML groups retain per-task descriptive
test maxima, separately by training pool; averaging these maxima is not global
model selection. There are 120 / 150 task marks and 20 / 25 available means, with
all 120 / 186 baseline source rows and 198 ML appendix cells retained. Ours is
bound separately to DeepSeek, Luna, and Gemini and shares each backbone's None. Missing
results are not zero-filled or averaged over a smaller task set. The 20260925
[shared Ours record](../../../outputs/paper/llm_vs_standard_baselines_20260924/ours_reported_results_20260925_three_backbones.json)
contains all 72 user-reported scores. Direct + Indirect means are 0.7209 / 0.7997
for DeepSeek, 0.6897 / 0.7871 for Luna, and 0.7360 / 0.8154 for Gemini
(literature-derived / TDC). Gemini adds 36 completed baseline cells / 12,819
predictions and 24 user-reported Ours cells. Summary scores and ordered cohort
labels were checked; the separately deferred provenance/cost audit stays deferred. Complete
task scores and Direct means are in the [current results tables](../../../outputs/paper/llm_vs_standard_baselines_20260924/RESULTS.md#complete-ours-scores)
and manuscript Appendix `app:ours-results`, both derived from that shared source.
This is a user-reported update, not new prediction-level validation.
The [conditioned figure](../../../outputs/paper/llm_vs_standard_baselines_20260924/conditioned/baseline_bars_mixed.png)
uses aligned DeepSeek adapter-v2 results with exactly the Luna messages.
The [TDC figure](../../../outputs/paper/llm_vs_standard_baselines_20260924/tdc/baseline_bars_mixed.png)
uses fresh matched adapter-v2 DeepSeek/Luna inference for Skin and Carcinogens.
The other four tasks retain historical DeepSeek adapter-v1 inference; records/prior
match Luna after v2 exact-field deduplication, but original rendering differs.
Generation settings remain model-specific; these are descriptive comparisons.
The 20260921 tool-visible/Direct-3 comparison is historical and does not supply
the current conditioned curve. The 20260923 Skin/Carcinogens curves are also
historical seed-1 results; the current 20260924 export incorporates the new ML
and 828-prediction LLM runs. The plotter itself performs no training or inference.

The original Luna Batch suite completed 12,819 predictions in 36 cells, with zero final failures and
$22.90500961 total Batch charges including repairs. Six-task mean Macro-F1 for
None / Direct / Direct + Indirect: conditioned 0.617613 / 0.624771 / 0.654253;
historical seed-1 TDC 0.656558 / 0.674375 / 0.673375. Current means are in
`llm_vs_standard_baselines.figures.*.task_mean_macro_f1`. Batch run/audit receipts:
[adopted Luna baseline](../../../outputs/paper/gpt6_luna_batch_b50_20260922/README.md).

The [ML appendix](../../../outputs/paper/llm_vs_standard_baselines_20260924/ML_APPENDIX.md)
pairs test Macro-F1 and AUROC for every task and method: 66 conditioned cells,
66 TDC-only cells and 66 TDC + Starling cells, including encoder/indirect controls
and both RASAR / RASAR+indirect retrieval policies.
All 198 AUROCs are recomputed from saved continuous prediction scores; full
precision and prediction paths are in the accompanying `ml_appendix.csv`.
Use that CSV for manuscript tables and round only at typesetting; artifact scope,
metric definitions and regeneration are documented in
[Appendix result sources](README.md#appendix-result-sources).

The [current ML Appendix](../../../outputs/paper/llm_vs_standard_baselines_20260924/ML_APPENDIX.md)
contains all 36 RASAR+indirect task/pool/policy cells. The source suites retain
validation selections, feature and donor audits, and resource receipts; Skin
uses `skin_indirect_level_mapped`. The original `retrieval_forest` registry entry
is provenance and its rendered report now routes to the current table.

The [six-task conditioned standard baseline table](../../../outputs/paper/conditioned_minimol_val_selected_20260921/RESULTS.md)
is now complete: MiniMol heads use official task parameters, per-seed val BCE
checkpoint selection and ensemble val Macro-F1 thresholds. BBB/Oral reuse the
adopted frozen controls; Skin/Ames/DILI/Carcinogens add 20 verified seed runs.
Together with the four fixed-k=3 KNN variants, all 30 cells cover
10,240 test predictions. Current head Macro-F1 is BBB 0.6309, Oral 0.6276,
Skin 0.5838, Ames 0.6473, DILI 0.5999, Carcinogens 0.5431 (mean 0.6055).
Registry: `conditioned_minimol_val_selected`. Historical CV head values are kept
as raw before/after references; head parameters also differ for some tasks.
The TDC-only/mixed standard heads were already complete and their saved inputs
and artifact hashes were rechecked unchanged. The subsequent four-task encoder/indirect
completion is reported below, with these frozen controls reused unchanged.

The original TDC LLM settings and validation-selected frozen-encoder heads, with
fixed-k=3 KNN, are consolidated in
[`comparison_all_settings.md`](../../../outputs/paper/tdc_minimol_val_selected_20260919/comparison_all_settings.md),
with English descriptions and CSV/JSON beside it: ten unique LLM settings plus
ten ML report settings (18 distinct results after identical TDC-only KNN aliases).
All task Macro-F1/accuracy values and ML AUROC were recomputed from saved
predictions against the same 2,225 test rows; LLM AUROC is not reported.
Registry: `tdc_all_settings_comparison`. The original all-task Bioavailability
head comparison remains at its historical Direct-3 experiment root.

In that frozen-head/KNN reference table, the highest observed mean Macro-F1 is Direct-3 + Indirect / Molecule-cap (0.7275),
followed by Group-balanced (0.7268). The strongest ML mean is TDC-only
MiniMol KNN (0.7133), a descriptive gap of 0.0141. Direct-3 alone (0.7196) exceeds
its exact-neighbor Morgan KNN (0.6994) by 0.0202. Added indirect evidence is
external information; the full set of comparisons is not uniformly data-matched,
and no significance claim is made.

## Paper KNN protocol

Both MiniMol embedding KNN and Morgan KNN use the original fixed k=3,
train-only references, existing condition policies and threshold 0.5. There is
no validation optimization of k. TDC LLM Direct-3 reuses the exact Morgan
neighbors and labels; MiniMol embedding neighbors can differ. MiniMol trainable
models retain the latest validation-selected checkpoints, thresholds and
auxiliary λ. Original KNN predictions and all current trained-model predictions
are preserved byte-for-byte.

## Adopted MiniMol + indirect assay

Current conditioned and TDC scores, including Skin's corrected auxiliary pool,
are maintained only in the [198-cell ML Appendix](../../../outputs/paper/llm_vs_standard_baselines_20260924/ML_APPENDIX.md)
and its full-precision CSV. The [Skin source correction](../../../outputs/paper/skin_indirect_level_mapped_20260925/README.md)
records its three validation-selected weights and verified inputs. The older
six-task and seed-42 experiment roots retain model provenance; their superseded
Skin scores and derived means are omitted from current documentation.

All models use validation BCE for checkpoints and auxiliary weight selection,
then validation Macro-F1 for the threshold. Auxiliary pools exclude heldout
parent/connectivity identities; shared scaffolds are allowed. No train+valid
refit or test-based model selection. Protocol and maintained commands:
[MiniMol README](../../../baselines/minimol/README.md#minimol--indirect-assay).

## TDC encoder finetuning and indirect assay (2026-09-22)

Both TDC-only and TDC + Starling main-label pools now have all three MiniMol
controls for six tasks: 240 new encoder seeds, 60 reused frozen seeds and 36
verified ensembles. Main rows and task head/condition settings match the frozen
controls. Auxiliary labels exclude each task's entire TDC valid/test parent and
connectivity union; auxiliary scaffold overlap is allowed. Original main TDC
overlap (BBB 4 rows, Ames 1) is preserved and reported separately. Starling main
labels retain the existing parent/scaffold exclusion.

| Pool | Frozen MiniMol | Main-only finetuned | + indirect assay |
|---|---:|---:|---:|
| TDC-only | 0.6633 | 0.6612 | 0.6542 |
| TDC + Starling labels | 0.6957 | 0.6614 | 0.6537 |

Values are six-task mean test Macro-F1. Each checkpoint and λ use TDC validation
BCE; ensemble thresholds use validation Macro-F1. λ candidates are {0.1,0.6,1};
no test-based selection or train+valid refit. Full task metrics, chosen λ values,
parameters, provenance and resources: [two-pool report](../../../outputs/paper/tdc_minimol_indirect_assay_20260922/RESULTS.md).
Registry: `tdc_minimol_indirect_assay`. The earlier head/KNN comparison remains
the frozen-encoder reference; the current LLM comparison table includes the new
encoder rows. KNN and LLM predictions are unchanged.

## Current TDC ML: external-validation selection (2026-09-19)

**Complete and verified:** all 12 MiniMol MLP cells rerun, covering both training
pools and six TDC test cohorts. Use the same official task hyperparameters,
Bioavailability fallback for Skin/Carcinogens, and five seeds. Each seed selects
its minimum TDC validation BCE checkpoint from 25 training epochs. The ensemble
probability threshold maximizes TDC validation Macro-F1. Only train rows receive
training updates; there is no CV or train+valid refit. Test labels do not select
checkpoints or thresholds. The mixed pool retains the same filtered Starling
train labels, with TDC-only valid/test.

| Method | TDC-only mean Macro-F1 | Mixed mean Macro-F1 |
|---|---:|---:|
| MiniMol MLP | 0.6633 | 0.6957 |
| MiniMol KNN | 0.7133 | 0.6405 |
| MiniMol KNN condition-first | 0.7133 | 0.6674 |
| Morgan KNN | 0.6994 | 0.6646 |
| Morgan KNN condition-first | 0.6994 | 0.7060 |

MLP means change from the official-parameter CV profile's 0.7271 → 0.6633
(TDC-only) and 0.7196 → 0.6957 (mixed). This comparison changes checkpoint
selection (including its criterion), calibration data and refitting protocol;
it does not isolate a single causal effect. The reported profile is the one
requested before these test results, not a per-task old/new winner. KNN is
unchanged. Verification covers 60 report cells / 22,250 test prediction rows,
including equivalent TDC-only KNN policy aliases; all 12 validation histories,
selected epochs and validation-optimal thresholds were audited.

[Full task F1/accuracy/AUROC and selected epochs](../../../outputs/paper/tdc_minimol_val_selected_20260919/RESULTS.md),
[CSV](../../../outputs/paper/tdc_minimol_val_selected_20260919/all_scores.csv).
Registry: `tdc_minimol_val_selected`. Historical profiles remain linked below.
The current profile is retained by user decision; no per-task test-score selection
among historical heads is used. AUROC uses probabilities and is threshold-independent.

## Historical TDC ML profiles

Earlier all-task Bioavailability-parameter heads and task-specific `SWEEP_RESULTS`
heads used train-only scaffold CV and OOF threshold selection. Their exact scores,
commands and verification remain in the original experiment roots:

- [Task-specific CV profile](../../../outputs/paper/tdc_minimol_sweep_results_20260919/RESULTS.md), registry `tdc_minimol_official_sweep`.
- [Original TDC-only and mixed comparison](../../../outputs/paper/tdc_knn3_baselines_20260919/comparison_all_settings.md).

These head scores are historical; current comparisons use the validation-selected
profile above. KNN predictions are unchanged across head profiles.

## TDC matched Morgan KNN Direct-3

Complete and independently verified: **8,900 / 8,900 predictions**, 24 cells over
six tasks and 2,225 TDC test rows. Every Direct context uses the same three saved
Morgan KNN training neighbors/labels; complete Indirect selections and candidate
audits match the previous corresponding strategy. All 2,225 None messages match
byte-for-byte and its predictions are reused, as are the TDC-only ML baselines.
New evidence contexts use adapter v2; previous 50/50 results used v1 rendering.

Equal-weight six-task means (one LLM run; no significance claim):

| Setting | Macro-F1 | Accuracy |
|---|---:|---:|
| None (reused) | 0.6316 | 0.6735 |
| Direct-3 | 0.7196 | 0.7608 |
| Direct-3 + Indirect / Neighbor-fill | 0.7168 | 0.7563 |
| Direct-3 + Indirect / Molecule-cap | 0.7275 | 0.7691 |
| Direct-3 + Indirect / Group-balanced | 0.7268 | 0.7655 |
| Morgan KNN (same three neighbors) | 0.6994 | 0.7574 |
| MiniMol KNN | 0.7133 | 0.7662 |

Direct-3 exceeds same-neighbor Morgan KNN by 0.0202 mean Macro-F1; adding
Molecule-cap Indirect adds 0.0079. Indirect improves four tasks but lowers BBB and
DILI. Excluding the retained same-parent cases (BBB #382, Ames #1146) gives
Direct-3 0.7194, Molecule-cap 0.7271 and Morgan KNN 0.6991. Current MiniMol MLP
comparisons use the validation-selected table above; the original sensitivity
artifact includes historical CV head scores.
Registry `budgeted_evidence_protocol.tdc_knn3_baselines` owns commands, recovery,
verification, comparison CSV/JSON and the common-cohort sensitivity check. Direct
is matched evidence; Indirect additionally consumes external Starling evidence.

## TDC scaffold test, fixed 50/50

The separately approved TDC evaluation is **complete_verified**: 13,350/13,350
LLM predictions over 2,225 test rows, with input/index/prior hashes, actual
prompts, output validity and metrics independently checked. No tool failures.
All six tasks' ML outputs are also independently verified (6,675 predictions
across three unique methods). Both condition-first KNN variants reproduce their
ordinary counterparts exactly, including probabilities and selected neighbors;
the five report names therefore represent three methods on these inputs.

Equal-weight six-task means (one LLM run; no significance claim):

| Setting | Macro-F1 | Accuracy |
|---|---:|---:|
| None | 0.6316 | 0.6735 |
| Direct / Neighbor-fill | 0.6749 | 0.7245 |
| Direct / Molecule-cap | 0.6782 | 0.7317 |
| Direct + Indirect / Neighbor-fill | 0.6816 | 0.7235 |
| Direct + Indirect / Molecule-cap | 0.7116 | 0.7545 |
| Direct + Indirect / Group-balanced | 0.6822 | 0.7314 |
| MiniMol KNN | 0.7133 | 0.7662 |
| Morgan KNN | 0.6994 | 0.7574 |

Direct + Indirect / Molecule-cap has the highest observed LLM mean: +0.0334
Macro-F1 over its matched Direct baseline. Use the current combined table above
for validation-selected MLP comparisons; the original experiment table retains
its historical CV head.
This is a descriptive test comparison, not test-based configuration selection.
Bioavailability drives the strongest LLM advantage; DILI and Carcinogens remain
well below the strongest ML baseline. Ames slightly declines for Molecule-cap.

Registry `budgeted_evidence_protocol.tdc_scaffold_baselines` pins the separate
data/run roots, verification, `comparison.json` and all per-task points in
`all_scores.csv`. The final `PROGRESS.md` snapshot is complete; its monitor exits.
BBB/Bioavailability/Ames/DILI use ADMET_Group fixed test files; Skin/Carcinogens
use frozen seed-1 TDC scaffold splits. Original split membership is retained.
BBB and Ames each have one normalized parent shared by official train/test; corresponding
Direct records are excluded, while ML preserves the official training data.
LLM retrieval uses external Starling evidence plus eligible TDC train labels;
ML uses TDC training labels only. This is not a matched-data comparison.

## TDC + Starling mixed-training ML baselines

The 2026-09-19 simple augmentation control is **complete_verified**: 30 cells,
11,125 predictions across six TDC test tasks and all five ML baselines. It preserves TDC train/valid/test rows,
adds only current Starling scaffold-train labels after excluding TDC valid/test
parents and raw/normalized scaffold groups (including empty scaffolds), and uses
ordinary equal-weight rows. Added Starling counts are BBB 2,289, Bioavailability
1,521, Skin 1,698, Ames 1,394, DILI 2,482 and Carcinogens 3,477.

Current MiniMol heads use source/condition features and TDC validation selection,
with the results in the current table above. Both KNN families reuse the mixed
pool and k=3. For conditionless TDC queries, `same_condition_then_null` means
**unspecified-condition-only KNN**, not matching a TDC assay condition. Therefore
its results can differ from unrestricted KNN on the mixed pool, whereas the two
policies coincide on TDC-only data. On Ames/Carcinogens, all added Starling labels
are conditioned, so this restricted policy excludes them all.

Registry `tdc_starling_mixed_baselines` retains the data/build provenance and
original CV-head results; `tdc_minimol_val_selected` owns current head results.
This augmentation is not fully evidence-matched to LLM retrieval: added ML labels
exclude the full heldout scaffold union, while the original LLM Direct profile
excludes the current query scaffold. No new model runner was introduced.

## Record rendering deduplication (offline verification)

The 2026-09-19 adapter v2 removes only exact duplicate displayed fields from
`Additional details`. All 13,350 completed TDC prepared inputs were checked:
890,000 record occurrences reconstruct exactly from retained details and
explicitly displayed fields, and all historical messages still render byte-for-byte.
The same inputs contain 47.22% fewer prompt characters after projection
(character count, not token count). All 61 relevant regression tests passed.
This offline audit alone does not measure performance; adapter v2 was subsequently
used by the completed Direct-3 suite above. The audit verifies preservation
of the previously model-visible sanitized evidence, not upstream source completeness.
Registry `budgeted_evidence_protocol.record_rendering_deduplication` links the
verification and TDC/Starling before-and-after examples.

## Aligned scaling record inventory

All three aligned suites are complete and verified on our own six-task
conditioned scaffold benchmark: **71,600 predictions / 210 task-setting cells**.
The registry's `scaling_records_review` records the consolidated recheck of saved
prediction/manifest hashes, recomputed metrics, score CSV hashes and registered
means against the original full audits.

| Suite | Predictions | Cells | Scores | Full audit |
|---|---:|---:|---|---|
| Validation 25/50/100 | 32,688 | 96 | [all scores](../../../outputs/paper/aligned_record_scaling_valid_20260918/figures/all_scores.csv) | [verification](../../../outputs/paper/aligned_record_scaling_valid_20260918/verification.json) |
| Test, original 16 settings | 32,768 | 96 | [val/test comparison](../../../outputs/paper/aligned_record_scaling_test_20260921/val_test_scores.csv) | [verification](../../../outputs/paper/aligned_record_scaling_test_20260921/verification.json) |
| Test, Direct-fill 100 + Indirect-fill 25/50/100 | 6,144 | 18 | [comparison](../../../outputs/paper/aligned_record_scaling_test_direct100_fill_20260922/comparison_scores.csv) | [verification](../../../outputs/paper/aligned_record_scaling_test_direct100_fill_20260922/verification.json) |

**2026-09-25 split check:** these scaling suites use the conditioned benchmark,
with Skin/Carcinogens test sizes 241/469. Both current test files still match the
saved scaling input hashes. The seed-1 to seed-42 correction applies to the
separate TDC benchmark (82/56 test rows), whose completed 50/50 Neighbor-fill
reruns are registered as `tdc_seed42_neighbor_fill`. It does not invalidate or
replace the conditioned scaling scores. Any TDC seed-42 budget sweep must be
registered separately and keep the benchmark identity explicit.

**Adopted naive LLM baseline: Neighbor-fill for both Direct and Indirect**, per
the user's 2026-09-22 clarification. The fixed-budget arms are Direct=50 and
Direct=50 + Indirect=50, each with one full-flat LLM prediction; see the
[baseline contract](BUDGETED_EVIDENCE.md#adopted-naive-llm-baseline-direct-neighbor-fill).
The original val-selected Direct reference is 50; the fixed-Direct 100 follow-up
compares against Direct 100. All 25/50/100 scaling points remain recorded below.
This clarification does not change saved predictions.

Validation retains prompt adapter v1; both test suites use v2 exact-field
deduplication. The fixed-Direct 100 follow-up was user-requested after inspecting
the original test sweep. These are descriptive single-run comparisons.

## Fixed Direct 50 + Indirect 10/25/50/75/100 (2026-09-25)

Conditioned scaffold test; both pools use Neighbor-fill. Six-task equal-weight Macro-F1.

| Task | +10 | +25 | +50 | +75 | +100 |
|---|---:|---:|---:|---:|---:|
| bbb_martins | 0.6900 | 0.7064 | 0.7039 | 0.7077 | 0.6902 |
| bioavailability_ma | 0.7671 | 0.7856 | 0.8014 | 0.7671 | 0.7961 |
| skin_reaction | 0.6372 | 0.6113 | 0.6467 | 0.6227 | 0.6062 |
| ames | 0.7258 | 0.7682 | 0.7288 | 0.7268 | 0.7515 |
| dili | 0.5817 | 0.6053 | 0.6114 | 0.6349 | 0.6212 |
| carcinogens | 0.5269 | 0.5760 | 0.5783 | 0.5588 | 0.5910 |
| Mean | 0.6548 | 0.6755 | 0.6784 | 0.6697 | 0.6760 |

25/50/100 reuse the original mixed-provider run; 10/75 use OpenRouter, same named model/generation/prompt. Provider execution equivalence is not established.

All 10,240 predictions verified; 6,144 from the original sweep and 4,096 added in this extension. No significance claim from one run per setting.

Full precision, exact commands and verification: [experiment record](../../../outputs/paper/conditioned_direct50_indirect_scaling_20260925/README.md). Registry: `conditioned_direct50_indirect_scaling`.

Ours (user-provided; raw predictions not audited), six-task mean: 0.6946 / 0.7183 / 0.7236 / 0.7198 / 0.7116. Per-task pairs: [comparison](../../../outputs/paper/conditioned_direct50_indirect_scaling_20260925/RESULTS.md#ours-vs-naive-baseline).

## upstream prompt alignment and validation scaling

The 2026-09-18 adapter pins upstream `3c36c264` full-flat v5 prose, preserves our
local JSON response format, and replaces raw tool text with cached single-molecule
analysis. Six-task validation prior coverage is 2,043/2,043; 18 rendered examples
cover None/Direct/Direct+Indirect. The portable six-task prompt bundle uses
upstream's two-field output with explicit label mappings. Registry
`budgeted_evidence_protocol.prompt_alignment` points to the source map, examples
and offline receipt. The aligned scaling results below are separate from the
older tool-visible baseline suite.

The six-task scaffold validation sweep is **complete_verified**: 32,688/32,688
predictions (14,301 Direct-stage + 18,387 Indirect-stage), zero invalid outputs,
32,688 prompt-condition checks, and no record-budget shortfalls. All source,
benchmark and five-baseline lineage checks passed. Macro-F1 below averages the
six tasks equally; None = 0.6249.

| Setting | 25 records | 50 records | 100 records |
|---|---:|---:|---:|
| Direct / Neighbor-fill | 0.6779 | **0.6858** | 0.6743 |
| Direct / Molecule-cap | 0.6698 | 0.6685 | **0.6843** |
| Direct 50 + Indirect / Neighbor-fill | 0.6943 | 0.6928 | **0.6990** |
| Direct 100 + Indirect / Molecule-cap | 0.6921 | **0.7030** | 0.6941 |
| Direct 100 + Indirect / Group-balanced | **0.7005** | 0.6886 | 0.6988 |

The column budget applies to Direct in the first two rows and to additional
Indirect records in the last three. Direct budgets were selected independently:
Neighbor-fill=50 and Molecule-cap=100; Group-balanced uses the latter's Direct
context. Both retrieval pools use 128 candidate molecules, min_similarity=0;
Indirect is parent-disjoint. Query conditions and cached single-molecule analysis
remain visible, with no raw tools. Predictions are independent full-flat calls.

More records do not monotonically improve performance. The highest observed
mean is Direct-cap 100 + Indirect-cap 50 (0.7030), +1.87 percentage points over
its fixed Direct reference and +7.81 points over None. This is one validation
run per setting, without a significance claim. The separately registered
2026-09-21 test sweep uses the same grid and validation-frozen Direct 50/100,
with adapter v2 exact-field deduplication. It completed on 2026-09-22;
see the test results below and `budgeted_evidence_protocol.aligned_record_scaling_test`.

Four figures retain the previous six-task panels and right-side five ML
baselines, with shared Macro-F1 limits [0.45, 0.85]. The Direct figure compares
two selector curves over 25/50/100; each Indirect figure compares three budget
curves over None → Direct → Direct+Indirect. Figures, portable CSVs, metadata
and verification are under
`outputs/paper/aligned_record_scaling_valid_20260918/figures/` and indexed by
`budgeted_evidence_protocol.aligned_record_scaling.comparison`.

## Aligned conditioned test record scaling (2026-09-22)

All **32,768 predictions / 96 cells** completed on the six-task conditioned
scaffold test set, with zero final failed queries. This is separate from the TDC
test benchmark. Direct completed 14,336 and Indirect completed 18,432 predictions;
the last resumed phase finished at 12:40 UTC. The result root is
`outputs/paper/aligned_record_scaling_test_20260921/`.

Macro-F1 below is the unweighted mean over six tasks. None is **0.6066** on test
(0.6249 on validation). Each entry is **validation / test**:

| Setting | 25 records | 50 records | 100 records |
|---|---:|---:|---:|
| Direct / Neighbor-fill | 0.6779 / 0.6402 | 0.6858 / 0.6379 | 0.6743 / 0.6413 |
| Direct / Molecule-cap | 0.6698 / 0.6408 | 0.6685 / 0.6434 | 0.6843 / 0.6530 |
| Direct 50 + Indirect / Neighbor-fill | 0.6943 / 0.6755 | 0.6928 / 0.6784 | 0.6990 / 0.6760 |
| Direct 100 + Indirect / Molecule-cap | 0.6921 / 0.6751 | 0.7030 / 0.6734 | 0.6941 / 0.6776 |
| Direct 100 + Indirect / Group-balanced | 0.7005 / 0.6696 | 0.6886 / 0.6729 | 0.6988 / 0.6712 |

Within the original 16-setting sweep, the highest observed test mean is Neighbor-fill Direct 50 + Indirect 50
(**0.6784**), +4.05 percentage points versus its fixed Direct 50 reference and
+7.19 points versus None. Validation's highest observed combination was
Molecule-cap Direct 100 + Indirect 50 (0.7030 val, 0.6734 test). Direct-cap 100
has the highest Direct-only test mean (0.6530). None of these test rankings
changes the frozen second-stage Direct budgets or establishes statistical
significance: there is one run per setting, and the validation adapter is v1
while test uses v2 exact-field deduplication. Val/test differences therefore
cannot be attributed exclusively to the split.

Increasing the record count is not consistently beneficial. Across the original
Indirect settings, mean test Macro-F1 spans 0.6696–0.6784. Skin's highest test cell remains
Direct-cap 100 (0.6590), above every Indirect setting; aggregate gains do not
imply gains for every task. All task-level values and paired validation/test
deltas are exported in `val_test_scores.csv` (96 rows). `verification.json`
records source/input hashes, every request/response/condition check and metrics
recomputed from predictions. Reproduce with the root's `verify_results.py`.

The user-requested Direct-fill 100 + Indirect-fill 25/50/100 follow-up is
complete and independently verified: 6,144 predictions across 18 cells on the
same conditioned scaffold test. Registry: `aligned_record_scaling_test_direct100_fill`.
Both evidence selectors are Neighbor-fill; values below are Macro-F1.

| Task | Direct 100 + Indirect 25 | Direct 100 + Indirect 50 | Direct 100 + Indirect 100 |
|---|---:|---:|---:|
| BBB | 0.6989 | 0.7183 | 0.7215 |
| Oral bioavailability | 0.7817 | 0.7635 | 0.7876 |
| Skin | 0.5987 | 0.6408 | 0.6028 |
| Ames | 0.7421 | 0.7395 | 0.7374 |
| DILI | 0.5963 | 0.6388 | 0.6249 |
| Carcinogens | 0.5677 | 0.5917 | 0.5992 |
| Six-task mean | 0.6642 | 0.6821 | 0.6789 |
| Original Direct 50 + same Indirect budget | 0.6755 | 0.6784 | 0.6760 |
| Difference (percentage points) | -1.12 | +0.37 | +0.29 |

Direct 100 alone has mean Macro-F1 0.6413; adding 25/50/100 Indirect records
changes it by +2.29/+4.08/+3.76 percentage points. The largest observed mean is
100+50 (0.6821), but these are single-run descriptive comparisons. This follow-up
was explicitly requested after viewing the original test results; it does not
replace the frozen validation-selected sweep. Its `verification.json` checks all
6,144 responses, request hashes, conditions, gold labels and recomputed scores,
plus exact Direct record IDs against Direct 100 and Indirect record IDs against
the original same-budget cells. All 25 source hashes match. Full precision and
per-task differences are in `comparison_scores.csv`; reproduce with the root's
`verify_results.py`, which reuses the original sweep's cell verifier. The run
finished at 2026-09-22 19:33 UTC after reducing concurrency from 512 to 128.

## Simple LLM baselines: no-threshold validation and test complete

Both fresh six-setting suites are complete and fully verified under
`outputs/paper/llm_baselines_50_visible_no_threshold_{valid,test}_20260917/`:
12,258 validation and 12,288 test predictions. Both use `min_similarity=0` for
direct and indirect, identity-visible prompts, 50+50 record budgets and the
same conditioned scaffold cohorts. All actual prompts passed condition checks.
No predictions from the historical 0.3 suites were reused. The five ML baselines
are unchanged and matched to each subset's current rows, labels and cohort hashes.
See registry `budgeted_evidence_protocol.baseline_no_threshold` for artifacts.
These 2026-09-17 runs retained the old 10-molecule Direct candidate limit, so
Neighbor-fill Direct was below 50 for 1,038/2,043 validation queries and
1,086/2,048 test queries, while Indirect reached 50 throughout. The paired
Direct-only and Direct + Indirect arms use identical Direct sets; source counts
are recorded in registry `baseline_no_threshold.record_count_audit`.
The aligned runs removed that limit and
report full record budgets; do not equate the historical nominal 50/50 setting
with the aligned runs' actual record counts.
Offline supportive/contradictory citation statistics now cover all 24,546
completed predictions. Each suite's `citations/` contains per-query parent/card
mappings, a 324-row task/setting/role/evidence-kind summary, and a hashed manifest.
Counts deduplicate records within roles; overlaps remain explicit. Effective
parent means condition on having citations. These use the original baseline
prompt, not a upstream-v4 rerun; no model calls or prediction changes were made.

Each root's `figures/baseline_curves.{png,pdf,svg}` shows None → Direct →
Direct + Indirect, with the five ML baselines on the right and a shared 0.45–0.80
Macro-F1 axis. CSV scores and a source/condition verification receipt accompany
each figure. Points are independent full-flat runs; None is shared, and
Group-balanced shares Molecule-cap's Direct point.

| Method | Validation mean Macro-F1 | Test mean Macro-F1 |
|---|---:|---:|
| MiniMol + head | 0.6382 | 0.6107 |
| MiniMol kNN, condition | 0.6173 | 0.5966 |
| MiniMol kNN, all | 0.5794 | 0.5717 |
| Morgan kNN, condition | 0.5775 | 0.5765 |
| Morgan kNN, all | 0.5545 | 0.5491 |
| LLM None | 0.6043 | 0.5522 |
| Direct / Neighbor-fill | 0.6501 | 0.5923 |
| Direct + Indirect / Neighbor-fill | 0.6693 | 0.6445 |
| Direct / Molecule-cap | 0.6561 | 0.5970 |
| Direct + Indirect / Molecule-cap | 0.6840 | 0.6566 |
| Direct + Indirect / Group-balanced | 0.6924 | 0.6515 |

Means give equal weight to the six tasks. These are single-run descriptive
results without replicate intervals or significance claims. The completed traces
retain 7 validation and 156 test tool-failure records (not failed predictions or
necessarily unique queries); the figure footnotes disclose these counts.

The preparation upgrade preserved 7,466 existing no-threshold outputs
byte-for-byte after draining in-flight requests. Its 67 targeted tests, four
exact BBB/DILI prompt checks, migration and live throughput receipts are under
`outputs/paper/analysis/baseline_preparation_upgrade_20260917/`. This operational
change is not a separate experimental setting or replicate. Joint/Quality and
annotation extraction remain paused.

## Historical thresholded baselines

The completed visible validation pilot (`baseline_pilot`, similarity ≥0.3,
12,258 verified predictions) and stopped thresholded test (`baseline_test`) are
historical references. Their exact roots, scores, figures, frozen runtime and
stop/verification receipts remain in `current_conditioned_results.json`.
Neither supplies predictions to the current no-threshold suites. The earlier
identity-blind attempt is also separate provenance, not a visible result.
The pilot used its original coordinator; it was not migrated to the global pool.

## Direct + Joint: paused implementation and source validation

Paused by user on 2026-09-17; baseline work is the next priority. The implemented
None / Direct / Direct + Indirect mode uses parent-disjoint retrieval for every
L2+ card, B=64 (sweep 16/64/128), and T=1. Current contracts are
`direct_indirect_budgeted.v3`, `record_diversity_annotation.v4`,
`record_assessment_input.v3`, `record_quality.v3`, and `budgeted_record_features.v3`.
See [BUDGETED_EVIDENCE.md](BUDGETED_EVIDENCE.md) for ownership and baseline entrypoints,
and [the pause receipt](receipts/joint_pause_review_20260917.json) for checkpoint
and verification details. **No formal current-revision prediction comparison has
completed.**

The current source/context audit recomputed 341 direct/selected records across
five real-tool contexts. A bounded PMID31006869 review repaired 17 Skin nonvoter
identities and rebuilt v8 retrieval plus the shared Parquet export. Skin valid #0
retains 61/64 indirect cards; three incorrectly attributed cards leave and its
11 direct cards are unchanged. Gold/split rows are unchanged. Source receipts:
`data/starling_data/skin_reaction/trace_review_20260917/README.md`.
The earlier 50-snippet review is bound to its pre-v8 snapshot; the three new cards
have a separate review. Known semantic and quality-rule gaps remain explicit in
`outputs/paper/joint_validation_20260917/context_input3_manual_review.json`.

The paused cache contains 6,530 successful unique jobs out of 174,800, with no
in-flight requests. Schema/quote validation does not establish semantic accuracy.
Formal preparation requires complete exact-input/prompt-bound assessments for
all candidate and direct cards; this partial cache is not a full benchmark cache.

Earlier native-only preparation checks and one successful Skin prediction used
older revisions and remain diagnostic provenance in
`receipts/direct_joint_implementation_20260917.json` and
`receipts/joint_semantic_rule_audit_20260917.json`. They do not validate current
prediction performance. The five existing baselines are unchanged; 60 six-task
scaffold valid/test artifact cells have a separate row/lineage audit, not a retrain.
The historical experiments below are not redesigned-paper comparison arms.

## Record diversity: current results

Registry: `replicate_suites.record_diversity_six_task_valid_20260916`.

Both the preceding anchored-diversity experiment and its molecule-local replay
are complete: **2,043 scaffold-valid queries, 11,741 query-level outputs per arm,
zero residual failed queries or validation errors**. The replay used 8,310 fresh
model calls, 3,063 verified reused prefix levels and 368 carry-forward updates.
Original prompts and priors were retained; Bio `measurement_scope.v1` was disabled.

Settings: similarity ≥ 0.3, at most 10 molecules per level, five newly unlocked
records per molecule/level, L1/L2 scaffold-disjoint and L3+ parent-disjoint.
The model was DeepSeek-V4-Flash-0731, with shared endpoint admission and retry
race width 6. These are single development runs on inspected validation data.

| Task | n | Final level | Historical budget control | Previous diversity | Completed molecule-local | Change (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Ames | 274 | L5 | 0.7272 | 0.7121 | 0.7104 | -0.17 |
| BBB | 397 | L5 | 0.7526 | 0.7588 | 0.7612 | +0.25 |
| Bioavailability | 262 | L6 | 0.8266 | 0.8117 | 0.7995 | -1.22 |
| Carcinogens | 469 | L7 | 0.6413 | 0.6192 | 0.6363 | +1.71 |
| DILI | 402 | L7 | 0.6763 | 0.6698 | 0.6334 | -3.64 |
| Skin | 239 | L3 | 0.6544 | 0.6508 | 0.6364 | -1.44 |

All scores are Macro-F1. Carcinogens improves relative to previous diversity;
Ames improves at L2–L4 but not at the final level. Bio L2 improves while L4–L6
decline; DILI and Skin final scores decline. Historical controls differ in
upstream card caps and some source/context fields and are not pure selector
ablations. Do not infer statistical significance from these single runs.

Per-level CSV, canonical PNG/SVG and paired source-card audit:
`outputs/paper/record_diversity_six_task_valid_20260916/selector_repair/final_review/REPORT.md`.
The registry pins the report/figure, predictions and completion verification.

### Latest selection-only repair

`record_selection_context.v2` expands relevant endpoint recognition and bounded
condition handling. The source-bound UGT overlay corrects low relative catalytic
capacity incorrectly encoded as negative activity; its full Bio replay adds no
selected-list changes. The release has **no inference scores**. It still has
species/system applicability, formulation and near-duplicate selection issues;
greater endpoint/outcome coverage is not evidence of globally better records.
Its exact manifests and selected lists are pinned in the report directory's
`selection_release.json`. Preparation/selection is not completed inference.

The 2026-09-17 working `record_selection_context.v3` repairs tested-species
attribution using explicit population/system fields rather than dose/platform
mentions in qualifiers. 114 relevant regression tests passed; full selection
replay and inference have not been run. It does not replace the frozen v2
selection release or any scores above.

### Retained pilots and preprocessing

The four-task 20-query, three-arm pilot is complete under
`diagnostics.record_diversity_progressive_pilot_20260916`; its restricted sample
does not estimate full-valid performance. DILI/Skin paired 20-parent pilots are
also complete: final baseline/diversity correct counts are 18/17 and 16/17.
Their detailed source reviews and metrics stay in the registered artifact roots.

The six-task feature extension resolved all 415,230 incremental annotation jobs,
including 4,862 initially failed jobs; all six caches cover 835,352 candidate
cards. Reviewed source/annotation corrections and exact input reuse receipts
remain in `completion_audit.json`, `full_feature_readiness.json` and the pilot
review artifacts. This does not certify whole-corpus semantic accuracy.
DILI/Skin preprocessing and full-valid evaluation are complete; the earlier
four-task pause is historical and must not be described as the live status.

The preceding global-coverage diagnosis is retained under the suite's
`diagnosis/REPORT.md`; it predates the completed molecule-local replay.
Current workflow, module boundaries and historical-policy compatibility are in
[RECORD_FEATURE_SELECTION.md](RECORD_FEATURE_SELECTION.md).

## Paused fixed-25, no-threshold progressive budget (2026-09-16)

`replicate_suites.fixed25_no_threshold_progressive_full_flat_20260915` is
`paused_by_user` at 2026-09-16 01:11 UTC to free endpoint resources. The coordinator
and three active valid inference process groups were stopped; partial successful
checkpoints and all prepared inputs are retained. No automatic resume; the suite
`pause_receipt.json` records the stop and verification. The prior explicit resume
at 2026-09-15 20:23 UTC was for progressive only. All six tasks passed preparation verification; each task runs
scaffold valid then test, with at most three active tasks. Full-flat is excluded;
the predecessor suite remains paused. The operational override is recorded in
`progressive_only_resume_receipt.json`, preserving the frozen preparation contract
and prepared inputs. Each level takes the top 25 eligible molecules from the shared
new/active pool, with no similarity cutoff, and adds up to five newly unlocked
cards per molecule. If fewer than 25 are eligible, it takes all and records the
shortfall. Source-family disjoint rules, frozen runtimes and execution shape are
inherited. Live progress is recorded in the suite's `execution_status.json`;
preparation or launch is not a completed model result.

## Expanded progressive budget and matched full-flat (2026-09-15)

`replicate_suites.expanded_budget_progressive_full_flat_20260915` freezes six tasks,
scaffold valid/test (4,091 queries, 23,510 levels per organization). Each level
selects at most ten molecules from one shared new/active pool and adds at most five
newly unlocked cards per selected molecule. Similarity is 0.3 for every task,
including DILI (previously 0); source-family L1/L2 remain scaffold-disjoint and
L3+ parent-disjoint. Current indices include Skin v7; indexed representative caps
remain unchanged. Preparation verifies selected card IDs against the census.
Bioavailability/Skin retain their legacy prompt/tools/prior; other tasks retain v5.

The frozen execution plan runs progressive with at most three active tasks, then matched full-flat
with all six tasks eligible. Each task runs valid then verified test and has a
512-request cap, including width-six failure/repair races; first attempts use one
request. HAProxy at `127.0.0.1:50010` routes to dgx007:50002, dgx011:50001 and
dgx014:50002, each capped at 512. A shared 1,536-permit broker admits requests
before HTTP deadlines begin. Full-flat consumes completed progressive inputs
byte-for-byte. Live state is the registered `execution_status.json`; preparation
or launch does not establish completed model results.

Paused by user at 2026-09-15 19:04 UTC. All 12 progressive task/subset runs are
complete and verified. Five matched full-flat runs are complete and verified:
BBB, Bioavailability, Skin and Ames valid, plus Skin test. The five remaining
active full-flat runners and both suite coordinators were stopped; DILI and
Carcinogens test had not started. Successful checkpoints are retained, and neither
suite resumes automatically from that pause. The fixed-25 suite was subsequently
explicitly resumed for progressive only, as described above; this 10-molecule
suite remains paused.

Final-level Macro-F1 for the five complete pairs only:

| Task / subset | Progressive | Matched full-flat | Progressive minus full-flat (pp) |
|---|---:|---:|---:|
| BBB / valid | 0.7526 | 0.7697 | -1.71 |
| Bioavailability / valid | 0.8266 | 0.7887 | +3.79 |
| Skin / valid | 0.6544 | 0.6187 | +3.58 |
| Ames / valid | 0.7272 | 0.7272 | 0.00 |
| Skin / test | 0.5995 | 0.5585 | +4.10 |

The [paired comparison](../../../outputs/paper/analysis/expanded_budget_full_flat_vs_progressive_20260915/README.md)
contains valid/test PNG and SVG curves, all 22 paired levels, recomputed metrics,
matched prepared-input verification and the pause receipt. Incomplete full-flat
runs are excluded. Each setting has one run; these differences do not establish
statistical significance. This expanded-budget comparison is separate from the
completed smaller-budget suite below.

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

## L3+ parent-disjoint progressive and matched full-flat (complete)

`replicate_suites.l3_parent_disjoint_matched_full_flat_20260914` is complete and
verified: valid 2,043 rows / 11,741 level outputs; test 2,048 rows / 11,769 level
outputs, zero failed queries. Each subset passed frozen-input, independent-request
and recomputed-metric checks; all valid checks preceded test inference. Source-family
L1/L2 retain scaffold exclusion; only L3+ use parent exclusion. The new full-flat
and hybrid progressive share frozen cards, tools and priors, including the paired
legacy Bioavailability/Skin runtime. Later Skin v7 source rebinding is separate.

Final-level Macro-F1 (one run per setting; no estimated run-to-run uncertainty):

| Task | Valid progressive | Valid full-flat | Test progressive | Test full-flat |
|---|---:|---:|---:|---:|
| BBB | 0.7612 | 0.7622 | 0.7082 | 0.6978 |
| Bioavailability | 0.7973 | 0.7928 | 0.7857 | 0.7779 |
| Skin | 0.6508 | 0.6174 | 0.5823 | 0.5813 |
| Ames | 0.7424 | 0.7221 | 0.7706 | 0.7970 |
| DILI | 0.6651 | 0.6522 | 0.6378 | 0.6410 |
| Carcinogens | 0.6278 | 0.6455 | 0.5575 | 0.5784 |

Both organizations use L3+ parent-disjoint retrieval in this table. Neither is
uniformly better across tasks; these single runs do not establish significance.
The suite's `valid_comparison_figure` and `test_comparison_figure` register the
six-panel PNG/SVG, exact per-level TSVs, and source/visual verification receipts.
Colors identify settings consistently across tasks and subsets. Valid also shows
original v5 scaffold full-flat as a historical reference: Bioavailability/Skin differ
in prompt/tools/prior and Ames/DILI in source repairs. Test has no original-v5
scaffold full-flat result and explicitly leaves that setting unavailable. None
belongs to the matched hybrid pair; five baselines use the same evaluation cohort.

This completed run used direct SSH to dgx007:50002, dgx011:50001 and dgx014:50002,
each with a shared 512-permit admission broker (1,536 total, including retry races).
The run plan, six subset/runtime verification receipts, endpoint probes and
checkpoint-preservation receipts remain under
`outputs/paper/l3_parent_disjoint_matched_full_flat_20260914/`. They record the
transport changes and reported model/settings; weight revision was not independently
verified. Endpoint burst probes are not full-completion throughput measurements.
The newer expanded-budget suite has its own transport and scientific contract.

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
the later Skin source rebinding and the completed matched full-flat comparison.

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

## GPT-6 Sol / Luna validation smoke (2026-09-22)

The [small Neighbor-fill diagnostic](../../../outputs/paper/gpt6_neighbor_fill_smoke_20260922/RESULTS.md)
uses 24 selected validation examples, Direct=50 + Indirect=50, shared cached
single-molecule analyses and current adapter v2. It completed 48 base calls,
8 existing-Carcinogens-calibration calls and 4 Luna formatting checks; total
reported cost is $3.548160475. Base readable-label matches are 16/24 for each
model, with schema validity 22/24 for Sol and 23/24 for Luna. Neither tested
prompt variation is adopted. This is a small diagnostic, not a test benchmark
or model-ranking result; registry `diagnostics.gpt6_neighbor_fill_smoke_20260922`
pins inputs, responses, errors and verification. Production prompts are unchanged.

## Carcinogens valid: GPT-5.6 Sol and prompt follow-up (complete)

The opt-in `gpt_calibration.v1` prompt follow-up is **complete and verified**,
finished through the normal chat-completions API and tracked by
`replicate_suites.carcinogens_scaffold_valid_gpt56_sol_prompt_v1_api_20260914`.
It keeps the same model, medium reasoning, omitted temperature, 20,480-token cap,
469 valid rows and seven progressive levels, with 128 concurrent requests.
It reuses 243 Batch single outputs after exact-message and benchmark-hash
verification; remaining calls use the normal API. All 469 single/None queries and
3,283 levels completed with zero failures and verified input/source/prepared hashes.
The reuse, transport handoff and billing receipts remain registered. Adapted
None/L1/L2/L3/L4/L5/L6/L7 Macro-F1 is
0.5403/0.6058/0.6203/0.6153/0.6199/0.6217/0.6235/0.6236; final accuracy is 0.6887.
Relative to the original Sol run, final Macro-F1 rises from 0.6009 to 0.6236
and accuracy from 0.6503 to 0.6887. This is a single-run prompt/transport
comparison; statistical significance and isolated prompt causality are untested.
The registered `comparison_figure` shows DeepSeek V4 Flash, Sol old prompt and
Sol adapted prompt over identical evidence/tools, with configuration-specific
None controls. Final Macro-F1 is 0.6238/0.6009/0.6236 respectively; final accuracy
is 0.7036/0.6503/0.6887. The figure receipt records the explicit prompt-comparison
audit and reproducible plotter command.
The registered `cost_verification` reports $244.16769045 including completed Batch
and smoke charges. Its 2026-09-14 20:14 UTC snapshot still had four remote batches
pending after cancellation returned HTTP 404; this is not a closed final bill.

`replicate_suites.carcinogens_scaffold_valid_gpt56_sol_batch_20260913` tracks
469 scaffold-valid rows with fresh Sol single/None and progressive reasoning.
It uses the existing matched-suite runner, frozen v5 source tools/cards, explicit
medium reasoning, omitted temperature and the same 20,480-token output cap.
The reusable `openrouter_batch` provider transport adds no task-specific prompt
or runner. All 469 queries and 3,283 level outputs completed with zero failed
queries; source/tool compatibility and resume history remain in the registered
receipts. Macro-F1 for None/L1/L2/L3/L4/L5/L6/L7 is
0.5168/0.5962/0.6063/0.6063/0.6005/0.5992/0.6009/0.6009; final accuracy is 0.6503.
The 51 completed batches report $104.8726226 for 938 single/None and 3,084
progressive requests (2,954 model-called level outputs), excluding the $0.00009
transport smoke. The registered
`completion_verification` receipt verifies current input/index/catalog hashes,
prepared-input hashes, output schemas, counts and raw per-batch billing usage.
The registered `deepseek_comparison_figure` compares the matched current v5
DeepSeek run: None/final Macro-F1 is 0.5923/0.6238 versus Sol 0.5168/0.6009;
final accuracy is 0.7036 versus 0.6503. DeepSeek is higher at every L1-L7.
Both are single runs with identical evidence/tools and independently generated
priors; GPT uses medium reasoning and omitted temperature, while DeepSeek uses
its existing default reasoning and temperature 0. The figure explicitly records
these generation differences; it is not a generation-setting ablation.
The paired trace diagnostic finds 38 more missed positives but 13 fewer false
positives for GPT at L7. Of GPT's 123 final false negatives, 115 were already
wrong at None; this is an association, not a demonstrated prior-causality effect.
Exploratory paired bootstrap over 319 molecule identities (all species rows kept
together, 5,000 resamples) gives a 95% percentile interval of [-0.0221, 0.0657]
for DeepSeek-minus-GPT Macro-F1. It does not measure run-to-run variation or
establish stable model superiority. Diagnostic artifacts are linked from the
suite. The separate implemented prompt follow-up is reported above; it does not
change this original comparison.

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
This restoration receipt covers preparation only. Later model results using the
frozen runtime are registered in the completed L3+ progressive/full-flat suites
above; the original restoration package is not another replicate. The v5 runs
and DILI source-repair replay retain their separate contracts.

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
(no repeat SD), using PARCC DeepSeek-V4-Flash-0731, matched prepared evidence,
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

## GPT-6 Luna Batch fixed-budget test (2026-09-22)

Complete and verified: conditioned and TDC scaffold test, six tasks each; None / Direct / Direct + Indirect, Neighbor-fill 50/50. All 12,819 predictions in 36 cells passed, with zero final failures. Actual Batch cost including repairs: $22.90500961 (conditioned $10.44522676; TDC $12.45978285). Six-task mean Macro-F1 for None / Direct / Direct + Indirect: conditioned 0.617613 / 0.624771 / 0.654253; TDC 0.656558 / 0.674375 / 0.673375. Full-precision scores: `outputs/paper/gpt6_luna_batch_b50_20260922/scores.csv`. Shared family runner, upstream adapter v2, medium reasoning, cached priors; no scientific prompt changes. Registry: `budgeted_evidence_protocol.gpt6_luna_batch_b50`. Progress, exact commands and completion audits: `outputs/paper/gpt6_luna_batch_b50_20260922/`. This is an adopted paper baseline. The retained `verify_results.py` audits saved outputs only; the completed temporary launcher/logs are archived. Code/cleanup review: `receipts/luna_baseline_maintenance_20260923.json`.
