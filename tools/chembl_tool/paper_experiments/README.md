# Molecular-evidence agent paper experiments

This directory contains the maintained paper-facing experiment layer. Task
science stays in `tools/chembl_tool/tasks/<task>/`; retrieval, visibility,
reasoning, validation, checkpointing, analysis, and plotting are shared.

The [scaffold follow-up review](RESULTS.md#scaffold-follow-up-maintenance-2026-10-18)
records the latest scoped code/result checks. The earlier
[paper maintenance review](RESULTS.md#maintenance-review-2026-09-29) retains unresolved
manuscript/provenance differences.

## Release preparation

As of 2026-09-30, the current work is preparing the codebase for release after
paper completion. Storage migration and cleanup are complete; broad code/documentation
consolidation and reproduction in an independent environment remain pending.

The `refactor contract` (local-only historical record) records the 2026-10-18 decision to
adopt upstream's package organization while preserving this working tree's
scientific behavior. Its semantic comparison and migration gates take precedence
over assuming that similarly named functions or disjointness policies are equal.

- Consolidate duplicate implementations and temporary entrypoints into the maintained
  shared modules. Check callers and registered reproduction commands before retiring code.
- Keep README files focused on current usage; preserve necessary historical commands,
  source snapshots and decisions through linked archives/receipts. The result registry
  owns result paths, and [retention rules](TRACE_RETENTION.md) own preservation boundaries.
- Inventory host-specific paths, external runtime dependencies and archived inputs;
  document installation/restoration before claiming a portable release. Verify retained
  entrypoints and saved-result exports after changes. A shared-disk archive alone is
  not a tested clean installation or a complete experiment replay.

The [storage completion record](RESULTS.md#storage-cleanup-2026-09-30) links both
maintenance receipts and restoration guidance. Keep the molecular tool service,
Joint extraction and paused experiments stopped. The [open review items](RESULTS.md#maintenance-review-2026-09-29),
including the Direct-budget discrepancy and Ours provenance limits, remain open;
cleanup does not resolve scientific or reporting differences.

## Current retrieval data

DILI/Carcinogens 的当前数据合同及保留的构建／维护入口见
[`NEW_TASK_SOURCE_DATA.md`](../common/starling/NEW_TASK_SOURCE_DATA.md)。二者已接入统一
Stage-03 恢复、catalog/index 重建和普通 Parquet level 导出；无需单独维护 runner 或画图程序。
当前 benchmark 均为 Starling-only gold_v4，清理后的检索源统一冻结在 `retrieval_final`。
DILI 新版 prompt-v5 源修复 replay 与历史 retrieval-review v5 分开登记；Carcinogens 保留 R18 诊断。结果、baseline 和图均从
`current_conditioned_results.json` 的对应 suite 读取。旧混合标签、旧 gold 和清理中间轮次
只作历史参考。源整合本身不产生模型结果；另行登记的 targeted replay 保留实际输入、请求和复用凭据。

## TDC DILI prompt and local replay maintenance

The [TDC-only handoff](shared/TDC_DILI_HANDOFF.md) provides the retained complete
system prompt, score-display change and completed results. Reuse the collaborator's
existing harness. Gold-v1 and other tasks keep their existing task prompts.
Registry `diagnostics.tdc_dili_prompt_20260924` owns this post-test diagnostic.
Its local `prompt_replay/run.py` only prepares/resumes the retained 96-query replay;
`summarize.py --root <saved-root>` scores saved requests without rebuilding inputs
or rewriting configuration. These require local archived traces; they are not new
paper runners. Rejected A/B generation switches are retired; their source is
archived beside frozen prompts/results. The combined arm was not run.

## Matched seed-42 TDC rerun

[Skin/Carcinogens run plan](../../../outputs/paper/tdc_seed42_neighbor_fill_20260924/README.md):
four OpenRouter suites, Luna/DeepSeek × two tasks, 128 requests each. Reuse the
family `--baseline-suite` entrypoint with `--baseline-settings none direct_fill
both_fill --retry-race-width 6 --transport-max-retries 0`. Retries share the
128-request cap; no additional task/model runner. All 828 predictions are verified,
with 414 identical model-input pairs and exact ML query alignment. Results and
completion receipts: registry `tdc_seed42_neighbor_fill`. The full ML Appendix
and current figures use the new cohorts.
Rebuild it with `python outputs/paper/llm_vs_standard_baselines_20260921/build.py
--seed42-appendix-output outputs/paper/llm_vs_standard_baselines_20260924`.
This export audits saved predictions and leaves historical tables intact. New Skin/Carcinogens runs
must use the registered seed-42 input manifest; the original six-task
`data/tdc_benchmark/benchmark_manifest.json` still pins their historical inputs.

## Scaffold-disjoint indirect follow-up

DeepSeek V4 Flash 0731 and GPT-6 Luna each completed 4,273 predictions / 12 cells
on Conditioned and current TDC test. Full scores and scope are linked from
[the results inventory](RESULTS.md#indirect-scaffold-disjoint-follow-up-2026-10-13).
Gemini scaffold inference has not run; the existing Gemini paper scores retain
parent-disjoint indirect retrieval.

| Operation | Existing entrypoint / configuration |
|---|---|
| Prepare | Family CLI `--baseline-suite --baseline-settings both_fill --direct-budget 50 --indirect-budget 50 --indirect-identity-policy scaffold_disjoint --prepare-only` plus the exact model/prior/profile options in each `run_plan.json` |
| Inference / resume | Same family CLI `--run-prepared`; use recorded per-task `inference_commands`. Luna profiles share `common.request_admission`, started with its recorded `admission_launch.json` command |
| Match historical rendering | `--record-rendering full_fields` only for DeepSeek TDC adapter-v1 references; default `exact_duplicate_fields.v1` remains adapter v2 |
| Audit retained results | The single experiment-local `outputs/paper/indirect_scaffold_disjoint_20261013/verify.py --root <experiment-root> --require-complete`; optional `--profiles` selects a recorded profile |
| Plot | Official `plot_assay_retrieval_curve.py` CSV/metadata interface; `--bar-content tasks` and `mean` produce separate panels; `excluded_bar_settings` hides unavailable arms |

The auditor is saved-result provenance, not a second runner. Completed temporary
controllers and console logs are archived; paths in historical launch receipts
refer to their original locations. Keep source snapshots, cached priors, prepared
requests, predictions and retrieval audit dependencies. See
[retention rules](TRACE_RETENTION.md) and the
[maintenance receipt](receipts/indirect_scaffold_maintenance_20261018.json).

## Paper scope

### Appendix result sources

The [2026-09-24 split audit](../../../outputs/paper/skin_tdc_upstream_split_20260924/SPLIT_AUDIT.md)
found that upstream's latest Skin and Carcinogens TDC test cohorts differ from the
local seed-1 ML/naive-LLM cohorts. Their reported Ours points in earlier combined
figures are not paired cohort comparisons. The separate
[Skin alignment run](../../../outputs/paper/skin_tdc_upstream_split_20260924/README.md)
uses upstream's exact split; registry `tdc_skin_upstream_split` owns its completion
status and results. The current ML Appendix now uses both seed-42 successors.
The new naive LLM runs use their exact molecule inputs; seed-1 LLM results remain historical. Other tasks' test agreement does not establish train/val agreement.

The [Carcinogens alignment run](../../../outputs/paper/carcinogens_tdc_upstream_split_20260924/README.md)
uses raw TDC scaffold seed 42: its 56 test parent/label pairs match upstream v2.
Train/valid are reconstructed from the upstream splitter; upstream's v2 files are
unpublished, so their membership is not independently verified. Registry
`tdc_carcinogens_upstream_split` records its two-pool ML completion and scores.

Use registry `budgeted_evidence_protocol.llm_vs_standard_baselines.ml_appendix`
for the current six-task ML tables. `ml_appendix.csv` is the full-precision
manuscript source (198 cells); `comparison.json` and each benchmark's
`all_scores.csv` also include the LLM rows. RASAR (six direct features) and
RASAR+indirect (6+2B features) have separate all-train/condition-first rows. The
[paired Macro-F1 / AUROC table](../../../outputs/paper/llm_vs_standard_baselines_20260924/ML_APPENDIX.md)
has three panels: conditioned, TDC-only and TDC + Starling labels, each with
66 task/method cells. Its [CSV](../../../outputs/paper/llm_vs_standard_baselines_20260924/ml_appendix.csv)
retains full precision, training pool, task, method, test size and prediction path.
Use the CSV for manuscript tables; round only when typesetting. Report panels
separately and retain the documented equivalent TDC-only KNN and RF policy aliases.

The [current comparison](../../../outputs/paper/llm_vs_standard_baselines_20260924/RESULTS.md)
retains all baseline variants and their accuracy in `comparison.json` and
per-benchmark `all_scores.csv`. LLM AUROC is unavailable; do not derive it from
hard decisions. The 20260921 comparison figures are historical tool-visible /
Direct-3 results; that directory retains the shared saved-prediction audit/export
script `build.py`. The current ML appendix is in the 20260924 result directory.

Training parameters, validation selection and exclusion rules remain in the
[MiniMol protocol](../../../baselines/minimol/README.md), [RF protocol](../../../baselines/RETRIEVAL_FOREST.md) and linked source-suite reports.

### Retrieval forest

**RASAR+indirect complete (2026-09-22):** all six tasks in conditioned, TDC-only
and TDC + Starling pools, each with all-train and condition-first retrieval. The
36 report cells use 30 distinct model suites (identical TDC-only policies reuse models).
Validation BCE selects RF parameters; validation Macro-F1 selects the threshold.
[Original settings and historical suite](../../../outputs/paper/retrieval_forest_20260922/RESULTS.md);
current Skin/Carcinogens replacements are in the unified Appendix above;
shared entrypoint: `python -m baselines.retrieval_forest {prepare,fit,finalize}`.
Every query excludes its own parent/connectivity from donors; auxiliary pools also
exclude the whole valid/test identity union. Registry: `retrieval_forest`.

**RASAR direct-only control complete (2026-09-25):** the same six tasks, three
training pools and two policies now have a matched six-feature control (36 report
cells / 30 unique suites). Use `fit` and `finalize` with `--feature-set direct`;
all indirect bucket columns are omitted. Skin/Carcinogens use current seed-42
TDC inputs. [Results and paired comparison](../../../outputs/paper/retrieval_forest_direct_20260925/RESULTS.md);
registry `retrieval_forest_direct`. The suite pins current cache hashes and retains
the same validation selection protocol.

### Current comparison figures

The two **final** layouts, adopted 2026-09-25, are maintained by
`plot_assay_retrieval_curve.py`. The former left-line/right-point comparison and
the early bars grouped by evidence stage are **retired**. The file/CLI name
`baseline-curves` is retained for input compatibility; it does not select the old style.
Legacy curve rendering remains only for historical scaling/audit reproduction.

Registry `budgeted_evidence_protocol.llm_vs_standard_baselines` owns the
[current results](../../../outputs/paper/llm_vs_standard_baselines_20260924/RESULTS.md),
`figure_variants`, `mean_figures`, and the common source files. Task and average
performance are separate figures; both task orientations remain supported:

| Layout | CLI option | Output stem | Use |
|---|---|---|---|
| Six vertical task panels (3 rows × 2 columns), no mean column | `--bar-layout mixed --bar-content tasks` | `baseline_bars_mixed` | Primary per-task layout |
| Six aligned horizontal task panels, no mean | `--bar-layout horizontal --bar-content tasks` | `baseline_bars_horizontal` | Alternate per-task layout |
| Standalone vertical six-task means | `--bar-layout mixed --bar-content mean` | `baseline_bars_mean` | Two benchmarks composed left/right in `mean_results.pdf` |

For separate benchmark figures, use `conditioned/baseline_bars_mixed.pdf` and
`tdc/baseline_bars_mixed.pdf` directly. Both have their own title and legend;
`main_results.pdf` is the optional stacked composition of these same panels.

The five ML groups are MM Direct, MM + Indirect, KNN, RASAR and RASAR+indirect.
Both task layouts use pastel palette B, gray ML bars, independent task scales
with axis breaks, and **no subtitle or footer prose**. Vertical task and mean
bars have no value labels or y-axis titles; ticks remain. The horizontal task
alternative retains four-decimal value labels. Exact scores and means are in
the receipts and result tables.

The current `mean_results.pdf` is the accepted design extended to Gemini:
both benchmark titles are above one shared color-legend row and a separate
abbreviation row, each centered across the full figure. The model names and
ticks retain their large fonts. The muted abbreviation strip (#748093) has
extra separation from the model groups. Standalone panel PDFs retain their own
legends; the LaTeX composite displays a single shared legend.

| Current style | Value |
|---|---|
| Vertical task font sizes | ticks 26 pt; groups/legend 28 pt; task titles 38 pt, normal weight in both layouts |
| Vertical task spacing | X tick labels only on the bottom panel in each column; row spacing 0.65, giving each task a taller plotting area |
| Mean panel font sizes | groups/ticks 28 pt; legend 20 pt; abbreviations 22 pt |
| Mean composite strip scale | legend 1.25; abbreviations 1.10 |
| Model-group underline / axis spine / bar outline | 1.8 / 0.7475 / 1.1 pt |
| Within-backbone / ML group-center spacing | 0.8 / 0.8 |

Columns/groups represent backbones; colors represent None, Naive Direct,
Naive Direct + Indirect, Ours Direct, and Ours Direct + Indirect. MM / R mean
MiniMol / RASAR, defined in the top abbreviation row. Solid gray denotes each
benchmark's main-task training pool; hatching denotes TDC + Starling.

Ours has separate DeepSeek V4 Flash, GPT-6 Luna, and Gemini 3.8 Flash curves, explicitly bound to
their backbones through `reported_curve_backbones`.
The user confirmed that Ours shares its backbone's None: draw that None once,
without adding a separate Ours None score to the source. Missing combinations
are never zero-filled: the mixed layout shows empty slots; horizontal panels omit settings missing for every task; vertical means leave missing slots empty. A partial task set has a **null mean**.
Gemini baseline inputs are the completed 36-cell
[`scores.csv`](../../../outputs/paper/gemini38_flash_api_b50_20260925/scores.csv).
`additional_curve_sources` binds each additional backbone to a SHA256-pinned CSV;
`curve_settings` supplies the three None / Direct / Direct + Indirect method IDs.
The existing exporter checks complete task/setting coverage and cohort sizes before
appending the rows. It rejects hash mismatches and method collisions. Gemini's
metrics were recomputed from all 12,819 saved predictions, whose ordered labels matched to
the current cohorts; full post-run provenance/cost reconciliation remains deferred
as recorded by the experiment. Ours is user-reported, not a prediction-level audit.
The inclusion receipt is `gemini_inclusion_verification.json` in the comparison root.
The existing exporter also regenerates the additional-source links in `RESULTS.md`
and source hashes/cell counts in `comparison.json` and `verification.json`.

Three-backbone panels retain the palette and model-name font sizes; only the ML
x-tick abbreviations use 22 pt to avoid overlap. Additional models still require
checking the fixed 16 × 9 inch mean panels and LaTeX clipping at manuscript size.
Future Ours curves require explicit backbone binding; do not fill missing results.


Means are unweighted arithmetic means of the six task Macro-F1 scores, **not**
pooled-query F1 or sample-size-weighted means. ML means average the per-task
maximum within each declared display group, separately by training pool; they
are descriptive test upper bounds, not the score of one globally selected method.
For the Appendix mean table, read `task_means` in each benchmark's
`baseline_bars_mean_receipt.json`: 18 naive-LLM means, 15 ML group means and
12 Ours means across both benchmarks. The 33 baseline values in
`tab:baseline-mean-results` use this exact convention. Task-only figures display
no means, although their receipts retain the same calculated values.
There are 120 conditioned / 150 TDC task marks, plus 20 / 25 available means.
All 120 / 186 raw baseline rows and 198 paired Macro-F1/AUROC appendix cells remain.

The shared [Ours record](../../../outputs/paper/llm_vs_standard_baselines_20260924/ours_reported_results_20260925_three_backbones.json)
owns all 72 user-reported scores supplied on 2026-09-25 for DeepSeek, Luna, and Gemini.
The [complete task-score tables](../../../outputs/paper/llm_vs_standard_baselines_20260924/RESULTS.md#complete-ours-scores)
and manuscript Appendix `app:ours-results` are derived from this source; refresh
both when updating the shared record.
Starling maps to conditioned. DeepSeek Direct + Indirect means are 0.720943 /
0.799718667 and Luna means are 0.6896595 / 0.7870845 (conditioned / TDC).
DeepSeek and Luna improve on all six conditioned tasks and five of six TDC tasks;
TDC exceptions are DeepSeek Carcinogens and Luna Skin. Gemini Direct / Direct +
Indirect means are 0.712062 / 0.736031167 (conditioned) and 0.807925333 /
0.815373333 (TDC): all six conditioned and four TDC tasks improve, TDC DILI ties,
and TDC Carcinogens declines. These are transcription
and rendering checks, not a new prediction-level audit. Experiment version and
budget metadata were not supplied with the replacement scores.
Baseline LLMs use Neighbor-fill
50/50. Skin/Carcinogens use matched seed-42 ML/LLM queries; other TDC tasks retain
the recorded DeepSeek adapter-v1 / Luna adapter-v2 difference. Carcinogens matches
upstream's 56 test parent/labels, with one raw Hg representation difference and
unpublished upstream v2 train/valid splits. New plot layouts do not change these boundaries.

#### Redraw both final layouts

From the repository root in the `vllm` environment:

```sh
for benchmark in conditioned tdc; do
  figure_dir="outputs/paper/llm_vs_standard_baselines_20260924/$benchmark"
  for layout in mixed horizontal; do
    MPLCONFIGDIR=/tmp/txagent-matplotlib PYTHONPATH=. OPENBLAS_NUM_THREADS=1 \
      python -m tools.chembl_tool.paper_experiments.plot_assay_retrieval_curve \
      --baseline-curves-csv "$figure_dir/points.csv" \
      --baseline-curves-metadata "$figure_dir/metadata.json" \
      --bar-layout "$layout" --bar-content tasks --analysis-dir "$figure_dir" \
      --output-stem "baseline_bars_$layout"
  done
  MPLCONFIGDIR=/tmp/txagent-matplotlib PYTHONPATH=. OPENBLAS_NUM_THREADS=1 \
    python -m tools.chembl_tool.paper_experiments.plot_assay_retrieval_curve \
    --baseline-curves-csv "$figure_dir/points.csv" \
    --baseline-curves-metadata "$figure_dir/metadata.json" \
    --bar-layout mixed --bar-content mean --analysis-dir "$figure_dir" \
    --output-stem baseline_bars_mean
done
```

Each render writes PNG/PDF/SVG, the selected baseline CSV and a JSON receipt.
No inference or training runs. CLI layout/content override metadata's `bar_layout` / `bar_content`;
explicit `--macro-f1-limits` overrides local axes, including the mean panel.

| Input / receipt field | Responsibility |
|---|---|
| `points.csv`, `all_scores.csv` | Audited baseline inputs and full table with available AUROC / prediction paths |
| `metadata.json: chart_style / bar_layout / bar_content / y_scale` | `bars` / `mixed` or `horizontal` / `tasks` / `task` |
| `task_labels` | Benchmark-specific panel titles; TDC uses BBB_Martins, Bioavailability_Ma, Skin_Reaction, AMES, DILI and Carcinogens_Lagunin; horizontal titles wrap at underscores |
| `curve_settings` | Backbone names and ordered None / Direct / Direct + Indirect method IDs |
| `additional_curve_sources` | Extra baseline CSV path and SHA256 per backbone, consumed by the existing exporter |
| `baseline_settings` | ML groups, declared candidates, display ticks, pool and pool label |
| `reported_curves_source` | Shared Ours JSON path relative to metadata and benchmark selector |
| `reported_curve_backbones` | Explicit reported-curve label to backbone mapping |
| `excluded_bar_settings` | Optional mapping from LLM backbone name to exact setting labels to omit from task bars, means, ticks and legends; omitted source values remain in `excluded_bar_scores` for audit. Configure ML groups with `baseline_settings` |
| `*_receipt.json: source_scores / baseline_selections` | Full candidate scores, paired winner metrics and ties |
| `*_receipt.json: bar_scores / task_means` | Model/setting bindings, task values, complete-task means and counts |
| `*_receipt.json: task_macro_f1_limits / mean_macro_f1_limits` | Task/mean scales |
| `*_receipt.json: rendered_counts` | Actual task/mean bars rendered; `point_counts` retains underlying source counts |

Update Ours in one dated shared source, preserve its predecessor, update both
metadata pointers and registry hash, then redraw both task orientations and the standalone mean panels. Receipts
pin point/metadata/source hashes. Refresh `figure_verification.json` after any
render; its old hashes must not be treated as current.

The latest [2026-10-14 scaffold-indirect figure variant](../../../outputs/paper/indirect_scaffold_disjoint_luna_20261014/figures/README.md)
uses this same renderer with separate task and mean panels. DeepSeek and Luna naive
Direct+Indirect use verified scaffold-disjoint scores. Ours Direct+Indirect and
Gemini naive Direct+Indirect are omitted because matching results are unavailable;
all retained references keep their original protocols. This variant is registered
under `indirect_scaffold_disjoint_luna.figures`. The original paper figures and
[earlier DeepSeek-only variant](../../../outputs/paper/indirect_scaffold_disjoint_20261013/figures/README.md)
remain frozen.

#### Audit and export saved results

The existing exporter now generates **both final layouts** and preserves the
current metadata/bindings. When source prediction auditing is needed:

```sh
MPLCONFIGDIR=/tmp/txagent-matplotlib PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python \
  outputs/paper/llm_vs_standard_baselines_20260921/build.py \
  --seed42-appendix-output outputs/paper/llm_vs_standard_baselines_20260924 \
  --seed42-llm-root outputs/paper/tdc_seed42_neighbor_fill_20260924
```

This audits the 36 seed-42 replacement cells, 9 corrected Skin auxiliary cells,
and 36 added RASAR cells, and
consumes the verified 828-prediction LLM suite. It regenerates 198 ML cells and
306 full comparison rows, preserving the prior prediction results. The maintained suite `verify.py` checks
those LLM predictions. The exporter refuses historical output roots and draws
new destinations from the current metadata rather than the retired template.
Do not import `build.py` as a library: historical audits execute at module scope.

The exporter renders six panels (two benchmarks × three variants) and copies
`main_results.tex` into a new output directory when absent, preserving existing
composition edits. It does not
compose the manuscript figures. `main_results.tex` is the single composition
source for `main_results.pdf`, `main_results_horizontal.pdf`, and
`mean_results.pdf`; it contains no scores or duplicate palette definitions.
The mean branch clips title/chart strips from the fixed 16 × 9 inch PDFs and
joins the TDC panel's legend strips into one centered row, keeping vector output.

From the current result directory, compile into a temporary directory to keep
LaTeX intermediates out of the results tree:

```sh
figure_build_dir=$(mktemp -d /tmp/txagent-figures.XXXXXX)
pdflatex -halt-on-error -output-directory="$figure_build_dir" main_results.tex
pdflatex -halt-on-error -output-directory="$figure_build_dir" \
  -jobname=main_results_horizontal '\def\BarLayout{horizontal}\input{main_results.tex}'
pdflatex -halt-on-error -output-directory="$figure_build_dir" \
  -jobname=mean_results '\def\FigurePart{mean}\input{main_results.tex}'
for stem in main_results main_results_horizontal mean_results; do
  cp "$figure_build_dir/$stem.pdf" "$stem.pdf"
  preview_size=2400
  if [ "$stem" = mean_results ]; then preview_size=3200; fi
  pdftoppm -scale-to "$preview_size" -png -singlefile "$stem.pdf" "$stem"
done
```

Inspect the final PNG and PDF after composition, not just SVG style attributes:
0.92 and 1.2 pt underlines rasterized identically in the 3200 px preview; 1.8 pt
was verified to change actual pixels. Refresh artifact/source hashes in
`figure_verification.json` and the registry after edits. `manuscript_preview.pdf`
is the saved 26-page compilation from this figure-editing round. The analog
manuscript has since changed and uses separate Starling/TDC figure files; this
preview and `manuscript_verification.json` are historical compilation evidence,
not verification of the current manuscript. Current plotting artifacts remain
the panels and composites listed above.

`--bar-content all` retains the former embedded-mean composition for historical
reproduction; it is not the current export default. The last combined design
is preserved in `design_snapshots/combined_before_split_20260925.zip`.

The user-requested [saved mixed design](../../../outputs/paper/llm_vs_standard_baselines_20260924/design_snapshots/20260925_backbone_pastel_mixed/README.md)
retains its reproducible source and inputs as a **frozen snapshot** preceding the
RASAR addition and naming update. Temporary
preview renderers and superseded current-root figures are archived in
`design_snapshots/retired_designs_20260925.zip`; they are not maintained entrypoints.

Current scores and names come from the unified comparison directory. [RASAR integration and code review](receipts/rasar_baseline_maintenance_20260925.json)
records the 198-cell export, 396-value paper audit, 62 passing tests and scoped cleanup.

### Skin auxiliary-source correction (2026-09-25)

All 9 affected Skin cells in the three training pools now use the collaborator's
release-level-mapped Parquet from commit `59c05729`: MiniMol+indirect and both
RASAR+indirect policies. [Protocol, corrected results and audit](../../../outputs/paper/skin_indirect_level_mapped_20260925/README.md)
are registered at `skin_indirect_level_mapped`. TDC retains the current seed-42
cohort. Six-feature RASAR, other ML methods/tasks and LLM predictions are unchanged.
Old Skin auxiliary scores in earlier suite reports (including the indirect side
of the original RASAR ablation) are historical; current comparisons use the unified
CSV. Source membership is release-level-mapped, not asserted to be strict L2+.

### LLM baseline runs

**Gemini 3.8 Flash paper baseline:** registry `gemini38_flash_api_b50`;
12,819 completed predictions / 36 cells for conditioned and TDC six-task test,
None / Direct 50 / Direct 50 + Indirect 50, both arms Neighbor-fill. TDC Skin and
Carcinogens use current seed-42 cohorts. Medium reasoning, 16,384 output tokens,
upstream adapter v2 and frozen priors; no model-specific scientific prompt.
The [baseline record](../../../outputs/paper/gemini38_flash_api_b50_20260925/README.md)
owns the protocol, two result tables and the full-precision `scores.csv` source.
It is included in both current task layouts and the mean figures, with its source
pinned under `llm_vs_standard_baselines.additional_baseline_sources`. Counts,
input hashes/labels and score summaries are checked; the user explicitly deferred
further provenance/cost reconciliation. Reuse family `--run-prepared` and the
shared admission broker; commands and historical Batch receipts stay in the
baseline record.

**GPT-6 Luna paper baseline:** registry `gpt6_luna_batch_b50`; 12,819 verified
predictions across two benchmarks and six tasks, zero final failures, actual cost
$22.90500961 including repairs. Shared family CLI:
`--baseline-suite --baseline-settings none direct_fill both_fill --direct-budget 50
--indirect-budget 50 --budgeted-prompt full_flat_v5 --min-similarity 0` plus the frozen
prior and provider settings in the [run record](../../../outputs/paper/gpt6_luna_batch_b50_20260922/README.md).
This is a standard-suite subset; record-scaling and Direct-3 modes reject the filter.
The completed temporary launcher is archived; `verify_results.py` only re-audits
saved outputs and performs no API calls. No Luna-specific scientific prompt was added.

**Conditioned standard baselines complete (2026-09-21):** all six MiniMol heads
now use train-only fitting, val BCE checkpoints and val Macro-F1 thresholds.
Skin/Ames/DILI/Carcinogens added 20 seeds; BBB/Oral reuse their adopted frozen results.
The [complete table](../../../outputs/paper/conditioned_minimol_val_selected_20260921/RESULTS.md)
includes the four fixed-k=3 KNN variants and current head results. Registry: `conditioned_minimol_val_selected`; commands remain
in the [shared MiniMol entrypoint documentation](../../../baselines/minimol/README.md#conditioned-external-validation-selection).

**Final adopted MiniMol + indirect assay profile (six tasks, 2026-09-22):**
all three controls are complete for BBB/Oral/Skin/Ames/DILI/Carcinogens: frozen
MiniMol, main-task encoder finetuning and indirect-assay joint training. The
four-task extension adds 80 seed runs, reuses current frozen heads and preserves
BBB/Oral results. All 18 ensemble cells use validation selection. Auxiliary
training excludes heldout parents/connectivity and permits shared scaffolds.
See the [protocol and commands](../../../baselines/minimol/README.md#minimol--indirect-assay),
[verified six-task results](../../../outputs/paper/minimol_indirect_assay_20260922/RESULTS.md)
and registry `budgeted_evidence_protocol.minimol_indirect_assay`.
The current LLM comparison tables and figures include these six-task encoder results. Training entrypoints
remain shared; the extension's launch/preflight scripts are archived source receipts.

**TDC encoder/indirect completion (2026-09-22):** both main training pools now
have six-task frozen, main-only encoder and indirect-assay controls: 240 new
seeds, 60 reused frozen seeds, 36 verified ensemble cells. Indirect pools exclude
TDC valid/test parents/connectivity and allow shared scaffolds. Main pools retain
existing split/Starling-label filters. [Results and parameters](../../../outputs/paper/tdc_minimol_indirect_assay_20260922/RESULTS.md);
registry `tdc_minimol_indirect_assay`. Use the shared MiniMol CLI.

**TDC fixed-k=3 reference (2026-09-19):** use
[`comparison_all_settings.md`](../../../outputs/paper/tdc_minimol_val_selected_20260919/comparison_all_settings.md)
and its CSV/JSON for all 20 LLM/ML report settings (18 distinct results).
MiniMol uses task-specific published parameters, TDC val BCE checkpoint selection
and val Macro-F1 threshold selection in both training pools. All 12 heads are
complete; this table retains k=3 KNN. The registry entries
`tdc_minimol_val_selected` and `tdc_all_settings_comparison` own this profile;
earlier CV head scores are historical. Protocol and commands are in
[MiniMol baselines](../../../baselines/minimol/README.md#tdc-external-validation-selection).

**2026-09-18 TDC test:** an explicitly approved external evaluation uses the
same six settings at fixed Direct 50 / Indirect 50. The existing family suite
accepts `--benchmark-manifest`; TDC train labels merge into Direct before
retrieval. Data, cached priors and outputs are isolated from conditioned runs.
See the [TDC profile](BUDGETED_EVIDENCE.md#tdc-scaffold-test-profile-2026-09-18)
and registry `budgeted_evidence_protocol.tdc_scaffold_baselines` for the completed
13,350-prediction verification and TDC-only ML comparison.

**2026-09-19 mixed ML and prompt review:** all five ML baselines also completed
TDC test with filtered Starling train labels (30 cells, 11,125 predictions).
The mixed-data protocol and historical CV command receipts are linked from
[MiniMol baselines](../../../baselines/minimol/README.md#tdc-test-with-added-current-starling-labels).
New local upstream-aligned inputs now remove exact duplicate displayed record
fields; adapter v2 is offline-verified and used by the completed 8,900-prediction
matched Morgan Direct-3 suite. The earlier 50/50 suite retains v1. See
[results](RESULTS.md) and the registry's `tdc_starling_mixed_baselines` and
`record_rendering_deduplication` entries under `budgeted_evidence_protocol`.

**2026-09-18 prompt alignment:** new budgeted runs default to adapted upstream v5
with cached single-molecule analysis and no raw tools. Pass
`--cached-prior-sources`; the local output JSON stays unchanged. Six-task task
export, prior provenance and examples are described in
[prompt alignment](BUDGETED_EVIDENCE.md#new-prompt-alignment-2026-09-18).
**Fixed-Direct indirect scaling (2026-09-25):** the conditioned test extension
holds Direct Neighbor-fill at 50 and compares Indirect Neighbor-fill 10/25/50/75/100.
The original 25/50/100 points are reused after full prompt/output checks; 75 and 10
each add 2,048 predictions through OpenRouter DeepSeek V4 Flash 0731 (256 concurrent
requests, failed-request race width 6). Registry `conditioned_direct50_indirect_scaling`
and the [experiment record](../../../outputs/paper/conditioned_direct50_indirect_scaling_20260925/README.md)
own exact commands and the completed 30-cell score table. All 10,240 predictions
across 10/25/50/75/100 are complete and verified. The same record also stores
30 user-provided Ours scores (raw predictions not audited). The square mean comparison
uses the existing plotter's `--indirect-budget-csv` plus
`--indirect-budget-baseline-csv`; PDF, source/figure hashes and reproduction commands
are linked from the experiment record and registry `comparison_figure`.

Aligned scaffold validation scaling is complete: **32,688 verified predictions**,
with four budget curves and all score points under registry
`budgeted_evidence_protocol.aligned_record_scaling.comparison`. Direct winners
are Neighbor-fill=50 and Molecule-cap=100. The 2026-09-21 conditioned test sweep
uses these frozen winners and adapter v2; all **32,768 predictions** completed
on 2026-09-22. Scores and the validation/test CSV are registered under
`budgeted_evidence_protocol.aligned_record_scaling_test`; see [test results](RESULTS.md#aligned-conditioned-test-record-scaling-2026-09-22). Reusable test/resume
commands are in [the scaling protocol](BUDGETED_EVIDENCE.md#test-record-scaling-and-resume). The user-requested Direct-fill 100 + Indirect-fill 25/50/100 test follow-up uses
`--baseline-phase indirect_fixed`; all **6,144 predictions / 18 cells** are complete
and verified under `aligned_record_scaling_test_direct100_fill`. The
[scaling inventory](RESULTS.md#aligned-scaling-record-inventory) links all three
suites (71,600 predictions). **The naive LLM baseline uses Neighbor-fill for
both Direct and Indirect**: Direct=50 versus the same Direct=50 + Indirect=50.
Use the [baseline contract](BUDGETED_EVIDENCE.md#adopted-naive-llm-baseline-direct-neighbor-fill)
and label scaling budgets separately. The completed aligned suites reached
their record budgets. The older nominal 50+50 valid/test suite uses
`--budgeted-prompt budgeted_v1` and retained a 10-molecule Direct candidate limit;
its Direct shortfalls are recorded under `baseline_no_threshold.record_count_audit`.

**2026-09-17 status:** Joint extraction, source/quality iteration and experiments
are paused by user; the six 50+50 simple LLM baselines completed no-threshold
scaffold validation and test (12,258 / 12,288 verified predictions). Both curve
figures and comparisons with the five existing ML baselines are registered in
`budgeted_evidence_protocol.baseline_no_threshold.suites`. See the
[baseline handoff and component map](BUDGETED_EVIDENCE.md#baseline-handoff) and
[verified pause receipt](receipts/joint_pause_review_20260917.json).
Do not automatically resume the annotation queue or treat preparation checks as
completed prediction comparisons.

| Current operation | Maintained entrypoint |
|---|---|
| All six simple LLM settings | `run_conditioned_assay_family_curve.py --baseline-suite` |
| TDC inputs and train-label records | Same runner, `--benchmark-manifest <registered suite manifest>`; Skin/Carcinogens use `tdc_seed42_neighbor_fill_20260924/data/benchmark_manifest.json` |
| Fresh TDC single-molecule analysis cache | `prepare_cached_single_analysis.py` |
| Current TDC-only / mixed ML head | `baselines.minimol.run_bioavailability_ma --threshold-strategy valid_macro_f1`; omit `--train-all` |
| Skin/Carcinogens seed-42 ML suites | `baselines.tdc_split_suite --root <prepared experiment root>`; input/source receipts in the linked suite READMEs |
| Conditioned/TDC MiniMol + indirect assay | `baselines.minimol.run_multitask {prepare,embed,train,finalize}`; frozen control reuses the head runner |
| Historical train-only CV | `baselines.minimol.run_train_cv`; separate selection protocol |
| RASAR+indirect | `baselines.retrieval_forest {prepare,fit,finalize}` |
| Mixed-label KNN | Existing `baselines.minimol.run_embedding_knn` and `baselines.structure_knn.run` |
| TDC matched Morgan Direct-3 | Same family suite, `--direct-knn-root <TDC ML root> --direct-budget 3` |
| Validation/test 25/50/100 sweep | Same runner, `--baseline-suite --baseline-record-scaling`; test adds `--direct-budget-selection` |
| One setting (also paused Quality/Joint) | Same runner, `--evidence-setting` and selectors |
| Verified six-setting trace curves | `plot_assay_retrieval_curve.py --budgeted-suite-root` |
| Final comparison bars from score CSV | Same plotter, `--baseline-curves-csv` + `--baseline-curves-metadata` + `--bar-layout mixed` or `horizontal` |
| Historical/scaling curves from score CSV | Same CSV flags; historical metadata retains the curve style |
| Score/accuracy overview | `plot_starling_model_comparison.py --budgeted-suite-root` |
| Supportive/contradictory citation statistics | `analyze_progressive_trace_adoption.py --budgeted-suite-root` |
| Current collaborator prompt bundle | [Shared prompt handoff](shared/README.md#task-prompts); apply task definitions and label mappings together |

Explicit run parameters and module ownership live in `BUDGETED_EVIDENCE.md`;
current means live in `RESULTS.md`, and exact artifacts in the registry.

New six-setting runs use `run_conditioned_assay_family_curve.py --baseline-suite`:
context preparation streams into one request pool across tasks and settings.
A shared CPU process pool loads each query candidate pool once across settings;
idle workers can serve any remaining task. The completed no-threshold valid/test
runs used the earlier six resident task workers;
their operational migration preserved all 7,466 already completed predictions.
See [global scheduling](BUDGETED_EVIDENCE.md#global-scheduling-for-the-six-baselines)
for the maintained command and checkpoint boundaries. The completed pilot used
its original coordinator.

The 2026-09-17 redesign uses **None / Direct / Direct + Indirect**. All L2+
records are indirect and use parent-disjoint retrieval. The new budgeted mode
of `run_conditioned_assay_family_curve.py` provides the five shared-pool selectors,
including Direct + Joint. Its B=64 and 16/64/128 research grid remains paused;
the completed baseline scaling grid is 25/50/100.
The existing five baselines are unchanged. See
[BUDGETED_EVIDENCE.md](BUDGETED_EVIDENCE.md) for the implemented objective, rules,
source bindings, CLI and validation boundaries.

The older matrix below is retained for existing artifact lineage and operations:

| Question | Retrieval source | Evidence organization | Maintained runner |
|---|---|---|---|
| Does literature-scale Starling retrieval outperform ChEMBL retrieval? | ChEMBL versus Starling | matched direct, full-flat, and full-mechanism views | `starling_benchmark_matrix.py` |
| What happens as progressively more distant evidence becomes available all at once? | Starling | one-shot cumulative full-flat levels | `run_conditioned_assay_family_curve.py` |
| Can the model revise an earlier decision when new indirect evidence is appended? | Starling | append-only progressive state and evidence | `run_conditioned_assay_progressive_curve.py` |
| Does mechanism grouping help relative to one flat evidence branch? | ChEMBL or Starling | full-flat versus full-mechanism over the same evidence | `starling_benchmark_matrix.py` |
| Does evaluation difficulty depend on structural generalization? | matched source and reasoning setup | scaffold versus parent-grouped random split | the same runner with the split flag |
| Does molecular identity visibility change behavior? | matched retrieval payload | `identity_blind` versus `deployment_visible_prefetched` | shared visibility contract and replay |

`deployment_visible` remains available for end-to-end agentic deployment
experiments. The paper-facing matched visibility comparison uses
`deployment_visible_prefetched`, which restores identities/structures while
replaying the blind retrieval and tool payload.

The historical one-shot family curve is identity-blind. The matched full-flat
control and append-only progressive curve both use deployment-visible-prefetched
inputs; they compare reasoning organization over identical cumulative cards and
tools. Neither comparison is a matched visibility ablation. Blind and visible
settings remain supported by the source/reasoning matrix.

The existing `plot_assay_retrieval_curve.py --conditioned-progressive-config-comparison`
accepts both matched full-flat and progressive roots through repeated
`--conditioned-progressive-config-task-root CONFIG:TASK=PATH` arguments. The frozen
prompt profiles may differ only after cumulative prepared-input, selection, and
retrieval hashes match. Matched organization checks accept the current v5 profile
and the frozen legacy v2 profile, while retaining exact prepared-input checks.
Add `--performance-only` for a Macro-F1 grid (up to three
tasks per row) with None and the five right-side baselines; omit it to retain
resource panels. Settings use the same colors and markers across task panels.
Cross-model organization plots use model colors and organization line styles;
explicit prompt comparisons use configuration colors.
Use `--conditioned-baseline-root` to select baselines from the same evaluation subset.

For the six-task L3+ policy figures, use the registry suite
`l3_parent_disjoint_matched_full_flat_20260914` and its
`valid_comparison_figure` / `test_comparison_figure` entries. Their receipts retain
hashed source metrics, plotting tables, `render_arguments`, and a `render_command`
that calls this same plotter function. No temporary script is required. These
mixed-policy/historical overlays use their explicit source audits; they do not
relax the ordinary CLI's matched-input checks. The valid v5 full-flat reference
has additional runtime/source differences, and test leaves that setting absent.

Suite-local `coordinator.py` files and frozen runtime directories are execution
provenance, not additional maintained entrypoints. Resume only from the suite's
registered plan and corresponding frozen runtime; new experiments use the shared
family/progressive runners. Do not delete these hashed snapshots as dead code.

## Shared request admission across launchers

When frozen runtimes require several launcher processes, use one host-local
admission broker and set the same `admission_socket` in every participating
provider configuration:

```bash
python -m tools.chembl_tool.common.request_admission \
  --socket /local/tmp/txagent-request-admission.sock --capacity 1024
```

Normal calls and parallel retry attempts share these permits. The provider's
request deadline starts after admission; `execution_provider.admission_wait_seconds`
records the separate wait. Connect to the direct model endpoint without a second
proxy queue. Cancellation awaits HTTP cleanup before releasing the permit;
process exit closes its lease connection. The async client cancels its HTTP work
if the broker connection is lost. Stop participating launchers before restarting
the broker. Client cancellation does not guarantee immediate server-side abort.
Sending `status\n` to the Unix socket returns capacity, active/waiting counts,
peak active permits and total grants. This is transport configuration only;
prompt, retrieval and generation contracts remain unchanged.
For multiple physical endpoints, use one broker per endpoint, shared by every
launcher using that endpoint; their capacities sum to the global budget. Current
addresses and allocations belong in the suite's provider config and run plan.

## Canonical benchmark

All active evaluation data lives at:

```text
data/conditioned_benchmark/<Task>/{scaffold,random}/
```

The two split schemes contain the same molecule-condition rows and labels:

- `scaffold`: parent- and Bemis-Murcko-scaffold-disjoint;
- `random`: exact parent-grouped 80/10/10 allocation; parent-disjoint but not
  scaffold-disjoint, with all conditions represented in all three splits.

The benchmark creation code, task-specific voting boundaries, condition and
parent construction, multi-vote held-out priority, complete lexicographic split
objectives, deletion-repair behavior, leakage policy, and required audit files
are frozen in:

```text
tools/chembl_tool/common/starling/CONDITIONED_BENCHMARK.md
data/conditioned_benchmark/manifest.json
data/conditioned_benchmark/migration_receipt.json
data/conditioned_benchmark/random_split_receipt.json
```

Historical molecule-only, `selected_vN`, task gold-vN, and source-build names
are provenance only. Runners and plots must use the canonical roots above.

## Maintained entrypoints

The requested Bioavailability/Skin legacy configuration is registered under
`replicate_suites.bio_skin_legacy_configuration_20260913`. Its `restoration.json`
contains verified frozen inputs and four commands for valid/test ×
progressive/full-flat, using the existing family runner from a pinned runtime.
Use those commands to restore the old prompt, tool context and prior together;
do not regenerate the prior or tool cache. This experimental package is not a
second maintained runner. It retains the confirmed Skin MDAM retrieval repair
and records preparation only. Later evaluations using this frozen runtime are
registered in the L3+ progressive/full-flat suites; see `RESULTS.md` for scores.

### Benchmark and evidence construction

```text
tools/chembl_tool/paper_experiments/rebuild_current_starling_retrieval.py
tools/chembl_tool/paper_experiments/export_current_starling_level_records.py
tools/chembl_tool/common/starling/publish_conditioned_benchmark.py
tools/chembl_tool/common/starling/build_conditioned_random_split.py
tools/chembl_tool/common/starling/build_record_supported_benchmark.py
tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py
tools/chembl_tool/paper_experiments/build_assay_family_catalog.py
tools/chembl_tool/paper_experiments/build_conditioned_source_family_purity.py
tools/chembl_tool/paper_experiments/build_bbb_source_family_purity.py
tools/chembl_tool/paper_experiments/build_bioavailability_vote_pure_source.py
```

`CURRENT_STARLING_RETRIEVAL.md` 是当前 Starling records、purity overlays、family catalogs 和
scaffold/random indices 的唯一操作说明。当前 retrieval 只冻结并读取实验实际使用的最新 Stage-03
canonical snapshot；统一入口负责
恢复、重建和逐哈希验证，不依赖个人 checkout。该文档也区分静态 record membership、split-specific
indexed representative cards 和 per-query model-visible cards，并说明 GitHub 协作数据。旧版本号只作为
run/receipt provenance 保留。

当前 family naming contract 是 `source_group_id -> family_key -> level`。新配置只通过
`common.experiment_retrieval.evidence_family()` 声明 canonical `family_key` 与 source-native groups；冻结
artifact 中的 `family_id`/`endpoint_group` 仅由兼容边界保留，跨 ChEMBL/Starling 对齐必须使用
`family_key`，不能使用 level number 或 `Mechanism.tier_N`。跨 source 对照应直接以 `family_key` join，
同时保留各 source 自己的 level。

Task-specific voter and source-review entrypoints are indexed in
`CONDITIONED_BENCHMARK.md`; they are intentionally not duplicated here.

### Model experiments

```text
tools/chembl_tool/paper_experiments/molecular_evidence_agent.py
tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
tools/chembl_tool/paper_experiments/run_conditioned_assay_family_curve.py
tools/chembl_tool/paper_experiments/run_conditioned_assay_progressive_curve.py
```

### Molecular tools and fresh priors

The three existing tools and `/tools/batch` remain the only molecular service
entrypoints; deployment and caching are documented in [the service README](../../service/README.md).
`tools/service/functional_group_tree.py` is an internal renderer, not a fourth tool
or a launcher. New progressive/matched-full-flat inputs use
`progressive_evidence_revision.v5`; retain historical results under their
recorded profiles.

Use the existing progressive runner's `--fresh-query-priors` to generate new
single/None branches, or add `--query-priors-only` to stop after those branches.
All six tasks are supported: BBB/Bioavailability/Skin reuse their native branch
builders and validators, with the shared query pool and retry races.
`--single-source-root` and `--query-prior-source-root` explicitly select frozen
priors; updating the tool service does not update that prior reasoning. Fresh
priors receive the query's complete properties text and tree section; progressive
and matched full-flat expose separate query/neighbor tree fields. Both reuse the
same service computation and caches. No neighbor-label branch is added.

The bounded tree diagnostic and preceding trace audit are linked from
`diagnostics.functional_group_tree_20260913` in the registry. Its scripts are
retained as execution snapshots, not additional runnable entrypoints. The
observed label results did not improve; these inspected cases are not a formal
benchmark rerun. See [RESULTS.md](RESULTS.md) for the outcome.

### Offline record diversity

The maintained preprocessing entrypoints are `prepare_record_diversity.py`
(inventory/frozen scope), `benchmark_record_diversity.py` (bounded speed probe),
and `run_record_diversity_extraction.py` (resumable annotations). Shared encoding
and selection live in `common/record_diversity.py`, `record_numeric_features.py`
and `record_selection_context.py`. See
[RECORD_FEATURE_SELECTION.md](RECORD_FEATURE_SELECTION.md) for inputs, API
boundaries, cache/overlay reuse, and known limitations.

The completed six-task valid comparison uses semantic `molecule_local.v1`;
the runner's opt-in `feature_based.v1` is a separate lexical apricot baseline.
Semantic experiments currently use registered prepared arms and the existing
progressive/matched runners. Suite-local preparation/repair scripts and frozen
runtimes are provenance, not extra maintained launchers. The newest endpoint/UGT
repair has selection replays only; completed performance and exact artifact
links are in [RESULTS.md](RESULTS.md#record-diversity-current-results).

### Analysis and figures

For fixed-Direct indirect-budget sweeps, the shared plotter accepts
`--indirect-budget-csv` with wide columns `task,10,25,...`. It plots all six tasks
and their equal-weight mean, validates completeness, and exports PNG/SVG/PDF
plus a source-hash receipt. [User-supplied Ours sweep and reproduction command](../../../outputs/paper/analysis/ours_indirect_budget_20260925/README.md).

```text
tools/chembl_tool/paper_experiments/summarize_starling_benchmark.py
tools/chembl_tool/paper_experiments/analyze_starling_direct_significance.py
tools/chembl_tool/paper_experiments/analyze_starling_best_agent_baselines.py
tools/chembl_tool/paper_experiments/analyze_progressive_trace_adoption.py
tools/chembl_tool/paper_experiments/plot_starling_model_comparison.py
tools/chembl_tool/paper_experiments/plot_assay_retrieval_curve.py
```

`plot_starling_model_comparison.py` is the single model/source/visibility
comparison figure entrypoint. `plot_assay_retrieval_curve.py` is the single
level-curve/resource figure entrypoint. Do not add a one-off plot module for a
new model row or a new progressive rerun.

### Baselines

```text
baselines/minimol/run_train_cv.py
baselines/minimol/run_embedding_knn.py
baselines/conditioned_knn.py
baselines/structure_knn/run.py
```

Condition-aware MiniMol is the paper-facing trained baseline whenever a task
has accepted condition groups. The two KNN families retain both
same-condition-then-null and unrestricted-train variants.

## Shared contracts

- Retrieval source, evidence organization, split, visibility, model, prompt,
  input, index, and family-catalog identities are independent manifest axes.
- A stored prediction may be reused only when all LLM-visible semantic hashes
  match. Directory names or matching molecule counts are insufficient.
- `none` may reuse a compatible frozen single/final state. Evidence-equivalent
  levels may carry forward without a model call.
- The current cumulative-family/progressive evaluation uses
  `scaffold_disjoint` for scaffold and `parent_disjoint` for random. The
  source/reasoning matrix exposes its identity policy explicitly in the
  manifest; a split name alone never silently changes an old matrix contract.
  Every policy prohibits the query parent itself.
- Evaluation indices remove valid+test benchmark-defining direct rows while
  preserving eligible non-direct mechanism evidence from held-out parents.
- Morgan retrieval uses the configured `0.30` minimum similarity. The current
  progressive protocol also applies its monatomic-element collision guard.
- Identity-blind prompts remove query/neighbor structures, identifiers, and
  source names. Visible-prefetched prompts restore them while holding retrieval
  and tool outputs fixed.
- LLM outputs pass shared JSON validation. Validation retries and provider
  failover are recorded in the trace; they do not alter label policy.
- The scaffold-valid progressive record-card budget ablation uses only the
  2/1, 4/2, and 8/4 roots and receipts registered in
  `current_conditioned_results.json`. BBB has current strict-voter-L1 results
  for all three budgets. Bioavailability 8/4 has two exact-contract full-curve
  runs, so the combined figure reports their mean and observed min-max range.
  Skin source-purity v7 applies the strict target-scope gate to gold and
  L1/L2/L3. A final reproducibility rebuild added 37 MDAM nonvoter outcomes to
  L2 without changing the benchmark split; this changed the index-selected
  surface for 1, 2, and 3 validation queries under 2/1, 4/2, and 8/4,
  respectively. The completed agent roots are retained as last-complete
  references pending targeted replay. The matched MiniMol/Morgan baselines
  remain current because their train/validation rows did not change.
- The ablation remains on the shared progressive runner: card limits are CLI
  parameters (`--initial-card-limit`, `--delta-card-limit`), while molecule
  quotas, selection order, prompt profile, and append-only semantics are fixed.
  Multi-configuration figures remain in the shared assay-curve plotter. Repeat
  the same `CONFIG:TASK=PATH` argument with a distinct path only for a verified
  exact-contract replicate; omitted cells are recorded explicitly. A
  selected-surface receipt is required whenever retrieval lineage hashes differ.
  The saved combined PNG/SVG, TSVs, and summary are under the following root;
  its Skin cells are stale until the targeted replay completes:
  `outputs/paper/analysis/progressive_record_card_budget_2_1_4_2_8_4/`.

Detailed group-level data flow is in [PIPELINE.md](PIPELINE.md). The cumulative
level and progressive protocols are in
[ASSAY_LEVEL_RETRIEVAL.md](ASSAY_LEVEL_RETRIEVAL.md).

## Current artifact registry

The only machine-readable index for paper-facing result roots and freshness is:

```text
tools/chembl_tool/paper_experiments/current_conditioned_results.json
```

It distinguishes:

- `current`: benchmark and retrieval hashes match;
- `current_via_zero_change_retrieval_receipt`: a split-scoped audit proves that
  every model-visible selected retrieval surface is unchanged across a
  lineage-only source/index repair;
- `last_complete_reference_pending_*`: a complete historical reference is
  retained because the corresponding current-condition matrix is incomplete;
- `stale_after_*_requires_*_replay`: a retained run is complete, but a later
  source/index repair changed model-visible retrieval for a bounded query set;
- `stale_retrieval_index_requires_targeted_replay`: labels may still match, but
  changed retrieval inputs prohibit publishing the old score as current.

At the 2026-09-05 snapshot, BBB scaffold-valid has fresh 2/1, 4/2, and 8/4
runs over the strict-voter-L1 v6 index; its random v6 index is built but still
requires LLM replay. Skin has unchanged matched trained/KNN
baselines, but the final MDAM L2 rebuild leaves 1/2/3 scaffold-validation
queries pending targeted replay for 2/1, 4/2, and 8/4. Its random agent cell
still requires full replay. The exact affected prefixes are frozen in
`receipts/skin_scaffold_valid_mdam_family_rebuild.json`.

Bioavailability scaffold-valid has fresh 2/1 and exact 8/4 replay
results. Its retained 4/2 curve is current through the nitrendipine
selected-surface zero-change receipt. The rejected L1-only 0.7148 run used a
different prompt contract and is neither a replicate nor a variance estimate.
Bioavailability random remains unaudited and requires a separate change audit
or replay. Its scaffold-test baselines have been retrained after the train-row
correction; older valid baseline scores remain pre-fix references.

The progressive runner accepts `--evaluation-subset test` with an explicit
output root and a complete test-specific `--single-source-root`. It uses the
same valid+test-heldout-filtered index as validation, records the selected
subset, and rejects cross-subset resume. Freeze the validation protocol before
running test. `--parallelism` is the shared pool size;
`--parallelism-per-task` optionally caps each task within that pool. The default
endpoint budget remains 512; an authorized override must be explicit in
`--endpoint-concurrency-budget`. Task caps are enforced before submission so
capped tasks do not occupy workers needed by other ready queries.

Both methods' three-task scaffold-test 4/2 curves have three complete runs,
including the original as replicate 1; all five matched baselines are complete.
Results, figures, tool audits, and the BBB/Skin L2→L3 trace analysis are registered
in `current_conditioned_results.json` and summarized in `RESULTS.md`. Earlier
valid/random tool outputs have not been audited against the serialized MolGpKa
correction; source-lineage compatibility alone does not establish tool equivalence.

The opt-in expanded budget uses `--initial-card-limit 5 --delta-card-limit 5
--combined-molecule-limit 10` in the shared progressive selector/runner. Later
levels rank new and active molecules together and only append newly unlocked
cards. Zero combined limit preserves the default separate quotas. The registered
`expanded_budget_progressive_full_flat_20260915` suite additionally freezes
mixed L1/L2 scaffold and L3+ parent exclusion, similarity 0.3 for every task,
and matching progressive/full-flat prepared inputs. Its run plan and execution
status define the authorized task/request limits and phase order.

As of 2026-09-16 01:11 UTC, that suite and its fixed-25/no-threshold successor are
`paused_by_user`; neither coordinator is running. All 12 expanded progressive
runs and five matched full-flat runs are verified. The successor retains verified
prepared inputs and partial progressive valid checkpoints for Ames, DILI and
Carcinogens; no complete split has passed final verification. Completed paired curves and scores
are linked from [RESULTS.md](RESULTS.md); successful checkpoints are retained.
Resume only on a subsequent user instruction, using the frozen suite plan and
the resume guidance in its artifact-root README.

For a controlled one-shot comparison, the existing family runner accepts
`--matched-progressive-root PATH --output-root NEW_PATH [--prepare-only]`.
It inherits the source split/subset and exact cumulative cards/tools, validates
current lineage, and judges every level independently without previous decisions.
Frozen inputs are copied atomically; altered destination inputs abort resume
instead of being overwritten alongside existing predictions.
After a source repair, a single matched run can add
`--level-reuse-source-root OLD_RUN`. The shared reuse gate requires the same
organization, model, generation settings, benchmark and priors, then compares
every prepared model input. Only unchanged prefixes are copied; the first
changed level and its successors run again. Each repetition must point to its
own old run. This option cannot be combined with `--replicate-ids`.
This visible matched control is distinct from the historical identity-blind,
per-assay full-flat curve; see `ASSAY_LEVEL_RETRIEVAL.md` for the contract.

Progressive and matched full-flat retry failed queries automatically after a
cooldown, skipping successful level checkpoints. Defaults are three extra rounds
(`--max-stage-requeues 3`) with 60-second initial backoff (`--retry-delay-s 60`),
doubling up to 900 seconds. Read `execution_status.json` on demand; exhausted
retries end as `needs_attention` with failure history retained. Restarting the
same command resumes checkpoints; no Codex polling loop is required.

For frozen-input repeats, use the same family runner with
`--matched-organizations progressive full_flat --replicate-ids 2 3`.
The suite runs the four jobs sequentially, with all selected tasks sharing each
job's prompt budget. It creates `replicate_02/{progressive,full_flat}/` and
`replicate_03/{progressive,full_flat}/`; source level predictions are never copied.
The original completed run remains replicate 1. None/query priors and tools stay
frozen, so these repeats measure level-reasoning variability conditional on those inputs.
To overlap complete runs, set `--concurrent-runs N`; each run uses `--parallelism`
slots, and their product must fit `--endpoint-concurrency-budget`. Provider-pool
capacity must also fit each run's slot count. The default remains sequential.

For a different model, add `--refresh-query-priors` to a matched suite. The same
runner generates fresh single/None branches under `single_cache/` using the
source's exact retrieval and prefetched tools, then replaces only the query prior,
None result and source index in the frozen level inputs. Both organizations share
the new model's priors. Source and resulting prepared hashes are recorded separately;
completed priors are hash-checked on resume. The prior stage shares the global and
per-task query caps, with sequential single/final calls and the legacy branch JSON
validation policy; six-way racing applies to subsequent level retries.
Tasks using `conditioned_query_priors.v1`, including Ames, refresh both branches
through the shared fresh-prior runtime, with the configured failure-only racing.
Only frozen tool retrievals are copied; old-model single/None predictions are not.
`--provider-pool-config PATH` forwards routing options without changing prompts and
freezes the public provider configuration in the suite manifest. Credentials are
read from `--env-file` (default `.env`) and are never written to receipts.
The one-run V4 Pro 0813 test suite is registered as
`replicate_suites.scaffold_test_pro_0813_4_2`; its `launch_receipt` contains the
exact command and `provider_pool.json` contains the
price-first routing policy and price caps. Resume with that command.

To plot repeats, register each run with the existing plotter's repeated
`--conditioned-progressive-config-task-root CONFIG:TASK=PATH` option and select
`--replicate-interval sd` for arithmetic mean ±1 sample SD (`ddof=1`) of
run-level scores. At least two runs per curve point are required. The default
remains observed min-max (`range`). Neither interval is a confidence interval;
fixed None and baseline references remain single points. The completed three-run
scaffold-test comparison, including all levels and baselines, is registered under
`replicate_suites.scaffold_test_4_2.comparison_analysis`.
The plotter compares actual prepared-input hashes across every replicate, not
only configuration/index hashes. Configuration comparisons read unique task/run
artifacts with up to eight threads for small comparisons or sixteen processes
for more than eight task/run inputs, then aggregate in the supplied order;
all completeness, input-equivalence and lineage checks still apply.
The registered `comparison_receipt` stores the
exact plotting command, individual run scores and verification results; the
suite's `launch_receipt` stores the run command. Use these receipts for exact
reproduction rather than creating another launcher or duplicating root lists.

Ames uses the same level-curve plotter: add
`--conditioned-progressive-config-task-root CONFIG:ames=PATH` and
`--conditioned-progressive-baseline-root ames=outputs/baselines/ames_scaffold_valid_v2`.
`--plot-tasks bbb_martins bioavailability_ma skin_reaction ames` fixes the panel
order and leaves unavailable current task curves explicitly empty. The progressive
overview also accepts Ames task roots and scales its width with the column count.
All five Ames v2 baselines cover 274 rows: condition-first KNN fills remaining
slots from unrestricted train molecules. Historical partial-coverage scores can
still be omitted with `--omit-mismatched-progressive-baselines`.
`--omit-baseline-tasks TASK` keeps None but excludes unavailable or
stale baselines for that task. The current four-column valid figure and exact
generation command are registered under
`result_families.record_card_budget_ablation.ames_valid_extension`.

To overlay different models in the same level/baseline figure, use the existing
configuration comparison with `--allow-model-comparison`. This additionally
requires identical prepared evidence/tools, card selection and retrieval lineage.
Model-derived priors/None and model-specific temperature/reasoning effort may
differ; generation differences are recorded in the comparison audit and must be
identified in the figure note. Same-model generation settings remain strict.
Full-flat/progressive within each model
must still share exact prepared inputs, including priors. None references are checked within
each model and plotted separately, while matched baselines appear once.
`--replicate-interval sd_if_repeated` shows sample SD for repeated configurations
and no interval for single runs; `sd` still rejects single-run configurations.
The completed Flash (three runs) versus Pro 0813 (one run) figure is registered in
`replicate_suites.scaffold_test_pro_0813_4_2.comparison_analysis`, with its exact
command, source checks and PNG/SVG hashes in the associated figure receipt.
This compact comparison (PNG/SVG, metric/reference TSVs, summary and receipt)
is tracked in Git for GitHub review. Raw run directories remain local artifacts.

`--retry-race-width 6` makes JSON repair calls and failed-level requeues race six
identical requests. The first fully validated response wins; remaining async HTTP
requests are cancelled and cleaned up before the caller publishes a single output.
Every request counts against both the global and per-task caps. SDK transport
retries must be zero. Full race receipts live in each level's `retry_races/`,
including invalid responses, failures, and cancellation outcomes; server-side
abort propagation is endpoint-dependent. Retained-output token metrics exclude
discarded/cancelled attempts; their available usage is in these receipts.

`suite_manifest.json` freezes the repeat plan; `suite_status.json` identifies the
active root, whose `execution_status.json` has query progress. An exclusive
`launcher.lock` prevents duplicate suite launches. Exhausted failures remain
`needs_attention`, while later suite jobs still run. No periodic Codex monitor is used.

Source-repair diagnostics use `--record-review-overlay` before cumulative
selection. Hash-bound `correct` decisions may repair endpoint/measurement/context
fields while preserving raw support, molecule identity and family membership;
corrected cards receive new content-addressed IDs. `--query-structure-map` accepts
an input-hash-bound `verified_query_structure.v1` map for DILI/Carcinogens. It uses
verified source-parent structures consistently for retrieval, tools and reasoning
while retaining frozen benchmark rows and leakage identities. Pass the same map
to fresh-prior preparation and evidence preparation; stale prior structures fail
closed. Task-specific rules under `--evidence-grounding` are pinned in manifests
and preserve the existing progressive update mechanism. Earlier Carcinogens rounds and their source-only decisions are indexed by the
registry's `cleanup_history` receipt. The final `retrieval_final` snapshot already
materializes the completed corrections; ordinary new runs use it without a
historical `--record-review-overlay`.

Ames fresh single/None preparation is available through the progressive runner
with `--tasks ames --fresh-query-priors` and an explicit output root. Task-local
`query_prior.py` owns its scientific prompt; tools, validation and concurrency
use the shared implementations. Completed compatible priors are resumable.
Changing `--parallelism` or `--endpoint-concurrency-budget` preserves compatible
checkpoints and records the previous settings in `execution_history`; model,
prompt, output-limit, and dataset changes still reject reuse.
The Ames full-flat control requires `--matched-progressive-root`; it does not
use the historical unpaired family-curve branch.

The family runner owns matched full-flat and repeat-suite CLI orchestration;
the progressive runner owns their shared query queue, retry rounds and summary.
`common/reasoning_race.py` is an internal adapter for cancellable first-valid
races through `openai_provider_pool.py` and `openai_reasoning_client.py`.
`reasoning_validation.py` remains the single validator. There is no separate
retry or replicate executable.

## OpenRouter Batch transport

The shared progressive and matched-family runners also accept provider
`transport: "openrouter_batch"`. The default remains `chat_completions` for PARCC
and ordinary OpenRouter requests. The reusable example is
[`provider_pools/gpt56_sol_openrouter_batch.json`](provider_pools/gpt56_sol_openrouter_batch.json).
Use the base model ID (`openai/gpt-5.6-sol`, without `:batch`); the transport calls
`POST /api/v1/batches` and polls the returned job ID. Set
`--retry-race-width 1 --transport-max-retries 0` for the progressive runner, or
`--retry-race-width 1` for matched-family runs. Batch pools require
`max_failovers: 0` and cannot mix with realtime providers.

`common/openrouter_batch.py` collects ready requests without changing
their messages. For the independent family `--baseline-suite` with exactly one
Batch provider, every prepared, still-needed request is queued ahead of result
processing. `parallelism` and `max_inflight` bound local result consumers, not
remote submissions: a slow batch does not hold back later independent queries.
The suite validates prepared hashes and skips valid saved outputs; journal
recovery reserves existing requests before creating new ones. No additional
runner or CLI flag is needed. Progressive/matched runs retain their bounded
logical requests and dependency ordering. In all modes,
`batch_options.max_requests` and `max_bytes` bound each submitted batch. The
example uses 128 result consumers and at most 128 requests/16 MiB per batch;
these are local settings, not provider limits. Multiple remote batches can be
pending together. Size the runner's global parallelism and provider capacity
together when running larger suites. The existing runner controls each query's
progressive dependencies; it never
submits L2 before that query's valid L1 result. Different tasks/queries may share
a batch. Remote completion windows, rather than realtime HTTP timeouts, determine
latency. GPT generation settings are explicit provider fields: `temperature: null`
omits sampling temperature, and `reasoning_effort: "medium"` requests reasoning.
The manifests record the actual settings; provider-returned reasoning and usage
are preserved, without assuming GPT exposes its full internal reasoning text.

Each run owns `batch_jobs/<provider>/job_*.json`: exact requests, remote IDs,
status/results and billing usage. `llm.batch` links each output to its custom ID
and journal. Repeating a stopped run recovers submitted results before making new
calls. A local stop does not cancel remote jobs. An interrupted POST with an
unknown outcome blocks resubmission until its remote job is reconciled; poll
errors never cause another submission. Keep journals with the reasoning traces,
and never use one journal directory concurrently from two clients.
Optional `batch_options.submit_interval_seconds` spaces submissions; independent
suites sharing a key should divide its submission allowance. An explicit HTTP
429 keeps its requests queued for retry after at least 60 seconds and any longer
server reset/retry time. Response headers are read before the body so a truncated
429 body remains a known rejection. Other interrupted POSTs remain ambiguous and
must be reconciled before restart; they are never automatically resubmitted.
Unchanged remote responses are not rewritten to disk. A journal's modification
time therefore reflects a status/result/error change, not the latest poll;
use the remote job status and runner process to diagnose a quiet queue.

For model comparisons, use the existing matched suite with
`--matched-organizations progressive --replicate-ids 1 --refresh-query-priors`.
This freezes source tools/cards, regenerates the new model's single/None, and
builds every progressive state with that model. It does not reuse DeepSeek
judgments as GPT judgments. The registered Carcinogens valid Sol Batch run is
`replicate_suites.carcinogens_scaffold_valid_gpt56_sol_batch_20260913`.
Matched prior replacement accepts the prompt-facing tool receipt after frozen
retrieval and query-identity validation. Tool content, status, warnings, errors
and supplied arguments must match; latency/cache metadata may differ. The
progressive input retains the original full tool receipt.

Carcinogens also offers the explicit `--prompt-adaptation gpt_calibration.v1`
option. It keeps the v5 payload/schema and changes only the task wording,
uncertainty-to-binary decision guidance, and symmetric analog-evidence weighting;
the single prior receives one matching instruction. Scientific wording lives in
`tasks/carcinogens/experiment_config.py:get_adapted_prompt_contract`, not in the
provider client. The default is unchanged; model names never select a prompt
automatically, and other tasks reject this profile until explicitly supported.

For this adaptation, add the option to the matched example below and choose a
new output directory. `--refresh-query-priors` regenerates both single and None
from the frozen tools before progressive/full-flat reasoning. The fresh
progressive runner instead requires `--fresh-query-priors`. Do not combine this
profile with the separate evidence-grounding ablation. Suite, prior and prepared
manifests pin the profile and serialized task-contract hash; changed profiles
cannot reuse prior checkpoints or completed levels. This is a prompt ablation,
not an unchanged-prompt model comparison: ordinary comparison plots reject
different reasoning policies. For an explicit adaptation comparison, the existing
plotter accepts `--allow-prompt-comparison` (plus `--allow-model-comparison` when
models differ). It verifies identical prepared evidence/tools after excluding
generated query priors and only the declared `prompt_adaptation`, keeps every
other policy invariant, and shows a separate None for each configuration.
Replicates still require identical prompt contracts. The completed adapted
Carcinogens valid run used normal `chat_completions`, tracked
by `replicate_suites.carcinogens_scaffold_valid_gpt56_sol_prompt_v1_api_20260914`.
Its launch, completion checks and provider configuration are linked from the
registry. It used the normal API at 128 concurrent requests. The stopped
Batch attempt and 243 hash-verified reused single outputs remain linked as
provenance; remote cancellation was unavailable (HTTP 404).

Example for a new matched Carcinogens valid run (change task, frozen source and
output directory for other experiments; copy the provider example to change its
model or capacity):

```bash
python -m tools.chembl_tool.paper_experiments.run_conditioned_assay_family_curve \
  --tasks carcinogens \
  --matched-progressive-root outputs/paper/reasoning_prompt_v5_scaffold_20260913/valid/standard/progressive \
  --matched-organizations progressive --replicate-ids 1 --refresh-query-priors \
  --output-root outputs/paper/carcinogens_sol_batch_replicate \
  --model openai/gpt-5.6-sol --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY_Mark_1 \
  --provider-pool-config tools/chembl_tool/paper_experiments/provider_pools/gpt56_sol_openrouter_batch.json \
  --parallelism 128 --parallelism-per-task 128 --endpoint-concurrency-budget 128 \
  --retry-race-width 1 --max-tokens 20480
```

Repeat the same command/output directory to resume. Preserve the submitted model,
generation settings and frozen inputs. Use `suite_status.json` for the active
stage; `single_cache/execution_status.json` counts complete single+None query
chains, and `replicate_01/progressive/execution_status.json` counts complete
progressive query chains. Count billing once per remote batch from `usage.cost`;
carry-forward level outputs are not additional model calls.

## Documentation map

- [Conditioned benchmark contract](../common/starling/CONDITIONED_BENCHMARK.md)
- [Current Starling data, rebuild, and sharing](CURRENT_STARLING_RETRIEVAL.md)
- [Pipeline and visibility](PIPELINE.md)
- [Assay/family-level and progressive retrieval](ASSAY_LEVEL_RETRIEVAL.md)
- [Current result status](RESULTS.md)
- [Trace retention](TRACE_RETENTION.md)
- [Current progressive trace viewer](../../trace_viewer/AGENTS.md)
- [MiniMol and KNN baselines](../../../baselines/minimol/README.md)
- [ClinTox source contract](../tasks/clintox/CLINTOX_BENCHMARK.md)

Historical experiments that are not part of the paper scope must not be linked
as active entrypoints from this page. Git history and explicitly retained
receipts provide provenance for removed no-go launchers and transient smoke
runs.

## Portable TDC BBB indirect ablation

The [task guide and Codex handoff](../tasks/bbb_martins/INDIRECT_ABLATION.md#codex-handoff) identify the official 406-row TDC test set and current assay-aware filter. The portable bundle is in `artifacts/chembl_tool/bbb_indirect_20260924/`. Historical one-pass and two-stage scores used the retired filter; the current filter has offline verification only. Preparation remains task-specific and `--run-prepared` uses the shared family runtime. Registry: `budgeted_evidence_protocol.tdc_bbb_indirect_test_iteration_20260924`.
