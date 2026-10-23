# Results and reproduction sources

## Frozen paper comparison

[Full-precision baseline results](paper/baselines.csv) contain **306 cells**:
198 ML cells (66 each for conditioned, TDC-only and TDC + Starling training)
and 108 naive-LLM cells (three backbones, two benchmarks, six tasks and three
settings). Filter `family=baseline` for the ML appendix; `training_pool` is empty
for LLM rows. Each row identifies its original prediction file by SHA-256.
[Source bindings](paper/sources.json) pin the original tables, exported files and
benchmark cohort hashes. Raw predictions, model weights and datasets are not
included in this compact result export; their distribution remains pending.

The export preserves the original numeric fields. All 306 cells were checked
against saved predictions, recomputing Macro-F1, accuracy and, for ML, AUROC.
This is a saved-result check, not new training or inference. LLM AUROC remains
empty because hard decisions are not continuous scores. Current TDC Skin and
Carcinogens use the seed-42 replacements; the nine corrected Skin auxiliary
cells and both RASAR variants are included.

[Ours reported scores](paper/ours_reported.json) preserve the supplied 72
Direct / Direct + Indirect values for the three backbones. None shares the
corresponding baseline. The file is copied byte-for-byte from the recorded
source; it does not establish prediction-level verification or matched budgets.
The manuscript's Direct=10 description versus the naive baseline's Direct=50
and the missing Ours budget provenance remain unresolved. Additional Gemini
source-provenance and cost reconciliation remain deferred.

All scores retain their frozen scientific protocols. They are not evaluations
of the revised retrieval rules in the integration branch. Six-task means are
unweighted; the paper's grouped ML bars use descriptive within-group test
maxima, not validation-selected method winners. Keep individual methods and
training pools visible when interpreting those summaries.

[Baseline recipes](../experiments/baselines/README.md) and the
[MiniMol guide](../predict/baselines/minimol/README.md) describe maintained ML
entrypoints. The [integration record](release_refactor.md) maps these to the
original configurations and distinguishes replay, retraining and pending work.
The [offline LLM verifier](../predict/prompts/README.md#frozen-paper-baseline)
checks frozen requests and responses through the migrated protocol adapter.
The maintained exporter below derives the appendices, figure values and the three
current manuscript PDFs. Do not manually edit frozen score tables or treat a
model's current defaults as the paper recipe.

Budget scans, Direct-3 controls and later scaffold-disjoint indirect follow-ups
remain separate experiments in the preserved source registry. They are outside
the current manuscript migration scope unless a specific paper item uses them.
Existing historical sources remain preserved.

## Rebuild paper tables

```bash
python -m analysis.prediction.paper_results \
  --source docs/paper \
  --output outputs/analysis/prediction/paper-tables-review
```

Use a new output directory. The command verifies source hashes and exports seven
CSV tables: ML, naive LLM, Ours reported, figure points, figure means, component
ablations and their means. It performs no inference or training. Missing tasks,
duplicate cells, mixed ML training pools and modified inputs fail validation.
[Display rules](paper/display.json) preserve the original group membership and
backbone bindings; each ML winner and tie is recorded. Full-precision baseline
and Ours values are unchanged. This is deterministic table derivation, not an
additional prediction audit.

Add `--figures` to the same command to render `main_results_starling.pdf`,
`main_results_tdc.pdf` and `mean_results.pdf`. Run from the repository root.
Rendering requires Matplotlib (verified with 3.10.8) and `pdflatex` with the
`standalone` and `graphicx` packages. CSV-only export requires neither dependency.
The two benchmark mean panels are retained in `conditioned/` and `tdc/` for
composition; temporary TeX compilation files are automatically removed.
The receipt records source/output hashes, renderer/template hashes and the
Matplotlib version. The original colours, task order, ML pooling and mean
composition are preserved. All three PDFs match the current manuscript pixel
for pixel at a 2400-pixel render width in the verified environment; PDF metadata
and binary hashes may differ. Font/library changes can affect rendering elsewhere.
The exporter only reads compact saved scores and does not need raw predictions,
model weights, services or the original checkout. Historical horizontal/combined
layouts and budget-scan plots are outside this maintained paper figure path.

[Component ablations](paper/ablations.csv) contain the 30 task/condition values
from Joseph's archived `gold_v1_ablation_test_20260925_v3.tsv`. The archive and
member hashes are pinned in [sources.json](paper/sources.json). Its five means
round to 0.721, 0.704, 0.714, 0.711 and 0.714, matching the manuscript. The source
TSV is rounded to six decimals; raw predictions have not been re-audited here.
Its control was chosen after viewing test results, and the Oral/Ames control
runs differ from the ablation runs; preserve that descriptive comparison limit.
The manuscript's sixth, similarity-only value (0.678) numerically matches the
main naive DeepSeek 50/50 baseline. Its exact ablation protocol binding remains
unverified, so it is not added as a sixth verified ablation condition.

Main supplies the paper baseline implementations and settings; Joseph supplies
transfer tools, context optimization, Ours and component ablations. Keep the
existing Joseph functionality and its source records rather than duplicating it.

## Existing FINAL collection

The following tables retain the branch's separately selected FINAL arms and
caveats. They are not substituted for the paper's user-reported Ours values above.

Synchronize these DeepSeek and Luna tables with the `finalizing` skill whenever a completed result is explicitly promoted. Each value must match a complete FINAL test arm and be documented in its report. A dash means no completed FINAL arm is recorded for that model and selection. Starling means Gold-v1; TDC includes the version pinned by each arm. Preserve older arms for audit and do not silently replace a selected cell with the highest score across prompts or versions.

### DeepSeek

| Task | Benchmark | Direct macro F1 | Direct + Indirect macro F1 |
| --- | --- | ---: | ---: |
| bbb_martins | Starling | 0.673178 | 0.725340 |
| bioavailability_ma | Starling | 0.768091 | 0.861340 |
| skin_reaction | Starling | 0.599320 | 0.615042 |
| ames | Starling | 0.708578 | 0.774211 |
| dili | Starling | 0.639755 | 0.659233 |
| carcinogens | Starling | 0.532669 | 0.647253 |
| bbb_martins | TDC | 0.865673 | 0.880814 |
| bioavailability_ma | TDC | 0.746709 | 0.830732 |
| skin_reaction | TDC | 0.634097 | 0.666875 |
| ames | TDC | 0.782289 | 0.799350 |
| dili | TDC | 0.819609 | 0.821346 |
| carcinogens | TDC | 0.817392 | 0.790611 |

The DeepSeek TDC Skin cells are complete 82-query **TDC-v1** upstream-v2 runs. The final pair was retained after reviewing test comparisons, and the same test cohort was used in later prompt and model iterations; treat these as historical descriptive scores, not an untouched TDC-v2 holdout estimate.

The DeepSeek TDC Carcinogens direct cell uses the validation-selected 56-query exact-score upstream-v3 arm. The separate Gold DILI `sr=0` control scored 0.680085 on 402 test queries; it is an exploratory FINAL successor and does not replace the validation-selected 0.659233 cell.

### Luna

| Task | Benchmark | Direct macro F1 | Direct + Indirect macro F1 |
| --- | --- | ---: | ---: |
| bbb_martins | Starling | 0.666929 | 0.722610 |
| bioavailability_ma | Starling | 0.760093 | 0.856127 |
| skin_reaction | Starling | 0.559252 | 0.564175 |
| ames | Starling | 0.667730 | 0.706003 |
| dili | Starling | 0.621456 | 0.665266 |
| carcinogens | Starling | 0.482328 | 0.623776 |
| bbb_martins | TDC | 0.828823 | 0.846330 |
| bioavailability_ma | TDC | 0.718372 | 0.774023 |
| skin_reaction | TDC | 0.700912 | 0.611005 |
| ames | TDC | 0.787868 | 0.811343 |
| dili | TDC | 0.790850 | 0.801545 |
| carcinogens | TDC | 0.841539 | 0.878261 |

The Luna Gold-v1 Skin direct arm uses a profile selected on 100 `valid_small` queries and scored 0.559252 on all 241 test queries. The Luna TDC Skin direct arm is an equivalent-cohort TDC-v1 replay filed in the TDC-v2 FINAL view. Its mixed successor was selected on all 40 TDC-v2 validation queries and scored 0.611005 on test; an exploratory test arm scored higher but did not change the validation choice. The Luna TDC Carcinogens mixed value uses the positive-gate prompt selected after viewing test results; its FINAL provenance records that decision. The Gold Carcinogens Luna arms are model-specific results below their existing DeepSeek counterparts. TDC-v2 AMES Luna exceeds both existing DeepSeek cells; its mixed profile was selected on a 100-query validation cohort.
