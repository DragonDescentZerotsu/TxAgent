# MiniMol baselines

This directory shares embedding, head and finetuning runtimes. Conditioned benchmarks,
TDC evaluation and the external Starling Table 2 reproduction remain separate.

## Shared runtime

- `embedding_runtime.py`: MiniMol loading, embedding, cache, and checkpoint
  provenance.
- `head_runtime.py`: the shared classification/regression MLP head, loss,
  corrected epoch-level learning-rate schedule, and prediction helpers.
- `condition_features.py`: optional train-derived categorical one-hot features.
  Both the train-only CV selector and final head runner accept
  `--condition-field`; omitting it preserves the molecule-only 512-dimensional
  input exactly.
- `baselines/conditioned_knn.py`: shared unique-molecule condition-first/null-
  fallback selection used by both Morgan and MiniMol embedding KNN runners.

## Conditioned external-validation selection

The six-task standard profile is `conditioned_minimol_val_selected`: BBB/Oral
reuse the adopted frozen controls; Skin, Ames, DILI and Carcinogens have now been
trained with the task-specific official head parameters below. Five seeds each
train for 25 epochs on train only, select checkpoints by validation BCE, then
select the ensemble threshold by validation Macro-F1 before test evaluation.
Use the existing `run_bioavailability_ma --threshold-strategy valid_macro_f1`
entrypoint with `--condition-field condition_group`; omit `--train-all`.

The [complete five-baseline table](../../outputs/paper/conditioned_minimol_val_selected_20260921/RESULTS.md)
contains all six tasks, original fixed-k=3 KNN, current heads, accuracy and AUROC.
[Current LLM comparisons](../../outputs/paper/llm_vs_standard_baselines_20260924/RESULTS.md)
use these heads and fixed-k=3 KNN on both benchmarks. Encoder finetuning and
indirect-assay experiments have a separate scope below; earlier CV heads remain
historical.

## Conditioned benchmark inputs and earlier matched baselines

All current inputs use `data/conditioned_benchmark/<Task>/{scaffold,random}/`.
The maintained entrypoints are:

- `run_bioavailability_ma.py`: shared trained head for a supplied task split.
- `run_train_cv.py`: train-only scaffold CV for epoch and OOF threshold selection.
- `run_embedding_knn.py`: cosine KNN over frozen MiniMol embeddings.

Use `--condition-field condition_group` for BBB, Bioavailability, Skin, and Ames.
Freeze the condition vocabulary from outer train; valid/test categories must
already exist in it. ClinTox has no accepted condition groups and omits this flag.
The earlier matched baselines below select the epoch by train-only scaffold-CV AUROC and freeze a macro-F1 threshold
from pooled OOF scores, and fit the final head on all train rows. Outer valid/test
labels are excluded from both selections. The adopted indirect-assay comparison
uses validation selection instead; see [its protocol](#minimol--indirect-assay).

For a dataset with a frozen, more conservative scaffold grouping, explicitly
pass `--scaffold-group-field bemis_murcko_scaffold` to `run_train_cv.py`.
This uses the aligned outer-train field and records its ordered hash; without
the flag, the existing RDKit Murcko grouping remains unchanged.
KNN uses unique train molecules only; the conditioned comparison retains
same-condition-then-null and unrestricted
train variants with frozen k=3. Ames valid v2 explicitly uses
`--condition-policy same_condition_then_null_then_all`: keep same-condition
neighbors, then unannotated-condition neighbors, and fill any remaining slots
from unrestricted train molecules by similarity. Molecules remain unique and
each fallback neighbor is marked `all_train_fallback`. This optional policy
does not change existing `same_condition_then_null` runs.

Current scores, artifact roots, input hashes, and checkpoint dependencies have
one authoritative index: [current_conditioned_results.json](../../tools/chembl_tool/paper_experiments/current_conditioned_results.json),
with tables in [RESULTS.md](../../tools/chembl_tool/paper_experiments/RESULTS.md).
The earlier three-task test comparison is under `matched_baselines.scaffold_test`;
use `budgeted_evidence_protocol.conditioned_minimol_val_selected` for current heads.
In that historical comparison, BBB/Skin reused train-only-selected checkpoints; Bioavailability was
retrained after the two-row train correction. Its older valid scores remain
pre-fix references and must not be reported as current.

Training diagnostics are in [HEAD_TRAINING_DIAGNOSTICS.md](HEAD_TRAINING_DIAGNOSTICS.md).
Historical gold-version and agent-retrieval sensitivities are not alternative
current benchmark inputs or score tables.

## TDC external-validation selection

The validation-selected MiniMol protocol retains the task-specific official
parameters below, but replaces train-only CV with the existing head runner's
validation path. Both training pools use the same TDC valid/test molecules and
labels. Train five seeds on train only for 25 epochs; select each checkpoint by
minimum TDC valid BCE, average their probabilities, and select the classification
threshold by ensemble valid Macro-F1. Evaluate test without further selection or
train+valid refitting. KNN remains fixed at k=3 and is reused.

Use `run_bioavailability_ma.py --threshold-strategy valid_macro_f1` with the
desired data/cache/output directories and official hyperparameters; omit
`--train-all` and `--decision-threshold`. Mixed training retains
`--condition-field model_condition`. `metrics.json` includes per-epoch
`validation_history`, and `valid_predictions.jsonl` supports threshold auditing.
The [current comparison](../../outputs/paper/llm_vs_standard_baselines_20260924/RESULTS.md)
and its full-precision ML Appendix are the reporting sources. Skin/Carcinogens
use the seed-42 successors below; the other four heads retain
`tdc_minimol_val_selected_20260919`. That original suite's Skin/Carcinogens scores
and combined LLM table are historical.
The previously completed CV scores are historical; do not select old/new winners
using test results. Threshold selection affects Macro-F1/accuracy, not AUROC.

## TDC task-specific official hyperparameters

Both the current validation-selected profile and historical
`tdc_minimol_official_sweep` CV profile use the upstream `SWEEP_RESULTS` entry
matching each task; Skin and Carcinogens fall back to Bioavailability:

| Task | Hidden width | Depth | Learning rate |
|---|---:|---:|---:|
| BBB | 2048 | 3 | 0.0001 |
| Bioavailability / Skin / Carcinogens | 512 | 3 | 0.0003 |
| Ames | 512 | 3 | 0.0001 |
| DILI | 512 | 4 | 0.0005 |

All use `combine=True`, dropout 0.1, batch size 32, weight decay 0.0001,
warmup 5, and 25 maximum epochs. Current epoch/threshold selection is described
above. This adopts published configurations, not the full upstream repeated-split
ensemble protocol: we use one fixed TDC train/valid split and average five seeds'
probabilities. Existing runners accept all settings directly.
The historical CV profile selected epochs/thresholds within train, then refit five
seeds. Its frozen commands/results remain under
`outputs/paper/tdc_minimol_sweep_results_20260919/`; the earlier all-task
Bioavailability-config profile is also historical.

## TDC encoder finetuning and indirect assay

Current TDC results combine the original four task cohorts with the completed
[Skin](../../outputs/paper/skin_tdc_upstream_split_20260924/README.md) and
[Carcinogens](../../outputs/paper/carcinogens_tdc_upstream_split_20260924/README.md)
seed-42 suites. Each successor replaces all 18 ML report cells across both pools;
the naive LLM reruns use the exact same query structures and labels. Skin uses
upstream's published split. Carcinogens matches all 56 upstream test parent/label
pairs, with one mercury representation difference; his train/valid are unpublished.
The [split audit](../../outputs/paper/skin_tdc_upstream_split_20260924/SPLIT_AUDIT.md)
records this boundary. The earlier seed-1 cells are historical, not alternatives.
Skin auxiliary methods additionally use the [level-mapped source correction](../../outputs/paper/skin_indirect_level_mapped_20260925/README.md); their earlier broad-pool scores are historical. Frozen/main-only controls retain the seed-42 suite results.

Shared orchestration: `python -m baselines.tdc_split_suite --root <prepared experiment root>`.
It delegates to existing modeling CLIs for the fixed Skin/Carcinogens head profile.
The linked READMEs own input preparation and exact commands. Resume checks inputs
and model-code hashes; historical verification checks frozen source ZIPs, allowing
maintenance of the live wrapper. Other tasks continue using their existing CLIs.

The original two-pool, six-task extension (Skin/Carcinogens cells superseded) is registered as
`budgeted_evidence_protocol.tdc_minimol_indirect_assay`. It compares the existing
frozen heads with main-only encoder finetuning and indirect-assay joint training
in each pool. Use the same task heads, seeds 1–5, 25 epochs, batches 32/32/128,
encoder LR 1e-5, and auxiliary λ grid {0.1, 0.6, 1} as the conditioned protocol
below. Select checkpoints and λ on TDC validation BCE, then ensemble thresholds
on TDC validation Macro-F1; do not reuse conditioned λ selections.

The shared `run_multitask prepare` phase takes `--condition-field none` for
TDC-only, or `--condition-field model_condition` for the mixed pool. Both use
`--preserve-main-split` to retain the exact existing main train/valid/test rows,
including original TDC identity overlap. This option only bypasses the main
split disjointness assertion: auxiliary rows still exclude the entire TDC
valid/test parent/connectivity union, before aggregation and normalization.
Shared auxiliary scaffolds are allowed. Added Starling main labels keep their
existing stricter parent/scaffold exclusion. Both pools share the same TDC
evaluation molecules and auxiliary target pool for each task.

The extension uses `outputs/paper/tdc_minimol_indirect_assay_20260922/` and reuses
the frozen heads under `tdc_minimol_val_selected_20260919`. Status and artifacts
are recorded in the registry; training and evaluation use the existing shared CLI.
All 240 new seeds and 36 ensemble cells are verified. Full F1/accuracy/AUROC,
validation λ scores, thresholds and resource measurements are in the extension
root's `RESULTS.md`, `comparison.json` and `all_scores.csv`.
Keep all candidate checkpoints and the linked frozen controls. Execution-only
launch/preflight scripts are archived in `training_source_snapshot.zip`; local
`verify.py [data]` and `summarize.py` audit/render the completed suite.
`maintenance.json` records source hashes and preservation of earlier results.

## TDC test with added current Starling labels

The augmentation suite reuses the same head and KNN entrypoints.
Its data manifest is `data/tdc_starling_mixed_20260919/manifest.json`; command,
progress and verification receipts are under
`outputs/paper/tdc_starling_mixed_baselines_20260919/` and registered as
`budgeted_evidence_protocol.tdc_starling_mixed_baselines` for the original CV run.
Current heads retain the validation-selection protocol and
`--condition-field model_condition`; Skin/Carcinogens rebuild their mixed rows
and exclusions against the new seed-42 heldout cohorts. Added Starling train labels exclude
the TDC valid/test parent and raw/normalized scaffold union, including empty
scaffolds. TDC valid/test contain only TDC rows. The historical CV grouping field
`cv_scaffold_group` is not used by the current validation-selected head.
Both KNN families retain their two policies: unrestricted train molecules and
`same_condition_then_null`. TDC queries have no condition, so the latter admits
only unspecified-condition train rows; it can differ on mixed data but is
identical on TDC-only data. This is separate from the
external Table 2 reproduction below and does not change the conditioned benchmark.

This suite is complete: 30 verified task/method cells and 11,125 test predictions.
The experiment-local `build_data.py`, `run.py`, and `verify.py` reproduce data,
record existing-module commands, and verify results, respectively; they are not
new maintained model APIs. Resume requires each stage's outputs, not just its
metrics marker. Per-task scores and TDC-only comparisons live in the registry's
linked `comparison.json` / `all_scores.csv` and `RESULTS.md`.

## External Starling paper Table 2 reproduction

- `run_starling_table2.py`: runs both TDC-only and TDC+literature conditions for
  Oral Bioavailability, LD50, or BBB. Classification runs can opt into the data
  author's collapsed-row training weight with `--extraction-weight sqrt` and an
  explicitly separate weighted-AUROC sensitivity with
  `--evaluation-extraction-weight sqrt`.
- `merge_starling_table2_shards.py`: validates and merges disjoint repetition
  shards.
- `starling_table2_data.py`: released CSV schema, task settings, labels, and
  paper reference values.

The immutable release snapshot is under `data/starling_table2_v3/`; the full
protocol, results, diagnostics, and artifact locations are in
[`STARLING_TABLE2_REPRODUCTION.md`](STARLING_TABLE2_REPRODUCTION.md).

This reproduction uses a different external release, task definitions, and
splits. Its Table 2 metrics must not be added to the current TxAgent Starling
benchmark ledger or canonical figures.

### Unseen evaluation conditions in source ablations

The head runner supports explicit `--unseen-condition-policy zero` when an
ablation removes a training source and its condition categories. The vocabulary
still comes only from training rows; unseen evaluation conditions append zeros
without creating a fitted category or using evaluation labels. The checkpoint
records this policy. The default remains `error`. DILI/Carcinogens Starling-only
training results and split-preservation receipts are registered separately from
the mixed-training baselines in `current_conditioned_results.json`.

## Appendix result tables

The complete six-task ML appendix covers conditioned and both TDC training pools:
[Macro-F1 / AUROC table](../../outputs/paper/llm_vs_standard_baselines_20260924/ML_APPENDIX.md)
and [full-precision CSV](../../outputs/paper/llm_vs_standard_baselines_20260924/ml_appendix.csv).
All 126 task/method cells are verified, including the encoder/indirect controls.
Appendix provenance and regeneration: [paper result sources](../../tools/chembl_tool/paper_experiments/README.md#appendix-result-sources).

## Paper KNN: fixed k=3

Both frozen MiniMol cosine KNN and Morgan Tanimoto KNN retain the original
k=3, threshold 0.5 and train-only references. Keep existing all-train and
condition-first policies. Do not choose k on validation. TDC Direct-3 uses the
exact three Morgan neighbors and labels; MiniMol uses its own embedding neighbors.
Trainable MiniMol heads and encoder/auxiliary finetuning retain the latest
validation-selected protocol. Results: `budgeted_evidence_protocol.llm_vs_standard_baselines`.

## MiniMol + indirect assay

**Adopted final setting (2026-09-20):** main-task scaffold split with
**parent-disjoint auxiliary evidence**. Exclude the union of valid/test normalized
parents and nonempty connectivity keys from auxiliary training; allow shared
scaffolds. This is intentional alignment with parent-disjoint LLM indirect
retrieval, not the stricter TDC+Starling label-augmentation profile. Global heldout
parent exclusion is stronger than per-query exclusion; numerical targets and
LLM text are not identical information interfaces. The scaffold-overlap audit is
informational and does not invalidate this adopted setting.

Compare frozen **MiniMol**, **MiniMol finetuned** on main labels, and
**MiniMol + indirect assay**. All use the same main head and train-derived
`condition_group` one-hot features. The latter two update the complete MiniMol v1
encoder; auxiliary heads are linear outputs over its 512-dimensional embedding.

| Task | Head width / depth | Head LR | Selected λ |
|---|---:|---:|---:|
| BBB | 2048 / 3 | 1e-4 | 1 |
| Oral | 512 / 3 | 3e-4 | 0.1 |
| Skin | 512 / 3 | 3e-4 | 1 |
| Ames | 512 / 3 | 1e-4 | 0.6 |
| DILI | 512 / 4 | 5e-4 | 0.1 |
| Carcinogens | 512 / 3 | 3e-4 | 0.6 |

All tasks use encoder LR 1e-5, main/auxiliary/evaluation batches 32/32/128,
25 epochs and seeds 1–5. Auxiliary heads use the main-head learning rate.

Shared settings: Adam, weight decay 1e-4, dropout 0.1, `combine=True`, five-epoch
warmup then cosine decay, gradient-norm cap 5 for finetuning. Depth 3 uses two
hidden layers plus the output. Main loss is BCE; auxiliary weights searched are
`{0.1, 0.6, 1}`. Each seed selects its checkpoint by main validation BCE; choose
auxiliary weight by ensemble validation BCE, then threshold by ensemble
validation Macro-F1 before test evaluation. No CV or train+valid refit. For BBB/Oral, the 0.6
supplement followed the first test evaluation; the four-task extension fixed the
complete candidate grid before training. Previously evaluated test cohorts are
not claimed to be untouched.

Use upstream's `parent_smiles`, JSON `pair_bucket_key`, and `canonical_measurements`
Parquet columns. Skin uses the release-level-mapped successor from commit
`59c05729` (3,287 rows / 67 buckets before heldout exclusion, upstream minimum
10 parents per bucket); other tasks retain their existing source pools. Within
each supplied pool, preparation adds no source/prediction/variance filter and
retains small or constant buckets after heldout exclusion.
Aggregate only within parent × bucket: median continuous values, or positive
fraction for binary −1/+1 measurements. Continuous targets use auxiliary-training
bucket mean/std (zero std becomes 1), with MSE and no log transform; binary
fractions use BCE. Only observed targets contribute. Each main batch receives
one auxiliary batch sampled uniformly over buckets, then parents, with replacement.
All retained targets are eligible, but each epoch is not an exhaustive auxiliary pass.
Valid ions that fail uncharging retain a charge-preserving fragment parent; the
same identity rule applies to benchmark and auxiliary rows before exclusion.

### Maintained entrypoints and adding tasks

Use one task-independent CLI: `python -m baselines.minimol.run_multitask`.

| Phase | Responsibility | Output |
|---|---|---|
| `prepare` | Exclude heldout identities, aggregate targets, cache graphs | `prepared/` |
| `embed` | Export pretrained embeddings when a compatible frozen cache is unavailable | `frozen/embeddings/` |
| `train` | Train one seed at one auxiliary weight; weight 0 is main-only | `aux_<weight>/seed_<n>/` |
| `finalize` | Select λ and threshold on validation, evaluate selected ensembles | `selection.json`, `finetuned/`, `indirect/` |

Implementation stays in three modules: `indirect_data.py` (targets),
`finetune_runtime.py` (graphs/encoder), and `run_multitask.py` (CLI/training/selection).
Frozen heads reuse `run_bioavailability_ma.py` with `--threshold-strategy valid_macro_f1`.

Example for a **new output directory** (the completed experiment is immutable):

```sh
TASK_RUN=outputs/paper/minimol_indirect_next/bbb_martins
python -m baselines.minimol.run_multitask prepare --prepared "$TASK_RUN/prepared" \
  --data-dir data/conditioned_benchmark/BBB_Martins/scaffold \
  --auxiliary data/artifacts/indirect_ml/bbb_martins_parent_pair_bucket_measurements.parquet
python -m baselines.minimol.run_multitask embed --prepared "$TASK_RUN/prepared" --output "$TASK_RUN/frozen"
python -m baselines.minimol.run_bioavailability_ma \
  --data-dir data/conditioned_benchmark/BBB_Martins/scaffold --output-dir "$TASK_RUN/frozen" \
  --condition-field condition_group --hidden-dim 2048 --depth 3 --lr 1e-4 \
  --epochs 25 --ensemble-size 5 --threshold-strategy valid_macro_f1
# One independent job: run seeds 1–5 for each weight 0, 0.1, 0.6, 1 on available GPUs.
CUDA_VISIBLE_DEVICES=0 python -m baselines.minimol.run_multitask train \
  --prepared "$TASK_RUN/prepared" --output "$TASK_RUN/aux_0.6/seed_1" \
  --seed 1 --aux-weight 0.6 --hidden-dim 2048 --lr 1e-4
python -m baselines.minimol.run_multitask finalize --prepared "$TASK_RUN/prepared" --output "$TASK_RUN"
```

Use the `vllm` environment and recorded MiniMol/Graphium dependencies. On node002,
the runtime dependencies were moved to shared storage on 2026-09-29. For the
commands above, set:

```sh
export PYTHONPATH=.:/path/to/txagent-assets/minimol/pydeps:/path/to/txagent-assets/minimol/runtime_41642328
```

`baselines.tdc_split_suite` uses this location by default; override its root with
`TXAGENT_MINIMOL_RUNTIME_ROOT` on another installation. Historical run receipts
retain the original `/local/tmp/txagent_minimol_*` paths; their dependency bytes
were preserved. Versions/checkpoint hashes remain in each run's `environment.json`
and graph receipts; the storage migration is recorded in
`/path/to/txagent-assets/migration_20260929.json`.
Cap native threads per process; distribute independent seeds over GPUs without
multiplying the batch size. Preparation defaults to 64 CPU workers per task;
cache deterministic per-SMILES graphs once and retain them for future runs.

For another binary task, supply its three condition-aware split files, compatible
Parquet, new output root and task-specific head parameters from the existing
parameter table. Seed count is explicit via `finalize --seeds` (default 1–5).
No dataset names are hardcoded in the maintained modules. All six tasks are now
complete. The four-task extension uses 32 concurrent seed jobs across eight GPUs
and four preparation processes with 32 CPU workers each; utilization is recorded
every 15 seconds. Reuse compatible frozen controls after checking input hashes
and agreement between pretrained graph embeddings and the original cache.

`finalize` validates all candidate checkpoints and scores before selection. An
unchanged, hash-bound selected model reuses predictions; altered inputs, models
or incomplete prediction files cannot silently reuse an evaluation. Completed
training/preparation directories are protected from accidental overwrites.
For an auxiliary-data-only refresh, use `finalize --indirect-only`: select and
evaluate positive-weight candidates without requiring or relabeling old weight-0
checkpoints. The default still evaluates both main-only and indirect ensembles.

### Adopted results and artifact layout

Registry: `budgeted_evidence_protocol.minimol_indirect_assay`, with Skin auxiliary
results superseded by `skin_indirect_level_mapped`. The original six-task
report: `outputs/paper/minimol_indirect_assay_20260922/RESULTS.md`; raw metrics:
`all_tasks_comparison.json` and `all_scores.csv`. There are 18 verified ensemble
cells, 6,144 test predictions and 6,129 validation predictions. Skin/Ames/DILI/
Carcinogens add 80 encoder-training seeds and reuse 20 frozen-head seeds from
`conditioned_minimol_val_selected_20260921`. BBB/Oral retain their original 50-seed
suite under `minimol_indirect_assay_20260920` without changing predictions.

- Per task: `prepared/` holds target/graph caches and hashes; `aux_<weight>/seed_<n>/`
  holds checkpoints and histories; `frozen/`, `finetuned/`, `indirect/` hold final predictions.
  The new four-task `frozen/` directories link to the existing verified controls.
- Each source suite retains its `manifest.json`, `verification.json`,
  `data_verification.json`, `preflight.json` and resource receipts. The aggregate
  comparison binds both suites by verification hashes. Import provenance is in
  the extension's `import_receipt.json`.
- Experiment-local launch/audit scripts are frozen command receipts, not APIs.
  Executed training source is preserved in `training_source_snapshot.zip`; use
  the maintained CLI for new tasks. In the four-task extension, `run.py` and
  `preflight.py` are archived in that ZIP under their original repository paths;
  execution-time paths in `manifest.json` remain historical provenance.
  Local `verify.py [data]` audits this completed suite. Its obsolete `summarize.py`
  is archived in the Skin correction's `superseded_reporting.zip`; current tables
  come from the shared baseline exporter.
  `maintenance.json` records the review, archive hashes and unchanged results.
  Do not overwrite completed run directories or copy the linked frozen caches.
