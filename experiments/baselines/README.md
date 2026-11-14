# Paper baseline recipes

Run from the repository root with the original frozen inputs and a fresh output
directory. Recipes select configurations; the shared Python commands implement
preparation, training and validation. They do not download or select benchmark
releases. Current sources and migration checks are in
发布整合状态 (internal document; not distributed in Git).

## Tinker supervised label prediction

`tinker_sft.sh` runs independent DeepSeek-V4.1-Flash-1017 LoRA sweeps for both
paper benchmarks, with train-only examples, validation selection and final test
evaluation. The [SFT contract and run guide](../../predict/baselines/TINKER_SFT.md)
documents the frozen source manifest, TxGemma-style settings and differences,
cost estimation, credentials, checkpoint resume and result artifacts.
Both suites are complete; see the 12-task results (internal document; not distributed in Git).
Use `python -m predict.baselines.tinker_sft verify --output <run-root>` for an
offline result audit, without running the training recipe or supplying credentials.

## Foundation encoders

The seven-model matrix is complete: 378 cells / 1,890 independent seed evaluations.
`foundation.sh` reuses the MiniMol training loop for frozen, main-only and auxiliary
settings. Uni-Mol 2 uses 84M; there are no new KNN arms. The default action is
**verify**, pointing to `seven_models_minimol_fixed_lambda_20261107`. Training
requires an explicit `run`; fresh preparation requires a new run identity.
The [foundation guide](../../predict/baselines/FOUNDATION_MODELS.md) owns the
entrypoint map, fixed lambda table, precision, runtime layout and multi-host resume
contract. Upstream repositories and weights share one external runtime directory.

```bash
bash experiments/baselines/foundation.sh verify
python -m analysis.prediction.baseline_results
```

The second command rebuilds the repository-only comparison from the
[result registry](results.json), including `docs/baseline_means.csv`, without
changing paper tables or figures.
Timing diagnostics remain optional: `benchmark_adaptation.sh` compares FP32,
BF16 and LoRA+BF16 on a reserved GPU; it does not launch the production matrix.

## Independent-seed replay and comparison

MiniMol and RASAR now have separate mean/SD/95% CI reports, replaying saved models
without retraining. Their historical ensemble reports remain intact. Each seed
selects its own candidate and threshold on validation only. RASAR retains four
feature/policy settings. Old deterministic KNN results have no seed SD or CI.
The unified report contains 504 repeated-run cells and 72 KNN references; see
results (internal document; not distributed in Git) for paths and findings.

Use a **new** output directory and the original compatible MiniMol/sklearn runtime:

```bash
python -m predict.baselines.repeated_runs prepare-forests \
  --source-root /path/to/original/TxAgent \
  --legacy-python /path/to/original/environment/bin/python \
  --output /path/to/new/rasar-replay
python -m predict.baselines.repeated_runs run-legacy \
  --output /path/to/new/rasar-replay --workers 16
```

For MiniMol use `prepare-legacy` instead of `prepare-forests`; the same CPU runner
resumes pending jobs and verifies the complete report. Its manifest supplies the
Graphium dependency paths. Audit a completed source with `summarize --output
<source> --expected-cells 54` (MiniMol) or `72` (RASAR). `summarize --sources <a> <b>
--output <new-combined-report> --expected-cells <total>` combines distinct seed
reports and rejects overlapping cells. The unified ML builder additionally brings
in the hash-bound deterministic KNN and historical ensemble reference columns.

## Morgan KNN, MiniMol KNN and frozen-encoder head

The historical recipe below retains ensemble reporting for trained heads.

```bash
TASK=bioavailability_ma TRAINING_POOL=conditioned \
  DATA_DIR=/path/to/splits OUTPUT_ROOT=/path/to/new/run \
  bash experiments/baselines/conditioned.sh
```

See the [MiniMol guide](../../predict/baselines/minimol/README.md#paper-baseline-recipe)
for task parameters, training pools, val-selected heads and exact embedding reuse.
Encoder fine-tuning and auxiliary training use the existing
`python -m predict.baselines.minimol.run_multitask {prepare,embed,train,finalize}`.
Retain the selected experiment's auxiliary data, condition field, seeds and
weight candidates; the current Skin indirect pool differs from its older
main-task-only training inputs.

## RASAR and RASAR+indirect

```bash
DATA_DIR=/path/to/splits SOURCE_PREPARED=/path/to/minimol/targets \
  PREPARED=/path/to/new/features OUTPUT_ROOT=/path/to/new/models \
  bash experiments/baselines/retrieval_forest.sh
```

`SOURCE_PREPARED` contains the audited `manifest.json` and `data.pt` from MiniMol
auxiliary preparation. It supplies observed targets, not learned embeddings.
An existing `PREPARED/manifest.json` skips feature construction; its pinned files
and inputs are still validated by the shared CLI. A new preparation needs
`DATA_DIR` and `SOURCE_PREPARED`; set `AUXILIARY=/new/path/source.parquet` if the
raw auxiliary source has moved. Its hash must match the frozen target manifest.

New feature caches contain exact split-file copies under `splits/` and use
manifest-relative paths (`retrieval_forest_features.v2`). Move the entire prepared
directory to keep it usable. Existing v1 caches remain unchanged: pass `DATA_DIR`
to the recipe, or `--data-dir` to `fit`/`finalize`, to locate relocated splits;
all three files must still match their original hashes. Do not rewrite the old
manifest, since model receipts bind its exact hash.

Move the complete model output directory, including all `candidates/` subdirectories.
RF models are loaded beside their training receipts and checked against saved
hashes. New receipts use relative filenames; old absolute filenames are accepted
as provenance without opening the old locations. Finalize preserves successful
prediction files on resume and verifies selection and saved prediction hashes.

The recipe runs six candidates (`max_features=sqrt/0.3`, leaf sizes 1/5/20),
five seeds 1–5 and 100 trees per seed. Valid BCE selects the candidate, ensemble
valid Macro-F1 selects the threshold, and test is evaluated once after selection.
`FEATURE_SETS="direct direct_indirect"` selects six direct features or direct
plus two features per auxiliary bucket. No condition one-hot is appended.

`POLICIES="0 1"` selects all-train and condition-first. When preparing conditioned
DILI/Carcinogens, set `CONDITION_POLICY=same_condition_then_null_then_all`;
otherwise the default is `same_condition_then_null`. Existing feature caches
retain their recorded policies. If both policies are identical, `POLICIES=0`
avoids duplicate fits; reporting aliases must follow the original result source.
`PYTHON_BIN` selects Python and `WORKERS` controls CPU parallelism (default 2).
Outputs are grouped as `<feature_set>/policy_<index>/`; completed output roots
are rejected. Resume an interrupted candidate or finalize explicitly through
`python -m predict.baselines.retrieval_forest`, using its original arguments.

## Frozen LLM baseline

Use `python -m predict.harnesses.branches` with the explicit paper-baseline mode.
The [frozen prompt contract](../../predict/prompts/README.md#frozen-paper-baseline)
owns commands for offline verification, candidate preparation, portable bundles,
API/Batch execution and recovery, including relocation and dependency requirements.
Use `--retrieval-protocol legacy_paper` to reproduce frozen paper inputs exactly.
The default `murcko` mode preserves prompt templates and generation settings, but
retrieved evidence and full messages can change. FINAL defaults are a separate recipe.

## Frozen LLM repetitions

`llm_repeats.sh STUDY_DIRECTORY` consumes the single `jobs.json` plan in that
study, using the existing naive and MolRAG preparation/execution commands.
The completed [DeepSeek study](../../outputs/analysis/baselines/deepseek_repeats_preflight_20261112/README.md)
covers four additional runs, both benchmarks and all six tasks for Direct,
Direct+Indirect and MolRAG. See results (internal document; not distributed in Git)
for five-run and new-four statistics and the historical-setting caveats.

The [Zero-shot supplement](../../outputs/analysis/baselines/deepseek_zero_shot_repeats_20261112/README.md)
uses the same launcher and scorer for four more `deepseek_none` runs per task.
The scorer accepts complete benchmark/task grids for any supported method subset;
Zero-shot retains frozen messages and priors with effective retrieval budgets 0/0.

Completed runs should be audited **offline**, into a new report directory:

```bash
python -m analysis.prediction.llm_repeats \
  --study outputs/analysis/baselines/deepseek_repeats_preflight_20261112 \
  --output /tmp/deepseek-repeats-audit
```

This verifies original and repeated naive predictions, uses the maintained
MolRAG full-cohort scorer, and writes `per_run.csv`, `summary.csv`, `results.json`
and `aggregation_status.json`. It rejects missing/duplicate repetitions,
pending MolRAG queries and changed source manifests. Reports are published
atomically only after all checks pass. No API calls or raw-run writes occur.
`--workers` controls parallel read-only naive verification (default 4).
Six-task SD is calculated across run-level means, never by averaging task SDs.

To execute an explicitly planned new study, inspect its provider JSONs and
endpoint capacity first, export `TXAGENT_PARCC_REPEAT_KEY`, then run:

```bash
SOURCE_ROOT=/path/to/original/TxAgent LLM_REPEAT_WORKERS=4 MOLRAG_PARALLELISM=128 \
  bash experiments/baselines/llm_repeats.sh /path/to/study
```

Each job contains `kind`, `root`, `provider` and `source` (plus repetition and
cohort metadata for reporting). Naive workers use 128 requests; MolRAG workers
use `MOLRAG_PARALLELISM` (default 128). Provider changes require new run identities;
MolRAG capacity overrides are recorded separately in execution sessions.
Credentials come from the process environment; a dummy `.env` file is unnecessary.
The study's old `jobs.tsv`, recovery subset and supervisor/monitor receipts are
historical records, not additional maintained launchers.

A `.done` marker means strict successful execution. MolRAG can process every
query yet exit nonzero for terminal failures; do not relabel that as all-success.
Its separate offline scorer retains failures in the denominator and the repeat
report can be complete under that documented penalty policy. Do not relaunch
completed inference just to recalculate statistics. A completed
`aggregation_status.json` makes the launch recipe skip inference without requiring
a credential; use the offline report command for rechecking.

## MolRAG selected reproduction

[Selection record](molrag_selected.json): direct evidence, four-shot Sim-CoT,
low effort, corrected parsing. The [single run guide](../../predict/baselines/MOLRAG.md)
owns the protocol, code map, raw/derived artifact layout and historical variants.
Selected scores (internal document; not distributed in Git) are the
canonical MolRAG result for this project.

```bash
# Recompute all three backbones offline into a NEW directory; no API calls.
bash experiments/baselines/molrag.sh /tmp/molrag-offline-check
```

Only the selected evidence/low-effort setting is maintained. Preparation fixes
four-shot test inference and model-specific generation settings; the runner keeps
resumability, concurrency, throttling and budget controls. See the run guide for
source export and new-run commands. Historical experiment artifacts in `outputs/`
are audit records, not maintained code or alternative settings.
