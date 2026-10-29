# Paper baseline recipes

Run from the repository root with the original frozen inputs and a fresh output
directory. Recipes select configurations; the shared Python commands implement
preparation, training and validation. They do not download or select benchmark
releases. Current sources and migration checks are in
[发布整合状态](../../docs/release_refactor.md).

## Morgan KNN, MiniMol KNN and frozen-encoder head

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
