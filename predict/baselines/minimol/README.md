# MiniMol baselines

## Shared implementation

- `embedding_runtime.py`: frozen MiniMol loading, embedding caches and checkpoint provenance.
- `head_runtime.py`: shared MLP, training schedule and prediction helpers.
- `condition_features.py`: train-derived categorical one-hot features.
- `run_bioavailability_ma.py`: shared head CLI for all six paper tasks.
- `run_embedding_knn.py`: frozen-embedding cosine KNN; shares the condition selector
  with Morgan KNN in `predict/baselines/conditioned_knn.py`.
- `run_multitask.py`: prepare/embed/train/finalize for encoder fine-tuning and
  indirect auxiliary targets, using `indirect_data.py` and `finetune_runtime.py`.

The integration status, current result sources and verification boundaries are
maintained in [发布整合状态](../../../docs/release_refactor.md).

## Paper baseline recipe

Run from the repository root with the frozen splits for the selected task/pool:

```bash
TASK=bioavailability_ma TRAINING_POOL=conditioned \
  DATA_DIR=/path/to/splits OUTPUT_ROOT=/path/to/new/run \
  bash experiments/baselines/conditioned.sh
```

`TASK` is required: `bbb_martins`, `bioavailability_ma`, `skin_reaction`,
`ames`, `dili` or `carcinogens`. It selects the original task-specific head
configuration. `TRAINING_POOL` selects `condition_group` for `conditioned`,
no condition features for `tdc_only`, or `model_condition` for `mixed`.
It does not build or change the supplied splits. The original prepared inputs
must include the appropriate fields; TDC Skin/Carcinogens use the current
seed-42 replacements. Explicit `CONDITION_FIELD=` disables one-hot features.

The head trains only on train for 25 epochs, with five seeds 1–5. Each seed's
checkpoint is selected by valid BCE; the ensemble threshold is selected by
valid Macro-F1, then applied to test. There is no train+valid refit. The condition
vocabulary is derived only from train. Unseen evaluation conditions fail by
default; `--unseen-condition-policy zero` explicitly uses an all-zero condition
vector without expanding the vocabulary. Head CLI arguments after the script
override settings, for example `--epochs 10 --ensemble-size 2`.

`TRAIN_ALL=1` retains the separate fixed-epoch, fixed-0.5 recipe without valid
selection. It still trains on the supplied train rows only; it does not merge
valid/test into train. This mode and the historical train-only CV experiments
are not the current paper protocol.

The recipe runs Morgan KNN, MiniMol cosine KNN and the head. Use
`METHODS="morgan_knn minimol_knn"` to select methods. KNN defaults to k=3 and
threshold 0.5. `CONDITION_POLICY` defaults to `same_condition_then_null`, or
`same_condition_then_null_then_all` for conditioned DILI/Carcinogens. Selection
uses the first source row for each molecule-condition pair, matches the query
condition first, then fills from null-condition rows; the latter policy also
fills from remaining parents. Null queries start with the null pool and may
also use the all-train fallback under the latter policy. Insufficient
pools retain unsupported queries and report coverage. Use
`CONDITION_POLICY=all_train_unique_molecules` for the all-train comparison.
The standalone KNN CLIs retain their `row_agnostic` default.

`EVALUATION_SPLIT=valid` selects validation output; it does not turn validation
into an unseen test set. Choose a fresh `OUTPUT_ROOT`. Each method writes
native metrics and predictions under `morgan_knn/`, `minimol_knn/` or
`minimol_head/`. The head also writes `best_ensemble.pt`, validation predictions
and per-epoch validation BCE. Predictions retain benchmark/condition metadata,
and metrics include condition cohorts and coverage.

## Runtime and exact cache reuse

`PYTHON_BIN`, `DEVICE`, `MINIMOL_SOURCE`, `EMBEDDING_CACHE_DIR` and
`REUSE_EMBEDDING_CACHE_DIR` configure execution. A portable MiniMol checkout
can live at `third_party/minimol`. For an existing archived runtime, set
`TXAGENT_MINIMOL_RUNTIME_ROOT` to a root containing `runtime_41642328/` and
`pydeps/`; the shell recipe adds both to `PYTHONPATH`. For direct Python CLI
use, supply the dependencies in the environment/PYTHONPATH as well. No runtime
is downloaded or installed by this recipe.

For exact historical reproduction, copy the original ordered `train.pt`,
`valid.pt`, `test.pt` to a new cache directory, confirm their SMILES/labels match
the input files, and pass it as `EMBEDDING_CACHE_DIR`. Preserve any recorded
split rename mapping (notably conditioned Ames). Do not point a writable run
at the only historical cache copy.

`REUSE_EMBEDDING_CACHE_DIR` instead assembles vectors by exact SMILES, retaining
current labels and rejecting vectors that conflict beyond the existing
floating-point tolerance. It can merge slightly different batch-computed
vectors for the same SMILES. This is useful general reuse, but cannot promise
bitwise-identical subsequent training. Fresh encoder computation can also
have small batch/device numerical differences.

## Historical protocols

`run_train_cv.py` retains the old train-only scaffold-CV/OOF selection workflow.
Its old validation results, including the BBB gold-v4 candidate comparison,
are historical and do not replace the current val-selected head results.
The historical training audit is [HEAD_TRAINING_DIAGNOSTICS.md](HEAD_TRAINING_DIAGNOSTICS.md).
Original result artifacts remain under `outputs/baselines/starling_conditioned_valid_v1/`
and `outputs/baselines/starling_conditioned_bbb_gold_v4_valid_v1/`; frozen
retrieval sensitivities also remain at their original result locations.
Current figures and scores use the single result registry described in the
integration document, not duplicated scores in this README.

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

The immutable release snapshot and its source contract are under
`data/legacy/artifacts/starling_table2_v3/`. The reproduction entrypoints remain
in this directory; the superseded standalone reproduction report was retired.

This reproduction uses a different external release, task definitions, and
splits. Its Table 2 metrics must not be added to the current TxAgent Starling
benchmark ledger or canonical figures.
