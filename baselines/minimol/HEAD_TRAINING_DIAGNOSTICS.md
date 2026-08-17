# MiniMol head train-only diagnostics

Updated: 2026-08-12.

This note records the training-only validation contract added after auditing the
Starling MiniMol head baseline. It applies to the frozen MiniMol encoder plus
task-specific binary MLP head, not the MiniMol embedding cosine-KNN baseline.

Scope boundary: the external Starling paper Table 2 CSV reproduction is recorded
separately in `STARLING_TABLE2_REPRODUCTION.md`. It reuses the corrected shared
head runtime, but its release data, task definitions, splits, and metrics do not
replace this current TxAgent benchmark contract or its canonical baseline.

## Problem in the previous formal run

The current scaffold-valid baseline under
`outputs/baselines/minimol_starling_valid/` trained five head seeds for a fixed
25 epochs with `--train-all`. That path did not read an inner validation split,
did not record per-epoch train/validation metrics, and did not select an epoch.
It evaluated outer scaffold-valid only after training.

The old learning-rate loop also called `scheduler.step(epoch)` before any
`optimizer.step()`. PyTorch warned that this skipped the first scheduled value.
The runner now uses five actual linear-warmup epochs and steps the scheduler
after each training epoch. A matched corrected-scheduler 25-epoch rerun changed
outer-valid AUROC by only -0.0019 BBB, -0.0006 Bioavailability, and -0.0017
Skin, so the call-order bug was real but does not explain the baseline result.

## New train-only contract

Entry point:

```text
python -m baselines.minimol.run_train_cv
```

The runner:

- reads only `train.jsonl` and its exact-SMILES-matched frozen embedding cache;
- constructs deterministic 5-fold `StratifiedGroupKFold` folds using canonical
  Bemis-Murcko scaffold groups;
- verifies that every train row appears in exactly one inner validation fold
  and that scaffold overlap is zero in every fold;
- records train and inner-valid BCE, AUROC, macro-F1, and accuracy for every
  fold and epoch;
- selects epoch by mean inner-valid AUROC, then mean macro-F1, then the earlier
  epoch;
- never reads outer `valid.jsonl` or `test.jsonl` during selection.

For speed, this first-stage diagnostic uses one deterministic head seed per
fold; the frozen outer-valid receipt uses the formal five-member ensemble. The
CV values are therefore selection diagnostics, not an unbiased estimate of the
five-member ensemble's performance.

Canonical diagnostic artifacts are:

```text
outputs/baselines/minimol_head_train_cv_scheduler_fixed/<Task>/scaffold/
outputs/baselines/minimol_head_train_cv_sweep/<Task>/scaffold/<config>/
```

Each directory contains `fold_epoch_metrics.jsonl`, `epoch_summary.json`, and
`summary.json`.

## Frozen compact regularization sweep

The sweep compared the existing 512-hidden/dropout 0.1/weight-decay 1e-4 head
with three fixed alternatives: dropout 0.3; dropout 0.3 plus weight decay 1e-3;
and 256 hidden units plus dropout 0.3 and weight decay 1e-3. Learning rate,
depth, batch size, folds, seed, and maximum 25 epochs were held fixed.

| task | train-only selected config | epoch | inner-valid AUROC | epoch-25 inner-valid AUROC | selected / epoch-25 valid BCE |
|---|---|---:|---:|---:|---:|
| BBB_Martins | hidden 256, dropout 0.3, wd 1e-3 | 6 | 0.7405 | 0.7336 | 0.5254 / 0.6560 |
| Bioavailability_Ma | hidden 512, dropout 0.3, wd 1e-3 | 5 | 0.7355 | 0.7038 | 0.5541 / 0.7789 |
| Skin_Reaction | hidden 256, dropout 0.3, wd 1e-3 | 6 | 0.6373 | 0.6002 | 0.6122 / 0.8120 |

The train-only curves show clear late-epoch overfitting. At epoch 25, train
AUROC is 0.9859, 0.9991, and 0.9824 respectively, while inner-valid BCE is
substantially worse than at the selected epoch. BBB ranking AUROC is relatively
flat; the Bioavailability and Skin AUROC drops are larger.

## Outer scaffold-valid receipt and decision

After freezing each config and epoch from train-only CV, a five-member ensemble
was retrained on all outer-train rows and evaluated once on scaffold-valid:

| task | old 25-epoch AUROC | CV-regularized AUROC | delta | old / new macro-F1 |
|---|---:|---:|---:|---:|
| BBB_Martins | 0.8252 | 0.7938 | -0.0314 | 0.7108 / 0.6746 |
| Bioavailability_Ma | 0.7204 | 0.7105 | -0.0099 | 0.6577 / 0.6258 |
| Skin_Reaction | 0.6349 | 0.6687 | +0.0338 | 0.5900 / 0.5914 |

Outputs are isolated under:

```text
outputs/baselines/minimol_starling_valid_train_cv_regularized/<Task>/scaffold/
```

The train-only procedure improves Skin but does not transfer uniformly to the
outer scaffold-valid distributions. It therefore does **not** replace the
current canonical 25-epoch MiniMol baseline. Formal test remains untouched.
The durable result is that the training diagnostics are now auditable and that
late-epoch overfitting exists; any future head promotion should use a matched
train-only selection protocol and require a frozen outer-valid promotion gate
before one formal test run.
