# Molecular foundation baselines

Seven pretrained encoders reuse the MiniMol head, training loop, condition tensors,
auxiliary sampler and evaluation functions. The completed experiment has **1,890
fits, 378 model/task/pool/setting cells and 1,890 independent seed evaluations**.
Uni-Mol 2 uses 84M. There are no new KNN arms and no production LoRA fits.

## Results and statistical contract

[Result registry](../../experiments/baselines/results.json) pins the completed
foundation run, MiniMol/RASAR replays and unified comparison. The compact
repository export (internal document; not distributed in Git) records all 576 cells without
changing paper files. The primary ML
comparison is `outputs/analysis/baselines/ml_mean_comparison_20261108/`:

- `scores.csv`: 504 five-seed cells plus 72 deterministic KNN references;
  separate `legacy_ensemble_macro_f1` values preserve the old MiniMol/RASAR scores.
- `per_seed.csv`: individual scores; `six_task_means.csv` and `MAIN_TABLE.md`:
  equally weighted task means, calculated within each seed before uncertainty.
- `verification.json`: source hashes and completeness checks. Original seed
  reports also audit predictions, validation selections and metric calculations.

For five independent seeds, report arithmetic mean, sample SD (`ddof=1`) and
`mean ± t(0.975,4) × SD / sqrt(5)`. The interval is not clipped to [0,1]. It
quantifies training randomness on the frozen split, not test-sampling or split
uncertainty. Five runs provide only an approximate normality-based interval.
KNN has one deterministic result: SD and CI are unavailable, not zero. Mean
per-seed F1 is distinct from F1 computed after averaging five probabilities.
Neither the historical paper export nor manuscript figures are updated.

The source reports are:

| Source | Cells / seeds | Selection on validation only |
| --- | --- | --- |
| Seven encoders, `seven_models_minimol_fixed_lambda_20261107/repeated_runs/` | 378 / 1,890 | Fixed MiniMol-derived lambda per pool/task; independent checkpoint and threshold per seed |
| `minimol_repeated_runs_20261107/` | 54 / 270 | Independent lambda, checkpoint and threshold per seed, replaying saved candidates |
| `rasar_repeated_runs_20261108/` | 72 / 360 | Independent RF hyperparameter candidate and threshold per seed, replaying saved models |

The foundation run is under `outputs/predictions/foundation/paper_v1/`; the two
legacy replays are under `outputs/analysis/baselines/`. The original 432-cell
neural-only comparison remains at the foundation run's `repeated_runs/combined/`.
The unified table adds RASAR and the old KNN references; it does not select
per-task winners using test scores. See results (internal document; not distributed in Git)
for findings and the separately retained historical Ours comparison.

## Encoder releases

| Encoder | Pinned release | Representation |
| --- | --- | --- |
| CheMeleon | Zenodo 15460715, `chemeleon_mp.pt` | Official Chemprop bond message passing and mean readout, 2048 dimensions |
| Uni-Mol 2 | `dptech/Uni-Mol2`, 84M | Official Uni-Mol Tools CLS representation, 768 dimensions |
| GROVER | Official `grover_base.pt` | Official `fingerprint_source=both` readout, 3200 dimensions |
| ChemBERTa 2 | `DeepChem/ChemBERTa-77M-MTR` | Pretrained first-token representation before the descriptor head, 384 dimensions |
| MoLFormer | `ibm-research/MoLFormer-XL-both-10pct`, `compat-v4` | Official pooled representation, 768 dimensions |
| SMI-TED | `ibm-research/materials.smi-ted`, Light 40 | Official differentiable encoder and pretrained autoencoder compression, 768 dimensions |
| NV-KERMT | `nvidia/NV-KERMT-70M-v2` | Official `fingerprint_source=both` graph readout, 3200 dimensions |

The ChemBERTa `77M` identifier denotes pretraining molecules, not encoder
parameters. KERMT's stochastic contrastive latent head and SMI-TED's reconstruction
heads are not part of these molecular representations. The live runtime manifest
pins every checkpoint SHA-256, Hugging Face revision and upstream Git commit.

## Frozen scientific contract

[Input bindings](../../experiments/baselines/foundation_sources.json) select the
18 original task/pool cohorts: six tasks × conditioned_train, tdc_only and mixed.
They include the Joseph TDC Skin/Carcinogens cohorts and updated Skin auxiliary
pool. Preparation checks the original split hashes, ordered molecules and labels;
`data.pt` and its manifest are copied unchanged. Train-derived condition
vocabularies/tensors, target normalization, parent/connectivity heldout exclusions
and uniform-bucket/within-bucket-parent auxiliary sampling are retained.
TDC-only has no condition columns; auxiliary heads do not receive main conditions.

`frozen` trains the task head on cached embeddings; `finetuned` trains the encoder
and task head on main labels; `indirect` adds observed auxiliary targets through
separate auxiliary heads. Fits use seeds 1–5, 25 epochs, batch 32, Adam with weight
decay 1e-4, five warmup epochs then cosine decay, encoder LR 1e-5 and the original
task-specific head parameters. Trainable encoders clip gradients at 5. Each main
batch receives one auxiliary batch when enabled. Each seed selects its epoch by
validation BCE and threshold by validation Macro-F1; no test selection or refit.

For the seven new encoders, lambda minimizes **mean MiniMol validation BCE over
five seeds** among 0.1/0.6/1.0; ties choose the smaller lambda. It is fixed across
models and seeds, not a majority vote. All 90 source MiniMol records are hash-bound.
MiniMol's own replay still selects lambda separately per seed.

| Task | conditioned_train | tdc_only | mixed |
| --- | --- | --- | --- |
| BBB | 0.1 | 0.1 | 0.1 |
| Oral | 0.1 | 1.0 | 0.1 |
| Skin | 1.0 | 0.6 | 0.1 |
| Ames | 0.6 | 0.1 | 0.1 |
| DILI | 0.1 | 0.6 | 0.1 |
| Carcinogens | 0.6 | 1.0 | 1.0 |

RASAR reuses six saved hyperparameter candidates (max_features sqrt/0.3, leaf
sizes 1/5/20), each with five 100-tree forests. Direct/indirect features and
all-train/condition-first policies stay separate. Each seed selects its own
candidate before test inference. Probability BCE uses the shared repeated-run
float64 clipping at 1e-12; original ensemble receipts retain sklearn log_loss.
`selected_aux_weight=0` in RF records is a non-applicable sentinel, not a neural
auxiliary lambda. Candidate paths/losses, selected model hash and threshold are
recorded in each seed's metrics. All 360 winning candidates were also checked
against original sklearn BCE; none changed. Saved models are loaded beside their receipts,
with exact prepared-feature and split bindings checked.

## Precision, inputs and runtime layout

Frozen embeddings are FP32. Uni-Mol's first cohort
`conditioned_train/bbb_martins` retains FP32; its subsequent trainable cohorts use
BF16 autocast with FP32 parameters, optimizer and loss. Other encoders use FP32.
TF32 and Uni-Mol activation checkpointing retain their recorded settings.
`execution_policy.json`, checkpoints and result rows identify precision; a cell
cannot mix precision across seeds. BF16 uses the same 84M weights.

Uni-Mol calls official Uni-Mol Tools conformer generation (`seed=42`, fast mode,
remove_hs, max_atoms=128); stable SMILES-hash seeds control atom cropping. These
input seeds are separate from training seeds 1–5. Source-bound per-molecule caches
are shared by CPU workers; failures are recorded and fail preparation. The local
aromaticity-restoration fix and upstream compatibility patches are recorded with
the runtime, not silently reapplied to saved scientific inputs.

All upstream repositories, weights, revisions and environments live under
`/data1/tianang/Projects/TxAgent-foundation-models/` with a pinned `manifest.json`.
This repository keeps adapters and `experiments/baselines/patches/`, not copies of
upstream repositories. Local SSD checkpoints and host input caches are transient;
completed checkpoints/predictions remain in the shared run archive. MiniMol replay
requires its original Graphium/PyTorch runtime, separate from the foundation env.

The completed fixed-lambda run reuses inputs/features/fits through `artifact_root`
and links into `seven_models_20261106/`. **Retain both directories.** The parent
contains historical ensembles, superseded grid fits and interrupted checkpoints;
it is not an active full-grid experiment. The old 3,150-fit plan became 1,890 fits;
transition ETA estimates remain historical records under `transition/`.

## Maintained entrypoints

| Capability | Owner |
| --- | --- |
| Official encoder adapters and feature/conformer preparation | `predict.baselines.foundation_models` |
| Verified host-local input cache | `predict.baselines.foundation_cache` |
| Frozen-input preparation, multi-host queue, publication and verification | `predict.baselines.foundation_suite`; recipe `experiments/baselines/foundation.sh` |
| Shared training, checkpoint/RNG resume and losses | `predict.baselines.minimol.run_multitask` |
| Per-seed selection/replay, mean/SD/CI and prediction audits | `predict.baselines.repeated_runs` |
| Shared RF candidate validation and historical ensemble finalization | `predict.baselines.retrieval_forest` |
| Unified ML comparison from audited sources | `analysis.prediction.baseline_results`; registry `experiments/baselines/results.json` |
| Optional FP32/BF16/LoRA timing diagnostic | `predict.baselines.benchmark_adaptation`; recipe `experiments/baselines/benchmark_adaptation.sh` |

Run from the repo root in the foundation environment:

```bash
# Default recipe action is offline verification of the completed fixed-lambda run.
bash experiments/baselines/foundation.sh verify
python -m analysis.prediction.baseline_results

# New run only; never prepare over the completed derived manifest.
python -m predict.baselines.repeated_runs select-auxiliary \
  --source-root outputs/analysis/baselines/minimol_repeated_runs_20261107 \
  --output /tmp/minimol_auxiliary_policy.json
RUN_NAME=my_new_run AUXILIARY_POLICY=/tmp/minimol_auxiliary_policy.json \
  bash experiments/baselines/foundation.sh prepare
RUN_NAME=my_new_run SCRATCH_ROOT=/local/tmp/my_new_run \
  bash experiments/baselines/foundation.sh run
```

`OUTPUT_ROOT`, `RUNTIME_ROOT`, `SOURCE_ROOT` and `PYTHON` override paths. A fresh
prepare without `AUXILIARY_POLICY` explicitly retains the older grid contract.
CPU replay commands are in the [recipes](../../experiments/baselines/README.md#independent-seed-replay-and-comparison).

For distributed runs, both hosts use the same manifest, precision policy and
shared per-job locks, with distinct local SSD scratch directories. Resume on the
owning host until its staged work is published. Defaults are four slots/GPU,
16 CPU evaluation slots, feature prefetch enabled; BF16 Uni-Mol consumes two GPU
slots and FP32 Uni-Mol is exclusive. `FEATURE_WORKERS`, `CONFORMER_WORKERS`,
`FEATURE_CPU_SLOTS`, `CONFORMER_CACHE` and `--gpus` control execution, not science.
Conformer warming supports `foundation_suite conformers --partition 0 --partitions 2`
(use partition 1 on the second host). Publication writes `training.json` last;
resumption restores optimizer/scheduler/sampler/RNG state. Auditing runs off the
GPU dispatch path. Completed results, rather than historical queue counters,
establish completeness: three externally recovered evaluations were missing
from an older controller's completed-job count; the counter logic is now fixed.

Hash-bound single-candidate checkpoints whose float32 probabilities saturate at
one require fresh validation-logit replay before test inference. This recovered
three CheMeleon Carcinogens evaluations without retraining; unhashed historical
scores and multi-candidate selection retain strict checks. Threshold search uses
cumulative confusion counts with the original thresholds and tie rule.

The diagnostic `outputs/analysis/baselines/lora_bf16_benchmark_20261107/` measures
short matched update blocks, excluding loading, validation and checkpoint I/O.
It does not measure whole-run time or accuracy, and did not change production
fits. Preserved reports and failed alternatives are audit artifacts, not extra
maintained experiment implementations.
