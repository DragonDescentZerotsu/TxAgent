# MiniMol baselines

This directory has one shared frozen-embedding runtime and two deliberately
separate experiment lineages.

## Shared runtime

- `embedding_runtime.py`: MiniMol loading, embedding, cache, and checkpoint
  provenance.
- `head_runtime.py`: the shared classification/regression MLP head, loss,
  corrected epoch-level learning-rate schedule, and prediction helpers.
- `condition_features.py`: optional train-derived categorical one-hot features.
  Both the train-only CV selector and final head runner accept
  `--condition-field`; omitting it preserves the molecule-only 512-dimensional
  input exactly.
- `predict/baselines/conditioned_knn.py`: shared unique-molecule condition-first/null-
  fallback selection used by both Morgan and MiniMol embedding KNN runners.

## Current TxAgent Starling benchmark

- `run_bioavailability_ma.py`: trained MiniMol head for a supplied TxAgent
  train/valid/test split.
- `run_train_cv.py`: train-only scaffold CV and epoch/configuration selection.
- `run_embedding_knn.py`: cosine KNN over frozen MiniMol embeddings.
- `tools/chembl_tool/paper_experiments/run_minimol_valid_matrix_gpt_oss_120b.py`:
  frozen serial launcher for the completed current-lineage `top_k=5`,
  `min_similarity=0` GPT-OSS-120B agent sensitivity. It delegates every task
  to the shared `starling_benchmark_matrix.py`; it is not a second matrix
  implementation.

The conditioned benchmark uses `--condition-field condition_group` for BBB,
Bioavailability, and Skin. The category vocabulary is frozen from the full
outer-train split, valid/test values must already exist in that vocabulary, and
outer valid/test labels are never used for epoch or threshold selection. After
selecting the epoch by train-only scaffold CV AUROC, the same CV's pooled OOF
scores determine a macro-F1 decision threshold. ClinTox has no accepted
condition groups and therefore uses the same runner without this option. Fresh
valid artifacts and the exact feature/threshold contract are under
`outputs/baselines/starling_conditioned_valid_v1/<Task>/minimol_train_retest/`.

The paper-facing MiniMol baseline is condition-aware whenever the task has an
accepted condition taxonomy. Fresh valid macro-F1 is 0.6912 for BBB, 0.5872 for
Bioavailability, 0.6030 for Skin, and 0.6520 for unconditioned ClinTox. The
train-only OOF thresholds are respectively 0.6474, 0.6056, 0.6306, and 0.2508.
For the three conditioned tasks, AUROCs are 0.7845, 0.7367, and 0.6776.
Bioavailability's earlier apparent 0.4748 regression was caused by evaluating
the same scores at an uncalibrated threshold of 0.5, not by the condition feature
or generic runner.

### BBB gold-v4 candidate matched baselines

The isolated `experimental_meaningful_cns_access_v4` /
`context_conditioned_selected_v3` candidate uses 3,053 train rows and 397
scaffold-valid rows. Its matched artifacts are under
`outputs/baselines/starling_conditioned_bbb_gold_v4_valid_v1/BBB_Martins/`;
the 398-row selected-v1 artifacts above remain historical and were not
overwritten. All five candidate baselines evaluate all 397 rows:

| method | Macro-F1 | accuracy | AUROC |
|---|---:|---:|---:|
| condition-aware MiniMol head | **0.6674** | 0.7053 | 0.7772 |
| MiniMol KNN same-condition then null | 0.6071 | 0.6952 | 0.6717 |
| MiniMol KNN all train | 0.5708 | 0.6650 | 0.6536 |
| Morgan KNN same-condition then null | 0.5958 | 0.7078 | 0.6512 |
| Morgan KNN all train | 0.6174 | 0.7204 | 0.6603 |

The head reruns the same 5-fold train-only scaffold-CV contract, selects epoch
3, and freezes the new pooled-OOF macro-F1 threshold at `0.62316`. Outer valid
and test labels remain excluded from epoch and threshold selection.

The training audit is in
[`HEAD_TRAINING_DIAGNOSTICS.md`](HEAD_TRAINING_DIAGNOSTICS.md). Canonical
paper-facing data lineages and metrics remain in
[`STARLING_BENCHMARK_RESULTS.md`](../../../tools/chembl_tool/paper_experiments/STARLING_BENCHMARK_RESULTS.md).

The six-condition MiniMol-retrieval sensitivity completed with zero failures:

| task | Starling direct | Starling full-flat |
|---|---:|---:|
| BBB | 0.652832 | **0.690073** |
| Bioavailability | 0.594314 | **0.623542** |
| Skin | 0.599776 | **0.612770** |

Artifacts are under
`outputs/paper/molecular_evidence_agent_scaffold_current_latest_valid_gpt_oss_120b_minimol_top5_nothreshold/`.
This is a valid-only sensitivity and does not replace each task's frozen
paper-facing retrieval setting.

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
