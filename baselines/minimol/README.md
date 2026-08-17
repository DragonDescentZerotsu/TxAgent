# MiniMol baselines

This directory has one shared frozen-embedding runtime and two deliberately
separate experiment lineages.

## Shared runtime

- `embedding_runtime.py`: MiniMol loading, embedding, cache, and checkpoint
  provenance.
- `head_runtime.py`: the shared classification/regression MLP head, loss,
  corrected epoch-level learning-rate schedule, and prediction helpers.

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

The training audit is in
[`HEAD_TRAINING_DIAGNOSTICS.md`](HEAD_TRAINING_DIAGNOSTICS.md). Canonical
paper-facing data lineages and metrics remain in
[`STARLING_BENCHMARK_RESULTS.md`](../../tools/chembl_tool/paper_experiments/STARLING_BENCHMARK_RESULTS.md).

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

The immutable release snapshot is under `data/starling_table2_v3/`; the full
protocol, results, diagnostics, and artifact locations are in
[`STARLING_TABLE2_REPRODUCTION.md`](STARLING_TABLE2_REPRODUCTION.md).

This reproduction uses a different external release, task definitions, and
splits. Its Table 2 metrics must not be added to the current TxAgent Starling
benchmark ledger or canonical figures.
