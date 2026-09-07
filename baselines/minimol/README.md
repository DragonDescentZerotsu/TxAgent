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
- `baselines/conditioned_knn.py`: shared unique-molecule condition-first/null-
  fallback selection used by both Morgan and MiniMol embedding KNN runners.

## Current TxAgent conditioned benchmark

All current inputs use `data/conditioned_benchmark/<Task>/{scaffold,random}/`.
The maintained entrypoints are:

- `run_bioavailability_ma.py`: shared trained head for a supplied task split.
- `run_train_cv.py`: train-only scaffold CV for epoch and OOF threshold selection.
- `run_embedding_knn.py`: cosine KNN over frozen MiniMol embeddings.

Use `--condition-field condition_group` for BBB, Bioavailability, Skin, and Ames.
Freeze the condition vocabulary from outer train; valid/test categories must
already exist in it. ClinTox has no accepted condition groups and omits this flag.
Select the epoch by train-only scaffold-CV AUROC, freeze a macro-F1 threshold
from pooled OOF scores, and fit the final head on all train rows. Outer valid/test
labels are excluded from both selections. KNN uses unique train molecules only;
the conditioned comparison retains same-condition-then-null and unrestricted
train variants with frozen k=3. Ames valid v2 explicitly uses
`--condition-policy same_condition_then_null_then_all`: keep same-condition
neighbors, then unannotated-condition neighbors, and fill any remaining slots
from unrestricted train molecules by similarity. Molecules remain unique and
each fallback neighbor is marked `all_train_fallback`. This optional policy
does not change existing `same_condition_then_null` runs.

Current scores, artifact roots, input hashes, and checkpoint dependencies have
one authoritative index: [current_conditioned_results.json](../../tools/chembl_tool/paper_experiments/current_conditioned_results.json),
with tables in [RESULTS.md](../../tools/chembl_tool/paper_experiments/RESULTS.md).
The completed three-task test comparison is under `matched_baselines.scaffold_test`.
BBB/Skin reuse verified train-only-selected checkpoints; Bioavailability was
retrained after the two-row train correction. Its older valid scores remain
pre-fix references and must not be reported as current.

Training diagnostics are in [HEAD_TRAINING_DIAGNOSTICS.md](HEAD_TRAINING_DIAGNOSTICS.md).
Historical gold-version and agent-retrieval sensitivities are not alternative
current benchmark inputs or score tables.

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
