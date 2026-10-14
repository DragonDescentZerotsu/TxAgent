# TxAgent

TxAgent predicts six molecular endpoints from frozen gold labels and retrieved
experimental evidence. The supported inference workflow is the FINAL flat
harness; direct-only inference uses its direct selection variant.

```bash
python -m predict.harnesses.branches --organization flat --help
python -m predict.harnesses.branches.matrix --help
python -m optimization.selection.search --help
python -m optimization.selection.pool --help
python -m predict.baselines.structure_knn.run --help
python -m predict.baselines.minimol.run_bioavailability_ma --help
```

`predict/` owns inference, transport, retrieval, baselines and traces.
`data/processing/` owns gold-label and evidence-library publication.
`optimization/` owns context selection, offline DILI scoring and reviewed
semantic grouping. `analysis/` owns independent evidence and prediction studies.
Deterministic chemistry tools live in `tools/service/`; manuscript fragments and
paper tables live in `docs/paper/`.

Gold/evidence use release manifests and `CURRENT`. TDC uses the single
`data/gold_labels/TDC/manifest.json` configuration. Prediction/prior templates
and all six task profiles live in `predict/prompts/`. Completed runs
and published data are immutable. See [predict/README.md](predict/README.md),
[optimization/README.md](optimization/README.md), [finalize.md](docs/finalize.md), and
[the selected result tables](docs/results.md).

Retained historical prompts and instruction snapshots live in `archive/`.
Git retains retired executable workflows and ChEMBL experiment reports.
[The cleanup inventory](docs/cleanup.md) lists past deletions and relocations;
[the cleanup log](docs/look/cleanup_candidates.md) records subsequent removals.

Semantic code and artifacts live under `optimization/semantic_buckets/` and
use `optimization.semantic_buckets` imports. Saved metadata uses this canonical
location; there is no root alias.

Experiment code follows [a small core for many experiments](docs/experiment_code.md):
reusable capability commands in Python, concrete variants in Bash recipes.
Conditioned Morgan KNN, MiniMol KNN and MiniMol head experiments share the
[conditioned baseline recipe](experiments/baselines/conditioned.sh); see
[its settings and outputs](predict/baselines/minimol/README.md#conditioned-baseline-recipe).
