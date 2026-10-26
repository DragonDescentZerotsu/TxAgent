# TxAgent

TxAgent predicts six molecular endpoints from frozen gold labels and retrieved
experimental evidence. The supported inference workflow is the FINAL flat
harness; direct-only inference uses its direct selection variant.

```bash
python -m predict.harnesses.branches --help
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
Deterministic chemistry tools live in `tools/service/`. Paper result sources and
verification limits are indexed in [docs/results.md](docs/results.md).

Gold/evidence use release manifests and `CURRENT`. TDC uses the single
`data/gold_labels/TDC/manifest.json` configuration. Prediction/prior templates
and all six task profiles live in `predict/prompts/`. Completed runs
and published data are immutable. See [predict/README.md](predict/README.md),
[optimization/README.md](optimization/README.md), [finalize.md](docs/finalize.md), and
[the selected result tables](docs/results.md).

Retained historical prompts live in `archive/`; agent instruction snapshots are
local maintenance files excluded from the release tree.
Git retains retired executable workflows and ChEMBL experiment reports.
[The cleanup log](docs/look/cleanup_candidates.md) records deletions and relocations.

Semantic code and artifacts live under `optimization/semantic_buckets/` and
use `optimization.semantic_buckets` imports. Saved metadata uses this canonical
location; there is no root alias.

Experiment code follows [a small core for many experiments](docs/experiment_code.md):
reusable capability commands in Python, concrete variants in Bash recipes.
Morgan KNN, MiniMol and RASAR use the shared
[baseline recipes](experiments/baselines/README.md), with explicit frozen inputs
and the original validation-selection settings.

The collaborator handoff is on `merge_release_v1`; main has not been replaced.
The [integration status](docs/release_refactor.md) records completed acceptance
and remaining input distribution and provenance work. See the
[reproduction inputs](docs/results.md#reproduction-inputs-and-distribution)
for what each restore or execution command requires.
