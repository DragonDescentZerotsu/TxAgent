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
Deterministic chemistry tools live in `tools/service/`.

Gold/evidence use release manifests and `CURRENT`. TDC uses the single
`data/gold_labels/TDC/manifest.json` configuration. Prediction/prior templates
and all six task profiles live in `predict/prompts/`. Completed runs
and published data are immutable. See [predict/README.md](predict/README.md),
[optimization/README.md](optimization/README.md).

Retained historical prompts live in `archive/`; agent instruction snapshots are
local maintenance files excluded from the release tree.
Git retains retired executable workflows and ChEMBL experiment reports.

Semantic code and artifacts live under `optimization/semantic_buckets/` and
use `optimization.semantic_buckets` imports. Saved metadata uses this canonical
location; there is no root alias.

Experiment code uses reusable capability commands in Python and concrete
variants in Bash recipes.
Morgan KNN, MiniMol and RASAR use the shared
[baseline recipes](experiments/baselines/README.md), with explicit frozen inputs
and the original validation-selection settings.

The collaborator handoff is on `merge_release_v1`; main has not been replaced.

The root `docs/` directory contains internal documentation and is excluded from
Git, along with local `logs/`. Public method and usage documentation remains next
to the implementation. Result aggregation commands that read `docs/paper/` require
the separately supplied internal inputs; a public checkout alone cannot rebuild
those internal tables. Frozen publication bundles retain their historical references.
