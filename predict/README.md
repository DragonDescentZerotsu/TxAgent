# Prediction package

`predict` is the canonical home for inference-time code, including harness
matrix launchers. Scientific comparisons live under `analysis/`, and their
derived artifacts live under `outputs/analysis/`.

```text
predict/
  baselines/                 Morgan, MiniMol, and structure-KNN baselines
  harnesses/
    branches/                complete direct/flat/full inference universe
      artifacts.py          branch input rows and historical trace artifacts
      direct.py              direct-only evidence organization
      flat.py                one collapsed mechanism branch
      full.py                one branch per mechanism family
      prompt.py              model-visible query and assay-score projection
      inference.py           validated single/group model calls
      visibility.py          identity-blind projection and leak checks
      runner.py              shared CLI preparation, resume, and summaries
      scheduler.py           throughput-oriented ready-prompt scheduling
      tasks/                 branch-specific task adapters and prompt assets
    progressive/             complete progressive inference universe
      retrieval_cache.py     indexed reads from the prepared SQLite cache
      prompt.py              exact progressive message construction
      grammar.py             version-owned reasoning-grammar rendering
      inference.py           model calls, validation, and state checkpoints
      runner.py              CLI, query preparation, resume, and manifests
      _records.py            private record-to-card prompt assembly
      state.py               append-only molecule/card state transitions
  tasks/                     benchmark meaning shared by both universes
    bbb_martins/
    bioavailability_ma/
    skin_reaction/
  retrieval/                 shared compact-index and neighbor primitives
    policies.py              molecule identity, exclusions, candidate ordering
    compact.py               compact evidence-index loading
    features.py              Morgan and MiniMol similarity backends
    retrieve.py              index search and evidence assembly
    assay_reranking/         offline V9 direct and BBB V19.1 cache generation
    cache/                   ignored reusable candidates and model scores
  llm_io/                    shared model-input and response contracts
  api_client/                shared OpenAI-compatible client and provider pool
  tools/                     inference-time tool client and prefetch helper
  traces/                    portable trace schema, catalog, and viewer
  utils/                     artifact I/O helpers
```

The two harness packages do not import each other. Branch inference owns all
direct/flat/full scheduling, task adapters, reranking, and branch reasoning.
Progressive inference owns its cumulative retrieval, append-only state,
task contracts, prompt, and runner. They share only low-level index primitives,
benchmark meaning, model transport, the tool client, traces, and artifact I/O.

```bash
python -m predict.harnesses.progressive --help
python -m predict.harnesses.branches --organization direct --task bbb_martins --help
python -m predict.harnesses.branches --organization flat --task bbb_martins --help
python -m predict.harnesses.branches --organization full --task bbb_martins --help
```

New harness runs write viewer traces under
`outputs/paper/live/<study>/<method>/<run>/<task>/<condition>/`. Live mode publishes the first
three samples and pauses; throughput mode is the full-batch default and runs
privately without a pilot gate. Full inference requires an explicit global
`--parallelism`; offline `--prepare-only` does not.
The original output checkpoints remain authoritative for resume. View standard
and progressive traces together with:

```bash
bash predict/traces/viewer/start_viewer.sh 8776
```

The mutable endpoint inventory is
`predict/api_client/providers/current_endpoints.json`. Update it when endpoint
hosts or ports change. At launch the shared client probes all candidates,
retains healthy exact-model endpoints, clamps effective parallelism to their
capacity, and records every outcome. A dead endpoint does not block inference
while another compatible endpoint is alive. Harnesses do not carry DGX-specific
connection code, and endpoint names do not enter run identity. The current
cache-V3 launch recipe is in
[`PROGRESSIVE_PREDICTION_RUNBOOK.md`](PROGRESSIVE_PREDICTION_RUNBOOK.md).

Assay-reranking caches have one visible split: current V3, BBB V24.1, and Oral
V25 artifacts live under `predict/retrieval/cache/assay_reranking/active/`;
retained historical profiles live under `archive/` and require an explicit
`--legacy` cache selection. See `predict/retrieval/assay_reranking/CACHE_LAYOUT.md`.

Inference code has a one-way package boundary: `predict` may read built artifact
contracts and benchmark paths from `data.processing`, but never imports
`tools.chembl_tool`. Old `tools/chembl_tool` entry points now point into
`predict` only for command compatibility; builders and audits remain outside the
inference package.

Canonical baseline module examples:

```bash
python -m predict.baselines.structure_knn.run --help
python -m predict.baselines.minimol.run_embedding_knn --help
python -m predict.baselines.minimol.run_train_cv --help
```
