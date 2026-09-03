# Prediction package

`predict` is the canonical home for inference-time code. Existing task and
paper entry points remain as compatibility wrappers, so saved commands and
resume artifacts keep working.

```text
predict/
  baselines/                 Morgan, MiniMol, and structure-KNN baselines
  harnesses/
    branches/                complete direct/flat/full inference universe
      artifacts.py          branch input rows and historical trace artifacts
      direct.py              direct-only evidence organization
      flat.py                one collapsed mechanism branch
      full.py                one branch per mechanism family
      payload.py             branch-specific model-visible query projection
      tasks/                 branch-specific task adapters and prompt assets
    progressive/             complete progressive inference universe
      runner.py              CLI, concurrency, checkpoints, and model calls
      state.py               append-only molecule/card state transitions
      retrieval.py           cumulative-family evidence selection
      prompt.py              progressive prompt renderer
      progressive.jinja      progressive prompt template
      molecule_card.yaml     executable model-visible molecule/card contract
      tasks/                 progressive task contracts
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
  llm_engine/                shared OpenAI-compatible client and provider pool
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
python -m predict.harnesses.branches.direct --task bbb_martins --help
python -m predict.harnesses.branches.flat --task bbb_martins --help
python -m predict.harnesses.branches.full --task bbb_martins --help
```

Completed model stages are copied to `predict/traces/runs/` by default. These
copies are for inspection; the original output checkpoints remain authoritative
for resume. View standard and progressive traces together with:

```bash
bash predict/traces/viewer/start_viewer.sh 8776
```

Provider profiles live under `predict.llm_engine.endpoints`. They supply endpoint
defaults to the shared transport; they do not introduce provider-specific
harness logic.

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
