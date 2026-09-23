# Prediction package rules

## Active data ownership

Active data lives with its semantic owner: gold-bound data under
`data/gold_labels/<Task>/<version>/`, evidence data under its task/release, and
shared reusable caches under `data/caches/`. `data/artifacts/` is audit-only and
must not be a required build or runtime input; complete retired products belong
under `data/legacy/`. Do not add compatibility symlinks.

The evidence-library pipeline owns scientific level assignment. Preserved
gold-version mappings live under `data/gold_labels/<Task>/level_mappings/<version>/`.
BBB and Bioavailability runtime consumers use the active release-owned
`data/evidence_libraries/<task>/<release>/level_mapping/`; Ames, DILI,
Carcinogens, and Skin keep their gold-owned mappings until reviewed replacements.
Voter membership may validate L1 coverage but must never derive or rewrite levels.
Corrections and publication belong to the evidence-library pipeline and must use
reviewed UID decisions with pinned input hashes.

Node-local `/local` may hold active working or staging state, including a running
job's reconstructable or resumable work, but it is never durable or canonical.
When the writer finishes, copy or move the closed tree to `/vast`, validate and
publish the `/vast` copy, then remove local staging. Never report work as durable,
complete, or published while it exists only under `/local`. If migration is needed
before completion, stop at a flushed boundary before copying and resume only after
the `/vast` copy validates.

Full-flat V5 artifact payloads are external build inputs, not tracked prediction
code. Their tracked contracts live under
`retrieval/assay_reranking/flat_v5_manifests/`; evidence and cache payloads remain
under their semantic owners. Restore compressed payloads with
`predict.retrieval.assay_reranking.artifact_bundle`, then run its one-time
`verify` command to create the ignored receipt. Inference may require and cheaply
check that receipt, but must not inventory, package, rebuild, or hash whole
payloads online. Missing, stale, or configuration-mismatched receipts are hard
preparation failures. Explicit cache paths remain authoritative; task/benchmark
defaults may select the Gold-v1 all-task or TDC bundle as documented by the flat
harness.

The six-task upstream Gold-v1 query-prior default is
`outputs/paper/assay_transfer_harness/joseph/query_priors/CURRENT.json`, with
hash-pinned `valid_small`, `valid`, and `test` batch roots. It applies only to
`full_flat_context_v5_six_tasks_upstream_v1`; explicit `--prior-root` remains
authoritative. TDC requires its own reviewed overlay, not a Gold prior default.

## Testing discipline

Do not add circular tests that merely assert newly written prompt prose or copy
implementation literals into the test. Prompt wording is validated with reviewed
input/output fixtures or a real pilot/evaluation. Automated tests should cover
executable behavior, failure modes, schemas, rendering validity, and provenance.

## Boundary

`predict/` owns everything required after benchmark and evidence artifacts have
already been built: task adapters, retrieval, prompt rendering, model calls,
scheduling, checkpoints, and traces.

- Canonical inference code must not import `tools.chembl_tool`.
- It may read immutable artifact schemas and current benchmark paths from
  `data.processing`, but it must not invoke builders, audits, or source-policy
  reconstruction during inference.
- Historical modules under `tools/chembl_tool` may import `predict` as thin
  compatibility wrappers. The dependency must never point back from `predict`.
- Place code by stable semantic ownership, not current caller count alone. A
  concrete task prompt, threshold, or policy may stay harness-independent when
  it is naturally reusable by future inference methods, even with one caller
  today. Future-aware placement is not permission to add unused interfaces,
  registries, or abstraction layers; keep the concrete implementation simple.

## Layout

- `harnesses/branches/` owns the complete direct/flat/full universe: task
  adapters, retrieval organization, branch reasoning, scheduling, and prompts.
- `harnesses/progressive/` owns the complete progressive universe: task
  contracts, cumulative retrieval, cards, state, prompt, and runner.
- These two packages must not import each other.
- `tasks/` contains harness-independent benchmark meaning, such as label scope,
  prompt profiles, thresholds, and task policies, regardless of how many
  harnesses currently import each item.
- `retrieval/` contains low-level compact-index and neighbor primitives used by
  both universes; it does not build evidence libraries. Keep this package small:
  `policies.py` owns normalization, exclusions, and candidate ordering;
  `retrieve.py` executes index search and assembles evidence; `compact.py` loads
  compact artifacts; and `features.py` owns retrieval feature backends.
- `llm_io/` contains only model-visible evidence/query contracts and
  response-validation used by both universes.
- `api_client/` owns OpenAI-compatible transport and endpoint profiles.
- `tools/`, `traces/`, and `utils/` contain the narrowly named runtime support
  indicated by their package names.

Every module should say what it owns and where delegated work occurs. Do not
hide harness semantics behind generic registries or generic `utils` modules.
File count is also a maintenance cost. Do not create a file for one helper,
dataclass, or stage of an existing flow; extend the clearest existing owner.
Split a module only when it has a genuinely independent responsibility that is
easier to understand and reuse on its own.
