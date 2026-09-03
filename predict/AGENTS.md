# Prediction package rules

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
- `llm_engine/` owns OpenAI-compatible transport and endpoint profiles.
- `tools/`, `traces/`, and `utils/` contain the narrowly named runtime support
  indicated by their package names.

Every module should say what it owns and where delegated work occurs. Do not
hide harness semantics behind generic registries or generic `utils` modules.
File count is also a maintenance cost. Do not create a file for one helper,
dataclass, or stage of an existing flow; extend the clearest existing owner.
Split a module only when it has a genuinely independent responsibility that is
easier to understand and reuse on its own.
