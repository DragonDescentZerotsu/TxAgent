# Prediction harness architecture

## Testing discipline

Do not add circular tests that merely assert newly written prompt prose or copy
implementation literals into the test. Prompt wording is validated with reviewed
input/output fixtures or a real pilot/evaluation. Automated tests should cover
executable behavior, failure modes, schemas, rendering validity, and provenance.

## Reader-first rule

A reader must be able to open any file in this directory and understand what
that file does without first tracing imports. Every module starts with a
docstring that explains:

1. its responsibility;
2. its inputs and outputs;
3. the execution steps it owns; and
4. which named shared component performs any delegated work.

Names such as `plan_query`, `run_harness`, or `prepare` are not explanations.
If such a shared operation exists, its docstring must describe the concrete
work: retrieval, branch construction, dependency scheduling, model calls,
checkpoint writes, and return value as applicable.

## Ownership

- Behavior used only by direct inference belongs in `branches/direct.py`. This includes
  direct-only retrieval, validation, prompt preparation, and branch decisions.
- Behavior used only by flat inference belongs in `branches/flat.py`.
- Behavior used only by full inference belongs in `branches/full.py`.
- Behavior shared by direct, flat, and full belongs under `branches/`; it is
  not shared with progressive unless progressive has a real caller.
- Progressive-only behavior belongs under `progressive/`.
- `branches/` and `progressive/` must not import each other.
- The launcher may own task selection, common CLI parsing, concurrency,
  dependency scheduling, retries, and resume. It must not choose an evidence
  scope or grouping policy.
- Do not create a generic `utils.py` or `utils/` dumping ground. Shared modules
  are named after their job, such as `runtime`, `retrieval`, or `artifacts`.

Small duplication in the public harness files is preferred when it lets a
reader understand an experiment from one file. Do not hide semantic behavior
behind a mode registry merely to remove a few repeated lines.

## Harness vocabulary

Each harness keeps the vocabulary of its own algorithm. Similar JSON shapes do
not justify a shared semantic class hierarchy.

- Direct, flat, and full are branch-based harnesses. Their configuration and
  semantic operations use explicit branch names such as `BranchDefinition`,
  `BranchRetrievalConfig`, and `retrieve_branches`. Do not use generic names
  such as `EvidenceGroupSpec` in the canonical prediction package.
- Progressive owns its level, state, molecule, and card vocabulary. It does not
  inherit a branch schema, and branch harnesses do not inherit progressive card
  types.
- Share neutral mechanics such as ranking, model transport, scheduling, traces,
  and artifact I/O only when there are real shared callers.
- A card or branch class should exist only when it enforces behavior or a stable
  contract used by multiple callers. Do not add wrapper types solely to make
  the harnesses look structurally alike.

## Standard reasoning lifecycle

Direct, flat, and full inference share the same execution lifecycle but not the
same evidence semantics:

```text
public harness
  -> select and organize evidence
  -> shared launcher prepares all query runs
  -> global scheduler executes ready model stages across queries
  -> stage runtime runs single, group, and final calls
  -> artifact code writes checkpoints, traces, predictions, and metrics
```

The public harness must make the first step explicit. Shared execution code may
operate on the resulting groups but may not silently reinterpret them.

## Current compatibility boundary

The repository still supports historical task commands under
`tools/chembl_tool/`. Those outer compatibility modules import canonical code
directly from `branches/` or `progressive/`; do not add a second compatibility
layer inside `predict/harnesses/`. No module under `predict` may import
`tools.chembl_tool`; compatibility always points from the old path into
`predict`, never back. When the remaining legacy `experiment_mode` dispatch is
removed, preserve old commands at the outermost CLI boundary rather than
carrying mode switches through the runtime.

## Progressive harness

`progressive/runner.py` exposes a different algorithm: evidence becomes visible
in ordered levels and each level updates an append-only reasoning state. The
package owns its retrieval, state, prompt, and task contracts. It shares the
model engine, tool client, trace format, benchmark meaning, and low-level index
operations, but no branch lifecycle code.

`progressive/molecule_card.yaml` is the executable model-visible card contract.
Edit its ordered field mappings to change the molecule and nested evidence-card
JSON; `progressive/state.py` validates and applies it, and the runner pins its
hash in each experiment manifest.
