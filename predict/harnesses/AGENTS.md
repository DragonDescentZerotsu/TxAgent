# Prediction harness architecture

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

## Full-batch execution contract

Direct, flat, full, and progressive full batches default to uninterrupted
throughput. They require an explicit global `--parallelism`, stream each runnable
prompt into one rolling ready queue, and backfill freed slots across every task
and condition in the invocation. Live/pilot behavior is opt-in only.

All harnesses read the mutable candidate inventory at
`predict/api_client/providers/current_endpoints.json` unless another inventory is
explicitly supplied. At launch they probe candidates concurrently, retain every
healthy endpoint advertising the exact model, and set effective parallelism to
the smaller of requested parallelism and healthy capacity. Record both values and
all probe outcomes. A dead or wrong-model endpoint never blocks a batch while one
compatible endpoint remains; fail only when none remain. Endpoint names never
enter scientific run identity. `--prepare-only` performs no endpoint probe.

## Current compatibility boundary

The repository still supports historical task commands under
`tools/chembl_tool/`. Those outer compatibility modules import canonical code
directly from `branches/` or `progressive/`; do not add a second compatibility
layer inside `predict/harnesses/`. No module under `predict` may import
`tools.chembl_tool`; compatibility always points from the old path into
`predict`, never back. When the remaining legacy `experiment_mode` dispatch is
removed, preserve old commands at the outermost CLI boundary rather than
carrying mode switches through the runtime.

## Flat prompt bundles

Versioned flat group prompts live under `branches/prompts/<version>/`. A bundle
owns the flat system message, task group instructions and schema, retrieval-mode
guidance, and upstream provenance. Resolve the named directory exactly and never
fall back to another version. Single-molecule and final prompts remain task-owned;
each flat bundle therefore pins the compatible task prompt profile. Preserve used
bundles byte-for-byte and create a new directory for later prompt changes.
The public flat CLI defaults to `joseph-flat-v2`. V2 keeps every nonempty
source-contract-approved scientific field in a core field,
`experimental_details`, or raw `extra_details`; it omits evidence-molecule names,
duplicate source structures, row/extraction metadata, and group comparison tools.
Its output claims cite stable card IDs. V1 and Tianang remain explicit historical
versions.

## Progressive harness

`progressive/runner.py` exposes a different algorithm: evidence becomes visible
in ordered levels and each level updates an append-only reasoning state. The
package owns its retrieval, state, prompt, and task contracts. It shares the
model engine, tool client, trace format, benchmark meaning, and low-level index
operations, but no branch lifecycle code.

Progressive prompt versions own their system/shared-user content, task instructions,
level overlays, and card schemas in `progressive/prompts/<version>/`. Create a new
directory for a behavior change; do not edit a version used by an experiment.
The active `reranked_progressive_v8/` bundle supports `assay-transfer`, `joint`,
and `morgan`. Historical assets are absent from this active checkout; their
aggregate hashes and removal scope are in `prompts/migration_receipt.json`.
Show transfer scores for assay-selected evidence: L1 molecule scores describe
scored gold contexts; later scores belong to individual records. Joint L1 shows
available scores and panel ranks from both methods. Morgan mode shows similarity
once per molecule; joint shows it for L1/L5 evidence, scoped by
`morgan_score_levels`. Assay-transfer mode shows Morgan similarity only for L5 evidence.
Do not restore the long shared ranking block or duplicate sampling instructions.
Assay transfer means learned transfer of reference assay evidence to the query
molecule. Preserve each selected record's approved source-schema experimental
columns in `experimental_details`; do not assume one schema per level. Omit nulls,
empty sections, duplicate core text, extraction metadata, and raw `extra_details`.
Hide `source_id` and `measurement_kind` from model-visible cards; keep provenance
internally. Semantic field coverage does not change record caps or retrieval.
Only the final v8 model-visible projection removes `protocol.version`,
`retrieval_by_level`, `prompt_mode`, molecule `analog_id`, record `evidence_family`, and
record `retrieved_by`. Keep them internally for rendering, grouping, and provenance.
Full level descriptions (`full_level_plan`), stable short card IDs, current family/level,
new-card markers, and score scopes remain visible.
Older Morgan-selected evidence retains its scoped similarity in later updates.
Strip duplicate similarity from model-visible tool text without modifying receipts.
V9 gold-train parents supply L1 rankings. BBB uses the verified
`--gold-context-mapping` membership to attach V10 L1 records and retire empty
contexts before parent deduplication. Assay-transfer/joint require exact rendered
V9 prompt hashes after redistribution by default. The explicit
`--allow-frozen-l1-vote-scores` opt-in accepts pre-redistribution vote percentages
only after verifying the original frozen prompt hash; audit the old/new counts
and accepted approximation. No other cache or identity check is bypassed. L2-L4
and oral L6 use identical finalized-cache assignments for Morgan and assay transfer;
L5 uses full mapped-V10 Morgan selection. No similarity floor or online cache build.
The default bundle is `reranking_caches_three_pools.yaml`; `--record_pool` (alias
`-record_pool`) selects `assay-transfer-trained` (default), `all_transfer_eligible`,
or `all`, mapped to stored `tool-accepted`, `tool-compatible`, or `all` respectively.
These retain existing bucket-membership/finite-scalar/all semantics, not a new
row-level eligibility filter. L1 and full-library Morgan L5 are unchanged.
Record both public and stored names in manifests. Support the compact and three-pool schemas,
using their declared parent count rather than imposing 75 or truncating a larger
pool. Prepared caches remain unusable until scoring/finalization completes.
Library, mapper, and cache bundle are explicit CLI inputs. Whole-library hash
differences require exact all-field equality of the complete cached-level rows
under an unchanged mapper, with the scope comparison recorded in the run audit.
Default reasoning uses PARCC LiteLLM and its explicit `LITE_LLM_KEY` credential;
do not substitute an OpenAI or OpenRouter credential.
External provider fallback is opt-in, and offline preparation requires no LLM credentials.
`prompt.py` is the public construction boundary; its private `_records.py` helper
applies record/card schemas, while `inference.py` owns calls and state checkpoints.
New manifests pin the complete asset bundle and assembly code.

`predict/harnesses/progressive/matrix.py` runs the explicitly requested full-depth validation sweep
through one shared inference pool (up to 2048 unique requests, without changing
the single-condition runner's 512 budget). It drives `runner.query_steps`, shares
exact rendered requests and generation contracts, prioritizes L1, and releases
dependent stages immediately. Condition `matrix_execution.json` receipts identify
the actual shared inference settings; runner manifests describe preparation.
Only validated responses are reusable; preserve alias-to-record provenance per
consumer and count reused stages separately from actual model calls. Matrix
raw runs stay under `outputs/paper/assay_transfer_harness/joseph/`; derived
analysis outputs and tables go under `outputs/analysis/prediction/`.

New matrices should use `--study` instead of inventing a descriptive flat
`--output-root`. The matrix writes one
`<study>/<method>/k<K>_m<M>_<YYYYMMDD_HHMM>/` leaf per retrieval condition and
keeps shared inference responses under `_batches/<batch-id>/`. The raw leaf is
complete only when `progressive_run_diagnostics.v2` has indexed the exact
visible molecule/group/card identifiers, recovered one frozen selected-context
label per L1 unit, and measured explicit private-reasoning references. The
diagnostic fields remain underscore-prefixed internal preparation provenance;
prompt renderers must continue to ignore them.

Organized runs require a prompt successor with a
`progressive_reasoning_references.v1` runtime contract. The renderer writes its
deterministic reference index beside the exact request, and diagnostics verify
and consume that same index. Use one global, unique `Molecule N` sequence across
L1 and nested later evidence, an independent `Evidence group N` sequence, and
stable `Cxx` card aliases. Matching is case-insensitive. Coverage is
observational: no mention is required, 0/K is valid, and coverage cannot alter
the response validator, retry policy, grammar, or prediction. Historical
weighted-v1 prompts with duplicate molecule labels remain auditable, but their
molecule and combined coverage are explicitly undefined rather than guessed.
