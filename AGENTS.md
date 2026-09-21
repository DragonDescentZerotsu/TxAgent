# TxAgent: resident molecular tools and evidence-retrieval reasoning

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

For prompt-only validation or prompt-surface comparisons, reuse an existing
prepared retrieval artifact and call the renderers directly. Do not rerun cache
materialization, whole-library scans, or cache integrity checks unless retrieval
selection inputs or selection code changed, or the user explicitly requests it.

An approved full-batch run must launch directly in uninterrupted throughput mode;
it must never send or wait for pilot responses before starting the full batch. For
the progressive matrix launcher, use `--execution-mode throughput` and do not pass
`--pilot-queries` or `--continue-after-pilot` in a full-batch command. Pilot or
canary inference is allowed only when explicitly requested as a separate run, with
its own artifact root, and must never gate an approved full-batch run.

Every newly launched LLM request must use `reasoning_effort=high`, never `max`.
Do not launch a workflow whose request configuration cannot enforce this setting.
Historical artifacts remain immutable; changing reasoning effort on resumed work
requires a fresh run identity unless the request bytes are otherwise unchanged.
The V10 measurement-resolution pipeline for AMES, DILI, Skin Reaction, and
Carcinogens is the sole exception: its gold pilots and later full extraction use
`reasoning_effort=low` under fresh run identities. All other requests remain high.

## Language convention

Use English for user-facing communication and newly generated documentation,
reports, plots, captions, and experiment conclusions. Keep code identifiers,
commands, paths, model names, experiment IDs, and machine-readable fields unchanged.

## Reranked Progressive v2

"Run reranked progressive" means the current BBB/oral harness
`python -m predict.harnesses.progressive --harness-version reranked-progressive-v2`.
The CLI default is `reranked-progressive-v2`. The active cache has exactly two
ranking modes; the default is `assay-transfer`.

| Mode | L1 | L2-L4 | L5 | L6 (oral only) |
|---|---|---|---|---|
| `assay-transfer` | Top 10 assay-ranked context cards | Assay transfer | Morgan | Assay transfer |
| `morgan` | Top 10 Morgan-ranked context cards | Morgan | Morgan | Morgan |

Default shape: up to ten ordered records per L1 molecule; independently select
up to 50 records per level from L2 onward, then group by normalized
parent into existing/new molecule cards. Preserve per-record conditions, append-only
evidence, scaffold-disjoint pools and no similarity floor. BBB ends at L5; oral at L6.
Joint remains available only through an explicit archived legacy cache. Skin is
not enabled for this version.

The active BBB/Oral Gold-v1 retrieval contract is
`ranked_uid_retrieval.v1`. Every task, split, and level owns an independent
immutable SQLite cache. L1 is scaffold-disjoint and contains 100 context cards.
L2 and later are parent-disjoint and contain every physical UID under the top
100 Morgan-ranked parents. Each expanded UID universe carries dense Morgan and,
where supported, assay-transfer ranks. The task `RELEASE_INDEX.json` locates the
level caches and evidence-owned projection. Prompt preparation reads only that
small index, selected manifests, indexed SQLite rows, and selected projection
rows. It must not reread source or mapping Parquet, normalize molecules,
recompute Morgan similarity, recheck disjointness, hash whole databases, or run
integrity scans. Those are one-time publication checks. Flat and progressive
use the same direct reader.

The active contract supports only the `all` pool and native `morgan` and
`assay-transfer` modes; joint and alternate pools are legacy-only. A caller may
request a separate K for every later level. K must be positive and no greater
than that query's expanded UID count; an oversized request fails before prompt
rendering. Every level is selected independently, with no cross-level deduplication.
The L1 cache resolves a selected `context_id` directly to its ordered physical
`source_row_uid` members. The prompt exposes up to its explicit per-card record
limit without representative-row collapse. Gold voter membership, not current
parent normalization, defines the L1 card and nested records. Historical vote
IDs are construction provenance only and never select a representative row. Parent
normalization organizes L2+ display cards. Display adaptation is
publication-time: semantic categories first, then a complete canonical
measurement/unit pair, then a complete source pair, with source text/no-unit
only for non-scalar records. Never mix a measurement and unit from different
origins.

`--score-cache-config` supplies only cache locations; it cannot override the version's
stage rules. Assay/joint stages require complete task/mapping-compatible cached scores;
never substitute a different cache or Morgan ranking on failure. Historical/custom
profiles require `--legacy` and load `prompts/legacy/`. Keep prior results immutable;
future behavior changes require a new version. See the active bundle README for
commands. Put new tables and validation reports under the Joseph directory below.

## Rules

Simplify code whenever possible. Before modifying or reviewing code, read and
follow `.agents/skills/simplicity-first/SKILL.md`.

In the evidence-library pipeline, **pruning always means record-level exclusion
from assay-transfer calibration**. Pruning never rejects or deletes a pair bucket,
and pruned records remain in the canonical and retrieval libraries. Call whole-
bucket pass/fail decisions `assay-transfer bucket eligibility`, never pruning. The
whole-bucket field is `assay_transfer_bucket_eligible`; the historical sidecar
field `bucket_eligible` is only a row-level compatibility alias.

Pair-bucket identity is intrinsic evidence metadata only: source, canonical
endpoint, canonical unit, and the source-specific canonical assay dimensions.
Gold-label membership, benchmark condition/context, split, voter identity,
parent assignment, and any hashes derived from them must never enter
`pair_bucket_key` or `canonical_pair_fields_json`. Those fields remain separate
record metadata for retrieval filtering, labeling, and audit.

The active conditioned benchmark for BBB, Oral Bioavailability, and Skin is
Gold v1. BBB and Oral V10 Stage 1 must protect every physical UID named by the
published Gold-v1 voter contract: a protected row cannot be dropped, repaired to
a different structure, or removed as an exact duplicate. For these two rebuilt
tasks, L1 is exactly the physical membership of published Gold-v1 cards. Vote
aggregation remains separate from visibility: each frozen `vote_id` contributes
once, while every physical member is retained and can be shown to the reasoning
model. In particular, Oral HF/local records coupled to one canonical claim
remain separate L1 records but collectively contribute the single frozen
Gold-v1 vote.

Canonical-claim grouping belongs only to the Gold voting artifacts. Evidence-
library records remain independent physical rows: do not attach
`canonical_claim_id` to them, join them through a Gold claim, or use Gold claim
membership for evidence-library deduplication. Gold may supply each physical
voter's own frozen label and condition for L1 display, while vote aggregation
stays exclusively in the Gold card and vote-unit mappings.

Every V10 Stage-1 source must declare exactly one endpoint role (a raw field or
an explicit constant), one raw measurement field, and one unit role (a raw
field, an explicit constant, or a reviewed exception). The only reviewed unit
exceptions for AMES, DILI, Carcinogens, and Skin Reaction are
`unitless_categorical` and `embedded_in_measurement`; they must be source-local,
reasoned, and recorded in both the extraction cache contract and mapping
manifest. Do not infer undeclared roles while building Stage 1.

### Evidence-library release gates

Materialize the pinned source universe before applying Gold protection. Gold
`voter_membership.parquet` supplies physical UID membership only: never use its
historical structure fields as source authority or reconstruct protected rows
from a prior evidence-library vote table. Protected UIDs retain the pinned source
structure and survive repair and exact deduplication; reviewed structure repairs
apply to nonprotected rows. A Gold card may legitimately contain multiple current
parents, so never repair, merge, drop, or reassign rows merely to force one parent
per card.

For any release with assay-transfer calibration, final Stage 3 requires a
hash-pinned record-pruning manifest. The required order is unpruned Stage 3 for
candidate generation, reviewed record decisions, then a final Stage-3 rebuild
that pins and applies those decisions. Missing or stale pruning is a hard failure,
not an optional skip. Previously reviewed prunes are the immutable minimum for a
successor refresh: new review may add exclusions, while a reversal requires a
separate explicit reviewed successor decision. Pruned rows remain in canonical,
Stage-3, level-mapping, and retrieval artifacts.

After the final Stage-3 rebuild, regenerate the complete release-owned level map
and its assay-transfer eligibility sidecar from that exact Stage-3 hash. Verify
exact Stage-3 UID coverage and exact Gold-v1 physical-voter L1 membership, then
refresh `data/evidence_libraries/level_mappings.v1.json` and immutable test pins.
That global file is only a hash-pinned locator; it does not derive levels. Publish
from a completed staging tree, archive the prior active tree, atomically install
the successor, and update `CURRENT` only after all hashes and row invariants pass.
Never publish build locks, incomplete markers, or working intermediates.

Reviewed measurement/unit maps should be extended as immutable successors rather
than rebuilt from scratch when their input contract remains compatible. Likewise,
new semantic atoms or buckets belong in a separate immutable semantic generation;
do not edit a selected semantic map, ranking, manifest, pair-record file, or live
cache in place.

AMES, DILI, and Carcinogens canonical reconciliation uses
`data/processing/evidence_library/versions/v10/canonical_reconciliation.py`.
The endpoint-aware V3 successor includes only exact units consumed by accepted
deterministic measurement resolution or successful LLM resolution. It performs
two endpoint-local passes over character-TF-IDF K-means clusters of at most 50
labels, using each provisional Pass-A label as the Pass-B input. Each request must
use local DeepSeek with `reasoning_effort=high`; preflight every configured
endpoint, exclude unavailable endpoints, and cap each selected endpoint at 256
in-flight requests. The current authorized V3 pool is exactly `dgx020:50002`,
`dgx005:50001`, and `dgx011:50001`. Read timeouts fail and retry the affected
request but do not permanently quarantine an otherwise reachable endpoint. Every
unit must survive with scale `1`; a failed or uncertain
cluster is identity-mapped. Preserve endpoint-specific rules and require complete,
hash-pinned agent review of any endpoint-local conflicts before immutable V3
publication. Other canonical categorical fields use the shared single-pass
embedding clustering implementation. Preserve V2 bundles and publish successors
only under `data_processing/canonicalization_v10/unit_reconciliation_v3/`; do not
add compatibility symlinks or update a task `CURRENT` pointer as part of unit-only
reconciliation.

V3.2 does not compare every numeric token in the selected span with the canonical
unit; explanatory numbers may be removed. The parsed physical scale/dimension,
qualifier, endpoint-basis, verbatim-span, and mandatory-unit guards remain. Prompts
must explicitly preserve percent, per/population, ratio, inhibition, control,
fold, and count semantics.

Parallel V3.2 OpenAI fallback may receive only a prefix-hashed snapshot of terminal
identity clusters and must write a separate append-only artifact. Use
`gpt-5.4-mini`, `reasoning_effort=high`, the model-supported 128,000 completion-token
ceiling, and `OPENAI_API_KEY_ONE` through `data/processing/llm_api.py`; never copy a
secret or write into the active local provider log.

V3.3 prompt wording must state that clustering is not unification: each ID is mapped
independently, and SI prefixes such as milli, micro, nano, and pico must remain
distinct even when those labels share one lexical cluster.

V3.4 treats standalone `x` beside a numeric factor as multiplication, not a fold
qualifier, and treats slash, `unit-1`, and `unit^-1` as equivalent per-denominator
notation. Prompt examples must retain percent, inhibition, control, ratio, count,
true fold, and per-denominator semantics.

Explain work concisely from the high-level result down to implementation details.
For new implementations, describe the files being added or changed, how they
connect, and the scope of the change. Prefer a small, coherent module over a
sprawling pipeline, reuse existing libraries, and keep tests proportional.

### Artifact hygiene

Raw assay-transfer harness runs for Joseph remain under
`outputs/paper/assay_transfer_harness/joseph/`. Every derived analysis, including
KNN, query-prior, oracle-union, and progressive-level comparisons, must be saved
under `outputs/analysis/<domain>/<study-id>/`, never inside or beside a raw harness
run. Every tabular or numeric result presented in a report must also be exported as
a TSV in the same study directory. Keep the report, TSV tables, and compact
provenance manifest together, and reference immutable input runs by path and hash.
Existing inference runs and reusable caches retain their own locations.

### Gold-v1 submodular context optimization

The reproducible Gold-only selector is `python -m optimization.gold_joint`. It
uses the three 100-query `valid_small` splits for BBB, Oral Bioavailability, and
Skin Reaction. BBB and Oral direct candidates come from the Gold-v1 L1 caches in
`ranked_level_retrieval_v3`; Skin direct candidates come from
`v9_skin_gold_v1_scaffold_morgan100_v1`, pinned to the v9.0.2 Skin
assay-transfer model. Indirect candidates come from each task's active
Gold-bound `ranked_level_retrieval_v3` L2+ universe. Do not substitute a TDC
query or L1 cache.

Direct selection chooses exactly ten Gold context cards. Its terms are normalized
Morgan-gated assay relevance, normalized Morgan-bit coverage, and capacity-
normalized log label diversity. Selecting a card never changes its frozen label
or ordered physical voter membership. The direct grid contains 27 crossed
profiles over gated assay `{.75,1,1.25}`, Morgan-bit coverage `{.25,.5,.75}`,
and label diversity `{.25,.5,.75}`, plus three gated-only controls.

Indirect selection combines all available L2+ UIDs for a query and chooses 50
records jointly, with no per-level quota or minimum. It retains normalized
Morgan-bit coverage. Semantic and level diversity use
`sum(log(1+n_group))` divided by the maximum feasible value for the same query,
capacities, and budget. The denominator is fixed before selection, so each term
is monotone submodular and remains in `[0,1]`. Skin has reviewed semantic bucket
assignments but no reviewed semantic relevance weights; its semantic-relevance
lambda is therefore recorded as unavailable and made effectively zero. Never
invent weights or borrow another task's scores.

The indirect grid validates 94 entries from the three existing normalized-gated
screens, records every source alias, deduplicates them to 75 scientific objective
settings, and crosses those settings with level weights `{.25,.5,.75}` for 225
profiles. A mixed selection freezes one reviewed direct manifest and combines it
with one indirect manifest; the direct panel must never be recomputed implicitly.

Use the isolated Apricot environment even though the successor's deterministic
lazy greedy implementation does not require an N-by-N matrix:

```bash
OPT_PY=/vast/projects/myatskar/design-documents/conda_env/apricot-select/bin/python
OPT_STAGE=/local/$USER/gold_joint_log_diversity_valid_small_v1
$OPT_PY -m optimization.gold_joint direct --output-root "$OPT_STAGE"
$OPT_PY -m optimization.gold_joint indirect --output-root "$OPT_STAGE"
$OPT_PY -m optimization.gold_joint validate \
  --manifest "$OPT_STAGE/direct/ga100_mc050_label050/bbb_martins/manifest.json"
$OPT_PY -m optimization.gold_joint compose \
  --direct-manifest "$OPT_STAGE/direct/<reviewed-profile>/<task>/manifest.json" \
  --indirect-manifest "$OPT_STAGE/indirect/<profile>/<task>/manifest.json" \
  --output "$OPT_STAGE/mixed/<profile>/<task>/manifest.json"
```

Direct manifests contain selected `context_id` values; indirect manifests contain
ordered `source_row_uid` values with their original levels; mixed manifests hash-
pin both. Validate every leaf manifest and the closed copy on `/vast`, then publish
under `outputs/analysis/record_selection/gold_joint_log_diversity_valid_small_v1/`.
Selection artifacts are derived analysis, not inference completion. These commands
must not launch an LLM request, update the Joseph run catalog, or choose a canonical
direct profile without reviewed validation results.

New Joseph progressive runs use a study/method/query-prior/run hierarchy under
`outputs/paper/assay_transfer_harness/joseph/`, for example
`contrastive/morgan/with_query_prior/k10_m3_YYYYMMDD_HHMM/`. The prior directory
is exactly `with_query_prior` for cached query priors or `no_query_prior` when no
prior is visible. Keep prompt, provider, task, cache, and hashes in the leaf
`run.json`; do not encode them into another flat top-level directory name.
Endpoint hosts, ports, and provider allocation never participate in a study,
batch, or run name and remain only in execution receipts. A multi-method matrix
splits into sibling method leaves and shares request/response reuse only through
`_batches/<batch-id>/`. Live traces mirror the same
study/method/query-prior/run identity under
`outputs/paper/live/`. Superseded layouts and prompt experiments belong under
`legacy/`; do not add compatibility symlinks.

The canonical navigational views for organized Joseph runs are
`outputs/paper/assay_transfer_harness/joseph/catalog.json` and
`outputs/paper/assay_transfer_harness/joseph/results_ledger.tsv`. The catalog is
the run-level index; the ledger is one row per run, task, and completed level and
must retain prepared or incomplete runs with blank metrics. Generate both from
leaf `run.json` and canonical metrics artifacts—never edit either view by hand.
Refresh them atomically whenever an organized run is prepared, changes status,
publishes diagnostics, or completes. A completed run is not fully published
until both views include it and their recorded hashes validate. Before answering
whether a run exists or reporting prior results, consult both views and verify
the referenced immutable leaf artifacts. Do not reorganize old runs physically;
use these catalog views to make historical layouts discoverable.

Every completed progressive run, starting with v12 progressive-simple, must
contain `visible_evidence.tsv`, `neighborhood_label_mix.per_query.tsv`,
`neighborhood_label_mix.summary.tsv`,
`reasoning_reference_coverage.per_query.tsv`,
`reasoning_reference_coverage.summary.tsv`, `diagnostics_manifest.json`, and
`report.md`. L1 label mix uses the frozen label of the exact molecule-condition
context that admitted the visible molecule; never infer one unconditional label
per parent. Reasoning coverage counts explicit visible identifiers such as
`Molecule 1`, `Evidence group 1`, and `C01` in the private reasoning stream.
Every new prompt family used by an organized run must declare an immutable
`progressive_reasoning_references.v1` contract. Prompt rendering and diagnostics
must consume the same deterministic reference index; molecule numbering is
global across the prompt, evidence-group numbering is independent, and record
aliases remain `Cxx`. Match identifiers case-insensitively. Repeated mentions
count once for coverage and separately as occurrences. This is an observational
measurement contract, not a requirement to mention every item: 0/K is a valid
result and must not trigger retries or response rejection. A missing label,
visible index, reasoning stream, or provenance receipt prevents completion.
Every active prompt bundle must also be self-contained: keep every Jinja and YAML
asset needed to render it inside its own version directory, and declare the
deterministic reference contract there. `asset_parent_version` is lineage-only
for historical bundles and is not allowed for new organized runs. Shared renderer
and reference-matching code remains centralized and is hash-pinned in run manifests.
Historical partial runs may emit the same diagnostics with an explicit partial
manifest and `diagnostic_gaps.tsv`; missing stages stay outside measured coverage
denominators and can never be promoted to complete.
Diagnostic labels and joins are post-run only and must never enter prompt
preparation or model-visible text. Preserve used prompt bundles; add a successor
bundle rather than changing their rendered prompt or reference rules. The
front-shelf v12, v13, and v14 successors are respectively
`reranked_progressive_l1_simple_v12_references_v1`,
`reranked_progressive_l1_simple_v13_references_v1`, and
`reranked_progressive_l1_simple_v14_references_v1`.

Human-authored documentation lives beside the code, data, prompt bundle, test,
or experiment it governs. Scoped `AGENTS.md`, `CLAUDE.md`, skill instructions,
executable prompt inputs, and reports stored with immutable artifacts remain
beside the content they govern.

Keep repository artifacts only when they are active build inputs or outputs,
immutable scientific records, expensive reusable caches, or compact evidence needed
to audit a published decision. Retain network and paid-model caches when they avoid
repeating external work, and document their scope and version.

Do not commit or publish agent scratch plans, navigation dumps, request batches,
review packets, raw progress logs, deterministic joins, duplicate exports, or
superseded pilots. Put only disposable, reconstructable scripts, checkpoints, and
working files in `/tmp`.
If durable navigation context is genuinely useful, write one small scoped Markdown
index rather than preserving the working directory that produced it.

After consolidating a review or processing stage, keep the canonical result, its
compact manifest or summary, and non-reconstructable decisions. Remove redundant
packets and intermediates once the consolidated result contains the needed source
identity, decision, rationale, reviewer provenance, and input hashes.

## Environment

TxAgent has multiple checkouts. Always inspect the hostname and resolved repository
root before choosing paths or reporting validation. This checkout under
`/vast/projects/myatskar/design-documents/joseph/TxAgent` is authoritative when work
starts here. Do not assume the node002-local `/data1/joseph/TxAgent` checkout exists
or is synchronized on another host.

### Recent failure-prevention checks

Before launching a prediction batch, reopen the provider-pool JSON from disk and
report the exact selected hosts, ports, per-endpoint `max_inflight`, aggregate
capacity, model, and reasoning effort. The mutable global candidate inventory is
`predict/api_client/providers/current_endpoints.json`; do not preserve a competing
run-specific endpoint list in this file.

The provider-pool `max_inflight` limit is local to one launcher, not a shared
endpoint semaphore. Overlapping launchers may use the same endpoints; report each
launcher's configured per-endpoint and aggregate capacity explicitly. Live SGLang
`/v1/loads` or scheduler metrics are observational and do not gate a requested
launch.

Do not silently accept the progressive matrix's 524,288-token default for these
structured L1 prompts. Review token counts from the closest completed artifact
and pass an explicit, justified `--max-tokens`; keep the required reasoning effort
unchanged. In the first V10.4 width-25 launch, 165 requests remained active after
all other work finished and began hitting the 3,600-second read timeout together,
opening both provider circuits and invoking cross-endpoint failover. A timeout
retry can duplicate expensive server work. Any changed token or timeout setting
requires a fresh run identity.

On `epyc-1-6`, do not start the resident molecular tool service with the bare
system `uvicorn`: `/usr/bin/python` lacks PyTorch. First verify the chosen runtime
with imports for `torch`, RDKit, and the service app, then start Uvicorn through
that same interpreter. A reachable port is not evidence that application startup
completed; require a successful `/health` response.

The legacy direct OpenAI-compatible pipeline still requires its configured API-key
environment variable even when the target is an unauthenticated local endpoint.
For direct DeepSeek requests to the local DGX services, explicitly set
`DEEPSEEK_API_KEY=EMPTY` (or the endpoint's documented placeholder) and verify the
rendered base URL before launch; an empty `api_key_env` in a separate provider-pool
configuration does not satisfy the direct client's credential preflight.

Cached query-prior reuse across gold releases is stable-identity based, never
positional. If a historical manifest names a removed input path, bind it to an
existing immutable input with a recorded hash. If the new release introduces a
query, generate only that missing prior under a fresh run identity and expose it
through a hash-pinned overlay; never mutate the historical cache or silently map
the new row by index.

Frozen V9 assay-transfer L1 caches and their original query priors use the V1
gold-label release. Historical V9 matrices must pass `--gold-label-version v1`;
never let the runner resolve `CURRENT` and rely on positional prior fallback.
Before launch, verify every cached-prior query SMILES against the selected V1
split and require complete 397-row BBB and 262-row Bioavailability coverage.

V10.4 L1 Morgan similarity is molecule-condition-context specific. During L1
cache construction, require a stable retrieval-parent rank for repeated parent
rows, but retain and validate each context's own similarity instead of requiring
one parent-wide similarity value.

A progressive matrix resume compares frozen matrix inputs byte-for-byte. If the
frozen invariant includes a live endpoint-preflight receipt, a prepare-only batch
may reject a later execution attempt after endpoint state changes even when its
scientific inputs are unchanged. Preserve the prepared batch, do not weaken or
rewrite its invariant, and launch the approved full batch under a fresh run and
batch identity after repeating preflight.

Before publishing a new immutable cache profile, add its exact profile name to
`ACTIVE_CACHE_PROFILES` and test that `cache_profile_root(profile)` resolves under
`active/`. The generic builder routes every unregistered profile to `archive/`
without treating that routing decision as a build error, so a successful build
alone does not prove that the cache was published to the intended shelf.

### Local staging and canonical publication

Before a high-throughput run, inspect the hostname, resolved repository root,
`df`, and `findmnt` for the exact candidate paths. `/vast/projects` is network NFS;
high-churn active working and staging writes can bottleneck there. When capacity
permits, a running job may keep reconstructable or resumable working state under
node-local `/local`. `/local` is never the durable or canonical home: as soon as
the writer finishes, copy or move the closed tree to `/vast`, validate the `/vast`
copy, publish from there, and remove the local staging tree. Never report work as
durable, complete, or published while its only copy is under `/local`. Do not use
`/tmp` unless its capacity and backing filesystem have first been confirmed suitable.

Never move or copy a tree with an active writer. If local working state must migrate
before completion, stop its writers at a flushed boundary, copy it to staging on
`/vast`, validate the copy, and only then resume from `/vast`.

When actually running from `/data1/joseph/TxAgent` on `node002`, use
`/data1/joseph/miniconda3/condabin/conda run -n txagent-glm`. Outside node002, use
the environment available to the active checkout and report it exactly; do not
claim node002 validation.

Data-processing code must use `data/processing/llm_api.py` for OpenAI-compatible
credentials and clients. Do not add task-local dotenv parsers or silently reuse an
OpenAI credential for OpenRouter. The shared loader reads the sibling
`therapeutic-tuning/distillation/.env`, resolves provider aliases, and fails closed
on missing or mismatched credentials.

Reusable OpenRouter route discovery, qualification, price filtering, and ranking
must use `data/processing/openrouter_provider_pool.py`; task runners must not embed
provider inventories. The default pool is DeepSeek V4 Flash 0731 only. Mixing in
DeepSeek V4.1 Flash requires explicit opt-in and a recorded profile snapshot hash.
Only model-provider routes with immutable gold qualification may receive work.

BBB and Oral assay-transfer record-pruning review uses `gpt-5.4-mini` with
`reasoning_effort=high`. Load the primary credential as `OPENAI_API_KEY_ONE`
through `data/processing/llm_api.py` from the sibling
`therapeutic-tuning/distillation/.env`; never read that file directly or copy its
secret value into code, commands, manifests, or documentation. The external
pruning quota refreshes daily at 8 PM America/New_York; inspect the current
ledger before reporting usage, and do not replace an active non-exhausted epoch
solely because the quota window refreshed.

## Current Objective

This project aims to build a reusable molecular evidence retrieval and reasoning system. BBB_Martins is the first proof-of-concept task;
currently the same workflow has been extended to Bioavailability_Ma and Skin_Reaction. ClinTox has been fully migrated to
`data/legacy/clintox/` and is no longer an active task. The overall flow is: given a query molecule,
first compute molecular properties, structural differences, and property differences through the resident FastAPI tool service, then retrieve experimental readings of similar molecules from the task-specific ChEMBL evidence
library, and finally hand the tool outputs and assay evidence to the reasoning LLM to comprehensively judge the
target label for that task.

Currently implemented ChEMBL reasoning tasks:

```text
tools/chembl_tool/tasks/bbb_martins/
tools/chembl_tool/tasks/bioavailability_ma/
tools/chembl_tool/tasks/skin_reaction/
```

## Current Conditioned Benchmark (2026-08-28)

The three tasks have only one active evaluation entry point:

```text
data/gold_labels/<Task>/v1/scaffold/
```

Do not directly reference historical molecule-only, `selected_vN`, BBB gold-vN, or
ClinTox source-build paths in runners, baselines, or result plots. They are only used for source provenance; the per-split hash/row-level equivalence relationships between the original paths and the current data are uniformly recorded
in `data/artifacts/gold_labels/conditioned_benchmark/migration_receipt.json`. The public path constants, publication entry points, and full contract are:

```text
data/processing/gold_labels/conditioned_benchmark.py
data/processing/gold_labels/publish_conditioned_benchmark.py
data/processing/gold_labels/README.md
```

| task | train / valid / test | current target |
|---|---:|---|
| BBB_Martins | 3,053 / 397 / 393 | experimentally meaningful systemic CNS access |
| Bioavailability_Ma | 1,958 / 262 / 269 | oral bioavailability under the reported condition |
| Skin_Reaction | 1,997 / 246 / 248 | skin sensitization/contact allergy |

All split rows use a unified molecule-condition schema. Rows without an external condition use
`no_reported_external_condition`, and the prompt renderer outputs no condition sentence for them. The train/valid/test parent identity and Bemis-Murcko scaffold overlap for the three active tasks are all 0.

The current split files for BBB, Bioavailability, and Skin are byte-identical to the conditioned cohorts that have already been evaluated. Existing predictions can only be reused when the manifest input hash matches the
migration receipt; they cannot be reused based solely on the old directory name.

Task-specific source voting and review remain in each task module. BBB's direct gold only accepts experimentally meaningful CNS access after systemic administration; Bioavailability's L1 includes only actual voter rows; Skin direct only accepts sensitization/contact-allergy final outcomes. Versioned source/retrieval contracts are provenance, not a second set of gold.

Current Starling random/scaffold frozen label decisions, formal GLM, MiniMol head, Morgan KNN,
MiniMol embedding cosine KNN, MiniMol/cosine agent retrieval, blind progress, Skin retrieval degradation,
Tier 1+2 final-only diagnostics, and code entry points are uniformly recorded in:

```text
tools/chembl_tool/paper_experiments/STARLING_BENCHMARK_RESULTS.md
```

The current paper-facing valid default prompt profiles are BBB `meaningful_cns_access_v1`, Bioavailability
`f20_evidence_calibrated_v2`, and Skin `sensitization_aligned_v2`. BBB v2/v3, BBB property-compatible selector,
Skin negative-transfer v3, train-ratio prior, and matched train-label agent are all reproducible but not promoted valid-only
historical experiments; they must not be used as the default pipeline or to run formal tests based on them. The current version, best valid condition, and artifact index are based on the canonical snapshot at the top of
`tools/chembl_tool/paper_experiments/RESULTS.md`.

Current Starling benchmark main run and summary entry points:

```text
data/processing/gold_labels/build_conditioned_benchmark.py
data/processing/gold_labels/build_record_supported_benchmark.py
tools/chembl_tool/tasks/bioavailability_ma/build_canonical_starling_source.py
tools/chembl_tool/paper_experiments/analyze_starling_parent_provenance.py
tools/chembl_tool/paper_experiments/analyze_starling_majority_thresholds.py
tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py
tools/chembl_tool/paper_experiments/build_minimol_retrieval_features.py
tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
tools/chembl_tool/paper_experiments/seed_starling_matrix_reuse.py
tools/chembl_tool/paper_experiments/analyze_starling_direct_significance.py
tools/chembl_tool/paper_experiments/analyze_starling_best_agent_baselines.py
tools/chembl_tool/paper_experiments/router_oof/cli.py
tools/chembl_tool/paper_experiments/summarize_minimol_retrieval_agent.py
tools/chembl_tool/paper_experiments/summarize_starling_benchmark.py
tools/chembl_tool/paper_experiments/plot_starling_benchmark_overview.py
tools/chembl_tool/paper_experiments/plot_starling_model_comparison.py
tools/chembl_tool/paper_experiments/watch_glm_tunnel_and_matrix.py
tools/chembl_tool/paper_experiments/plot_starling_with_minimol_agent.py
tools/chembl_tool/paper_experiments/run_minimol_valid_matrix_gpt_oss_120b.py
tools/chembl_tool/paper_experiments/run_assay_retrieval_curve.py
tools/chembl_tool/paper_experiments/build_assay_family_catalog.py
tools/chembl_tool/paper_experiments/run_conditioned_assay_family_curve.py
predict/harnesses/progressive.py
tools/chembl_tool/paper_experiments/audit_conditioned_assay_prompt_lengths.py
tools/chembl_tool/paper_experiments/plot_assay_retrieval_curve.py
tools/chembl_tool/paper_experiments/summarize_coverage_selector_llm_matrix.py
tools/chembl_tool/paper_experiments/analyze_coverage_selector_retrieval_changes.py
tools/chembl_tool/paper_experiments/plot_coverage_selector_llm_matrix.py
tools/chembl_tool/paper_experiments/run_minimol_retrieval_agent_experiment.py
tools/chembl_tool/paper_experiments/run_train_ratio_prior_experiment.py
tools/chembl_tool/paper_experiments/train_ratio_prior_analysis.py
predict/baselines/minimol/run_bioavailability_ma.py --train-all
predict/baselines/minimol/run_train_cv.py
predict/baselines/minimol/run_embedding_knn.py
predict/baselines/conditioned_knn.py
predict/baselines/structure_knn/run.py
```

The Bioavailability nondirect context overlay for the conditioned cumulative-family and the ClinTox source-native support
bridge are built by `tasks/bioavailability_ma/build_nondirect_assay_context.py` and
`tasks/clintox/build_flat_assay_support_evidence.py` respectively; both only generate versioned source artifacts and do not duplicate the reasoning
runner. The full contract, current valid progress, and reproduction commands are uniformly documented in `ASSAY_LEVEL_RETRIEVAL.md`.

`plot_starling_model_comparison.py` is the sole Starling overview entry point for GPT-OSS-20B, GPT-OSS-120B, train-label baselines, and subsequent
ablations. New complete model/visibility summaries are appended via the reproducible
`--comparison-metrics`; matched method experiments continue to be appended via the reproducible `--experiment-metrics`.
Do not add separate overview/bar-chart modules or formal subplots for individual new experiments. The complete comparison summary must use the same task/split/subset/sample counts and baselines as the reference; experiment metrics must provide an anchor row aligned with existing candidate conditions per task/split, and the plotter will validate `n` and macro-F1, then hide duplicate anchors and only plot new experiment rows.

`router_oof/` is a train-only KNN-vs-direct-agent experiment isolated from the formal valid/test matrix. It fixes scaffold-or-parent
5-fold OOF, fold-specific heldout-parent filtered index, Morgan `k=3`, gpt-oss-120b
`identity_blind + parent_disjoint` direct agent, and trains Logistic/HistGBDT separately for BBB/Bioavailability/Skin;
v1–v3.1 deployable routers must not share task parameters, nor add train folds to the evaluation
subset of `starling_benchmark_matrix.py`. The only shared-parameter experiment is the frozen, non-deployable train-only transfer termination diagnosis.
The current v2 router does not take fold-dependent absolute reference size or deterministically repeated k=3 margin/entropy as input; it performs
nested train-only selection within the three general profiles `knn_compact`, `query_knn`, `query_knn_evidence`, and calibrates
`P(agent-only-correct)` and `P(KNN-only-correct)` separately, using the difference between the two as the routing score. Only when the nested OOF paired-bootstrap
promotion gate passes both the macro-F1 improvement and accuracy guardrail is it deployed; otherwise it explicitly falls back to KNN. The v1 `router/`
artifact is retained as historical diagnosis; v2 writes to `router_v2/` and `router_features_v2.*`, and must not overwrite or mix tables.
v3 is an independent output-aware post-selector: it trains only on KNN/agent disagreement rows, with target being agent-only-correct vs.
KNN-only-correct; it retains the original query/KNN/evidence features and compares 18-feature base, 49-feature
evidence, and 70-feature evidence+structured-trace profiles via nested OOF. The two disagreement directions use independent thresholds,
and strictly fall back to KNN when the promotion gate fails. v3 writes to `post_selector_features_v3.jsonl` and `post_selector_v3/`,
and must not overwrite v1/v2; permutation/dropout stability requires additional calls and cannot be fabricated from existing traces.
v3.1 reuses the same frozen feature rows, but selects family/profile/threshold separately for each disagreement direction, and uses a
fold-heldout sigmoid-calibrated ensemble. Each direction must route at least 20 train rows and have a one-sided 95% Wilson lower bound for agent-win precision
> 0.5; overall train promotion uses accuracy paired-bootstrap lower CI > 0 as the primary gate. It writes to `post_selector_v31/` and `post_selector_valid_result_v31.json`, and must not overwrite v3. The valid receipt must
separately indicate the train-only promotion and the held-out valid evidence gate; valid does not change the frozen policy.
After v3.1, only two frozen train-only termination diagnostics are allowed: `post_selector_v31_learning_curve/` fixes
direction specs for matched-size curves; `post_selector_v31_transfer/` fixes a task-balanced Logistic shared
representation, retaining task-specific calibration/threshold. The current transfer gate has failed, and the router main method line stops;
no new shared family/profile may be searched based on valid, and the formal test has not yet run.
Protocols, commands, and gates are uniformly recorded in `tools/chembl_tool/paper_experiments/ROUTER_OOF_IMPLEMENTATION_PLAN.md`.

`watch_glm_tunnel_and_matrix.py` is the resumable monitoring entry point for the long GLM matrix: check `/v1/models`, SSH tunnel, and the unique
launcher; on disconnection, stop the current process group, and after reconnection rely on `--skip-existing` to resume. Completion counts must pass the four-layer gate of task prediction,
single/final status, expected group count, and group status; do not count only final files. This entry point does not store passwords;
Duo approval remains the user's responsibility.

Data construction entry point:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m data.processing.gold_labels.build_conditioned_benchmark
```

The current split uses a unified condition-aware schema. Label provenance, source review, parent identity, and conflict records
are stored in audit artifacts in the same directory. Before formal evaluation, the train-only retrieval index must be rebuilt against the heldout detailed labels of the valid+test union;
the existing evidence index built from the full Starling source cannot be used directly for the current benchmark.

The old `data/gold_labels/legacy/processed/BBB_Martins/B3DB_cleaned/test/test_efflux.jsonl` and
`data/gold_labels/legacy/processed/{Bioavailability_Ma,ClinTox,Skin_Reaction}` are historical inputs for existing TDC experiments and no longer represent the current benchmark for the three
migrated tasks above; the old ClinTox split also does not represent the new parent-normalized reconstruction.
The current strict ClinTox split is located at `data/legacy/clintox/gold_labels/conditioned_benchmark/scaffold/`. Historical results and reproduction commands may be retained,
but must be clearly marked as historical lineage.

## Design Principles

1. Do not write BBB logic as a one-off script. BBB is the first task, but tool services, retrieval protocols, and LLM input/output formats should support more tasks later.
2. ChEMBL neighbor retrieval is an evidence prefetch / context assembly step in the pipeline, not a function tool currently exposed to the LLM, nor a current FastAPI service tool. Later models for pKa, logD, solubility, toxicity, target affinity, PK properties, etc., will be integrated under a common tool contract.
3. Long-initialization models should be resident. Slow-start models and large indexes should be loaded at service startup and called via FastAPI endpoints, avoiding repeated initialization per query.
4. Evidence retrieval only provides evidence, not a substitute for reasoning. The retrieval payload must retain assay description, activity values, endpoint semantics, similarity, and uncertainty.
5. The default production/group-level retrieval unit remains molecule-level evidence: first find similar molecules, then expand
   assay/activity evidence. There is also an isolated Starling assay-level scaling experiment: first select a cumulative assay prefix based on frozen biological relevance,
   delete direct-outcome rows for valid+test parents, then retrieve query-scaffold-disjoint molecules from the retained source records,
   and merge identical molecules across assays into one flat branch. This experiment must not
   modify the production family mapping; the protocol is in
   `tools/chembl_tool/paper_experiments/ASSAY_LEVEL_RETRIEVAL.md`.
   Conditioned cumulative-family assay experiments use `assay_compact.raw_v3`:
   at most three representative record cards per assay×molecule with complete
   raw card fields and support text. They do not apply field-level truncation
   and do not require or call a support-summary model.
   The isolated visible progressive experiment now uses
   `conditioned_assay_progressive_visible.v8`: candidate generation is global
   molecule-similarity retrieval within each cumulative record-family pool,
   without a per-assay neighbor cap. L1 selects at most 10 molecules; later
   families append bounded new-molecule and active-molecule card deltas; all
   prior cards remain visible. Assay identity remains card provenance and a
   diversity tie-break only. Family assignment is record-level: a physical
   assay may contribute cards to several levels, and its earliest level is only
   catalog ordering/coverage metadata, never a visibility or retrieval gate.
   BBB source-purity v5 restricts L1 to current accepted non-prediction voters
   plus records that replay the experimental CNS-access gold contract; predicted
   BBB outcomes and missing/generic proxies move to near-direct, while predicted
   passive-permeability or efflux readouts remain in their mechanism family.
   The v5 row ledger audits all 581,708 source records and rejects cross-family
   efflux/influx precedence violations before an index can be published.
   Its runner and artifacts must not replace the
   cumulative-family or geometric assay-prefix pipelines.
6. LLM reasoning is divided into concurrent evidence branches and a final summary: the single-molecule branch judges physicochemical property priors; the paper-facing
   group-level branch judges analog transferability by a small number of data-source-independent mechanism families; the final level summarizes all
   evidence. Fine-grained `Tier.endpoint_group` is only used for source-local normalization, retrieval audit, and legacy native runner,
   and must not be used as one parallel LLM branch per group in new tasks. Task extension specifications are in `tools/chembl_tool/tasks/AGENTS.md`.

## Current Resident Tool Services

Currently implemented common tool service entry points:

```text
tools/service/app.py
  FastAPI app. Registers tools and provides /health, /tools, /tools/batch, /tools/{tool_name}/invoke,
  /tools/invoke, /tools/{tool_name}.

tools/service/config.py
  Service configuration. Reads MolGpKa, bounded batch workers, persistent cache, and native-thread budget; mmpdb uses
  the mmpdblib installed in the current Python environment, no source path environment variable needed.

tools/service/registry.py
  ToolRegistry. Responsible for initializing tools, reusing shared instances, unified invoke/batch invoke, persistent cache, and single-flight.

tools/service/cache.py
  Versioned SQLite/WAL persistent cache, in-process LRU, and single-flight common implementation.

tools/service/runtime.py
  Limits PyTorch/OpenMP/MKL/OpenBLAS/NumExpr native threads to prevent request-level concurrency from nesting thread explosion.

tools/service/molgpka_predictor.py
  ResidentMolGpKaPredictor; acid/base weights are loaded only once per service process.

tools/service/schemas.py
  Unified schemas such as ToolRequest / ToolResponse / ToolError.

tools/service/errors.py
  Unified tool exceptions.

tools/service/tools/base.py
  BaseTool abstraction.

tools/service/tools/rdkit_properties.py
  molecule_properties v1. Computes RDKit descriptors, MolGpKa pKa/logD, AccFG top-level functional groups.

tools/service/tools/properties_compare.py
  properties_compare v1. Compares molecule_properties output of two molecules, excluding functional groups.

tools/service/tools/mmp_structure_compare.py
  mmp_structure_compare v1. Compares structure only: Morgan Tanimoto, similarity bucket, mmpdb matched-pair transformation, MCS.
```

Current relevant test entry points:

```text
tests/service/test_registry.py
tests/service/test_molgpka_predictor.py
tests/service/test_rdkit_properties.py
tests/service/test_properties_compare.py
tests/service/test_mmp_structure_compare.py
```

Single-process development startup command:

```bash
uvicorn tools.service.app:app --host 127.0.0.1 --port 8765
```

node002 formal high-throughput startup, caching, batch endpoints, thread budget, and rolling switch specifications are maintained in:

```text
tools/service/README.md
```

The formal benchmark uses 32 Uvicorn process workers, 8 bounded batch workers per process, 1 thread per native
inference, and may stage the active versioned SQLite/WAL cache under node-local `/local/tmp`; after the writer closes, it must be copied to validated storage on `/vast` and removed locally. The harness submits a fixed tool bundle for one sample
via `/tools/batch` at once; the service performs persistent cache and single-flight
deduplication by tool/input/version. This layer only consumes the unified retrieval payload and does not depend on the internal implementations of Morgan, MiniMol, coverage selector, or future retrievers.
Do not duplicate tool-prefetch/cache logic in new retrieval methods.

Currently, the service layer temporarily freezes only three general tools:

```text
molecule_properties
properties_compare
mmp_structure_compare
```

Previously planned `rdkit_properties` and `ml_pka` are no longer exposed as independent service tools; they have been merged into `molecule_properties`. `chembl_neighbors` is currently not a resident service tool, nor a DeepSeek-callable tool. ChEMBL neighbor retrieval is still prefetched and injected as group evidence context by each task's `retrieve_neighbors.py` / `run_reasoning_pipeline.py` before calling the LLM; it can be wrapped as a service tool or task endpoint later if needed.

## LLM-Visible Output Convention

Tool responses retain structured JSON for workflow, debugging, caching, and testing; the final content shown to the LLM uses only:

```text
ToolResponse.output.text
```

Do not stuff structured fields such as `raw_features`, `comparisons`, `mcs`, `transformation`, `metadata` into the LLM prompt as a whole. When needing to distinguish tools in the prompt, you can add short titles, for example:

```text
[molecule_properties]
<output.text>

[properties_compare]
<output.text>

[mmp_structure_compare]
<output.text>
```

In all tool-facing LLM text, numbers are rounded to at most two decimal places. None / missing / not applicable values should be described in natural language, e.g., `not applicable`, avoiding exposing Python/JSON internal representations directly to the LLM.

The evidence library may retain rule-derived fields such as `evidence_direction`, `evidence_strength`, `endpoint_group_reason`,
`assay_reason` internally for debugging and auditing; these fields must not be sent to the reasoning LLM. All task
group prompts must convert ChEMBL, Starling, or other source rows into `minimal_evidence.v1` via `tools/chembl_tool/common/evidence_contract.py`. This contract only describes source, molecule, group, endpoint/measurement,
evidence/context text, optional role/scope, quality/uncertainty, and provenance; it does not include label votes, threshold
policies, or deterministic overrides.

## OpenAI-compatible / GLM-5.2 Adaptation Record

Starting 2026-08-01, the paper/Starling GLM runner defaults to connecting directly to dgx008 vLLM via a local SSH tunnel:

```text
base_url: http://127.0.0.1:50000/v1
model: nvidia/GLM-5.2-NVFP4
api key env: GLM_LOCAL_API_KEY (runner auto-injects a non-sensitive placeholder when loopback vLLM has no auth)
reasoning_effort: "" (omit the API parameter; matches the historical LiteLLM runs)
```

First establish the tunnel:

```bash
ssh -fNT parcc-glm
```

The old Penn LiteLLM can still be explicitly used as a fallback:

```bash
--api-key-env GLM_API_KEY \
--base-url https://litellm.parcc.upenn.edu/v1 \
--model zai-org/GLM-5.2-FP8 \
--reasoning-effort ""
```

`zai-org/GLM-5.2-FP8` is the old LiteLLM request alias; old responses and existing traces actually report
`hosted_vllm/nvidia/GLM-5.2-NVFP4`. Do not describe the old/new paths as a comparison between FP8 and NVFP4 models.
The formal runner continues to use the historical `--disable-thinking --reasoning-effort ""`. Here, `--disable-thinking` simply does not send the DeepSeek-style `thinking` parameter; the empty `reasoning_effort` makes the client completely omit that API parameter; it does not disable GLM's own reasoning, and the provider-returned `reasoning_content` or `reasoning` is still written to the trace. The previously measured 64/128/256/512 concurrency high-throughput numbers used `reasoning_effort=none`, which is an endpoint ceiling diagnosis with reasoning disabled, not adopted as the formal default, and cannot be used to estimate the speedup ratio of the current reasoning-enabled agent pipeline.

OpenAI-compatible responses may place thinking text in `reasoning_content` or `reasoning`; the shared client accepts both.

### Current Conditioned Benchmark Formal Run Defaults

The four tasks uniformly read `data/gold_labels/<Task>/v1/scaffold/` and freeze it as:

```text
endpoint: http://127.0.0.1:50000/v1
model: nvidia/GLM-5.2-NVFP4
reasoning: --disable-thinking --reasoning-effort "" (consistent with historical GLM settings, still saves reasoning)
visibility_mode: identity_blind
neighbor_identity_policy: parent_disjoint
run_operational_first: false
endpoint_concurrency_budget: 512
```

Here, 512 is the global endpoint request budget for a single formal launcher, not allowing each layer to parallelize and multiply by 512 again.
Reasoning-enabled valid stress runs have proven 512/384 unstable for long group prompts, so the current default execution shape is `--parallelism 128` for the global prompt pool. Independent output-root matrix launchers can use the same endpoint simultaneously;
each launcher independently adheres to and records its own global budget and endpoint allocations, and states known aggregate load in status reports.
Outer fan-out across multiple splits, tasks, or conditions within the same launcher must share that launcher's slots.
512 only represents the single-launcher hard cap for this formal workflow, not a recommended concurrency.

The matrix uses only one cross-task/condition ready queue; `--parallelism` is the launcher's global outstanding prompt limit.
single/group/final branches share the pool; final is enqueued only after its dependencies succeed; frozen single dependencies across conditions are dynamically unlocked per sample.
Do not reintroduce condition lanes, whole-batch phase barriers, or split the same requested matrix into multiple launchers to bypass the matrix's budget; this does not restrict independent output-root matrices from sharing the endpoint concurrently.

Every approved multi-variant full batch must place all requested tasks, retrieval
widths, prompt versions, query-prior modes, and other condition axes into that one
matrix invocation and one rolling ready queue. Never serialize variants with a
shell loop or start a separate matrix launcher for each version or width. Prepare
conditions incrementally, enqueue each request as soon as its inputs are ready,
and immediately backfill every free endpoint slot while any runnable request from
any variant remains. A slow tail, preparation, validation, or finalization for one
variant must not gate runnable work from another. If the matrix CLI cannot express
a requested axis, extend the shared matrix first rather than approximating the run
with separate launchers.
Independent evidence-level ablations are variants under this rule: pass every
requested level to one matrix invocation, never a shell loop of scalar levels.

Multi-endpoint inference must degrade gracefully. Before launch, probe every
configured endpoint for the exact model, use all compatible endpoints that are
currently healthy, and set effective parallelism no higher than their combined
capacity. An unavailable endpoint is recorded in the execution receipt but must
not block or delay a run while at least one compatible endpoint is alive. Block
only when no compatible endpoint is available. If an endpoint fails during a
run, preserve completed responses and continue on the remaining pool; a recovered
endpoint may rejoin at the next safe launcher or resume boundary. Endpoint
availability may change execution capacity, never prompts, retrieval, condition
identity, or run names.

All prediction full-batch launchers use the mutable candidate inventory
`predict/api_client/providers/current_endpoints.json`. Update that one file when
hosts or ports change; do not encode endpoint names in run IDs. Full inference
requires an explicit `--parallelism`. The launcher records that requested budget,
selects every healthy exact-model candidate concurrently, and records the lower
effective budget when live capacity is smaller. Offline `--prepare-only` work does
not probe endpoints. Throughput is the default; a pilot requires the explicit
`--execution-mode live` choice and is never inserted into a full-batch run.

The new formal matrix performs fresh `parent_disjoint` retrieval directly from the
held-out-filtered index, no longer relying on operational artifacts,
same-parent diff, or `reuse_plan.json`. `none` must still run, but its
identity policy is marked as not applicable. New datasets first run valid for
pipeline/completeness checks, then run test after freezing settings; do not
adjust model, prompt, threshold, or label policy based on test except for
concurrency. Existing operational, deployment-visible, matched-prefetch, and
reuse artifacts are retained as historical lineage, not deleted, and not mixed
into the new v4 main results.

The current code has completed the following implementation gates; GPT-OSS-120B
v2 valid is complete. On 2026-08-10, Bioavailability full-flat/full-mechanism
and corresponding Morgan KNN, MiniMol embedding KNN, MiniMol trained head
frozen scaffold-test were first completed. The 2026-08-13 BBB DeepSeek
residual-adjudication valid gate failed; BBB method development stops and the
formal test is no longer listed as a pending item; Skin formal test and GLM v2
valid/test still must wait for independent promotion/artifact gates:

1. runner defaults to `identity_blind + parent_disjoint` and allows fresh-run for that combination;
2. parent-disjoint fresh-run does not require operational `reuse_plan.json` and
   outputs to a separate `runs_identity_blind_parent_disjoint/`;
3. default launcher uses a single 128-slot global prompt pool and prevents
   `parallelism` from exceeding the global 512 budget;
4. manifest explicitly saves endpoint, served model, reasoning, visibility,
   identity policy, effective concurrency, evaluation subset, and `operational_staging_used=false`;
5. valid/test partitions, manifest, and held-out index use the same valid+test
   union; the full matrix must still pass `n_failed_runs=0`, identity leak=0,
   parent conflict=0, and held-out overlap=0.

GLM's adherence to tool choice and long structured output may be unstable. All
tasks uniformly check required JSON fields and allowed values through
`tools/chembl_tool/common/reasoning_validation.py`; currently the default is at most 4 total attempts, and each
validation error and attempt count must be recorded in the trace. This
validation layer cannot modify valid predictions or implement task-specific
label policy. When longer final output is needed, explicitly increase
`--max-tokens`; do not use postprocess to patch benchmark labels.

## Evidence-library release versioning

Evidence-library construction code is version-first and release-local. The
canonical layout is:

```text
semantic_buckets/
  artifacts.py
  prompts/
  policies/
  releases/<task>/<version>/
  provenance/
  history/
  audits/
  tests/

data/processing/evidence_library/
  shared/v1/
  views.py
  evidence_library.py
  heldout_index.py
  compact_artifacts.py
  assay_catalog.py
  stage_artifact_store.py
  bucket_informativeness.py
  versions/
    v7/
      build_*.py
      pair_bucket_build.py
      prompts/
      tasks/<task>/
    v8/
      build_*.py
      pair_bucket_build.py
      prompts/
      tasks/<task>/

data/processing/gold_labels/
  benchmark_dataset.py
  conditioned_benchmark.py
  build_conditioned_benchmark.py
  publish_conditioned_benchmark.py
```

`shared/v1/` is immutable construction code pinned by both V7 and V8. It
contains only dependency-closed mechanics shared across releases; it does not
own task policies, active prompts, mappings, pruning rules, or release-specific
module resolution. Never change existing pinned behavior after a release uses
it. An incompatible common change creates `shared/v2/`.

Each `versions/vN/` root owns the small set of construction modules shared across
tasks in that release. Do not add a second `versions/vN/shared/` layer. Heldout
filtering, gold-label substitution, evidence catalogs, neighbor indices,
bucket-informativeness analysis, and semantic/readout sidecars are consumers of
a completed library and stay at the unversioned `evidence_library/` root.
Semantic/readout code, prompts, policies, releases, provenance, history, and audits
are colocated under the repository-top-level `semantic_buckets/` owner. Published
payloads live under `semantic_buckets/releases/<task>/<version>/` and are selected
by their manifest. Stable mechanics remain in that package; active policy and
reviewed decisions are versioned separately and may diverge by release.

Start a new release by copying the complete preceding release directory, then
modify the copy. Never make a new release inherit implementation or assets from
another release at runtime. Before publication, remove superseded prompts,
one-off review drivers, and inactive code from the copied release; preserve only
the selected construction path and non-reconstructable scientific decisions.

Prompt, mapping, and rule versions are component versions, not aliases for the
library release number. Keep only the selected component generation in an active
release. Compact lineage needed for a retired generation belongs under
`data/legacy/evidence_library_construction/`; agent scratch and review-process intermediates remain outside the
repository.

Published version directories are frozen. Do not retrofit a behavior change into
an older version; create the next version instead. `data/evidence_libraries/<task>/<version>/` contains the
corresponding built data, and each published task's `CURRENT` file selects
the active release. Raw data and gold labels remain external inputs; scientific
construction mappings and reviewed inputs belong to the release that consumes
them.

Construction code must import canonical `shared/vN` or `versions/vN` packages
directly. Do not create compatibility symlinks, forwarding modules, dynamic
fallbacks, or construction modules under `tools/chembl_tool/`. The real task-specific
reasoning and retrieval workflows under `tools/chembl_tool/tasks/` remain independent of
construction-code ownership.

## ChEMBL task workflow directory

Specific tasks are placed in:

```text
tools/chembl_tool/tasks/<task_name>/
```

Each task directory maintains only task-specific rules, scoring, endpoint
assignment, default paths, output filenames, and reasoning prompt / final
schema. Cross-task shared workflows are not placed under the `tasks/`
directory, but under:```text
tools/chembl_tool/common/task_workflows/
  screen_assays.py
  rescore_outputs.py
  summarize_outputs.py
  assay_report.py
  evidence_library.py
  distance_assay_manifest.py
  retrieve_neighbors.py
  chembl_exact_context.py
  reasoning_batch.py
  global_prompt_pool.py
  reasoning_stage_runtime.py

tools/chembl_tool/common/evidence_contract.py
tools/chembl_tool/common/identity_blind.py
tools/chembl_tool/common/json_utils.py
tools/chembl_tool/common/openai_reasoning_client.py
tools/chembl_tool/common/reasoning_calls.py
tools/chembl_tool/common/reasoning_payload.py
tools/chembl_tool/common/reasoning_validation.py
tools/chembl_tool/common/final_decision_prior.py
tools/chembl_tool/common/prompt_profile.py
tools/chembl_tool/common/molecule_identity.py
tools/chembl_tool/common/retrieval_policy.py
tools/chembl_tool/common/neighbor_selection.py
tools/chembl_tool/common/coverage_reasoning.py
tools/chembl_tool/common/retrieval_ablation.py
tools/chembl_tool/common/retrieval_replay.py
tools/chembl_tool/common/experiment_retrieval.py
tools/chembl_tool/common/evidence_distance.py
tools/chembl_tool/common/distance_index.py
tools/chembl_tool/common/distance_retrieval.py
tools/chembl_tool/common/scalar_knn.py
data/processing/evidence_library/evidence_library.py
data/processing/evidence_library/assay_catalog.py
data/processing/gold_labels/benchmark_dataset.py
data/processing/gold_labels/build_conditioned_benchmark.py
data/processing/evidence_library/heldout_index.py
tools/chembl_tool/common/assay_retrieval.py
```

Responsibilities of these common workflows:

```text
screen_assays.py
  Scans ChEMBL assays, calls task-specific scoring.scored_row, exports assay
  candidates, activity evidence, and report.

rescore_outputs.py
  Re-scores existing candidate CSVs, suitable for stricter rules, reordering
  tiers, or adjusting thresholds; if rules are relaxed, re-run screen_assays.py.
  Supports `--only-filter-activities` for scenarios where candidates are already determined
  and only need to re-filter activity evidence based on existing candidates.

summarize_outputs.py / assay_report.py
  Generate health checks and Markdown reports.

evidence_library.py
  Builds molecule-level evidence rows, RDKit fingerprints, and neighbor index
  from assay candidates + activity evidence. Task only configures input paths,
  output filenames, index version, and assign_endpoint_group. Supports
  `--workers` parallel standardization of molecules / index building; long
  tasks print elapsed, rate, and ETA progress.

distance_assay_manifest.py
  E12's generic ChEMBL assay scanning and frozen manifest workflow. Task-local
  classifier only decides family, scope, quality, and mapping reason; common
  implementation handles source manifest, inclusion/exclusion audit, activity
  export, and graph/config provenance.

retrieve_neighbors.py
  Source-local / legacy native retrieval: performs analog retrieval for
  fine-grained Tier.endpoint_group, including molecule identity policy, Tanimoto
  ranking, similarity threshold, similarity bucket, and JSONL batch retrieval
  CLI. Paper-facing direct/flat/mechanism views are assembled by
  experiment_retrieval.py on top, organized by mechanism family.

chembl_exact_context.py
  Optional exact-query ChEMBL context and shared-assay enrichment. Not enabled
  by default in benchmarks to avoid prospective evaluation data leakage.

reasoning_batch.py
  Multi-molecule batch parameters, manifest, logs, result collection, and
  predictions/metrics/report common implementation. Actual prompt scheduling is
  uniformly delegated to the global prompt pool. Supports `--groups`
  pass-through to task pipeline for targeted group smoke tests; supports
  `--final-only-source-batch` reuse of existing single/group artifacts, and uses `--final-only-groups`
  to strictly prune visible groups before re-summarizing final (cannot use
  `--groups` instead of this filter); metrics include positive-class
  precision/recall/F1, confusion matrix, and prediction distribution.

global_prompt_pool.py / reasoning_stage_runtime.py
  The former provides a unique ready queue across tasks/conditions and a global
  prompt concurrency limit; the latter provides single/group/final stage
  checkpoints, dependency unlocking, atomic artifact writing, and breakpoint
  recovery. Successful prerequisite rewrites invalidate old final/trace; final
  executes only after single and exact expected group set all succeed. All five
  existing batch wrappers must accept the common `--prepare-only` seed command;
  DILI, not yet migrated to the shared retrieval contract, only allows
  native/operational/standard default combinations; the common parser rejects
  disguised parent-disjoint or coverage ablation. Pool internal sample keys must
  use absolute batch directory plus query index; `batch_id` is only unique
  within a single batch root and cannot serve as a global key across folds/roots.
  Single dependency source keys must use the same absolute directory contract.

evidence_contract.py
  The unique schema/normalizer for `minimal_evidence.v1`. Old ChEMBL-like rows can be
  dynamically converted at prompt time; new source builders should call
  `attach_minimal_evidence()` at library build time. This module only describes evidence, does
  not predict labels.

identity_blind.py
  Uniformly implements identity redaction, harness-prefetched tool evidence, and
  matched-prefetch tool replay. Paper task runners can only select
  visibility/tool-execution contract through this module; cannot duplicate
  redaction or replay logic within tasks.

reasoning_calls.py / json_utils.py
  Shared single/group branch calls, frozen single analysis reuse, group payload
  transport bound, JSON extraction, and atomic publication tools for
  JSON/JSONL/trace in the same directory. Transport bound can only
  deterministically sample oversized evidence rows; cannot change evidence
  source, label policy, or inference settings.

reasoning_payload.py
  Shared LLM query identity surface for five task pipelines, exact-match/
  shared-assay cleaning, single-line JSONL reading, explicit env-file parsing,
  and single/group/final trace serialization. Tasks only bind their own
  prediction fields; must not copy and drift these frozen field contracts per
  task.

molecule_identity.py / retrieval_policy.py
  Data-source and task-agnostic whole-record, RDKit fragment/molecular-parent,
  and mixture-component standardization and neighbor exclusion policy.
  Operational retains same-parent evidence; parent_disjoint additionally excludes
  and backfills within existing similarity thresholds. Here parent is not
  pharmacological active moiety, nor does it infer prodrug/metabolite
  relationships.

neighbor_selection.py / coverage_reasoning.py
  Two orthogonal pluggable contracts: the former only selects neighbors from
  candidates that have passed similarity, identity, and evidence gates; the
  latter only controls LLM-visible analog-set context. `standard` context is
  strictly a no-op; `coverage_aware` provides anonymous statistics of query Morgan
  feature/atom-environment coverage, per-neighbor marginal coverage, and region
  size, without changing retrieval.json, neighbor set, task JSON schema, or
  tool-prefetch/cache. Coverage context does not expose SMILES, fingerprint bit
  IDs, element labels, or molecular identity, and currently only supports Morgan
  retrieval features. Visible-only `coverage_mmp_ledger` is an independent opt-in
  profile: it reuses the resident `mmp_structure_compare` per-neighbor MCS/MMP text, then
  organizes rank-by-rank complementarity, redundancy, and uncovered region
  ledger with Morgan marginal feature statistics; does not reimplement MCS/MMP,
  does not modify raw retrieval, neighbor set, task JSON schema, or default
  `standard` / `coverage_aware` paths. Morgan feature coverage cannot be
  interpreted as atom coverage; without matched-pair transformation, specific
  fragment correspondences must be marked as unresolved.

retrieval_ablation.py
  Computes LLM-visible retrieval/group input hashes, supports deterministic
  reuse of full samples and independent mechanism branches, and records
  provenance. This module cannot change evidence, thresholds, or predictions.

retrieval_replay.py
  Performs matched-prefetch replay based on frozen retrieval/tool artifacts,
  checks sample coverage and input consistency; only for visibility/tool-
  execution control, not a substitute for agentic deployment-visible main
  experiments.

experiment_retrieval.py
  Maps source-local endpoint groups to task-declared direct/mechanism families;
  ensures full_flat and full_mechanism use the same evidence union, and only
  changes reasoning organization.evidence_distance.py / distance_index.py / distance_retrieval.py
  E12 independent code line, implements D-root/C-family tree contract: each C family has exactly one aggregate H1 child, each H1 has at most
  one optional H2 child. Each C/H1/H2 tree node independently takes at most 3 neighbors and shares the same similarity threshold;
  multiple target/measurement families within a child share the node budget. The public builder/retrieval/audit can materialize
  D, D+C, D+C+H1, D+C+H1+H2 flat/mechanism views, and verify base parity, nestedness, branch stability,
  and node budget; this remains an engineering line isolated from the old paper matrix, not yet registered as a paper LLM condition.
  In addition to graph hop validation, a same-molecule causal continuity audit must be performed: if the assay molecule only changes the system
  state, while the downstream endpoint actually acts on another unobserved substrate, mark it as `requires_query_role` or
  `context_only`, and it must not enter the main H1/H2. All future task releases must fully declare `FamilySelfRelevanceAudit` and pass
  `validate_self_relevance_audit(..., require_publishable=True)`; prompts cannot replace missing substrate/target roles.
  These modules must not be registered into the old `EXPERIMENT_MODES`, nor change the old paper matrix or old index.

scalar_knn.py
  Shared scalar KNN baseline implementation; currently used for Bioavailability numeric direct-F control, must be reported separately from LLM agent conditions.

openai_reasoning_client.py
  Shared OpenAI-compatible JSON completion, bounded tool-call loop, resident tool service calls, and trace
  serialization for all tasks. Provider/model/base URL are configured by runtime parameters; task files do not copy client runtime.

reasoning_validation.py
  Provides common required fields, allowed values, and required tool result validation for single/group/final branches; single/group automatically extract the top-level `required_json_schema` enum from the sent
  `a | b | c` and validate it, with illegal near-synonym values triggering same-setting retry. Currently defaults to at most
  4 total attempts.
  Must not rewrite predictions when the response is valid, nor delete evidence or change inference settings through retry.

final_decision_prior.py
  Provides an explicit opt-in final-stage decision profile. Default `standard` is strictly no-op;
  `train_ratio_tiebreak_v1` only allows BBB/Bio final-only valid diagnostics to use frozen train majority on true evidence ties,
  and requires `evidence_state`, boolean `prior_used`, and predictions to pass cross-field validation. Failed BBB
  `direct_anchored_residual_v1` / `direct_override_recheck_v1` only retain frozen artifacts and no-go conclusions; dedicated implementations have been removed,
  and formal tests must not be started. No profile may become a batch quota.

prompt_profile.py
  Only responsible for manifest provenance, historical default mapping, and branch-reuse consistency gate for task prompt profiles. Specific task
  instructions/schema continue to reside in each task's `prompt_profiles.py`; task semantics must not be stuffed into shared modules, nor should profile parsing logic be duplicated in the runner.

common/starling/evidence_library.py
  Profile-driven parquet ingestion. Profiles only declare SMILES, endpoint, value, unit, context, scope, role,
  and group mappings; the public implementation handles canonicalization, missing SMILES statistics, molecule-level aggregation, representative
  examples, provenance, and neighbor-index compatible evidence rows.

common/starling/benchmark_dataset.py
  Starling direct gold-label construction public engine: source-row decisions, RDKit fragment-parent aggregation, 70% record-majority,
  random/scaffold train/valid/test split, audit artifacts, and summary. Task-specific thresholds,
  population/scope/unit/free-text rules can only be provided by task adapters.

common/starling/build_benchmark_datasets.py
  Historical TDC-compatible three-task CLI; reads frozen source revision/local parquet, generates
  `data/gold_labels/legacy/processed_starling/<Task>/{random,scaffold}/` and task/root summaries. BBB new mainline does not use this entry.

common/starling/build_record_supported_benchmark.py
  Constructs scaffold-only quality splits from frozen binary parents; default `record_supported_v2`, while providing configurable lineage/seed shared allocation to the BBB new builder.
  Lexicographic MILP first minimizes
  held-out singleton and valid/test imbalance, then optimizes label balance, and finally maximizes first-version valid reuse. Output contains only
  changed split/audit; root-level source rejection/conflict provenance continues to read the first-version directory to avoid duplicate data.

common/starling/publish_conditioned_benchmark.py / conditioned_benchmark.py
  The only publication/path entry for the current four tasks; source-specific builders first write to `data/.build/conditioned_benchmark_sources/`,
  then the publisher unifies schema, paths, and hash receipts. Versioned BBB build/migration scripts serve only as source QA provenance,
  and must not become runner inputs.

common/starling/heldout_index.py
  Deletes valid+test parents from full-source Starling evidence rows according to `rdkit_fragment_parent.v1`, rebuilds
  train/evaluation isolated retrieval index, and writes source/exclusion SHA-256, exclusion counts, and zero-overlap audit.
  Recomputes and validates held-out parent keys during build; source evidence rows with unparseable parents are conservatively excluded.
```

Typical task wrapper files:

```text
screen_assays.py
rescore_outputs.py
summarize_outputs.py
report.py
build_evidence_library.py
retrieve_neighbors.py
chembl_exact_context.py
run_reasoning_pipeline.py
run_reasoning_batch.py
```

These wrappers should remain thin, only configuring task-specific parameters; do not duplicate public implementations across multiple tasks.

Common command templates:

```bash
# Full scan of ChEMBL assays
python -m tools.chembl_tool.tasks.<task_name>.screen_assays \
  --chembl-sqlite tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db \
  --out-dir outputs/chembl_tool/tasks/<task_name>/assay_screening/raw \
  --min-score 40 \
  --progress-every 10000 \
  --export-activities

# Rescore existing candidates
python -m tools.chembl_tool.tasks.<task_name>.rescore_outputs \
  --in-dir outputs/chembl_tool/tasks/<task_name>/assay_screening/raw \
  --out-dir outputs/chembl_tool/tasks/<task_name>/assay_screening/<version> \
  --min-score 40 \
  --filter-activities

# When candidates are already determined, only re-filter activity evidence
python -m tools.chembl_tool.tasks.<task_name>.rescore_outputs \
  --in-dir outputs/chembl_tool/tasks/<task_name>/assay_screening/raw \
  --out-dir outputs/chembl_tool/tasks/<task_name>/assay_screening/<version> \
  --only-filter-activities

# Generate health check
python -m tools.chembl_tool.tasks.<task_name>.summarize_outputs \
  --out-dir outputs/chembl_tool/tasks/<task_name>/assay_screening/<version>

# Build evidence library and neighbor index
python -m tools.chembl_tool.tasks.<task_name>.build_evidence_library \
  --workers 128 \
  --progress-every 50000

# Retrieve analog neighbors
python -m tools.chembl_tool.tasks.<task_name>.retrieve_neighbors \
  --query-smiles '<SMILES>' \
  --top-k-per-group 3 \
  --min-similarity 0.3

# Batch reasoning
python -m tools.chembl_tool.tasks.<task_name>.run_reasoning_batch \
  --input-jsonl <input.jsonl> \
  --parallelism 1 \
  --batch-id <batch_id>
```

When only a few endpoint groups need to be run for debug/smoke, add to the pipeline or batch:

```bash
--groups "Tier 3.some_endpoint_group" "Tier 4.another_endpoint_group"
```

Resume from breakpoints uniformly using:

```bash
python -m tools.chembl_tool.tasks.<task_name>.run_reasoning_batch \
  --input-jsonl <input.jsonl> \
  --batch-id <batch_id> \
  --skip-existing
```

`--skip-existing` will skip molecules that already have
`reasoning/batches/<batch_id>/runs/<batch_id>_idxNNNNN/final_reasoning_output.json`
; existing stdout/stderr logs are not overwritten, and partial runs without final output will be re-run.

Output directories are unified as:

```text
outputs/chembl_tool/tasks/<task_name>/
  assay_screening/
  evidence_library/
  reasoning/
    single_runs/
    batches/
```

View final paper trace:

```bash
bash tools/trace_viewer/start_viewer.sh 8776
```

Viewer by default registers Starling random, Starling scaffold, and historical TDC test. New v4 datasets by default scan
`runs_identity_blind_parent_disjoint/`; historical datasets continue to scan identity-blind, matched-prefetch,
deployment-visible, and deployment-visible parent-disjoint four legacy paper roots. Pages display per-sample
single-molecule, mechanism-family/flat/direct, and final stages, and recursively display
common JSON, tool calls, retrieval evidence, and provenance. Old task-specific reasoning outputs are no longer supported;
for other datasets, explicitly append trace root after the port.

## Paper experiment split and visualization entry

Detailed operational specifications for the frozen paper matrix are in `tools/chembl_tool/paper_experiments/AGENTS.md`. By default without
`--split`, use test and write to `outputs/paper/molecular_evidence_agent/`; validation diagnostic reruns uniformly add
`--split valid`, with artifacts isolated to `outputs/paper/molecular_evidence_agent_valid/`. Reusable entries include:

The existing `test` / `valid` and 2026-07-23 frozen results come from the old TDC lineage and should be retained as historical results;
`--split test|valid` currently cannot be interpreted as Starling's `random|scaffold`. New Starling formal experiments must explicitly choose
`data/gold_labels/legacy/processed_starling/<Task>/random/test.jsonl` or `scaffold/test.jsonl`, use the corresponding train-only
retrieval index, and write to a new output root/batch ID isolated from both TDC and the other Starling split. Before completing input wiring,
test-parent exclusion, and zero-overlap audit, existing paper metrics must not be renamed as Starling results.

Bioavailability `record_supported_v2`, Skin `record_supported_v2`, and BBB
`experimental_meaningful_cns_access_v3` subsequent paper-facing
structural-analog main results default to
`identity_blind + parent_disjoint` fresh-run; no longer run operational first, nor require operational diff/reuse plans.
The old lineage's operational -> parent-disjoint process and same-parent exposure statistics are retained only as historical sensitivity
artifacts. If operational comparison is needed in the future, it must be an explicit opt-in ablation using a separate root, and must not become a main matrix dependency.

The Starling main entry for new datasets remains
`tools.chembl_tool.paper_experiments.starling_benchmark_matrix`, and the formal default must resolve to
`--visibility-mode identity_blind --neighbor-identity-policy parent_disjoint`, outputting to each lineage root's
`runs_identity_blind_parent_disjoint/`. Each split must cover the full condition/sample set and pass failure,
query-SMILES leak, visibility-contract, parent identity, and held-out overlap audits. Old `identity_blind + operational`
results remain historical supplemental controls and must not be presented as current main results.

On 2026-08-04, the first version of `record_agreement70_split811_v1` scaffold-valid completed GLM, GPT-OSS-20B/120B three sets of
22-condition blind matrix; GLM achieved
6887/6887 strict success, and each of the two GPTs had one unfixable context-limit sample counted as error per the predetermined policy. GPT's two sets of
deployment-visible+parent-disjoint supplementary matrices are also complete; GLM visible same-contract matrix also reached 6887/6887 strict success,
and has been added to the canonical blind+visible overall figure.
Random-valid GLM still has two Bioavailability ChEMBL full sample-conditions failing strict gates. Full paths, metrics,
failure policy, and coverage context/MMP-ledger results are in
`tools/chembl_tool/paper_experiments/STARLING_BENCHMARK_RESULTS.md`.

```bash
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent --split valid ...
python -m tools.chembl_tool.paper_experiments.summarize_results --split valid
python -m tools.chembl_tool.paper_experiments.audit_prefetch_contract --split valid
python -m tools.chembl_tool.paper_experiments.parent_disjoint_ablation --split valid --materialize
python -m tools.chembl_tool.paper_experiments.plot_retrieval_claims_overview \
  --analysis-dir outputs/paper/molecular_evidence_agent_valid/analysis \
  --output outputs/paper/molecular_evidence_agent_valid/analysis/figures/retrieval_claims_overview.svg \
  --png-output outputs/paper/molecular_evidence_agent_valid/analysis/figures/retrieval_claims_overview_highres.png \
  --data-split valid

python -m tools.chembl_tool.paper_experiments.analyze_coverage_performance \
  --split valid \
  --analysis-dir outputs/paper/molecular_evidence_agent_valid/analysis

python -m tools.chembl_tool.paper_experiments.plot_coverage_performance \
  --analysis-dir outputs/paper/molecular_evidence_agent_valid/analysis \
  --output outputs/paper/molecular_evidence_agent_valid/analysis/figures/coverage_performance_relationship.svg \
  --data-split valid
```

`summarize_parent_disjoint_results.py` currently switches splits via explicit `--operational-root`, `--parent-disjoint-root`, and
`--output-dir`; detailed valid commands are in the paper-experiments directory documentation.

The 2026-07-23 valid matrix has been extended: three visibility/tool-execution regimes each with 26 conditions, 2,713
sample-conditions and 0 failures; prefetch audit is 2,713/2,713; parent-disjoint is 22 conditions,
2,275 sample-conditions and 0 failures. Test identity-blind and deployment-visible each have 26 conditions,
matched-prefetch remains the original 21 conditions. Measured results are in `tools/chembl_tool/paper_experiments/RESULTS.md`.

The old TDC performance overview still uses `plot_retrieval_claims_overview.py` as the only template; the current v4 model, visibility,
baseline, and matched ablation overall figures are uniformly generated by `plot_starling_model_comparison.py`. Coverage and performance gain old
diagnostic figures use `plot_coverage_performance.py`. All entries reuse `paper_figure_style.py`, and each formal figures directory only
retains canonical SVG and one high-resolution PNG, without preview, QA, or one-off overviews.

## MiniMol baseline (historical results and current v4 scaffold-valid boundary)

MiniMol baseline code is in:

```text
baselines/minimol/
  run_bioavailability_ma.py
  run_embedding_knn.py
  run_direct_gpu_sweep.sh
  run_hparam_sweep.py
```

`run_bioavailability_ma.py` name is retained from the first Bioavailability_Ma experiment, but it is actually a generic JSONL
binary classifier runner. Input split convention:

```text
train.jsonl / valid.jsonl / test.jsonl
fields:
  drug: SMILES
  Y: 0/1 label
```

The old commands, sweeps, and metrics listed below use historical TDC or strict-conflict Starling splits. The current baseline uniformly reads
`data/gold_labels/<Task>/v1/scaffold/`, and runs MiniMol head, Morgan KNN,
and MiniMol embedding KNN per condition-aware cohort. Historical molecule-only baselines must not be mixed with current cohorts. This diagnostic does not read or tune on
test; all settings must still be frozen before formal test.

The previous strict-conflict formal Starling baseline uses `--train-all`:
does not read `valid.jsonl`, each
ensemble member trains fixed epochs on all `train.jsonl`, and evaluates test with frozen `threshold=0.5`.
This mode must not perform test-selected early stopping or threshold tuning; validation metrics in output are null.
These historical random/scaffold results and output roots are in
`tools/chembl_tool/paper_experiments/STARLING_BENCHMARK_RESULTS.md`; the sweep values listed below are all still
old TDC lineage.

MiniMol embedding retrieval-only control reuses the above formal baseline's saved, row-aligned
`embeddings/{train,test}.pt`, L2 normalizes, then takes top-3 molecules from the same split train by cosine similarity.
Like Morgan KNN, it uses unweighted majority vote, with score as the proportion of positive-class neighbors; no new head is trained, and no GPU reoccupation is needed.
Entry and output are respectively:

```text
baselines/minimol/run_embedding_knn.py
outputs/baselines/minimol_embedding_knn_starling/<Task>/<random|scaffold>/
```

MiniMol feature agent ablation differs from the above label-vote KNN: it only replaces the agent pipeline's neighbor
ranking from Morgan/Tanimoto with L2-normalized MiniMol/cosine, keeping evidence source, top-k, GLM,
and inference settings unchanged. The current v4 full random/scaffold fresh parent-disjoint -> paired summary
-> figure recoverable entry is:

```bash
python -m tools.chembl_tool.paper_experiments.run_minimol_retrieval_agent_experiment
```

MiniMol agent retrieval is another independent experimental line: it does not do KNN vote from gold train labels, but replaces the Morgan/Tanimoto neighbor ranking of the ChEMBL /
Starling evidence index with MiniMol v1 embedding cosine,
then feeds the retrieved evidence into a frozen GLM agent pipeline. It keeps source/group/top-k/numeric min-similarity,
prompt/tool/model and fresh parent-disjoint contract unchanged, and writes to an independent output root. Official entry, feature
store contract, checkpoint/cache parity, test-parent audit and full commands see
`tools/chembl_tool/paper_experiments/AGENTS.md`; do not mix this agent ablation with
the label-vote baseline of `baselines/minimol/run_embedding_knn.py` as one item.

Runtime environment and implementation notes:

```text
conda env: intern
MiniMol source reference: /data1/tianang/Projects/minimol

For actual runs, prefer the minimol package already installed in the intern environment. The
minimol/ckpts/minimol_v1/state_dict.pth in the source directory is currently a Git LFS pointer, not a directly torch.load-able weight.

Share `baselines/minimol/embedding_runtime.py` for two compatibility patches:
  1. Graphium CPU/fake-graph featurization defaults to float16, which triggers scipy.sparse dtype errors;
     the runner forces float32 adjacency/pyg graph in-process.
  2. MiniMol checkpoint predates PyTorch 2.6 weights_only=True default; when initializing MiniMol, temporarily
     call torch.load with weights_only=False.

MiniMol featurization sets featurization_n_jobs=1 to avoid joblib subprocesses losing the above in-process patches.
```

Evaluation criteria:

```text
MiniMol embeddings + leaderboard-style TaskHead.
Each ensemble member is trained only on train, using valid BCE loss to select the best epoch.
Default ensemble_size=5, epochs=25, threshold=0.5.
accuracy / macro-F1 use threshold=0.5; AUROC uses probability score.
valid-tuned threshold metrics are also written to metrics.json, but the main report uses fixed 0.5.
```

Single-task baseline command template:

```bash
/data1/tianang/anaconda3/condabin/conda run -n intern python -m baselines.minimol.run_bioavailability_ma \
  --data-dir data/gold_labels/legacy/processed/<TaskName> \
  --output-dir outputs/baselines/minimol/<task_name>
```

BBB_Martins uses MiniMol original `SWEEP_RESULTS['bbb_martins']` hyperparameters:

```bash
/data1/tianang/anaconda3/condabin/conda run -n intern python -m baselines.minimol.run_bioavailability_ma \
  --data-dir data/gold_labels/legacy/processed/BBB_Martins \
  --output-dir outputs/baselines/minimol/bbb_martins \
  --hidden-dim 2048 \
  --depth 3 \
  --lr 0.0001
```

When GPU status or a specific GPU is needed, it must be run outside the sandbox; NVML / CUDA may not be visible inside the sandbox,
causing the runner to fall back to CPU. A reliable approach is to explicitly bind the GPU via shell:

```bash
env CUDA_VISIBLE_DEVICES=4 /data1/tianang/anaconda3/condabin/conda run -n intern python -m baselines.minimol.run_bioavailability_ma \
  --data-dir data/gold_labels/legacy/processed/ClinTox \
  --output-dir outputs/baselines/minimol/clintox
```

ClinTox / Skin_Reaction are not in the MiniMol original `SWEEP_RESULTS` table. The current hyperparameter search for these two tasks uses
the 11 unique head configurations that appear in the MiniMol ADMET sweep table:

```text
(hidden_dim, depth, lr)
(512, 3, 0.0001)
(512, 3, 0.0003)
(512, 4, 0.0003)
(512, 4, 0.0005)
(1024, 3, 0.0003)
(1024, 3, 0.0005)
(1024, 4, 0.0001)
(1024, 4, 0.0005)
(2048, 3, 0.0001)
(2048, 4, 0.0003)
(2048, 4, 0.0005)
```

The reliable entry point for GPU sweep is a direct shell script; it reuses existing embedding cache and uses `env CUDA_VISIBLE_DEVICES=<gpu>`
to launch each training job directly:

```bash
baselines/minimol/run_direct_gpu_sweep.sh \
  clintox \
  data/gold_labels/legacy/processed/ClinTox \
  outputs/baselines/minimol/clintox/embeddings \
  outputs/baselines/minimol_sweeps_gpu \
  4,5,6,7

baselines/minimol/run_direct_gpu_sweep.sh \
  skin_reaction \
  data/gold_labels/legacy/processed/Skin_Reaction \
  outputs/baselines/minimol/skin_reaction/embeddings \
  outputs/baselines/minimol_sweeps_gpu \
  4,5,6,7
```

`run_hparam_sweep.py` is a stdlib Python launcher, but in the current environment, nested `conda run` has shown CUDA
invisibility / CPU fallback; when GPU sweep is needed, prefer `run_direct_gpu_sweep.sh`.

Current MiniMol baseline / sweep results:

```text
Bioavailability_Ma, fixed h=512 d=3 lr=0.0003:
  output: outputs/baselines/minimol/bioavailability_ma/
  test macro-F1 0.5674, accuracy 0.7813, AUROC 0.6721
  MiniMol paper/README reports Bioavailability Ma AUROC 0.689 +/- 0.020, so this is close.

BBB_Martins, MiniMol sweep config h=2048 d=3 lr=0.0001:
  output: outputs/baselines/minimol/bbb_martins/
  test macro-F1 0.8186, accuracy 0.8878, AUROC 0.9322

ClinTox, default h=512 d=3 lr=0.0003:
  output: outputs/baselines/minimol/clintox/
  test macro-F1 0.5651, accuracy 0.9301, AUROC 0.6770

ClinTox, 11-config GPU sweep selected by valid AUROC:
  summary: outputs/baselines/minimol_sweeps_gpu/clintox/sweep_summary.json
  selected h=512 d=3 lr=0.0001
  valid AUROC 0.6969
  test macro-F1 0.5745, accuracy 0.9371, AUROC 0.6347
  Note: default h=512 d=3 lr=0.0003 has higher observed test AUROC 0.6770; do not use test to select config.

Skin_Reaction, default h=512 d=3 lr=0.0003:
  output: outputs/baselines/minimol/skin_reaction/
  test macro-F1 0.4864, accuracy 0.5854, AUROC 0.5872

Skin_Reaction, 11-config GPU sweep selected by valid AUROC:
  summary: outputs/baselines/minimol_sweeps_gpu/skin_reaction/sweep_summary.json
  selected h=1024 d=4 lr=0.0001
  valid AUROC 0.7475
  test macro-F1 0.5795, accuracy 0.6098, AUROC 0.6055
```

## ChEMBL assay activity transfer benchmark

This independent benchmark is used to study: within the same ChEMBL assay endpoint, based only on the structural similarity of two molecules,
can we determine whether activity can be transferred from neighbor to query. The first version does not call LLM, only establishes a
Tanimoto threshold baseline, as the minimum control for subsequent DeepSeek / other LLM assay-transfer reasoning.

Code entry:

```text
tools/chembl_tool/activity_transfer_benchmark/
  __init__.py
  analyze_mcs_results.py
  benchmark_mcs_runtime.py
  build_llm_eval_set.py
  build_task_llm_eval_set.py
  plot_llm_run_comparison.py
  run_benchmark.py
  run_llm_benchmark.py
  run_task_assay_benchmark.py
```

Function of `run_benchmark.py`:

```text
1. Read ChEMBL activities from tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db.
2. The continuous-value main analysis only uses pchembl_value, filters standard_relation='=', standard_flag=1,
   and by default excludes records with non-empty data_validity_comment and potential_duplicate=1.
3. Aggregate duplicate pChEMBL means for the same molecule within the same assay_id + standard_type.
4. Read Morgan fingerprints from tools/chembl_tool/chembl_data/chembl_36_fps/chembl_36.fps.gz.
5. Sample molecule pairs within each assay endpoint, compute Tanimoto and |delta pChEMBL|.
6. Label rule: |delta pChEMBL| <= 0.5 is similar; >= 1.0 is different; in between is ambiguous.
7. Scan Tanimoto threshold, output accuracy, macro-F1, balanced accuracy, precision/recall.
8. Compute assay-specific enrichment: within each assay endpoint, first compute random pair background rate,
   then compare similar-rate lift, fold lift, and median-delta reduction across similarity buckets.
9. Optional dynamic range filter: filter low-information assay endpoints by molecule-level pChEMBL range and IQR.
10. Auxiliary analysis conservatively handles binary activity_comment, mapping only clear active / inactive class comments.
11. Generate TSV/GZ data, metrics, SVG charts, and Chinese report.
```

Function of `benchmark_mcs_runtime.py`:

```text
Sample pairs from continuous_pairs.tsv.gz by similarity bucket, use RDKit FindMCS to compute
MCS atom coverage, and estimate the MCS computation time for the full set of pairs under multiprocessing. The worker will
set OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS / RDKIT_NUM_THREADS etc. to 1,
to avoid contention between RDKit or underlying library internal threads and outer process parallelism. Long tasks will output progress to stderr:
completed, rate, elapsed, ETA, and timeout count.
Full MCS should use `--full-scan` for streaming reading and writing results, to avoid keeping all pairs, tasks, and results in memory.
If RDKit FindMCS does not return on the last few pairs, the process may hang on tail pending futures;
in that case, first terminate the stuck process, keep `.tmp`, then use `--finalize-existing` to generate summary/report from existing results
and `missing_result_indices.tsv`.
```

Function of `analyze_mcs_results.py`:

```text
Read full or partial MCS TSV, exclude ambiguous labels, scan mean MCS coverage threshold,
and rescan Tanimoto threshold on the same set of observed pairs, output threshold metrics,
MCS coverage bucket summary, Tanimoto x MCS heatmap, SVG charts, and Chinese report.
```

Function of `build_llm_eval_set.py`:

```text
Extract a small-scale LLM evaluation set from dynamic_v1 continuous_pairs.tsv.gz. By default, read existing MCS full-scan
partial results, keep only non-ambiguous pairs with observed MCS, and stratify sample by label x Tanimoto bucket.
Default output 3,000 pairs, similar/different each 1,500, each similarity bucket 500.
Output JSONL/TSV, summary.json, and Chinese report, for reuse by LLM benchmark.
```

Function of `run_llm_benchmark.py`:

```text
Run assay activity transfer LLM benchmark using OpenAI-compatible endpoint. Default model is local vLLM
hosted gpt-oss-120b, can also run DeepSeek/OpenAI-compatible hosted endpoint; default input is
dynamic_v1_llm_3k/eval_pairs.jsonl. The prompt hides query pChEMBL, only exposes reference molecule's
pChEMBL, assay context, Tanimoto, bucket, and MCS coverage. Optionally call mmp_structure_compare / properties_compare in the current tool server; output per-sample run JSON, predictions.jsonl,
metrics.json, Chinese report, model-vs-baseline SVG figure, and trace_messages.jsonl readable by trace_viewer.
Currently also supports HF prompt/completion/metadata format: completion A/B maps to similar/different,
original metadata retained in input_record.hf_metadata, and outputs grouped metrics by similarity_bucket, assay_type.
Default max-tool-rounds=3, breakpoint resume uses --skip-existing.
```

Function of `plot_llm_run_comparison.py`:

```text
Summarize two LLM runs and full-valid baseline, output overall, similarity_bucket, assay_type three-level
macro-F1 comparison figures, TSV, and Markdown report. Comparison artifacts go under
outputs/chembl_tool/activity_transfer_benchmark/comparisons/, do not put them in llm_runs/.
```

Current MCS runtime test results:

```text
outputs/chembl_tool/activity_transfer_benchmark/mcs_runtime/

dynamic_v1 continuous pairs: 1,989,152
timeout=1s, workers=128, chunksize=1:
  sampled 6,000 pairs, throughput ~490 pairs/s, full estimate ~1.1 h,
  timeout rate ~16%.

timeout=2s, workers=128, chunksize=1:
  sampled 3,000 pairs, throughput ~285 pairs/s, full estimate ~1.9 h,
  timeout rate ~13%.

workers=256 did not materially improve over 128 in the sampled test, likely due to process scheduling
and timeout-tail overhead. Prefer 128 workers first for full MCS runs.

Recommended full command:

python -m tools.chembl_tool.activity_transfer_benchmark.benchmark_mcs_runtime \
  --run-id dynamic_v1_mcs_full_t2_w128_stream \
  --full-scan \
  --workers 128 \
  --timeout-s 2 \
  --chunksize 1 \
  --progress-every 10000

Cleanup command after hang:

python -m tools.chembl_tool.activity_transfer_benchmark.benchmark_mcs_runtime \
  --run-id dynamic_v1_mcs_full_t2_w128_stream \
  --finalize-existing \
  --workers 128 \
  --timeout-s 2

MCS threshold analysis command:

python -m tools.chembl_tool.activity_transfer_benchmark.analyze_mcs_results \
  --run-id dynamic_v1_mcs_t2_analysis
```

Current MCS full-scan partial results:

```text
outputs/chembl_tool/activity_transfer_benchmark/mcs_runtime/dynamic_v1_mcs_full_t2_w128_stream/
  mcs_sample_results.tsv.tmp
  missing_result_indices.tsv
  summary.json
  report_zh.md

Completed 1,987,665 / 1,989,152 pairs, completion rate 99.9252%, missing 1,487.
timeout=2s observed timeout count is 218,664, about 11.0%.
```

Current MCS threshold analysis results:

```text
outputs/chembl_tool/activity_transfer_benchmark/mcs_analysis/dynamic_v1_mcs_t2_analysis/

non-ambiguous usable pairs: 1,500,676
best mean MCS coverage threshold: 0.70
best MCS macro-F1: 0.5538
best MCS balanced accuracy: 0.5542
best Tanimoto threshold on same subset: 0.48
best Tanimoto macro-F1 on same subset: 0.5688
best Tanimoto balanced accuracy on same subset: 0.5689

Conclusion: mean MCS coverage has activity-transfer signal, but as a standalone global threshold it does not exceed
Tanimoto. It is more suitable as a supplementary feature for later LLM / learned classifier, not a replacement for Tanimoto.
```

Current 3K LLM eval set:

```text
outputs/chembl_tool/activity_transfer_benchmark/llm_eval_sets/dynamic_v1_llm_3k/
  eval_pairs.jsonl
  eval_pairs.tsv
  summary.json
  report_zh.md

samples: 3,000
assay endpoints: 2,463
label counts: similar 1,500 / different 1,500
similarity buckets: 500 pairs per bucket
baseline on this intentionally balanced set:
  Tanimoto>=0.50 macro-F1 0.4977, balanced accuracy 0.5020
  Tanimoto>=0.48 macro-F1 0.4903, balanced accuracy 0.4963
  MCS>=0.70 macro-F1 0.4941, balanced accuracy 0.4963
```

Current gpt-oss-120b 3K LLM benchmark:

```text
outputs/chembl_tool/activity_transfer_benchmark/llm_runs/gpt_oss_120b_dynamic_v1_llm_3k_tools/
  manifest.json
  predictions.jsonl
  metrics.json
  report_zh.md
  figures/model_vs_baselines.svg
  runs/

model: gpt-oss-120b via local vLLM
tool service: http://127.0.0.1:8765
vLLM base URLs used: http://127.0.0.1:8001-8004/v1
api key used for this local vLLM server: EMPTY
n: 3,000, failed: 0, tool calls: 1,017
wall time: ~15.2 min with --parallelism 16 across 4 vLLM ports
usage: prompt_tokens 4,725,225; completion_tokens 1,651,970; total_tokens 6,377,195

LLM metrics:
  accuracy 0.5310
  balanced accuracy 0.5310
  macro-F1 0.5309

same-set baselines:
  Tanimoto>=0.50 macro-F1 0.4977
  Tanimoto>=0.48 macro-F1 0.4903
  MCS>=0.70 macro-F1 0.4941

gray zone subset, Tanimoto 0.40-0.70:
  n=818
  LLM macro-F1 0.5390
  Tanimoto>=0.50 macro-F1 0.4683

Conclusion: on this deliberately label- and similarity-bucket-balanced 3K stress test,
gpt-oss-120b + tools clearly exceeds the simple threshold baseline on the same set, but absolute performance is still weak.
This 3K set is not the full dynamic_v1 distribution and should not be directly compared one-to-one with full-data best Tanimoto macro-F1 ~0.569;
it is more suitable as an initial test of whether LLM can supplement structural thresholds on difficult samples.
```

Current DeepSeek-v4-pro 3K LLM benchmark:

```text
outputs/chembl_tool/activity_transfer_benchmark/llm_runs/deepseek_v4_pro_dynamic_v1_llm_3k_tools_thinking/
  manifest.json
  predictions.jsonl
  metrics.json
  report_zh.md
  trace_messages.jsonl
  runs/

model: deepseek-v4-pro via https://api.deepseek.com
api key source: DEEPSEEK_API_KEY from .env
tool service: http://127.0.0.1:8765
n: 3,000, ok: 2,994, failed: 6, tool calls: 6,009
wall time: ~81.8 min with --parallelism 60
usage: prompt_tokens 13,141,595; completion_tokens 6,991,635; total_tokens 20,133,230
reasoning_content saved: 2,994 / 2,994 ok samples

LLM metrics:
  accuracy 0.5471
  balanced accuracy 0.5472
  macro-F1 0.5378
  similar recall 0.4052
  different recall 0.6892

same-success-subset baselines:
  Tanimoto>=0.50 macro-F1 0.4981
  MCS>=0.70 macro-F1 0.4941

gray zone subset, Tanimoto 0.40-0.70:
  n=818
  LLM macro-F1 0.5090
  Tanimoto>=0.50 macro-F1 0.4683

Conclusion: DeepSeek-v4-pro is the LLM run with the highest overall metrics on the current 3K stress-test,
macro-F1 0.5378 is higher than gpt-oss-120b no-thinking's 0.5309 and thinking's 0.5253.
But the improvement is small and mainly comes from more conservative prediction of different; similar recall is low.
In the more critical Tanimoto 0.40-0.70 gray zone, DeepSeek is lower than both gpt-oss runs.
Considering tool calls, tokens, and time, the current cost-effectiveness is not as good as local gpt-oss.
```

Command to build the 3K eval set:

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.build_llm_eval_set \
  --run-id dynamic_v1_llm_3k
```

Command to run the local gpt-oss-120b LLM benchmark:

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.run_llm_benchmark \
  --run-id gpt_oss_120b_dynamic_v1_llm_3k_tools \
  --parallelism 16 \
  --base-urls http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1 \
  --max-tool-rounds 1 \
  --max-tokens 1024 \
  --timeout-s 180 \
  --skip-existing \
  --api-key EMPTY \
  --progress-every 100
```

Command to run the DeepSeek-v4-pro thinking LLM benchmark / resume from checkpoint:

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.run_llm_benchmark \
  --run-id deepseek_v4_pro_dynamic_v1_llm_3k_tools_thinking \
  --model deepseek-v4-pro \
  --base-url https://api.deepseek.com \
  --env-file .env \
  --api-key-env DEEPSEEK_API_KEY \
  --parallelism 60 \
  --max-tool-rounds 3 \
  --max-tokens 20480 \
  --timeout-s 300 \
  --skip-existing \
  --reasoning-effort high \
  --enable-thinking \
  --progress-every 100
```

Typical full baseline command:

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.run_benchmark \
  --run-id chembl36_activity_transfer_v1 \
  --max-total-pairs 2000000 \
  --max-pairs-per-assay 5000 \
  --binary-max-total-pairs 500000 \
  --workers 32
```

Recommended dynamic-range filtered command:

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.run_benchmark \
  --run-id chembl36_activity_transfer_dynamic_v1 \
  --min-pchembl-range 2.0 \
  --min-pchembl-iqr 0.75 \
  --max-total-pairs 2000000 \
  --max-pairs-per-assay 5000 \
  --binary-max-total-pairs 500000 \
  --workers 32
```

Output directory:

```text
outputs/chembl_tool/activity_transfer_benchmark/<run_id>/
  continuous_pairs.tsv.gz
  continuous_assay_endpoint_summary.tsv
  continuous_threshold_metrics.tsv
  continuous_similarity_bucket_summary.tsv
  continuous_assay_bucket_enrichment.tsv
  continuous_enrichment_summary.tsv
  binary_pairs.tsv.gz
  binary_assay_endpoint_summary.tsv
  binary_threshold_metrics.tsv
  binary_similarity_bucket_summary.tsv
  binary_assay_bucket_enrichment.tsv
  binary_enrichment_summary.tsv
  manifest.json
  report_zh.md
  figures/
    threshold_metrics.svg
    label_rates_by_bucket.svg
    median_delta_by_bucket.svg
    pair_counts_by_bucket.svg
    delta_lift_similar_rate_by_bucket.svg
    fold_lift_similar_rate_by_bucket.svg
    median_delta_reduction_by_bucket.svg
    binary_threshold_metrics.svg
    binary_label_rates_by_bucket.svg
    binary_delta_lift_similar_rate_by_bucket.svg
```

Current complete results:

```text
outputs/chembl_tool/activity_transfer_benchmark/chembl36_activity_transfer_v1/
outputs/chembl_tool/activity_transfer_benchmark/chembl36_activity_transfer_dynamic_v1/
```

The unfiltered v1 continuous-value main analysis includes about 40k assay-endpoints and about 1.98 million sampled pairs. The best single
Tanimoto threshold is about 0.45, with macro-F1 about 0.56. Assay-specific enrichment shows that close analogs
have a macro similar-rate lift of about +0.13 relative to their respective assay background, while distant / very_distant are negative;
structural similarity has a weak to moderate transfer signal, but should not be used alone as a criterion for activity transfer.

dynamic_v1 uses `pchembl_range >= 2.0` and `pchembl_iqr >= 0.75`, the continuous-value main analysis retains about 20k
assay-endpoints and about 1.99 million sampled pairs. Compared to unfiltered v1, similar/different labels are more balanced,
median |delta pChEMBL| is higher, and the macro similar-rate lift for close analogs is about +0.16;
this version is more suitable as the primary data for subsequent LLM assay-transfer benchmarks.

## Task-specific native runner records

From here, the BBB, ClinTox, Skin_Reaction code entry points, old task prompt/schema, DeepSeek run parameters, and interim results
are used to reproduce the native/legacy workflow of `tools/chembl_tool/tasks/<task>/run_reasoning_pipeline.py`. The current paper method is based on
`tools/chembl_tool/paper_experiments/`, each task's `experiment_config.py`, and
`tools/chembl_tool/tasks/AGENTS.md`; when there is a conflict, the old `Tier.endpoint_group` branch, task-specific
prediction policy, or historical "next steps" must not be restored to the paper runner.

## BBB code entry points

```text
tools/chembl_tool/tasks/bbb_martins/endpoint_groups.py
  Rules for BBB endpoint_group, evidence_direction, evidence_strength.

tools/chembl_tool/tasks/bbb_martins/build_evidence_library.py
  Entry point for building the BBB_Martins evidence library. Only the task default path, output file names, and endpoint assignment
  configuration are retained; common build logic is in tools/chembl_tool/common/task_workflows/evidence_library.py.

tools/chembl_tool/tasks/bbb_martins/retrieve_neighbors.py
  Entry point for BBB_Martins neighbor retrieval. Only the default index path is retained; common retrieval logic is in
  tools/chembl_tool/common/task_workflows/retrieve_neighbors.py.

tools/chembl_tool/tasks/bbb_martins/run_reasoning_pipeline.py
  Entry point for the BBB_Martins reasoning pipeline. Handles neighbor retrieval, single-molecule analysis,
  group-level concurrent reasoning, final summary, trace saving, and final-only rerun.

tools/chembl_tool/tasks/bbb_martins/run_reasoning_batch.py
  Entry point for BBB_Martins batch reasoning. Calls the single-molecule pipeline by query_index, supports molecule-level parallelism,
  optional trace saving/merging, prediction report, accuracy, and macro-F1 evaluation.

tools/trace_viewer/viewer.html
  Final paper trace visualization page. Scans identity-blind/deployment-visible conditions, views single-sample
  single/group/final messages, reasoning, tool calls, retrieval evidence, and generic JSON response.
  Does not include old task-specific structured field adaptation.

tools/trace_viewer/start_viewer.sh
  Registers Starling random/scaffold and historical TDC paper traces in a temporary, restricted serving root; the first argument is the port,
  subsequent optional arguments are trace roots to register.

tools/chembl_tool/tasks/bbb_martins/
  Other BBB evidence cleaning, scoring, reporting, and output aggregation scripts.
```

## ClinTox legacy archive

ClinTox is retired from active task registries. Its frozen lineage, code, tests,
sources, gold labels, evidence library, and no-promotion result are preserved in:

```text
data/legacy/clintox/
```

Historical data paths remain read aliases, but no active builder, publisher, or
reasoning runner may select ClinTox.

## Skin_Reaction code entry points

Skin_Reaction task-specific details are recorded in:

```text
tools/chembl_tool/tasks/skin_reaction/AGENTS.md
```

Current status:

```text
Completed task wrapper, assay scoring, endpoint grouping, evidence library, neighbor retrieval,
single/group/final reasoning pipeline, and batch wrapper.

Current label mapping:
  Y=1 -> risk
  Y=0 -> no_risk

Current v1 benchmark batch:
  outputs/chembl_tool/tasks/skin_reaction/reasoning/batches/skin_reaction_calib_50_v1
  n=82, failed=0
  accuracy=0.682927, macro-F1=0.678140
  positive precision=0.733333, recall=0.702128, F1=0.717391
  confusion matrix: TN=23 FP=12 FN=14 TP=33
```

Main entry points:

```text
tools/chembl_tool/tasks/skin_reaction/constants.py
  Skin_Reaction label and prediction mapping.

tools/chembl_tool/tasks/skin_reaction/rules.py
  Screening keywords and exclusion rules for skin sensitization, direct skin reaction, phototoxicity, irritation/corrosion, skin exposure,
  and weak/context evidence.

tools/chembl_tool/tasks/skin_reaction/scoring.py
  Assay retention/exclusion and scoring entry point. Both screen_assays.py and rescore_outputs.py call scored_row().

tools/chembl_tool/tasks/skin_reaction/endpoint_groups.py
  Rules for Tier.endpoint_group, evidence_direction, evidence_strength, and endpoint assignment.

tools/chembl_tool/tasks/skin_reaction/build_evidence_library.py
  Entry point for building the evidence library. By default reads assay_screening/v1, outputs molecule evidence, neighbor index, and meta.

tools/chembl_tool/tasks/skin_reaction/retrieve_neighbors.py
  Analog retrieval entry point. The current benchmark uses top-k-per-group=3, min-similarity=0.35.

tools/chembl_tool/tasks/skin_reaction/run_reasoning_pipeline.py
  Single-molecule reasoning pipeline: retrieval prefetch, single-molecule branch, group-level concurrent reasoning,
  final summary, trace saving, and final-only rerun.

tools/chembl_tool/tasks/skin_reaction/run_reasoning_batch.py
  Batch reasoning wrapper. Reuses common reasoning_batch.py, outputs predictions, metrics, report, logs,
  runs, and combined trace.
```

## BBB evidence grouping criteria

In the second phase of BBB, retrieval is not performed per assay. Instead, it should be grouped by:

```text
Tier -> endpoint_group
```

Generate retrieval groups.

### Tier 1: direct BBB / brain exposure

Suggested endpoint groups:

```text
direct_brain_plasma
  bpr
  brain/plasma
  b/p
  bbr
  ratio
  ratio auc

direct_unbound_brain
  k(p,uu,brain)
  k(p,uu,csf)
  kp
  fu

direct_logbb_or_brain_level
  logbb
  log bb
  brain level
  brain concentration
  brain penetration index
  bpi

direct_brain_uptake_or_perfusion
  drug uptake
  drug uptake(free)
  uptake
  brain uptake
```

### Tier 2: passive permeability / barrier model

Suggested endpoint groups:

```text
passive_papp
  papp
  logpapp
  logp app
  papp e-6

passive_caco2
  caco-2 papp
  caco-2 permeability
  pcaco2

passive_generic_permeability
  permeability
  permeability coefficient
  peff
  logpeff
  log pe
  pc
  pm
  pbbb

passive_transport_or_recovery
  drug transport
  drug recovery
```

### Tier 3: efflux transporter

Tier 3 must separate strong and weak evidence; transporter inhibition must not be directly interpreted as efflux substrate.

Suggested endpoint groups:

```text
efflux_functional_ratio_or_bidirectional
  efflux ratio
  ratio
  ratio_papp
  papp a to b (mean)
  papp b to a (mean)
  papp

efflux_transport_or_accumulation
  drug transport
  drug uptake
  activity
  flu intensity
  rfu
  fluorescence

efflux_atpase_or_probe
  ratio_atpase activity
  atpase
  relative jc-1 accumulation

efflux_inhibition_or_binding
  inhibition
  ic50
  ki
  ec50
  kd
  km
  kon
  k_off
  ratio ic50
  ratio ec50
  fc
```

### Tier 4: influx transporter

Tier 4 must also distinguish functional uptake from ordinary binding/inhibition.

Suggested endpoint groups:

```text
influx_functional_uptake_or_transport
  drug uptake
  uptake
  drug transport
  transport

influx_kinetic_or_substrate
  km
  vmax
  jmax
  kin

influx_inhibition_or_binding
  inhibition
  ic50
  ki
  kd
  kon
  k_off
  ec50
  ratio ic50
```

### Unknown / weak context

The following endpoints can only be used as context-dependent evidence, not strong interpretation alone:

```text
activity
ratio
inhibition
survival
cc50
gi50
ec90
mic
flu intensity
rfu
fluorescence
```

If these endpoints appear in a clear assay context, they can be promoted by endpoint group rules; otherwise they should be marked as:

```text
endpoint_group: context_dependent
evidence_strength: weak
```

## BBB evidence library

A molecule-level library should be built from current assay candidates and activity evidence. Each evidence entry must include at least:

```text
molecule_chembl_id
canonical_smiles
standard_inchi_key
assay_chembl_id
assay_id
assay_tier
endpoint_group
endpoint_group_reason
assay_description
target_chembl_id
target_pref_name
target_genes
standard_type
standard_relation
standard_value
standard_units
pchembl_value
data_validity_comment
activity_comment
evidence_direction
evidence_strength
evidence_reason
```

Where:

```text
evidence_direction:
  supports_bbb_crossing
  argues_against_bbb_crossing
  efflux_risk
  influx_support
  permeability_support
  context_dependent
  unknown_direction

evidence_strength:
  strong
  moderate
  weak
  context_dependent
```

`endpoint_group` is jointly determined by `assay_tier + standard_type + assay_description + target_genes`. You must not look only at `standard_type`, because endpoints such as `ratio`, `activity`, `inhibition` have different meanings in different assay contexts.

## Neighbor retrieval design

Input:

```text
query_smiles
top_k_per_group: default 3
min_similarity: default 0.3
groups: optional list[Tier.endpoint_group]
exclude_exact: default true
```

Process:

```text
1. Standardize the query molecule, generate canonical SMILES, InChIKey, Morgan fingerprint.
2. Build group membership based on Tier.endpoint_group in the evidence library.
3. For each group, independently compute Tanimoto between the query and the group's molecule fingerprints.
4. Each group returns top 3 non-identical neighbors, by default filtering out very distant analogs with Tanimoto < 0.3.
5. Aggregate all assay/activity evidence for the same neighbor molecule under that group.
6. Return the group-level retrieval payload.
```

Same-molecule exclusion criteria:

```text
same molecule_chembl_id
same full standard_inchi_key
same InChIKey connectivity layer, i.e. the first block before "-"
same canonical_smiles
```

Similarity buckets:

```text
very_close_analog: Tanimoto >= 0.95
close_analog: 0.80 <= Tanimoto < 0.95
moderate_analog: 0.60 <= Tanimoto < 0.80
weak_analog: 0.40 <= Tanimoto < 0.60
distant_analog: 0.20 <= Tanimoto < 0.40
very_distant_analog: Tanimoto < 0.20
```

By default, `min_similarity=0.3` is used to filter very distant analogs, reducing evidence that is almost non-transferable in sparse groups. Retained low-similarity analogs should still be handed to the group-level LLM to judge transferability. The LLM prompt must clearly state: `distant_analog` and `very_distant_analog` cannot be used as positive or negative evidence unless there is a strong medicinal chemistry rationale from shared scaffold and assay mechanism.

The current implementation uses precomputed ChEMBL fingerprints:

```text
tools/chembl_tool/chembl_data/chembl_36_fps/chembl_36.fps.gz
```

Build the fingerprint/group index for the BBB evidence molecule subset. This subset currently has about 50,000 molecules,
and at query time doing `BulkTanimotoSimilarity` for each group is fast enough. If later expansion to full ChEMBL background neighbor retrieval is needed,
a background index can be added without changing the BBB MVP payload contract.

### Query ChEMBL exact context / shared-assay enrichment

The current implementation retains an optional evidence-rich enhancement:

```text
tools/chembl_tool/tasks/bbb_martins/chembl_exact_context.py
```

It can use query full InChIKey to look up ChEMBL exact molecule, read ChEMBL compound properties,
query molecule's BBB-relevant evidence rows, and query activity in retrieved neighbor assays,
generating two types of shared-assay context:

```text
same_endpoint_activity:
  same assay_chembl_id + same normalized standard_type

same_assay_different_endpoint_activity:
  same assay_chembl_id + different standard_type
```

This type of information uses the query molecule's known ChEMBL experimental records. To avoid data leakage in prospective evaluation,
it must be disabled by default, so `query_chembl_context` / `exact_query_chembl_context` will not be added, and the query's
ChEMBL exact evidence will not be injected into the single-molecule prompt, group prompt, or batch evaluation. Only when explicitly passed:

```bash
--enable-chembl-exact-context
```

is it allowed for use as a retrospective / evidence-rich case study. Default benchmark, accuracy, and macro-F1 reports
should keep this switch off.

## LLM reasoning flow

The legacy native runner for BBB_Martins is divided into concurrent evidence branches and a final summary. The current paper runner first merges the
following source-local groups into the mechanism families declared by `experiment_config.py`.

### Group-level reasoning

In the legacy native runner, each `Tier.endpoint_group` executes independently:

```text
input:
  query molecule
  top 3 neighbors for this group
  cleaned raw ChEMBL assay/activity evidence rows
  tier and endpoint_group

available tools:
  mmp_structure_compare
  properties_compare

not sent:
  query standard_inchi_key
  query fingerprint
  group n_candidate_molecules
  evidence_direction
  evidence_strength
  endpoint_group_reason
  assay_reason

output:
  group_id
  useful_for_bbb_reasoning: true/false
  transferability:
    high
    moderate
    low
    not_applicable
  evidence_direction:
    supports_bbb_crossing
    argues_against_bbb_crossing
    efflux_risk
    influx_support
    neutral_or_unclear
  confidence:
    high
    moderate
    low
  reasoning_summary
  key_evidence:
    molecule_chembl_id
    similarity
    similarity_bucket
    assay_signal
    activity_values
    tool_summary
    transferability
    effect_on_bbb_reasoning
  caveats
```

`key_evidence` is the current format. The old `key_neighbors` is no longer used.

Each group's DeepSeek conversation, reasoning, tool calls, and tool messages are saved to the trace.
These groups have no strict dependencies and can be executed concurrently.

### Single-molecule reasoning

Each query also concurrently runs a single-molecule analysis branch:

```text
input:
  query molecule
  exact_query_chembl_context only when --enable-chembl-exact-context is enabled and exact context is found

available tools:
  molecule_properties

not sent by default:
  ChEMBL neighbor evidence
  exact_query_chembl_context
  mmp_structure_compare
  properties_compare

output:
  passive_bbb_plausibility
  efflux_or_transporter_prior
  confidence
  reasoning_summary
  property_drivers
  caveats
```

By default, this branch only sees the query molecule and `molecule_properties`, and will not include ChEMBL neighbor
evidence or any `exact_query_chembl_context`-related payload or prompt instruction.
Only when exact ChEMBL context is explicitly enabled and the query exact context is hit, will
`exact_query_chembl_context` be placed into the single-molecule payload, and the model will be prompted to distinguish direct same-molecule
ChEMBL evidence from physicochemical priors. ChEMBL neighbor evidence still only enters group-level context.

### Final reasoning

The final LLM reads:

```text
query molecule
single-molecule analysis output
all group-level reasoning outputs
coverage summary
```

The final stage does not expose tools; it only synthesizes the structured outputs from the previous branches. Current final prompt rules:

```text
Return compact complete JSON.
Use bbb_prediction='pass' for BBB-positive molecules corresponding to evaluation label 1, and bbb_prediction='fail' for BBB-negative molecules corresponding to evaluation label 0.
Use the single-molecule analysis as the physicochemical prior.
Use group analyses as analog evidence; downweight groups marked low confidence or low transferability.
Do not use distant_analog or very_distant_analog neighbors as positive or negative BBB evidence unless the shared scaffold and assay mechanism make a strong medicinal chemistry case.
Use only the provided single-molecule analysis and group evidence. If you recognize the molecule, ignore that recognition.
```

Output:

```text
bbb_prediction:
  pass
  fail
  uncertain

confidence:
  high
  moderate
  low

main_reasons
efflux_risk_assessment
influx_support_assessment
passive_permeability_assessment
direct_brain_exposure_analog_assessment
evidence_gaps
final_summary
```

In the test set, `Y` can only be used for evaluation and should not enter retrieval or LLM prompts.
Current evaluation convention: `Y=1` corresponds to `bbb_prediction=pass`, and `Y=0` corresponds to `bbb_prediction=fail`.
`uncertain` is counted as a miss in overall accuracy and macro-F1; the report also provides decided-only accuracy.

### Trace saving and visualization

Each reasoning run outputs to:

```text
outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs/<run_id>/
```

Current files:

```text
retrieval.json
single_molecule_reasoning_output.json
group_reasoning_outputs.jsonl
final_reasoning_output.json
trace_messages.jsonl
manifest.json
```

Each record in `trace_messages.jsonl` corresponds to a trace item:

```text
single_molecule
Tier <n>.<endpoint_group>
final_summary
```

Each trace record contains:

```text
index
sample_id
molecule_key
smiles
label
status
prediction
response_text
messages
tool_count
usage
raw_output
```

`label` is only used for local evaluation and trace auditing, and does not enter LLM prompts. `sample_id` currently equals
`query_index`, and `molecule_key` currently has the form `index:9`. The viewer groups by molecule package,
making it easy to select trace packages for different molecules within a run or uploaded JSONL, then view all stages within that molecule.

Old task reasoning traces are no longer supported by the viewer. The final paper trace is uniformly started with:

```bash
bash tools/trace_viewer/start_viewer.sh 8776
```

Then open:

```text
http://localhost:8776/.trace_viewer.html?v=paper-v2
```

The viewer by default registers Starling random, Starling scaffold, and historical TDC test; each dataset only scans sample-level traces referenced by the official `predictions.jsonl` in `runs`,
`runs_deployment_visible_prefetched`, `runs_deployment_visible`, and
`runs_deployment_visible_parent_disjoint`, and does not merge metrics across datasets. For parent-disjoint samples, the viewer also reads the manifest and `reuse.json`, displays the identity policy,
and distinguishes reruns after retrieval changes from artifact reuse when LLM-visible input is unchanged. The retention and cleanup rules for old task reasoning directories are in `tools/chembl_tool/paper_experiments/TRACE_RETENTION.md`.

Common pipeline commands:

```bash
# Historical TDC native runner: run a complete test_efflux molecule
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline \
  --query-index 0 \
  --top-k-per-group 3 \
  --min-similarity 0.3 \
  --max-workers 4 \
  --max-tool-rounds 10 \
  --timeout-s 300 \
  --max-tokens 8192 \
  --model deepseek-v4-pro \
  --run-id <run_id>

# Only rerun the final summary of an existing run
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline \
  --resume-final-from-run-dir outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs/<run_id> \
  --timeout-s 300 \
  --max-tokens 8192 \
  --model deepseek-v4-pro

# Historical TDC batch reproduction; new Starling benchmark must not reuse this input path or full-source index
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch \
  --input-jsonl data/gold_labels/legacy/processed/BBB_Martins/B3DB_cleaned/test/test_efflux.jsonl \
  --parallelism 1 \
  --top-k-per-group 3 \
  --min-similarity 0.3 \
  --max-tool-rounds 10 \
  --timeout-s 300 \
  --max-tokens 8192 \
  --model deepseek-v4-pro \
  --batch-id <batch_id>
```

By default, the stderr progress of each single-molecule run is printed to the console in real time, e.g.,
`[idx00003 stderr] [bbb_reasoning_pipeline] group done: ...`, while also being fully saved to
`outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/logs/`. If you only want to write log files and not display progress on the console, add
`--no-stream-logs`.

Batch resume uses `--skip-existing`. When rerunning with the same `--batch-id`, the script checks
`reasoning/batches/<batch_id>/runs/<batch_id>_idxNNNNN/final_reasoning_output.json`:
if it exists, the molecule is considered complete and skipped, without overwriting existing stdout/stderr logs; if it does not exist, the molecule is rerun.
Therefore, a partial run after interruption will automatically complete, and completed results will enter the new predictions, metrics, report, and
batch `trace_messages.jsonl` summary. This logic is uniformly implemented by `tools/chembl_tool/common/task_workflows/reasoning_batch.py`,
shared by BBB_Martins, Bioavailability_Ma, ClinTox, and Skin_Reaction.

Batch outputs:

```text
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/manifest.json
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/predictions.jsonl
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/metrics.json
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/report.md
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/trace_messages.jsonl
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/logs/
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/runs/
```

Each molecule's independent run within the batch is kept inside the batch directory:

```text
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/runs/<batch_id>_idx00000/
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/runs/<batch_id>_idx00001/
...
```

Old task batches can generate merged traces for offline auditing, but the final paper runner must use
`--no-combine-traces`, keeping only each molecule's own trace. The final viewer is fixed to start with:

```bash
bash tools/trace_viewer/start_viewer.sh 8776
```

The viewer builds the sample list from the condition's `predictions.jsonl`, then loads per-run traces and retrieval as needed.
When adding a new task, the common stage/message/tool/JSON contract must be followed; do not add task-specific adaptations for prediction,
Tier, expert-policy, or `key_evidence` effect fields to the viewer.

### LLM usage and cost estimation

DeepSeek API responses return token usage, but do not directly return dollar costs in each response.
To estimate batch costs, aggregate tokens from the usage saved in traces, then multiply by DeepSeek's official pricing.
Prices change; before production estimation, you must check the official page:

```text
https://api-docs.deepseek.com/quick_start/pricing/
```

The final paper viewer is only responsible for trace auditing and visualization, and does not include provider-specific pricing or cost estimation.
To estimate the cost of an old run, aggregate tokens offline from the usage field of traces and use the official price on the day of the run.

As of 2026-05-12, the official page shows the current discounted price for `deepseek-v4-pro` as:

```text
input cache hit:  $0.003625 / 1M tokens
input cache miss: $0.435    / 1M tokens
output:           $0.87     / 1M tokens
```

The discount is valid until 2026-05-31 15:59 UTC; the original prices are:

```text
input cache hit:  $0.0145 / 1M tokens
input cache miss: $1.74   / 1M tokens
output:           $3.48   / 1M tokens
```

Aggregate usage from the `trace_messages.jsonl` of a run or batch:

```bash
jq -s 'reduce .[] as $r (
  {calls:0,prompt:0,completion:0,total:0,cache_hit:0,cache_miss:0,reasoning:0};
  .calls += (if ($r.usage//null) then 1 else 0 end) |
  .prompt += (($r.usage.prompt_tokens // 0) | tonumber) |
  .completion += (($r.usage.completion_tokens // 0) | tonumber) |
  .total += (($r.usage.total_tokens // 0) | tonumber) |
  .cache_hit += (($r.usage.prompt_cache_hit_tokens // $r.usage.prompt_tokens_details.cached_tokens // 0) | tonumber) |
  .cache_miss += (($r.usage.prompt_cache_miss_tokens // 0) | tonumber) |
  .reasoning += (($r.usage.completion_tokens_details.reasoning_tokens // 0) | tonumber)
)' outputs/chembl_tool/tasks/<task_name>/reasoning/batches/<batch_id>/trace_messages.jsonl
```

If estimating a standalone single-molecule run, replace the path with:

```text
outputs/chembl_tool/tasks/<task_name>/reasoning/single_runs/<run_id>/trace_messages.jsonl
```

Where `output_tokens` uses the `completion_tokens` from usage.

Cost formula:

```text
cost =
  cache_hit_tokens  / 1,000,000 * cache_hit_price
+ cache_miss_tokens / 1,000,000 * cache_miss_price
+ output_tokens     / 1,000,000 * output_price
```

Current BBB_Martins smoke estimation baseline:

```text
single molecule example:
  14 LLM calls
  prompt 186,050 tokens
  completion 54,562 tokens
  cache_hit 121,088
  cache_miss 64,962
  estimated discounted cost: ~$0.076 / molecule

3 molecule parallelism=3 smoke:
  40 LLM calls
  prompt 459,509 tokens
  completion 152,893 tokens
  cache_hit 340,352
  cache_miss 119,157
  estimated discounted cost: ~$0.186 total, ~$0.062 / molecule
```

These are only for rough estimation. Different molecules' endpoint group coverage, tool rounds, and final prompt length will vary;
for a full budget, first sample 3-10 molecules, multiply the average cost by the number of molecules, and leave margin.

## FastAPI resident service standard

The service initializes slow resources at startup, and each subsequent tool invoke reuses the loaded objects:

```text
RDKit standardization config
MolGpKa import/model state
AccFG import/state
mmpdb Python package import
Other slow-start ML models or large indexes later
```

Current endpoints:

```text
GET /health
GET /tools
POST /tools/{tool_name}/invoke
POST /tools/invoke
POST /tools/{tool_name}
```

Among them, `/tools/{tool_name}/invoke` is the standard general tool interface. `/tools/invoke` and `/tools/{tool_name}` are compatibility entry points. `/tasks/bbb_martins/retrieve`, `/tasks/bbb_martins/reason`, `/tasks/bbb_martins/predict` will be added later if task-level orchestration is needed.

### General ToolRequest

```json
{
  "request_id": "string",
  "tool_name": "molecule_properties",
  "version": "v1",
  "input": {},
  "options": {
    "timeout_s": 120,
    "return_debug": false
  }
}
```

### General ToolResponse

```json
{
  "request_id": "string",
  "tool_name": "molecule_properties",
  "version": "v1",
  "status": "ok",
  "output": {
    "text": "LLM-readable natural language output",
    "..."
  },
  "warnings": [],
  "errors": [],
  "metadata": {
    "started_at": "ISO-8601",
    "finished_at": "ISO-8601",
    "latency_ms": 0,
    "model_or_index_version": "string"
  }
}
```

On failure:

```json
{
  "status": "error",
  "output": null,
  "warnings": [],
  "errors": [
    {
      "code": "INVALID_SMILES",
      "message": "Could not parse query SMILES.",
      "recoverable": true
    }
  ]
}
```

## Tool: molecule_properties v1

Input:

```json
{
  "query_smiles": "CCO"
}
```

Responsibilities:

```text
1. Standardize the query SMILES, returning canonical_smiles and standard_inchi_key.
2. Compute easily interpretable RDKit descriptors.
3. Compute acidic/basic pKa and logD via MolGpKa.
4. Identify the top-level functional groups via AccFG.
5. Generate natural language output.text as the only tool text directly consumed by the LLM.
```

Main output fields:

```text
output.text
output.query
output.properties
output.functional_groups
output.raw_features
```

`properties` at least cover:

```text
MolGpKa pKa/logD features
RDKit molecular weight
logP
TPSA
HBD/HBA
rotatable bonds
formal charge
heavy atom count
aromatic rings
fraction Csp3
QED
rule flags
```

## Tool: properties_compare v1

Input:

```json
{
  "query_smiles": "CCN(CC)CC",
  "reference_smiles": "CCO",
  "query_label": "query",
  "reference_label": "reference"
}
```

Responsibilities:

```text
1. Reuse the same molecule_properties tool instance in the registry to avoid re-initializing MolGpKa/AccFG.
2. Compare all non-functional-group properties in molecule_properties.
3. Delta is defined as query_value - reference_value.
4. Generate natural language output.text explaining which properties increased, decreased, or are not applicable.
```

Main output fields:

```text
output.text
output.query
output.reference
output.comparisons
```

Functional groups are not compared in this tool. FG information is only returned separately by `molecule_properties`.

## Tool: mmp_structure_compare v1

Input:

```json
{
  "query_smiles": "CCN(CC)CC",
  "reference_smiles": "CCO",
  "query_label": "query",
  "reference_label": "reference"
}
```

Responsibilities:

```text
1. Compute Morgan fingerprint Tanimoto similarity.
2. Mark structural similarity by similarity bucket.
3. Call mmpdb fragmentation / matched-pair logic to describe interpretable matched-pair transformations.
4. Compute RDKit MCS coverage to help judge common scaffold proportion.
5. Generate natural language output.text as an LLM-readable structural difference explanation.
```

Main output fields:

```text
output.text
output.query
output.reference
output.similarity
output.transformation
output.mcs
```

`mmp_structure_compare` does not return descriptor/property deltas. All property difference comparisons must use `properties_compare`.

Structural similarity buckets:

```text
very_close_analog: Tanimoto >= 0.95
close_analog: 0.80 <= Tanimoto < 0.95
moderate_analog: 0.60 <= Tanimoto < 0.80
weak_analog: 0.40 <= Tanimoto < 0.60
distant_analog: 0.20 <= Tanimoto < 0.40
very_distant_analog: Tanimoto < 0.20
```

## ChEMBL neighbor retrieval status

`chembl_neighbors` remains an important evidence retrieval concept for BBB_Martins reasoning, but the current implementation is not a DeepSeek function tool, nor is it exposed as a registered resident tool of `tools/service/`. The current code entry point is:

```text
tools/chembl_tool/tasks/bbb_martins/retrieve_neighbors.py
```

Legacy native runner invocation:

```text
run_reasoning_pipeline.py reads the BBB neighbor index before LLM calls,
prefetches top 3 neighbors per group by Tier.endpoint_group,
cleans internal derived fields, then puts neighbors/evidence_rows into the group-level user prompt.
```

Therefore, group-level DeepSeek sees neighbor evidence but cannot actively call `chembl_neighbors`. The only tools it can call are:

```text
mmp_structure_compare
properties_compare
```

If BBB evidence retrieval is later placed into a resident service, reuse the existing ToolRequest / ToolResponse contract and load the BBB evidence library and fingerprint group index at server startup. Do not duplicate the service framework.

## Concurrency strategy

The service can execute concurrently:

```text
molecule_properties
properties_compare
mmp_structure_compare
ChEMBL neighbor retrieval / evidence prefetch for each group (pipeline internal; can be service-ized later)
group-level LLM reasoning for each group
```

Concurrency boundaries between the legacy native runner and the current paper runner:

```text
1. Source-local retrieval/normalization can maintain endpoint-group granularity.
2. The legacy native runner's group-level reasoning granularity is endpoint group; the paper runner must use mechanism family.
3. Final reasoning must wait for all enabled mechanism-family/group reasoning to complete.
4. Each query must have a request_id/run_id, and all intermediate artifacts must be traceable.
```

## Implementation plan

### Phase 0: Documentation and interface freeze

Status:

```text
Completed and recorded in this file with current tool entry points and LLM-visible output conventions.
```

Currently frozen service tool names:

```text
molecule_properties
properties_compare
mmp_structure_compare
```

### Phase 1: BBB evidence library

Status:

```text
Core entry points implemented:
tools/chembl_tool/tasks/bbb_martins/endpoint_groups.py
tools/chembl_tool/tasks/bbb_martins/build_evidence_library.py
```

Goals remain unchanged:

```text
Can generate molecule-level evidence from v6 assay/activity output.
Each evidence has endpoint_group, evidence_direction, evidence_strength.
Tests cover Tier.endpoint_group mapping rules.
```

### Phase 2: BBB neighbor retrieval

Status:

```text
Core entry point implemented:
tools/chembl_tool/tasks/bbb_martins/retrieve_neighbors.py
```

Current behavior:

```text
Given a SMILES, return top 3 non-identical neighbors per Tier.endpoint_group.
Return full assay description and activity rows.
Can batch process test_efflux.jsonl.
Default min_similarity=0.3 filters very distant analogs; retained low-similarity analogs continue to be bucket-marked.
```

### Phase 3: Resident FastAPI service

Status:

```text
Implemented:
tools/service/app.py
tools/service/config.py
tools/service/registry.py
tools/service/schemas.py
tools/service/errors.py
tools/service/tools/base.py
tools/service/tools/rdkit_properties.py
tools/service/tools/properties_compare.py
tools/service/tools/mmp_structure_compare.py
```

Current acceptance criteria:

```text
Initialize slow resources like MolGpKa, AccFG, mmpdb at service startup.
POST /tools/{tool_name}/invoke can call molecule_properties, properties_compare, mmp_structure_compare.
GET /tools can list tool schemas and versions.
All tool output.text is LLM-visible text, with numbers rounded to at most two decimal places.
```

Note: `chembl_neighbors` is currently not registered as a service tool, nor is it an LLM-callable tool; it is an evidence prefetch/context assembly inside the BBB pipeline. DeepSeek only receives prefetched group evidence. If service-ization is needed later, add `tools/service/tools/chembl_neighbors.py`, but do not replace or duplicate the existing general tool framework.

### Phase 4: LLM reasoning payloads

Status:

```text
Core orchestration implemented:
tools/chembl_tool/tasks/bbb_martins/run_reasoning_pipeline.py
```

Current behavior:

```text
1. Read queries from test_efflux.jsonl.
2. Use retrieve_neighbors.py to get top 3 neighbors per Tier.endpoint_group.
3. Concurrently execute single-molecule analysis, exposing only molecule_properties.
4. Concurrently execute each group-level analysis, exposing only mmp_structure_compare and properties_compare.
5. After all group and single branches complete, execute final summary without exposing tools.
6. Save retrieval, single, group, final, trace_messages, and manifest.
7. Support --resume-final-from-run-dir to reuse existing retrieval/single/group outputs and only rerun the final summary.
```

Current DeepSeek invocation conventions:

```text
OpenAI SDK
base_url=https://api.deepseek.com
model=deepseek-v4-pro
thinking enabled
reasoning_effort=high
```

API key is read by default from `--env-file .env` as `DEEPSEEK_API_KEY`. `run_reasoning_pipeline.py`
will cause the value in `.env` to override the same-named environment variable already present in the current shell; this ensures that when running batch directly from the shell,
the project `.env` takes precedence. To temporarily switch keys, explicitly pass `--api-key-env <ENV_NAME>` and configure the corresponding variable in `.env`.

Current trace acceptance criteria:

```text
Completely save system/user/assistant/tool messages.
Save assistant reasoning_content.
Save assistant tool_calls.
Save ToolResponse.output.text as tool message content.
Save molecule_key to allow viewer grouping by molecule trace package.
```

### Phase 5: BBB_Martins evaluation

Status:

```text
Single-molecule end-to-end smoke runs completed, output directory:
outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs/<run_id>/
```

Verified examples:

```text
query_index=0:
  run_id=first_efflux_full_key_evidence_20260505_182029
  final prediction=fail
  confidence=moderate
  19 group outputs, 21 trace records, 57 key_evidence items

query_index=9:
  run_id=last_efflux_full_key_evidence_20260505_185201
  final prediction=fail after final-only prompt rerun
  confidence=moderate
  19 group outputs, 21 trace records, 57 key_evidence items
```

Subsequent evaluation work:

```text
Batch run retrieval + reasoning on test_efflux.jsonl.
During evaluation, only compare against Y at the end; do not pass Y to tools or LLM.
Report coverage, prediction accuracy, uncertain rate, and typical success/failure cases.
Batch trace supports multiple molecule packages; viewer groups by molecule_key for browsing.
```

### Phase 6: Extension to other tasks

New tasks only add:

```text
tools/<domain_tool>/tasks/<task_name>/
tools/service/tasks/<task_name>.py
```

Corresponding outputs are uniformly placed in:

```text
outputs/<domain_tool>/tasks/<task_name>/
  assay_screening/
  evidence_library/
  reasoning/
    single_runs/
    batches/
```

If a new model or index is needed, add:

```text
tools/service/tools/<tool_name>.py
```

Do not duplicate existing service frameworks, tool schemas, or request/response standards. Do not duplicate the following task workflow helpers either:

```text
tools/chembl_tool/common/task_workflows/screen_assays.py
tools/chembl_tool/common/task_workflows/rescore_outputs.py
tools/chembl_tool/common/task_workflows/summarize_outputs.py
tools/chembl_tool/common/task_workflows/assay_report.py
tools/chembl_tool/common/task_workflows/evidence_library.py
tools/chembl_tool/common/task_workflows/retrieve_neighbors.py
tools/chembl_tool/common/task_workflows/chembl_exact_context.py
tools/chembl_tool/common/task_workflows/reasoning_batch.py
```

The `run_reasoning_batch.py` of a new task should act as a thin wrapper calling
`tools/chembl_tool/common/task_workflows/reasoning_batch.py`, configuring only:

```text
default_input
default_batch_root
default_index
pipeline_module
prediction_field
positive/negative label mapping
```

Task-specific pipelines, retrieval, prompts, and final schemas can remain in their respective task directories; when these parts stabilize enough to be general, extract common modules.

New tasks must keep task-specific outputs within the generic JSON response and follow the unified
stage/message/tool contract. The viewer will recursively display these JSONs; do not add task-specific rendering branches.

## What is not done currently

```text
Do not train a BBB classifier.
Do not treat each assay as an independent retrieval group.
Do not interpret IC50/inhibition directly as substrate or transport.
Do not expose the Y of test_efflux.jsonl to retrieval or LLM prompts.
Do not re-import or re-initialize MolGpKa, AccFG, mmpdb, ML models, or large indexes on each query.
Do not put descriptor/property deltas into mmp_structure_compare; property differences uniformly go through properties_compare.
Do not show the entire structured JSON of tools to the LLM; the LLM only sees output.text by default.
Do not add source path environment variables for the installed mmpdb.
```

## Historical BBB phased next steps

The following is a plan left from the early BBB single-molecule MVP phase, now superseded by the current paper experiment plan, and is only used to explain historical implementation:

```text
1. Batch evaluate test_efflux.jsonl to form a prediction/label comparison table and error analysis.
2. Continue auditing evidence weighting in the final summary, adding explicit adjudication fields if necessary.
3. If other systems need to reuse retrieval, wrap chembl_neighbors as a service tool or task endpoint; the current BBB pipeline continues to use it as internal evidence prefetch.
4. Later integrate more resident ML tools, such as slower pKa/logD, solubility, PK, or toxicity models.
```
