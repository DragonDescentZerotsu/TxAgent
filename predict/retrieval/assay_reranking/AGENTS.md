# Assay-reranking boundary

## Active independent-level release

`ranked_uid_retrieval.v1` is the active BBB/Oral Gold-v1 retrieval contract.
Each task, split, and level owns one immutable cache. L1 is scaffold-disjoint
and contains the top 100 frozen context cards. L2+ is parent-disjoint and
contains every physical UID under the top 100 Morgan-ranked parents; it uses
only the `all` pool. The same expanded UID universe carries dense Morgan and,
where supported, assay-transfer ranks. Active modes are `morgan` and
`assay-transfer`; joint and alternate pools require an archived legacy bundle.

The task release index connects every level to one evidence-owned projection.
At L1, resolve the embedded `context_id -> source_row_uid[]` membership and
expose ordered physical Gold members up to the explicit per-card limit. Do not
use voter construction IDs, choose a representative row, or regroup L1 by the
current parent normalizer. At later levels, select ranked UIDs directly and
hydrate their deduplicated union. A caller may set K independently for every
later level to any positive value up to that query's available expanded UID
count; an oversized request must fail before rendering and must never be
silently clamped. Levels are independent. Runtime reads only small manifests,
indexed ranking rows, and selected projection rows; source equivalence, database
hashes, integrity, and disjointness are publication checks.

The V27 safety and Skin training-prompt successors are indexed in
`MODEL_CONTRACTS.md`. Their indirect query side uses the query SMILES and the
candidate record's training-visible query fields. Keep the original Starling
visibility rules: known-only measurements, support text, and other hidden
fields never appear on the query side. The training-pair renderer remains
byte-exact to Starling; the candidate-context inference adapter has its own
hash-pinned `candidate_copy_v1` cache identity because live queries lack their
own assay record. Skin also uses the pinned
canonical endpoint; preserve its older source-endpoint/whitespace cache unchanged.
Direct TDC safety successors score only the existing frozen L1 Morgan-100 cards.
For every safety or Skin V27 query, rank the distinct parents in the combined
L2+ evidence pool once, then expand only the selected parents to unique physical
L2+ UIDs. The default is the top 100; the opt-in Gold Carcinogens and TDC AMES
successors take the first 40 from each query's frozen shared top-100 list under
new `shared_parent40_v1` identities. Never choose a different parent set per level,
recompute the frozen Morgan ordering for these successors, or score the full
evidence library per query. Level directories are storage
partitions for the existing harness, not independent parent selections; the
general V27 tool is unchanged across those partitions. If one parent ID has
multiple projected SMILES, use the lexicographically smallest one for its
one-time Morgan comparison, while retaining every physical record. For these
successor caches, full-flat displays every physical L2+ record under the shared
selected parents, regardless of level; the legacy 50-per-level display limit does
not apply. Preserve distinct source UIDs even when external record IDs repeat.
Do not use prompt-size checks as cache-generation or ranking-validation gates.
Use a new shared-parent cache identity and
reuse finalized scores or stopped score journals only when the exact model and
prompt-derived score key match. All seven model
prompts must pass the sampled real-dataset audit before GPU scoring. On dgx027,
use four distinct `srun` GPU ranks with device `0` inside each rank and shard
index `SLURM_LOCALID`; publish only closed and validated local staging trees.
For V27 safety successors, `build_safety_v27_ranked_retrieval tokenize` may
precompute the exact A/B token prefixes on an EPYC allocation from closed
`prompts.parquet` files. The optional `--tokenized-root` scoring path must
match the prompt-file hash, model revision, shard order, and token-file hashes;
it does not change score keys or existing score journals.
For large prepared levels, `--tokenize-workers` may parallelize independent
shards; keep `--num-shards` consistent between tokenization and GPU scoring.
For Gold Carcinogens and TDC Ames, `--supplement-41-50` prepares a separate
score-only rank window from each query's frozen top-50 parent list. It requires
an explicit output root and source parent universe, and cannot write an active
cache index. CPU-tokenize this supplement independently; do not auto-queue GPU
scoring. Once its scores and the top-40 release are closed, a fresh top-50
preparation may reuse both exact score-key sources. Never edit either source or
silently activate the top-50 successor.

The canonical `active/flat_v5/<benchmark>/<task>/<role>/<method>/<release>/<variant>`
tree holds independent task releases; the Gold and TDC V27 successor YAMLs join
their L1 and L2+ release indexes for the harness. In those YAMLs, Gold
Carcinogens and TDC Ames L2+ select `general_candidate_copy_shared_parent40_v1`;
other safety L2+ tasks select `general_candidate_copy_shared_parent100_v2`.
Validated `staging/` trees are not published releases. Keep GPU writers
separate from closed scored stages. Publish only a validated closed release to
its unoccupied canonical path, then inventory and verify the matching bundle
before inference. Gold Carcinogens and TDC Ames may also have opt-in
`shared_parent40_partial_snapshot_v1` releases built from closed score journals.
They retain the full Morgan-40 universe and original Morgan ranks, but only
genuinely scored records receive assay-transfer ranks. Empty assay-ranked
query/levels are valid. Never invent missing scores, call these fully scored,
replace their resumable sources, or switch the default bundle to a snapshot.
Use the dedicated YAML and verified receipt for inference.
The upstream V5 harness may read DILI and Carcinogens through L7 from these
releases. Its safety L5 method is assay-transfer; do not apply the legacy
BBB/Oral Morgan-only L5 stage rule to a shared V27 safety index.

## Full-flat V5 artifact contract

The full-flat-context-v5 family uses the tracked manifests in
`flat_v5_manifests/<benchmark>/<task>.json`. A manifest enumerates the selected
cache release indexes, cache payloads, evidence manifests, evidence projections,
and the exact cache-bundle YAML; it does not move those files out of their
semantic owners. The checked-in manifest may therefore be complete while the
payloads are absent from a fresh checkout.

Use `python -m predict.retrieval.assay_reranking.artifact_bundle inventory` to
rebuild a manifest only when the selected release inputs change. After restoring
compressed release parts with `restore`, run `verify` once. The resulting ignored
receipt under `data/caches/assay_reranking/active/flat_v5/receipts/` is the launch
contract: flat preparation requires it and performs only manifest/configuration
and size/mtime checks. `--verify-flat-artifacts` explicitly repeats the expensive
hash pass. Never add a runtime source scan, whole-database hash, integrity scan,
or cache builder to the harness.

Canonical restored cache identities use
`flat_v5/<benchmark>/<task>/<role>/<method>/<release>/<variant>`. The historical
V24.1 BBB and V25 Oral names remain compatibility aliases; when a canonical
successor is present, `runtime.cache_profile_root()` resolves it, otherwise the
legacy location remains usable. Do not add compatibility symlinks or overwrite a
legacy cache in place. Direct and matrix flat launchers select the TDC-v1 bundle
for TDC runs and the all-task Gold-v1 bundle for Gold safety tasks; an explicit
`--assay-transfer-cache` always wins.

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
job's resumable journals, prepared caches, score journals, and checkpoints, but it
is never durable or canonical. As soon as writers finish, copy or move the closed
tree to `/vast`, validate and publish the `/vast` copy, then remove local staging.
Never report a cache as durable, complete, or published while it exists only under
`/local`. If migration is needed before completion, stop every writer at a flushed
boundary before copying and resume only after the `/vast` copy validates.

## Testing discipline

Do not add circular tests that merely assert newly written prompt prose or copy
implementation literals into the test. Prompt wording is validated with reviewed
input/output fixtures or a real pilot/evaluation. Automated tests should cover
executable behavior, failure modes, schemas, rendering validity, and provenance.

This package prepares reusable assay-transfer scores before a reasoning harness runs.
Harnesses consume finalized caches and must not call these checkpoints online.

Direct gold-label transfer supports task-specific V9, V10.3, the Oral V10.3.0.2
display successor, and V10.4 checkpoints. Record-level transfer supports the task-specific
BBB/Bioavailability/Skin V19.1 checkpoints, the opt-in
BBB V21 all-record cache, BBB V24.1 level-specific L2-L4 checkpoints, and Oral V25
level-specific L2/L3/L4/L6 checkpoints. Add no compatibility fallback or alternate checkpoint.

`runtime.py` owns the pinned task-to-role mapping and cache contract; `v9.py` owns its
shared direct-gold prompt and Morgan train-neighbor prefilter; `v19_1.py` owns its record-level prompt.
`progressive_levels.py` is the independent offline BBB/Bioavailability/Skin cache
builder for the current Stage 3 level sidecar. Keep direct and indirect scientific
contracts separate.

Current progressive membership is assigned from normalized-v7 Stage 3 records by
`data.processing.evidence_library.versions.v7.progressive_level_mapping`. The sidecar
uses current vote-ledger membership rather than the stored `retrieval_source_id` tag.
BBB has L1-L5, Bioavailability has L1-L6, and Skin has L1-L4. Skin exposure records
outside the current progressive curve retain a null level instead of being forced in.

For BBB, L1 contains current direct voters and is globally excluded from this score
cache. L2 is current near-direct evidence plus former-L1 records that do not currently
vote; L3-L5 are current passive, efflux, and influx assignments. Do not globally remove
L2-L5 records merely because their parent occurs in validation/test or another query.
Every retained L2-L5 molecule instead passes per-query `scaffold_disjoint` exclusion.

Build the BBB L2-L5 cache from Stage 3, not Stage 6/7/8 held-out views. Construct fresh
parent-molecule identities and Morgan fingerprints, then take independent top-75 pools
per level with no similarity floor. Globally omit the 260 pinned unresolved-endpoint
records and the 5 pinned rows without a resolvable parent identity. The latter cannot be
Morgan-ranked or checked under `scaffold_disjoint`; do not fall back to raw SMILES. Keep
every other nonnumeric or assay-transfer-ineligible L2-L5 row, and retain its explicit
out-of-domain flag in the level sidecar.

The 260 endpoint exclusions have no usable source endpoint and therefore never entered
the assay-transfer prompt contract. They may be reconsidered only through a reviewed,
source-specific endpoint mapping; do not synthesize `missing_endpoint` prompts.

L1 reuses the V9 gold cache at Morgan width 75. L2-L5 use the BBB V19.1 checkpoint and
the V19.1 raw/source-field prompt projection. Copy the retrieved record's assay context
to the hidden-value query side and use the normalized-v7 parent SMILES for the known
molecule. The direct-BBB projection extension is explicit OOD because L2 direct-source
records were outside numeric-indirect checkpoint training.

The opt-in V21 BBB profile is separate from that default hybrid. It independently
ranks raw Stage 3 records at L1-L5, including current direct voters at L1, with the
same per-query top-75 scaffold-disjoint parent pools. Its query side copies each
record's source-native assay context while hiding result-bearing fields exactly as in
the V21 training prompt.

The V24.1 BBB profile is another separate hybrid: frozen gold-train V9 supplies L1,
then V10 Stage 3 records join exactly to Tianang's `source_row_uid` mapping and use
the pinned L2, L3, and L4 checkpoints. Each level takes 75 distinct scaffold-disjoint
parent molecules with no similarity floor, then expands and scores every record in
V24.1 assay-transfer bucket eligibility. Its V20/V23.2 prompt projection uses V10
`bbb_semantic_display.v2`; Experiment B copies only non-result assay context.
Measurement text, support text, extra details, BBB labels, conclusions, uncertainty,
and other result-bearing fields remain hidden. L5 is not part of this profile.

The Oral V25 profile is cache-only and does not alter a reasoning harness. It joins
V10 Stage 3 rows exactly through Tianang's `source_row_uid` mapping and builds
independent Morgan-75 scaffold-disjoint L2/L3/L4/L6 pools for valid and test. Rendering
reuses the released Oral V25/V24 field projection and template: Experiment B copies
source-native assay context but hides measurement text, support text, extra details,
substrate status, and comparator exposure. Copied query context also excludes the
retrieved molecule's name, which is not an assay condition. L3/L4/L6 are continuous-only; L2 includes
continuous and ordinal measurements, both with the released source-native display.
Parent exclusion checks every represented parent-SMILES form, so expansion cannot
reintroduce same-scaffold records through another tautomer sharing that parent ID.
No BBB binary semantic-display transform applies. Frozen V9 supplies the separate
Bioavailability L1 test cache.

Use `v25_oral_levels prepare --subset valid --l2-only`, then
`score --subset valid --level L2 --device 0`, then
`finalize --subset valid --l2-only`; repeat for test. L2 lives in the separate
`v25_oral_uid_levels_morgan75_l2` profile and is read with `levels=("L2",)`.
For L3 use `prepare --subset valid --single-level L3`,
`score --subset valid --level L3 --device 0`, then
`finalize --subset valid --single-level L3`; repeat for test. Its profile is
`v25_oral_uid_levels_morgan75_l3`, read with `levels=("L3",)`.
`--single-level L2` also selects the existing separate L2 profile.
Large levels can use `score --num-shards 2 --shard-index 0` and shard index 1 on
different GPUs. Finalization requires the exact complete prompt-key union and
rejects duplicates across all journals before publishing.
The existing L2/L4/L6 cache files and their audited original builder provenance remain
unchanged. Finalization stages journal scores in a connection-local temporary table;
missing, duplicate, or invalid journals can be corrected and retried without cleanup.

Build Bioavailability L2-L6 under the same Stage 3, top-75, no-floor, and per-query
`scaffold_disjoint` policy. Globally exclude L1 and the one pinned row without a
resolvable parent identity. Its `hf_bioavailability` L2 nonvoters use an explicit OOD
projection extension because that direct source was outside V19.1 indirect training.
Build Skin L3-L4 with its task-specific V19.1 checkpoint under the same Stage 3,
top-75, no-floor, and per-query `scaffold_disjoint` policy. L4 is the distinct-hazard
phototoxicity/irritation/local-damage family, not sensitization-label evidence. The
`direct_skin_reaction` L3 projection extension is explicit OOD because that source was
outside V19.1 numeric-indirect training. There is no checkpoint fallback.

`three_pools.py` owns the separate Morgan-100 BBB V24.1/oral V25 pool expansion.
It does not alter legacy builders or cache files. `tool-accepted` means released
level-bucket membership only; `tool-compatible` means any finite scalar, including
encoded outcomes; `all` includes non-scalar records. No pool filters on
`assay_transfer_eligible`. Select 100 distinct scaffold-disjoint parents independently
per pool/level, then expand all matching records. Each task/split stores pool
assignments against one shared, exact-prompt-key score table. L1 references the
existing task-specific V9 Morgan-100 gold-condition cache.

Before preparation, every checkpoint must pass released-training/copied-context
prompt comparisons. Scalars retain their legacy task-specific display. Non-scalars
use original source measurement text with no separate unit on either side, while
support/result fields remain known-side only. Their projection has a separate hash.
Mapped records with invalid parent identities block preparation for an explicit
decision. The level-mapping-compatibility V10 rebuild excludes its 220 approved
Stage-1 UIDs; any remaining unmapped source UID now blocks preparation.
The v2 parent normalizer retains a validated charged fragment only when neutralizing
the selected parent fails. It does not introduce inorganic counterion component
matches into previously resolved salts; the six affected mapped records are retained.
Historical training/cache inputs may resolve through the explicitly verified
pre-rebuild V10 archive, but only at their original content hashes. Score reuse still
requires exact model revision, prompt, template, projection, and scoring identities.

The following is the archived three-pool builder workflow, not the active V2
runtime: `python -m predict.retrieval.assay_reranking.three_pools prepare --task TASK --subset SPLIT`,
then `score --task TASK --subset SPLIT --level LEVEL --device GPU --num-shards N --shard-index I`,
then `finalize --task TASK --subset SPLIT`. Preparation builds all three pool views;
`load_top_ranked_records(..., task=TASK, subset=SPLIT, pool=POOL)` selects a view.
Scorers process only the missing union of prompt keys. Finalization rejects invalid,
missing, extra, or duplicate journal keys before consolidating scores.

The active V10.3 L1 successor reranks the exact official conditioned-gold Morgan-100
parent pools for BBB and Bioavailability with the pinned task checkpoint. It preserves
the task's existing label adapter and all official training contexts. The corresponding
V3 successor copies the complete immutable V3 database and changes only L1 assay scores
and ranks over the identical parent set; all L2+ assignments are hash-checked unchanged.

The cache-only `v10_3_best_scaffold_morgan100_v1` release is separate from that V3
successor. It uses V1 gold and explicit scaffold-disjoint Morgan-100 pools for valid and
test, with BBB V10.3 step 120 and Oral V10.3.0.2 step 160. BBB scores may be reused only
by exact prompt/model identity; Oral scores must not reuse the older V10.3 checkpoint.
Oral direct-gold rows keep the frozen V10.3 prompt. Its pinned record-level contract is
canonical measurement/unit first with atomic fallback to a complete source pair; never
mix origins. Publishing this cache does not authorize evidence-library mapping, a V3
database, or runtime activation.

`ranked_evidence_retrieval_parent_v1` is the archived combined V1 predecessor.
Its publisher consumed the current task V10 Stage-3 records and
finalized release-owned level map, the V10.3 BEST scaffold-disjoint L1 cache, and
complete BBB V24.1/Oral V25 later-level scores. It writes one evidence database
per task and one rankings database per split; `source_row_uid` is the only
evidence join. L1 is fixed across the three pool views, L2+ pool membership is
independent, and L5 is Morgan-only. Runtime may read only the split manifest and
indexed rows from those two databases. It must not read Parquet, normalize
molecules, compute fingerprints, rescore prompts, hash whole databases, or run
integrity checks. Publication performs those checks once and must reject any L1
membership-card parent overlap, any L2+ physical parent overlap, or any selected
record without a valid display. L1 grouping comes strictly from frozen Gold-v1
voter membership: the card parent overrides a physical record's normalized parent
for ranking and presentation, while the physical parent remains payload provenance.

The V10.4 L1 successor follows the same immutable replacement contract against the
V2 conditioned benchmark. It uses V10.4's stable parent-ID/context-ID ordering,
parent-SMILES prompts, top-100 scaffold-disjoint parents, and one fixed L1 pool shared
by all three later-level pool views. Record-level binary/canonical-source display
adapters are audited separately and never injected into direct-gold L1 prompts.
For BBB V2, explicitly disable the obsolete V1 `bbb_remaining49` gold-context mapping;
the official V2 contexts and V10.4 ranking receipt are the identity authority.

Verify the shared Morgan fingerprint database against exact archived pre-V2 Stage 3
source snapshots when the live source paths have drifted; do not rebuild a scientific
cache merely to match mutable paths. V10.4 context-form Morgan similarities may differ
from the V3 database's parent-scoped values. A successor must match the exact 100-parent
set, retain the base database's parent-scoped Morgan similarity/rank, record the mismatch
count, and prove the L2+ row hash is unchanged.

For a V2 successor of the three-pool cache, keep the prior complete cache's candidate
snapshot fixed: its Stage 3 records, UID-level mapping, and source contract define the
unchanged later-level candidate universe. Released training snapshots validate the
checkpoint and renderer; they do not replace that candidate snapshot. Resolve drifted
historical data inputs only through exact-hash archive receipts. Builder, legacy-builder,
and runtime hashes in an old cache are historical code receipts rather than materialized
data inputs; reuse still re-renders every stored assignment and requires the exact cache,
model, scoring-contract, template, and projection identities.

Preflight the Python environment before three-pool preparation by importing `rdkit`,
`transformers`, and `duckdb`, then verify the pinned problematic BBB salts normalize
successfully. On dgx007, the verified environment for this workflow is
`/vast/projects/myatskar/design-documents/conda_env/openrlhf_tfv4/bin/python`
(RDKit 2025.09.4, Transformers 4.57.6, DuckDB 1.5.3). Do not prepare with
`open_rlhf_intern` (its RDKit 2023.09.6 rejects those parents) or the system Python
(it lacks Transformers); an import-only check is insufficient without the identity
probe.

Generated candidates and scores are ignored by Git. The active
`ranked_level_retrieval_v4` profile lives under
`data/caches/assay_reranking/active/`; its display-ready payload projection lives
under the owning V10 evidence release. The V3 cache is its immutable predecessor.
Earlier V10.3, V10.4, V24.1,
V25, and three-pool profiles retain their existing locations under
`predict/retrieval/cache/assay_reranking/active/`. All other profiles live under
`archive/` and require an explicit legacy path. They are reusable retrieval
inputs, not paper outputs. Prefer adding a clear section to an existing owner
over adding a small helper module.
