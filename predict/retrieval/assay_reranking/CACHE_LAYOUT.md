# Assay-reranking cache layout

## Molecular representation policy

`predict/retrieval/policies.py` defines `query_parent_evidence_source.v1` through
`source_smiles`, `evidence_smiles`, and the shared `decide_candidate` filter.
Successor projections, level manifests and indexes pin this policy explicitly.

| Operation | Direct L1 | Indirect L2+ |
| --- | --- | --- |
| Assay-transfer query | Normalized query parent | Normalized query parent |
| Assay-transfer evidence | Original frozen context molecule | Complete source SMILES |
| Card identity and display | Frozen context ID and structure; preserve conditions and ordered voters | Parent for one component; canonical complete source for multiple fragments |
| Morgan comparison | Query parent versus frozen context structure | Query parent versus the displayed/grouping structure |

RDKit determines connected components. Multi-fragment sources, including salts,
remain one evidence entity; fragment order does not change their grouping identity.
Their measurements are neither split nor assigned to a chosen constituent. Direct
contexts are never merged by parent. Existing fingerprint settings remain explicit
in each cache contract (including direct-card chirality).

### Current exclusion rules

By default, `policies.py` applies `fragment_parent_connectivity.v3` to source identities,
independently of scoring and display. The approved rules are:

| Check | Exclusion behavior |
| --- | --- |
| Whole record / selected parent | Exclude identical records, whole-record connectivity variants, equal selected parents and equal selected-parent connectivity; salt and stereo changes combined remain excluded |
| Multiple components | Retain exact selected-parent-to-component matching in either direction; do not compare arbitrary secondary components with each other or expand their stereo/scaffold matches |
| Generic scaffold-disjoint | Additionally compare nonempty, non-chiral Murcko scaffolds of the selected parents, preserving bond types; shared empty acyclic scaffolds do not exclude |
| Declared DILI/Carcinogens parent-disjoint | Refine only legacy ElementGraph + formula + net-charge groups using completed stereo-free tautomer normalization; distinct completed keys proceed to generic parent checks, while equal keys, missing structures or incomplete enumeration remain excluded |
| Declared DILI/Carcinogens scaffold-disjoint | Apply the same bond-preserving, non-chiral selected-parent Murcko gate as other tasks, after the retained parent refinement; do not use frozen topology groups for retrieval exclusion |

The task-specific rules live in `../task_identity.py` and require the index's
`new_task_tautomer_identity.v2` declaration and matching task metadata. Missing or
mixed contracts fail validation; refinement is recorded as
`complete_stereo_free_tautomer_within_legacy_group.v1`. Single-heavy-atom queries
also retain the element eligibility gate before fingerprint ranking.
Source structures and normalized/effective representations stay separate;
source-backed preparation rejects missing source identity rather than substituting
a displayed parent. Mixture measurements remain attached to the complete source.

Version 3 unifies retrieval scaffold checks, including DILI/Carcinogens. Their
frozen `MURCKO_TOPOLOGY_` fields remain provenance/grouping metadata in the default
`murcko` mode; benchmark splits and gold construction are unchanged. Historical
DILI/Carcinogens baseline predictions and the October scaffold-indirect follow-up
used the earlier topology gate. Explicit
paper-baseline `legacy_paper` preparation preserves that old behavior, including
coarse parent groups and group representatives, and requires exact frozen-input
parity. See the [protocol switch](../../prompts/README.md#retrieval-protocol-selection).
It does not reinterpret FINAL's frozen cache releases. New
preparations require a new run identity. Reading a frozen ranked cache cannot
recover candidates excluded when that cache was built.

These runtime rules do not rebuild benchmark splits, restore candidates removed
by heldout-L1 prefiltering, or regenerate frozen SQLite/UID releases. Existing
caches retain their own contracts and candidate universes; the integration
[acceptance status](../../../docs/release_refactor.md) records the validation limits.

`rebuild_policy` prepares immutable all-task successors using the shared builders.
The recipe is `experiments/retrieval_cache/molecular_policy.sh`. Publication requires
complete scoring, candidate/source audits and verified bundle receipts. The default
remains the previously verified release until its replacement is fully published.
Legacy payloads, catalogs and receipts are not rewritten or redirected; their weaker
historical filtering is described in the new rebuild study rather than relabeled in place.

Successor score reuse follows declared source priority: the same benchmark's frozen
scaffold cache, its parent cache, then compatible caches from the other benchmark.
Fallback sources fill missing exact prompt/model/contract keys only. Preserve the
chosen frozen score when another run produced a different numerical value; never
average scores or rescore merely to reconcile duplicate runs. Source priority and
reused counts are recorded in the preparation. Changed prompts still require scoring.

After reuse, `rebuild_policy worksets` unions pending exact keys across compatible
Gold/TDC and parent/scaffold preparations for one task. Score these worksets in
disjoint shards across workers, then reuse their closed journals in each candidate
cache. Worksets contain scoring inputs only; they are not retrieval releases.

## Cache identities and active selection

New or restored caches use the semantic identity layout:

```text
flat_v5/<benchmark>/<task>/<role>/<method>/<release>/<variant>
```

`role` is `l1` or `l2plus`; `method` is the ranking/selection method; and
`release` and `variant` are independent fields. For example, the BBB V24.1
successor is:

```text
flat_v5/gold_v1/bbb_martins/l2plus/assay_transfer/v24_1/morgan75
```

Native prediction defaults to the declarative cache set
`flat_v5_manifests/scaffold_disjoint_parent100_v2/catalog.json`. It binds six
Gold-v1 tasks and all six canonical TDC tasks. Each task
binds an immutable L1 index and a scaffold-disjoint L2+ index. Retrieval L1 and
optimizer direct-card L1 remain distinct bindings. Numeric YAML configurations
are retained for historical decoding and frozen build provenance.
The catalog is a binding document, not the payload: its files must be restored
and verified separately. It does not identify the inputs used by the frozen
paper baseline adapter or prove which release produced user-reported Ours scores.

`flat_v5_manifests/parent_disjoint_parent100_v1/catalog.json` binds the same
scaffold-disjoint L1 indexes and the published parent-disjoint L2+ indexes.
Direct L1 always requires scaffold-disjointness; only indirect L2+ offers both
policies. The parent-disjoint Gold Ames/DILI/Carcinogens bindings retain their
published Morgan-only coverage. Switching policies never invents missing scores
or redirects to a different candidate universe.

All active TDC bindings and receipts use `tdc`. Scaffold-disjoint payloads live
under `flat_v5/tdc/<task>/l2plus/assay_transfer/<model-release>/scaffold_disjoint_parent100_v1/`.
Canonical query hashes bind the frozen cohorts. Historical headers and paths are
preserved in `data/legacy/tdc/` through the shared relocation resolver.

The cache index declares levels, disjointness, score availability and any
incomplete-score ordering. The prompt release does not choose those policies.
Prediction accepts `--retrieval-cache` and `--prompt`; see
[the prediction interface](../../README.md).

Generate and verify a task bundle with its actual label lineage:

```bash
python -m predict.retrieval.assay_reranking.artifact_bundle inventory \
  --task bbb_martins --benchmark gold_v1 \
  --cache-config predict/retrieval/assay_reranking/flat_v5_manifests/scaffold_disjoint_parent100_v2/catalog.json \
  --output predict/retrieval/assay_reranking/flat_v5_manifests/scaffold_disjoint_parent100_v2/gold_v1/bbb_martins.json
python -m predict.retrieval.assay_reranking.artifact_bundle verify \
  --manifest predict/retrieval/assay_reranking/flat_v5_manifests/scaffold_disjoint_parent100_v2/gold_v1/bbb_martins.json
```

Verification hashes the evidence/cache payloads once and writes an ignored
receipt under `receipts/<cache-set>/<benchmark>/<task>.json`. Preparation checks
the receipt's manifest, size and mtime without repeating payload scans.
`--verify-flat-artifacts` explicitly refreshes it. Evidence payloads remain under
`data/evidence_libraries/<task>/<release>/retrieval_projection/`; the cache index
pins the exact projection and release.

The compact BBB/Oral Gold-v1 direct ranking sources remain under
`predict/retrieval/cache/assay_reranking/active/v10_3_best_scaffold_morgan100_v1/`.
Skin uses the corrected source at
`data/caches/assay_reranking/active/flat_v5/gold_v1/skin_reaction/l1/assay_transfer/v9_0_2/scaffold_disjoint_parent100_v1/scoring/`.
Its prediction index, canonical voter projection and SQLite rankings live beside
`scoring/`. The obsolete Skin scaffold ranking source has been retired.
Current source hashes and the size gate are in `direct_gold_v1_release.json`;
validate all six task/split caches with
`python -m predict.retrieval.assay_reranking.v9 validate-release`. These source
rankings are distinct from the derived V4 level databases and from TDC caches.

Cache renames use `data/caches/assay_reranking/relocations.json` through the
shared path resolver. Skin's corrected source and canonical voter cache were
relocated byte-for-byte; their frozen profile/provenance fields and content IDs
remain unchanged. Newly built indexes use the uniform profile above. The deleted
obsolete source is not redirected to scientifically different corrected scores.

- `ranked_level_retrieval_v4`: current BBB/Oral Gold-v1 retrieval. Every
  task, split, and level owns one independent SQLite database. L1 contains 100
  scaffold-disjoint Gold context cards and their exact ordered physical voter
  UIDs. Each later level contains every UID belonging to its 100 nearest
  parent-disjoint Morgan parents. The same expanded UID universe carries a
  dense Morgan rank plus, where supported, an assay-transfer score and rank.
  Morgan and assay panels can therefore be selected independently and hydrated
  once through their deduplicated UID union.
- `ranked_level_retrieval_gold_v1_addon_v2`: current Morgan-only Gold-v1
  retrieval for Ames, DILI, and Carcinogens. It preserves the V1 L1 context
  cards and adds independent valid/test L2+ caches over all physical UIDs under
  the top 100 parent-disjoint Morgan parents. Level-0 rows are excluded.
- `ranked_level_retrieval_gold_v1_addon_l1_assay_safety_best_v1`: immutable
  assay-ranked L1 successor for Ames, DILI, and Carcinogens. Its L1 Morgan
  universe and all later-level artifacts are copied unchanged from the add-on
  V2 cache.
- `ranked_level_retrieval_tdc_v1_l1_assay_task_best_v1`: immutable TDC-v1 L1
  successor for BBB, Bioavailability, and Skin using their task-specific
  validation-selected checkpoints over the frozen Morgan-100 universe.
- `ranked_level_retrieval_gold_v1_addon_v1`: immutable L1-only predecessor.
- `ranked_level_retrieval_v3`: immutable predecessor bound to the earlier V10
  evidence projection.
- `ranked_level_retrieval_v2`: immutable 100-row predecessor.
- `cache_matched_retrieval_v3`: retained immutable predecessor.
- `v24_1_bbb_uid_levels_morgan75`: active BBB level-specific assay-transfer
  scores.
- `v25_oral_uid_levels_morgan75`, `_l2`, and `_l3`: active Oral V25
  level-specific assay-transfer scores.

All other profiles are retained unchanged under `archive/`. Active code never
searches that directory or substitutes an archived profile. Historical decoding must use the exact archived manifest or cache
path recorded by the original run.

Every task/split/level directory is self-describing through `VERSION.json`.
Runtime reads the small task index, one level database, and only the selected
UID payloads from the evidence-owned Parquet. It does not normalize molecules,
rerank, or rerun display adaptation.

The reader exposes the same three stages directly:

1. `load_ranked_panels(...)` selects independent Morgan and assay-transfer UID
   orders and returns their deduplicated UID union.
2. `hydrate_uids(...)` fetches that union from the evidence-owned projection.
3. `load_candidates(...)` is the flat/progressive convenience path that performs
   both operations and returns the existing prompt-ready card shape.

## Rebuild and use

### Scaffold-disjoint L2+ successors

The opt-in `ranked_level_retrieval_*_scaffold_disjoint_v1.yaml` configurations
cover all six Gold-v1 tasks, Oral/Skin TDC-v1, and BBB/Ames/DILI/Carcinogens
TDC-v2. Each release uses variant `scaffold_disjoint_parent100_v1` under its
actual benchmark identity; existing L1 references and defaults are preserved.

Use `build_ranked_uid_retrieval` with `--neighbor-identity-policy scaffold_disjoint`,
an explicit `--benchmark gold|tdc`, and the pinned evidence manifest.
The builder retains V24.1 BBB, V25 Oral, V27 Skin training-template, and V27
general safety scoring. BBB/Oral L5 remain explicitly Morgan-only. BBB/Oral
select 100 parents independently per level; Skin/safety select 100 across L2+,
then expand every physical UID at each level. Shared levels may contain fewer
than 100 represented parents when selected parents have no records at that level.

Scaffold filtering uses `predict.retrieval.policies`: matching nonempty
scaffolds, parent identities, and connectivity variants are excluded per query.
Acyclic molecules retain the existing empty-scaffold behavior. Replenishment
ranks the complete eligible evidence pool rather than trimming a predecessor.
`--score-reuse-root` can repeat; both source splits are searched by exact scoring
key, and incompatible models or conflicting values fail. With explicit
`--rescore-conflicting-reuse`, conflicting values are rejected, recorded in
`score_reuse_conflicts.json`, and their keys are queued for fresh scoring.
`--query-identity-root` reads frozen query identities from the matching predecessor
without inheriting its candidate pool; this preserves synthetic identities for
wildcard SMILES in Gold Skin. Preparation checks
published database identity metadata; full hashing belongs to final validation.

For shared pools, run `prepare-universe` before `prepare-level`. Run all test
levels before validation scoring. `pretokenize --tokenized-root ... --num-shards N`
freezes CPU tokenization for `score` with the same shard count. Finalize only
closed, complete journals, then run `index`, `validate`, and artifact inventory
and verification with the matching benchmark and new output paths. A prepared
or partial index is not a published complete cache.

Active preparation may use node-local storage, but it is not durable there. As
soon as a writer finishes, move the closed stage to `/vast`, validate and publish
the `/vast` copy, then remove local staging. First publish the display-ready
evidence projection, then prepare each independent level:

```bash
python -m predict.retrieval.assay_reranking.build_ranked_uid_retrieval build-evidence \
  --task bbb_martins --profile ranked_level_retrieval_v4 \
  --evidence-release v10_main_universe_v3 \
  --evidence-manifest STAGE_EVIDENCE/bbb_martins/VERSION.json
python -m predict.retrieval.assay_reranking.build_ranked_uid_retrieval prepare-level \
  --task bbb_martins --subset valid --level L2 --output-root STAGE_CACHE \
  --profile ranked_level_retrieval_v4 --evidence-release v10_main_universe_v3 \
  --evidence-manifest STAGE_EVIDENCE/bbb_martins/VERSION.json
python -m predict.retrieval.assay_reranking.build_ranked_uid_retrieval score \
  --task bbb_martins --subset valid --level L2 --output-root STAGE_CACHE \
  --evidence-manifest STAGE_EVIDENCE/bbb_martins/VERSION.json --device 0
python -m predict.retrieval.assay_reranking.build_ranked_uid_retrieval finalize \
  --task bbb_martins --subset valid --level L2 --output-root STAGE_CACHE \
  --evidence-manifest STAGE_EVIDENCE/bbb_martins/VERSION.json
```

Repeat for valid/test and each task level, write the task release index, validate
the closed stage, then publish by an atomic rename beside the canonical target.
Never copy a live writer. If a refreshed V10 release produces the exact same
projection bytes, `rebind-evidence` can update provenance without recomputing
ranks; any projection-byte change fails and requires a rebuild.

```bash
python -m predict.retrieval.assay_reranking.build_ranked_uid_retrieval index \
  --task bbb_martins --profile ranked_level_retrieval_v4 \
  --evidence-release v10_main_universe_v3 --output-root STAGE_CACHE \
  --evidence-manifest data/evidence_libraries/bbb_martins/v10_main_universe_v3/retrieval_projection/ranked_evidence_v2/VERSION.json
python -m predict.retrieval.assay_reranking.build_ranked_uid_retrieval validate \
  --task bbb_martins --profile ranked_level_retrieval_v4 \
  --evidence-release v10_main_universe_v3 --output-root STAGE_CACHE \
  --evidence-manifest data/evidence_libraries/bbb_martins/v10_main_universe_v3/retrieval_projection/ranked_evidence_v2/VERSION.json
```

Progressive and flat preparation accept a common default with repeatable
per-level overrides, for example:

```bash
--records-per-level 50 --level-record-limit L2=20 --level-record-limit L6=80
```

L1 is capped at 100 cards. A later-level K may be any positive value up to that
query's expanded UID count; a larger request fails before prompt rendering.
Retrieval opens the task release index, the selected level databases, and the
evidence-owned Parquet projection directly. It does not construct a replacement
database or rescore prompts.

Canonical TDC uses `data/gold_labels/TDC/manifest.json` and cache paths
`flat_v5/tdc/<task>/{l1,l2plus,scores,l1_cards}/`. The manifest binds all six
tasks and preserves their selected split/ranking policies. `l1_cards` is the
optimizer full-molecule candidate universe and has no computed assay scores;
`l1` retains frozen parent-grouped retrieval. The edition-specific YAMLs are
archived in `data/legacy/tdc/configs/`. Historical path reads use explicit
relocations; they never choose a current cohort.
