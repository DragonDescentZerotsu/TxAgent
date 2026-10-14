# Assay-reranking cache layout

## Canonical full-flat V5 identities

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

The historical `v24_1_bbb_uid_levels_morgan75` and Oral V25 profile names remain
valid compatibility aliases. `runtime.cache_profile_root()` uses the canonical
location automatically when that successor has been restored, and otherwise
continues to resolve the historical location. Generate and verify the tracked
task/benchmark bundle manifest with:

```bash
python -m predict.retrieval.assay_reranking.artifact_bundle inventory \
  --task bbb_martins --benchmark gold_v1 \
  --cache-config predict/retrieval/assay_reranking/ranked_level_retrieval_v4.yaml \
  --output predict/retrieval/assay_reranking/flat_v5_manifests/gold_v1/bbb_martins.json
python -m predict.retrieval.assay_reranking.artifact_bundle verify \
  --manifest predict/retrieval/assay_reranking/flat_v5_manifests/gold_v1/bbb_martins.json
```

Verification hashes the evidence/cache payloads once and writes an ignored
receipt. Full-flat V5 preparation checks the receipt's manifest, size, and mtime
fingerprints; it does not repeat the payload hash scan. Use
`--verify-flat-artifacts` only when intentionally refreshing the receipt.

The current cache is
`data/caches/assay_reranking/active/ranked_level_retrieval_v4/`. Evidence
payloads remain with their semantic owners under the active V10 release at
`data/evidence_libraries/<task>/<release>/retrieval_projection/ranked_evidence_v2/`.

The compact Gold-v1 direct ranking sources are published in normal Git under
`predict/retrieval/cache/assay_reranking/active/`: BBB and Oral use
`v10_3_best_scaffold_morgan100_v1`, while Skin uses
`v9_skin_gold_v1_scaffold_morgan100_v1`. Their aggregate hashes and size gate
are in `direct_gold_v1_release.json`; validate all six task/split caches with
`python -m predict.retrieval.assay_reranking.v9 validate-release`. These source
rankings are distinct from the derived V4 level databases and from TDC caches.

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
searches that directory or substitutes an archived profile. Historical replay
must opt in with `--legacy` and provide the exact archived manifest or cache
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
