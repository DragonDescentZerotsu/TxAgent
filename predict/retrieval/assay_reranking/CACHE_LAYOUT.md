# Assay-reranking cache layout

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
