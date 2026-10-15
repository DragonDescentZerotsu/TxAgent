# Context selection

Pipeline: **canonical candidate pool → optimize each JSON grid configuration →
export ordered selections → evaluate with the prediction harness on validation →
freeze the winner for test**. Differently weighted objective values do not choose
the hyperparameter winner.

| File | Responsibility |
| --- | --- |
| `pool.py` | Prepare L1 caches; read candidates, exact assay scores and ledger projections. |
| `objectives.py` | Objective mathematics, Morgan features and lazy greedy optimization. |
| `policies.py` | Explicit JSON profiles and direct/indirect scientific requirements. |
| `search.py` | Grid execution, multiprocessing, diagnostics, export, composition and validation. |
| `record_scores.py` | Source-bound deterministic DILI, Bioavailability, BBB, Ames and Carcinogens score caches. |
| `grids/` | Explicit named profile configurations, without Python experiment factories. |

Direct search retrieves 100 canonical context–molecule cards and selects 10 distinct
cards. Parent normalization is used only for filtering and declared disjointness.
Ranking and Morgan coverage use the full molecule, including charge, fragments and
stereochemistry. Different contexts for one molecule may occupy separate slots;
sharing a parent does not merge cards or change their authoritative binary labels.
Canonical card membership preserves physical voters. The existing Gold Skin member
adapter remains explicitly pinned; its member labels retain that published adapter's
card-label convention.

Indirect search selects 50 physical UIDs jointly across L2+, without level quotas.
Both policies preserve ordered physical UID exports and fail membership/disjointness
checks before prediction. All indirect weights come from a selected ledger-bound
projection: reviewed membership means or explicitly provisional task-level means.
A provisional weight does not invent a semantic family.

BBB indirect pools automatically exclude prediction-only evidence before objective
optimization. Experimental and mixed experimental/predicted records remain eligible.
No `--prediction-filter` argument is required; a supplied source-bound filter remains
supported. New selections record `bbb_prediction_only_excluded_l2plus.v2` and the
excluded UID count. Frozen direct L1 membership is unchanged.

Assay scores match the exact query, card/UID, level and parent under their frozen
model/prompt contract. Missing scores remain missing and candidates remain eligible.
An L1 score snapshot must declare `context_identity=canonical_context_molecule.v1`,
`cache_index_sha256`, `model`, `prompt_version` and `scoring_contract_version`.
Parent-level maxima cannot score individual cards. Snapshot databases contain
`query_id, level, item_id, parent_id, score, score_key`; manifests pin their databases
and source files. New selection metadata uses `context_selection.v1` and the shared
physical UID manifest consumed through prediction's `--uids-file` interface.

The DILI geometric policy retains `sqrt(dili_tool_score * semantic_weight)`, complete
record-score coverage, a frozen canonical direct panel, square-root group diversity
and tied diversity coefficients. The standalone scorer in `../dili_score/` is unchanged.

```bash
TASK=bbb_martins BENCHMARK=gold_v1 DIRECT_CACHE=/path/RELEASE_INDEX.json \
  OUTPUT_ROOT=/path/fresh-direct bash experiments/selection/direct.sh
TASK=dili BENCHMARK=tdc CACHE_BUNDLE=/path/cache.yaml GRID=dili_geomean_tied16 \
  OUTPUT_ROOT=/path/fresh-indirect bash experiments/selection/indirect.sh \
  --direct-manifest /path/direct/manifest.json --dili-score-cache /path/scores/manifest.json
python -m optimization.selection.search compose \
  --direct-manifest /path/direct/manifest.json --indirect-manifest /path/indirect/manifest.json \
  --output /path/fresh-combined/manifest.json
```

Declare `--score-snapshot` when frozen exact assay scores are available. Custom grids
are JSON files with `kind`, `diversity` (`log` or `sqrt`) and a `profiles` list using
the existing objective coefficient fields. `--profiles` selects named entries for
frozen held-out evaluation; `--workers` controls CPU parallelism.

The Oral scaffold Luna recipe is `experiments/selection/oral_scaffold_luna.sh`:

```bash
STUDY_ID=oral_scaffold_luna_YYYYMMDD PYTHON_BIN=/path/to/python \
  bash experiments/selection/oral_scaffold_luna.sh
```

It compares the three pinned historical Luna profiles per benchmark, retaining
the frozen direct panels. Gold selects on its existing 100-query `valid_small`;
TDC selects on its full 64-query `valid`. Both winners are frozen before either
test run. Ranking uses macro F1, then accuracy, then profile name, and rejects
incomplete batches or incompatible query/model identities. Model responses are
checked against the declared served model and optional provider.

Generated studies live under the gitignored
`outputs/analysis/context_selection/bioavailability_ma/scaffold/luna/<study-id>/`;
raw batches remain under `outputs/predictions/`. Two explicit three-profile JSON
grids live beside the existing grids. Missing scaffold semantic projections are
materialized by `search extend-semantics` from existing reviewed memberships and
weights, without provisional assignments. `compose --cache-bundle` binds a frozen
Gold direct panel to that explicit cache and preserves its cache-provided scores;
canonical direct panels continue to use ordinary `compose`.
The recipe explicitly enables `--exclude-disjoint-violations` for TDC and Gold
test selection: candidates
that fail the current identity policy are removed before optimization and listed
by query, level, physical UID and reason in the selection audit. Frozen caches
and scores remain intact; the filter does not backfill the original parent pool.
Without this flag, an incompatible candidate still aborts the search.

Prepared caches are indexed by
`data/caches/assay_reranking/active/flat_v5/canonical_l1_cards_v1/manifest.json`.
Each task has valid/test L1 databases and an evidence projection. They contain no
new model scores and have not replaced old caches or benchmark configurations.
The cache preparation command is:

```bash
python -m optimization.selection.pool prepare-l1 --task bbb_martins \
  --train-path /path/train_molecule_condition_labels.jsonl \
  --query-path valid=/path/valid_molecule_condition_labels.jsonl \
  --query-path test=/path/test_molecule_condition_labels.jsonl \
  --membership-path /path/voter_membership.parquet --output /path/fresh-cache
```

Use `--membership-projection` instead for a published member adapter. Omit both only
when each canonical source row directly supplies its physical UID and label.
Old parent-grouped caches are rejected for new direct search. Per-level optimization,
separate mixed optimization and query-mean score imputation are retired. Historical
scientific artifacts remain frozen; new configurations require new output identities.
