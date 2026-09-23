# Record-selection optimization

## Gold-v1 direct and joint L2+ successor

`optimization.gold_joint` adds immutable direct and Gold-v1 indirect selection
modes without changing the legacy per-level grids below. Direct mode chooses 10 L1 context
cards using normalized Morgan-gated assay relevance, normalized Morgan-bit
coverage, and normalized log label diversity. Indirect mode pools every L2+
candidate and chooses 50 physical UIDs jointly using the same relevance and bit
coverage terms, reviewed semantic relevance where available, and normalized log
semantic and level diversity. It imposes no per-level quota.

The capacity-aware log term is
`sum(log(1+n_group)) / max_feasible_sum(log(1+x_group))`. The denominator is
fixed for each query and budget, which keeps the score monotone submodular and
within `[0, 1]`. Skin uses its Gold-v1 v9.0.2 direct rankings; its semantic
relevance is unavailable and therefore has an effective weight of zero.

```bash
OPT_PY=/vast/projects/myatskar/design-documents/conda_env/apricot-select/bin/python
OPT_STAGE=/local/$USER/gold_joint_log_diversity_valid_small_v1
$OPT_PY -m optimization.gold_joint direct --output-root "$OPT_STAGE"
$OPT_PY -m optimization.gold_joint indirect --output-root "$OPT_STAGE"
```

Direct mode accepts `--benchmark gold_v1|tdc_v1` and
`--subset valid_small|valid|test`; `--profiles` restricts a held-out selection
to an already frozen winner. Gold-v1 supports all six active tasks, while TDC-v1
supports all six active tasks. Full-split evaluation
uses separate immutable grids, for example:

```bash
$OPT_PY -m optimization.gold_joint direct --subset valid \
  --output-root /local/$USER/gold_joint_log_diversity_full_v1
$OPT_PY -m optimization.gold_joint direct --subset test \
  --profiles ga100_mc025_label025 \
  --output-root /local/$USER/gold_joint_log_diversity_test_winner_v1
$OPT_PY -m optimization.gold_joint direct --benchmark tdc_v1 --subset valid \
  --output-root /local/$USER/tdc_direct_valid_v1
```

`predict.harnesses.branches.six_task_query_priors` builds matching full-split
query-prior overlays. `--reuse-root` reuses compatible `valid_small` artifacts
by `benchmark_row_id` and query SMILES while hash-pinning every source; it never
uses positional fallback.

The current direct grid has 39 profiles (the 27-profile grid plus 12 low-assay
follow-ups). The historical indirect grid retains 94 source aliases,
deduplicates them to 75 objectives, and crosses three level-diversity weights for
225 profiles. `compose` requires a reviewed direct manifest and only writes a
hash-pinned reference to that frozen panel; it never selects a winner. See the
root `AGENTS.md` for validation and publication requirements.

The `sqrt48` indirect successor selects 50 L2+ UIDs jointly for Gold-v1
`valid_small` or TDC-v1 full `valid`. It also accepts `test` and `--profiles`
to reproduce frozen validation winners on the held-out split without running
the full grid. Its 48 profiles cross gated assay
`{.75,1}`, semantic relevance `{.1,.25,.5}`, and molecular coverage, semantic
diversity, and level diversity each `{.1,.25}`. The molecular term is the same
linear normalized Morgan-bit coverage as direct selection; only semantic and
level diversity change to capacity-normalized `sum(sqrt(n_group))`. The
denominator is the maximum feasible value at budget 50 under the actual group
capacities, fixed before greedy selection. Skin's semantic-relevance lambda is
recorded but effectively zero because no reviewed weights exist.

```bash
$OPT_PY -m optimization.gold_joint indirect --grid sqrt48 \
  --benchmark gold_v1 --subset valid_small --workers 8 \
  --output-root /local/$USER/gold_sqrt48_v1
$OPT_PY -m optimization.gold_joint indirect --grid sqrt48 \
  --benchmark tdc_v1 --subset valid --workers 8 \
  --output-root /local/$USER/tdc_sqrt48_v1
```

`compose` binds a selected indirect leaf to an already frozen, benchmark- and
query-matched direct leaf; it never reselects the direct panel. These manifests
are selection artifacts, not downstream prediction results. The full-flat
matrix consumes one per-task grid through `--mixed-selection-grid-manifest`.

The V27 safety/Skin shared-parent caches are separate candidate releases, not
an automatic optimizer input. `gold_joint direct` reads frozen L1 indexes;
`gold_joint indirect` still reads its reviewed BBB/Oral/Skin L2+ universes and
does not include Ames, DILI, or Carcinogens. Do not substitute a new V27 safety
index into that objective without reviewing semantic coverage and recording a
new, hash-pinned optimizer run. The six-task upstream flat prompt likewise
does not select an optimizer or cache by itself.

For the reviewed Skin v6 semantic-only successor, rerun Skin in a new output
root with `--skin-semantic-release v10_main_universe_v6`. The selector checks
the v6 release hashes and its reviewed v5/v6 evidence binding, uses the v6
record-level weights, and leaves any unweighted candidate eligible. If a query
contains an unweighted candidate, its semantic-relevance term is inactive for
that query rather than assigning an invented weight. The default remains v5
for historical reproduction.

This package selects ordered physical V10 evidence records from each query's
active `ranked_level_retrieval_v4` universe: every physical UID beneath the
top 100 Morgan-ranked parents. It optimizes compact cache rows and does not
hydrate evidence payloads.

Published runs that pin `ranked_level_retrieval_v3` remain immutable historical
replays; changing the default does not rewrite them.

For a fixed panel of K records, the objective is:

```text
M + lambda_a*A + lambda_mc*Cmol + lambda_sr*Srel + lambda_sd*Csem
```

All five optimized components are normalized to `[0, 1]`: mean query Morgan
similarity, mean assay-transfer score, covered Morgan fingerprint bits divided
by all bits in the candidate universe, mean reviewed semantic weight, and
distinct semantic buckets divided by K. Both coverage terms are monotone
submodular. Mean pairwise Morgan distance remains a diagnostic but is not part
of the objective. A level with no published assay score records an unavailable
assay term and uses an effective assay lambda of zero.

The full grid contains 384 profiles over assay
`{0,.05,.1,.25,.5,1}`, molecular coverage `{0,.05,.1,.25}`, semantic
relevance `{0,.1,.25,.5}`, and semantic coverage `{0,.05,.1,.25}`. Screening
is label-blind: identical panels collapse, nine requested anchors are retained,
and Pareto layers plus deterministic farthest-point coverage select exactly 32
profiles. K is fixed at 10 independently at every later level.

Generate manifests (not run automatically):

```bash
APRICOT_PY=/vast/projects/myatskar/design-documents/conda_env/apricot-select/bin/python
$APRICOT_PY -m optimization.grid_search \
  --output-root /local/$USER/record_selection_grid_v1 --workers 32
```

The complete grid writes consolidated TSVs and 32 selected
`flat_preselected_uids.v1` manifests. Publish the closed, validated tree to
`outputs/analysis/record_selection/morgan_assay_feature_semantic_grid_v1/`.

Use a single manifest with the fixed full-flat V6 configuration, or give the
screen manifest to the matrix launcher for all 64 profile-task batches:

```bash
$APRICOT_PY -m predict.harnesses.branches --organization flat \
  --harness-version full-flat-context-v6-oral-high-low \
  --task bbb_martins \
  --reranking assay-transfer-contrastive \
  --morgan-primary-parent-width 25 \
  --l1-min-contrast 0 \
  --l1-molecules 10 \
  --records-per-level 10 \
  --flat-preselected-uids outputs/analysis/record_selection/morgan_assay_feature_semantic_grid_v1/selected/a000_mc000_sr000_sd000/bbb_martins/manifest.json \
  --prepare-only
```

```bash
python -m predict.harnesses.branches.matrix \
  --preselected-grid-manifest outputs/analysis/record_selection/morgan_assay_feature_semantic_grid_v1/screen_manifest.json \
  --provider-pool-config predict/api_client/providers/full_flat_context_v5_lambda_grid_four_endpoint_512_high.json \
  --target-total-load-per-endpoint 512 --load-samples 6 \
  --max-tokens 65536 --request-timeout-s 7200 \
  --execution-mode throughput --skip-pilots
```

The harness validates the manifest and UID order against its active cache, then
uses the existing evidence projection and the
`full_flat_context_v6_oral_high_low`
renderer. The optimizer never supplies record text or prompt content.

The label-blind Morgan-gated analysis uses the successor objective
`M + lambda_a*A + lambda_ga*MGA + lambda_mc*Cmol + lambda_sr*Srel +
lambda_sd*Csem`, where `MGA` is mean `Morgan * assay`. Its requested 324-profile
grid is generated without screening or inference:

```bash
$APRICOT_PY -m optimization.grid_search \
  --grid gated-324 \
  --output-root /local/$USER/record_selection_gated_324_properties_v1 \
  --workers 32 --query-set valid_small
```

The closed analysis includes overall `profiles.tsv`, task-level
`profile_properties.by_task_level.tsv`, query-level diagnostics, and ordered UID
selections. Pruning is a separate reviewed step.

The fixed-gate successor searches `{0,.25,.5}` for additive assay, molecular
coverage, semantic relevance, and semantic coverage while fixing gated assay at
1. It retains the prior best profile, the all-zero profile, and each one-term
`.5` ablation, then finds a maximum profile set whose average exact UID overlap
is at most `.8` while retaining equal numbers from the three additive-assay
strata. The sole anchor-anchor exception is recorded explicitly.

```bash
$APRICOT_PY -m optimization.grid_search \
  --grid gated-fixed-81 \
  --output-root /local/$USER/record_selection_gated_fixed_81_v1 \
  --workers 32 --query-set valid_small
```

The normalized-gate successor uses
`(M + alpha*MGA)/(1 + alpha) + lambda_mc*Cmol_normalized + lambda_sr*Srel + lambda_sd*Csem`.
Morgan-bit coverage is divided by the deterministic coverage-only greedy value
for the same query, level, and K. Its 36-profile factorial crosses assay gate
`{.75,1,1.25}`, molecular coverage `{.25,.5,.75}`, and two independent semantic
weights `{.05,.1}`; three assay-only controls bring the total to 39 profiles.

```bash
$APRICOT_PY -m optimization.grid_search \
  --grid normalized-gated-39 \
  --output-root /local/$USER/record_selection_normalized_gated_39_v1 \
  --workers 32 --query-set valid_small
```

The high-semantic repeat keeps the same assay gate, molecular coverage, and
three assay-only controls, while crossing semantic relevance and coverage at
`{.25,.5}`:

```bash
$APRICOT_PY -m optimization.grid_search \
  --grid normalized-gated-semantic-high-39 \
  --output-root /local/$USER/record_selection_normalized_gated_semantic_high_39_v1 \
  --workers 32 --query-set valid_small
```
