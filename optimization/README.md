# Record-selection optimization

This package selects ordered physical V10 evidence records from each query's
immutable `ranked_level_retrieval_v3` universe: every physical UID beneath the
top 100 Morgan-ranked parents. It optimizes compact cache rows and does not
hydrate evidence payloads.

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
