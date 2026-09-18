# Record-selection optimization

This package selects physical V10 evidence records from the immutable
`ranked_level_retrieval_v3` universe. It reads the task release index, optimizes
compact ranking rows, and hydrates only selected `source_row_uid` values from the
release-owned evidence projection.

The combined objective selects exactly K records independently from every later
level:

```text
mean(Morgan similarity)
+ assay_lambda * mean(assay-transfer score)
+ parent_lambda * sum(sqrt(records per parent)) / K
+ diversity_lambda * selected fingerprint-union coverage
```

Fingerprint-union coverage is the fraction of radius-2, 2,048-bit Morgan
features present in the level's candidate universe that are covered by selected
parents. It is monotone and submodular and does not construct a pairwise
similarity matrix. Levels without a published assay-transfer model are evaluated
only with `assay_lambda=0`; missing scores are never replaced.

Run the frozen three-point parent-diversity diagnostic with the installed
environment:

```bash
APRICOT_PY=/vast/projects/myatskar/design-documents/conda_env/apricot-select/bin/python
$APRICOT_PY -m optimization.select_records --task bbb_martins
$APRICOT_PY -m optimization.select_records --task bioavailability_ma
```

The default is K=10, `parent_lambda` values `0`, `0.05`, and `0.15`, assay lambda
values `0` and `0.25`, and diversity lambda values `0`, `0.25`, and `1.0`.
Results are written under
`outputs/analysis/record_selection/morgan_assay_feature_diversity_v1/`. L1, gold
labels, semantic-bucket terms, and inference remain outside this milestone.
