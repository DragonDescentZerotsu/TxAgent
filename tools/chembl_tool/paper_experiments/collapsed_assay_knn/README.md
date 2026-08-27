# Collapsed-assay Morgan neighbor validation

This isolated diagnostic predicts the frozen scaffold-validation labels for BBB_Martins,
Bioavailability_Ma, and Skin_Reaction. It does not read ClinTox or any test split.

For each task, the runner reads the benchmark train/valid rows and the canonical Stage 06
collapsed records. A numerical assay feature must be a valid, assay-transfer-eligible
`continuous_median` row from the `indirect` partition with a finite
`finite_scalar_value`. The complete `pair_bucket_key` is the feature identity.

The four reported conditions are:

1. the query molecule's own indirect-assay vector;
2. the unweighted mean indirect-assay vector of its Morgan neighbors;
3. the same mean vector plus the neighbors' positive benchmark-label fraction;
4. the traditional unweighted Morgan label vote.

Train rows use leave-one-out neighbors. Valid rows retrieve only from train. Sparse assay
values are standardized using train observations, missing values are zero, and each assay
has a separate presence feature. The learned conditions use one fixed balanced L2 logistic
regression. Every valid query remains in the primary cohort.

Run the frozen validation sweep:

```bash
python -m tools.chembl_tool.paper_experiments.collapsed_assay_knn.run
```

The default output is `outputs/paper/collapsed_assay_knn_v1/`. It contains input hashes,
feature catalogs, sparse matrices, top-25 neighbor records, row-level predictions,
machine-readable metrics, and `REPORT.md`. The sweep reports K=5,10,25 without choosing a
winner or authorizing a test run.

Compare fixed L2 and L1 logistic regression and random forest, each with and without the
presence half of the frozen vectors:

```bash
python -m tools.chembl_tool.paper_experiments.collapsed_assay_knn.model_ablation
```

This reuses the v1 matrices and neighbor rankings without rebuilding features. It writes to
`outputs/paper/collapsed_assay_model_ablation_v1/` and remains validation-only/report-only.

Select K independently for each presence-enabled L2/random-forest neighbor condition and
for direct-label KNN using four train-only scaffold folds:

```bash
python -m tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_k_cv
```

The fixed candidate grid is K=3,5,10,15,25. Each fold derives its assay vocabulary and
standardization only from the fold reference rows. Pooled OOF macro-F1 selects K, with the
smaller K winning exact ties. The selected conditions are then refit on the complete train
split and evaluated once on scaffold-valid. The default output is
`outputs/paper/collapsed_assay_k_selection_cv_v1/`; ClinTox and test splits are not read.
