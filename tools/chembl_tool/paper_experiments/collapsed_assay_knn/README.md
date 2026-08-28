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

Run the current v4 selection with five train-only scaffold folds:

```bash
python -m tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_k_cv
```

The K grid is 3,5,10,15,25,40. L2 uses numerical values without presence indicators. Random
forest retains the value+presence representation and jointly selects K, `max_depth` in
`{None,20}`, and `min_samples_leaf` in `{1,5}`; query-self RF selects only the RF profile.
The additional `Indirect + indirect categorical` surface one-hot encodes each eligible
`(pair_bucket_key, canonical_category_id)` and averages those indicators across neighbors.
Parent/assay category conflicts use an unweighted mode and exact ties are dropped.

Each fold fits numerical standardization and both feature catalogs only from its reference
rows. Pooled OOF macro-F1 selects the configuration; exact ties prefer smaller K, larger
leaves, then finite depth. The frozen selections are evaluated once on scaffold-valid and
written to `outputs/paper/collapsed_assay_k_selection_cv_v4/`. The report also compares the
shared conditions with v3. ClinTox and test splits are not read, and v1-v3 artifacts remain
frozen. `validation_macro_f1.tsv` is the wide score table without embedded configurations;
`selected_k.tsv` reports the selected neighbor counts separately. The long-form
`valid_metrics.tsv` retains accuracy and RF depth/leaf provenance.

Run the repeated-CV v5.1 LR feature ladder:

```bash
python -m tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_v5
```

This uses three repeated five-fold scaffold partitions, K=3,5,10,25,40, and
`C=0.1,1,10`. Candidate configurations are selected by mean pooled OOF macro-F1; the
standard error across repeats is reported but does not override the winner. Numerical LR
features remain value-only. The runner compares all-K means, observed-only means,
similarity-weighted observed-only means, fixed-K3 versus same-K label scores, and
within-assay retained-category distributions with train support at least two. Outputs are
written to `outputs/paper/collapsed_assay_feature_selection_v5_1/`.

Run the minimal matched RF follow-up:

```bash
python -m tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_v5_rf
```

RF retains numerical neighbor-coverage fractions and compares only flexible
`depth=None, leaf=1` and regularized `depth=20, leaf=5` profiles on four retained recipes.
Its outputs are in `outputs/paper/collapsed_assay_feature_selection_rf_v5_1/`. The retained
v5.1 LR macro-F1 is 0.6095/0.6874/0.5455 for BBB/Bioavailability/Skin; RF and the rich
categorical representation do not beat the retained LR recipe on any task. ClinTox and test
splits remain unread.

Run the unified observed-only v6 strategy:

```bash
python -m tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_v6
```

V6 first selects the direct-label K from `3,5,10,25,40` using three repeated five-fold
train-only scaffold partitions, then freezes that task-level K. Every neighbor numerical
surface uses the observed-only per-assay mean. The explicit `indirect plus labels` surface
uses the frozen label K while selecting its assay K independently. Query-self and the two
other neighbor surfaces do not receive a label feature. LR selects `C` from `0.1,1,10` and
uses numerical values without presence indicators. RF selects between the same two v5.1
profiles and retains numerical neighbor-coverage fractions. Categorical features remain
within-assay retained-category distributions and do not include direct labels.

Outputs are written to `outputs/paper/collapsed_assay_unified_observed_only_v6/`.
`validation_macro_f1.tsv` is the complete wide score table; `selected_assay_k.tsv` and
`selected_label_k.tsv` report the two K roles separately, and `model_selections.tsv` retains
LR C and RF profile provenance. This is validation-only: ClinTox and all test splits remain
unread.

Run the filtered observed-continuous/all-neighbor-categorical v7 strategy:

```bash
python -m tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_v7
```

V7 keeps the v5.1 filters: numerical assays must have nonzero reference-train variance and
categorical assay-category features need reference-train support of at least two. Continuous
neighbor values use observed-only means. Categorical one-hot values instead average across
all K neighbors, so missing categorical assays contribute zero. Uniform and Morgan-similarity
weighting are compared, with one weighting choice applied to both blocks. Direct-label votes
remain unweighted.

Standalone direct-label K is selected first from `3,5,10,25,40` and frozen per task. Combined
models then compare that K with using their selected assay K. LR and RF retain the v6 C/profile
grids and presence conventions. Outputs are in
`outputs/paper/collapsed_assay_filtered_observed_weighting_v7/`; the score, assay-K, label-K,
and weighting tables are separate. ClinTox and test splits remain unread.

The minimum train-fold support threshold applies to both numerical assays and
categorical assay-category columns and can be changed without altering the v7
protocol. The completed support-sensitivity refits are:

```bash
python -m tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_v7 \
  --min-feature-support 6 \
  --output-root outputs/paper/collapsed_assay_filtered_observed_weighting_support6_v7

python -m tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_v7 \
  --min-feature-support 10 \
  --output-root outputs/paper/collapsed_assay_filtered_observed_weighting_support10_v7
```

Each refit repeats all train-only K, weighting, label-mode, LR C, and RF profile
selection before fitting the complete train split and evaluating scaffold-valid.
The support-6 full-train catalogs contain `6/576/149` numerical and
`291/76/765` categorical features for BBB/Bioavailability/Skin; support 10 leaves
`2/303/110` numerical and `125/41/360` categorical features. The standalone
direct-label K remains `3/3/5`. Both runs retain all `366/209/245` validation rows
and do not read ClinTox or test.
