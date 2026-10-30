# RASAR and RASAR+indirect

Maintained entrypoint: `python -m baselines.retrieval_forest {prepare,fit,finalize}`.
Reporting names: **RASAR** uses six direct features; **RASAR+indirect** adds the
indirect measurement/similarity pairs. Both reuse this one RF entrypoint.
Feature construction is in `baselines/retrieval_features.py`; it reuses the Morgan
fingerprint definition, condition selector and frozen indirect targets used by
the existing baselines. No MiniMol embeddings or learned encoder are used.

For each query, the vector has `6 + 2B` columns:

- Three main-label neighbors, each represented by `(label, Morgan similarity)`.
- One nearest eligible neighbor **within each indirect bucket**, represented by
  `(aggregated standardized target, Morgan similarity)`.

Main neighbors use either all-train or the existing condition-first policy.
After membership selection, the three positions are ordered by descending
similarity, with stable source-row ties. Indirect ties use the frozen parent order.
Condition affects only main-neighbor selection; there is no condition one-hot.
Unfilled pairs are NaN and use sklearn's native missing-value support.

## Identity and source contract

All train/valid/test queries exclude their own normalized parent/connectivity
from both donor pools. This is essential when generating training features: a
query's own label or a salt/stereochemical variant cannot become an input.
Auxiliary pools additionally exclude the entire valid+test identity union, before
target aggregation/normalization, exactly as in MiniMol + indirect assay.
Scaffold overlap is permitted for auxiliary donors. Original main train/valid/test
rows and the stricter exclusion for added Starling main labels are preserved.
The new RF query-level exclusion can change the original TDC KNN donors for the
few pre-existing TDC main identity overlaps; the earlier KNN results are unchanged.

Use `--source-prepared` to point to the matching audited MiniMol `prepared/` cache.
The RF preparation verifies its hashes and main-row alignment, then reads only
its molecular identities and auxiliary numeric targets. Continuous values are
within-parent/bucket medians with training-pool z-scaling; binary values are
positive fractions. Skin uses the release-level-mapped auxiliary successor from
commit `59c05729`; the other tasks retain their original source pools. All
small/constant buckets surviving heldout exclusion remain. See the
[Skin correction](../outputs/paper/skin_indirect_level_mapped_20260925/README.md).
Saved donor indices bind every feature to its original main row or auxiliary target.

## Training and selection

The adopted initial sweep has six candidates: `max_features` in `{sqrt, 0.3}`
and `min_samples_leaf` in `{1, 5, 20}`. Other settings: Gini splits, unrestricted
depth, bootstrap sampling, no class weighting, seeds 1–5 and 100 trees per seed
(500 trees in the averaged probability). `sqrt` tests strong feature subsampling;
0.3 gives the three direct-label columns more opportunities alongside thousands
of indirect columns. Leaf sizes control model complexity.

Fit only main training rows. Select the candidate by five-seed mean-probability
validation BCE, then select its decision threshold by validation Macro-F1.
Evaluate only that candidate on test, reporting Macro-F1, accuracy and AUROC.
There is no CV, out-of-bag selection or train+valid refit.
Two policies with identical features across all splits reuse one model suite.

## Direct-only RASAR control

Use `--feature-set direct` for both `fit` and `finalize` to train on exactly the
first six columns: three main-label/Morgan-similarity pairs. The default remains
`direct_indirect`. Both modes reuse the same audited `prepared/` cache; direct-only
reads and validates only the direct feature/donor files and main split inputs,
without loading auxiliary arrays. Training and metrics receipts record the feature
set and actual dimension; finalization rejects candidates from the other mode.

Keep the six-candidate grid, seeds, trees, validation selection and donor exclusions
above. With six inputs, sklearn resolves `sqrt` to two features per split and `0.3`
to one; the hyperparameter rule is held fixed, not its absolute feature count.
Do not reuse the selected hyperparameters or threshold from the indirect model.

The [2026-09-25 direct-only suite](../outputs/paper/retrieval_forest_direct_20260925/RESULTS.md)
provides the matched control across both benchmarks, all six tasks, all three
training pools and both policies. It uses the current seed-42 Skin/Carcinogens
TDC caches (82/56 test rows), including their mixed-pool counterparts. Its manifest
pins the exact source caches; prior indirect results and predictions are retained.
Registry: `retrieval_forest_direct`. Per-candidate command receipts and the suite
report give the complete reproduction commands.

```bash
python -m baselines.retrieval_forest prepare \
  --data-dir <main-split-dir> --source-prepared <matching-indirect-cache> \
  --prepared <new-feature-dir> --workers 32
python -m baselines.retrieval_forest fit \
  --prepared <feature-dir> --policy 0 --max-features 0.3 --min-leaf 5 \
  --output <policy-dir>/candidates/features_0.3_leaf_5 --workers 20
# Run every candidate before finalization; policy 1 is condition-first.
python -m baselines.retrieval_forest finalize \
  --prepared <feature-dir> --policy 0 --output <policy-dir>
```

For conditioned DILI/Carcinogens, preparation uses
`--condition-policy same_condition_then_null_then_all`; other cells use the
default `same_condition_then_null`. Arrays are cached once per task/training pool
and reused across RF candidates. Native threads are capped to avoid oversubscription;
RF parallelism is over trees on CPUs. Use `/tmp` for temporary files on node002.

The [current ML Appendix](../outputs/paper/llm_vs_standard_baselines_20260924/ML_APPENDIX.md)
is the reporting source for both RF policies. It combines the original conditioned
and four unchanged TDC tasks from
`outputs/paper/retrieval_forest_20260922/manifest.json` with the seed-42
[Skin](../outputs/paper/skin_tdc_upstream_split_20260924/README.md) and
[Carcinogens](../outputs/paper/carcinogens_tdc_upstream_split_20260924/README.md) suites.
Each successor replaces four RF cells across TDC-only / TDC + Starling, rebuilding
features and heldout exclusions with the same protocol. Earlier seed-1 cells are
historical. The matched naive LLM runs now use these same query inputs.

Each source suite retains selected parameters, thresholds, dimensions and resource
measurements. Registry `retrieval_forest` records the original suite and its
superseded TDC tasks; `tdc_skin_upstream_split` / `tdc_carcinogens_upstream_split` own
the successors. Carcinogens agrees with upstream v2 at test parent/label level;
his train/valid are unpublished and one mercury raw representation differs.

Current reporting uses the unified 198-cell CSV/Appendix and five ML display
groups registered at `llm_vs_standard_baselines`. Internal `rf_direct`,
`rf_indirect` and legacy figure-group keys remain stable identifiers; their
display names are RASAR and RASAR+indirect. Frozen historical receipts retain
the names and counts used when those experiments were completed.

Skin RASAR+indirect in all three pools is superseded by registry
`skin_indirect_level_mapped` (67 buckets / 140 features). Its six direct columns
and donor indices match the previous caches exactly, so the RASAR control is
unchanged. Current scores resolve through the unified Appendix above; old Skin
auxiliary scores in original experiment reports are historical.
