# Luna Skin Reaction final collection

The Luna directory preserves the frozen optimization parameters, both prompt bundles, full validation and test results, and per-query traces with served-route provenance. The two test winners were chosen from validation and reselected on each benchmark’s test cache without retuning.

| Split | Benchmark | Selection | Profile | Complete | Macro F1 | Accuracy |
|---|---|---|---|---:|---:|---:|
| test | gold_v1 | direct_plus_indirect | `ga050_mc010_sr050_sd010_ld010_rd010` | 241/241 | 0.564175 | 0.692946 |
| test | tdc_v2 | direct_plus_indirect | `ga100_mc010_sr025_sd010_ld010_rd010` | 82/82 | 0.605770 | 0.634146 |
| validation | gold_v1 | direct_plus_indirect | `ga025_mc010_sr010_sd010_ld010_rd010` | 239/239 | 0.601590 | 0.707113 |
| validation | gold_v1 | direct_plus_indirect | `ga050_mc010_sr010_sd010_ld010_rd010` | 239/239 | 0.631386 | 0.732218 |
| validation | gold_v1 | direct_plus_indirect | `ga050_mc010_sr050_sd010_ld010_rd010` | 239/239 | 0.646527 | 0.744770 |
| validation | tdc_v2 | direct | `ga100_mc000_label025` | 40/40 | 0.648411 | 0.675000 |
| validation | tdc_v2 | direct_plus_indirect | `ga075_mc010_sr010_sd010_ld010_rd010` | 40/40 | 0.636618 | 0.675000 |
| validation | tdc_v2 | direct_plus_indirect | `ga100_mc010_sr025_sd010_ld010_rd010` | 40/40 | 0.670330 | 0.700000 |
| validation | tdc_v2 | direct_plus_indirect | `ga100_mc025_sr010_sd025_ld025_rd025` | 40/40 | 0.601139 | 0.650000 |

All arms completed with zero failed queries. The Gold-v1 and TDC-v2 test cohorts, caches, and prompt versions differ, so their absolute scores are separate results. The validation results selected the profiles; the test results did not change their parameters.

The canonical [Luna index](../../../paper/assay_transfer_harness/joseph/final/luna/index.json) and [full-test collection](../../../paper/assay_transfer_harness/joseph/final/collection.json) identify the published arms. Numerical data are in [results.tsv](results.tsv); [provenance.json](provenance.json) pins the collection, parameters, prompts, and source analyses.
