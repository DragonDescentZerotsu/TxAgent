# DILI joint direct and indirect search with partitioned V2 semantic weights

The four indirect profiles below were chosen by validation macro F1, then applied once to test with their direct panels frozen. Gold uses 100 `valid_small` and 402 test queries; TDC-v2 uses 47 validation and 96 test queries. All listed runs completed with zero failed queries. The test scores did not choose any profiles.

## Frozen profiles and outcomes

| Benchmark | Model | Direct profile | Indirect profile | Validation direct F1 | Validation mixed F1 | Test direct F1 | Test mixed F1 | Test delta | Test direct accuracy | Test mixed accuracy |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| gold | luna | `ga125_mc025_label000` | `ga075_mc010_sr035_sd010_ld010_rd010` | 0.668475 | 0.688935 | 0.621456 | 0.665266 | +0.043810 | 0.711443 | 0.783582 |
| gold | deepseek | `ga100_mc025_label025` | `ga075_mc010_sr025_sd010_ld010_rd010` | 0.657189 | 0.708544 | 0.639755 | 0.659233 | +0.019478 | 0.758706 | 0.798507 |
| tdc_v2 | luna | `ga075_mc010_label010` | `ga075_mc035_sr075_sd035_ld035_rd035` | 0.840678 | 0.886199 | 0.790850 | 0.801545 | +0.010695 | 0.791667 | 0.802083 |
| tdc_v2 | deepseek | `ga075_mc025_label000` | `ga050_mc025_sr025_sd025_ld025_rd025` | 0.831541 | 0.879672 | 0.819609 | 0.821346 | +0.001737 | 0.822917 | 0.822917 |

Complete profile-level metrics, confusion counts, immutable run paths, and output hashes are in [validation.tsv](validation.tsv) and [test.tsv](test.tsv). [manifest.json](manifest.json) pins each completed matrix, completion receipt, metrics file, predictions file, selection grid, semantic snapshot, score cache manifest, and provider configuration.

## Selection and provenance

- Luna direct panels were separately screened across three profiles on each benchmark. Its Gold winner was `ga125_mc025_label000`; its TDC-v2 winner was `ga075_mc010_label010`. DeepSeek kept its previously fixed Gold `ga100_mc025_label025` and TDC-v2 `ga075_mc025_label000` direct panels.
- Gold indirect validation screened the prior five profiles plus the next five ranked profiles for each model. The additional five produced both Gold winners. TDC-v2 indirect validation screened five profiles per model. The two top DeepSeek TDC-v2 profiles tied on macro F1 and accuracy; the lexically first name was frozen.
- Indirect selection used the active Gold-v1 DILI parent-100 cache or the TDC-v2 DILI parent-disjoint V3 exact-score overlay (652,993 scores), respectively. Relevance combined the DILI tool score and task semantic score by geometric mean; semantic, level, and record diversity coefficients were tied within each profile.
- The semantic weights came from the partitioned V2 DILI snapshot with status `complete_unselected`, as an explicit experimental input. The active semantic release and `CURRENT` pointer were not changed. A hash-pinned TDC extension supplied scores where the candidate snapshot did not cover the TDC universe. This fallback covered 714 distinct universe pairs; the selected TDC Luna test winner used V2 weights for all 4,800 indirect records.
- TDC-v2 DeepSeek direct controls reused earlier OpenRouter-only runs. Their L1 score rows were verified identical between the V2 and V3 TDC overlay (12,232 rows; zero set differences). Gold DeepSeek direct controls were run fresh on the same OpenRouter-only route because older Gold controls used a mixed OpenRouter/RunPod pool.
- All new model requests used high reasoning effort, 65,536 max tokens, 7,200-second timeout, uninterrupted throughput mode, and immediate failure requeues (up to three stage requeues). Luna used OpenRouter standard routing with a launcher-local cap of 64; DeepSeek-V4-Flash-0731 used OpenRouter provider-auto routing with a launcher-local cap of 512.

## Interpretation

The validation and test gains are comparisons within each model and benchmark. Provider routes differ between Luna and DeepSeek, so their scores do not isolate a model effect. The semantic snapshot remains unselected and should be reviewed before use as a production release.
