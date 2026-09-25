# Gold-v1 DeepSeek selected test controls in FINAL

Status: complete. Six complete direct-plus-indirect test arms are selected by the larger observed macro-F1 between the previous FINAL arm and the repeated full control. This decision was made after viewing test results and is therefore descriptive, not a validation-selected performance estimate. The older arms and their hashes remain in the FINAL collection.

| Task | Previous FINAL | Selected | Selected path |
|---|---:|---:|---|
| ames | 0.774211 | 0.774211 | `gold_v1/ames/direct_plus_indirect` |
| bbb_martins | 0.725340 | 0.732966 | `deepseek/test/gold_v1/bbb_martins/direct_plus_indirect_coverage_control_20260925_v1` |
| bioavailability_ma | 0.861340 | 0.861340 | `gold_v1/bioavailability_ma/direct_plus_indirect` |
| carcinogens | 0.647253 | 0.658598 | `deepseek/test/gold_v1/carcinogens/direct_plus_indirect_repeated_control_20260925_v1` |
| dili | 0.659233 | 0.680085 | `deepseek/test/gold_v1/dili/direct_plus_indirect_sr000_v3_control` |
| skin_reaction | 0.615042 | 0.618458 | `deepseek/test/gold_v1/skin_reaction/direct_plus_indirect_sr005_control_20260925_v1` |

BBB, Carcinogens, and Skin were added as successor arms with full per-query traces, predictions, metrics, source manifests, route ledgers, selected records, parameter snapshots, and an upstream-v2 prompt snapshot. DILI semantic weight 0 was already packaged in FINAL with its upstream-v3 prompt and parameter snapshot; its selected pointer now names that existing arm. Oral and AMES retain their previous FINAL arms. The separate frozen-direct ablations and budget scaling are summarized in `final/ablations/`; no ablation variant replaced a full control.

The selected paths are in `collection.json` under `selected_gold_v1_deepseek_test_best`. Gold DILI also updates `canonical_dili.gold_v1.deepseek.direct_plus_indirect`. No compressed archive was rebuilt.
